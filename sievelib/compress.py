"""
compress.py -- make the model GENERATE from a compressed KV cache (R8).

Every other path in this repository measures what an allocation WOULD cost:
`probe.sieve_probe_attention` attends over full-precision keys and only records
the query, and `run_h0` computes the error offline. R8 has to answer "does this
matter for a task", so the model must actually read quantized and evicted keys
and the answer it produces must be scored. This module is that path.

DESIGN (bugs/8_router_endtask/plan.md section 2):

  * SIMULATED QUANTIZATION. Each key is quantized and dequantized at its own
    width with the SAME quantizer the measurement uses (`quant.quantize_keys`:
    per-token norm, random rotation, Lloyd-Max, norm correction). Accuracy
    depends only on the dequantized values; bit-packing is speed and memory,
    which is R14's question.
  * PREFILL AT FULL PRECISION, DECODE COMPRESSED. The cache is built exactly as
    it is today, then decode reads a compressed view of it -- how a deployed
    system works, and the population `run_h0` measured (decode steps).
  * K ONLY, V EXACT, matching the paper (co-design plan B5).
  * A PROTECTED WINDOW. The last `W` prompt tokens -- which hold the question --
    and every token generated during decode stay at full precision in EVERY arm.
    That is the recent window of KIVI and the observation window of SnapKV, it
    is identical across arms so it cannot favour one, and it is a vanishing
    share of the budget (32 of 32,768). Only the CONTEXT, the prompt tokens
    before the window, is compressed.
  * EVICTION IS PER KV HEAD. A GQA model stores one K row per (KV head, token);
    the n_rep query heads sharing it cannot keep different sets. The score is
    the observation window's attention, summed over the window's queries and
    over the group's query heads -- SnapKV's own GQA pooling, and the strongest
    published prefill-time eviction rule, so the baseline is not a straw man.
  * ALLOCATE ONCE, AT DECODE START, NEVER UPGRADE. Simulated quantization keeps
    the full-precision K, so a re-budget could move a token from 2 bits up to 8;
    a real cache that stored it at 2 bits has discarded what that would need.
    Upgrading is only realizable with retained refinement tiers (ROADMAP R11).
    On a 5-60 token answer the allocation is effectively prefill-time anyway.
  * THE ATTENTION IS `probe._sdpa`, unchanged. With compression off this
    function returns exactly what `sieve_probe_attention` returns, which is the
    path L1 validates as a drop-in for the native attention to 0.000 logit
    difference. Compression changes the keys and adds a mask; nothing else.

QUESTION-AGNOSTIC MODE (plan.md 11-12). P0 found SnapKV at 1.00 at every budget:
with the question inside the observation window, the window vote points at the
needle, so eviction becomes an oracle. A deployed compress-once cache (prefix
caching, multi-turn) compresses BEFORE the question exists. In that mode the
driver prefills the CONTEXT only (the window = its last W tokens), compresses,
and then prefills the QUESTION through this function with compression ON: a
q_len > 1 call while `STATE.enabled` reads the compressed context under a
causal mask, exactly as each of its rows would as a decode step
(tests/test_r8.py::test_question_prefill_compressed pins that equality). The
question and the answer are positions >= ctx_len, so they stay full precision.

Nothing here mutates the KV cache. The compressed keys are substituted into the
attention call, so the cache stays full precision and one prefill can serve
every arm: crop the cache back to the prefill length and run the next arm.
"""
from __future__ import annotations
import torch

from . import quant
from .probe import _sdpa

try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
except ImportError as e:  # pragma: no cover
    raise ImportError("needs transformers>=4.48") from e

IMPL = "sieve_compress"


class CompressState:
    """Everything the attention function needs, set by the driver per prompt
    and per arm. A module-level singleton, like probe.STATE, because the
    attention interface gives the function no other channel."""

    def __init__(self):
        self.reset_prompt()

    def reset_prompt(self):
        # prefill: capture the observation window's attention
        self.capture = False
        self.window_start = 0      # absolute position where the protected window begins
        self.ctx_len = 0           # compressible context tokens = window_start
        self.score: dict[int, torch.Tensor] = {}   # layer -> [Hkv, ctx_len]  SnapKV vote
        # H2O: attention each context token received from EVERY prefill query,
        # not only the window's. Costs roughly one extra prefill's attention, so
        # it is captured only when an arm asks for it.
        self.h2o = False
        self.score_h2o: dict[int, torch.Tensor] = {}   # layer -> [Hkv, ctx_len]
        # P2: the interior needs PER-QUERY-HEAD attention (a GQA group's heads
        # attend differently, and the group allocation weighs them), the window
        # queries themselves (the cascade re-scores them against base-tier keys,
        # and the noise model is fitted on the last one), and the scaling.
        self.score_h: dict[int, torch.Tensor] = {}     # layer -> [H, ctx_len]
        self.qwin: dict[int, torch.Tensor] = {}        # layer -> [H, w, d] window queries
        self.scaling: dict[int, float] = {}
        # the FP arm's first `capture_q` DECODE queries -- the queries every
        # per-head output error is measured for, and the only oracle information
        # in R8. Not step 0 alone: step 0's query attends near the KEY as it emits
        # the first digit, while digits 2..k are copied by later queries that
        # attend further along. An allocation can look perfect at step 0 and drop
        # the rest of the answer (found on the CPU pilot: plan.md 10.2).
        self.capture_q = 0
        self.qdec: dict[int, list] = {}                # layer -> [ [H, d] per step ]
        self.reset_arm()

    def reset_arm(self):
        # decode: the arm's compressed view of the context
        self.enabled = False
        self.kdeq: dict[int, torch.Tensor] = {}    # layer -> [Hkv, ctx_len, d] dequantized
        self.evict: dict[int, torch.Tensor] = {}   # layer -> [Hkv, ctx_len] True = evicted
        self.bits: dict[int, torch.Tensor] = {}    # layer -> [Hkv, ctx_len] long, for the audit


STATE = CompressState()


def _layer(module) -> int:
    li = getattr(module, "layer_idx", None)
    if li is None:
        # probe.py falls back to a counter; R8 must not, because the arm's
        # per-layer keys are looked up by index and a counter that drifts by one
        # would attend over another layer's keys without any error.
        raise RuntimeError(
            "attention module has no layer_idx; sieve_compress cannot map the "
            "arm's compressed keys to layers on this architecture")
    return int(li)


def mixed_quantize_keys(K: torch.Tensor, bits: torch.Tensor, R: torch.Tensor,
                        norm_correct: bool = True):
    """Quantize every key at ITS OWN width. Width 0 = evicted.

    K: [..., d] float32 rows; bits: same leading shape, long. Returns
    (dequantized keys, evicted mask). Evicted rows are returned unchanged -- they
    are removed by the attention mask, never read -- so a caller that forgets the
    mask sees full-precision keys, not zeros, and the mistake shows up in the
    audit rather than as a silently wrong attention.

    Reuses `quant.quantize_keys` rather than reimplementing it: that quantizer
    treats every row independently (its norm, its rotation, its levels), so
    quantizing the rows of one width as a gathered subset is exactly the same as
    quantizing everything and selecting. tests/test_r8.py::T-R8-3 pins it.
    """
    if bits.shape != K.shape[:-1]:
        raise ValueError(f"bits {tuple(bits.shape)} does not match keys {tuple(K.shape[:-1])}")
    out = K.clone()
    for b in torch.unique(bits).tolist():
        b = int(b)
        if b <= 0:
            continue
        m = bits == b
        out[m] = quant.quantize_keys(K[m], b, R, norm_correct)
    return out, bits <= 0


def _capture_window(li, query, key, scaling):
    """Observation-window attention, accumulated per layer during prefill.

    Robust to chunked prefill: each call knows the absolute position of its
    queries (the cache holds everything before them), so the window may fall in
    one chunk or straddle two, and the rows are simply summed."""
    q_len, k_len = query.shape[2], key.shape[2]
    base = k_len - q_len                          # absolute position of query row 0
    lo = max(STATE.window_start - base, 0)
    C = STATE.ctx_len
    if lo >= q_len or C <= 0:
        return
    qw = query[0, :, lo:, :].float()              # [H, w, d]
    K = key[0].float()                            # [Hkv, k_len, d]
    H, w, d = qw.shape
    Hkv = K.shape[0]
    r = H // Hkv
    # query head h uses kv head h // r -- the mapping repeat_kv and logits_gqa use
    s = torch.einsum("grwd,gkd->grwk", qw.reshape(Hkv, r, w, d), K) * scaling
    pos = torch.arange(lo, q_len, device=s.device) + base
    j = torch.arange(k_len, device=s.device)
    s = s.masked_fill(j.view(1, 1, 1, -1) > pos.view(1, 1, -1, 1), float("-inf"))
    a = torch.softmax(s, dim=-1)
    sc_h = a[..., :C].sum(dim=2).reshape(H, C)    # per query head, summed over the window
    sc = sc_h.view(Hkv, r, C).sum(1)              # ...and over the group: SnapKV's vote
    STATE.score[li] = sc if li not in STATE.score else STATE.score[li] + sc
    STATE.score_h[li] = sc_h if li not in STATE.score_h else STATE.score_h[li] + sc_h
    # the window's queries, in position order (a window straddling two prefill
    # chunks arrives as two consecutive blocks)
    STATE.qwin[li] = qw if li not in STATE.qwin else torch.cat([STATE.qwin[li], qw], dim=1)
    STATE.scaling[li] = float(scaling)


def _capture_h2o(li, query, key, scaling):
    """H2O's heavy-hitter score: the attention each CONTEXT token receives,
    summed over every prefill query (causal) and over the KV group.

    Why it is here: the observation-window score (SnapKV) asks only what the
    QUESTION attends to, and a multi-hop chain link that the question never
    mentions can score low and be evicted -- the first CPU pilot's evict arm
    answered variable tracking with the first variable five times. H2O also
    counts attention from OTHER context tokens, so a link that a later link
    points at gets credit. That was the hypothesis.

    MEASURED, AND REFUTED (plan.md 9): the raw prefill sum is dominated by a
    POSITIONAL bias. Under causal attention token j is seen by every query after
    it, so its sum decays roughly as ln(L/j): Spearman(score, position) =
    -0.40..-0.47 on Llama-3.2-1B, and a needle at depth 0.93 ranked at the 61st
    percentile -- outside the kept half even at B = 4. SnapKV ranked the same
    needles in the top 0.6%. This reproduces SnapKV's published advantage over
    H2O on long-context retrieval, which is useful as a check that both
    baselines behave like their papers; it makes evict_h2o a WEAK baseline, not
    the fair one.

    NOT the paper's `accum` corner, despite the family resemblance. `accum`
    sums over DECODE steps, where every query sees every token, so it carries
    no causal-count bias. This sums over PREFILL queries, which are causal.

    Faithful to H2O's raw sum; the bias is H2O's, not a bug here (T: brute-force
    column sum, tests/test_r8.py::test_h2o_capture). Query rows are processed in
    blocks so the [group, rows, keys] logits stay ~1 GB at any context. The
    QK^T product runs in the model's dtype (bf16 on a GPU) and only the softmax
    in float32: the score is used for a top-k ranking, not a measurement."""
    q_len, k_len = query.shape[2], key.shape[2]
    base = k_len - q_len
    C = STATE.ctx_len
    if C <= 0:
        return
    K = key[0]                                    # [Hkv, k_len, d]
    Hkv, H = K.shape[0], query.shape[1]
    r = H // Hkv
    QB = max(1, int(2 ** 28 // max(H * k_len, 1)))
    j = torch.arange(k_len, device=K.device)
    acc = torch.zeros(Hkv, C, dtype=torch.float32, device=K.device)
    # An early chunk's keys only reach k_len, which may be SHORTER than the
    # context: its queries can only attend to the first k_len context tokens,
    # so they contribute to acc[:, :k_len] and nothing beyond. (The window
    # capture never meets this -- the window is the last rows, so k_len >= C.)
    m = min(C, k_len)
    for a0 in range(0, q_len, QB):
        a1 = min(q_len, a0 + QB)
        qb = query[0, :, a0:a1, :]                 # [H, b, d]
        b = a1 - a0
        s = torch.einsum("grbd,gkd->grbk", qb.reshape(Hkv, r, b, -1), K).float() * scaling
        pos = torch.arange(a0, a1, device=K.device) + base
        s = s.masked_fill(j.view(1, 1, 1, -1) > pos.view(1, 1, -1, 1), float("-inf"))
        acc[:, :m] += torch.softmax(s, dim=-1)[..., :m].sum(dim=(1, 2))
    STATE.score_h2o[li] = acc if li not in STATE.score_h2o else STATE.score_h2o[li] + acc


def _additive(mask, dtype):
    """An attention mask as an additive float mask (0 = attend). HF may hand a
    custom attention function a boolean mask (True = attend) or a float one."""
    if mask.dtype == torch.bool:
        z = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
        return z.masked_fill(~mask, torch.finfo(dtype).min)
    return mask.to(dtype)


def _question_view(li, query, key, attention_mask):
    """Question-agnostic mode: a MULTI-token call (the question's prefill) over a
    compressed context. Returns the keys with the compressed context substituted
    and an explicit additive mask = causal + evicted + the caller's mask. The
    mask is always explicit, so _sdpa never falls back to its own causal logic."""
    q_len, k_len = query.shape[2], key.shape[2]
    C = STATE.ctx_len
    kd = STATE.kdeq[li]
    if kd.shape[1] != C or k_len - q_len < C:
        raise RuntimeError(
            f"layer {li}: question prefill over a compressed context of {kd.shape[1]} "
            f"tokens (expected {C}) with only {k_len - q_len} cached -- the cache was "
            f"not cropped back to the context length before the question")
    key = torch.cat([kd.unsqueeze(0).to(key.dtype), key[:, :, C:, :]], dim=2)
    H = query.shape[1]
    neg = torch.finfo(query.dtype).min
    m = torch.zeros(1, H, q_len, k_len, dtype=query.dtype, device=query.device)
    pos = torch.arange(q_len, device=query.device) + (k_len - q_len)
    j = torch.arange(k_len, device=query.device)
    m.masked_fill_((j.view(1, -1) > pos.view(-1, 1)).view(1, 1, q_len, k_len), neg)
    ev = STATE.evict[li]
    if bool(ev.any()):
        evq = ev.repeat_interleave(H // key.shape[1], dim=0).to(query.device)   # [H, C]
        m[0, :, :, :C].masked_fill_(evq.unsqueeze(1), neg)
    if attention_mask is not None:
        m = torch.minimum(m, _additive(attention_mask[:, :, -q_len:, :k_len], query.dtype))
    return key, m


def sieve_compress_attention(module, query, key, value, attention_mask=None,
                             scaling=None, dropout=0.0, **kwargs):
    if kwargs.get("softcap"):
        # probe._sdpa does not apply logit softcapping, and L1 rejects such
        # models in run_h0 for that reason. Refuse loudly rather than generate
        # from a subtly different attention.
        raise NotImplementedError("sieve_compress does not support attention softcapping")
    if scaling is None:
        scaling = module.head_dim ** -0.5
    q_len, k_len = query.shape[2], key.shape[2]
    li = _layer(module)

    if q_len > 1 and STATE.enabled and li in STATE.kdeq:
        # question-agnostic mode: the question is prefilled AFTER compression and
        # reads the compressed context (never during the context prefill, where
        # STATE.enabled is False)
        key, m = _question_view(li, query, key, attention_mask)
        return _sdpa(query, key, value, m, scaling, True), None

    if q_len > 1:                                  # prefill: always full precision
        out = _sdpa(query, key, value, attention_mask, scaling, True)
        if STATE.capture:
            _capture_window(li, query, key, scaling)
            if STATE.h2o:
                _capture_h2o(li, query, key, scaling)
        return out, None

    if STATE.capture_q and len(STATE.qdec.setdefault(li, [])) < STATE.capture_q:
        STATE.qdec[li].append(query[0, :, 0].detach().float())    # record, never alter

    if STATE.enabled and li in STATE.kdeq:        # decode, compressed
        C = STATE.ctx_len
        kd = STATE.kdeq[li]
        if kd.shape[1] != C or k_len < C:
            raise RuntimeError(
                f"layer {li}: compressed context has {kd.shape[1]} tokens, "
                f"expected {C}, cache holds {k_len} -- the cache was not cropped "
                f"back to the prefill length between arms")
        key = torch.cat([kd.unsqueeze(0).to(key.dtype), key[:, :, C:, :]], dim=2)
        ev = STATE.evict[li]
        if bool(ev.any()):
            H = query.shape[1]
            evq = ev.repeat_interleave(H // key.shape[1], dim=0)       # [H, C]
            m = torch.zeros(1, H, 1, k_len, dtype=query.dtype, device=query.device)
            m[0, :, 0, :C].masked_fill_(evq, torch.finfo(query.dtype).min)
            if attention_mask is not None:
                # min, not +: two finfo.min added overflow to -inf, and min
                # keeps a position masked if EITHER mask masks it
                m = torch.minimum(m, attention_mask[:, :, :, :k_len].to(query.dtype))
            attention_mask = m

    out = _sdpa(query, key, value, attention_mask, scaling, False)
    return out, None


def install():
    ALL_ATTENTION_FUNCTIONS[IMPL] = sieve_compress_attention


# ----------------------------------------------------------------- per-arm setup
def cache_len(past) -> int:
    return int(past.get_seq_length())


def crop_to(past, length: int):
    """Truncate the cache to `length` tokens, on every transformers version.

    `DynamicCache.crop`'s POSITIVE argument means "keep the first N" up to 5.17
    and is removed in 5.18; a NEGATIVE argument means "remove N from the end" in
    old and new versions alike. So always pass a negative count. A cluster on a
    newer transformers than this machine would otherwise have mis-cropped every
    arm after the first, silently letting each arm read the previous arm's
    generated tokens. crop(0) is skipped: the docstring warns it is not always a
    no-op."""
    extra = cache_len(past) - int(length)
    if extra > 0:
        past.crop(-extra)
    elif extra < 0:
        raise RuntimeError(f"cache holds {cache_len(past)} tokens, cannot crop UP to {length}")
    if cache_len(past) != int(length):
        raise RuntimeError(f"crop left {cache_len(past)} tokens, wanted {length}")


def apply_bits(past, bits_per_layer: dict[int, torch.Tensor], R: torch.Tensor,
               norm_correct: bool = True, keys_fn=None):
    """Build the arm's compressed view of the context from the cache's
    full-precision keys and switch compression on.

    `bits_per_layer[li]` is [Hkv, ctx_len] long. The cache is read, not written.
    `keys_fn(li, K)` (paper-table quantization baselines, kv_quant_baselines.py)
    replaces the TurboQuant quantizer for an arm that keeps every key at one
    width; None is the unchanged path."""
    from .probe import cache_kv
    STATE.reset_arm()
    C = STATE.ctx_len
    for li, bits in bits_per_layer.items():
        K, _ = cache_kv(past, li)                  # [Hkv, L, d], cache dtype
        Kc = K[:, :C, :].float()
        if keys_fn is not None:
            if bool((bits <= 0).any()) or torch.unique(bits).numel() != 1:
                raise ValueError("keys_fn arms keep every key at one width")
            kd, ev = keys_fn(li, Kc), bits.to(Kc.device) <= 0
        else:
            kd, ev = mixed_quantize_keys(Kc, bits.to(Kc.device), R.to(Kc.device),
                                         norm_correct)
        STATE.kdeq[li] = kd.to(K.dtype)
        STATE.evict[li] = ev
        STATE.bits[li] = bits
    STATE.enabled = True


def bits_audit() -> dict:
    """What the arm actually spent, over the context tokens. Every arm is
    supposed to spend B bits per context token; this is how that is checked
    rather than assumed (T-R8-4)."""
    if not STATE.bits:
        return {"bits_per_token": float("nan"), "evict_frac": 0.0}
    tot = sum(float(b.double().sum()) for b in STATE.bits.values())
    n = sum(b.numel() for b in STATE.bits.values())
    ev = sum(float((b <= 0).double().sum()) for b in STATE.bits.values())
    return {"bits_per_token": tot / max(n, 1), "evict_frac": ev / max(n, 1)}

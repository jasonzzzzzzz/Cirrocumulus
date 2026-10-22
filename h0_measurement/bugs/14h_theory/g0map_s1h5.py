#!/usr/bin/env python3
"""G0 (ARCHITECTURES_PLAN.md §5): MODEL_FAMILY.md's per-layer map at 32K tokens instead of 1,600, on five calibration
prompts per model, in the deployed single store's form (theory.md §13). CPU only; it needs no GPU result.

capture  One model, loaded once (adapters_s1h5._load: fp32, eager; full depth, or DEPTH's partial depth), runs every
         prompt with an exact attention function in place of each layer's eager_attention_forward: fused SDPA where
         that is exact, a query-chunked softmax where the layer has learned sinks (GPT-OSS; transformers' SDPA drops
         them) or softcapped logits. Unknown attention kwargs are refused. Each global layer's view (the last 48
         queries, every row's key and value, the scaling, the sinks) goes to VIEWS/<model>/ (../14_logs/14h_theory/g0views/).
         Prompts: position 0 holds the first token of the model's chat template (its BOS, or its first role token),
         then raw text: PG-19 and four LongBench v2 contexts (a code repository, a table, an agent's history log, a
         multi-document news set), each at L = 32768. Controls at 1,600 tokens of PG-19: with that first token, and
         without it (MODEL_FAMILY.md's input exactly, adapters_s1h5._ids).
analyze  One view -> findings/g0/<model>/<view>.json (+ .pt with the per-head-step ratios): flatness; score noise;
         Theorem M's grid in the store's form (keys: the 4-bit Lloyd-Max base plus an r-bit dithered residual,
         r = 1..4; values 3..8 dithered bits; exact read 1/8 or 1/4) and in the uniform form (MODEL_FAMILY.md's grid);
         part 8's slack at 6- and 8-bit keys with the score noise sigma_w (nested_s1h5.depth's measurement), and
         Theorem M against those draws. Read sets: the store's own vote (its base keys; R1 by construction).
summary  Per model and layer: families, allocations, certification widths, the slack law per architecture
         (findings/g0/summary_g0.json, tables_g0.md, widths_<model>.json).

    python g0map_s1h5.py capture --model gemma3 [--L 32768] [--prompts pg19,code] [--no-controls] [--layers 0,5]
    python g0map_s1h5.py analyze <view.pt> [<view.pt> ...] [--draws 2] [--vote store|r6]
    python g0map_s1h5.py summary [--models gemma3,gptoss]
"""
from __future__ import annotations

import collections
import contextlib
import glob
import hashlib
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
METH = os.path.join(os.path.dirname(HERE), "14h_methodology_improve")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))              # sievelib, .h0_corpus, .hf_cache
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if METH not in sys.path:
    sys.path.append(METH)
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cert_s1h5 as K  # noqa: E402
import rquant_s1h5 as RQ  # noqa: E402

DT = torch.float64
N_FIX, N_VOTE = 48, 32                 # the last 48 positions stay exact: 32 cast the vote, 16 are measured
L_MAIN = 32768
PROMPTS = ("pg19", "code", "table", "agent", "news")
LB2 = os.path.join(ROOT, ".h0_corpus/longbench_v2/data-2b48e494.json")
LB2_IDS = {"code": "66fa208bbb02136c067c5fc1", "table": "66f2a414821e116aacb2a3af",      # e2e_s1h5.TEXTS
           "agent": "671b170cbb02136c067d4f4a", "news": "66efaf70821e116aacb234bd"}    # news: Multi-Document QA
CONTROLS = (("pg19", 1600, True), ("pg19", 1600, False))
DEPTH = {"llama70b": 12, "glm45air": 9, "minimaxm2": 6}     # layers loaded: MODEL_FAMILY.md §2's partial depths
# The views moved from .g0views/ here to ../14_logs/14h_theory/g0views/ on 2026-10-22 (14_logs/README.md); G0_VIEWS overrides.
VIEWS = os.environ.get("G0_VIEWS", os.path.normpath(os.path.join(HERE, "..", "14_logs", "14h_theory", "g0views")))
OUT = os.environ.get("G0_OUT", os.path.join(HERE, "findings", "g0"))
ROT_K, ROT_V = 50, 150                 # the tier's rotations (exp4bit_s1h5, nested_s1h5, R6)
STORE_R, STORE_V = (1, 2, 3, 4), (3, 4, 5, 6, 8)
UNI_K, UNI_V = (4, 5, 6, 8), (4, 5, 6, 8)
FRACS = (8, 4)                         # exact read 1/8, 1/4
DEPTH_CFGS = (("uniform 6/4", dict(kind="uniform", kb=6)), ("nested 4+2/4", dict(kind="nested", res_bits=2)),
              ("uniform 8/4", dict(kind="uniform", kb=8)), ("nested 4+4/4", dict(kind="nested", res_bits=4)))
DEPTH_PRED = {"uniform 6/4": "uniform:6/4/8", "nested 4+2/4": "store:2/4/8", "uniform 8/4": "uniform:8/4/8",
              "nested 4+4/4": "store:4/4/8"}


def _atomic_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def _atomic_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(obj, fh, indent=1)
    os.replace(path + ".tmp", path)


# ------------------------------------------------------------------ prompts
def text(name: str, n_chars: int = 400000) -> str:
    """PG-19's first book from char 20,000 (adapters_s1h5._text, e2e_s1h5.text), or a LongBench v2 context."""
    if name == "pg19":
        f = sorted(glob.glob(os.path.join(ROOT, ".h0_corpus/pg19", "pg19_*.txt")))[0]
        return open(f, encoding="utf-8").read()[20000:20000 + n_chars]
    item = next(x for x in json.load(open(LB2)) if x["_id"] == LB2_IDS[name])
    return item["context"][:n_chars]


def encoder(key: str):
    """(encode(text) -> ids without special tokens, the first token of the model's chat template, its string)."""
    import adapters_s1h5 as A
    repo = A.MODELS[key][0]
    snap = A._snap(repo)
    if os.path.exists(os.path.join(snap, "tiktoken.model")):        # Moonshot: pure-Python BPE, template by jinja2
        import jinja2
        tc = json.load(open(os.path.join(snap, "tokenizer_config.json")))
        tmpl = jinja2.Template(tc["chat_template"]).render(messages=[{"role": "user", "content": "x"}],
                                                           add_generation_prompt=True, bos_token=tc.get("bos_token", ""))
        specials = {v["content"]: int(k_) for k_, v in tc.get("added_tokens_decoder", {}).items()}
        hit = max((s_ for s_ in specials if tmpl.startswith(s_)), key=len, default=None)
        if hit is None:
            raise NotImplementedError(f"{key}: the chat template does not start with a special token: {tmpl[:40]!r}")
        return A.TikTokenBPE(os.path.join(snap, "tiktoken.model")).encode, specials[hit], hit
    if os.path.exists(os.path.join(snap, "tekken.json")) and not os.path.exists(os.path.join(snap, "tokenizer.json")):
        return A.TekkenBPE(os.path.join(snap, "tekken.json")).encode, 1, "<s>"   # Mistral's chat format starts with <s>
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(repo)
    tmpl = tk.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False, add_generation_prompt=True)
    first = tk.encode(tmpl, add_special_tokens=False)[0]
    return (lambda s: tk.encode(s, add_special_tokens=False)), int(first), tk.convert_ids_to_tokens(int(first))


def prompt_ids(key: str, name: str, L: int, prefix: bool = True, enc=None) -> tuple:
    """(ids [1, L], spec): the first chat-template token (prefix) then the text's first tokens. Without a prefix, PG-19
    at any L is adapters_s1h5._ids's input (MODEL_FAMILY.md's)."""
    import adapters_s1h5 as A
    if not prefix and name == "pg19":
        ids = A._ids(key, L)[0].tolist()
        first, first_s = None, None
    else:
        encode, first, first_s = enc or encoder(key)
        body = encode(text(name))
        head = [first] if prefix else []
        assert len(body) >= L - len(head), (key, name, len(body), L)
        ids = head + body[:L - len(head)]
    t = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    sha = hashlib.sha256(t.numpy().astype("int64").tobytes()).hexdigest()[:16]
    return t, dict(prompt=name, L=L, prefix=bool(prefix), first_token=first if prefix else None,
                   first_token_str=first_s if prefix else None, ids_sha=sha)


def tag_of(name: str, L: int, prefix: bool) -> str:
    return f"{name}{'' if prefix else 'raw'}_{L}"


# ------------------------------------------------------------------ exact attention at any length
HANDLED = {"dropout", "scaling", "softcap", "s_aux", "sliding_window"}
IGNORED = {"position_ids", "cache_position", "output_attentions", "use_cache", "is_causal", "cu_seq_lens_q",
           "cu_seq_lens_k", "max_length_q", "max_length_k", "past_key_values", "position_embeddings"}
REFUSED = {"position_bias", "head_mask", "attention_bias", "alibi", "attn_bias"}


def _check_kwargs(kw: dict):
    for k_, v_ in kw.items():
        if k_ in REFUSED and v_ is not None:
            raise NotImplementedError(f"attention kwarg {k_!r} is not supported by the exact attention")
        if k_ not in HANDLED and k_ not in IGNORED and v_ is not None and v_ is not False:
            raise NotImplementedError(f"unknown attention kwarg {k_!r} ({type(v_).__name__}): refusing rather than "
                                      "ignoring it")


def _visible(m: torch.Tensor) -> torch.Tensor:
    return m if m.dtype == torch.bool else m > torch.finfo(m.dtype).min / 2


def _chunked_attention(query, key, value, mask, scaling, softcap=None, sink=None, max_elems=1 << 27):
    """Exact softmax attention by blocks of queries: query [B, H, Lq, d], key/value [B, Hkv, Lk, d(v)] (GQA, no
    expansion), mask [B, 1 or H, Lq, Lk] additive float or bool (None: causal, keys aligned at the end), softcap as
    transformers' eager (tanh before the mask), sink [H] a per-head logit with value 0 (GPT-OSS). -> [B, Lq, H, dv]."""
    B, H, Lq, d = query.shape
    Hkv, Lk, dv = key.shape[1], key.shape[2], value.shape[-1]
    r = H // Hkv
    out = query.new_empty(B, Lq, H, dv)
    cq = max(1, min(Lq, max_elems // max(1, H * Lk)))
    off = Lk - Lq
    for b in range(B):
        qb = query[b].reshape(Hkv, r, Lq, d)
        mb = None if mask is None else mask[b if mask.shape[0] > 1 else 0]
        for i0 in range(0, Lq, cq):
            i1 = min(Lq, i0 + cq)
            c = i1 - i0
            if mb is None:
                lo, hi = 0, min(Lk, i1 + off)
            else:
                vis = _visible(mb[:, i0:i1])
                cols = vis.reshape(-1, Lk).any(0).nonzero()
                if cols.numel() == 0:
                    raise RuntimeError("a block of queries sees no key")
                lo, hi = int(cols[0]), int(cols[-1]) + 1
            s = torch.einsum("grcd,gkd->grck", qb[:, :, i0:i1], key[b, :, lo:hi]) * scaling
            if softcap is not None:
                s = torch.tanh(s / softcap) * softcap
            if mb is None:
                ii = torch.arange(i0, i1).view(c, 1) + off
                jj = torch.arange(lo, hi).view(1, hi - lo)
                s = s.masked_fill((jj > ii).view(1, 1, c, hi - lo), float("-inf"))
            else:
                m = mb[:, i0:i1, lo:hi]
                if m.dtype == torch.bool:
                    m = torch.zeros(m.shape, dtype=s.dtype).masked_fill(~m, float("-inf"))
                s = s + (m.view(1, 1, c, hi - lo) if m.shape[0] == 1 else m.reshape(Hkv, r, c, hi - lo))
            if sink is not None:
                s = torch.cat([s, sink.to(s.dtype).reshape(Hkv, r, 1, 1).expand(Hkv, r, c, 1)], -1)
                s = s - s.amax(-1, keepdim=True)
                p = torch.softmax(s, -1)[..., :-1]
            else:
                p = torch.softmax(s, -1)
            o = torch.einsum("grck,gkd->grcd", p.to(value.dtype), value[b, :, lo:hi])
            out[b, i0:i1] = o.reshape(H, c, dv).transpose(0, 1)
    return out


def _scaling(module, scaling) -> float:
    """The scaling transformers' eager function uses: the one passed, else head_dim^-0.5 (Gemma's eager default)."""
    return float(scaling) if scaling is not None else float(module.head_dim ** -0.5)


def exact_attention(module, query, key, value, attention_mask, *a, dropout=0.0, scaling=None, softcap=None, s_aux=None,
                    sliding_window=None, **kw):
    """transformers' eager attention, computed exactly without the [H, L, L] weights: SDPA where the layer has neither
    sinks nor softcapping nor a narrow window, else _chunked_attention (which reads only each block's visible keys).
    Returns (output [B, Lq, H, dv], None)."""
    if a:
        raise NotImplementedError("positional arguments after attention_mask")
    _check_kwargs(kw)
    if dropout and module.training:
        raise NotImplementedError("dropout in training mode")
    scaling = _scaling(module, scaling)
    sink = s_aux if s_aux is not None else getattr(module, "sinks", None)
    if sliding_window is not None and attention_mask is None:
        raise NotImplementedError("a sliding window without a mask")
    if sliding_window is not None:                       # the window must be in the mask: the last row sees <= W keys
        Lk = key.shape[2]
        last = _visible(attention_mask[0, :, -1]).reshape(-1, Lk).any(0)
        if int(last.sum()) > int(sliding_window):
            raise RuntimeError(f"the mask does not encode the sliding window {sliding_window}")
    narrow = sliding_window is not None and 4 * int(sliding_window) < key.shape[2]
    if sink is None and softcap is None and not narrow:      # a masked SDPA would compute every (query, key) pair
        B, H, Lq, d = query.shape
        Hkv, Lk = key.shape[1], key.shape[2]
        r = H // Hkv
        k_ = key[:, :, None].expand(B, Hkv, r, Lk, key.shape[-1]).reshape(B, H, Lk, key.shape[-1])
        v_ = value[:, :, None].expand(B, Hkv, r, Lk, value.shape[-1]).reshape(B, H, Lk, value.shape[-1])
        o = torch.nn.functional.scaled_dot_product_attention(query, k_, v_, attn_mask=attention_mask, scale=scaling,
                                                             is_causal=attention_mask is None and query.shape[2] > 1)
        return o.transpose(1, 2).contiguous(), None
    return _chunked_attention(query, key, value, attention_mask, scaling, softcap, sink), None


def text_attn_modules(model) -> dict:
    """{layer_idx: attention module} of the text decoder only: a multimodal checkpoint's vision or audio tower can reuse
    the same layer indices (Gemma 4), which adapters_s1h5._attn_modules would merge. Refuses duplicate indices."""
    try:
        dec = model.get_decoder()
    except Exception:                                    # noqa: BLE001
        dec = model
    mods = {}
    for m in dec.modules():
        if "Attention" in type(m).__name__ and hasattr(m, "scaling") and getattr(m, "layer_idx", None) is not None:
            if m.layer_idx in mods and mods[m.layer_idx] is not m:
                raise RuntimeError(f"two attention modules with layer_idx {m.layer_idx}")
            mods[m.layer_idx] = m
    return mods


@contextlib.contextmanager
def exact_hooks(model, on_layer=None):
    """Swap the text decoder's attention modules' eager_attention_forward (module-level, looked up at call time) for
    exact_attention; on_layer(li, module, query, key, value, kw) sees each call first. Yields a Counter of calls per
    layer_idx."""
    mods = text_attn_modules(model)
    if not mods:
        raise RuntimeError("no attention modules found")
    calls = collections.Counter()
    undo = []
    for nm in sorted({type(m).__module__ for m in mods.values()}):
        mm = sys.modules[nm]
        if not hasattr(mm, "eager_attention_forward"):
            raise RuntimeError(f"{nm} has no eager_attention_forward")
        orig = mm.eager_attention_forward

        def wrap(module, query, key, value, attention_mask, *a, **kw):
            li = getattr(module, "layer_idx", None)
            calls[li] += 1
            if on_layer is not None:
                on_layer(li, module, query, key, value, kw)
            return exact_attention(module, query, key, value, attention_mask, *a, **kw)
        mm.eager_attention_forward = wrap
        undo.append((mm, orig))
    try:
        yield calls
    finally:
        for mm, orig in undo:
            mm.eager_attention_forward = orig


# ------------------------------------------------------------------ capture
def provenance(model, key: str) -> dict:
    """Compare loaded weights with the checkpoint's own tensors, bit for bit after the dtype cast: the input norm and
    the attention projections of the first and last attention layers, where the checkpoint stores them unquantized.
    A random init passes every self-consistency check (Gemma 3's did, until 2026-10-12); this one it fails."""
    import adapters_s1h5 as A
    from safetensors import safe_open
    snap = A._snap(A.MODELS[key][0])
    idxf = os.path.join(snap, "model.safetensors.index.json")
    if os.path.exists(idxf):
        wm = json.load(open(idxf))["weight_map"]
    else:
        with safe_open(os.path.join(snap, "model.safetensors"), framework="pt") as fh:
            wm = {k_: "model.safetensors" for k_ in fh.keys()}
    params = dict(model.named_parameters())
    att = sorted(text_attn_modules(model))
    checked = {}
    for li in sorted({att[0], att[-1]}):
        for suf in (f"layers.{li}.input_layernorm.weight", f"layers.{li}.self_attn.q_proj.weight",
                    f"layers.{li}.self_attn.k_proj.weight"):
            mk = [n for n in params if n.endswith(suf) and not any(x in n for x in ("vision", "audio"))]
            ck = [n for n in wm if n.endswith(suf) and not any(x in n for x in ("vision", "audio", "mtp"))]
            if len(mk) != 1 or len(ck) != 1:
                continue
            with safe_open(os.path.join(snap, wm[ck[0]]), framework="pt") as fh:
                t = fh.get_tensor(ck[0])
            if t.dtype not in (torch.bfloat16, torch.float16, torch.float32):
                continue                                   # quantized in the checkpoint: dequantized at load
            prm = params[mk[0]].detach()
            checked[suf] = bool(prm.shape == t.shape and torch.equal(prm, t.to(prm.dtype)))
    if not checked or not all(checked.values()):
        raise RuntimeError(f"{key}: loaded weights do not match the checkpoint: {checked or 'nothing comparable'}")
    return checked


def view_path(key: str, tag: str, li: int) -> str:
    return os.path.join(VIEWS, key, f"g0view_{key}_{tag}_L{li}.pt")


def capture(key: str, L: int = L_MAIN, prompts=PROMPTS, controls: bool = True, layers=None, k_load=None) -> dict:
    """Every prompt's views of one model (see the module docstring); skips prompts whose views are all on disk."""
    import adapters_s1h5 as A
    from transformers import AutoConfig
    repo = A.MODELS[key][0]
    cfg = AutoConfig.from_pretrained(repo)
    tc = getattr(cfg, "text_config", None) or cfg
    n_layers = tc.num_hidden_layers
    k_load = k_load or DEPTH.get(key, n_layers)
    glob_l = A.global_layers(tc)
    want = [li for li in glob_l if li < k_load and (layers is None or li in layers)]
    runs = [(p, L, True) for p in prompts] + (list(CONTROLS) if controls else [])
    todo = [(p, Lp, pre) for p, Lp, pre in runs if not all(os.path.exists(view_path(key, tag_of(p, Lp, pre), li))
                                                            for li in want)]
    man_path = os.path.join(VIEWS, key, "manifest.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else dict(model=key, repo=repo, runs={})
    if not todo:
        print(f"{key}: every view on disk ({len(runs)} prompts x {len(want)} layers)", flush=True)
        return man
    enc = encoder(key)
    t0 = time.time()
    model = A._load(key, k=k_load).eval()
    prov = provenance(model, key)
    t_load = time.time() - t0
    import transformers
    man.update(n_layers=n_layers, layers_loaded=k_load, global_layers=[li for li in glob_l if li < k_load], want=want,
               t_load=round(t_load, 1), transformers=transformers.__version__, torch=torch.__version__,
               first_token=enc[1], first_token_str=enc[2], label=A.MODELS[key][2], provenance=prov)
    attn_layers = sorted(text_attn_modules(model))
    for p, Lp, pre in todo:
        tag = tag_of(p, Lp, pre)
        ids, spec = prompt_ids(key, p, Lp, pre, enc=enc)
        meta = dict(model=key, repo=repo, family="gqa", **spec, tag=tag)
        written = []

        def on_layer(li, module, query, key_, value, kw, _meta=meta, _written=written):
            if li not in want:
                return
            H, Hkv = query.shape[1], key_.shape[1]
            Lq = query.shape[2]
            sink = kw.get("s_aux") if kw.get("s_aux") is not None else getattr(module, "sinks", None)
            sc = _scaling(module, kw.get("scaling"))
            rec = dict(_meta, layer=li, L=Lq, scaling=sc, pos=torch.arange(Lq - N_FIX, Lq),
                       q=query[0, :, -N_FIX:].detach().float().reshape(Hkv, H // Hkv, N_FIX, -1).clone(),
                       K=key_[0].detach().float().clone(), V=value[0].detach().float().clone(),
                       sink=None if sink is None else sink.detach().float().reshape(Hkv, H // Hkv).clone(),
                       softcap=kw.get("softcap"), sliding_window=kw.get("sliding_window"))
            _atomic_save(rec, view_path(key, tag, li))
            _written.append(li)
        t1 = time.time()
        with exact_hooks(model, on_layer) as calls:
            with torch.no_grad():
                model(input_ids=ids, use_cache=False, logits_to_keep=1)
        t_fwd = time.time() - t1
        bad = [li for li in attn_layers if calls[li] != 1]
        if bad or sorted(written) != want:
            raise RuntimeError(f"{key} {tag}: attention calls {dict(calls)}; views written {sorted(written)} != {want}")
        man["runs"][tag] = dict(spec, t_forward=round(t_fwd, 1), n_views=len(written), attn_layers=attn_layers)
        _atomic_json(man, man_path)
        print(f"{key} {tag}: forward {t_fwd:.0f} s, {len(written)} views", flush=True)
    del model
    print(f"{key}: load {t_load:.0f} s, total {time.time() - t0:.0f} s -> {os.path.join(VIEWS, key)}", flush=True)
    return man


# ------------------------------------------------------------------ the analysis setup
def setup(path: str, vote: str = "store", fracs=FRACS) -> dict:
    """{frac: S} for one GQA view: S as exp4bit_s1h5.load_setup (the keys nested_s1h5.draw and theory5_s1h5._bound
    read), the last 48 positions exact (sinks folded into them as a row of value 0), queries 32..47 measured. The read
    set is the vote's 1/frac of the context: vote "store" = the single store's own vote (its 4-bit Lloyd-Max base
    keys, the tier's rotation; softmax with the fixed rows; R6's question pass), "r6" = load_setup's (R6's sketch,
    rotation 7, softmax over the sketch only)."""
    import adapters_s1h5 as A
    import nested_s1h5 as N
    from sievelib import quant as Q
    from sievelib.kv_quant_baselines import fp8_e4m3
    d0 = torch.load(path, weights_only=False)
    if d0.get("family", "gqa") != "gqa":
        raise SystemExit("GQA views only")
    q, Kx, Vx = d0["q"].to(DT), d0["K"].to(DT), d0["V"].to(DT)
    pos, sc, L = d0["pos"], float(d0["scaling"]), int(d0["L"])
    G, r, n_all, d = q.shape
    dv = Vx.shape[-1]
    C = L - n_all
    s_all = torch.einsum("grnd,gld->grnl", q, Kx) * sc
    s_all = s_all.masked_fill(~(torch.arange(L).view(1, 1, 1, L) <= pos.view(1, 1, n_all, 1)), K.NEG)
    s_ctx, sf = s_all[..., :C], s_all[..., C:]
    Kc, Vc = Kx[:, :C], Vx[:, :C]
    lf_all = torch.logsumexp(sf, -1)
    of_all = torch.einsum("grnf,gfd->grnd", torch.softmax(sf, -1), Vx[:, C:])
    if d0.get("sink") is not None:
        sk = d0["sink"].to(DT).view(G, r, 1).expand(G, r, n_all)
        lf2 = torch.logaddexp(lf_all, sk)
        of_all, lf_all = of_all * torch.exp(lf_all - lf2).unsqueeze(-1), lf2
    Rk, Rv = Q.random_rotation(d, "cpu", DT, ROT_K), Q.random_rotation(dv, "cpu", DT, ROT_V)
    qv = q[:, :, :N_VOTE]
    if vote == "store":
        Kb = N.nested_quantize(Kc, 4, 1, Rk, 0)["Xb"]
        slv = torch.einsum("grnd,gcd->grnc", qv, Kb) * sc
        pv = torch.exp(slv - torch.logaddexp(torch.logsumexp(slv, -1), lf_all[:, :, :N_VOTE]).unsqueeze(-1))
    elif vote == "r6":
        Kl = Q.quantize_keys(Kc.float(), 4, Q.random_rotation(d, "cpu", torch.float32, 7)).to(DT)
        pv = torch.softmax(torch.einsum("grnd,gcd->grnc", qv, Kl) * sc, -1)
    else:
        raise ValueError(vote)
    score = pv.sum((1, 2))
    ms = slice(N_VOTE, n_all)
    s, lf, of, qm = s_ctx[:, :, ms], lf_all[:, :, ms], of_all[:, :, ms], q[:, :, ms]
    n = n_all - N_VOTE
    o = A._attend(s, Vc, lf, of)
    e8 = (A._attend(torch.einsum("grnd,gcd->grnc", qm, fp8_e4m3(Kc.float()).to(DT)) * sc,
                    fp8_e4m3(Vc.float()).to(DT), lf, of) - o).norm(dim=-1)
    base = dict(q=qm, Kc=Kc, Vc=Vc, s=s, lf=lf, of=of, o=o, e8=e8, Rk=Rk, Rv=Rv, sc=sc, G=G, r=r, n=n, C=C, d=d, dv=dv,
                L=L, layer=int(d0.get("layer", -1)), model=d0.get("model"), tag=d0.get("tag"), sink=d0.get("sink") is not None,
                vote=vote)
    out = {}
    for f in fracs:
        v_ = K.topk_keep(score, C // f)
        km = v_.view(G, 1, 1, C)
        out[f] = dict(base, km=km, keep=v_.unsqueeze(1).expand(G, n, C), frac=f,
                      c=A._attend(s.masked_fill(~km, K.NEG), Vc, lf, of))
    return out


# ------------------------------------------------------------------ Theorem M, memory-safe at 32K
def coin_terms(qr: torch.Tensor, step: torch.Tensor, sc: float, chunk: int | None = None) -> tuple:
    """From per-coordinate key steps step [G, C, d] (a uniform tier repeats one step per row) and rotated queries
    qr [G, r, n, d]: (E Z, Var Z, ldb), each [G, r, n, C], with Z = e^{-delta - ldb} the unbiased tail's weight factor.
    E e^{-k delta} = prod_j sinhc(k a_j), a_j = sc |qr_j| step_j / 2 (Lemma N; theory3_s1h5.exact_coin_moments), and
    Var Z = (E Z)^2 expm1(Delta), Delta = sum_j [g(2 a_j) - 2 g(a_j)], g = log sinhc: no cancellation (Lemma P')."""
    import nested_s1h5 as N
    import theory3_s1h5 as K3
    G, d = qr.shape[0], qr.shape[-1]
    C = step.shape[1]
    ldb = RQ.debias_log(N.coin_moments_pc(qr, step, sc))
    qa = (sc / 2.0) * qr.double().abs().reshape(G, -1, 1, d)
    chunk = chunk or max(8, (1 << 22) // (G * qa.shape[1] * d))
    g1 = torch.empty(G, qa.shape[1], C, dtype=DT)
    dl = torch.empty_like(g1)
    for c0 in range(0, C, chunk):
        a = qa * step[:, c0:c0 + chunk].double().unsqueeze(1)
        l1 = K3._log_sinhc(a).sum(-1)
        g1[:, :, c0:c0 + chunk] = l1
        dl[:, :, c0:c0 + chunk] = K3._log_sinhc(2.0 * a).sum(-1) - 2.0 * l1
    shp = (*qr.shape[:-1], C)
    EZ = torch.exp(g1.view(shp) - ldb)
    return EZ, EZ * EZ * torch.expm1(dl.view(shp).clamp_min(0.0)), ldb


def weights_of(S: dict) -> tuple:
    """(w, u2): the exact normalized weights w_i / Z [G, r, n, C] and |v_i - o|^2, by expansion (no [.., C, d])."""
    lZ = torch.logaddexp(torch.logsumexp(S["s"], -1), S["lf"])
    w = torch.exp(S["s"] - lZ.unsqueeze(-1))
    G, C = S["G"], S["C"]
    vv = (S["Vc"] * S["Vc"]).sum(-1).view(G, 1, 1, C)
    vo = torch.einsum("grnd,gcd->grnc", S["o"], S["Vc"])
    u2 = (vv - 2.0 * vo + (S["o"] * S["o"]).sum(-1, keepdim=True)).clamp_min(0.0)
    return w, u2


def theorem_m(S: dict, w: torch.Tensor, u2: torch.Tensor, unread: torch.Tensor, EZ, VarZ, n2) -> tuple:
    """Theorem M (theory.md §10) for the unbiased tail: E|N|^2 / Z^2 per head-step and its key, value and bias parts.
    theory3_s1h5.second_moment's terms, with |v - o|^2 and the bias vector by expansion (memory-safe at 32K)."""
    wT = w * unread
    w2 = wT * wT
    key = (w2 * VarZ * u2).sum(-1)
    val = (w2 * (VarZ + EZ * EZ) * n2).sum(-1)
    a = wT * (1.0 - EZ)
    bv = torch.einsum("grnc,gcd->grnd", a, S["Vc"]) - a.sum(-1, keepdim=True) * S["o"]
    bias = (bv * bv).sum(-1)
    return key + val + bias, dict(key=key, value=val, bias=bias)


def key_steps(S: dict, form: str, x: int) -> torch.Tensor:
    """Per-coordinate key steps [G, C, d]: form "store" = the nested store at 4 + x bits (nested_s1h5.nested_quantize;
    deterministic given the data), "uniform" = R6's dithered tier at x bits (one step per row, repeated)."""
    import nested_s1h5 as N
    G, C, d = S["G"], S["C"], S["d"]
    if form == "store":
        return N.nested_quantize(S["Kc"], 4, x, S["Rk"], 0)["step"]
    y = S["Kc"] @ S["Rk"].T
    st = 2.0 * y.abs().amax(-1) / (2 ** x - 1)
    return st.unsqueeze(-1).expand(G, C, d)


def value_n2(S: dict, vb: int) -> torch.Tensor:
    """E|e_i|^2 = dv D_i^2 / 12 of R6's dithered value tier at vb bits, [G, 1, 1, C]."""
    yv = S["Vc"] @ S["Rv"].T
    stv = 2.0 * yv.abs().amax(-1) / (2 ** vb - 1)
    return RQ.value_moments(stv.view(S["G"], 1, 1, S["C"]), S["dv"])[0]


def _q(x: torch.Tensor, p: float) -> float:
    x = x.flatten().double()
    return float(torch.quantile(x, p)) if x.numel() <= 16_000_000 else float(x.sort().values[int(p * (x.numel() - 1))])


def grid(Sf: dict, keep_pred=tuple(DEPTH_PRED.values())) -> tuple:
    """Theorem M's grid: per (form, key width, value bits, 1/frac) the median, p90 and share above 1 of the predicted
    RMS error over FP8's actual error, and the key term's median share; flatness per 1/frac; score noise per key
    width. Returns (json-able dict, {config: ratios [head-steps] float32}, {config in keep_pred: predicted MSE})."""
    S8 = Sf[8]
    G, r, n, C = S8["G"], S8["r"], S8["n"], S8["C"]
    e8 = S8["e8"].clamp_min(1e-30)
    w, u2 = weights_of(S8)
    unr = {f: (~Sf[f]["km"]).expand(G, r, n, C) for f in Sf}
    flat = {}
    for f in Sf:
        wu = w * unr[f]
        flat[f] = dict(unread_second_moment_share=float(((wu * wu * u2).sum(-1) / (w * w * u2).sum(-1)
                                                         .clamp_min(1e-300)).median()),
                       unread_mass=float(wu.sum(-1).median()))
    qr = S8["q"] @ S8["Rk"].T
    qn = S8["q"].norm(dim=-1, keepdim=True)
    n2s = {vb: value_n2(S8, vb) for vb in sorted(set(STORE_V) | set(UNI_V))}
    res, rat, sig, preds = {}, {}, {}, {}
    wun = w * unr[8]
    for form, kxs, vbs in (("store", STORE_R, STORE_V), ("uniform", UNI_K, UNI_V)):
        for kx in kxs:
            st = key_steps(S8, form, kx)
            EZ, VarZ, _ = coin_terms(qr, st, S8["sc"])
            if form == "store":
                import nested_s1h5 as N
                sg = N.score_sigma_pc(qr, st, S8["sc"])
            else:
                sg = RQ.score_sigma(qn, st[..., 0].reshape(G, 1, 1, C), S8["sc"]).expand(G, r, n, C)
            sig[f"{form}:{kx}"] = dict(median_unread=float(sg[unr[8]].median()),
                                       w_median=float(((wun * sg).sum(-1) / wun.sum(-1).clamp_min(1e-300)).median()),
                                       max_median=float(sg.masked_fill(~unr[8], 0.0).amax(-1).median()))
            for vb in vbs:
                for f in Sf:
                    tot, pt = theorem_m(S8, w, u2, unr[f], EZ, VarZ, n2s[vb])
                    rr = tot.clamp_min(0.0).sqrt() / e8
                    nm = f"{form}:{kx}/{vb}/{f}"
                    res[nm] = dict(med=float(rr.median()), p90=_q(rr, 0.9), above=float((rr > 1).double().mean()),
                                   key_share=float((pt["key"] / tot.clamp_min(1e-300)).median()),
                                   agg=float(tot.clamp_min(0.0).sum() / (S8["e8"] ** 2).sum().clamp_min(1e-300)))
                    rat[nm] = rr.flatten().float()
                    if nm in keep_pred:
                        preds[nm] = tot
            del EZ, VarZ, st
    return dict(flat=flat, sigma=sig, cfg=res, fp8_err_median=float(S8["e8"].median())), rat, preds


def depth_part(S8: dict, preds: dict, draws: int = 2, delta: float = 1e-3) -> dict:
    """nested_s1h5.depth's measurement on the store's read set: part 8 (hkz+exact) at uniform 6/4 and 8/4 and nested
    4+2/4 and 4+4/4: coverage, median and p90 bound / error, share certified within FP8, sigma_w (attention-weighted
    score noise of the unread rows) and its max, median over head-steps; and the measured MSE over Theorem M's
    (preds: config -> predicted MSE per head-step), pooled over head-steps."""
    import nested_s1h5 as N
    import theory5_s1h5 as K5
    km = S8["km"]
    w = torch.softmax(torch.cat([S8["s"], S8["lf"].unsqueeze(-1)], -1), -1)[..., :-1] * (~km)
    out = {}
    for nm, cfg in DEPTH_CFGS:
        cov, ratio, cert, sw, smax, e2 = [], [], [], [], [], []
        for sd in range(draws):
            D = N.draw(S8, sd, **cfg)
            b = K5._bound("hkz+exact", S8, D, km, S8["c"], D["oT"], delta)
            e = D["err"]
            cov.append(float((e <= b * (1 + 1e-9) + 1e-15).double().mean()))
            ratio.append((b / e.clamp_min(1e-30)).flatten())
            cert.append(float((b <= S8["e8"]).double().mean()))
            sw.append(float(((w * D["sig"]).sum(-1) / w.sum(-1).clamp_min(1e-300)).median()))
            smax.append(float(D["sig"].masked_fill(km, 0.0).amax(-1).median()))
            e2.append(e * e)
            del D
        rt = torch.cat(ratio)
        pr = preds[DEPTH_PRED[nm]]
        m2 = torch.stack(e2).mean(0)
        out[nm] = dict(coverage=min(cov), ratio_med=float(rt.median()), ratio_p90=_q(rt, 0.9),
                       certified_within_fp8=sum(cert) / len(cert), sigma_w_med=sum(sw) / len(sw),
                       sigma_max_med=sum(smax) / len(smax), mse_meas_over_pred=float(m2.sum() / pr.sum()),
                       mse_meas_over_pred_med=float((m2 / pr.clamp_min(1e-300)).median()))
    return out


def mem_estimate_gb(view: str) -> float:
    """analyze's peak RSS from the view's shapes, fitted on two 32K views (Llama-3.1-8B: G r n C = 16.8 M, G C d =
    33.5 M; Gemma 3: 8.4 M, 67 M; both 21.3 GB with the jobs' malloc settings), times 1.25."""
    d0 = torch.load(view, mmap=True, weights_only=False)
    G_, r, n_all, d = d0["q"].shape
    C = int(d0["L"]) - n_all
    return 1.25 * (0.85 * G_ * r * (n_all - N_VOTE) * C + 0.21 * G_ * C * d) / 1e6 + 1.0


_HELD = []


def reserve_memory(gb: float, budget_gb: float | None = None, token_gb: float = 8.0):
    """Hold ceil(gb / token_gb) of the node's memory tokens (flock'd files in /dev/shm, one set per Slurm job) until the
    process exits, so that parallel analyze processes never overcommit the node. All or nothing, with random
    back-off. Off outside Slurm or with G0_NO_RESERVE set."""
    import fcntl
    import random
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("G0_NO_RESERVE"):
        return
    budget = budget_gb or float(os.environ.get("G0_MEM_BUDGET_GB", "640"))
    n_tok = max(1, int(budget // token_gb))
    k = min(n_tok, max(1, math.ceil(gb / token_gb)))
    d = f"/dev/shm/g0tok_{os.environ['SLURM_JOB_ID']}"
    os.makedirs(d, exist_ok=True)
    t0 = time.time()
    while True:
        held = []
        for i in random.sample(range(n_tok), n_tok):
            fh = open(os.path.join(d, f"t{i}"), "a")
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held.append(fh)
                if len(held) == k:
                    _HELD.extend(held)
                    if time.time() - t0 > 1:
                        print(f"(waited {time.time() - t0:.0f} s for {k} x {token_gb:.0f} GB of memory)", flush=True)
                    return
            except BlockingIOError:
                fh.close()
        for fh in held:
            fh.close()
        time.sleep(3 + 7 * random.random())


def result_path(view: str) -> str:
    """findings/g0/<model>/<view stem without g0view_>.json; the model is the view's folder (VIEWS/<model>/)."""
    stem = os.path.basename(view)[:-3]
    model = os.path.basename(os.path.dirname(os.path.abspath(view))) if stem.startswith("g0view_") else "other"
    return os.path.join(OUT, model, stem.replace("g0view_", "") + ".json")


def backfill(view: str, setup_fn=None, grid_fn=None) -> bool:
    """Add the grid's later fields (agg: the layer's aggregate MSE over FP8's, sum over head-steps of Theorem M's
    prediction over sum of FP8's squared errors) to a result written before they existed; reruns setup and grid only."""
    path = result_path(view)
    if not os.path.exists(path):
        return False
    res = json.load(open(path))
    if all("agg" in v for v in res["cfg"].values()):
        return False
    reserve_memory(mem_estimate_gb(view))
    g = (grid_fn or grid)((setup_fn or setup)(view))[0]
    for nm, v in g["cfg"].items():
        res["cfg"].setdefault(nm, {}).update(agg=v["agg"])
    _atomic_json(res, path)
    print(f"{os.path.basename(view)}: agg added", flush=True)
    return True


def analyze(view: str, draws: int = 2, vote: str = "store", out: str | None = None) -> dict:
    reserve_memory(mem_estimate_gb(view))
    t0 = time.time()
    Sf = setup(view, vote=vote)
    S8 = Sf[8]
    g, rat, preds = grid(Sf)
    t1 = time.time()
    dp = depth_part(S8, preds, draws=draws) if draws else {}
    res = dict(view=os.path.basename(view), model=S8["model"], layer=S8["layer"], tag=S8["tag"], L=S8["L"], C=S8["C"],
               G=S8["G"], r=S8["r"], d=S8["d"], dv=S8["dv"], sink=S8["sink"], vote=vote, draws=draws, **g, depth=dp,
               t_grid=round(t1 - t0, 1), t_depth=round(time.time() - t1, 1))
    path = out or result_path(view)
    _atomic_save(rat, path[:-5] + ".pt")
    _atomic_json(res, path)
    c = res["cfg"]
    print(f"{res['view']}: unread 2nd-moment share {g['flat'][8]['unread_second_moment_share']:.4f}; tail / FP8 "
          f"(median) store 4+2/4 {c['store:2/4/8']['med']:.2f}, 4+2/6 {c['store:2/6/8']['med']:.2f}, 4+4/4 "
          f"{c['store:4/4/8']['med']:.2f}; part 8 at 4+2 "
          f"{dp.get('nested 4+2/4', {}).get('ratio_med', float('nan')):.1f}x; "
          f"({res['t_grid']} + {res['t_depth']} s)", flush=True)
    return res


# ------------------------------------------------------------------ summary
def traffic(kb: float, vb: float, frac: int) -> float:
    """R6's traffic model (layermap_s1h5.traffic): tier 1 for the unread rows, FP16 for the read 1/frac."""
    return (1.0 - 1.0 / frac) * (kb + vb) / 32.0 + 1.0 / frac


def cfg_traffic(nm: str) -> float:
    form, rest = nm.split(":")
    kx, vb, f = (int(x) for x in rest.split("/"))
    return traffic(4 + kx if form == "store" else kx, vb, f)


def choose(cfg: dict, form: str = "store", target: str = "p90") -> tuple:
    """The cheapest configuration of `form` meeting the target (p90 or med <= 1), ties broken by the lower median; None
    when no configuration meets it."""
    ok = [(cfg_traffic(nm), v["med"], nm) for nm, v in cfg.items() if nm.startswith(form + ":") and v[target] <= 1.0]
    return min(ok)[2] if ok else None


def split_of(share: float) -> float:
    """k* - v* = 1/2 log2(a / b) from the key term's share a / (a + b) at equal widths (MODEL_FAMILY.md §1.4)."""
    share = min(max(share, 1e-12), 1 - 1e-12)
    return 0.5 * math.log2(share / (1.0 - share))


def load_results(models=None) -> dict:
    """{model: {tag: {layer: result}}} from findings/g0/<model>/*.json."""
    out = collections.defaultdict(lambda: collections.defaultdict(dict))
    for path in sorted(glob.glob(os.path.join(OUT, "*", "*.json"))):
        if os.path.basename(path).startswith(("summary", "widths")):
            continue
        r = json.load(open(path))
        if (models and r["model"] not in models) or "depth" not in r:          # MLA results: g0mla_s1h5
            continue
        out[r["model"]][r["tag"]][r["layer"]] = r
    return out


def summary(models=None, L_main: int = L_MAIN) -> dict:
    """Per model: per-layer families and allocations over the main prompts (those at L_main), and the slack law."""
    R = load_results(models)
    import numpy as np
    summ, rows_law = {}, []
    UNIFORM_STORE = ("store:1/4/8", "store:2/4/8", "store:2/5/8", "store:2/6/8", "store:3/4/8", "store:4/4/8",
                     "store:4/6/8", "store:4/8/8")
    UNIFORM_OLD = ("uniform:6/4/8", "uniform:6/6/8", "uniform:8/4/8")
    for m, by_tag in sorted(R.items()):
        main = [t for t in by_tag if t.endswith(f"_{L_main}")]
        layers = sorted(set().union(*[set(by_tag[t]) for t in main])) if main else []
        per_layer = {}
        for li in layers:
            rs = {t: by_tag[t][li] for t in main if li in by_tag[t]}
            ch = {t: choose(r["cfg"], "store", "p90") for t, r in rs.items()}
            chm = {t: choose(r["cfg"], "store", "med") for t, r in rs.items()}
            tr = [cfg_traffic(c) if c else traffic(8, 8, 4) for c in ch.values()]
            pick = max(((cfg_traffic(c) if c else 9.0), c) for c in ch.values())[1]
            fl = [r["flat"]["8"]["unread_second_moment_share"] for r in rs.values()]
            per_layer[li] = dict(
                prompts=sorted(rs), choice_p90=ch, choice_med=chm, traffic_max_p90=max(tr), calibrated=pick,
                flatness_med=float(np.median(fl)), flatness_range=[min(fl), max(fl)],
                split_66=[split_of(r["cfg"]["uniform:6/6/8"]["key_share"]) for r in rs.values()],
                sigma6_w=[r["sigma"]["uniform:6"]["w_median"] for r in rs.values()],
                sigma_store2_w=[r["sigma"]["store:2"]["w_median"] for r in rs.values()],
                p8_store2=[r["depth"].get("nested 4+2/4", {}).get("ratio_med") for r in rs.values()],
                p8_store4=[r["depth"].get("nested 4+4/4", {}).get("ratio_med") for r in rs.values()],
                misses={c: sum(r["cfg"][c]["p90"] > 1.0 for r in rs.values()) for c in UNIFORM_STORE + UNIFORM_OLD})
            for t, r in rs.items():
                for nm in ("uniform 6/4", "uniform 8/4", "nested 4+2/4", "nested 4+4/4"):
                    dd = r["depth"].get(nm)
                    if dd and dd["ratio_med"] > 0 and dd["coverage"] is not None:
                        rows_law.append(dict(model=m, layer=li, tag=t, cfg=nm, sigma_w=dd["sigma_w_med"],
                                             ln_ratio=math.log(dd["ratio_med"]), coverage=dd["coverage"]))
        n_l = max(1, len(per_layer))
        summ[m] = dict(
            layers=layers, prompts=main,
            traffic_calibrated=sum(cfg_traffic(v["calibrated"]) if v["calibrated"] else traffic(8, 8, 4)
                                   for v in per_layer.values()) / n_l,
            unmet=[li for li, v in per_layer.items() if v["calibrated"] is None],
            misses_any_prompt={c: sum(v["misses"][c] > 0 for v in per_layer.values()) for c in UNIFORM_STORE + UNIFORM_OLD},
            loose_4p2=[li for li, v in per_layer.items() if any(x is not None and x > 20 for x in v["p8_store2"])],
            per_layer=per_layer)
    law = {}
    groups = collections.defaultdict(list)
    for x in rows_law:
        groups[("all", x["model"])].append(x)
        if x["cfg"].startswith("uniform"):
            groups[("uniform", x["model"])].append(x)
            groups[("uniform", "*")].append(x)
    for (kind, m), xs in groups.items():
        if len(xs) < 3:
            continue
        s_ = np.array([x["sigma_w"] for x in xs])
        y = np.array([x["ln_ratio"] for x in xs])
        A_ = np.stack([np.ones_like(s_), s_], 1)
        coef, *_ = np.linalg.lstsq(A_, y, rcond=None)
        res = y - A_ @ coef
        law[f"{kind}:{m}"] = dict(n=len(xs), a=float(coef[0]), b=float(coef[1]),
                                  r2=float(1 - (res ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-300)),
                                  max_abs_resid=float(abs(res).max()),
                                  ref_resid_med=float(np.median(y - (0.51 + 13.9 * s_))))
    full = dict(models=summ, slack_law=law, slack_rows=rows_law)
    _atomic_json(full, os.path.join(OUT, "summary_g0.json"))
    for m, v in summ.items():
        _atomic_json({str(li): dict(calibrated=x["calibrated"], loose_4p2=li in v["loose_4p2"])
                      for li, x in v["per_layer"].items()}, os.path.join(OUT, f"widths_{m}.json"))
    for m, v in summ.items():
        print(f"{m}: {len(v['layers'])} layers x {len(v['prompts'])} prompts; calibrated store traffic "
              f"{v['traffic_calibrated']:.3f}; unmet {v['unmet']}; 4+2/4 misses {v['misses_any_prompt']['store:2/4/8']}, "
              f"4+2/6 {v['misses_any_prompt']['store:2/6/8']}, 4+4/4 {v['misses_any_prompt']['store:4/4/8']}; "
              f"loose at 4+2: {v['loose_4p2']}", flush=True)
    for k_, v_ in sorted(law.items()):
        print(f"slack law {k_}: ln(bound/error) = {v_['a']:.2f} + {v_['b']:.1f} sigma_w (n {v_['n']}, R2 {v_['r2']:.3f}, "
              f"max resid {v_['max_abs_resid']:.2f}; median residual against 0.51 + 13.9 sigma_w "
              f"{v_['ref_resid_med']:+.2f})", flush=True)
    return full


def main():
    argv = sys.argv[1:]
    cmd = argv[0] if argv else ""
    opts, args, i = {}, [], 1
    while i < len(argv):
        if argv[i].startswith("--"):
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                opts[argv[i]] = argv[i + 1]
                i += 2
            else:
                opts[argv[i]] = True
                i += 1
        else:
            args.append(argv[i])
            i += 1
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    if cmd == "capture":
        lay = opts.get("--layers")
        capture(opts["--model"], int(opts.get("--L", L_MAIN)),
                tuple(opts["--prompts"].split(",")) if "--prompts" in opts else PROMPTS,
                controls="--no-controls" not in opts, layers=[int(x) for x in lay.split(",")] if lay else None,
                k_load=int(opts["--k-load"]) if "--k-load" in opts else None)
    elif cmd == "analyze":
        for v in args:
            if "--force" not in opts and os.path.exists(result_path(v)):
                print(f"{v}: on disk")
                continue
            analyze(v, draws=int(opts.get("--draws", 2)), vote=opts.get("--vote", "store"))
    elif cmd == "backfill":
        for v in args:
            backfill(v)
    elif cmd == "summary":
        summary(opts["--models"].split(",") if "--models" in opts else None)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()

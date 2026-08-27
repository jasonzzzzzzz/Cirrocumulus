"""
validate.py -- three INDEPENDENT checks of the probe.

  L1  drop-in fidelity   model(sieve_probe) vs model(sdpa) -> identical LM logits.
  L2  capture fidelity   attention rebuilt from captured q + cache K,V matches what
                         the model actually produced.
  L3  external reference our softmax weights vs HuggingFace's own `eager` +
                         output_attentions. Independent implementation, so this is
                         the only level that can catch a bug SHARED between the
                         capture path and our sdpa path -- e.g. a wrong GQA
                         expansion, which L2 alone would happily pass.

v2 fixes:
  * L1 and L3 validate the PROBE, which is architecture-dependent but not
    size-dependent. They now run against a small same-architecture proxy
    (config `validate_with`), instead of loading a 70B model four times.
  * They run in float32, so tolerances can be ~1e-4 instead of the 5e-2 that bf16
    forced -- tight enough to actually catch a subtle indexing error.
"""
from __future__ import annotations
import torch
from . import probe as P


def _load(model_id, impl, dtype=torch.float32, **kw):
    from transformers import AutoModelForCausalLM
    dev = {"": 0} if torch.cuda.is_available() else None
    return AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, device_map=dev, attn_implementation=impl,
        trust_remote_code=True, **kw).eval()


def level1_dropin(model_id, ctx=512, tol=1e-4, dtype=torch.float32, **kw):
    """The probe must not change the model's output at all."""
    torch.manual_seed(0)
    ids = torch.randint(100, 20000, (1, ctx))
    out = {}
    for impl in ("sdpa", "sieve_probe"):
        m = _load(model_id, impl, dtype, **kw)
        with torch.no_grad():
            out[impl] = m(ids.to(m.device)).logits[0, -1].float().cpu()
        del m
        torch.cuda.empty_cache()
    d = (out["sdpa"] - out["sieve_probe"]).abs().max().item()
    return d < tol, d


def level2_capture(state, past, tol=2e-2):
    """Rebuild attention from what we captured and compare with the model's own
    output. Runs on the real model at real context, hence the looser bf16 tol."""
    from .probe import cache_kv, repeat_kv
    worst = 0.0
    for li, q in state.q.items():
        K, V = cache_kv(past, li)
        K, V = K.float().cpu(), V.float().cpu()
        n_rep = q.shape[0] // K.shape[0]
        s = torch.einsum("hd,hld->hl", q, repeat_kv(K[None], n_rep)[0]) * state.scaling[li]
        if state.mask[li] is not None:
            s = s + state.mask[li][: s.shape[-1]][None, :]
        o = torch.einsum("hl,hld->hd", torch.softmax(s, -1), repeat_kv(V[None], n_rep)[0])
        ref = state.ref_out[li]
        worst = max(worst, ((o - ref).norm(dim=-1)
                            / ref.norm(dim=-1).clamp_min(1e-6)).max().item())
    return worst < tol, worst


def level3_external(model_id, ctx=384, tol=1e-3, dtype=torch.float32, **kw):
    """Compare against HuggingFace's independent eager implementation."""
    torch.manual_seed(0)
    ids = torch.randint(100, 20000, (1, ctx))
    m = _load(model_id, "eager", dtype, **kw)
    with torch.no_grad():
        ref = m(ids.to(m.device), output_attentions=True, use_cache=False)
    ref_attn = [a[0, :, -1].float().cpu() for a in ref.attentions]
    del m, ref
    torch.cuda.empty_cache()

    from .probe import cache_kv, repeat_kv
    m2 = _load(model_id, "sieve_probe", dtype, **kw)
    P.STATE.reset(); P.STATE.enabled = True
    with torch.no_grad():
        pre = m2(ids[:, :-1].to(m2.device), use_cache=True)
        out = m2(ids[:, -1:].to(m2.device), past_key_values=pre.past_key_values,
                 use_cache=True)
    P.STATE.enabled = False

    worst = 0.0
    for li, q in P.STATE.q.items():
        if li >= len(ref_attn):
            continue
        K, _ = cache_kv(out.past_key_values, li)
        K = K.float().cpu()
        n_rep = q.shape[0] // K.shape[0]
        s = torch.einsum("hd,hld->hl", q, repeat_kv(K[None], n_rep)[0]) * P.STATE.scaling[li]
        if P.STATE.mask[li] is not None:
            s = s + P.STATE.mask[li][: s.shape[-1]][None, :]
        a = torch.softmax(s, -1)
        r = ref_attn[li][:, : a.shape[-1]]
        worst = max(worst, (a - r).abs().max().item())
    del m2, out, pre
    torch.cuda.empty_cache()
    return worst < tol, worst
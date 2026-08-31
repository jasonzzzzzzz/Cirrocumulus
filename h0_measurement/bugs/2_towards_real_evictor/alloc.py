"""
alloc.py -- reverse water-filling, and measurement of what it actually buys.

METHODOLOGY (v2, after audit). Two things are kept strictly separate:

  * the THEOREM is used only to CHOOSE bit-widths. It is first-order and its cost
    model is a heuristic.
  * the REPORTED ERROR is measured by EXACT recomputation -- real quantized keys,
    real softmax, real output. Nothing downstream assumes the expansion is accurate.
    `lin_ratio` records predicted/measured so the expansion is itself testable.

Two bugs fixed in this pass, either of which could invert the conclusion:

  1. UNITS. Evicting token i costs a_i^2 w_i^2 * 1; quantizing it at b bits costs
     a_i^2 w_i^2 * Var(delta_i), with Var(delta) in nats^2 -- an ABSOLUTE quantity.
     The previous version divided Var(delta) by tau^2 while leaving the eviction
     cost at 1.0, mixing units. At tau=2.5 that told the allocator a 1-bit key
     costs 0.42 against eviction's 1.0 when the true cost is 2.6: a 6x error that
     systematically refused to evict. sig2 is now absolute throughout; the
     tau-relative value is reported as `c{b}_rel` for comparison with the table.

  2. BIAS. A constant shift in logits has exactly zero effect on the output
     (softmax shift-invariance), so E[delta^2] overstates the cost -- by 37% at
     1 bit on real data. We use the variance. We additionally fit
     delta_i = (alpha-1) s_i + eps_i and report alpha (a pure temperature
     distortion: order-preserving and correctable by one per-head scalar) and
     Var(eps). Var(delta), which still contains the alpha term, drives allocation
     -- the conservative choice, since we do not assume the correction is applied.
"""
from __future__ import annotations
import math
import torch


BAND_MIN = 2.0   # a head is "in the productive band" if the interior beats the
                 # best corner by at least this factor; below it, routing to the
                 # corner is the right engineering call and SIEVE adds nothing.


def _cost_vector(sig2: dict[int, float], maxb: int, device) -> torch.Tensor:
    """Dense cost lookup over 0..maxb. `bit_list` may be sparse (e.g. no 7-bit
    tier), so unmeasured widths are filled by geometric extension (4x per bit)
    from the nearest measured width below."""
    have = sorted(sig2)
    vals = []
    for b in range(maxb + 1):
        if b in sig2:
            vals.append(sig2[b])
        else:
            lo = max(x for x in have if x <= b)
            vals.append(sig2[lo] / (4.0 ** (b - lo)))
    return torch.tensor(vals, dtype=torch.float64, device=device)


def noise_model(s: torch.Tensor, shat: dict[int, torch.Tensor]) -> dict:
    """Fit delta_i = (alpha-1) s_i + eps_i per bit-width. sig2 is ABSOLUTE
    Var(delta) with sig2[0] = 1.0 as the eviction cost."""
    sd = s.double()
    sc = sd - sd.mean()
    ss = float((sc @ sc).item())
    sig2, alpha, resid = {0: 1.0}, {}, {}
    for b, sv in shat.items():
        h = sv.double()
        sig2[b] = float((h - sd).var(unbiased=False).item())
        a = float(((h - h.mean()) @ sc / max(ss, 1e-30)).item())
        alpha[b] = a
        resid[b] = float((h - h.mean() - a * sc).var(unbiased=False).item())
    return dict(sig2=sig2, alpha=alpha, resid=resid)


def waterfill(w2: torch.Tensor, sig2: dict[int, float], budget_bits: float,
              maxb: int = 8, iters: int = 60) -> torch.Tensor:
    """Minimise sum_i w2_i * sig2[b_i] subject to sum_i b_i <= budget * L.

    Per token the Lagrangian is w2_i*c_b + lambda*b, so we take the argmin over the
    available tiers directly. Tiers above the lower convex envelope (e.g. a 1-bit
    tier costing more than eviction) are never selected -- no explicit envelope
    construction is required.
    """
    bits = sorted(sig2)
    cvec = torch.tensor([sig2[b] for b in bits], dtype=torch.float64, device=w2.device)
    bt = torch.tensor(bits, dtype=torch.float64, device=w2.device)
    target = budget_bits * w2.numel()
    w2c = w2.clamp_min(1e-300)

    def alloc_for(lam):
        return (cvec[None, :] + lam * bt[None, :] / w2c[:, None]).argmin(dim=1)

    lo, hi = 1e-30, 1e30
    idx = alloc_for(lo)
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        idx = alloc_for(mid)
        if float(bt[idx].sum().item()) > target:
            lo = mid
        else:
            hi = mid
    return bt[idx].long()


def exact_error(s: torch.Tensor, shat: dict[int, torch.Tensor], V: torch.Tensor,
                b: torch.Tensor, o: torch.Tensor | None = None) -> float:
    """Relative output error under a per-token allocation, computed exactly.

    `o` is the exact (unquantized) output. It does not depend on the allocation,
    so callers comparing many allocations for one head -- quant_metrics runs
    three corners plus one per configured evictor, per budget -- should compute
    it once and pass it in. Recomputed here when omitted.
    """
    used = set(int(x) for x in torch.unique(b).tolist()) - {0}
    missing = used - set(shat)
    if missing:
        raise ValueError(f"allocation used bit-widths {sorted(missing)} with no "
                         f"measured logits; add them to bit_list")
    sd, Vd = s.double(), V.double()
    if o is None:
        o = torch.softmax(sd, -1) @ Vd
    keep = b > 0
    if not bool(keep.any()):
        return 1.0
    sh = sd.clone()
    for bit, sv in shat.items():
        m = b == bit
        if bool(m.any()):
            sh[m] = sv.double()[m]
    sh = sh[keep]
    ah = torch.softmax(sh - sh.max(), -1)
    oh = ah @ Vd[keep]
    return float((oh - o).norm().item() / max(float(o.norm().item()), 1e-12))


def sensitivity_metrics(s: torch.Tensor, V: torch.Tensor, n_sink: int = 4) -> dict:
    """CHEAP: no quantization. Safe to run on every decode step."""
    L = s.numel()
    if L < 128:
        return {}
    sd, Vd = s.double(), V.double()
    a = torch.softmax(sd, -1)
    o = a @ Vd
    w = (Vd - o).norm(dim=-1)
    sens = a * w

    out = {"L": L, "tau": float(sd.std().item())}
    order = torch.argsort(sd, descending=True)
    out["tau_nosink"] = (float(sd[order[n_sink:]].std().item())
                         if L > n_sink + 64 else float("nan"))
    pos = sens[sens > 0]
    out["ladder_bits"] = (float(torch.log2(pos).std().item())
                          if pos.numel() > 64 else float("nan"))
    pa = a[a > 0]
    out["ladder_bits_a_only"] = (float(torch.log2(pa).std().item())
                                 if pa.numel() > 64 else float("nan"))
    out["w_cv"] = float((w.std() / w.mean().clamp_min(1e-12)).item())
    asort = torch.sort(a, descending=True).values
    out["top1"] = float(asort[0].item())
    out["n95"] = int(torch.searchsorted(
        torch.cumsum(asort, 0),
        torch.tensor(0.95, dtype=a.dtype, device=a.device)).item()) + 1
    out["eff_frac"] = out["n95"] / L
    out["entropy"] = float(-(a * torch.log(a.clamp_min(1e-300))).sum().item())
    return out


def quant_metrics(s: torch.Tensor, shat: dict[int, torch.Tensor], V: torch.Tensor,
                  budgets=(1, 2, 3, 4), maxb: int = 8,
                  practical_scores: dict[str, torch.Tensor] | None = None,
                  practical_score: torch.Tensor | None = None) -> dict:
    """EXPENSIVE: needs quantized logits at every bit-width in `shat`.

    `practical_scores` maps an evictor label (see sievelib.evict) to a selection
    score available WITHOUT seeing the current query -- lagged attention, which
    is all H2O/SnapKV/TOVA/StreamingLLM ever get. The oracle eviction corner
    ranks by true sensitivity a_i||v_i-o||, which requires the very weights it
    is trying to avoid computing; it is an upper bound on any real evictor, so
    the honest headline is the gain over the PRACTICAL corner.

    Every evictor gets its own `*_<label><B>` columns. The unsuffixed
    `err_practical<B>` / `gain_best_practical<B>` / `in_band_practical<B>`
    aggregate them by taking the STRONGEST (lowest-error) practical corner per
    head. That is deliberately the conservative direction for H0: it makes the
    baseline as hard to beat as any deployable evictor could make it, so an
    in-band verdict cannot be blamed on having picked a weak competitor. It is
    also mildly optimistic about evictor selection (you would not know per head
    which one wins), which is why the per-evictor columns are kept -- rerun any
    analysis against a single fixed evictor from those.

    `practical_score` (singular tensor) is the pre-registry spelling, kept so
    older callers and tests still work; it is labelled "practical".
    """
    L = s.numel()
    if L < 128 or not shat:
        return {}
    scores = dict(practical_scores or {})
    if practical_score is not None:
        scores.setdefault("practical", practical_score)
    for nm, ps in scores.items():
        if ps.numel() != L:
            raise ValueError(
                f"practical score {nm!r} has {ps.numel()} entries but the head "
                f"has {L} live positions -- the score is mis-aligned with the "
                f"logits and would rank the wrong tokens")
    sd, Vd = s.double(), V.double()
    a = torch.softmax(sd, -1)
    o = a @ Vd
    w2 = (a * (Vd - o).norm(dim=-1)) ** 2
    tau2 = max(float(sd.var(unbiased=False).item()), 1e-12)

    nm = noise_model(s, shat)
    sig2 = nm["sig2"]
    out: dict[str, object] = {}
    if scores:
        out["n_practical"] = len(scores)
    order = torch.argsort(sd, descending=True)
    top = order[: max(64, L // 100)]
    rt = torch.argsort(torch.argsort(sd[top])).double()
    rt = rt - rt.mean()
    for b in sorted(shat):
        out[f"c{b}_abs"] = sig2[b]
        out[f"c{b}_rel"] = sig2[b] / tau2
        out[f"alpha{b}"] = nm["alpha"][b]
        out[f"resid{b}_abs"] = nm["resid"][b]
        out[f"evict_beats_b{b}"] = float(sig2[b] > 1.0)
        rh = torch.argsort(torch.argsort(shat[b].double()[top])).double()
        rh = rh - rh.mean()
        out[f"spearman_top_b{b}"] = float(
            (rt @ rh / (rt.norm() * rh.norm()).clamp_min(1e-12)).item())

    cv = _cost_vector(sig2, maxb, w2.device)
    for B in budgets:
        if int(B) not in shat:
            continue
        bw = waterfill(w2, sig2, float(B), maxb)
        e_wf = exact_error(s, shat, V, bw, o)
        e_un = exact_error(s, shat, V, torch.full_like(bw, int(B)), o)
        m = max(1, int(round(B * L / maxb)))
        be = torch.zeros_like(bw)
        be[torch.argsort(w2, descending=True)[:m]] = maxb
        e_ev = exact_error(s, shat, V, be, o)
        out[f"err_wf{B}"] = e_wf
        out[f"err_uniform{B}"] = e_un
        out[f"err_evict{B}"] = e_ev
        out[f"gain_u{B}"] = e_un / max(e_wf, 1e-12)
        out[f"gain_e{B}"] = e_ev / max(e_wf, 1e-12)
        out[f"gain_best{B}"] = min(e_un, e_ev) / max(e_wf, 1e-12)
        out[f"in_band{B}"] = float(out[f"gain_best{B}"] >= BAND_MIN)
        if scores:
            e_best, who = None, ""
            for nm_, ps in scores.items():
                bp = torch.zeros_like(bw)
                bp[torch.argsort(ps.double(), descending=True)[:m]] = maxb
                e_p = exact_error(s, shat, V, bp, o)
                out[f"err_practical_{nm_}_b{B}"] = e_p
                out[f"gain_practical_{nm_}_b{B}"] = e_p / max(e_wf, 1e-12)
                out[f"oracle_evict_advantage_{nm_}_b{B}"] = e_p / max(e_ev, 1e-12)
                if e_best is None or e_p < e_best:
                    e_best, who = e_p, nm_
            out[f"err_practical{B}"] = e_best
            out[f"gain_practical{B}"] = e_best / max(e_wf, 1e-12)
            out[f"gain_best_practical{B}"] = min(e_un, e_best) / max(e_wf, 1e-12)
            out[f"in_band_practical{B}"] = float(
                out[f"gain_best_practical{B}"] >= BAND_MIN)
            out[f"oracle_evict_advantage{B}"] = e_best / max(e_ev, 1e-12)
            out[f"best_evictor{B}"] = who
        out[f"evict_frac{B}"] = float((bw == 0).double().mean().item())
        out[f"mean_bits{B}"] = float(bw.double().mean().item())
        pred = math.sqrt(
            float((w2 * cv[torch.full_like(bw, int(B))]).sum().item())
            / max(float((w2 * cv[bw]).sum().item()), 1e-300))
        out[f"lin_ratio{B}"] = pred / max(out[f"gain_u{B}"], 1e-12)
    return out


def head_metrics(s, shat, V, budgets=(1, 2, 3, 4), maxb=8, n_sink=4,
                 practical_scores=None, practical_score=None) -> dict:
    m = sensitivity_metrics(s, V, n_sink)
    if m and shat:
        m.update(quant_metrics(s, shat, V, budgets, maxb,
                               practical_scores=practical_scores,
                               practical_score=practical_score))
    return m

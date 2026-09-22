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

from . import evict as _EV


_WARNED_RANKED_INTERIOR = False   # one-shot guard, see quant_metrics/interior_raw

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
    feasible = alloc_for(hi)
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        idx = alloc_for(mid)
        if float(bt[idx].sum().item()) > target:
            lo = mid
        else:
            hi = mid
            feasible = idx
    return bt[feasible].long()


def waterfill_floor(w2: torch.Tensor, sig2: dict[int, float], budget_bits: float,
                    maxb: int, floor: torch.Tensor | None) -> torch.Tensor:
    """waterfill with the `floor` positions held at `maxb`, BUDGET-MATCHED: the
    rest are water-filled on what those positions leave, so the head still
    spends exactly budget_bits * L.

    For a LAGGED allocator, `floor` is the set its score knows nothing about
    (Evictor.unseen): at minimum the token appended this step. Water-filling
    them on their score of 0 evicts them, and the current query attends to its
    own token -- the job214* R3 defect: lag cost 6.4-7.0x on real attention
    with them zeroed vs 1.13-1.46x floored (R3-fresh-token-test.py).
    """
    if floor is None or not bool(floor.any()):
        return waterfill(w2, sig2, budget_bits, maxb)
    L, n_f = w2.numel(), int(floor.sum().item())
    out = torch.full((L,), int(maxb), dtype=torch.long, device=w2.device)
    if n_f < L:
        rest = max(0.0, (budget_bits * L - maxb * n_f) / (L - n_f))
        out[~floor] = waterfill(w2[~floor], sig2, rest, maxb)
    return out


def exact_error(s: torch.Tensor, shat: dict[int, torch.Tensor], V: torch.Tensor,
                b: torch.Tensor, o: torch.Tensor | None = None) -> float:
    """Relative output error under a per-token allocation, computed exactly.

    `o` is the exact unquantized output. It does not depend on the allocation,
    so a caller comparing many allocations for one head -- quant_metrics now
    runs one corner per (evictor, policy) cell per budget -- should compute it
    once and pass it in. Recomputed here when omitted.
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


def evict_error_curve(s: torch.Tensor, shat_max: torch.Tensor, V: torch.Tensor,
                      order: torch.Tensor, Ks, o: torch.Tensor | None = None,
                      chunk: int = 32768) -> dict[int, float]:
    """Eviction-corner error for EVERY kept-token count in `Ks`, in one pass.

    An eviction corner keeps the top-K tokens by some ranking, all at maxb, and
    drops the rest. Walking K therefore walks a NESTED family of kept sets, so
    the softmax numerator and denominator are running sums along `order` and the
    whole curve costs about one `exact_error` instead of one per K. That is what
    makes E1's K* diagnostic and the (evictor x policy) grid affordable.

    Exact, not an approximation: softmax is shift-invariant, so factoring out a
    single global max leaves oh(K) = sum_{i<K} w_i v_i / sum_{i<K} w_i equal to
    the softmax over the kept subset. Accumulated in chunks so peak memory stays
    O(chunk*d) rather than O(L*d).
    """
    sd, Vd = s.double(), V.double()
    if o is None:
        o = torch.softmax(sd, -1) @ Vd
    onorm = max(float(o.norm().item()), 1e-12)
    L = sd.numel()
    z = shat_max.double()[order]
    w = torch.exp(z - z.max())
    ks = sorted({max(1, min(int(k), L)) for k in Ks})
    num = torch.zeros_like(o)
    den = torch.zeros((), dtype=torch.float64, device=o.device)
    out, prev = {}, 0
    for k in ks:
        for lo in range(prev, k, chunk):
            hi = min(lo + chunk, k)
            ww = w[lo:hi]
            num = num + (ww[:, None] * Vd[order[lo:hi]]).sum(0)
            den = den + ww.sum()
        prev = k
        oh = num / den.clamp_min(1e-300)
        out[k] = float((oh - o).norm().item() / onorm)
    return out


def _kstar_grid(m: int, points: int) -> list[int]:
    """Geometric ladder of kept-token counts from 1 to m, m always included."""
    if m <= 1:
        return [1]
    n = max(2, int(points))
    ks = {1, m}
    for i in range(1, n):
        ks.add(max(1, int(round(m ** (i / (n - 1))))))
    return sorted(ks)


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
    # THE LADDER IS NOT A PREDICTION -- READ THIS BEFORE QUOTING IT.
    #
    #   log2 a_i = s_i/ln2 - log2 Z        (Z = sum_j exp(s_j), a constant in i)
    #   => std_i(log2 a_i) = std_i(s_i)/ln2 = tau/ln2   IDENTICALLY.
    #
    # So `ladder_bits_a_only` equals tau/ln2 as algebra, not as a measurement:
    # verified to 8e-16 relative error per head across all 24 configurations of
    # the E1/E2 campaign, and pinned by tests/test_units.py::test_ladder_identity.
    # Plotting one against the other plots x against x.
    #
    # `ladder_bits` adds the value term ||v_i - o|| and is therefore the only one
    # of the two with any empirical content -- but that content is small by our
    # own measurement (<=0.035 bits; the two ladders agree to 0.4-1.2%), so
    # "ladder ~ tau/ln2 to ~1%" is a statement about how negligible the value
    # term is, which we already report separately.
    #
    # The theory's real forward prediction is `lin_ratio{B}` in quant_metrics:
    # the linearized cost model against the exactly-recomputed gain. Quote that.
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
                  n95: float | None = None, corner=None,
                  practical_score: torch.Tensor | None = None,
                  interior_raw: dict[str, torch.Tensor] | None = None,
                  interior_unseen: dict[str, torch.Tensor] | None = None) -> dict:
    """EXPENSIVE: needs quantized logits at every bit-width in `shat`.

    The eviction corner is built on TWO axes (see sievelib/evict.py):

      WHO   `practical_scores` maps a corner label to a selection score computed
            WITHOUT the current query -- lagged attention, which is all
            H2O/SnapKV/TOVA/StreamingLLM ever get. The `oracle` corner is added
            here, ranking by the true sensitivity a_i||v_i-o||; it needs the very
            weights it is trying to avoid computing, so it is an upper bound on
            any real evictor rather than a baseline anything can field.

      HOW MUCH  `corner.policies`: `frac` keeps B*L/maxb tokens (the literature's
            definition, linear in L); `abs` caps that at max(kappa*n95, floor),
            a multiple of the head's OWN support, and records the bits it
            declined to spend.

    The oracle is ALWAYS measured and keeps its legacy column names, so
    `gain_best<B>` stays exactly the v7 quantity and the monotonicity check
    (`gain_best_practical >= gain_best`) is computable row by row in one frame.
    The VERDICT columns are the `*_practical` family, which never see the oracle.

    `practical_score` (singular tensor) is the pre-registry spelling, kept so
    older callers and tests still work; it is labelled "practical".
    """
    global _WARNED_RANKED_INTERIOR
    L = s.numel()
    if L < 128 or not shat:
        return {}
    if corner is None:
        corner = _EV.CornerSpec()
    scores = dict(practical_scores or {})
    if practical_score is not None:
        scores.setdefault("practical", practical_score)
    for nm, ps in scores.items():
        if nm == _EV.ORACLE:
            raise ValueError("'oracle' is computed here, not supplied -- it "
                             "needs the current step's attention")
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

    # R3 / E2b -- the LAGGED SENSITIVITY, for a practical interior.
    #
    # w2 = (a*||v-o||)^2 uses the CURRENT step's attention `a`, which is the
    # oracle information the corner demotion took away. The deployable analogue
    # rebuilds the same functional from a lagged score: normalise it to a
    # distribution `ap`, form the lagged output `op = ap @ V` -- computable from
    # the KV cache alone, nothing from the current query -- and take
    # w2p = (ap*||v-op||)^2. The allocation DECISION is then lagged while the
    # reported error stays exact recomputation against the true logits, which is
    # precisely the asymmetry a deployed system faces.
    # `interior_raw` carries the same scores WITHOUT score()'s ordinal
    # "never evict at birth" bump. The bump is correct for a corner (it is a
    # ranking) and wrong here (this is a magnitude): normalised, it hands a
    # handful of fresh positions 20-33% of the distribution, worst on the
    # concentrated heads the interior cares about. Fall back to the ranked score
    # only for old callers, and say so, because the fallback is biased.
    raw = dict(interior_raw or {})
    # Positions each lagged score knows NOTHING about (Evictor.unseen): held
    # at the top tier below, never water-filled on their score of 0. Callers
    # that pass no mask get `raw <= 0`, which is exactly those positions for an
    # attention history (a softmax is never exactly 0 where it was observed).
    unseen_in = dict(interior_unseen or {})
    unseen: dict[str, torch.Tensor] = {}
    w2p: dict[str, torch.Tensor] = {}
    for nm in corner.interior_scores:
        ps = raw.get(nm)
        if ps is None:
            ps = scores.get(nm)
            if ps is not None and not _WARNED_RANKED_INTERIOR:
                _WARNED_RANKED_INTERIOR = True
                import warnings
                warnings.warn(
                    f"interior score {nm!r} came from the RANKED tensor; the "
                    f"freshness bump inflates w2p. Pass interior_raw=.",
                    RuntimeWarning, stacklevel=2)
        if ps is None:
            continue
        ap = ps.double().clamp_min(0)
        tot = float(ap.sum().item())
        if not (tot > 0) or not math.isfinite(tot):
            continue                  # a flat/degenerate score cannot allocate
        ap = ap / tot
        op = ap @ Vd
        w2p[nm] = (ap * (Vd - op).norm(dim=-1)) ** 2
        u = unseen_in.get(nm)
        if u is None:
            u = ps.double() <= 0
        if u.numel() != L:
            raise ValueError(f"interior_unseen[{nm!r}] has {u.numel()} entries, "
                             f"the head has {L} live positions")
        unseen[nm] = u.to(w2.device, torch.bool)

    # Corner rankings. The oracle is a CONFIGURED corner like any other (listed
    # by default); it is the only one that sees the current step.
    orc = corner.oracle_label
    rank = {}
    if orc is not None:
        rank[orc] = torch.argsort(w2, descending=True)
    for nm, ps in scores.items():
        rank[nm] = torch.argsort(ps.double(), descending=True)
    # Corner ranked by the lagged SENSITIVITY rather than raw lagged attention:
    # apples-to-apples with the w2p interior (see CornerSpec.interior_rank_corner).
    if corner.interior_rank_corner:
        for nm, wp in w2p.items():
            # unseen positions first, as the raw-score corner's bump does
            wr = wp.clone()
            if bool(unseen[nm].any()):
                wr[unseen[nm]] = float(wp.max().item()) + 1.0
            rank[f"{nm}_w2p"] = torch.argsort(wr, descending=True)
    have_maxb = int(maxb) in shat
    out["n_practical"] = len(scores)

    cv = _cost_vector(sig2, maxb, w2.device)
    for B in budgets:
        if int(B) not in shat:
            continue
        bw = waterfill(w2, sig2, float(B), maxb)
        e_wf = exact_error(s, shat, V, bw, o)
        e_un = exact_error(s, shat, V, torch.full_like(bw, int(B)), o)
        out[f"err_wf{B}"] = e_wf
        out[f"err_uniform{B}"] = e_un
        out[f"gain_u{B}"] = e_un / max(e_wf, 1e-12)
        out[f"evict_frac{B}"] = float((bw == 0).double().mean().item())
        out[f"mean_bits{B}"] = float(bw.double().mean().item())
        pred = math.sqrt(
            float((w2 * cv[torch.full_like(bw, int(B))]).sum().item())
            / max(float((w2 * cv[bw]).sum().item()), 1e-300))
        out[f"lin_ratio{B}"] = pred / max(out[f"gain_u{B}"], 1e-12)
        # R3 / E2b: the practical interior. One waterfill + one exact_error per
        # (budget, score). `err_wf_pp` is strictly >= err_wf by construction --
        # the lagged allocator is choosing from strictly less information -- so
        # the gap between them prices what the interior pays for being honest.
        for nm, wp in w2p.items():
            bwp = waterfill_floor(wp, sig2, float(B), maxb, unseen[nm])
            e_wf_pp = exact_error(s, shat, V, bwp, o)
            out[f"err_wf_pp{B}_{nm}"] = e_wf_pp
            out[f"interior_lag_cost{B}_{nm}"] = e_wf_pp / max(e_wf, 1e-12)
            out[f"evict_frac_pp{B}_{nm}"] = float((bwp == 0).double().mean().item())
            out[f"unseen_frac_pp{B}_{nm}"] = float(unseen[nm].double().mean().item())
            out[f"gain_u_pp{B}_{nm}"] = e_un / max(e_wf_pp, 1e-12)

        if not have_maxb or not rank:
            continue                      # no top tier -> no eviction corner

        # How many tokens each policy lets the corner keep, and what that costs.
        # `abs` deliberately spends LESS than the budget allows; recording the
        # spend makes the comparison explicit instead of smuggled.
        mtok = {p: _EV.corner_tokens(p, float(B), L, maxb, n95,
                                     corner.kappa, corner.floor)
                for p in corner.policies}
        for p, mk in mtok.items():
            out[f"corner_tokens{B}_{p}"] = int(mk)
            out[f"corner_bits_used{B}_{p}"] = mk * maxb / L
        pol0 = "frac" if "frac" in mtok else list(mtok)[0]
        mfrac = mtok[pol0]

        # One cumulative pass per ranking covers every policy's K, plus the K*
        # ladder for the oracle -- see evict_error_curve.
        ks_extra = _kstar_grid(mfrac, corner.kstar_points) if corner.kstar else []
        err = {}
        for nm, idx in rank.items():
            need = set(mtok.values()) | (set(ks_extra) if nm == orc else set())
            curve = evict_error_curve(s, shat[int(maxb)], V, idx, need, o)
            err[nm] = curve
            for p, mk in mtok.items():
                e = curve[max(1, min(int(mk), L))]
                out[f"err_e{B}_{nm}_{p}"] = e
                out[f"gain_e{B}_{nm}_{p}"] = e / max(e_wf, 1e-12)

        # Legacy names == oracle at the fractional budget, so `gain_best<B>`
        # stays bit-for-bit the v7 quantity and the rerun remains comparable.
        e_ev = None
        if orc is not None:
            e_ev = out[f"err_e{B}_{orc}_{pol0}"]
            out[f"err_evict{B}"] = e_ev
            out[f"gain_e{B}"] = e_ev / max(e_wf, 1e-12)
            out[f"gain_best{B}"] = min(e_un, e_ev) / max(e_wf, 1e-12)
            out[f"in_band{B}"] = float(out[f"gain_best{B}"] >= BAND_MIN)

        # THE VERDICT CORNER: strongest DEPLOYABLE corner at the full budget.
        # `min` picks the lowest error, i.e. the hardest competitor -- the
        # conservative direction, so an in-band verdict cannot be dismissed as
        # having been measured against a weak evictor.
        #
        # ONLY when EVERY configured evictor scored. On the first decode step the
        # lagged evictors have no history, but `recency` needs none -- so the min
        # would collapse onto the one corner that is always available and happens
        # to be the weakest, and `gain_best_practical` would measure the interior
        # against StreamingLLM alone. Measured cost of not doing this: the band
        # fraction came out 29 points too high on average (up to +49), and the
        # band-vs-ctx curve turned NON-MONOTONE, bending back up at 128k.
        # A campaign without `recency` never saw it, because then nothing scored
        # at step 0 and these columns were absent -- which is why round 1
        # (accum-only) agrees with round 2 filtered to 0.5 pts and with round 2
        # unfiltered to 29. NaN here, so a median/dropna excludes the row.
        _complete = bool(scores) and (set(corner.practical).issubset(scores)
                                      or set(scores) == {"practical"})
        if _complete:
            e_best, who = None, ""
            for nm in scores:
                e_p = out[f"err_e{B}_{nm}_{pol0}"]
                if e_ev is not None:
                    out[f"oracle_evict_advantage{B}_{nm}"] = e_p / max(e_ev, 1e-12)
                if e_best is None or e_p < e_best:
                    e_best, who = e_p, nm
            out[f"err_practical{B}"] = e_best
            out[f"gain_practical{B}"] = e_best / max(e_wf, 1e-12)
            out[f"gain_best_practical{B}"] = min(e_un, e_best) / max(e_wf, 1e-12)
            out[f"in_band_practical{B}"] = float(
                out[f"gain_best_practical{B}"] >= BAND_MIN)
            out[f"best_evictor{B}"] = who
            if e_ev is not None:
                out[f"oracle_evict_advantage{B}"] = e_best / max(e_ev, 1e-12)

            # THE SYMMETRIC CELL: both sides decide from lagged information.
            # This is the only comparison in the study that a reviewer cannot
            # attribute to information asymmetry, so it is the honest headline.
            # `_pp` = practical interior AND practical corner.
            for nm, wp in w2p.items():
                e_wf_pp = out[f"err_wf_pp{B}_{nm}"]
                # Corner ranked the literature's way (raw lagged attention)...
                out[f"gain_pp{B}_{nm}"] = min(e_un, e_best) / max(e_wf_pp, 1e-12)
                out[f"in_band_pp{B}_{nm}"] = float(
                    out[f"gain_pp{B}_{nm}"] >= BAND_MIN)
                # ...and ranked by the same w2p the interior used, which removes
                # the last scrap of asymmetry in the OTHER direction.
                k = f"err_e{B}_{nm}_w2p_{pol0}"
                if k in out:
                    e_sym = min(e_best, out[k])
                    out[f"gain_pp_sym{B}_{nm}"] = min(e_un, e_sym) / max(e_wf_pp, 1e-12)
                    out[f"in_band_pp_sym{B}_{nm}"] = float(
                        out[f"gain_pp_sym{B}_{nm}"] >= BAND_MIN)

        # E1's policy-free diagnostic: the smallest kept-token count that gets
        # within kstar_tol of the FULL-budget corner. K* << B*L/maxb means the
        # corner was winning on slack the budget handed it, not on merit.
        if corner.kstar and ks_extra and orc is not None:
            curve = err[orc]
            thr = (1.0 + corner.kstar_tol) * curve[max(1, min(int(mfrac), L))]
            ks_ok = [k for k in sorted(curve) if k <= mfrac and curve[k] <= thr]
            kstar = ks_ok[0] if ks_ok else int(mfrac)
            out[f"kstar{B}"] = int(kstar)
            out[f"kstar_frac{B}"] = kstar / max(int(mfrac), 1)
            if n95 is not None and math.isfinite(float(n95)) and float(n95) > 0:
                out[f"kstar_over_n95{B}"] = kstar / float(n95)
    return out


def head_metrics(s, shat, V, budgets=(1, 2, 3, 4), maxb=8, n_sink=4,
                 practical_scores=None, corner=None, practical_score=None,
                 interior_raw=None, interior_unseen=None, extra=None) -> dict:
    m = sensitivity_metrics(s, V, n_sink)
    if m and shat:
        m.update(quant_metrics(s, shat, V, budgets, maxb,
                               practical_scores=practical_scores,
                               interior_raw=interior_raw,
                               interior_unseen=interior_unseen,
                               n95=m.get("n95"), corner=corner,
                               practical_score=practical_score))
        # co-design (bugs/co-design/plan.md S3). `extra` is None for every
        # caller that has not opted in, so the dict above is returned bit for
        # bit as before -- pinned by test_codesign_invariants (T1).
        if extra is not None and m:
            m.update(extra_metrics(s, shat, V, m, extra, maxb=maxb,
                                   corner=corner))
    return m


# =============================================================================
# CO-DESIGN (h0_measurement/bugs/co-design/plan.md) -- two measurement columns.
#
# Everything below is ADDITIVE: nothing above calls it except the guarded hook
# in head_metrics, and with `extra=None` that hook does not fire. `quant_metrics`
# itself is not edited, so a run without the knobs is unchanged.
#
#   G  GROUP ALLOCATION.  Every gain this project reports is measured per QUERY
#      head, but K is quantized once per KV head (run_h0.py: quantize_keys on
#      K[Hkv], logits_gqa against it) and a tiered cache stores one bit-width per
#      (KV head, token). With n_rep query heads per KV head -- 4 for llama31-8b /
#      mistral-7b / qwen3-8b, 8 for llama33-70b and qwen3-30b, 1 for qwen15-moe --
#      the per-head allocation is not storable. These columns measure what the
#      group constraint costs. ROADMAP R12, and the only open item that can shrink
#      the headline.
#
#   C  CASCADE SCORE.  The honest (lagged) interior sits 3.9-24.0 band points
#      below the oracle-interior cell (R3-cells.csv). The base tier is already
#      quantized at every measured step, so `softmax(shat[bc])` is a CURRENT-query
#      attention estimate available to a deployed system for the cost of reading
#      the base tier. These columns price how much of that gap it recovers.
#
# Column prefixes are `grp_`, `cs_`, `csv_`, `gain_grp`, `gain_cs`, `gain_csv`,
# `err_wf_grp`, `err_wf_cs`, `err_wf_csv`, `err_e<B>_grp`, `err_e<B>_cs` -- none
# of which is matched by report.py's `gain_pp3_` / `gain_e3_` / `corner_bits_used3_`
# scans, boundary.py's `gain_pp`, or drift.py's `interior_lag_cost{B}_first`
# (plan.md B6, pinned by T7).
# =============================================================================


def waterfill_group(w2_stack: torch.Tensor, sig2_list: list[dict],
                    budget_bits: float, maxb: int = 8,
                    floor: torch.Tensor | None = None,
                    iters: int = 60) -> torch.Tensor:
    """ONE allocation for the `G` query heads that share a KV head (plan.md S1).

    `w2_stack` is [G, L]; `sig2_list` holds each head's own noise model. Per
    token the group pays the SUM of its heads' costs, so

        cost_g[i, b] = sum_h w2[h, i] * sig2_h[b]
        b_g[i]       = argmin_b  cost_g[i, b] + lambda * b

    with lambda bisected to spend `budget_bits * L`. That is exactly `waterfill`
    with a per-token cost vector instead of one head's `w2 * sig2` -- and for
    G = 1 it reduces to it bit for bit (`waterfill` divides the Lagrangian by
    w2, which is a positive rescaling of the same argmin), which is what T4
    checks and what makes qwen15-moe (n_rep = 1) the control cell.

    `floor` is the group's UNION of positions no head's score has seen; they are
    held at `maxb` and the rest water-filled on what is left, as in
    `waterfill_floor`.
    """
    if w2_stack.dim() != 2:
        raise ValueError(f"w2_stack must be [G, L], got {tuple(w2_stack.shape)}")
    G, L = w2_stack.shape
    if len(sig2_list) != G:
        raise ValueError(f"{G} heads in the group but {len(sig2_list)} noise models")
    if G == 1:
        # THE n_rep = 1 CONTROL MUST BE EXACT BY CONSTRUCTION, not by numerical
        # luck. `waterfill` minimises cvec[b] + lam*b/w2 while the group form
        # below minimises the same thing multiplied through by w2 -- the same
        # argmin in exact arithmetic, but a*b/a is not a in IEEE754, so the two
        # bisections can part company by one tier on a borderline token (4.5e-6
        # in the reported error when this was measured). qwen15-moe is the only
        # n_rep = 1 model in the registry and it is what validates every group
        # column, so it delegates instead of re-deriving.
        return (waterfill(w2_stack[0], sig2_list[0], budget_bits, maxb)
                if floor is None or not bool(floor.any())
                else waterfill_floor(w2_stack[0], sig2_list[0], budget_bits,
                                     maxb, floor))
    bits = sorted(sig2_list[0])
    for k, s2 in enumerate(sig2_list[1:], 1):
        if sorted(s2) != bits:
            raise ValueError(
                f"head {k} of the group offers bit-widths {sorted(s2)} against "
                f"{bits} for head 0 -- a shared allocation needs a shared tier set")
    dev = w2_stack.device
    bt = torch.tensor(bits, dtype=torch.float64, device=dev)
    # [G, n_bits] -- each head's cost at each tier, so cost = w2.T @ C is [L, n_bits]
    C = torch.tensor([[s2[b] for b in bits] for s2 in sig2_list],
                     dtype=torch.float64, device=dev)

    def _solve(w2s: torch.Tensor, budget: float) -> torch.Tensor:
        n = w2s.shape[1]
        if n == 0:
            return torch.zeros(0, dtype=torch.long, device=dev)
        cost = w2s.double().T @ C                      # [n, n_bits]
        target = budget * n
        lo, hi = 1e-30, 1e30
        feasible = (cost + hi * bt[None, :]).argmin(dim=1)
        for _ in range(iters):
            mid = math.sqrt(lo * hi)
            idx = (cost + mid * bt[None, :]).argmin(dim=1)
            if float(bt[idx].sum().item()) > target:
                lo = mid
            else:
                hi = mid
                feasible = idx
        return bt[feasible].long()

    if floor is None or not bool(floor.any()):
        return _solve(w2_stack, float(budget_bits))
    n_f = int(floor.sum().item())
    out = torch.full((L,), int(maxb), dtype=torch.long, device=dev)
    if n_f < L:
        rest = max(0.0, (float(budget_bits) * L - maxb * n_f) / (L - n_f))
        out[~floor] = _solve(w2_stack[:, ~floor], rest)
    return out


def _sens(a: torch.Tensor, Vd: torch.Tensor, o: torch.Tensor | None = None):
    """(a * ||v - o||)^2 -- the sensitivity the allocator water-fills on, and the
    output `o` it was formed against. `o = a @ V` when not supplied."""
    if o is None:
        o = a @ Vd
    return (a * (Vd - o).norm(dim=-1)) ** 2, o


def _rel(w2: torch.Tensor, o_true: torch.Tensor) -> torch.Tensor:
    """`w2` rescaled so that summing it ACROSS heads minimises the total
    RELATIVE error, which is the quantity this project reports.

    sum_i w2_i * sig2[b_i] is the first-order proxy for the head's squared
    ABSOLUTE output error, while `exact_error` returns ||o_hat - o|| / ||o||.
    Summing raw w2 over a KV group therefore weights each head by ||o_h||^2:
    the head with the largest output norm captures the shared allocation and
    its neighbours are starved, which shows up in exactly the per-head RELATIVE
    numbers the band counts. Dividing by ||o_h||^2 first makes the group
    objective the sum of relative squared errors -- the same thing the reported
    metric measures.

    For a single head this is a positive rescaling of the whole cost, so the
    argmin is unchanged and `waterfill` is unaffected: the n_rep = 1 control
    stays exact either way.
    """
    return w2 / max(float((o_true @ o_true).item()), 1e-300)


def group_prepass(heads: list[dict], n_rep: int, budgets, maxb: int = 8,
                  coarse_bits=(), group: bool = True) -> list[dict]:
    """Per-head `extra` payloads for one LAYER (plan.md S2).

    `heads` is one dict per query head, in head order, each carrying the tensors
    the group allocation needs:

        s       [L] finite logits            shat  {bit: [L]}
        V       [L, d] value rows            raw   {score name: [L]} lagged, unbumped
        unseen  {score name: [L] bool}

    Query head h belongs to KV head `h // n_rep`, which is the mapping
    `quant.logits_gqa` and `run_h0.py`'s `V[h // n_rep]` already use.

    Returns a list aligned with `heads`; element h is passed to `head_metrics`
    as `extra=`. Each carries this head's precomputed `sig2`/`o`/`w2p`/`op` -- so
    `extra_metrics` does NOT recompute them -- plus the allocation its group
    agreed on. The per-head loop in run_h0.py is unchanged; this runs before it.
    """
    n_rep = max(1, int(n_rep))
    budgets = [int(B) for B in budgets]
    coarse_bits = [int(b) for b in coarse_bits]
    per: list[dict] = []
    for h in heads:
        sd, Vd = h["s"].double(), h["V"].double()
        a = torch.softmax(sd, -1)
        w2, o = _sens(a, Vd)
        sig2 = noise_model(h["s"], h["shat"])["sig2"]
        w2p, op, unseen = {}, {}, {}
        for nm, ps in (h.get("raw") or {}).items():
            ap = ps.double().clamp_min(0)
            tot = float(ap.sum().item())
            if not (tot > 0) or not math.isfinite(tot):
                continue
            ap = ap / tot
            w2p[nm], op[nm] = _sens(ap, Vd)
            u = (h.get("unseen") or {}).get(nm)
            unseen[nm] = (ps.double() <= 0) if u is None else u.to(w2.device, torch.bool)
        w2c, w2cv = {}, {}
        for bc in coarse_bits:
            if bc not in h["shat"]:
                raise ValueError(
                    f"coarse_bits {bc} has no quantized logits; add it to bit_list")
            ac = torch.softmax(h["shat"][bc].double(), -1)
            w2c[bc], _ = _sens(ac, Vd)                     # exact o   -- the bound
            for nm, opn in op.items():                     # lagged o  -- deployable
                w2cv[(bc, nm)], _ = _sens(ac, Vd, opn)
        per.append(dict(sig2=sig2, o=o, w2=w2, w2p=w2p, op=op, unseen=unseen,
                        w2c=w2c, w2cv=w2cv, L=int(sd.numel()),
                        # every cross-head sum below is taken on these
                        rw2=_rel(w2, o),
                        rw2p={k: _rel(v, o) for k, v in w2p.items()},
                        rw2cv={k: _rel(v, o) for k, v in w2cv.items()}))

    # Only what extra_metrics actually reads. w2/w2p/op/unseen are inputs to the
    # group allocation and are dropped here: at ctx 131072 a [L] float64 per head
    # is 1 MB, and a 32-head layer would otherwise carry a few hundred MB of dead
    # tensors into the per-head loop.
    out: list[dict] = []
    for h_i in range(len(heads)):
        out.append(dict(budgets=budgets, coarse_bits=coarse_bits, maxb=int(maxb),
                        n_rep=n_rep, kv_head=h_i // n_rep,
                        sig2=per[h_i]["sig2"], o=per[h_i]["o"],
                        w2c=per[h_i]["w2c"], w2cv=per[h_i]["w2cv"], group=None))
    if not group:
        return out

    # One shared allocation per KV group, per mode, per budget. Heads in a group
    # must have the same live positions: the additive mask is ONE [L] vector per
    # layer (probe.py STATE.mask[li], applied to every head in run_h0.py), so a
    # mismatch is structural and must stop the run rather than be padded over.
    for g0 in range(0, len(heads), n_rep):
        grp = list(range(g0, min(g0 + n_rep, len(heads))))
        Ls = {per[i]["L"] for i in grp}
        if len(Ls) != 1:
            raise ValueError(
                f"KV group {g0 // n_rep} has heads with different live lengths "
                f"{sorted(Ls)} -- the group cannot share one allocation")
        sig2s = [per[i]["sig2"] for i in grp]
        bits: dict[str, torch.Tensor] = {}
        rank: dict[str, torch.Tensor] = {}
        # _rel is a per-head POSITIVE RESCALE, so for a one-head group it cannot
        # change the argmin -- but waterfill's bisection is a finite search, and
        # a rescaled w2 makes it visit different lambdas and land one tier away
        # on a borderline token. Skipping it at n_rep = 1 keeps the control cell
        # bit-exact against the per-head path; it only ever matters for a real
        # cross-head sum.
        one = len(grp) == 1
        kw2 = "w2" if one else "rw2"
        kwp = "w2p" if one else "rw2p"
        kwc = "w2cv" if one else "rw2cv"

        # Keys are (name, budget) TUPLES, never formatted strings: with a coarse
        # width and a budget both in the name, "csv_3_accum_3" has two readings
        # and the reader would silently pick one.
        # oracle-information group allocation: the pure cost of the constraint
        # `rw2`, not `w2`: a cross-head sum must be of RELATIVE errors (see _rel).
        W = torch.stack([per[i][kw2] for i in grp])
        for B in budgets:
            bits[("or", B)] = waterfill_group(W, sig2s, float(B), maxb, None)
        rank["or"] = torch.argsort(W.sum(0), descending=True)

        # lagged (deployable today) -- only scores every head in the group has
        common = set.intersection(*[set(per[i]["w2p"]) for i in grp]) if grp else set()
        for nm in sorted(common):
            Wp = torch.stack([per[i][kwp][nm] for i in grp])
            fl = torch.stack([per[i]["unseen"][nm] for i in grp]).any(0)
            for B in budgets:
                bits[(f"pp_{nm}", B)] = waterfill_group(Wp, sig2s, float(B), maxb, fl)
            wr = Wp.sum(0)
            if bool(fl.any()):                 # unseen first, as the corner bumps them
                wr = wr.clone(); wr[fl] = float(wr.max().item()) + 1.0
            rank[f"pp_{nm}"] = torch.argsort(wr, descending=True)

        # group + cascade: the design a cache can actually store
        for bc in coarse_bits:
            for nm in sorted(common):
                key = (bc, nm)
                if not all(key in per[i][kwc] for i in grp):
                    continue
                Wc = torch.stack([per[i][kwc][key] for i in grp])
                for B in budgets:
                    bits[(f"csv_b{bc}_{nm}", B)] = waterfill_group(
                        Wc, sig2s, float(B), maxb, None)

        for i in grp:
            out[i]["group"] = dict(bits=bits, rank=rank, size=len(grp))
    return out


def extra_metrics(s: torch.Tensor, shat: dict[int, torch.Tensor], V: torch.Tensor,
                  base: dict, extra: dict, maxb: int = 8, corner=None) -> dict:
    """The co-design columns for one head (plan.md S3).

    Reads the errors `quant_metrics` already produced from `base` and adds only
    what is new, so nothing here duplicates the per-head path. Every quantity it
    needs beyond that -- sig2, the exact output `o`, the lagged sensitivities --
    arrives precomputed in `extra` from `group_prepass`.

    Aggregate direction only. A constrained allocation CAN beat an unconstrained
    one on a single head, for the same reason the corner's monotonicity is not a
    per-head bound (quant_metrics' note, R3-report 2.6: 9-32% of head-rows already
    violate the analogous claim). No assertion here is per head.
    """
    out: dict[str, object] = {}
    if not extra or not shat:
        return out
    L = s.numel()
    maxb = int(extra.get("maxb", maxb))
    budgets = [int(B) for B in extra.get("budgets", ())]
    coarse_bits = [int(b) for b in extra.get("coarse_bits", ())]
    grp = extra.get("group")
    sig2 = extra.get("sig2") or noise_model(s, shat)["sig2"]
    Vd = V.double()
    o = extra.get("o")
    if o is None:
        o = torch.softmax(s.double(), -1) @ Vd
    # Every corner column below keeps `frac` tokens. That is the policy every
    # configuration in this project runs (`abs` was refuted by E1), and naming
    # it in the column is only honest if it was actually configured -- so when
    # it was not, the corner columns are WITHHELD rather than quietly computed
    # under a policy the rest of the run did not use.
    have_frac = corner is None or "frac" in tuple(corner.policies)
    have_maxb = int(maxb) in shat and have_frac
    if grp:
        out["n_rep"] = int(extra.get("n_rep", 1))
        out["kv_head"] = int(extra.get("kv_head", 0))
        out["grp_size"] = int(grp.get("size", 1))

    for B in budgets:
        if int(B) not in shat:
            continue
        e_wf = base.get(f"err_wf{B}")
        e_un = base.get(f"err_uniform{B}")
        if e_wf is None or e_un is None:
            continue
        e_best = base.get(f"err_practical{B}")      # per-head corner, may be absent

        def _emit(tag: str, bits: torch.Tensor, corner_err=None,
                  need_corner: bool = True):
            """One constrained allocation: its error, what the constraint cost
            against the per-head oracle interior, and the band-style gain.

            The `cost` column needs only `err_wf` and is always emitted. The
            `gain` column is a band statistic and is WITHHELD (not silently
            recomputed against uniform alone) when its corner is missing --
            e.g. on a decode step where a lagged evictor has no history yet,
            which is the case `quant_metrics` handles the same way for
            `gain_best_practical`. A column that quietly changes denominator
            between rows would be averaged across both by any reader."""
            e = exact_error(s, shat, V, bits, o)
            out[f"err_wf_{tag}{B}"] = e
            out[f"{tag}cost{B}"] = e / max(e_wf, 1e-12)
            out[f"evict_frac_{tag}{B}"] = float((bits == 0).double().mean().item())
            if corner_err is not None:
                out[f"gain_{tag}{B}"] = min(e_un, corner_err) / max(e, 1e-12)
            elif not need_corner:
                out[f"gain_u_{tag}{B}"] = e_un / max(e, 1e-12)
            return e

        # ---- G: the group constraint ---------------------------------------
        # The corner must be group-constrained too. A group interior measured
        # against a PER-HEAD corner would compare something a cache can store
        # against something it cannot, and understate the gain (plan.md B2).
        e_grp_practical = None
        if grp:
            if have_maxb and grp["rank"]:
                mk = max(1, min(int(_EV.corner_tokens(
                    "frac", float(B), L, maxb, None,
                    4.0 if corner is None else corner.kappa,
                    256 if corner is None else corner.floor)), L))
                for nm, idx in grp["rank"].items():
                    curve = evict_error_curve(s, shat[int(maxb)], V, idx, {mk}, o)
                    out[f"err_e{B}_grp_{nm}_frac"] = curve[mk]
                    if nm.startswith("pp_"):        # the DEPLOYABLE group corner
                        e_grp_practical = (curve[mk] if e_grp_practical is None
                                           else min(e_grp_practical, curve[mk]))
            for (name, Bk), bits in grp["bits"].items():
                if int(Bk) != int(B):
                    continue
                # `or` is the pure cost of the constraint under ORACLE
                # information: there is no deployable corner to compare it to,
                # so it is reported against uniform, under its own column name
                # (`gain_u_grp_or_<B>`) rather than sharing `gain_` with the
                # corner-based ones.
                if name == "or":
                    _emit("grp_or_", bits, None, need_corner=False)
                else:
                    _emit(f"grp_{name}_", bits, e_grp_practical)

        # ---- C: the cascade -------------------------------------------------
        for bc in coarse_bits:
            w2c = (extra.get("w2c") or {}).get(bc)
            if w2c is None:
                continue
            # the bound: current query, base-tier keys, EXACT o
            bits = waterfill(w2c, sig2, float(B), maxb)
            e_cs = _emit(f"cs_b{bc}_", bits, e_best)
            # ...and the corner ranked by the SAME coarse sensitivity, which is
            # the cascade's symmetric cell (mirrors gain_pp_sym in quant_metrics)
            if have_maxb and e_best is not None:
                mk = max(1, min(int(_EV.corner_tokens(
                    "frac", float(B), L, maxb, None,
                    4.0 if corner is None else corner.kappa,
                    256 if corner is None else corner.floor)), L))
                curve = evict_error_curve(s, shat[int(maxb)], V,
                                          torch.argsort(w2c, descending=True),
                                          {mk}, o)
                out[f"err_e{B}_cs{bc}_frac"] = curve[mk]
                out[f"gain_cs_sym_b{bc}_{B}"] = (
                    min(e_un, e_best, curve[mk]) / max(e_cs, 1e-12))
            # the deployable variant: lagged o, so no V read at the current step
            for (bc2, nm), w2cv in (extra.get("w2cv") or {}).items():
                if bc2 != bc:
                    continue
                _emit(f"csv_b{bc}_{nm}_", waterfill(w2cv, sig2, float(B), maxb), e_best)

        # ---- the GROUPING's OWN marginal cost --------------------------------
        # `grp_pp_*cost` divides by err_wf, so it carries the LAG cost as well as
        # the grouping, and the lag tail is heavy (per-head lag reads max 172x on
        # 448 synthetic-prompt head-rows while grouping alone reads max 3.8x).
        # Dividing by the matching PER-HEAD allocation isolates what the shared
        # allocation actually costs, which is the R12 question. Measured at
        # n_rep = 2: median 1.005 against 1.165 for the conflated ratio.
        if grp:
            for (name, Bk) in list(grp["bits"]):
                if int(Bk) != int(B) or name == "or":
                    continue
                num = out.get(f"err_wf_grp_{name}_{B}")
                if name.startswith("pp_"):
                    den = base.get(f"err_wf_pp{B}_{name[3:]}")
                elif name.startswith("csv_"):
                    den = out.get(f"err_wf_csv_{name[4:]}_{B}")
                else:
                    den = None
                if num is not None and den:
                    out[f"grp_{name}_over_head{B}"] = num / max(den, 1e-12)
    return out

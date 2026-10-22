#!/usr/bin/env python3
"""G0c (ARCHITECTURES_PLAN.md §4–§5): the MLA map in the single store's form. CPU only.

MLA caches one row per token, the latent x_i = [c_i, k_rope,i] (blocks 0 and 1). Head h's score is
sc <q'_h, x_i> with the absorbed query q'_h = [W_UK,h^T q_nope,h, q_rope,h], and its value is W_UV,h c_i. One store
therefore serves the keys and the values.

Lemma N_lat (Lemma N for a latent). Store each block of x_i as the nested coin-free store (theory.md §13: a
deterministic 4-bit Lloyd-Max base plus an r-bit subtractive-dithered residual on the base cell). Given the base
indices, the residual errors are independent across coordinates, blocks and rows, uniform on +-D_bij / 2 and
independent of the data: Lemma N's proof uses nothing about how the row is read. Hence
  - head h's score error is delta_hi = -sum_b <beta_hb, eps_bi>, beta_hb = sc R_b q'_hb: Theorem N's per-coordinate
    coin quantities, summed over the blocks;
  - head h's value error is A_h eps_0i with A_h = W_UV,h R_0^T: it shares eps_0 with delta (coupled, as MLA's uniform
    tier, mla_m_s1h5);
  - (e, delta) is jointly symmetric (eps -> -eps), the one property cert_s1h5.tail_bound_rand_coupled uses beyond the
    per-row inputs, which have per-coordinate forms: sigma^2 = sum_bj beta_j^2 D_j^2 / 12, the a.s. bound
    sum_bj |beta_j| D_j / 2, the coin moments' A2 and A4, E|W_h e|^2 = sum_j |A_:j|^2 D_0j^2 / 12 and
    |W_h e| <= min(|W_h|_2 sqrt(sum_j D_0j^2) / 2, sum_j |A_:j| D_0j / 2);
  - the vote reads sc <q'_h, base(x_i)>, coin-free: R1 holds. In the question pass a layer's dithered latent values
    reach only higher layers, whose votes read other coins (Lemma C's filtration), as for GQA.
Theorem M_lat: mla_m_s1h5.second_moment_mla with per-coordinate half-widths a_ij = D_0ij / 2. Every per-coordinate
tilted moment (m1, m2, v2) is a closed form in a_ij beta_j, and the proof is unchanged. theorem_m_mla evaluates it by
blocks of rows without [head-steps, rows, d] tensors.

    python g0mla_s1h5.py capture --model dsv2lite [--L 32768] [--prompts ..] [--no-controls]
    python g0mla_s1h5.py analyze <view.pt> [...] [--draws 2]
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g0map_s1h5 as G  # noqa: E402

K, RQ = G.K, G.RQ
DT = torch.float64
MLA_L = {"dsv2lite": 32768, "moonlight": 8192}     # Moonlight's max_position_embeddings is 8,192
LAT_R, LAT_U = (1, 2, 3, 4), (4, 5, 6, 8)
ROPE = ("same", "u8", "exact")


# ------------------------------------------------------------------ capture
def mla_view(meta: dict, li: int, module, query, key_, value, latent, k_rot) -> dict:
    """One MLA layer's view: absorbed queries [1, H, 48, d_c + d_R], latent rows [1, L, d_c + d_R], W_UV [1, H, dv, d_c],
    blocks, and checks: the absorption against the module's own keys and values, and the reference output of the 48
    queries from the module's attention inputs."""
    H = query.shape[1]
    L = query.shape[2]
    nope, dv, rank = module.qk_nope_head_dim, module.v_head_dim, module.kv_lora_rank
    Wkv = module.kv_b_proj.weight.detach().float().view(H, nope + dv, rank)
    W_UK, W_UV = Wkv[:, :nope], Wkv[:, nope:]
    c, kr = latent[0, 0].float(), k_rot[0, 0].float()                 # [L, rank], [L, rope]
    q48 = query[0, :, -G.N_FIX:].float()                               # [H, 48, nope + rope]
    q_abs = torch.einsum("hnd,hdc->hnc", q48[..., :nope], W_UK)
    qv = torch.cat([q_abs, q48[..., nope:]], -1).unsqueeze(0)
    X = torch.cat([c, kr], -1).unsqueeze(0)
    rows = torch.arange(max(0, L - 512), L)
    k_exp = torch.cat([torch.einsum("hdc,lc->hld", W_UK, c[rows]), kr[rows].unsqueeze(0).expand(H, -1, -1)], -1)
    v_exp = torch.einsum("hdc,lc->hld", W_UV, c[rows])
    sc = float(meta["scaling"])
    s = torch.einsum("hnd,hld->hnl", q48, key_[0].float()) * sc
    pos = torch.arange(L - G.N_FIX, L)
    s = s.masked_fill(torch.arange(L).view(1, 1, L) > pos.view(1, -1, 1), float("-inf"))
    ref = torch.einsum("hnl,hld->hnd", torch.softmax(s, -1), value[0].float()).unsqueeze(0)
    return dict(meta, family="mla", layer=li, L=L, q=qv.clone(), K=X.clone(), V=c.unsqueeze(0).clone(),
                W=W_UV.unsqueeze(0).clone(), blocks=[(0, rank), (rank, rank + kr.shape[-1])], v_is_k=True, pos=pos,
                ref=ref.clone(), sink=None,
                checks=dict(absorb_k=float((k_exp - key_[0, :, rows].float()).norm() / key_[0, :, rows].float().norm()),
                            absorb_v=float((v_exp - value[0, :, rows].float()).norm() / value[0, :, rows].float().norm())))


def capture(key: str, L: int | None = None, prompts=G.PROMPTS, controls: bool = True, layers=None, k_load=None) -> dict:
    """g0map_s1h5.capture for an MLA model: the latent from each wanted layer's expand_kv, the queries from the exact
    attention's inputs."""
    import adapters_s1h5 as A
    from transformers import AutoConfig
    L = L or MLA_L.get(key, G.L_MAIN)
    repo = A.MODELS[key][0]
    cfg = AutoConfig.from_pretrained(repo)
    n_layers = cfg.num_hidden_layers
    k_load = k_load or n_layers
    want = [li for li in range(k_load) if layers is None or li in layers]
    runs = [(p, L, True) for p in prompts] + (list(G.CONTROLS) if controls else [])
    todo = [r for r in runs if not all(os.path.exists(G.view_path(key, G.tag_of(*r), li)) for li in want)]
    man_path = os.path.join(G.VIEWS, key, "manifest.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else dict(model=key, repo=repo, runs={})
    if not todo:
        print(f"{key}: every view on disk", flush=True)
        return man
    enc = G.encoder(key)
    t0 = time.time()
    model = A._load(key, k=k_load).eval()
    prov = G.provenance(model, key)
    mods = G.text_attn_modules(model)
    if not all(hasattr(mods[li], "expand_kv") for li in want):
        raise RuntimeError(f"{key}: not every wanted layer has expand_kv (MLA)")
    import transformers
    man.update(n_layers=n_layers, layers_loaded=k_load, want=want, t_load=round(time.time() - t0, 1),
               transformers=transformers.__version__, first_token=enc[1], first_token_str=enc[2], provenance=prov,
               label=A.MODELS[key][2])
    lat = {}
    undo = []
    for li in want:
        m = mods[li]
        orig = m.expand_kv

        def ewrap(kv_nope, k_rot, _orig=orig, _li=li):
            lat[_li] = (kv_nope.detach(), k_rot.detach())
            return _orig(kv_nope, k_rot)
        m.expand_kv = ewrap
        undo.append(m)
    try:
        for p, Lp, pre in todo:
            tag = G.tag_of(p, Lp, pre)
            ids, spec = G.prompt_ids(key, p, Lp, pre, enc=enc)
            meta = dict(model=key, repo=repo, **spec, tag=tag)
            written = []

            def on_layer(li, module, query, key_, value, kw, _meta=meta, _written=written, _tag=tag):
                if li not in want:
                    return
                sc = G._scaling(module, kw.get("scaling"))
                rec = mla_view(dict(_meta, scaling=sc), li, module, query, key_, value, *lat.pop(li))
                G._atomic_save(rec, G.view_path(key, _tag, li))
                _written.append(li)
            t1 = time.time()
            with G.exact_hooks(model, on_layer) as calls:
                with torch.no_grad():
                    model(input_ids=ids, use_cache=False, logits_to_keep=1)
            bad = [li for li in sorted(mods) if calls[li] != 1]
            if bad or sorted(written) != want:
                raise RuntimeError(f"{key} {tag}: calls {dict(calls)}; views {sorted(written)} != {want}")
            man["runs"][tag] = dict(spec, t_forward=round(time.time() - t1, 1), n_views=len(written))
            G._atomic_json(man, man_path)
            print(f"{key} {tag}: forward {time.time() - t1:.0f} s, {len(written)} views", flush=True)
    finally:
        for m in undo:
            del m.expand_kv
    return man


# ------------------------------------------------------------------ the analysis
def setup(path: str, fracs=G.FRACS) -> dict:
    """{frac: S} for one MLA view, in g0map_s1h5.setup's layout with G = 1 and r = H: latent-space quantities (Vc = the
    latents c_i, o and c the latent outputs) plus W (head maps). The read set is the store's vote over the base latent
    (each block's nested base, rotations 50 + b as mla_m_s1h5 and adapters_s1h5), softmax with the fixed rows."""
    import adapters_s1h5 as A
    import nested_s1h5 as N
    from sievelib import quant as Q
    d0 = torch.load(path, weights_only=False)
    if d0.get("family") != "mla":
        raise SystemExit("MLA views only")
    q, X, W = d0["q"].to(DT), d0["K"].to(DT), d0["W"].to(DT)
    blocks = [tuple(b) for b in d0["blocks"]]
    pos, sc, L = d0["pos"], float(d0["scaling"]), int(d0["L"])
    G_, r, n_all, D = q.shape
    dc = blocks[0][1]
    C = L - n_all
    s_all = torch.einsum("grnd,gld->grnl", q, X) * sc
    s_all = s_all.masked_fill(~(torch.arange(L).view(1, 1, 1, L) <= pos.view(1, 1, n_all, 1)), K.NEG)
    s_ctx, sf = s_all[..., :C], s_all[..., C:]
    Xc = X[:, :C]
    Vc = Xc[..., :dc]
    lf_all = torch.logsumexp(sf, -1)
    of_all = torch.einsum("grnf,gfd->grnd", torch.softmax(sf, -1), X[:, C:, :dc])
    Rs = [Q.random_rotation(b - a, "cpu", DT, 50 + i) for i, (a, b) in enumerate(blocks)]
    Xb = torch.cat([N.nested_quantize(Xc[..., a:b], 4, 1, Rs[i], 0)["Xb"] for i, (a, b) in enumerate(blocks)], -1)
    slv = torch.einsum("grnd,gcd->grnc", q[:, :, :G.N_VOTE], Xb) * sc
    pv = torch.exp(slv - torch.logaddexp(torch.logsumexp(slv, -1), lf_all[:, :, :G.N_VOTE]).unsqueeze(-1))
    score = pv.sum((1, 2))
    head = lambda x: torch.einsum("grod,grnd->grno", W, x)              # noqa: E731
    checks = dict(d0.get("checks") or {})
    if d0.get("ref") is not None:
        o48 = A._attend(s_ctx, Vc, lf_all, of_all)
        ref = d0["ref"].to(DT)
        checks["recon"] = float(((head(o48) - ref).norm(dim=-1) / ref.norm(dim=-1).clamp_min(1e-12)).max())
    ms = slice(G.N_VOTE, n_all)
    s, lf, of, qm = s_ctx[:, :, ms], lf_all[:, :, ms], of_all[:, :, ms], q[:, :, ms]
    n = n_all - G.N_VOTE
    o = A._attend(s, Vc, lf, of)
    X8 = A._fp8_blocks(Xc.float(), blocks).to(DT)
    o8 = A._attend(torch.einsum("grnd,gcd->grnc", qm, X8) * sc, X8[..., :dc], lf, of)
    e8 = (head(o8) - head(o)).norm(dim=-1)
    Wn = torch.linalg.matrix_norm(W, ord=2)                              # [1, H]
    A0 = torch.einsum("grod,jd->groj", W, Rs[0])                         # W_h R_0^T
    base = dict(q=qm, X=Xc, Vc=Vc, s=s, lf=lf, of=of, o=o, e8=e8, W=W, Wn=Wn, A0=A0, colA2=(A0 * A0).sum(-2),
                Rs=Rs, blocks=blocks, sc=sc, G=G_, r=r, n=n, C=C, D=D, dc=dc, L=L, layer=int(d0.get("layer", -1)),
                model=d0.get("model"), tag=d0.get("tag"), checks=checks, head=head)
    out = {}
    for f in fracs:
        v_ = K.topk_keep(score, C // f)
        km = v_.view(G_, 1, 1, C)
        out[f] = dict(base, km=km, keep=v_.unsqueeze(1).expand(G_, n, C), frac=f,
                      c=A._attend(s.masked_fill(~km, K.NEG), Vc, lf, of))
    return out


def block_steps(S: dict, b: int, spec) -> torch.Tensor | None:
    """Per-coordinate steps [1, C, d_b] of block b: ("store", r) the nested store at 4 + r bits, ("uniform", kb) the
    dithered tier at kb bits (one step per row), ("exact", None) none."""
    import nested_s1h5 as N
    form, x = spec
    a, e = S["blocks"][b]
    Xb = S["X"][..., a:e]
    if form == "exact":
        return None
    if form == "store":
        return N.nested_quantize(Xb, 4, x, S["Rs"][b], 0)["step"]
    y = Xb @ S["Rs"][b].T
    st = 2.0 * y.abs().amax(-1) / (2 ** x - 1)
    return st.unsqueeze(-1).expand(*st.shape, e - a)


def _tilt(x: torch.Tensor) -> tuple:
    """(L(x), T(x)): L = coth x - 1/x (the tilted uniform's mean on [-1, 1]), T = 1 - 2 coth(x)/x + 2/x^2 (its second
    moment); series near 0 (mla_m_s1h5._langevin, _tilt2)."""
    import mla_m_s1h5 as MM
    return MM._langevin(x), MM._tilt2(x)


def block_coin(S: dict, b: int, st) -> dict:
    """Block b's share of the coin quantities, each [1, H, n, C]: l1 = sum_j log sinhc(a_j beta_j), l2 = the same at
    2 a_j beta_j, the moments' A2/A4 bounds (rquant_s1h5.coin_moments_pc's dict), sigma^2 and the a.s. bound; zeros
    for an exact block (st None)."""
    import nested_s1h5 as N
    import theory3_s1h5 as K3
    G_, r, n, C = S["G"], S["r"], S["n"], S["C"]
    shp = (G_, r, n, C)
    if st is None:
        z = torch.zeros(shp, dtype=DT)
        return dict(l1=z, l2=z, mom=dict(lmu_lo=z, lmu_hi=z, lmu2_hi=z, lmu4_hi=z), sig2=z, bas=z)
    a, e = S["blocks"][b]
    qr = S["q"][..., a:e] @ S["Rs"][b].T
    qa = (S["sc"] / 2.0) * qr.abs().reshape(G_, -1, 1, e - a)
    l1 = torch.empty(G_, qa.shape[1], C, dtype=DT)
    l2 = torch.empty_like(l1)
    ch = max(8, (1 << 22) // (G_ * qa.shape[1] * (e - a)))
    for c0 in range(0, C, ch):
        aa = qa * st[:, c0:c0 + ch].unsqueeze(1)
        l1[:, :, c0:c0 + ch] = K3._log_sinhc(aa).sum(-1)
        l2[:, :, c0:c0 + ch] = K3._log_sinhc(2.0 * aa).sum(-1)
    return dict(l1=l1.view(shp), l2=l2.view(shp), mom=N.coin_moments_pc(qr, st, S["sc"]),
                sig2=N.score_sigma_pc(qr, st, S["sc"]) ** 2, bas=N.score_bound_pc(qr, st, S["sc"]))


def combine(*parts) -> dict:
    """Blocks' coin quantities summed (independent blocks): E Z, E Z^2 [1, H, n, C], ldb, mom, sigma, b_as."""
    l1 = sum(p_["l1"] for p_ in parts)
    l2 = sum(p_["l2"] for p_ in parts)
    mom = {k_: sum(p_["mom"][k_] for p_ in parts) for k_ in parts[0]["mom"]}
    ldb = RQ.debias_log(mom)
    return dict(EZ=torch.exp(l1 - ldb), EZ2=torch.exp(l2 - 2.0 * ldb), ldb=ldb, mom=mom,
                sig=torch.sqrt(sum(p_["sig2"] for p_ in parts)), bas=sum(p_["bas"] for p_ in parts))


def coin_sums(S: dict, steps: list) -> tuple:
    """(log E Z + ldb, log E Z^2 + 2 ldb, ldb, mom, sigma, b_as) over every block; steps per block (None: exact)."""
    bc = [block_coin(S, b, st) for b, st in enumerate(steps)]
    cb = combine(*bc)
    return (sum(p_["l1"] for p_ in bc), sum(p_["l2"] for p_ in bc), cb["ldb"], cb["mom"], cb["sig"], cb["bas"])


def theorem_m_mla_multi(S: dict, D0, coins: dict, masks: dict, chunk: int = 64) -> dict:
    """Theorem M_lat for one latent block (steps D0 [1, C, dc], or None: exact) under several score-noise totals
    (coins: name -> combine(...) over every block) and read sets (masks: name -> unread [1, H, n, C]), in one pass over
    blocks of rows: the latent's tilted moments m1, m2, v2 and A_h m1, A_h m2 are computed once. Returns
    {(coin name, mask name): (E|N|^2 / Z^2 [1, H, n], parts)} in head space (mla_m_s1h5.second_moment_mla's terms)."""
    G_, r, n, C = S["G"], S["r"], S["n"], S["C"]
    W, A0, colA2 = S["W"], S["A0"], S["colA2"]
    lZ = torch.logaddexp(torch.logsumexp(S["s"], -1), S["lf"])
    w0 = torch.exp(S["s"] - lZ.unsqueeze(-1))
    head_c = torch.einsum("grod,gcd->grco", W, S["Vc"])                  # W_h c_i [1, H, C, do]
    head_o = torch.einsum("grod,grnd->grno", W, S["o"])                  # W_h o   [1, H, n, do]
    beta0 = S["sc"] * (S["q"][..., :S["dc"]] @ S["Rs"][0].T)            # [1, H, n, dc]
    keys = [(cn, mn) for cn in coins for mn in masks]
    ws = {mn: w0 * m for mn, m in masks.items()}
    acc = {k_: dict(var=torch.zeros(G_, r, n, dtype=DT), m1=torch.zeros(G_, r, n, S["dc"], dtype=DT),
                    key=torch.zeros(G_, r, n, dtype=DT), value=torch.zeros(G_, r, n, dtype=DT),
                    cross=torch.zeros(G_, r, n, dtype=DT)) for k_ in keys}
    for c0 in range(0, C, chunk):
        sl = slice(c0, c0 + chunk)
        u = head_c[:, :, None, sl] - head_o.unsqueeze(-2)                # [1, H, n, c, do]
        u2 = (u * u).sum(-1)
        if D0 is not None:
            a0 = (D0[:, sl] / 2.0).unsqueeze(1).unsqueeze(1)             # [1, 1, 1, c, dc]
            x1 = a0 * beta0.unsqueeze(-2)
            Lx1, _ = _tilt(x1)
            Lx2, Tx2 = _tilt(2.0 * x1)
            m1 = a0 * Lx1
            m2 = a0 * Lx2
            v2 = a0 * a0 * Tx2 - m2 * m2
            Am1 = torch.einsum("grod,grncd->grnco", A0, m1)
            Am2 = torch.einsum("grod,grncd->grnco", A0, m2)
            P1, P2 = (u * Am1).sum(-1), (u * Am2).sum(-1)
            M1 = (Am1 * Am1).sum(-1)
            Q2 = torch.einsum("grncd,grd->grnc", v2, colA2) + (Am2 * Am2).sum(-1)
        for cn, cb in coins.items():
            ez, ez2 = cb["EZ"][..., sl], cb["EZ2"][..., sl]
            for mn in masks:
                wc = ws[mn][..., sl]
                w2 = wc * wc
                a_ = acc[(cn, mn)]
                kq = (1.0 - 2.0 * ez + ez2) * u2
                if D0 is None:
                    eyy, eyq = kq, (1.0 - ez) ** 2 * u2
                    crs = torch.zeros_like(kq)
                    ze2 = torch.zeros_like(kq)
                else:
                    crs = -2.0 * (ez * P1 - ez2 * P2)
                    ze2 = ez2 * Q2
                    eyy = kq + crs + ze2
                    eyq = (1.0 - ez) ** 2 * u2 - 2.0 * (1.0 - ez) * ez * P1 + ez * ez * M1
                    a_["m1"] += torch.einsum("grnc,grncd->grnd", wc * ez, m1)
                a_["var"] += (w2 * (eyy - eyq)).sum(-1)
                a_["key"] += (w2 * kq).sum(-1)
                a_["value"] += (w2 * ze2).sum(-1)
                a_["cross"] += (w2 * crs).sum(-1)
    out = {}
    for cn, cb in coins.items():
        for mn in masks:
            a_ = acc[(cn, mn)]
            aw = ws[mn] * (1.0 - cb["EZ"])                                 # sum_i a_i W_h (c_i - o), in closed form
            EY = torch.einsum("grod,grnd->grno", W, torch.einsum("grnc,gcd->grnd", aw, S["Vc"])) \
                - aw.sum(-1, keepdim=True) * head_o
            if D0 is not None:
                EY = EY - torch.einsum("grod,grnd->grno", A0, a_["m1"])
            tot = a_["var"] + (EY * EY).sum(-1)
            out[(cn, mn)] = (tot, dict(key=a_["key"], value=a_["value"], cross=a_["cross"], mean=(EY * EY).sum(-1)))
    return out


def theorem_m_mla(S: dict, unread: torch.Tensor, steps: list, chunk: int = 64, parts: bool = False):
    """Theorem M_lat for one store configuration (steps per block, None = exact) and one read set."""
    cb = combine(*[block_coin(S, b, st) for b, st in enumerate(steps)])
    tot, pt = theorem_m_mla_multi(S, steps[0], {"c": cb}, {"m": unread}, chunk)[("c", "m")]
    return (tot, pt) if parts else tot


def traffic_mla(S: dict, lat: tuple, rope: str, frac: int) -> float:
    """Bits read per row and step against the FP16 latent row (d_c + d_R elements), R6's model: tier 1 for the unread
    rows, FP16 for the read 1/frac."""
    dc, dR = S["dc"], S["D"] - S["dc"]
    bc = 4 + lat[1] if lat[0] == "store" else lat[1]
    bR = {"same": bc, "u8": 8, "exact": 16}[rope]
    return (1.0 - 1.0 / frac) * (dc * bc + dR * bR) / (16.0 * (dc + dR)) + 1.0 / frac


def rope_steps(S: dict, lat: tuple, rope: str):
    if rope == "exact":
        return None
    return block_steps(S, 1, lat if rope == "same" else ("uniform", 8))


def grid(Sf: dict) -> tuple:
    """Theorem M_lat's grid: per (latent form and width, rope block, 1/frac) the median, p90 and share above 1 of the
    predicted RMS error over FP8's in head space, the key, value and coupling shares, and the traffic."""
    S8 = Sf[8]
    G_, r, n, C = S8["G"], S8["r"], S8["n"], S8["C"]
    e8 = S8["e8"].clamp_min(1e-30)
    lZ = torch.logaddexp(torch.logsumexp(S8["s"], -1), S8["lf"])
    w = torch.exp(S8["s"] - lZ.unsqueeze(-1))
    head_c = torch.einsum("grod,gcd->grco", S8["W"], S8["Vc"])
    head_o = torch.einsum("grod,grnd->grno", S8["W"], S8["o"])
    u2 = ((head_c * head_c).sum(-1).unsqueeze(2) - 2.0 * torch.einsum("grno,grco->grnc", head_o, head_c)
          + (head_o * head_o).sum(-1, keepdim=True)).clamp_min(0.0)
    unr = {f: (~Sf[f]["km"]).expand(G_, r, n, C) for f in Sf}
    flat = {f: dict(unread_second_moment_share=float(((w * unr[f]) ** 2 * u2).sum(-1).div((w * w * u2).sum(-1)
                                                                                           .clamp_min(1e-300)).median()),
                    unread_mass=float((w * unr[f]).sum(-1).median())) for f in Sf}
    rope_cache = {}

    def rope_coin(spec):
        if spec not in rope_cache:
            rope_cache[spec] = block_coin(S8, 1, block_steps(S8, 1, spec))
        return rope_cache[spec]
    res, rat, preds = {}, {}, {}
    for lat in [("store", x) for x in LAT_R] + [("uniform", x) for x in LAT_U]:
        D0 = block_steps(S8, 0, lat)
        lc = block_coin(S8, 0, D0)
        coins = {rope: combine(lc, rope_coin(spec)) for rope, spec in
                 (("same", lat), ("u8", ("uniform", 8)), ("exact", ("exact", None)))}
        mt = theorem_m_mla_multi(S8, D0, coins, unr)
        for (rope, f), (tot, pt) in mt.items():
            rr = tot.clamp_min(0.0).sqrt() / e8
            nm = f"{lat[0]}:{lat[1]}/{rope}/{f}"
            den = (pt["key"] + pt["value"]).clamp_min(1e-300)
            res[nm] = dict(med=float(rr.median()), p90=G._q(rr, 0.9), above=float((rr > 1).double().mean()),
                           key_share=float((pt["key"] / den).median()),
                           cross_share=float((pt["cross"].abs() / den).median()),
                           traffic=traffic_mla(S8, lat, rope, f),
                           agg=float(tot.clamp_min(0.0).sum() / (S8["e8"] ** 2).sum().clamp_min(1e-300)))
            rat[nm] = rr.flatten().float()
            if f == 8:
                preds[nm] = tot
        del lc, coins, mt, D0
    return dict(flat=flat, cfg=res, fp8_err_median=float(S8["e8"].median())), rat, preds


def draw(S: dict, sd: int, lat: tuple, rope: str) -> dict:
    """One draw of the latent store (each block separately; the rope block per `rope`) and the unbiased tail."""
    import adapters_s1h5 as A
    import nested_s1h5 as N
    Xh = S["X"].clone()
    steps = []
    for b, spec in enumerate((lat, lat if rope == "same" else (("uniform", 8) if rope == "u8" else ("exact", None)))):
        a, e = S["blocks"][b]
        if spec[0] == "exact":
            steps.append(None)
            continue
        if spec[0] == "store":
            nz = N.nested_quantize(S["X"][..., a:e], 4, spec[1], S["Rs"][b], 1000003 * sd + 1 + b)
            Xh[..., a:e], st = nz["Xh"], nz["step"]
        else:
            Xh[..., a:e], st1 = RQ.dither_quantize(S["X"][..., a:e], spec[1], S["Rs"][b], 1000003 * sd + 1 + b)
            st = st1.unsqueeze(-1).expand(*st1.shape, e - a)
        steps.append(st)
    cb = combine(*[block_coin(S, b, st) for b, st in enumerate(steps)])
    ldb, mom, sig, bas = cb["ldb"], cb["mom"], cb["sig"], cb["bas"]
    sh = torch.einsum("grnd,gcd->grnc", S["q"], Xh) * S["sc"]
    shd = sh - ldb
    Vh = Xh[..., :S["dc"]]
    oT = A._mix(S["s"], shd, S["Vc"], Vh, S["keep"], S["lf"], S["of"])
    return dict(Xh=Xh, Vh=Vh, shd=shd, ldb=ldb, mom=mom, sig=sig, bas=bas, steps=steps, oT=oT,
                err=(S["head"](oT) - S["head"](S["o"])).norm(dim=-1))


def cert(S: dict, D: dict, delta: float = 1e-3) -> torch.Tensor:
    """Theorem R for coupled rows in head space (cert_s1h5.tail_bound_rand_coupled), with Lemma N_lat's per-coordinate
    inputs; the latent block's steps make the value error (a block-0-exact store has none)."""
    W, A0, colA2 = S["W"], S["A0"], S["colA2"]
    c = S["c"]
    hc, hVh = S["head"](c), torch.einsum("grod,gcd->grco", W, D["Vh"])
    rhat = ((hVh * hVh).sum(-1).unsqueeze(2) - 2.0 * torch.einsum("grno,grco->grnc", hc, hVh)
            + (hc * hc).sum(-1, keepdim=True)).clamp_min(0.0).sqrt()
    D0 = D["steps"][0]
    if D0 is None:
        nu2 = torch.zeros(1, dtype=DT)
        numax = torch.zeros(1, dtype=DT)
    else:
        nu2 = (torch.einsum("grj,gcj->grc", colA2, D0 * D0) / 12.0).unsqueeze(2)
        nm1 = S["Wn"].unsqueeze(-1) * (D0 * D0).sum(-1).sqrt().unsqueeze(1) / 2.0
        nm2 = torch.einsum("grj,gcj->grc", colA2.sqrt(), D0) / 2.0
        numax = torch.minimum(nm1, nm2).unsqueeze(2)
    oc = (S["head"](D["oT"]) - hc).norm(dim=-1)
    return K.tail_bound_rand_coupled(S["s"], D["shd"], D["sig"], S["km"], rhat, numax, nu2, oc, S["lf"], delta=delta,
                                     b_as=D["bas"], mom=D["mom"], ldebias=D["ldb"])


CERT_CFGS = ((("store", 2), "same"), (("store", 4), "same"), (("uniform", 6), "same"), (("uniform", 8), "same"))


def analyze(view: str, draws: int = 2, delta: float = 1e-3, out: str | None = None) -> dict:
    G.reserve_memory(G.mem_estimate_gb(view))
    t0 = time.time()
    Sf = setup(view)
    S8 = Sf[8]
    g, rat, preds = grid(Sf)
    t1 = time.time()
    dp = {}
    for lat, rope in CERT_CFGS:
        cov, ratio, certf, e2 = [], [], [], []
        for sd in range(draws):
            D = draw(S8, sd, lat, rope)
            b = cert(S8, D, delta)
            e = D["err"]
            cov.append(float((e <= b * (1 + 1e-9) + 1e-15).double().mean()))
            ratio.append((b / e.clamp_min(1e-30)).flatten())
            certf.append(float((b <= S8["e8"]).double().mean()))
            e2.append(e * e)
        rt = torch.cat(ratio)
        pr = preds[f"{lat[0]}:{lat[1]}/{rope}/8"]
        m2 = torch.stack(e2).mean(0)
        wu = torch.softmax(torch.cat([S8["s"], S8["lf"].unsqueeze(-1)], -1), -1)[..., :-1] * (~S8["km"])
        sig_w = float(((wu * D["sig"]).sum(-1) / wu.sum(-1).clamp_min(1e-300)).median())
        dp[f"{lat[0]}:{lat[1]}/{rope}"] = dict(coverage=min(cov), ratio_med=float(rt.median()), ratio_p90=G._q(rt, 0.9),
                                               certified_within_fp8=sum(certf) / len(certf),
                                               mse_meas_over_pred=float(m2.sum() / pr.sum()), sigma_w_med=sig_w)
    res = dict(view=os.path.basename(view), model=S8["model"], layer=S8["layer"], tag=S8["tag"], L=S8["L"], C=S8["C"],
               H=S8["r"], dc=S8["dc"], dR=S8["D"] - S8["dc"], checks=S8["checks"], draws=draws, **g, cert=dp,
               t_grid=round(t1 - t0, 1), t_cert=round(time.time() - t1, 1))
    path = out or G.result_path(view)
    G._atomic_save(rat, path[:-5] + ".pt")
    G._atomic_json(res, path)
    c = res["cfg"]
    print(f"{res['view']}: unread 2nd-moment share {g['flat'][8]['unread_second_moment_share']:.4f}; tail / FP8 (median) "
          f"store 4+2 {c['store:2/same/8']['med']:.2f}, 4+4 {c['store:4/same/8']['med']:.2f}; coupling share "
          f"{c['store:2/same/8']['cross_share']:.1e}; coupled Theorem R at 4+2: cover "
          f"{dp['store:2/same']['coverage']:.3f}, {dp['store:2/same']['ratio_med']:.1f}x; recon "
          f"{S8['checks'].get('recon', float('nan')):.1e} ({res['t_grid']} + {res['t_cert']} s)", flush=True)
    return res


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
        capture(opts["--model"], int(opts["--L"]) if "--L" in opts else None,
                tuple(opts["--prompts"].split(",")) if "--prompts" in opts else G.PROMPTS,
                controls="--no-controls" not in opts, k_load=int(opts["--k-load"]) if "--k-load" in opts else None)
    elif cmd == "analyze":
        for v in args:
            if "--force" not in opts and os.path.exists(G.result_path(v)):
                print(f"{v}: on disk")
                continue
            analyze(v, draws=int(opts.get("--draws", 2)))
    elif cmd == "backfill":
        for v in args:
            G.backfill(v, setup_fn=setup, grid_fn=grid)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()

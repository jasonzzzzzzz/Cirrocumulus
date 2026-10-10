#!/usr/bin/env python3
"""R5 adapters (adapters_s1h5.py) on CPU. --fast: tiny random-weight models of every family built from
transformers' own classes (GQA, GPT-OSS sinks + banded layers, Gemma 3 local/global + QK-norm,
Granite-4.0-H Mamba-2 + attention, DeepSeek-V2/V3 MLA, Kimi-Linear's NoPE MLA): the adapter must
reconstruct each global layer's output from its view, MLA's absorption must be exact, the bounds must
hold. Default adds the six downloaded models (truncated to their first global layers, real weights, a
1,600-token pg19 passage), one subprocess each, and writes findings/R5_adapters.{md,json}.

    OMP_NUM_THREADS=8 .venv/bin/python -u h0_measurement/bugs/14h_methodology_improve/test_adapters_s1h5.py --fast
"""
import json, os, subprocess, sys, tempfile, time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import adapters_s1h5 as A  # noqa: E402

OK, BAD = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
fails = 0


def check(name, cond, detail=""):
    global fails
    print(f"  {OK if cond else BAD}  {name} {detail}")
    if not cond:
        fails += 1


def _tiny(kind):
    """A tiny random-weight model of one family (float32, eager) and its global layers."""
    import transformers as T
    common = dict(vocab_size=128, hidden_size=64, intermediate_size=96, num_attention_heads=4, max_position_embeddings=512)
    if kind == "llama":
        cfg = T.LlamaConfig(num_hidden_layers=2, num_key_value_heads=2, head_dim=16, **common)
        cls = T.LlamaForCausalLM
    elif kind == "gptoss":
        cfg = T.GptOssConfig(num_hidden_layers=2, num_key_value_heads=2, head_dim=16, num_local_experts=4,
                             num_experts_per_tok=2, sliding_window=8, layer_types=["sliding_attention", "full_attention"],
                             **common)
        cls = T.GptOssForCausalLM
    elif kind == "gemma3":
        cfg = T.Gemma3TextConfig(num_hidden_layers=6, num_key_value_heads=2, head_dim=16, sliding_window=8,
                                 query_pre_attn_scalar=16, **common)
        cls = T.Gemma3ForCausalLM
    elif kind == "granite":
        cfg = T.GraniteMoeHybridConfig(num_hidden_layers=2, num_key_value_heads=2, num_local_experts=4,
                                       num_experts_per_tok=2, layer_types=["mamba", "attention"], mamba_n_heads=4,
                                       mamba_d_head=32, mamba_n_groups=1, mamba_d_state=16, mamba_expand=2,
                                       mamba_chunk_size=16, position_embedding_type="nope", **common)
        cls = T.GraniteMoeHybridForCausalLM
    elif kind in ("dsv2", "dsv3"):
        mk = T.DeepseekV2Config if kind == "dsv2" else T.DeepseekV3Config
        cfg = mk(num_hidden_layers=2, num_key_value_heads=4, kv_lora_rank=32, q_lora_rank=None if kind == "dsv2" else 24,
                 qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16, n_routed_experts=4, num_experts_per_tok=2,
                 moe_intermediate_size=32, first_k_dense_replace=1, n_group=1, topk_group=1, **common)
        cls = T.DeepseekV2ForCausalLM if kind == "dsv2" else T.DeepseekV3ForCausalLM
    else:
        raise ValueError(kind)
    torch.manual_seed(0)
    cfg._attn_implementation = "eager"
    m = cls(cfg).float().eval()
    if kind == "gptoss":                                   # sinks large enough to matter
        for mod in m.modules():
            if hasattr(mod, "sinks"):
                mod.sinks.data.normal_(0, 2.0)
    return m, A.global_layers(m.config)


def _tiny_kimi(L):
    """Kimi-Linear's MLA path (DeepseekV3Attention, q_proj, identity position embeddings) at tiny size."""
    import transformers as T
    from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Attention
    cfg = T.DeepseekV3Config(hidden_size=64, num_attention_heads=4, num_key_value_heads=4, q_lora_rank=None,
                             kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16, num_hidden_layers=4,
                             rope_interleave=False, vocab_size=128)
    cfg._attn_implementation = "eager"
    torch.manual_seed(1)
    attn = DeepseekV3Attention(cfg, 3).float().eval()
    x = torch.randn(1, L, 64)
    mask = torch.full((L, L), float("-inf")).triu(1).view(1, 1, L, L)
    pe = (torch.ones(1, L, 8), torch.zeros(1, L, 8))
    with A.capture(torch.nn.ModuleList([attn]), [3]) as rec:
        with torch.no_grad():
            attn(x, pe, mask)
    return rec


def test_fast():
    print("\n[S1h R5] adapters: tiny random-weight models of every family (CPU)")
    L = 96
    ids = torch.randint(0, 128, (1, L), generator=torch.Generator().manual_seed(0))
    for kind in ("llama", "gptoss", "gemma3", "granite", "dsv2", "dsv3", "kimi"):
        try:
            if kind == "kimi":
                rec, layers, n_all = _tiny_kimi(L), [3], 4
            else:
                m, layers = _tiny(kind)
                with A.capture(m, layers) as rec:
                    with torch.no_grad():
                        m(ids, use_cache=False)
                n_all = m.config.num_hidden_layers
            res = A.report_layers(rec, L, n_meas=8, n_vote=8)
        except Exception as e:  # noqa: BLE001
            check(f"{kind}: runs", False, f"({type(e).__name__}: {str(e)[:300]})")
            continue
        recon = max(x["recon"] for x in res)
        absorb = max(max(x.get("absorb_k", 0.0), x.get("absorb_v", 0.0)) for x in res)
        valid = all(x["lemma2_ok"] == 1.0 and x["l4_tailx_ok"] == 1.0 and x.get("l4_tail_ok", 1.0) == 1.0 for x in res)
        fam = {x["family"] for x in res}
        extra = ""
        if kind == "gptoss":                                # negative control: drop the sinks
            r0 = {li: dict(rec[li], sink=None) for li in rec}
            no_sink = max(x["recon"] for x in A.report_layers(r0, L, n_meas=8, n_vote=8))
            extra = f"; without the sinks the reconstruction misses by {no_sink:.2e}"
            valid = valid and no_sink > 1e-3 and all(x["sink"] for x in res)
        if kind == "gemma3":
            valid = valid and layers == [5]
        if kind == "granite":
            valid = valid and layers == [1]
        check(f"{kind}: global layers {layers} of {n_all} ({'/'.join(sorted(fam))}, G x r = "
              f"{res[0]['G']} x {res[0]['r']}, row width {res[0]['dk']}); the view reconstructs the layer's output; "
              f"MLA absorption exact; Lemma 2 and Lemma 4 hold", recon < 1e-5 and absorb < 1e-5 and valid,
              f"(reconstruction {recon:.1e}, absorption {absorb:.1e}{extra})")


def test_real():
    print("\n[S1h R5] adapters: the six downloaded models (+ Llama-3.2-1B), truncated, real weights, CPU, 1,600 tokens")
    tmp = tempfile.mkdtemp(prefix="s1h5_adapt_")
    allres = []
    for key in A.MODELS:
        out = os.path.join(tmp, f"{key}.json")
        env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "8"), PYTHONUNBUFFERED="1")
        t0 = time.time()
        r = subprocess.run([sys.executable, "-u", os.path.join(HERE, "adapters_s1h5.py"), "--model", key, "--out", out],
                           capture_output=True, text=True, env=env, timeout=3500)
        if r.returncode != 0 or not os.path.exists(out):
            check(f"{key}: runs", False, f"(rc {r.returncode}; {(r.stderr or r.stdout)[-1500:]})")
            continue
        res = json.load(open(out))
        allres.append(res)
        lay = res["layers"]
        recon = max(x["recon"] for x in lay)
        absorb = max(max(x.get("absorb_k", 0.0), x.get("absorb_v", 0.0)) for x in lay)
        valid = all(x["lemma2_ok"] == 1.0 and x["l4_tailx_ok"] == 1.0 and x.get("l4_tail_ok", 1.0) == 1.0 for x in lay)
        check(f"{key} ({res['label']}): {len(lay)} global layer(s) {res['global_layers']}; reconstruction, absorption, "
              f"Lemma 2 and Lemma 4 hold", recon < 1e-4 and absorb < 1e-4 and valid,
              f"(reconstruction {recon:.1e}, absorption {absorb:.1e}; {time.time() - t0:.0f} s)")
    if allres:
        _write(allres)


def _write(allres):
    fdir = os.path.join(HERE, "findings")
    json.dump(allres, open(os.path.join(fdir, "R5_adapters.json"), "w"), indent=1)
    cols = [("G x r", lambda x: f"{x['G']} x {x['r']}"), ("row (k/v)", lambda x: f"{x['dk']}/{x['dv']}"),
            ("recon", lambda x: f"{x['recon']:.0e}"), ("B_min .01", lambda x: f"{x['bmin_frac_med']:.3f}"),
            ("vote eps", lambda x: f"{x['eps_vote_med']:.3f}"), ("cert/oracle", lambda x: f"{x['cert_over_union_med']:.1f}"),
            ("hp/oracle", lambda x: f"{x['cert_hp_over_union_med']:.1f}"), ("b max", lambda x: f"{x['b_max_med']:.1f}"),
            ("rel tailx", lambda x: f"{x['rel_tailx_med']:.4f}"), ("rel FP8", lambda x: f"{x['rel_fp8_med']:.4f}"),
            ("rel 4/4", lambda x: f"{x['rel_d4_med']:.4f}"), ("tailx<=FP8", lambda x: f"{x['err_tailx_fp8']:.2f}"),
            ("L5 cover", lambda x: f"{x['l5_tailx_cover']:.3f}"), ("L5/err", lambda x: f"{x['l5_over_err_tailx_med']:.0f}"),
            ("L5<=FP8", lambda x: f"{x['l5_tailx_fp8']:.2f}"), ("+C/32", lambda x: f"{x['l5_tailx_p32_fp8']:.2f}"),
            ("+C/8", lambda x: f"{x['l5_tailx_p8_fp8']:.2f}"),
            ("L5b cover", lambda x: f"{min(x['l5b_tailx_cover'], x['l5b_tailx_p32_cover'], x['l5b_tailx_p8_cover']):.3f}"),
            ("L5b<=FP8 +C/8", lambda x: f"{x['l5b_tailx_p8_fp8']:.2f}")]
    lines = ["# R5 adapters — the certificate on each architecture (CPU smokes, real weights)", "",
             "Written by `test_adapters_s1h5.py` (code: `adapters_s1h5.py`). Each model truncated to its first",
             "global softmax layers, float32 on CPU, one 1,600-token pg19 passage; context = the first 1,552 rows,",
             "the next 32 positions vote (tier-1 attention summed over each row group: 1/8 of the rows), the last 16",
             "are measured. Tier 1 = the store's rotated Lloyd-Max at 4 bits (MLA: the latent, once). Medians over",
             "heads and steps; shares are of head-steps. A smoke, not a result: one passage, early layers only.", "",
             "| model | layer | " + " | ".join(c for c, _ in cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for res in allres:
        for x in res["layers"]:
            lines.append(f"| {res['model']} | {x['layer']} | " + " | ".join(f(x) for _, f in cols) + " |")
    lines += ["", "Columns: G x r = row groups x query heads sharing one row set; row = key/value width per group "
              "(MLA: the 576-wide latent, values its first 512); recon = max relative difference between the "
              "adapter's output and the model's own; B_min .01 = rows per head for missed mass <= 0.01 (share of "
              "context); vote eps = the vote's missed mass; cert/oracle, hp/oracle = certified rows (worst-case, "
              "z = 5 bound) over the shared-set oracle's at 0.01; b max = the largest worst-case score bound "
              "(nats); rel = output error / |o| (tailx = the vote's rows exact, the rest at 4 bits; 4/4 = dense "
              "4-bit); tailx<=FP8 = share of head-steps with tailx's error within FP8's; L5 cover = share with the "
              "error within Lemma 5's bound (should be >= 0.999); L5/err = median bound over error; L5<=FP8 = share "
              "certified at FP8's level at the vote's rows, +C/32 and +C/8 with that many more rows read exactly; "
              "L5b = Lemma 5 with a score-bias allowance of 0.5 sigma (its worst coverage over the three reads, and "
              "its share certified at FP8's level with C/8 more rows)."]
    open(os.path.join(fdir, "R5_adapters.md"), "w").write("\n".join(lines) + "\n")
    print(f"  wrote findings/R5_adapters.md ({sum(len(r['layers']) for r in allres)} layers)")


if __name__ == "__main__":
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    test_fast()
    if "--fast" not in sys.argv:
        test_real()
    print(f"\n{'ALL R5 ADAPTER TESTS PASSED' if not fails else f'{fails} R5 ADAPTER TEST(S) FAILED'}")
    sys.exit(1 if fails else 0)

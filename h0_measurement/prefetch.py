#!/usr/bin/env python3
"""
prefetch.py -- download weights on the LOGIN node.

Compute nodes on most SLURM clusters have no outbound internet, so weights must be
staged into a shared cache before any job is submitted. Run this first, always.

Stages both the model itself AND its `validate_with` proxy -- run_h0.py loads the
proxy for L1/L3 probe validation before it touches the real model, so a job whose
proxy is missing dies at validation with a GPU already allocated.

  python h0_measurement/prefetch.py              # all models in models.yaml
  python h0_measurement/prefetch.py -m qwen3-8b
"""
from __future__ import annotations
import argparse, os, sys, yaml
from huggingface_hub import snapshot_download

ALLOW = ["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*", "*.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(__import__("pathlib").Path(__file__).with_name("models.yaml")))
    ap.add_argument("-m", "--models", nargs="*", default=None)
    ap.add_argument("--cache", default=os.environ.get("HF_HOME"))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    tags = args.models or [m["tag"] for m in cfg["models"]]

    # Resolve the hub cache HERE and pass it to snapshot_download explicitly.
    # Mutating os.environ at this point cannot work: huggingface_hub froze its
    # HF_HOME/HF_HUB_CACHE constants when it was imported at module scope above,
    # so --cache used to be silently ignored and weights went wherever the
    # inherited env pointed. $HF_HOME/hub matches the layout the shell scripts pin.
    cache_dir = os.path.join(args.cache, "hub") if args.cache else None
    if cache_dir:
        print(f"HF_HOME = {args.cache}\nhub cache = {cache_dir}")

    # A model needs TWO sets of weights staged: its own, and the small proxy that
    # L1/L3 probe validation loads (`validate_with`). Missing the proxy fails the
    # job at the first validation step, after it has already been allocated a GPU.
    # Proxies are shared between models, so collect ids first and download once.
    wanted: dict[str, list[str]] = {}
    for m in cfg["models"]:
        if m["tag"] not in tags:
            continue
        wanted.setdefault(m["id"], []).append(m["tag"])
        vwith = m.get("validate_with")
        if vwith:
            wanted.setdefault(vwith, []).append(f"{m['tag']}:validate_with")

    fail = 0
    for model_id, who in wanted.items():
        print(f"\n=== {' '.join(who)}  ({model_id}) ===", flush=True)
        try:
            # No resume_download=: deprecated and ignored since hub 0.23, it only
            # printed a UserWarning into every job's stderr. Resume is automatic.
            p = snapshot_download(model_id, allow_patterns=ALLOW,
                                  max_workers=8, cache_dir=cache_dir)
            sz = sum(os.path.getsize(os.path.join(r, f))
                     for r, _, fs in os.walk(p) for f in fs) / 1e9
            print(f"  ok  {sz:.1f} GB -> {p}")
        except Exception as e:
            fail += 1
            print(f"  FAILED: {e}", file=sys.stderr)
            if "gated" in str(e).lower() or "401" in str(e):
                print("  -> accept the license on the HF model page, then "
                      "`huggingface-cli login`", file=sys.stderr)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
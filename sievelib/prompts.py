"""prompts.py -- long-context prompt families.

Families are chosen to span the attention regimes the theory distinguishes:
  niah  retrieval  -> expect sharp heads, wide sensitivity ladder
  qa    mixed
  cont  continuation -> expect diffuse heads, narrow ladder
If niah does not show a wider ladder than cont, the haystack is not inducing
retrieval behaviour -- point H0_CORPUS at real text and rerun.
"""
from __future__ import annotations
import os, random

FILLER = [
    "The harbour master kept a ledger of every vessel that passed the breakwater.",
    "Rain moved across the valley in long grey sheets, erasing the far ridge.",
    "She counted the tiles on the ceiling twice and arrived at different numbers.",
    "The committee adjourned without resolving the question of the eastern boundary.",
    "Copper wire salvaged from the old exchange lay coiled in the corner of the shed.",
    "By the third winter the orchard had gone entirely to seed and nobody minded.",
    "He read the timetable aloud, though there had been no train since September.",
    "The letter arrived unfranked, which the clerk considered a deliberate insult.",
]


def _corpus_text(corpus_dir, need_chars, seed):
    if not corpus_dir or not os.path.isdir(corpus_dir):
        return None
    files = sorted(f for f in os.listdir(corpus_dir) if f.endswith((".txt", ".md")))
    if not files:
        return None
    rng = random.Random(seed)
    rng.shuffle(files)
    buf = []
    n = 0
    for f in files:
        t = open(os.path.join(corpus_dir, f), encoding="utf-8", errors="ignore").read()
        buf.append(t); n += len(t)
        if n >= need_chars:
            break
    return "\n\n".join(buf) if n >= need_chars * 0.6 else None


def build(tok, family, ctx, seed=0, corpus_dir=None):
    rng = random.Random(seed)
    target = int(ctx * 0.92)
    txt = _corpus_text(corpus_dir or os.environ.get("H0_CORPUS"), target * 5, seed)
    synthetic = txt is None
    if synthetic:
        parts, n = [], 0
        while n < target:
            parts.append(rng.choice(FILLER)); n += 12
        txt = " ".join(parts)
    ids = tok(txt, add_special_tokens=False).input_ids[:target]
    txt = tok.decode(ids)

    if family == "niah":
        code = rng.randint(10000, 99999)
        needle = f"\n\nThe access code for vault {rng.randint(10, 99)} is {code}.\n\n"
        cut = int(len(ids) * rng.uniform(0.15, 0.85))
        txt = tok.decode(ids[:cut]) + needle + tok.decode(ids[cut:])
        txt += "\n\nQuestion: what is the access code mentioned above? Answer:"
    elif family == "qa":
        txt += "\n\nSummarize the three most important claims in the document above:"
    return txt, synthetic
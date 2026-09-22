"""
tasks_ruler.py -- RULER-style end tasks on the project's PG-19 haystack (R8).

Why not the in-house `niah` family: one 5-digit needle scored by substring match,
and it passed 6/6, 4/4, 3/3 on every cell at full precision. Single-needle
retrieval is known to survive aggressive KV compression, so it cannot separate
the arms. It stays here as `niah_single` -- the smoke test the compressed path
must pass at a generous budget -- and the measurement uses the three RULER tasks
that are known to discriminate:

  niah_multikey    n_keys needles with DIFFERENT keys; ask for one. The others
                   are distractors, so keeping the wrong heavy-hitter is visible.
  niah_multivalue  one key with n_values values in different places; ask for all.
                   Recall across several distant spots.
  vt               variable tracking: a chain VAR X1 = v, VAR X2 = VAR X1, ...;
                   ask which variables hold v. Multi-hop, every link must survive.

Templates, answer prefixes and string-match scoring follow RULER (Hsieh et al.,
2024) so the numbers are comparable in kind; the haystack is the project's own
validated PG-19 window (`prompts._build_haystack`), so the text is exactly what
every other measurement in this project read. Raw text, no chat template --
the convention of the whole pipeline, including the calibration pass P2 will
read the router from.

Seeds live in their own namespace ("ruler", task, prompt_idx), so no R8 needle
coincides with a run_h0 `niah` needle at the same prompt index.
"""
from __future__ import annotations
import random
import re

from . import prompts

TASKS = ("niah_single", "niah_multikey", "niah_multivalue", "vt")

# answer length budget per task, in tokens -- enough for the answer, short
# enough that a model which rambles does not wander into another needle
MAX_NEW = {"niah_single": 24, "niah_multikey": 24, "niah_multivalue": 64, "vt": 64}

_ADJ = ["amber", "arctic", "brass", "cobalt", "crimson", "dusky", "ember", "fallow",
        "gilded", "hollow", "ivory", "jagged", "kindred", "lunar", "mossy", "nimble",
        "ochre", "pewter", "quiet", "russet", "silver", "tawny", "umber", "velvet",
        "woven", "zinc", "copper", "misty", "rustic", "sable"]
_NOUN = ["anchor", "beacon", "cinder", "dagger", "easel", "falcon", "garnet", "harbor",
         "island", "jasper", "kettle", "lantern", "marble", "needle", "orchard", "parcel",
         "quarry", "raven", "saddle", "thistle", "urchin", "violin", "walnut", "yarrow",
         "zephyr", "compass", "lattice", "meadow", "pylon", "sonnet"]

_NIAH_PREFIX = ("Some special magic numbers are hidden within the following text. "
                "Make sure to memorize it. I will quiz you about the numbers "
                "afterwards.\n\n")
_VT_PREFIX = ("Memorize and track the chain(s) of variable assignment hidden in the "
              "following text.\n\n")


def _key(rng, used):
    while True:
        k = f"{rng.choice(_ADJ)}-{rng.choice(_NOUN)}"
        if k not in used:
            used.add(k)
            return k


def _num(rng, digits, used):
    while True:
        v = str(rng.randint(10 ** (digits - 1), 10 ** digits - 1))
        if v not in used:
            used.add(v)
            return v


def _var(rng, used):
    while True:
        v = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5))
        if v not in used:
            used.add(v)
            return v


def _insert(tok, ids, needles, depths):
    """Splice `needles` into the haystack token ids at fractional `depths`, in
    order. Cuts on token boundaries and joins decoded segments, as prompts.build
    does for the in-house needle; each needle is padded with blank lines so it
    reads as its own sentence whatever the cut lands on."""
    n = len(ids)
    cuts = sorted(max(1, min(n - 1, int(round(d * n)))) for d in depths)
    parts, prev = [], 0
    for c, nd in zip(cuts, needles):
        parts.append(tok.decode(ids[prev:c]))
        parts.append(f"\n\n{nd}\n\n")
        prev = c
    parts.append(tok.decode(ids[prev:]))
    return "".join(parts), cuts


def build(tok, task, ctx, *, prompt_idx, corpus_dir=None, require_real=False,
          n_keys=4, n_values=4, n_hops=4):
    """One prompt. Returns (text, meta); `meta['expected']` is what scoring
    looks for, `meta['distractors']` what it must not confuse it with."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; one of {TASKS}")
    corpus_dir = prompts.resolve_corpus_dir(corpus_dir)
    ids, meta = prompts._build_haystack(tok, ctx, corpus_dir, prompt_idx, require_real)
    rng = random.Random(prompts._seed("ruler", task, prompt_idx))
    used_k, used_v = set(), set()
    distractors: list[str] = []

    if task in ("niah_single", "niah_multikey"):
        n = 1 if task == "niah_single" else n_keys
        keys = [_key(rng, used_k) for _ in range(n)]
        vals = [_num(rng, 7, used_v) for _ in range(n)]
        q = rng.randrange(n)
        needles = [f"One of the special magic numbers for {k} is: {v}."
                   for k, v in zip(keys, vals)]
        order = list(range(n))
        rng.shuffle(order)                        # the queried needle is not always first
        needles = [needles[i] for i in order]
        expected = [vals[q]]
        distractors = [v for i, v in enumerate(vals) if i != q]
        prefix = _NIAH_PREFIX
        question = (f"\n\nWhat is the special magic number for {keys[q]} mentioned "
                    f"in the provided text? The special magic number for {keys[q]} "
                    f"mentioned in the provided text is")
    elif task == "niah_multivalue":
        key = _key(rng, used_k)
        vals = [_num(rng, 7, used_v) for _ in range(n_values)]
        needles = [f"One of the special magic numbers for {key} is: {v}." for v in vals]
        expected = list(vals)
        prefix = _NIAH_PREFIX
        question = (f"\n\nWhat are all the special magic numbers for {key} mentioned "
                    f"in the provided text? The special magic numbers for {key} "
                    f"mentioned in the provided text are")
    else:  # vt
        used_n: set[str] = set()
        value = _num(rng, 5, used_v)
        names = [_var(rng, used_n) for _ in range(n_hops + 1)]
        needles = [f"VAR {names[0]} = {value}."] + \
                  [f"VAR {names[i]} = VAR {names[i - 1]}." for i in range(1, n_hops + 1)]
        expected = list(names)
        prefix = _VT_PREFIX
        question = (f"\n\nQuestion: Find all variables that are assigned the value "
                    f"{value} in the text above. Answer: According to the chain(s) "
                    f"of variable assignment in the text above, {len(names)} "
                    f"variables are assigned the value {value}, they are:")

    # the chain must read forward for vt, so depths are sorted and needles keep
    # their order; for niah the needles were already shuffled above
    depths = sorted(rng.uniform(0.05, 0.95) for _ in needles)
    body, cuts = _insert(tok, ids, needles, depths)
    text = prefix + body + question
    meta.update(family=f"ruler_{task}", task=task, prompt_idx=prompt_idx,
                expected=expected, distractors=distractors,
                needle_depths=[round(d, 4) for d in depths], n_needles=len(needles),
                # text == context + question: the question-agnostic driver
                # compresses the context before the question exists (plan.md 12)
                question=question)
    return text, meta


_NUM = re.compile(r"\d{5,}")


def score(task, pred, meta) -> dict:
    """RULER's string_match_all -- the share of expected strings present in the
    prediction -- as the primary score, plus two diagnostics chosen for R8:

      first_ok     single/multikey: the FIRST long number in the answer is the
                   right one. string_match_all credits an answer that states the
                   right value AND then rambles through the others; first_ok
                   does not.
      distractor   multikey: another key's value appears -- what keeping the
                   wrong needle looks like, which is exactly the failure an
                   eviction corner can cause.
    """
    p = pred.strip()
    exp = meta["expected"]
    hits = sum(1 for e in exp if e in p)
    out = {"score": hits / max(len(exp), 1), "hits": hits, "n_expected": len(exp),
           "distractor": any(d in p for d in meta.get("distractors", [])),
           "first_ok": float("nan")}
    if task in ("niah_single", "niah_multikey"):
        m = _NUM.search(p)
        out["first_ok"] = float(bool(m) and m.group(0) == exp[0])
    return out

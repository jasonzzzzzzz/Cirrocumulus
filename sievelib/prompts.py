"""prompts.py -- long-context prompt families.

Families are chosen to span the attention regimes the theory distinguishes:
  niah  retrieval  -> expect sharp heads, wide sensitivity ladder
  qa    mixed
  cont  continuation -> expect diffuse heads, narrow ladder
If niah does not show a wider ladder than cont, the haystack is not inducing
retrieval behaviour -- point H0_CORPUS at real text and rerun. report.py turns
that sentence into a gate (`family_gate`) instead of a comment.

Two things here are deliberate and easy to "fix" into uselessness:

1. THE HAYSTACK IS SHARED ACROSS FAMILIES AT A GIVEN PROMPT INDEX. niah, qa and
   cont built with the same `prompt_idx` see byte-identical haystack text; only
   the needle and the trailing question differ. That makes the niah-vs-cont
   ladder comparison PAIRED per (layer, head) -- the gate measures the effect of
   the retrieval task, not the difference between two random slices of a novel.
   Give the families independent text and the gate keeps working but gets much
   noisier. (This used to happen by accident: the old seed was
   `1000*p + len(family)`, and len("niah") == len("cont") == 4.)

2. A CORPUS THAT IS PRESENT BUT TOO SMALL IS AN ERROR, NOT A FALLBACK. The old
   code returned synthetic filler whenever it gathered less than 60% of its
   character estimate, so a corpus one book short of a 128k window silently
   produced exactly the filler run this module exists to avoid. `build` now
   confirms the window really yields `ctx` tokens after tokenisation, grows the
   window if the text is denser than estimated, and raises under `require_real`.
"""
from __future__ import annotations
import hashlib, json, os, random

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

# Chars budgeted per token when sizing a window. English prose through a Llama or
# Qwen BPE runs ~4; 5 is slack so the first read almost always suffices. Being
# wrong is not silent -- _build_haystack grows the window and re-tokenises.
CHARS_PER_TOKEN = 5.0
CTX_FILL = 0.92                  # leave room for the needle and the question
MAX_GROW = 4                     # window-growth attempts before giving up


def _seed(*parts) -> int:
    """Deterministic across processes. random.Random() falls back to hash() for
    tuples, and hash() of a str is salted per process unless PYTHONHASHSEED is
    pinned -- which would make every prompt in a campaign irreproducible."""
    s = "|".join(str(p) for p in parts)
    return int(hashlib.sha256(s.encode()).hexdigest()[:16], 16)


def resolve_corpus_dir(explicit=None) -> str | None:
    d = explicit or os.environ.get("H0_CORPUS")
    return d if d and os.path.isdir(d) else None


def corpus_index(corpus_dir) -> list[tuple[str, int]]:
    """(name, size) for every text file. Sizes come from stat, not from reading."""
    if not corpus_dir or not os.path.isdir(corpus_dir):
        return []
    out = []
    for f in sorted(os.listdir(corpus_dir)):
        if f.endswith((".txt", ".md")):
            p = os.path.join(corpus_dir, f)
            if os.path.isfile(p):
                out.append((f, os.path.getsize(p)))
    return out


def corpus_sha(corpus_dir) -> str:
    """Short id of the exact text staged, recorded in every parquet row."""
    if not corpus_dir:
        return ""
    man = os.path.join(corpus_dir, "MANIFEST.json")
    if os.path.isfile(man):
        try:
            return json.load(open(man)).get("corpus_sha", "")
        except Exception:
            pass
    idx = corpus_index(corpus_dir)
    h = hashlib.sha256()
    for name, size in idx:
        h.update(f"{name}:{size}|".encode())
    return h.hexdigest()[:16]


def _read_window(path, offset, n_bytes) -> str:
    """Read one slice instead of the whole book: a 128k prompt needs ~600 KB out
    of files that run to 3 MB. Trim to line boundaries so a window never starts
    or ends mid-word."""
    with open(path, "rb") as f:
        f.seek(max(0, offset))
        raw = f.read(n_bytes)
    txt = raw.decode("utf-8", errors="ignore")
    nl = txt.find("\n")
    if 0 <= nl < 4096:
        txt = txt[nl + 1:]
    nl = txt.rfind("\n")
    if nl > len(txt) - 4096:
        txt = txt[: nl + 1]
    return txt


def corpus_window(corpus_dir, need_chars, key) -> dict | None:
    """One haystack window, keyed by `key` (the prompt index).

    Prefers a single book large enough to cover the whole window: splicing two
    unrelated novels puts an artificial topic boundary in the middle of the
    context, which is itself an attention artifact. Books are dealt round-robin
    from a corpus-level shuffle, so consecutive prompt indices get DIFFERENT
    books rather than colliding at random.
    """
    idx = corpus_index(corpus_dir)
    if not idx:
        return None
    total = sum(n for _, n in idx)
    need = int(need_chars)

    big = [(f, n) for f, n in idx if n >= need]
    order = random.Random(_seed("corpus-order", corpus_sha(corpus_dir)))
    if big:
        big = list(big)
        order.shuffle(big)
        name, size = big[key % len(big)]
        rng = random.Random(_seed("offset", name, key))
        off = rng.randrange(0, max(1, size - need))
        txt = _read_window(os.path.join(corpus_dir, name), off, int(need * 1.05))
        return {"text": txt, "doc": name, "offset": off, "spliced": False}

    if total < need:
        return None
    files = list(idx)
    order.shuffle(files)
    start = (key * max(1, len(files) // 4)) % len(files)
    files = files[start:] + files[:start]
    buf, got, used = [], 0, []
    for name, size in files:
        buf.append(_read_window(os.path.join(corpus_dir, name), 0, need - got + 4096))
        used.append(name); got += size
        if got >= need:
            break
    return {"text": "\n\n".join(buf), "doc": ",".join(used), "offset": 0,
            "spliced": True}


def _filler(target_tokens, rng) -> str:
    parts, n = [], 0
    while n < target_tokens:
        parts.append(rng.choice(FILLER)); n += 12
    return " ".join(parts)


def _build_haystack(tok, ctx, corpus_dir, hay_key, require_real):
    """Return (ids, meta). Verifies by TOKENISING at the real ctx: a character
    estimate is not evidence that the window holds `target` tokens."""
    target = int(ctx * CTX_FILL)
    meta = {"synthetic": True, "source": "filler", "doc": None, "offset": None,
            "spliced": False, "corpus_sha": corpus_sha(corpus_dir),
            "n_haystack_tokens": 0}

    if corpus_dir:
        want = target * CHARS_PER_TOKEN
        for attempt in range(MAX_GROW):
            w = corpus_window(corpus_dir, want, hay_key)
            if w is None:
                break
            ids = tok(w["text"], add_special_tokens=False).input_ids
            if len(ids) >= target:
                meta.update(synthetic=False, source="corpus", doc=w["doc"],
                            offset=w["offset"], spliced=w["spliced"],
                            n_haystack_tokens=target)
                return ids[:target], meta
            want *= 1.6                       # denser text than estimated; grow
        if require_real:
            raise RuntimeError(
                f"H0_CORPUS={corpus_dir!r} cannot supply a {target:,}-token "
                f"haystack (needed ~{int(target * CHARS_PER_TOKEN):,} chars, "
                f"corpus holds {sum(n for _, n in corpus_index(corpus_dir)):,}). "
                f"Stage more text:  python h0_measurement/prefetch_corpus.py "
                f"--check-ctx {ctx}")

    if require_real:
        raise RuntimeError(
            "no usable H0_CORPUS, so the haystack would be synthetic filler. "
            "Stage one:  python h0_measurement/prefetch_corpus.py")
    rng = random.Random(_seed("filler", hay_key))
    ids = tok(_filler(target, rng), add_special_tokens=False).input_ids[:target]
    meta["n_haystack_tokens"] = len(ids)
    return ids, meta


def build(tok, family, ctx, seed=0, corpus_dir=None, *,
          prompt_idx=None, require_real=False):
    """Build one prompt. Returns (text, meta).

    meta is a dict, not the old `synthetic` bool: the provenance of the text
    (which book, which offset, spliced or not, which corpus) has to reach the
    parquet or a result cannot be traced back to what the model actually read.

    `prompt_idx` keys the haystack. Families sharing a prompt_idx share the
    haystack -- see the module docstring; the gate in report.py depends on it.
    """
    corpus_dir = resolve_corpus_dir(corpus_dir)
    hay_key = prompt_idx if prompt_idx is not None else seed
    ids, meta = _build_haystack(tok, ctx, corpus_dir, hay_key, require_real)
    meta.update(family=family, prompt_idx=hay_key, needle_tok=-1,
                needle_code="", needle_char_start=-1, needle_char_end=-1)

    txt = tok.decode(ids)
    if family == "niah":
        rng = random.Random(_seed("needle", hay_key, family))
        code = rng.randint(10000, 99999)
        needle = f"\n\nThe access code for vault {rng.randint(10, 99)} is {code}.\n\n"
        cut = int(len(ids) * rng.uniform(0.15, 0.85))
        head = tok.decode(ids[:cut])
        txt = head + needle + tok.decode(ids[cut:])
        txt += "\n\nQuestion: what is the access code mentioned above? Answer:"
        # CHARACTER offsets, because they are exact: this function does the
        # concatenation, so it knows precisely where the needle sits. `needle_tok`
        # is an index into the HAYSTACK's tokens, which does not survive
        # decode -> concat -> re-tokenise (BOS shifts everything, and BPE merges
        # across the two seams). run_h0.py converts these to token positions with
        # the tokenizer's offset mapping; validity.needle_mass needs that span.
        meta.update(needle_tok=cut, needle_frac=cut / max(1, len(ids)),
                    needle_code=str(code), needle_text=needle,
                    needle_char_start=len(head),
                    needle_char_end=len(head) + len(needle))
    elif family == "qa":
        txt += "\n\nSummarize the three most important claims in the document above:"
    return txt, meta


def preflight(tok, ctx, corpus_dir=None, require_real=False, n_prompts=1):
    """Build the haystack once, on CPU, before any GPU is touched.

    The whole point of the gate is that it fires in seconds rather than after a
    4xH100 allocation is already held, and that it proves the corpus yields real
    text AT THIS ctx rather than merely that H0_CORPUS is set.
    """
    corpus_dir = resolve_corpus_dir(corpus_dir)
    _, meta = _build_haystack(tok, ctx, corpus_dir, 0, require_real)
    idx = corpus_index(corpus_dir)
    need = int(ctx * CTX_FILL * CHARS_PER_TOKEN)
    meta["n_books"] = len(idx)
    meta["n_books_full_window"] = sum(1 for _, n in idx if n >= need)
    meta["corpus_chars"] = sum(n for _, n in idx)
    meta["n_prompts"] = n_prompts
    return meta

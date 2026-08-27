#!/usr/bin/env python3
"""
prefetch_corpus.py -- stage the real haystack on the LOGIN node.

Same reason as prefetch.py: compute nodes have no outbound internet, so the text
the measurement attends over has to be on shared disk before anything is
submitted. Unset H0_CORPUS means synthetic filler, and a full campaign on filler
measures how a model degenerates on 8 sentences tiled ~970x, not how it behaves
on a long document (h0_measurement/bugs/1_from_synthetic_to_real_corpus/).

PG-19 is Project Gutenberg books published before 1919, so we fetch them from
Gutenberg directly rather than through `datasets`: no new dependency (this venv
has none, and the pyarrow episode in the README is a warning about adding them),
a pinned ID list instead of a dataset revision, and a MANIFEST that can be
re-verified offline on a compute node.

  python h0_measurement/prefetch_corpus.py                     # 40 books -> .h0_corpus/pg19
  python h0_measurement/prefetch_corpus.py --check-ctx 131072  # is it big enough?
  python h0_measurement/prefetch_corpus.py --verify            # offline sha256 re-check
"""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, random, re, sys, time, urllib.error, urllib.request

# Pinned candidate list: pre-1919 Project Gutenberg, ordered LONG FIRST so that a
# small --books still yields books that clear the 128k window on their own. IDs
# that 404 are skipped, not fatal -- the run continues until --books succeed.
CANDIDATES = [
    (2600, "war-and-peace"),            (1184, "count-of-monte-cristo"),
    (135,  "les-miserables"),           (145,  "middlemarch"),
    (766,  "david-copperfield"),        (1023, "bleak-house"),
    (996,  "don-quixote"),              (2413, "madame-bovary"),
    (1399, "anna-karenina"),            (28054,"brothers-karamazov"),
    (2554, "crime-and-punishment"),     (963,  "little-dorrit"),
    (967,  "nicholas-nickleby"),        (968,  "martin-chuzzlewit"),
    (883,  "our-mutual-friend"),        (821,  "dombey-and-son"),
    (580,  "pickwick-papers"),          (599,  "vanity-fair"),
    (6593, "tom-jones"),                (507,  "adam-bede"),
    (6688, "mill-on-the-floss"),        (7469, "daniel-deronda"),
    (2701, "moby-dick"),                (4300, "ulysses"),
    (1257, "three-musketeers"),         (3300, "wealth-of-nations"),
    (1260, "jane-eyre"),                (768,  "wuthering-heights"),
    (1400, "great-expectations"),       (730,  "oliver-twist"),
    (98,   "tale-of-two-cities"),       (1342, "pride-and-prejudice"),
    (158,  "emma"),                     (161,  "sense-and-sensibility"),
    (141,  "mansfield-park"),           (105,  "persuasion"),
    (121,  "northanger-abbey"),         (345,  "dracula"),
    (84,   "frankenstein"),             (174,  "picture-of-dorian-gray"),
    (1661, "adventures-sherlock-holmes"),(2852,"hound-of-the-baskervilles"),
    (108,  "return-of-sherlock-holmes"),(834,  "memoirs-sherlock-holmes"),
    (244,  "study-in-scarlet"),         (76,   "huckleberry-finn"),
    (74,   "tom-sawyer"),               (120,  "treasure-island"),
    (164,  "twenty-thousand-leagues"),  (103,  "around-the-world-80-days"),
    (36,   "war-of-the-worlds"),        (5230, "invisible-man"),
    (35,   "time-machine"),             (829,  "gullivers-travels"),
    (514,  "little-women"),             (113,  "secret-garden"),
    (289,  "wind-in-the-willows"),      (236,  "jungle-book"),
    (55,   "wonderful-wizard-of-oz"),   (16,   "peter-pan"),
    (2814, "dubliners"),                (219,  "heart-of-darkness"),
    (1695, "man-who-was-thursday"),     (863,  "mysterious-affair-at-styles"),
    (1228, "origin-of-species"),        (205,  "walden"),
    (1497, "the-republic"),             (2680, "meditations"),
    (2591, "grimms-fairy-tales"),       (1727, "the-odyssey"),
]

# Gutenberg serves the same book under several paths and mirrors; try in order.
URL_PATTERNS = [
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://gutenberg.pglaf.org/cache/epub/{id}/pg{id}.txt",
]
UA = "Mozilla/5.0 (compatible; h0-corpus-fetch/1; research use)"

# Every Gutenberg file carries an identical ~1 KB licence header and a ~19 KB
# footer. Left in, they would be the one passage repeated across every book in
# the corpus -- a miniature version of the filler artifact this whole change
# exists to remove. PG-19 the dataset strips them too.
START_RE = re.compile(r"^\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*$",
                      re.M | re.I)
END_RE = re.compile(r"^\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*$",
                    re.M | re.I)
SMALLPRINT_RE = re.compile(r"^\*END\*THE SMALL PRINT.*$", re.M | re.I)


def strip_boilerplate(txt: str) -> tuple[str, bool]:
    """Return (body, clean). clean=False means the markers were not found."""
    clean = True
    m = START_RE.search(txt) or SMALLPRINT_RE.search(txt)
    if m:
        txt = txt[m.end():]
    else:
        clean = False
    m = END_RE.search(txt)
    if m:
        txt = txt[: m.start()]
    else:
        clean = False
    # Gutenberg text is CRLF; normalise so char offsets and byte offsets agree
    # for ASCII prose, which is what prompts._window seeks with.
    return txt.replace("\r\n", "\n").strip() + "\n", clean


def fetch(book_id: int, timeout: int, retries: int = 2) -> str | None:
    for pat in URL_PATTERNS:
        url = pat.format(id=book_id)
        for attempt in range(retries):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    return r.read().decode("utf-8", errors="ignore")
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    break                      # wrong path for this id, try next
                time.sleep(1.5 * (attempt + 1))
            except Exception:
                time.sleep(1.5 * (attempt + 1))
    return None


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def corpus_sha(entries) -> str:
    """One id for the whole corpus: hash of the sorted per-file digests. Goes in
    every parquet row so a result can be traced to the exact text it saw."""
    h = hashlib.sha256()
    for e in sorted(entries, key=lambda x: x["file"]):
        h.update(e["sha256"].encode())
    return h.hexdigest()[:16]


def default_out() -> str:
    if os.environ.get("H0_CORPUS"):
        return os.environ["H0_CORPUS"]
    root = pathlib.Path(__file__).resolve().parent.parent
    return str(root / ".h0_corpus" / "pg19")


def report_capacity(entries, ctx, n_prompts, cpt) -> bool:
    """Does this corpus support n_prompts DISTINCT windows at this ctx?

    prompts.build asks for ctx*0.92 tokens and budgets `cpt` chars per token.
    A book that clears that on its own gives a window with no mid-context splice
    between two unrelated works; below that the builder has to concatenate.
    """
    need = int(ctx * 0.92 * cpt)
    big = [e for e in entries if e["n_chars"] >= need]
    total = sum(e["n_chars"] for e in entries)
    print(f"\n  ctx {ctx:,}  needs ~{need:,} chars/prompt  ({n_prompts} prompts)")
    print(f"  books >= that size : {len(big)} / {len(entries)}")
    print(f"  total corpus       : {total:,} chars")
    if len(big) >= n_prompts:
        print(f"  OK -- {n_prompts} prompts get distinct single-book windows")
        return True
    if total >= need * n_prompts:
        print(f"  WARN -- only {len(big)} books clear one window, so some prompts "
              f"will be spliced from several books (recorded as corpus_spliced).")
        return True
    print(f"  TOO SMALL -- need >= {need * n_prompts:,} chars for {n_prompts} "
          f"disjoint prompts. Re-run with a larger --books.")
    return False


def load_manifest(out) -> dict | None:
    p = os.path.join(out, "MANIFEST.json")
    return json.load(open(p)) if os.path.isfile(p) else None


def do_verify(out) -> int:
    """Offline: no network. Safe to call from a compute node as a precondition."""
    man = load_manifest(out)
    if not man:
        print(f"FAIL: no MANIFEST.json in {out}", file=sys.stderr)
        return 1
    bad = []
    for e in man["files"]:
        p = os.path.join(out, e["file"])
        if not os.path.isfile(p):
            bad.append(f"{e['file']}: missing")
        elif sha256_file(p) != e["sha256"]:
            bad.append(f"{e['file']}: sha256 mismatch")
    if bad:
        print(f"FAIL: {len(bad)} problem(s) in {out}", file=sys.stderr)
        for b in bad[:10]:
            print(f"  {b}", file=sys.stderr)
        return 1
    print(f"ok  {len(man['files'])} books, {man['total_chars']:,} chars, "
          f"corpus_sha={man['corpus_sha']}  ({out})")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="corpus dir (default $H0_CORPUS "
                    "or $PROJECT_ROOT/.h0_corpus/pg19)")
    ap.add_argument("--books", type=int, default=40, help="how many to stage")
    ap.add_argument("--ids", nargs="*", type=int, default=None,
                    help="override the pinned Gutenberg id list")
    ap.add_argument("--min-chars", type=int, default=120_000,
                    help="discard books shorter than this after stripping")
    ap.add_argument("--check-ctx", type=int, default=131072)
    ap.add_argument("--n-prompts", type=int, default=6)
    ap.add_argument("--chars-per-token", type=float, default=5.0)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--sleep", type=float, default=0.4, help="between downloads")
    ap.add_argument("--verify", action="store_true", help="offline sha256 check only")
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()

    out = args.out or default_out()
    if args.verify:
        sys.exit(do_verify(out))

    os.makedirs(out, exist_ok=True)
    cands = ([(i, f"book{i}") for i in args.ids] if args.ids else CANDIDATES)

    entries, failed, short = [], [], []
    for bid, slug in cands:
        if len(entries) >= args.books:
            break
        name = f"pg19_{bid:05d}_{slug}.txt"
        path = os.path.join(out, name)
        if os.path.isfile(path) and not args.force:
            n = os.path.getsize(path)
            if n >= args.min_chars:
                entries.append({"file": name, "gutenberg_id": bid, "slug": slug,
                                "n_chars": n, "sha256": sha256_file(path)})
                print(f"  have {name}  {n:,}")
                continue
        raw = fetch(bid, args.timeout)
        time.sleep(args.sleep)
        if raw is None:
            failed.append(bid); print(f"  MISS {bid} {slug}", file=sys.stderr); continue
        body, clean = strip_boilerplate(raw)
        if len(body) < args.min_chars:
            short.append((bid, len(body))); print(f"  skip {slug} ({len(body):,} < "
                                                  f"{args.min_chars:,})"); continue
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)
        entries.append({"file": name, "gutenberg_id": bid, "slug": slug,
                        "n_chars": len(body), "sha256": sha256_file(path),
                        "boilerplate_stripped": clean})
        print(f"  ok   {name}  {len(body):,}{'' if clean else '  (markers not found)'}")

    if not entries:
        print("no books staged", file=sys.stderr)
        sys.exit(1)

    man = {"source": "project-gutenberg (PG-19 = pre-1919 Gutenberg)",
           "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "n_books": len(entries),
           "total_chars": sum(e["n_chars"] for e in entries),
           "corpus_sha": corpus_sha(entries),
           "files": sorted(entries, key=lambda e: e["file"])}
    with open(os.path.join(out, "MANIFEST.json"), "w") as f:
        json.dump(man, f, indent=1)

    print(f"\nstaged {len(entries)} books, {man['total_chars']:,} chars -> {out}")
    if failed:
        print(f"  {len(failed)} id(s) unreachable: {failed}")
    ok = report_capacity(entries, args.check_ctx, args.n_prompts, args.chars_per_token)
    print(f"\n  export H0_CORPUS={out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

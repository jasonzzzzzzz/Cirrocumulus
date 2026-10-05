"""tasks_s1h.py -- R14 Stage 1h, R3b: aggregation and latent-association tasks for the
project's harness. Standalone (no model, no runner chain): run_s1h3b.py installs them into
sievelib.tasks_ruler for one run (tasks_ruler is not edited). Data in data/r3b/, pinned by
sha256 (DATA_SHA256).

TASKS
  cwe            RULER's common-words extraction (NVIDIA/RULER@c3f5e3b4; template and answer
                 prefix verbatim). A numbered list in which 10 words appear freq_cw times and
                 every other word 3 times, after RULER's one-shot example (40 words: 10 x 10,
                 30 x 3, with its answer). Answer: the 10 common words. It needs the whole
                 list, so a question-time read of 1/8 of the rows is stressed where retrieval
                 is not.
  fwe            RULER's frequent-words extraction: coded 6-letter words with Zipf counts
                 floor(N k^-alpha / zeta(alpha)), vocabulary (ctx - 50) // 50, rank 1 replaced
                 by '...'. Answer: ranks 2-4.
  nolima         NoLiMa (Modarressi et al., ICML 2025; HF amodaresi/NoLiMa@378115b1; Adobe
                 Research License, non-commercial research only; data/r3b/nolima_LICENSE)
                 one-hop questions: the needle and the question share no content word ("Yuki
                 lives next to the Semper Opera House" / "Which character has been to
                 Dresden?"). Answer: the character's name. NoLiMa's task template verbatim.
  nolima_direct  the same prompt (haystack, needle, character, depth) with NoLiMa's direct
                 question, which repeats the needle's words ("Which character lives next to
                 the Semper Opera House?"): the control for the question-time vote.

DIFFICULTY (task_config_r3b): cwe's freq_cw (RULER: 30) and fwe's alpha (RULER: 2.0).
DEVIATIONS FROM THE SOURCES (each for this harness)
  - Raw text, no chat template, as every R14 task.
  - The context ends where the question begins (the harness compresses the context before
    the question exists): cwe/fwe's '\\nQuestion: ...' plus RULER's answer prefix is the
    question; nolima's is the template's text after the haystack plus '\\nAnswer:'.
  - The context fills prompts.CTX_FILL (0.92) of ctx, as the RULER tasks' haystack does:
    binary search on the number of distinct words (RULER fits ctx - tokens_to_generate).
  - cwe: the words are the 8,050 lowercase single words of wonderwords 2.2.0's three lists
    (RULER also keeps acronyms and phrases); the example's 40 words are disjoint from the
    list's; no english_words.json fallback (build raises if the pool runs out).
  - Scoring is RULER's string_match_all on whole words, case-insensitive ('art' is not found
    in 'party'); nolima is NoLiMa's 'contains' on the whole name.
  - nolima: the haystack is the project's PG-19 window (not NoLiMa's shuffled books); the
    character is drawn among the needle set's names that the window does not contain; the
    depth is uniform in [0.05, 0.95]; prompt p uses one-hop pair NOLIMA_ORDER[p mod 32].
"""
from __future__ import annotations
import hashlib
import json
import os
import random
import re
import string
from functools import lru_cache

import numpy as np

from sievelib import prompts, tasks_ruler as TR

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "r3b")
GEN_VERSION = "s1h_r3b_v1"
TASKS = ("cwe", "fwe", "nolima", "nolima_direct")
NOLIMA_TASKS = ("nolima", "nolima_direct")
DATA_SHA256 = {
    "nounlist.txt": "0778f51872f2aa89e0745c0cc37471a13e3d2161a79fcfdc03b6390b10604a55",
    "adjectivelist.txt": "66814d46b7e292c83e839d12fe2fa083c8f66779b14b294e1be4e1342c5d4131",
    "verblist.txt": "9fbf5e4e69b8869aebd88e6d545312ff9f878a747327ede5f6dbbba5c27b08ad",
    "nolima_needle_set.json": "b197f5633668be11aa5855144716b3009d6246c5fb05a508c37db0d08303aa99",
    "nolima_needle_set_ONLYDirect.json": "0f44e4d277f72bd05780a6cf449b7691636837e1e8da254d3ab73fa3b38c071e",
}
SOURCES = dict(ruler="NVIDIA/RULER@c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a", wonderwords="mrmaxguns/wonderwordsmodule@v2.2.0",
               nolima="huggingface.co/datasets/amodaresi/NoLiMa@378115b1f136b6ba78f90f78682bc55f70ec3ddd")

# RULER (scripts/data/synthetic/constants.py), verbatim
CWE_TEMPLATE = ("Below is a numbered list of words. In these words, some appear more often than others. "
                "Memorize the ones that appear most often.\n{context}\nQuestion: What are the 10 most common "
                "words in the above list?")
CWE_ANSWER_PREFIX = " Answer: The top 10 words that appear most often in the list are:"
FWE_TEMPLATE = ("Read the following coded text and track the frequency of each coded word. Find the three most "
                "frequently appeared coded words. {context}\nQuestion: Do not provide any explanation. Please "
                "ignore the dots '....'. What are the three most frequently appeared words in the above coded text?")
FWE_ANSWER_PREFIX = " Answer: According to the coded text above, the three most frequently appeared words are:"
NOLIMA_ANSWER_PREFIX = "\nAnswer:"
MAX_NEW = {"cwe": 120, "fwe": 50, "nolima": 32, "nolima_direct": 32}   # RULER's tokens_to_generate; nolima short
DEFAULT_CFG = {"freq_cw": 30, "alpha": 2.0}
CWE_NUM_CW, CWE_FREQ_UCW = 10, 3
CWE_EXAMPLE = (40, 10, 3)                  # RULER's one-shot at >= 4K: 40 words, common x 10, the rest x 3
FWE_WORDLEN, FWE_CHARS_PER_VOCAB, FWE_GEN = 6, 50, 50
DEPTH_LO, DEPTH_HI = 0.05, 0.95


# ------------------------------------------------------------------ config
def task_config_r3b(freq_cw=DEFAULT_CFG["freq_cw"], alpha=DEFAULT_CFG["alpha"]) -> dict:
    f, a = int(freq_cw), float(alpha)
    if f != float(freq_cw) or f <= CWE_FREQ_UCW:
        raise ValueError(f"freq_cw must be an integer above {CWE_FREQ_UCW}, got {freq_cw!r}")
    if not 1.0 < a <= 4.0:
        raise ValueError(f"alpha must lie in (1, 4], got {alpha!r}")
    return {"freq_cw": f, "alpha": a}


def parse_cfg(s: str) -> dict:
    """'freq_cw=100,alpha=1.5' -> validated dict (missing knobs take RULER's values)."""
    raw = {}
    for item in [x for x in str(s).split(",") if x]:
        k, v = item.split("=")
        if k not in DEFAULT_CFG:
            raise ValueError(f"unknown R3b difficulty knob {k!r}")
        raw[k] = float(v) if k == "alpha" else int(v)
    return task_config_r3b(**raw)


def cfg_str(cfg: dict) -> str:
    c = task_config_r3b(**cfg)
    return f"freq_cw={c['freq_cw']},alpha={format(c['alpha'], 'g')}"


# -------------------------------------------------------------------- data
def _data(name: str) -> str:
    return os.path.join(DATA, name)


@lru_cache(maxsize=None)
def _verified(name: str) -> str:
    p = _data(name)
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if h != DATA_SHA256[name]:
        raise RuntimeError(f"{p}: sha256 {h[:12]} is not the pinned {DATA_SHA256[name][:12]}")
    return p


@lru_cache(maxsize=None)
def cwe_pool() -> tuple:
    words = set()
    for f in ("nounlist.txt", "adjectivelist.txt", "verblist.txt"):
        for line in open(_verified(f)):
            w = line.strip()
            if re.fullmatch(r"[a-z]+", w):
                words.add(w)
    return tuple(sorted(words))


@lru_cache(maxsize=None)
def nolima_sets() -> tuple:
    main = {e["id"]: e for e in json.load(open(_verified("nolima_needle_set.json")))}
    direct = {e["id"]: e for e in json.load(open(_verified("nolima_needle_set_ONLYDirect.json")))}
    return main, direct


@lru_cache(maxsize=None)
def nolima_pairs() -> tuple:
    """The one-hop (needle id, test) pairs, sorted (32); every one has a direct twin. The
    direct set leaves out a test whose direct question takes no argument (0402, 0405:
    'Which character is vegan?'); its twin is that question (_direct_args)."""
    main, direct = nolima_sets()
    out = tuple((i, t) for i in sorted(main) for t in sorted(main[i]["tests"]) if "onehop" in main[i]["questions"])
    for i, t in out:
        if i not in direct or "direct" not in direct[i]["questions"]:
            raise RuntimeError(f"NoLiMa pair {i}/{t} has no direct question")
        _direct_args(direct[i], t, main[i]["tests"][t]["input_args"])
    return out


def _direct_args(ed: dict, test: str, args):
    if test in ed["tests"]:
        return ed["tests"][test]["input_args"]
    if re.search(r"\{\d+\}", ed["questions"]["direct"] + ed["needle"]):
        raise RuntimeError(f"NoLiMa {ed['id']}/{test}: no direct test and the direct question takes arguments")
    return args


@lru_cache(maxsize=None)
def nolima_order() -> tuple:
    """A fixed permutation of the pairs: consecutive prompts get distinct pairs."""
    idx = list(range(len(nolima_pairs())))
    random.Random(prompts._seed(GEN_VERSION, "nolima-order")).shuffle(idx)
    return tuple(idx)


def _fill(s: str, char: str, args) -> str:
    s = s.replace("{CHAR}", char)
    for i, a in enumerate(args):
        s = s.replace("{%d}" % (i + 1), a)
    if re.search(r"\{(\d+|CHAR)\}", s):
        raise ValueError(f"unfilled placeholder in {s!r}")
    return s


# ---------------------------------------------------------------- helpers
def _ntok(tok, text: str) -> int:
    return len(tok(text, add_special_tokens=False).input_ids)


def _fit(ntok_of, lo: int, hi: int, target: int, what: str) -> int:
    """The largest n in [lo, hi] with ntok_of(n) <= target (ntok_of increasing in n)."""
    if ntok_of(lo) > target:
        raise ValueError(f"{what}: even {lo} words exceed {target:,} tokens")
    if ntok_of(hi) <= target:
        raise ValueError(f"{what}: {hi} words (the most allowed) fit in {target:,} tokens; the pool runs out")
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ntok_of(mid) <= target:
            lo = mid
        else:
            hi = mid
    return lo


def _numbered(words) -> str:
    return " ".join(f"{i + 1}. {w}" for i, w in enumerate(words))


def zeta(s: float, n: int = 50) -> float:
    """Riemann zeta for s > 1 (Euler-Maclaurin after n terms; |error| < 1e-12 here).
    Replaces scipy.special.zeta (no scipy in the shared .venv)."""
    if s <= 1:
        raise ValueError("zeta needs s > 1")
    tot = sum(k ** -s for k in range(1, n))
    return (tot + n ** (1 - s) / (s - 1) + 0.5 * n ** -s + s * n ** (-s - 1) / 12
            - s * (s + 1) * (s + 2) * n ** (-s - 3) / 720)


def _base_meta(corpus_dir, source):
    return {"synthetic": False, "source": source, "doc": "", "offset": 0, "spliced": False,
            "corpus_sha": prompts.corpus_sha(corpus_dir) if corpus_dir else None}


# --------------------------------------------------------------------- cwe
def build_cwe(tok, ctx, *, prompt_idx, freq_cw, corpus_dir=None):
    perm = list(cwe_pool())
    random.Random(prompts._seed(GEN_VERSION, "cwe", prompt_idx)).shuffle(perm)
    n_ex, f_ex, u_ex = CWE_EXAMPLE
    ex_words, words = perm[-n_ex:], perm[:-n_ex]
    ex_list = ex_words[:CWE_NUM_CW] * f_ex + ex_words[CWE_NUM_CW:] * u_ex
    random.Random(prompts._seed(GEN_VERSION, "cwe-example", prompt_idx)).shuffle(ex_list)
    example = (CWE_TEMPLATE.format(context=_numbered(ex_list)) + CWE_ANSWER_PREFIX + " "
               + _numbered(ex_words[:CWE_NUM_CW]))
    head = example + "\n" + CWE_TEMPLATE.split("{context}")[0]
    question = "\n" + CWE_TEMPLATE.split("{context}")[1].lstrip("\n") + CWE_ANSWER_PREFIX

    def context(n):
        lst = words[:CWE_NUM_CW] * freq_cw + words[CWE_NUM_CW:n] * CWE_FREQ_UCW
        random.Random(prompts._seed(GEN_VERSION, "cwe-order", prompt_idx, n)).shuffle(lst)
        return head + _numbered(lst)

    target = int(ctx * prompts.CTX_FILL)
    n = _fit(lambda k: _ntok(tok, context(k)), CWE_NUM_CW + 1, len(words), target, f"cwe ctx {ctx} freq_cw {freq_cw}")
    meta = _base_meta(corpus_dir, "ruler_cwe_wonderwords")
    meta.update(family="r3b_cwe", task="cwe", prompt_idx=prompt_idx, expected=list(words[:CWE_NUM_CW]),
                distractors=list(words[CWE_NUM_CW:n]), n_words=n, freq_cw=int(freq_cw), freq_ucw=CWE_FREQ_UCW,
                n_entries=CWE_NUM_CW * int(freq_cw) + CWE_FREQ_UCW * (n - CWE_NUM_CW), target_needle_depth=None,
                target_needle_rank=None, question=question, query_term="")
    return context(n) + question, meta


# --------------------------------------------------------------------- fwe
def fwe_vocab(ctx, prompt_idx) -> list:
    rng = random.Random(prompts._seed(GEN_VERSION, "fwe", prompt_idx))
    size = max((ctx - FWE_GEN) // FWE_CHARS_PER_VOCAB, 8)
    vocab = set()
    while len(vocab) < size:
        vocab.add("".join(rng.choices(string.ascii_lowercase, k=FWE_WORDLEN)))
    vocab = sorted(vocab)
    rng.shuffle(vocab)
    vocab[0] = "..."                      # RULER: the top rank is noise
    return vocab


def fwe_counts(n, size, alpha) -> list:
    z = zeta(alpha)
    return [int(n * k ** -alpha / z) for k in range(1, size + 1)]


def build_fwe(tok, ctx, *, prompt_idx, alpha, corpus_dir=None):
    vocab = fwe_vocab(ctx, prompt_idx)
    head = FWE_TEMPLATE.split("{context}")[0]
    question = "\n" + FWE_TEMPLATE.split("{context}")[1].lstrip("\n") + FWE_ANSWER_PREFIX

    def context(n):
        lst = [w for w, m in zip(vocab, fwe_counts(n, len(vocab), alpha)) for _ in range(m)]
        random.Random(prompts._seed(GEN_VERSION, "fwe-order", prompt_idx, n)).shuffle(lst)
        return head + " ".join(lst)

    target = int(ctx * prompts.CTX_FILL)
    # every word is >= 1 token and the counts keep >= 1/2 of N, so N <= 2 * target
    n = _fit(lambda k: _ntok(tok, context(k)), 100, 2 * target, target, f"fwe ctx {ctx} alpha {alpha}")
    c = fwe_counts(n, len(vocab), alpha)
    if not c[1] > c[2] > c[3] > c[4]:
        raise ValueError(f"fwe p{prompt_idx}: ranks 2-5 are not strictly ordered ({c[:6]})")
    meta = _base_meta(corpus_dir, "ruler_fwe_coded")
    meta.update(family="r3b_fwe", task="fwe", prompt_idx=prompt_idx, expected=list(vocab[1:4]),
                distractors=[w for w, m in zip(vocab[4:], c[4:]) if m > 0], n_words=n, alpha=float(alpha),
                vocab_size=len(vocab), top_counts=c[:6], target_needle_depth=None, target_needle_rank=None,
                question=question, query_term="")
    return context(n) + question, meta


# ------------------------------------------------------------------ nolima
def _template_parts():
    main, direct = nolima_sets()
    ts = {e["task_template"] for e in list(main.values()) + list(direct.values())}
    if len(ts) != 1:
        raise RuntimeError("NoLiMa needle sets carry more than one task template")
    head, tail = ts.pop().split("{haystack}")
    return head, tail


def nolima_item(prompt_idx: int, haystack_text: str) -> dict:
    """Prompt p's pair, character, depth, needle and both questions. The character is
    drawn among the names the haystack does not contain (whole word)."""
    main, direct = nolima_sets()
    pid, test = nolima_pairs()[nolima_order()[int(prompt_idx) % len(nolima_pairs())]]
    e, ed = main[pid], direct[pid]
    args = e["tests"][test]["input_args"]
    dargs = _direct_args(ed, test, args)
    rng = random.Random(prompts._seed(GEN_VERSION, "nolima", prompt_idx))
    depth = rng.uniform(DEPTH_LO, DEPTH_HI)
    names = [c for c in e["character_set"] if not re.search(rf"(?<![A-Za-z]){re.escape(c)}(?![A-Za-z])", haystack_text)]
    if not names:
        raise ValueError(f"nolima p{prompt_idx}: every character name occurs in the haystack")
    char = names[rng.randrange(len(names))]
    needle = _fill(e["needle"], char, args)
    if _fill(ed["needle"], char, dargs) != needle:
        raise RuntimeError(f"NoLiMa {pid}/{test}: the direct set's needle differs")
    return dict(pair=f"{pid}/{test}", reasoning_type=e["reasoning_type"], character=char, depth=depth, needle=needle,
                q_onehop=_fill(e["questions"]["onehop"], char, args), q_direct=_fill(ed["questions"]["direct"], char, dargs),
                term_onehop=_term(e["questions"]["onehop"], args), term_direct=_term(ed["questions"]["direct"], dargs),
                n_names_free=len(names))


def _term(tmpl: str, args) -> str:
    """The test argument a question names ('' if it names none)."""
    m = re.search(r"\{(\d+)\}", tmpl)
    return args[int(m.group(1)) - 1] if m else ""


def build_nolima(tok, task, ctx, *, prompt_idx, corpus_dir=None, require_real=False):
    corpus_dir = prompts.resolve_corpus_dir(corpus_dir)
    ids, meta = prompts._build_haystack(tok, ctx, corpus_dir, prompt_idx, require_real)
    it = nolima_item(prompt_idx, tok.decode(ids))
    body, cuts = TR._insert(tok, ids, [it["needle"]], [it["depth"]])
    head, tail = _template_parts()
    q = it["q_direct"] if task == "nolima_direct" else it["q_onehop"]
    question = tail.format(question=q) + NOLIMA_ANSWER_PREFIX
    meta.update(family=f"r3b_{task}", task=task, prompt_idx=prompt_idx, expected=[it["character"]], distractors=[],
                needle_depths=[round(it["depth"], 4)], n_needles=1, target_needle_rank=0,
                target_needle_depth=round(it["depth"], 4), nolima_pair=it["pair"], needle=it["needle"],
                reasoning_type=it["reasoning_type"], n_names_free=it["n_names_free"],
                query_term=it["term_direct" if task == "nolima_direct" else "term_onehop"], question=question)
    return head + body + question, meta


# ---------------------------------------------------------------- the API
def build(tok, task, ctx, *, prompt_idx, corpus_dir=None, require_real=False, cfg=None):
    """One prompt: (text, meta); text = context + meta['question']."""
    c = task_config_r3b(**(cfg or DEFAULT_CFG))
    if task == "cwe":
        return build_cwe(tok, ctx, prompt_idx=prompt_idx, freq_cw=c["freq_cw"],
                         corpus_dir=prompts.resolve_corpus_dir(corpus_dir))
    if task == "fwe":
        return build_fwe(tok, ctx, prompt_idx=prompt_idx, alpha=c["alpha"],
                         corpus_dir=prompts.resolve_corpus_dir(corpus_dir))
    if task in NOLIMA_TASKS:
        return build_nolima(tok, task, ctx, prompt_idx=prompt_idx, corpus_dir=corpus_dir, require_real=require_real)
    raise ValueError(f"unknown R3b task {task!r}; one of {TASKS}")


def generation_limit(task) -> int:
    return MAX_NEW[task]


def _word_re(w: str):
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(w)}(?![A-Za-z0-9])", re.IGNORECASE)


def score(task, pred, meta) -> dict:
    """RULER's string_match_all on whole words, case-insensitive (cwe, fwe); NoLiMa's
    'contains' on the whole name (nolima*). distractor: cwe/fwe state a non-answer word
    of the list."""
    p = str(pred).strip()
    exp = [str(x) for x in meta["expected"]]
    if task in ("cwe", "fwe"):
        words = set(re.findall(r"[a-z]+", p.lower()))
        hits = sum(1 for e in exp if e in words)
        dis = bool(words & set(meta.get("distractors") or []))
    else:
        hits = sum(1 for e in exp if re.search(rf"(?<![A-Za-z]){re.escape(e)}(?![A-Za-z])", p))
        dis = False
    return {"score": hits / max(len(exp), 1), "hits": hits, "n_expected": len(exp), "distractor": dis,
            "first_ok": float("nan")}


_ITEM = re.compile(r"[^\S\n]*(?:\d+[.)]|[-*•])[^\S\n]*[^\d\n]*")


def stops_at_line(text: str) -> bool:
    """The 'r8list' stop rule: run_r8's stop at the first newline after content, except
    that a line holding one list item ('3. word', '- word') does not stop it, so an answer
    listed one item per line runs on to the first line that is not an item. A one-line
    list ('1. a 2. b') is not one item and stops at its newline."""
    t = text.lstrip()
    if "\n" not in t:
        return False
    return not _ITEM.fullmatch(t.split("\n")[-2])


# ------------------------------------------------- masks (the runner's helpers)
def answer_tokens_wb(tok, ids, expected) -> dict:
    """s1d_lib.answer_tokens on whole words, case-insensitive: which tokens of FP's
    answer spell an expected value; the span ends where FP completes the last value it
    states. Returns dict(vmask, span_end, found, text); found holds the strings as FP
    wrote them."""
    ids = [int(x) for x in ids]
    n = len(ids)
    full = tok.decode(ids) if n else ""
    starts, ends, prev = [], [], 0
    for i in range(n):
        e = len(tok.decode(ids[:i + 1]))
        starts.append(min(prev, e))
        ends.append(e)
        prev = e
    occ, span_char, found = [], 0, []
    for x in expected:
        x = str(x)
        if not x:
            continue
        ms = list(_word_re(x).finditer(full))
        if not ms:
            continue
        found.append(ms[0].group(0))
        span_char = max(span_char, ms[0].end())
        occ += [(m.start(), m.end()) for m in ms]
    occ = [(s, e) for s, e in occ if s < span_char]
    span_end = sum(1 for s in starts if s < span_char) if found else 0
    vmask = [bool(found) and i < span_end and any(starts[i] < e and ends[i] > s for s, e in occ) for i in range(n)]
    return dict(vmask=vmask, span_end=span_end, found=found, text=full)


def answer_positions_wb(tok, text, expected, ctx_len):
    """run_r8.answer_positions on whole words, case-insensitive, vectorized: a cwe
    context holds up to 3,000 occurrences, which the original's per-token scan over
    every span cannot afford at 128K."""
    import torch
    enc = tok(text, return_offsets_mapping=True)
    mark = np.zeros(len(text) + 1, dtype=np.int64)
    for e in expected:
        for m in _word_re(str(e)).finditer(text):
            mark[m.start()] += 1
            mark[m.end()] -= 1
    inside = np.cumsum(mark)[:-1] > 0
    cs = np.concatenate([[0], np.cumsum(inside)])
    off = np.asarray(enc["offset_mapping"][:ctx_len], dtype=np.int64).reshape(-1, 2)
    m = torch.zeros(ctx_len, dtype=torch.bool)
    if len(off):
        hit = (cs[off[:, 1]] - cs[off[:, 0]]) > 0
        m[:len(hit)] = torch.from_numpy(hit)
    return m

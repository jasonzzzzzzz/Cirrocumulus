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
import weakref
from types import MappingProxyType

from . import prompts

TASKS = ("niah_single", "niah_multikey", "niah_multivalue", "vt")

# These were implicit arguments to build() in every R8 result through
# 2026-09-22. Keep one canonical record so old sidecars/routes can be treated
# as this exact legacy configuration, while harder-task runs are explicit.
DEFAULT_TASK_CONFIG = {"n_keys": 4, "n_values": 4, "n_hops": 4}

# The prompt algorithm and its RNG namespace. Adding target-needle provenance
# below is metadata-only, so this stays at v1: old and new runs construct the
# same prompt bytes for a given task/config/prompt index.
TASK_GENERATION_VERSION = "ruler_pg19_v1"
TARGET_NEEDLE_PROVENANCE_VERSION = "queried_needle_v1"

# Versioned contrastive task used only by the V3 panel workflow.  These
# constants intentionally do not enter ``build``: the legacy RULER path above
# is a byte-level experimental contract and must remain unchanged.
MULTIKEY_PANEL_TASK = "niah_multikey"
MULTIKEY_PANEL_TASK_VARIANT = "multikey_panel_v1"
MULTIKEY_PANEL_VERSION = "contrastive_multikey_panel_v1"
MULTIKEY_PANEL_RNG_VERSION = "named_streams_v1"
MULTIKEY_PANEL_RNG_NAMESPACES = {
    "haystack": "niah_multikey_panel_v1/haystack",
    "target": "niah_multikey_panel_v1/target",
    "distractor": "niah_multikey_panel_v1/distractor",
    "placement": "niah_multikey_panel_v1/placement",
}
MULTIKEY_PANEL_KEY_SUFFIX_CONTRACT = (
    "token_ids(key)==common_prefix_token_ids+[unique_suffix_token_id]"
)
MULTIKEY_PANEL_N_CLUSTERS = 4
MULTIKEY_PANEL_CLUSTER_SIZE = 12
MULTIKEY_PANEL_N_QUERIES = 4
MULTIKEY_PANEL_TARGET_RANKS = (5, 17, 30, 42)
MULTIKEY_PANEL_TARGET_DEPTHS = (0.15, 0.38, 0.62, 0.85)


def task_config(n_keys=4, n_values=4, n_hops=4):
    """Validate and return the lossless task-difficulty provenance record."""
    raw = {"n_keys": n_keys, "n_values": n_values, "n_hops": n_hops}
    out = {}
    for key, value in raw.items():
        try:
            integer = int(value)
            exact = not isinstance(value, bool) and float(value) == integer
        except (TypeError, ValueError, OverflowError):
            exact = False
        if not exact or integer < 1:
            raise ValueError(f"task difficulty counts must be positive integers, got {raw}")
        out[key] = integer
    return out


def task_tag(config, *, include_default=False):
    """Filesystem-safe tag. Legacy/default runs retain their old filenames."""
    c = task_config(**config)
    if not include_default and c == DEFAULT_TASK_CONFIG:
        return ""
    return f"k{c['n_keys']}_v{c['n_values']}_h{c['n_hops']}"

# answer length budget per task, in tokens -- enough for the answer, short
# enough that a model which rambles does not wander into another needle
MAX_NEW = {"niah_single": 24, "niah_multikey": 24, "niah_multivalue": 64, "vt": 64}
GENERATION_LIMIT_VERSION = "difficulty_v1"


def generation_limit(task, config):
    """Answer budget that preserves k4/v4/h4 and grows with harder outputs.

    The model often spells VT answers as full assignments rather than a compact
    name list. Each added hop therefore gets 16 tokens; each added requested
    value gets 8. These are output limits only and do not alter task prompts.
    """
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; one of {TASKS}")
    c = task_config(**config)
    if task == "vt":
        return MAX_NEW[task] + 16 * max(c["n_hops"] - 4, 0)
    if task == "niah_multivalue":
        return MAX_NEW[task] + 8 * max(c["n_values"] - 4, 0)
    return MAX_NEW[task]

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
    cfg = task_config(n_keys, n_values, n_hops)
    n_keys, n_values, n_hops = (cfg["n_keys"], cfg["n_values"], cfg["n_hops"])
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; one of {TASKS}")
    corpus_dir = prompts.resolve_corpus_dir(corpus_dir)
    ids, meta = prompts._build_haystack(tok, ctx, corpus_dir, prompt_idx, require_real)
    rng = random.Random(prompts._seed("ruler", task, prompt_idx))
    used_k, used_v = set(), set()
    distractors: list[str] = []
    target_needle_rank = None

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
        # Zero-based rank in the actual insertion order. This lookup consumes
        # no randomness and therefore cannot perturb prompt generation.
        target_needle_rank = order.index(q)
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
    target_needle_depth = (round(depths[target_needle_rank], 4)
                           if target_needle_rank is not None else None)
    meta.update(family=f"ruler_{task}", task=task, prompt_idx=prompt_idx,
                expected=expected, distractors=distractors,
                needle_depths=[round(d, 4) for d in depths], n_needles=len(needles),
                target_needle_rank=target_needle_rank,
                target_needle_depth=target_needle_depth,
                task_config=cfg,
                # text == context + question: the question-agnostic driver
                # compresses the context before the question exists (plan.md 12)
                question=question)
    return text, meta


_PANEL_CLUSTER_STEMS = (
    "panel cluster alpha key",
    "panel cluster bravo key",
    "panel cluster charlie key",
    "panel cluster delta key",
)
# Tokenizer validation is moderately expensive (an oversized suffix pool is
# deliberate), but the valid key pool is prompt-independent.  Cache it for the
# life of the tokenizer so a 40-context job pays this cost once, not 40 times.
_PANEL_KEY_POOL_CACHE = weakref.WeakKeyDictionary()


def _plain_token_ids(tok, text):
    """Tokenize one key without BOS/chat decoration and return plain ints."""
    encoded = tok(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        if len(ids) != 1:
            raise ValueError("panel key tokenizer unexpectedly returned a batch")
        ids = ids[0]
    return tuple(int(x) for x in ids)


def _panel_suffix_literals():
    """A stable, oversized search pool; only tokenizer-validated entries survive."""
    words = list(dict.fromkeys(_ADJ + _NOUN + [
        "alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
        "golf", "hotel", "india", "juliet", "kilo", "lima", "mike",
        "november", "oscar", "papa", "quebec", "romeo", "sierra",
        "tango", "uniform", "victor", "whiskey", "xray", "yankee",
    ]))
    # Three-digit strings are single tokens under several long-context model
    # tokenizers and provide a robust fallback when uncommon words split.
    return tuple(f" {word}" for word in words) + tuple(
        f" {number:03d}" for number in range(1000)
    )


def _contrastive_key_pool(tok, literal_prefix, minimum):
    """Find keys with exactly one varying final token.

    Grouping by all but the last token is stronger than assuming that a space or
    an English word is one token for every model tokenizer.  Every returned key
    has the same non-empty token prefix and a distinct final token id.
    """
    try:
        tokenizer_cache = _PANEL_KEY_POOL_CACHE.setdefault(tok, {})
    except TypeError:  # unusual tokenizer wrappers can be unhashable
        tokenizer_cache = {}
    cached = tokenizer_cache.get(literal_prefix)
    if cached is None:
        groups = {}
        for suffix in _panel_suffix_literals():
            key = literal_prefix + suffix
            ids = _plain_token_ids(tok, key)
            if len(ids) < 2:
                continue
            prefix, suffix_id = ids[:-1], ids[-1]
            group = groups.setdefault(prefix, [])
            if all(old[2] != suffix_id for old in group):
                group.append((key, suffix, suffix_id))
        if groups:
            prefix, entries = min(groups.items(),
                                  key=lambda item: (-len(item[1]), item[0]))
            cached = (prefix, tuple(entries))
            tokenizer_cache[literal_prefix] = cached
        else:
            cached = ((), ())
    prefix, entries = cached
    if len(entries) < minimum:
        raise ValueError(
            f"tokenizer cannot construct {minimum} contrastive keys for "
            f"{literal_prefix!r}; largest one-token-suffix group has {len(entries)}"
        )
    return prefix, entries


def _panel_cluster_identities(tok, prompt_idx, n_distractors=11):
    """Create targets before distractors so pool extension cannot move them.

    This helper accepts a larger distractor count solely to make the independence
    contract testable.  The public v1 panel fixes it to eleven.
    """
    if isinstance(prompt_idx, bool) or int(prompt_idx) != prompt_idx:
        raise ValueError(f"prompt_idx must be an integer, got {prompt_idx!r}")
    prompt_idx = int(prompt_idx)
    if isinstance(n_distractors, bool) or int(n_distractors) != n_distractors:
        raise ValueError("n_distractors must be an integer")
    n_distractors = int(n_distractors)
    if n_distractors < 1:
        raise ValueError("n_distractors must be positive")

    pools = []
    for stem in _PANEL_CLUSTER_STEMS:
        prefix_ids, candidates = _contrastive_key_pool(
            tok, stem, minimum=n_distractors + 1)
        pools.append((stem, prefix_ids, candidates))

    # All four target identities and values are completed before a distractor
    # RNG is opened.  Changing distractor count or generation cannot perturb
    # these records.
    used_values = set()
    clusters = []
    for cluster, (stem, prefix_ids, candidates) in enumerate(pools):
        rng = random.Random(prompts._seed(
            MULTIKEY_PANEL_RNG_NAMESPACES["target"], prompt_idx, cluster))
        candidate_idx = rng.randrange(len(candidates))
        key, suffix_literal, suffix_token_id = candidates[candidate_idx]
        target = {
            "cluster": cluster,
            "key": key,
            "value": _num(rng, 7, used_values),
            "suffix_literal": suffix_literal,
            "suffix_token_id": suffix_token_id,
            "common_prefix_token_ids": prefix_ids,
        }
        clusters.append({
            "cluster": cluster,
            "literal_prefix": stem,
            "common_prefix_token_ids": prefix_ids,
            "target": target,
            "candidates": candidates,
            "target_candidate_idx": candidate_idx,
        })

    for record in clusters:
        cluster = record["cluster"]
        rng = random.Random(prompts._seed(
            MULTIKEY_PANEL_RNG_NAMESPACES["distractor"], prompt_idx, cluster))
        remaining = [entry for i, entry in enumerate(record["candidates"])
                     if i != record["target_candidate_idx"]]
        rng.shuffle(remaining)
        distractors = []
        for key, suffix_literal, suffix_token_id in remaining[:n_distractors]:
            distractors.append({
                "cluster": cluster,
                "key": key,
                "value": _num(rng, 7, used_values),
                "suffix_literal": suffix_literal,
                "suffix_token_id": suffix_token_id,
                "common_prefix_token_ids": record["common_prefix_token_ids"],
            })
        if len(distractors) != n_distractors:
            raise ValueError(f"cluster {cluster} has too few contrastive distractors")
        record["distractors"] = tuple(distractors)
        del record["candidates"]
        del record["target_candidate_idx"]
    return tuple(clusters)


def _multikey_panel_depths():
    """Fixed increasing depths with target anchors at the frozen four ranks."""
    anchors = ((-1, 0.03),) + tuple(zip(
        MULTIKEY_PANEL_TARGET_RANKS, MULTIKEY_PANEL_TARGET_DEPTHS
    )) + ((MULTIKEY_PANEL_N_CLUSTERS * MULTIKEY_PANEL_CLUSTER_SIZE, 0.97),)
    depths = [None] * (MULTIKEY_PANEL_N_CLUSTERS * MULTIKEY_PANEL_CLUSTER_SIZE)
    for (left_rank, left_depth), (right_rank, right_depth) in zip(
            anchors[:-1], anchors[1:]):
        width = right_rank - left_rank
        for rank in range(left_rank + 1, right_rank):
            fraction = (rank - left_rank) / width
            depths[rank] = round(
                left_depth + fraction * (right_depth - left_depth), 4)
    for rank, depth in zip(MULTIKEY_PANEL_TARGET_RANKS,
                           MULTIKEY_PANEL_TARGET_DEPTHS):
        depths[rank] = depth
    if any(depth is None for depth in depths):
        raise AssertionError("panel depth construction left an empty rank")
    if any(a >= b for a, b in zip(depths, depths[1:])):
        raise AssertionError("panel depths must be strictly increasing")
    return tuple(depths)


def build_multikey_panel(tok, ctx, *, prompt_idx, corpus_dir=None,
                         require_real=False):
    """Build one immutable four-query contrastive multikey panel.

    Returns ``(context, provenance, queries)``.  ``context`` ends immediately
    after the shared 48-needle haystack and contains no question.  ``queries``
    is a tuple of four read-only mappings ordered by fixed depth slot.  A runner
    can therefore prefill/compress the context once and independently append
    each query from the exact same cache boundary and bit allocation.
    """
    if isinstance(prompt_idx, bool) or int(prompt_idx) != prompt_idx:
        raise ValueError(f"prompt_idx must be an integer, got {prompt_idx!r}")
    prompt_idx = int(prompt_idx)
    if isinstance(ctx, bool) or int(ctx) != ctx or int(ctx) < 1:
        raise ValueError(f"ctx must be a positive integer, got {ctx!r}")
    ctx = int(ctx)
    corpus_dir = prompts.resolve_corpus_dir(corpus_dir)

    # Addition by a namespace-derived constant preserves distinct consecutive
    # corpus keys while separating this task from every legacy prompt family.
    haystack_key = prompt_idx + prompts._seed(
        MULTIKEY_PANEL_RNG_NAMESPACES["haystack"])
    ids, source_meta = prompts._build_haystack(
        tok, ctx, corpus_dir, haystack_key, require_real)
    clusters = _panel_cluster_identities(
        tok, prompt_idx, n_distractors=MULTIKEY_PANEL_CLUSTER_SIZE - 1)
    depths = _multikey_panel_depths()

    rotation = prompt_idx % MULTIKEY_PANEL_N_CLUSTERS
    cluster_to_query_idx = tuple(
        (cluster + rotation) % MULTIKEY_PANEL_N_CLUSTERS
        for cluster in range(MULTIKEY_PANEL_N_CLUSTERS)
    )
    query_idx_to_cluster = tuple(
        (query_idx - rotation) % MULTIKEY_PANEL_N_CLUSTERS
        for query_idx in range(MULTIKEY_PANEL_N_QUERIES)
    )

    n_needles = MULTIKEY_PANEL_N_CLUSTERS * MULTIKEY_PANEL_CLUSTER_SIZE
    ranked = [None] * n_needles
    for record in clusters:
        cluster = record["cluster"]
        query_idx = cluster_to_query_idx[cluster]
        rank = MULTIKEY_PANEL_TARGET_RANKS[query_idx]
        ranked[rank] = record["target"]

    other = [item for record in clusters for item in record["distractors"]]
    placement_rng = random.Random(prompts._seed(
        MULTIKEY_PANEL_RNG_NAMESPACES["placement"], prompt_idx))
    placement_rng.shuffle(other)
    free_ranks = [rank for rank, item in enumerate(ranked) if item is None]
    if len(other) != len(free_ranks):
        raise AssertionError("panel distractors do not fill every non-target rank")
    for rank, item in zip(free_ranks, other):
        ranked[rank] = item

    needles = [
        f"One of the special magic numbers for {item['key']} is: {item['value']}."
        for item in ranked
    ]
    body, cuts = _insert(tok, ids, needles, depths)
    context = _NIAH_PREFIX + body

    needle_keys = tuple(item["key"] for item in ranked)
    needle_values = tuple(item["value"] for item in ranked)
    needle_clusters = tuple(item["cluster"] for item in ranked)
    needle_suffix_token_ids = tuple(item["suffix_token_id"] for item in ranked)
    common_prefixes = tuple(
        tuple(record["common_prefix_token_ids"]) for record in clusters)

    provenance = dict(source_meta)
    provenance.update(
        family="ruler_niah_multikey_panel",
        task=MULTIKEY_PANEL_TASK,
        task_variant=MULTIKEY_PANEL_TASK_VARIANT,
        task_generation_version=MULTIKEY_PANEL_VERSION,
        panel_version=MULTIKEY_PANEL_VERSION,
        panel_rng_version=MULTIKEY_PANEL_RNG_VERSION,
        panel_rng_namespaces=dict(MULTIKEY_PANEL_RNG_NAMESPACES),
        key_suffix_contract=MULTIKEY_PANEL_KEY_SUFFIX_CONTRACT,
        prompt_idx=prompt_idx,
        haystack_key=haystack_key,
        n_clusters=MULTIKEY_PANEL_N_CLUSTERS,
        cluster_size=MULTIKEY_PANEL_CLUSTER_SIZE,
        n_queries=MULTIKEY_PANEL_N_QUERIES,
        n_needles=n_needles,
        target_needle_ranks=MULTIKEY_PANEL_TARGET_RANKS,
        target_needle_depths=MULTIKEY_PANEL_TARGET_DEPTHS,
        needle_depths=depths,
        cluster_rotation=rotation,
        cluster_to_query_idx=cluster_to_query_idx,
        query_idx_to_cluster=query_idx_to_cluster,
        cluster_literal_prefixes=_PANEL_CLUSTER_STEMS,
        cluster_common_prefix_token_ids=common_prefixes,
        needle_keys=needle_keys,
        needle_values=needle_values,
        needle_clusters=needle_clusters,
        needle_suffix_token_ids=needle_suffix_token_ids,
        insertion_token_cuts=tuple(cuts),
    )

    queries = []
    for query_idx, cluster in enumerate(query_idx_to_cluster):
        record = clusters[cluster]
        target = record["target"]
        rank = MULTIKEY_PANEL_TARGET_RANKS[query_idx]
        depth = MULTIKEY_PANEL_TARGET_DEPTHS[query_idx]
        cluster_distractors = record["distractors"]
        all_other = [item for item in ranked if item is not target]
        question = (
            f"\n\nWhat is the special magic number for {target['key']} mentioned "
            f"in the provided text? The special magic number for {target['key']} "
            f"mentioned in the provided text is"
        )
        query = {
            "query_idx": query_idx,
            "cluster": cluster,
            "target_cluster": cluster,
            "question": question,
            "expected": (target["value"],),
            "target_key": target["key"],
            "target_value": target["value"],
            "target_suffix_token_id": target["suffix_token_id"],
            "target_common_prefix_token_ids": tuple(
                target["common_prefix_token_ids"]),
            "target_needle_rank": rank,
            "target_needle_depth": depth,
            # Existing scoring reads ``distractors``.  It intentionally gets all
            # 47 alternatives, while the two unprefixed fields below preserve
            # the 11 within-cluster identities for strict provenance checks.
            "distractors": tuple(item["value"] for item in all_other),
            "all_distractor_values": tuple(item["value"] for item in all_other),
            "distractor_keys": tuple(item["key"] for item in cluster_distractors),
            "distractor_values": tuple(item["value"] for item in cluster_distractors),
        }
        queries.append(MappingProxyType(query))

    if len(queries) != MULTIKEY_PANEL_N_QUERIES:
        raise AssertionError("panel must expose exactly four questions")
    return context, provenance, tuple(queries)


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

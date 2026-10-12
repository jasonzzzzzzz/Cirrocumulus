"""tasks_r7b.py -- R7b's HELMET items at 128K (plan_r7b.md §3): R4's HELMET tasks unchanged (tasks_s1h4: templates,
demonstrations, scoring, the split), from R7b's item manifest (data/manifest_r7b.json, make_manifest_r7b.py).

ITEMS. The manifest extends R6's (../../14h_methodology_improve/data/r6/manifest_r6.json) in the same order: R6's 41 items
per task first (checked by the builder), then fresh items 41-110 of the RAG and ICL tasks (kilt_nq, kilt_hotpotqa,
icl_trec_coarse, icl_banking77), which no earlier run used. msmarco's test file has only 41 distinct queries and R4-R6 ran
all of them, so re-ranking stays at R6's 41 items; R7b's re-ranking units reuse R6's items 11-40 (plan_r7b.md §8, a
decision for the user). The RAG / re-rank records live in data/helmet_items/ (inside r7b/), found through HM_ITEMS, a
path relative to tasks_s1h4.HM_DIR.

install() points tasks_s1h4 at R7b's manifest and item files for one run; uninstall() restores R4's. run_r7b.py calls them
for HELMET (suite r4) blocks. install refuses an unpinned manifest unless check_manifest=False (the builder only).
"""
from __future__ import annotations

import os

import tasks_s1h4 as T

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "data", "manifest_r7b.json")
MANIFEST_SHA256 = "82d22c12d492f2f96e2f21e19511bab9da9701eaaa8ed4ee58dd52573fd1e60d"   # make_manifest_r7b.py, 2026-10-12
ITEMS_DIR = os.path.join(HERE, "data", "helmet_items")
HM_ITEMS = os.path.relpath(ITEMS_DIR, T.HM_DIR)
N_ITEMS = {"kilt_nq": 111, "kilt_hotpotqa": 111, "icl_trec_coarse": 111, "icl_banking77": 111, "msmarco_rerank_psg": 41}
R6_N_ITEMS = 41
FRESH_ITEMS = range(41, 111)              # RAG and ICL only
GEN_VERSION = "s1h_r7b_v1"
HM_SHA256_R7B = {                         # make_manifest_r7b.py, 2026-10-12 (msmarco's file is R6's, byte for byte)
    f"{HM_ITEMS}/kilt_nq.jsonl": "3935cff6e2928cdb234ebf50e417713f840f70bd657184c6209e6c2c64286c3f",
    f"{HM_ITEMS}/kilt_hotpotqa.jsonl": "8bcbcc744aa04f1666784d903caceecf7b1b6adb0da78652b1a44855110cec03",
    f"{HM_ITEMS}/msmarco_rerank_psg.jsonl": "fbb52a1334421298accc1e2eb4adf361592a713ff81b2865d06254234001f777",
}

_SAVED = {}


def install(pins=None, check_manifest=True):
    """tasks_s1h4 reads R7b's manifest and item files until uninstall()."""
    if _SAVED:
        raise RuntimeError("tasks_r7b is already installed")
    if check_manifest and (MANIFEST_SHA256 is None or not HM_SHA256_R7B):
        raise RuntimeError("R7b's HELMET manifest is not built and pinned (make_manifest_r7b.py, then tasks_r7b."
                           "MANIFEST_SHA256 and HM_SHA256_R7B)")
    _SAVED.update(MANIFEST=T.MANIFEST, MANIFEST_SHA256=T.MANIFEST_SHA256, HM_ITEMS=T.HM_ITEMS,
                  HM_SHA256=dict(T.HM_SHA256), N_ITEMS_HM=T.N_ITEMS_HM, GEN_VERSION=T.GEN_VERSION)
    T.MANIFEST, T.MANIFEST_SHA256 = MANIFEST, MANIFEST_SHA256 if check_manifest else None
    T.HM_ITEMS, T.N_ITEMS_HM, T.GEN_VERSION = HM_ITEMS, max(N_ITEMS.values()), GEN_VERSION
    T.HM_SHA256.update(pins if pins is not None else HM_SHA256_R7B)
    _clear()


def uninstall():
    if not _SAVED:
        return
    T.MANIFEST, T.MANIFEST_SHA256, T.HM_ITEMS = _SAVED["MANIFEST"], _SAVED["MANIFEST_SHA256"], _SAVED["HM_ITEMS"]
    T.N_ITEMS_HM, T.GEN_VERSION = _SAVED["N_ITEMS_HM"], _SAVED["GEN_VERSION"]
    T.HM_SHA256.clear()
    T.HM_SHA256.update(_SAVED["HM_SHA256"])
    _SAVED.clear()
    _clear()


def installed() -> bool:
    return bool(_SAVED)


def _clear():
    for f in (T.manifest, T.jsonl, T._verified):
        f.cache_clear()

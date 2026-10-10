"""tasks_s1h6.py -- R14 Stage 1h, R6: HELMET at 128K from R6's item manifest (data/r6/manifest_r6.json,
make_r6_manifest.py; plan.md amendment "R6 frozen"). R6 runs R4's HELMET tasks unchanged (tasks_s1h4: templates,
demonstrations, scoring, the split, the closed-book twin); only the item list is longer. install() points tasks_s1h4 at
R6's manifest and item files for one run, uninstall() restores R4's; run_s1h5.py calls them for the R6 presets.

ITEMS. N_ITEMS = 41 per task in R4's order (the first 20 are R4's manifest, checked by make_r6_manifest.py). Items
0-10 have been run before (R4: 0-9; R5's pilot: 5 and 10); R6's units are R6_ITEMS = 11-40 of every task (30 per task).
"""
from __future__ import annotations
import os

import tasks_s1h4 as T

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "data", "r6", "manifest_r6.json")
MANIFEST_SHA256 = "48f4f9b1e11c464b41a71d9337e0c6b40fd37cc890758406c2647310ab1f3a7e"   # make_r6_manifest.py, 2026-10-10
HM_ITEMS = "r6_items"           # under T.HM_DIR
N_ITEMS = 41
R6_ITEMS = range(11, 41)
GEN_VERSION = "s1h_r6_v1"
HM_SHA256_R6 = {                # make_r6_manifest.py, 2026-10-10
    "r6_items/kilt_nq.jsonl": "e77e49ed59dd0430ee018fbc0ced942b71eceae8ccc700970ef4e7f560389cc1",
    "r6_items/kilt_hotpotqa.jsonl": "f0dea905efce4fdfa429179a914052a4fa58d5a1068673da76fcf489a76b63e9",
    "r6_items/msmarco_rerank_psg.jsonl": "fbb52a1334421298accc1e2eb4adf361592a713ff81b2865d06254234001f777",
}

_SAVED = {}


def install(pins=None, check_manifest=True):
    """tasks_s1h4 reads R6's manifest and item files until uninstall()."""
    if _SAVED:
        raise RuntimeError("tasks_s1h6 is already installed")
    if check_manifest and MANIFEST_SHA256 is None:
        raise RuntimeError("R6's manifest is not pinned (tasks_s1h6.MANIFEST_SHA256)")
    _SAVED.update(MANIFEST=T.MANIFEST, MANIFEST_SHA256=T.MANIFEST_SHA256, HM_ITEMS=T.HM_ITEMS,
                  HM_SHA256=dict(T.HM_SHA256), N_ITEMS_HM=T.N_ITEMS_HM, GEN_VERSION=T.GEN_VERSION)
    T.MANIFEST, T.MANIFEST_SHA256 = MANIFEST, MANIFEST_SHA256 if check_manifest else None
    T.HM_ITEMS, T.N_ITEMS_HM, T.GEN_VERSION = HM_ITEMS, N_ITEMS, GEN_VERSION
    T.HM_SHA256.update(pins if pins is not None else HM_SHA256_R6)
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

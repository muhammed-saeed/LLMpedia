#!/usr/bin/env python3
"""
funnel_analysis.py — Read-only pipeline funnel analyser for LLMPedia run directories.

HOW IT WORKS
============
The LLMPedia BFS crawler stores its runtime state in a set of append-only JSONL files.
This script reads those files in six sequential passes, builds an in-memory model of
every candidate that flowed through the pipeline, and produces a full funnel report,
summary CSV/JSON files, and optional matplotlib figures.

Pass 1 — Queue (load_queue)
    Reads queue.json / queue.jsonl.  Reconstructs the entity tree: who was enqueued
    at which hop, who their parent was, and what the final queue status is (pending /
    working / done / failed).  One EntityStats object is created per unique (subject,
    hop) pair.

Pass 2 — Articles (load_articles)
    Reads articles_wikitext.jsonl and articles_meta.jsonl.  Records the number of raw
    wikilink / category candidates extracted from each generated article.  These are
    the raw candidates before any deduplication.

Pass 3 — Pre-NER dedup (load_pre_ner)
    Reads ner_decisions.jsonl (stage=="pre_ner_dedup_summary" rows), ner_lowconf.jsonl
    (canonical-dedup rejection rows), and plural_s_dedup.jsonl (plural/singular variant
    rejections).  Counts how many candidates were dropped by (a) global seen-set dedup,
    (b) within-batch dedup, and (c) plural/singular variant matching, leaving the
    "canonical_kept" count that actually enters the NER model.

Pass 4 — NER (load_ner)
    Reads ner_decisions.jsonl again, looking at per-phrase decision rows.  Each phrase
    is classified as:
      accepted            is_ne=True and passes confidence gate
      not_named_entity    is_ne=False
      below_conf_threshold  is_ne=True but confidence < threshold
      parse/call failure  model output could not be parsed (strict gate rejects)

Pass 5 — Similarity (load_similarity)
    Reads similarity_decisions.jsonl.  Classifies each candidate as:
      accepted            not a duplicate
      skipped             already in the embedding index
      rejected            duplicate, with sub-reasons:
        llm_duplicate           LLM confirmed duplicate
        within_batch            cosine-similar to another candidate in the same wave
        above_threshold_db      embedding distance alone exceeded threshold (no LLM)
        above_threshold_batch   same but against the wave batch
        operational_failure     batch/SDK error → conservative reject

Pass 6 — Finalize (finalize)
    Cross-joins similarity outcomes with queue records to count how many accepted
    candidates were actually inserted into the queue as new entities.

Output
------
  full_report.txt       Human-readable report with rate tables and per-hop breakdown.
  funnel_summary.json   All computed numbers in structured JSON.
  hop_funnel.csv        One row per hop with all key metrics.
  entity_funnel.csv     One row per entity with all per-entity metrics.
  candidate_funnel.csv  One row per candidate with stage decisions.
  queue_tree.csv        Parent-child tree edges from the queue.
  figures/              Matplotlib PNG figures (if matplotlib is installed):
    01_funnel_stages.png      Absolute counts through each pipeline stage.
    02_ner_reject_reasons.png NER rejection reason breakdown.
    03_sim_reject_reasons.png Similarity rejection reason breakdown.
    04_hop_breakdown.png      Per-hop bar chart of key metrics.
    05_survival_rates.png     Per-hop stage-survival rate line chart.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ============================================================
# Optional tqdm
# ============================================================
try:
    from tqdm import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

def _progress(iterable, *, desc: str = "", total: int = None, **kw):
    if _HAS_TQDM:
        return _tqdm(iterable, desc=desc, total=total, **kw)
    if desc:
        print(f"  {desc}...", flush=True)
    return iterable

# ============================================================
# Optional matplotlib
# ============================================================
try:
    import matplotlib
    matplotlib.use("Agg")          # headless — no display required
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

# ============================================================
# Helpers
# ============================================================

WIKILINK_RX = re.compile(r"\[\[([^\]]+)\]\]")
DASH_RX = re.compile(r"[\-_\u2010\u2011\u2012\u2013\u2014\u2212]+", re.UNICODE)

CANONICAL_DEDUP_STAGES = {
    "queue_dedup_pre_ner",
    "queue_dedup_batch_pre_ner",
    "queue_dedup",
    "queue_dedup_batch",
}

SIMILARITY_FAILURE_REASONS = {
    "missing_in_batch_output",
    "sdk_exception",
    "no_output_file",
}

SIMILARITY_SKIP_STAGE = "similarity_filter_skip_indexed"


def read_json(path: str, default: Any) -> Any:
    if not path or not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def read_jsonl(path: str):
    """Streaming generator — yields one dict at a time, never loads whole file."""
    if not path or not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def count_jsonl_lines(path: str) -> int:
    """Fast line count for progress display (no JSON parsing)."""
    if not path or not os.path.exists(path):
        return 0
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def canon(s: Any) -> str:
    if not isinstance(s, str):
        s = str(s)
    t = s.strip().lower()
    t = DASH_RX.sub(" ", t)
    t = re.sub(r"[^\w\s]", "", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t, flags=re.UNICODE).strip()
    return t


def json_counter(counter: Counter) -> str:
    return json.dumps(dict(counter), ensure_ascii=False, sort_keys=True)


def rate(n: int, d: int) -> Optional[float]:
    return round(float(n) / float(d), 4) if d else None


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def extract_wikilinks(wikitext: str) -> Tuple[int, List[str]]:
    if not isinstance(wikitext, str) or not wikitext:
        return 0, []
    raw = WIKILINK_RX.findall(wikitext)
    seen = set()
    out = []
    for item in raw:
        target = (item.split("|", 1)[0] or "").strip()
        if not target:
            continue
        low = target.lower()
        if low.startswith(("category:", "file:", "image:", "media:")):
            continue
        if target not in seen:
            seen.add(target)
            out.append(target)
    return len(raw), out


def default_out_dir(run_dir: str) -> str:
    """Output is a SUBFOLDER inside run_dir."""
    return os.path.join(os.path.abspath(run_dir), "funnel_analysis")


def write_csv(path: str, rows: List[dict]):
    ensure_dir(os.path.dirname(path))
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(["empty"])
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Similarity normalization
# ============================================================

def is_similarity_failure_reason(reason: str) -> bool:
    if not reason:
        return False
    if reason in SIMILARITY_FAILURE_REASONS:
        return True
    return reason.startswith("batch_status=") or reason.startswith("http_status=")


def normalize_similarity(stage: str, decision: str, reason: str) -> Tuple[str, str, bool]:
    st = (stage or "").strip()
    dc = (decision or "").strip().lower()
    rs = (reason or "").strip()

    if st == SIMILARITY_SKIP_STAGE:
        return "skipped", "already_in_embedding_index", False

    if dc == "accept":
        return "accepted", (rs or "accepted"), False

    if dc == "skip":
        return "skipped", (rs or "skip"), False

    if dc == "reject":
        if is_similarity_failure_reason(rs):
            return "rejected", rs, True
        if st in {
            "similarity_batch_http_error",
            "similarity_batch_missing_output",
            "similarity_batch_fallback_all",
        }:
            return "rejected", (rs or st), True
        return "rejected", (rs or "reject"), False

    return "unknown", (rs or st or "unknown"), False


# ============================================================
# Data classes
# ============================================================

@dataclass
class EntityStats:
    subject: str
    hop: int
    parent_subject: Optional[str] = None
    parent_hop: Optional[int] = None
    queue_status: Optional[str] = None

    # ── Processing flags ─────────────────────────────────────
    # These track whether the entity was actually processed by
    # the NER / similarity pipeline (vs. just having an article
    # generated).  Entities that were never processed should NOT
    # contribute to funnel "candidates generated" counts.
    ner_processed: bool = False
    sim_processed: bool = False
    has_article: bool = False

    raw_wikilink_occurrences: int = 0
    raw_wikilink_unique: int = 0
    extracted_link_targets: int = 0
    extracted_categories: int = 0
    ner_candidates: int = 0

    canonical_pre_ner_input: int = 0
    canonical_kept: int = 0
    canonical_rejected_total: int = 0
    canonical_rejected_seen: int = 0
    canonical_rejected_batch: int = 0
    canonical_rejected_plural: int = 0
    canonical_rejected_other: int = 0
    canonical_reason_counts: Counter = field(default_factory=Counter)

    pre_ner_summary_pre_logged: int = 0
    pre_ner_summary_filtered_logged: int = 0
    pre_ner_summary_post_canonical_logged: int = 0

    ner_accepted: int = 0
    ner_rejected: int = 0
    ner_reject_reasons: Counter = field(default_factory=Counter)
    ner_summary_parse_modes: Counter = field(default_factory=Counter)
    ner_summary_error_count: int = 0

    # NER sub-reason breakdowns
    ner_rejected_not_ne: int = 0
    ner_rejected_low_conf: int = 0
    ner_rejected_parse_fail: int = 0

    similarity_accepted: int = 0
    similarity_rejected: int = 0
    similarity_skipped: int = 0
    similarity_reject_reasons: Counter = field(default_factory=Counter)
    similarity_skip_reasons: Counter = field(default_factory=Counter)
    similarity_failure_reasons: Counter = field(default_factory=Counter)

    # Similarity sub-reason breakdowns
    sim_rejected_llm_duplicate: int = 0
    sim_rejected_within_batch: int = 0
    sim_rejected_above_threshold_db: int = 0
    sim_rejected_above_threshold_batch: int = 0
    sim_rejected_failure: int = 0
    sim_rejected_other: int = 0

    inserted_children: int = 0

    def to_row(self) -> dict:
        return {
            "subject": self.subject,
            "hop": self.hop,
            "parent_subject": self.parent_subject or "",
            "parent_hop": "" if self.parent_hop is None else self.parent_hop,
            "queue_status": self.queue_status or "",
            "ner_processed": self.ner_processed,
            "sim_processed": self.sim_processed,
            "has_article": self.has_article,
            "raw_wikilink_occurrences": self.raw_wikilink_occurrences,
            "raw_wikilink_unique": self.raw_wikilink_unique,
            "extracted_link_targets": self.extracted_link_targets,
            "extracted_categories": self.extracted_categories,
            "ner_candidates": self.ner_candidates,
            "canonical_pre_ner_input": self.canonical_pre_ner_input,
            "canonical_kept": self.canonical_kept,
            "canonical_rejected_total": self.canonical_rejected_total,
            "canonical_rejected_seen": self.canonical_rejected_seen,
            "canonical_rejected_batch": self.canonical_rejected_batch,
            "canonical_rejected_plural": self.canonical_rejected_plural,
            "canonical_rejected_other": self.canonical_rejected_other,
            "canonical_reason_counts": json_counter(self.canonical_reason_counts),
            "pre_ner_summary_pre_logged": self.pre_ner_summary_pre_logged,
            "pre_ner_summary_filtered_logged": self.pre_ner_summary_filtered_logged,
            "pre_ner_summary_post_canonical_logged": self.pre_ner_summary_post_canonical_logged,
            "post_canonical_candidates": self.canonical_kept,
            "ner_accepted": self.ner_accepted,
            "ner_rejected": self.ner_rejected,
            "ner_rejected_not_ne": self.ner_rejected_not_ne,
            "ner_rejected_low_conf": self.ner_rejected_low_conf,
            "ner_rejected_parse_fail": self.ner_rejected_parse_fail,
            "ner_reject_reason_counts": json_counter(self.ner_reject_reasons),
            "ner_summary_parse_modes": json_counter(self.ner_summary_parse_modes),
            "ner_summary_error_count": self.ner_summary_error_count,
            "similarity_accepted": self.similarity_accepted,
            "similarity_rejected": self.similarity_rejected,
            "similarity_skipped": self.similarity_skipped,
            "sim_rejected_llm_duplicate": self.sim_rejected_llm_duplicate,
            "sim_rejected_within_batch": self.sim_rejected_within_batch,
            "sim_rejected_above_threshold_db": self.sim_rejected_above_threshold_db,
            "sim_rejected_above_threshold_batch": self.sim_rejected_above_threshold_batch,
            "sim_rejected_failure": self.sim_rejected_failure,
            "sim_rejected_other": self.sim_rejected_other,
            "similarity_reject_reason_counts": json_counter(self.similarity_reject_reasons),
            "similarity_skip_reason_counts": json_counter(self.similarity_skip_reasons),
            "similarity_failure_reason_counts": json_counter(self.similarity_failure_reasons),
            "inserted_children": self.inserted_children,
            "canonical_survival": rate(self.canonical_kept, self.canonical_pre_ner_input),
            "ner_survival": rate(self.ner_accepted, self.canonical_kept),
            "similarity_survival": rate(self.similarity_accepted, self.ner_accepted),
            "overall_survival_to_inserted": rate(self.inserted_children, self.canonical_pre_ner_input),
        }


@dataclass
class CandidateRecord:
    parent_subject: str
    parent_hop: int
    candidate: str
    candidate_canon: str

    source_in_article: bool = False
    source_from_diag_meta: bool = False

    canonical_stage: str = "unknown"
    canonical_reason: str = ""
    canonical_source_stage: str = ""

    ner_stage: str = "not_run"
    ner_reason: str = ""
    ner_parse_mode: str = ""
    ner_source: str = ""

    similarity_stage: str = "not_run"
    similarity_reason: str = ""
    similarity_event_stage: str = ""
    similarity_is_failure: bool = False
    duplicate_of: str = ""
    max_similarity: Optional[float] = None
    max_db_similarity: Optional[float] = None
    max_batch_similarity: Optional[float] = None

    inserted_to_queue: bool = False
    inserted_subject: str = ""
    inserted_hop: Optional[int] = None

    def to_row(self) -> dict:
        return {
            "parent_subject": self.parent_subject,
            "parent_hop": self.parent_hop,
            "candidate": self.candidate,
            "candidate_canon": self.candidate_canon,
            "source_in_article": self.source_in_article,
            "source_from_diag_meta": self.source_from_diag_meta,
            "canonical_stage": self.canonical_stage,
            "canonical_reason": self.canonical_reason,
            "canonical_source_stage": self.canonical_source_stage,
            "ner_stage": self.ner_stage,
            "ner_reason": self.ner_reason,
            "ner_parse_mode": self.ner_parse_mode,
            "ner_source": self.ner_source,
            "similarity_stage": self.similarity_stage,
            "similarity_reason": self.similarity_reason,
            "similarity_event_stage": self.similarity_event_stage,
            "similarity_is_failure": self.similarity_is_failure,
            "duplicate_of": self.duplicate_of,
            "max_similarity": "" if self.max_similarity is None else self.max_similarity,
            "max_db_similarity": "" if self.max_db_similarity is None else self.max_db_similarity,
            "max_batch_similarity": "" if self.max_batch_similarity is None else self.max_batch_similarity,
            "inserted_to_queue": self.inserted_to_queue,
            "inserted_subject": self.inserted_subject,
            "inserted_hop": "" if self.inserted_hop is None else self.inserted_hop,
        }


# ============================================================
# Main analyzer
# ============================================================

class FunnelAnalyzer:
    def __init__(self, run_dir: str):
        self.run_dir = os.path.abspath(run_dir)
        self.paths = {
            "queue_json":               os.path.join(self.run_dir, "queue.json"),
            "queue_jsonl":              os.path.join(self.run_dir, "queue.jsonl"),
            "articles_meta_jsonl":      os.path.join(self.run_dir, "articles_meta.jsonl"),
            "articles_wikitext_jsonl":  os.path.join(self.run_dir, "articles_wikitext.jsonl"),
            "ner_decisions_jsonl":      os.path.join(self.run_dir, "ner_decisions.jsonl"),
            "ner_lowconf_jsonl":        os.path.join(self.run_dir, "ner_lowconf.jsonl"),
            "plural_s_dedup_jsonl":     os.path.join(self.run_dir, "plural_s_dedup.jsonl"),
            "similarity_decisions_jsonl": os.path.join(self.run_dir, "similarity_decisions.jsonl"),
            "run_meta_json":            os.path.join(self.run_dir, "run_meta.json"),
        }

        self.run_meta = read_json(self.paths["run_meta_json"], {})
        self.queue_records: Dict[Tuple[str, int], dict] = {}
        self.children_by_parent: Dict[Tuple[str, int], List[Tuple[str, int]]] = defaultdict(list)
        self.entity_stats: Dict[Tuple[str, int], EntityStats] = {}

    def _entity(self, subject: str, hop: int) -> EntityStats:
        key = (subject, hop)
        if key not in self.entity_stats:
            self.entity_stats[key] = EntityStats(subject=subject, hop=hop)
        return self.entity_stats[key]

    # ─────────────────────────────────────────────────────────────
    def load_queue(self):
        print("[1/6] Loading queue...", flush=True)
        queue_rows: List[dict] = []

        if os.path.exists(self.paths["queue_json"]):
            data = read_json(self.paths["queue_json"], [])
            if isinstance(data, list):
                for rec in data:
                    if isinstance(rec, dict) and rec.get("subject") is not None:
                        queue_rows.append(rec)

        if not queue_rows:
            for rec in read_jsonl(self.paths["queue_jsonl"]):
                if rec.get("subject") is None:
                    continue
                if "event" in rec:
                    queue_rows.append(rec)

        deduped: Dict[Tuple[str, int], dict] = {}
        for rec in queue_rows:
            subject = str(rec.get("subject"))
            hop = safe_int(rec.get("hop"), 0)
            deduped[(subject, hop)] = rec

        self.queue_records = deduped

        for (subject, hop), rec in self.queue_records.items():
            ent = self._entity(subject, hop)
            ent.parent_subject = rec.get("parent_subject")
            if rec.get("parent_hop") is not None:
                ent.parent_hop = safe_int(rec.get("parent_hop"), 0)
            ent.queue_status = rec.get("status")
            ps = rec.get("parent_subject")
            ph = rec.get("parent_hop")
            if ps is not None and ph is not None:
                self.children_by_parent[(str(ps), safe_int(ph, 0))].append((subject, hop))

        print(f"      → {len(self.queue_records):,} queue records", flush=True)

    def load_articles(self):
        print("[2/6] Loading articles...", flush=True)
        wt_count = 0
        for rec in read_jsonl(self.paths["articles_wikitext_jsonl"]):
            subject = rec.get("subject")
            if subject is None:
                continue
            hop = safe_int(rec.get("hop"), 0)
            ent = self._entity(str(subject), hop)
            ent.has_article = True
            raw_count, unique_targets = extract_wikilinks(rec.get("wikitext", ""))
            ent.raw_wikilink_occurrences = raw_count
            ent.raw_wikilink_unique = len(unique_targets)
            wt_count += 1
            if wt_count % 50000 == 0:
                print(f"      wikitext rows: {wt_count:,}", flush=True)

        meta_count = 0
        for rec in read_jsonl(self.paths["articles_meta_jsonl"]):
            subject = rec.get("subject")
            if subject is None:
                continue
            hop = safe_int(rec.get("hop"), 0)
            ent = self._entity(str(subject), hop)
            if rec.get("diag"):
                ner_candidates = rec.get("ner_candidates") or []
                if isinstance(ner_candidates, list):
                    ent.ner_candidates = max(ent.ner_candidates, len(ner_candidates))
                meta_count += 1
                continue
            links = rec.get("links") or []
            categories = rec.get("categories") or []
            if isinstance(links, list):
                ent.extracted_link_targets = len(links)
            if isinstance(categories, list):
                ent.extracted_categories = len(categories)
            meta_count += 1

        print(f"      → {wt_count:,} wikitext + {meta_count:,} meta rows", flush=True)

    def load_pre_ner(self):
        print("[3/6] Loading pre-NER dedup...", flush=True)
        nd_count = 0
        for rec in read_jsonl(self.paths["ner_decisions_jsonl"]):
            if rec.get("stage") != "pre_ner_dedup_summary":
                continue
            subject = rec.get("current_entity") or rec.get("subject")
            if subject is None:
                continue
            hop = safe_int(rec.get("hop"), 0)
            ent = self._entity(str(subject), hop)
            # ── Mark as NER-processed: this entity went through the pipeline ──
            ent.ner_processed = True
            pre_count = safe_int(rec.get("pre_dedup_count"), 0)
            post_count = safe_int(rec.get("post_dedup_count"), 0)
            filtered = safe_int(rec.get("filtered_count"), max(pre_count - post_count, 0))
            ent.pre_ner_summary_pre_logged = max(ent.pre_ner_summary_pre_logged, pre_count)
            ent.pre_ner_summary_post_canonical_logged = max(ent.pre_ner_summary_post_canonical_logged, post_count)
            ent.pre_ner_summary_filtered_logged = max(ent.pre_ner_summary_filtered_logged, filtered)
            ent.canonical_pre_ner_input = max(ent.canonical_pre_ner_input, pre_count)
            ent.canonical_kept = max(ent.canonical_kept, post_count)
            ent.canonical_rejected_total = max(ent.canonical_rejected_total, filtered)
            nd_count += 1

        lc_count = 0
        for rec in read_jsonl(self.paths["ner_lowconf_jsonl"]):
            stage = rec.get("stage", "")
            if stage not in CANONICAL_DEDUP_STAGES:
                continue
            subject = rec.get("current_entity") or rec.get("subject")
            if subject is None:
                continue
            hop = safe_int(rec.get("hop"), 0)
            ent = self._entity(str(subject), hop)
            reason = str(rec.get("rejection_reason") or stage)
            ent.canonical_reason_counts[reason] += 1
            if "batch" in reason:
                ent.canonical_rejected_batch += 1
            elif "seen" in reason:
                ent.canonical_rejected_seen += 1
            else:
                ent.canonical_rejected_other += 1
            lc_count += 1

        pl_count = 0
        for rec in read_jsonl(self.paths["plural_s_dedup_jsonl"]):
            subject = rec.get("current_entity") or rec.get("subject")
            if subject is None:
                continue
            hop = safe_int(rec.get("hop"), 0)
            ent = self._entity(str(subject), hop)
            stage = str(rec.get("stage") or "")
            reason = "plural_variant_batch" if "batch" in stage else "plural_variant"
            ent.canonical_reason_counts[reason] += 1
            ent.canonical_rejected_plural += 1
            pl_count += 1

        print(
            f"      → {nd_count:,} pre_ner_dedup_summaries, {lc_count:,} lowconf canon, "
            f"{pl_count:,} plural rows",
            flush=True,
        )

    def load_ner(self):
        print("[4/6] Loading NER decisions...", flush=True)
        row_count = 0

        for rec in read_jsonl(self.paths["ner_decisions_jsonl"]):
            subject = rec.get("current_entity") or rec.get("subject")
            if subject is None:
                continue
            hop = safe_int(rec.get("hop"), 0)
            ent = self._entity(str(subject), hop)
            stage = rec.get("stage", "")

            if stage == "ner_run_summary_global":
                ent.ner_processed = True
                for pm in rec.get("parse_modes", []) or []:
                    ent.ner_summary_parse_modes[str(pm)] += 1
                ent.ner_summary_error_count += safe_int(rec.get("chunk_failures"), 0)
                continue

            if stage == "ner_run_summary":
                ent.ner_processed = True
                parse_mode = rec.get("parse_mode")
                if parse_mode:
                    ent.ner_summary_parse_modes[str(parse_mode)] += 1
                if parse_mode == "exception":
                    ent.ner_summary_error_count += 1
                continue

            phrase = rec.get("phrase")
            if phrase is None or "decision_reason" not in rec:
                continue

            # ── Mark as NER-processed: we have per-phrase decisions ──
            ent.ner_processed = True

            passed = bool(rec.get("passed_threshold"))
            reason = str(rec.get("decision_reason") or "")
            source = str(rec.get("source") or "")

            if passed:
                ent.ner_accepted += 1
            else:
                ent.ner_rejected += 1
                ent.ner_reject_reasons[reason or "rejected"] += 1
                r_low = reason.lower()
                if "not_named_entity" in r_low or r_low == "not_ne":
                    ent.ner_rejected_not_ne += 1
                elif "below_conf" in r_low or "missing_confidence" in r_low:
                    ent.ner_rejected_low_conf += 1
                elif any(x in r_low for x in ("parse_failed", "parse_fail", "ner_call_failed", "ner_parse")):
                    ent.ner_rejected_parse_fail += 1
                elif source == "missing":
                    ent.ner_rejected_parse_fail += 1
                else:
                    ent.ner_rejected_not_ne += 1

            row_count += 1
            if row_count % 500000 == 0:
                print(f"      ner_decisions rows: {row_count:,}", flush=True)

        # ── Fallback reconciliation ──────────────────────────────
        # ONLY apply to entities that were actually NER-processed.
        # This prevents inflating unprocessed entities with article
        # metadata counts.
        for ent in self.entity_stats.values():
            if not ent.ner_processed:
                # Entity was never NER-processed — do NOT fill in
                # canonical_pre_ner_input from ner_candidates.
                continue

            if ent.canonical_pre_ner_input == 0:
                ent.canonical_pre_ner_input = ent.ner_candidates
            if ent.canonical_kept == 0:
                ner_total = ent.ner_accepted + ent.ner_rejected
                if ner_total > 0:
                    ent.canonical_kept = ner_total
            if ent.canonical_rejected_total == 0 and ent.canonical_pre_ner_input >= ent.canonical_kept:
                ent.canonical_rejected_total = ent.canonical_pre_ner_input - ent.canonical_kept
            if ent.ner_candidates == 0 and ent.canonical_pre_ner_input:
                ent.ner_candidates = ent.canonical_pre_ner_input

        print(f"      → {row_count:,} ner_decisions per-phrase rows parsed", flush=True)

    def load_similarity(self):
        print("[5/6] Loading similarity decisions...", flush=True)
        row_count = 0

        for rec in read_jsonl(self.paths["similarity_decisions_jsonl"]):
            subject = rec.get("parent_entity") or rec.get("current_entity")
            candidate = rec.get("candidate")
            if subject is None or candidate is None:
                continue

            hop = safe_int(rec.get("hop"), 0)
            ent = self._entity(str(subject), hop)
            # ── Mark as similarity-processed ──
            ent.sim_processed = True

            stage = str(rec.get("stage") or "")
            decision = str(rec.get("decision") or "")
            reason = str(rec.get("reason") or "")
            sim_stage, sim_reason, is_failure = normalize_similarity(stage, decision, reason)

            if sim_stage == "accepted":
                ent.similarity_accepted += 1
            elif sim_stage == "rejected":
                ent.similarity_rejected += 1
                ent.similarity_reject_reasons[sim_reason] += 1
                if is_failure:
                    ent.similarity_failure_reasons[sim_reason] += 1
                    ent.sim_rejected_failure += 1
                else:
                    r_low = sim_reason.lower()
                    if "llm_duplicate" in r_low:
                        ent.sim_rejected_llm_duplicate += 1
                    elif "within_batch" in r_low:
                        ent.sim_rejected_within_batch += 1
                    elif "above_threshold_db" in r_low:
                        ent.sim_rejected_above_threshold_db += 1
                    elif "above_threshold_batch" in r_low:
                        ent.sim_rejected_above_threshold_batch += 1
                    else:
                        ent.sim_rejected_other += 1
            elif sim_stage == "skipped":
                ent.similarity_skipped += 1
                ent.similarity_skip_reasons[sim_reason] += 1

            row_count += 1
            if row_count % 500000 == 0:
                print(f"      similarity_decisions rows: {row_count:,}", flush=True)

        print(f"      → {row_count:,} similarity_decisions rows parsed", flush=True)

    def finalize(self):
        print("[6/6] Finalizing queue tree...", flush=True)
        for (parent_subject, parent_hop), children in self.children_by_parent.items():
            ent = self._entity(parent_subject, parent_hop)
            ent.inserted_children = len(children)

        # ── Compute processing stats ────────────────────────────
        total = len(self.entity_stats)
        ner_proc = sum(1 for e in self.entity_stats.values() if e.queue_status == "done")
        sim_proc = sum(1 for e in self.entity_stats.values() if e.sim_processed)
        has_art  = sum(1 for e in self.entity_stats.values() if e.has_article)
        unproc   = sum(1 for e in self.entity_stats.values()
                       if e.has_article and e.queue_status != "done")
        print(f"      → {total:,} entities "
              f"(NER-processed: {ner_proc:,}, sim-processed: {sim_proc:,}, "
              f"articles: {has_art:,}, unprocessed: {unproc:,})", flush=True)

    # ─────────────────────────────────────────────────────────────
    def hop_summary_rows(self) -> List[dict]:
        grouped: Dict[int, List[EntityStats]] = defaultdict(list)
        for ent in self.entity_stats.values():
            grouped[ent.hop].append(ent)

        rows: List[dict] = []
        for hop in sorted(grouped):
            ents = grouped[hop]
            # Separate processed vs unprocessed
            processed   = [e for e in ents if e.queue_status == "done"]
            unprocessed = [e for e in ents if e.has_article and e.queue_status != "done"]

            canonical_reasons = Counter()
            ner_reasons = Counter()
            ner_parse_modes = Counter()
            sim_reject_reasons = Counter()
            sim_skip_reasons = Counter()
            sim_failure_reasons = Counter()
            for ent in ents:
                canonical_reasons.update(ent.canonical_reason_counts)
                ner_reasons.update(ent.ner_reject_reasons)
                ner_parse_modes.update(ent.ner_summary_parse_modes)
                sim_reject_reasons.update(ent.similarity_reject_reasons)
                sim_skip_reasons.update(ent.similarity_skip_reasons)
                sim_failure_reasons.update(ent.similarity_failure_reasons)

            row = {
                "hop": hop,
                "entities": len(ents),
                "entities_processed": len(processed),
                "entities_unprocessed": len(unprocessed),
                "entities_ner_processed": sum(1 for e in ents if e.queue_status == "done"),
                "entities_sim_processed": sum(1 for e in ents if e.sim_processed),
                "entities_with_article": sum(1 for e in ents if e.has_article),
                # ── Candidate counts from PROCESSED entities only ──
                "raw_wikilink_occurrences": sum(e.raw_wikilink_occurrences for e in ents),
                "raw_wikilink_unique": sum(e.raw_wikilink_unique for e in ents),
                "extracted_link_targets": sum(e.extracted_link_targets for e in ents),
                "extracted_categories": sum(e.extracted_categories for e in ents),
                "ner_candidates": sum(e.ner_candidates for e in ents),
                "ner_candidates_processed": sum(e.ner_candidates for e in processed),
                "ner_candidates_unprocessed": sum(e.ner_candidates for e in unprocessed),
                "canonical_pre_ner_input": sum(e.canonical_pre_ner_input for e in ents),
                "canonical_kept": sum(e.canonical_kept for e in ents),
                "canonical_rejected_total": sum(e.canonical_rejected_total for e in ents),
                "canonical_rejected_seen": sum(e.canonical_rejected_seen for e in ents),
                "canonical_rejected_batch": sum(e.canonical_rejected_batch for e in ents),
                "canonical_rejected_plural": sum(e.canonical_rejected_plural for e in ents),
                "canonical_rejected_other": sum(e.canonical_rejected_other for e in ents),
                "canonical_reason_counts": json_counter(canonical_reasons),
                "ner_accepted": sum(e.ner_accepted for e in ents),
                "ner_rejected": sum(e.ner_rejected for e in ents),
                "ner_rejected_not_ne": sum(e.ner_rejected_not_ne for e in ents),
                "ner_rejected_low_conf": sum(e.ner_rejected_low_conf for e in ents),
                "ner_rejected_parse_fail": sum(e.ner_rejected_parse_fail for e in ents),
                "ner_reject_reason_counts": json_counter(ner_reasons),
                "ner_summary_parse_modes": json_counter(ner_parse_modes),
                "ner_summary_error_count": sum(e.ner_summary_error_count for e in ents),
                "similarity_accepted": sum(e.similarity_accepted for e in ents),
                "similarity_rejected": sum(e.similarity_rejected for e in ents),
                "similarity_skipped": sum(e.similarity_skipped for e in ents),
                "sim_rejected_llm_duplicate": sum(e.sim_rejected_llm_duplicate for e in ents),
                "sim_rejected_within_batch": sum(e.sim_rejected_within_batch for e in ents),
                "sim_rejected_above_threshold_db": sum(e.sim_rejected_above_threshold_db for e in ents),
                "sim_rejected_above_threshold_batch": sum(e.sim_rejected_above_threshold_batch for e in ents),
                "sim_rejected_failure": sum(e.sim_rejected_failure for e in ents),
                "sim_rejected_other": sum(e.sim_rejected_other for e in ents),
                "similarity_reject_reason_counts": json_counter(sim_reject_reasons),
                "similarity_skip_reason_counts": json_counter(sim_skip_reasons),
                "similarity_failure_reason_counts": json_counter(sim_failure_reasons),
                "inserted_children": sum(e.inserted_children for e in ents),
            }
            row["canonical_survival"] = rate(row["canonical_kept"], row["canonical_pre_ner_input"])
            row["ner_survival"] = rate(row["ner_accepted"], row["canonical_kept"])
            row["similarity_survival"] = rate(row["similarity_accepted"], row["ner_accepted"])
            row["overall_survival_to_inserted"] = rate(row["inserted_children"], row["canonical_pre_ner_input"])
            rows.append(row)
        return rows

    def overall_summary(self) -> dict:
        hop_rows = self.hop_summary_rows()
        totals = Counter()
        for row in hop_rows:
            for k, v in row.items():
                if isinstance(v, int) and k != "hop":
                    totals[k] += v
        return {
            "run_dir": self.run_dir,
            "seed": self.run_meta.get("seed"),
            "mode": self.run_meta.get("mode"),
            "domain": self.run_meta.get("domain"),
            "max_depth": self.run_meta.get("max_depth"),
            "max_subjects": self.run_meta.get("max_subjects"),
            "entities_total": len(self.entity_stats),
            **dict(totals),
        }

    def tree_rows(self) -> List[dict]:
        rows = []
        for (subject, hop), ent in sorted(
            self.entity_stats.items(), key=lambda x: (x[0][1], x[0][0].lower())
        ):
            rows.append({
                "subject": subject,
                "hop": hop,
                "parent_subject": ent.parent_subject or "",
                "parent_hop": "" if ent.parent_hop is None else ent.parent_hop,
                "inserted_children": ent.inserted_children,
                "queue_status": ent.queue_status or "",
                "ner_processed": ent.ner_processed,
                "sim_processed": ent.sim_processed,
                "has_article": ent.has_article,
            })
        return rows

    def analyze(self) -> dict:
        self.load_queue()
        self.load_articles()
        self.load_pre_ner()
        self.load_ner()
        self.load_similarity()
        self.finalize()

        entity_rows = [
            self.entity_stats[k].to_row()
            for k in sorted(self.entity_stats, key=lambda x: (x[1], x[0].lower()))
        ]
        return {
            "overall_summary": self.overall_summary(),
            "hop_summary": self.hop_summary_rows(),
            "entity_rows": entity_rows,
            "tree_rows": self.tree_rows(),
        }


# ============================================================
# Report rendering helpers
# ============================================================

def pct(n: int, d: int) -> str:
    if not d:
        return "  n/a"
    return f"{100.0 * n / d:5.1f}%"


def _bar(n: int, total: int, width: int = 28) -> str:
    if total <= 0 or n < 0:
        return "░" * width
    filled = int(round(min(1.0, n / total) * width))
    return "█" * filled + "░" * (width - filled)


def _row(label: str, n: int, d: int, bar: bool = True, indent: int = 2) -> str:
    sp = " " * indent
    b = f"  [{_bar(n, d)}]" if bar else ""
    return f"{sp}{label:<40} {n:>9,}  {pct(n, d)}{b}"


def _stage_row(label: str, inp: int, out: int, loss: int) -> str:
    return (
        f"  {label:<30} {inp:>10,}  →  {out:>10,}  "
        f"loss={loss:>9,}  ({pct(loss, inp)} lost, {pct(out, inp)} surv)"
    )


# ============================================================
# Full detailed report
# ============================================================

def render_full_report(summary: dict) -> str:
    overall = summary["overall_summary"]
    hop_rows = summary["hop_summary"]
    entity_rows = summary.get("entity_rows", [])

    lines: List[str] = []

    def h1(title: str):
        lines.append("=" * 74)
        lines.append(f"  {title}")
        lines.append("=" * 74)

    def h2(title: str):
        lines.append("")
        lines.append(f"── {title} " + "─" * max(0, 70 - len(title)))

    def blank():
        lines.append("")

    # ── Header ──────────────────────────────────────────────────
    h1("LLMPedia Pipeline Funnel Report")
    lines.append(f"  Run dir : {overall.get('run_dir', '')}")
    lines.append(f"  Seed    : {overall.get('seed', '')}")
    lines.append(
        f"  Mode    : {overall.get('mode', '')}  |  "
        f"Domain: {overall.get('domain', '')}  |  "
        f"Depth: {overall.get('max_depth', '')}  |  "
        f"Max subjects: {overall.get('max_subjects', '')}"
    )

    total_ents = overall.get("entities_total", 0)
    ner_proc   = overall.get("entities_ner_processed", 0)
    sim_proc   = overall.get("entities_sim_processed", 0)
    with_art   = overall.get("entities_with_article", 0)
    unproc     = overall.get("entities_unprocessed", 0)

    lines.append(
        f"  Entities: {total_ents:,}  |  "
        f"With articles: {with_art:,}  |  "
        f"NER-processed: {ner_proc:,}  |  "
        f"Unprocessed: {unproc:,}"
    )

    # ── 0. Processing Coverage ──────────────────────────────────
    if unproc > 0:
        h2("0. Processing Coverage Warning")
        blank()
        lines.append(
            f"  ⚠  {unproc:,} entities ({pct(unproc, total_ents).strip()}) had articles generated"
        )
        lines.append(
            f"     but were NEVER processed through the NER/similarity pipeline."
        )
        lines.append(
            f"     These entities produced {overall.get('ner_candidates_unprocessed', 0):,} "
            f"candidate mentions in their articles,"
        )
        lines.append(
            f"     but those candidates never entered the funnel."
        )
        lines.append(
            f"     All funnel numbers below reflect ONLY the {ner_proc:,} actually-processed entities."
        )
        blank()

        lines.append(f"  {'Hop':>3}  {'Total':>10}  {'Processed':>10}  {'Unprocessed':>12}  {'% processed':>12}")
        lines.append("  " + "-" * 54)
        for row in hop_rows:
            h = row["hop"]
            tot = row["entities"]
            proc = row["entities_processed"]
            unp = row["entities_unprocessed"]
            pp = f"{100.0 * proc / tot:.1f}%" if tot > 0 else "n/a"
            lines.append(f"  {h:>3}  {tot:>10,}  {proc:>10,}  {unp:>12,}  {pp:>12}")
        blank()

    # ── 1. Overall pipeline totals ───────────────────────────────
    h2("1. Overall Pipeline Totals")
    blank()
    lines.append(
        f"  {'Stage':<32} {'IN':>11}  {'OUT':>11}  {'LOSS':>11}  LOSS%   SURV%"
    )
    lines.append("  " + "-" * 74)

    gen       = overall.get("ner_candidates_processed", overall.get("ner_candidates", 0))
    can_in    = overall.get("canonical_pre_ner_input", gen)
    can_kept  = overall.get("canonical_kept", 0)
    can_loss  = overall.get("canonical_rejected_total", can_in - can_kept)
    ner_in    = can_kept
    ner_out   = overall.get("ner_accepted", 0)
    ner_loss  = overall.get("ner_rejected", 0)
    sim_skip  = overall.get("similarity_skipped", 0)
    sim_out   = overall.get("similarity_accepted", 0)
    sim_rej   = overall.get("similarity_rejected", 0)
    sim_in    = ner_out + sim_skip
    inserted  = overall.get("inserted_children", 0)

    lines.append(_stage_row("Candidates generated", gen, gen, 0))
    lines.append(_stage_row("Pre-NER dedup  → canon_kept", can_in, can_kept, can_loss))
    lines.append(_stage_row("NER filter     → ner_accepted", ner_in, ner_out, ner_loss))
    lines.append(_stage_row("Sim filter     → sim_accepted", sim_in, sim_out, sim_rej + sim_skip))
    lines.append(_stage_row("Queue insert   → inserted", sim_out, inserted, sim_out - inserted))
    blank()

    # ── 2. Pre-NER dedup breakdown ───────────────────────────────
    h2("2. Pre-NER Dedup Losses")
    blank()

    seen_total   = sum(r.get("canonical_rejected_seen",   0) for r in hop_rows)
    batch_total  = sum(r.get("canonical_rejected_batch",  0) for r in hop_rows)
    plural_total = sum(r.get("canonical_rejected_plural", 0) for r in hop_rows)
    other_total  = sum(r.get("canonical_rejected_other",  0) for r in hop_rows)

    lines.append(f"  Total pre-NER rejects: {can_loss:,}  (out of {can_in:,} input candidates)")
    blank()
    lines.append(f"  {'Reason':<40} {'Count':>9}   % total  Bar")
    lines.append("  " + "-" * 72)
    for label, n in [
        ("Already seen (global canon)",       seen_total),
        ("Within-batch duplicate",            batch_total),
        ("Plural/singular variant",           plural_total),
        ("Other",                             other_total),
    ]:
        lines.append(_row(label, n, can_loss))

    all_canon_reasons: Counter = Counter()
    for er in entity_rows:
        try:
            all_canon_reasons.update(json.loads(er.get("canonical_reason_counts") or "{}"))
        except Exception:
            pass
    if all_canon_reasons:
        blank()
        lines.append("  Top canonical reject reasons:")
        for reason, count in all_canon_reasons.most_common(10):
            lines.append(f"    {reason:<47}  {count:>9,}")

    # ── 3. NER filter breakdown ──────────────────────────────────
    h2("3. NER Filter")
    blank()

    ner_not_ne     = sum(r.get("ner_rejected_not_ne",     0) for r in hop_rows)
    ner_low_conf   = sum(r.get("ner_rejected_low_conf",   0) for r in hop_rows)
    ner_parse_fail = sum(r.get("ner_rejected_parse_fail", 0) for r in hop_rows)

    lines.append(f"  Input (post-canonical):  {ner_in:>11,}")
    lines.append(_row("Accepted (is_ne=true)",                   ner_out,   ner_in))
    lines.append(_row("Rejected total",                          ner_loss,  ner_in))
    blank()
    lines.append(f"  {'Reject sub-reason':<40} {'Count':>9}   % rejected  Bar")
    lines.append("  " + "-" * 72)
    for label, n in [
        ("not_named_entity  (is_ne=false)",          ner_not_ne),
        ("below_conf_threshold / missing conf",       ner_low_conf),
        ("parse/call failure  (strict gate)",         ner_parse_fail),
    ]:
        lines.append(_row(label, n, ner_loss))

    all_parse_modes: Counter = Counter()
    for er in entity_rows:
        try:
            all_parse_modes.update(json.loads(er.get("ner_summary_parse_modes") or "{}"))
        except Exception:
            pass
    if all_parse_modes:
        blank()
        total_chunks = sum(all_parse_modes.values())
        lines.append(f"  NER parse modes  (total chunks: {total_chunks:,}):")
        for mode, count in all_parse_modes.most_common():
            lines.append(f"    {mode:<32}  {count:>9,}  {pct(count, total_chunks)}")

    all_ner_reasons: Counter = Counter()
    for er in entity_rows:
        try:
            all_ner_reasons.update(json.loads(er.get("ner_reject_reason_counts") or "{}"))
        except Exception:
            pass
    if all_ner_reasons:
        blank()
        lines.append("  Top NER reject reasons:")
        for reason, count in all_ner_reasons.most_common(10):
            lines.append(f"    {reason:<47}  {count:>9,}")

    # ── 4. Similarity filter breakdown ──────────────────────────
    h2("4. Similarity Filter")
    blank()

    sim_total = sim_out + sim_rej + sim_skip

    sim_rej_llm  = sum(r.get("sim_rejected_llm_duplicate",         0) for r in hop_rows)
    sim_rej_wb   = sum(r.get("sim_rejected_within_batch",          0) for r in hop_rows)
    sim_rej_db   = sum(r.get("sim_rejected_above_threshold_db",    0) for r in hop_rows)
    sim_rej_bt   = sum(r.get("sim_rejected_above_threshold_batch", 0) for r in hop_rows)
    sim_rej_fail = sum(r.get("sim_rejected_failure",               0) for r in hop_rows)
    sim_rej_oth  = sum(r.get("sim_rejected_other",                 0) for r in hop_rows)

    lines.append(f"  Total candidates entering similarity: {sim_total:>11,}")
    blank()
    lines.append(f"  {'Outcome':<40} {'Count':>9}   % total  Bar")
    lines.append("  " + "-" * 72)
    lines.append(_row("Accepted  (not duplicate)",                    sim_out,  sim_total))
    lines.append(_row("Skipped   (already indexed in DB)",            sim_skip, sim_total))
    lines.append(_row("Rejected  (duplicate)",                        sim_rej,  sim_total))

    if sim_rej > 0:
        blank()
        lines.append(f"  Rejected sub-reasons  (out of {sim_rej:,} rejected):")
        lines.append(f"  {'Reason':<40} {'Count':>9}   % rejected  Bar")
        lines.append("  " + "-" * 72)
        for label, n in [
            ("llm_duplicate  (LLM confirmed dup)",           sim_rej_llm),
            ("within_batch_duplicate",                        sim_rej_wb),
            ("above_threshold_db  (embedding, no LLM)",      sim_rej_db),
            ("above_threshold_batch  (embedding, no LLM)",   sim_rej_bt),
            ("operational_failure  (batch/SDK error)",        sim_rej_fail),
            ("other",                                         sim_rej_oth),
        ]:
            lines.append(_row(label, n, sim_rej))

    all_sim_reasons: Counter = Counter()
    for er in entity_rows:
        try:
            all_sim_reasons.update(json.loads(er.get("similarity_reject_reason_counts") or "{}"))
        except Exception:
            pass
    if all_sim_reasons:
        blank()
        lines.append("  Top similarity reject reasons (raw):")
        for reason, count in all_sim_reasons.most_common(10):
            lines.append(f"    {reason:<47}  {count:>9,}")

    all_fail_reasons: Counter = Counter()
    for er in entity_rows:
        try:
            all_fail_reasons.update(json.loads(er.get("similarity_failure_reason_counts") or "{}"))
        except Exception:
            pass
    if all_fail_reasons:
        blank()
        lines.append("  Operational failure reasons:")
        for reason, count in all_fail_reasons.most_common(10):
            lines.append(f"    {reason:<47}  {count:>9,}")

    # ── 7. Accept / reject rate summary ───────────────────────
    h2("7. Accept / Reject Rate Summary")
    blank()

    def _rate_row(label: str, accepted: int, rejected: int, skipped: int = 0):
        total = accepted + rejected + skipped
        acc_pct  = f"{100.0 * accepted  / total:6.2f}%" if total else "    n/a"
        rej_pct  = f"{100.0 * rejected  / total:6.2f}%" if total else "    n/a"
        skip_pct = f"{100.0 * skipped   / total:6.2f}%" if total else "    n/a"
        return (
            f"  {label:<28}  total={total:>10,}  "
            f"accept={accepted:>10,} ({acc_pct})  "
            f"reject={rejected:>10,} ({rej_pct})"
            + (f"  skip={skipped:>10,} ({skip_pct})" if skipped else "")
        )

    lines.append(_rate_row("Pre-NER dedup",  can_kept, can_loss))
    lines.append(_rate_row("NER filter",     ner_out,  ner_loss))
    lines.append(_rate_row("Similarity",     sim_out,  sim_rej, sim_skip))
    blank()

    # Per-stage rejection reason rates
    lines.append("  Pre-NER dedup — rejection reason rates:")
    lines.append(f"  {'Reason':<42} {'Count':>9}  {'% of rejects':>13}  {'% of total input':>17}")
    lines.append("  " + "-" * 84)
    for label, n in [
        ("already seen (global canon)",    seen_total),
        ("within-batch duplicate",         batch_total),
        ("plural/singular variant",        plural_total),
        ("other",                          other_total),
    ]:
        r_pct = f"{100.0 * n / can_loss:6.2f}%" if can_loss else "   n/a"
        i_pct = f"{100.0 * n / can_in  :6.2f}%" if can_in   else "   n/a"
        lines.append(f"    {label:<40}  {n:>9,}  {r_pct:>13}  {i_pct:>17}")
    blank()

    lines.append("  NER filter — rejection reason rates:")
    lines.append(f"  {'Reason':<42} {'Count':>9}  {'% of rejects':>13}  {'% of NER input':>15}")
    lines.append("  " + "-" * 82)
    for label, n in [
        ("not_named_entity  (is_ne=false)",          ner_not_ne),
        ("below_conf_threshold / missing conf",       ner_low_conf),
        ("parse / call failure  (strict gate)",       ner_parse_fail),
    ]:
        r_pct = f"{100.0 * n / ner_loss:6.2f}%" if ner_loss else "   n/a"
        i_pct = f"{100.0 * n / ner_in  :6.2f}%" if ner_in   else "   n/a"
        lines.append(f"    {label:<40}  {n:>9,}  {r_pct:>13}  {i_pct:>15}")
    blank()

    if sim_rej > 0 or sim_skip > 0:
        sim_total_check = sim_out + sim_rej + sim_skip
        lines.append("  Similarity filter — rejection reason rates:")
        lines.append(f"  {'Reason':<42} {'Count':>9}  {'% of rejects':>13}  {'% of total':>11}")
        lines.append("  " + "-" * 78)
        for label, n in [
            ("accepted (not duplicate)",                 sim_out),
            ("skipped  (already indexed)",               sim_skip),
            ("llm_duplicate  (LLM confirmed dup)",       sim_rej_llm),
            ("within_batch_duplicate",                   sim_rej_wb),
            ("above_threshold_db  (embedding only)",     sim_rej_db),
            ("above_threshold_batch",                    sim_rej_bt),
            ("operational_failure",                      sim_rej_fail),
            ("other",                                    sim_rej_oth),
        ]:
            r_pct = f"{100.0 * n / sim_rej          :6.2f}%" if sim_rej           else "   n/a"
            t_pct = f"{100.0 * n / sim_total_check  :6.2f}%" if sim_total_check   else "   n/a"
            lines.append(f"    {label:<40}  {n:>9,}  {r_pct:>13}  {t_pct:>11}")
    blank()

    # ── 8. By-hop breakdown ──────────────────────────────────────
    h2("8. By-Hop Breakdown")
    blank()

    cw = 10
    header = (
        f"  {'Hop':>3}  {'Ents':>{cw}}  {'Proc':>{cw}}  {'CandIN':>{cw}}  "
        f"{'CanonKpt':>{cw}}  {'NER_acc':>{cw}}  "
        f"{'Sim_skip':>{cw}}  {'Sim_acc':>{cw}}  {'Sim_rej':>{cw}}  {'Inserted':>{cw}}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for row in hop_rows:
        lines.append(
            f"  {row['hop']:>3}  "
            f"{row['entities']:>{cw},}  "
            f"{row['entities_processed']:>{cw},}  "
            f"{row['canonical_pre_ner_input']:>{cw},}  "
            f"{row['canonical_kept']:>{cw},}  "
            f"{row['ner_accepted']:>{cw},}  "
            f"{row['similarity_skipped']:>{cw},}  "
            f"{row['similarity_accepted']:>{cw},}  "
            f"{row['similarity_rejected']:>{cw},}  "
            f"{row['inserted_children']:>{cw},}"
        )

    # ── 9. Survival rates per hop ────────────────────────────────
    h2("9. Survival Rates by Hop")
    blank()
    lines.append(f"  {'Hop':>3}  {'canon_surv':>12}  {'ner_surv':>11}  {'sim_surv':>10}  {'overall':>10}")
    lines.append("  " + "-" * 54)

    def _fmtr(v: Optional[float]) -> str:
        return f"{v:.2%}" if v is not None else "     n/a"

    for row in hop_rows:
        lines.append(
            f"  {row['hop']:>3}  "
            f"{_fmtr(row.get('canonical_survival')):>12}  "
            f"{_fmtr(row.get('ner_survival')):>11}  "
            f"{_fmtr(row.get('similarity_survival')):>10}  "
            f"{_fmtr(row.get('overall_survival_to_inserted')):>10}"
        )

    # ── 10. Per-hop mean ± std statistics ───────────────────────
    h2("10. Per-Hop Statistics  (mean ± std across PROCESSED entities)")
    blank()

    import statistics as _stats

    # Collect per-entity values grouped by hop — ONLY from processed entities
    _hop_data: Dict[int, Dict[str, List[float]]] = {}
    _stat_keys = [
        ("canonical_survival",           "canon_surv"),
        ("ner_survival",                 "ner_surv"),
        ("similarity_survival",          "sim_surv"),
        ("overall_survival_to_inserted", "overall_surv"),
        ("canonical_kept",               "canon_kept_n"),
        ("ner_accepted",                 "ner_acc_n"),
        ("similarity_accepted",          "sim_acc_n"),
        ("inserted_children",            "inserted_n"),
    ]
    for er in entity_rows:
        # Only include processed entities in statistics
        if er.get("queue_status") != "done":
            continue
        h = er.get("hop")
        if h is None:
            continue
        if h not in _hop_data:
            _hop_data[h] = {short: [] for _, short in _stat_keys}
        for full, short in _stat_keys:
            v = er.get(full)
            if v is not None:
                _hop_data[h][short].append(float(v))

    def _ms(vals: List[float], pct: bool = False) -> str:
        """Format mean ± std; optionally as percentage."""
        if not vals:
            return "        n/a"
        mu  = sum(vals) / len(vals)
        std = (_stats.pstdev(vals) if len(vals) > 1 else 0.0)
        if pct:
            return f"{mu*100:6.1f} ± {std*100:5.1f}%"
        return f"{mu:8.1f} ± {std:7.1f}"

    # ── 10a. Survival rates ──────────────────────────────────────
    lines.append("  10a. Survival rates per hop  (mean ± population std across PROCESSED entities)")
    blank()
    hdr = (f"  {'Hop':>3}  {'N processed':>11}  "
           f"{'canon_surv':>18}  {'ner_surv':>18}  "
           f"{'sim_surv':>18}  {'overall':>18}")
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))

    for h in sorted(_hop_data.keys()):
        d = _hop_data[h]
        n = max((len(d[k]) for k in d if d[k]), default=0)
        lines.append(
            f"  {h:>3}  {n:>11,}  "
            f"{_ms(d['canon_surv'], pct=True):>18}  "
            f"{_ms(d['ner_surv'],   pct=True):>18}  "
            f"{_ms(d['sim_surv'],   pct=True):>18}  "
            f"{_ms(d['overall_surv'], pct=True):>18}"
        )

    blank()
    lines.append("  Note: std measures entity-level variance within each hop.")
    lines.append("  High std = some entities are processed very differently (e.g. niche")
    lines.append("  vs. well-connected topics). Low std = consistent pipeline behaviour.")
    lines.append("  Only entities that were actually NER/sim-processed are included.")

    blank()
    # ── 10b. Candidate counts ────────────────────────────────────
    lines.append("  10b. Candidate counts per entity per hop  (mean ± std, PROCESSED only)")
    blank()
    hdr2 = (f"  {'Hop':>3}  {'N processed':>11}  "
            f"{'canon_kept':>20}  {'ner_accepted':>20}  "
            f"{'sim_accepted':>20}  {'inserted':>20}")
    lines.append(hdr2)
    lines.append("  " + "-" * (len(hdr2) - 2))

    for h in sorted(_hop_data.keys()):
        d = _hop_data[h]
        n = max((len(d[k]) for k in d if d[k]), default=0)
        lines.append(
            f"  {h:>3}  {n:>11,}  "
            f"{_ms(d['canon_kept_n']):>20}  "
            f"{_ms(d['ner_acc_n']):>20}  "
            f"{_ms(d['sim_acc_n']):>20}  "
            f"{_ms(d['inserted_n']):>20}"
        )

    blank()
    lines.append("  Note: values are raw candidate counts per entity (not rates).")
    lines.append("  Large std at deep hops reflects the power-law distribution of")
    lines.append("  wikilink counts across articles.")
    blank()

    blank()
    lines.append("=" * 74)
    lines.append("  END OF REPORT")
    lines.append("=" * 74)
    return "\n".join(lines)


# ============================================================
# Short / legacy render
# ============================================================

def render_console(summary: dict) -> str:
    overall = summary["overall_summary"]
    hop_rows = summary["hop_summary"]
    lines = []
    lines.append(f"Run: {overall.get('run_dir')}")
    lines.append(f"Seed: {overall.get('seed')}")
    lines.append(
        f"Entities: {overall.get('entities_total', 0)} | "
        f"NER-processed: {overall.get('entities_ner_processed', 0)} | "
        f"Unprocessed: {overall.get('entities_unprocessed', 0)}"
    )
    lines.append(
        "Overall funnel (processed only): "
        f"generated={overall.get('ner_candidates_processed', 0)} -> "
        f"canonical_kept={overall.get('canonical_kept', 0)} -> "
        f"ner_accept={overall.get('ner_accepted', 0)} -> "
        f"sim_accept={overall.get('similarity_accepted', 0)} -> "
        f"inserted={overall.get('inserted_children', 0)}"
    )
    lines.append(
        "Losses: "
        f"canonical_reject={overall.get('canonical_rejected_total', 0)} | "
        f"ner_reject={overall.get('ner_rejected', 0)} | "
        f"sim_reject={overall.get('similarity_rejected', 0)} | "
        f"sim_skip={overall.get('similarity_skipped', 0)}"
    )
    lines.append("")
    lines.append("By hop:")
    for row in hop_rows:
        lines.append(
            f"  hop={row['hop']}: entities={row['entities']} (proc={row['entities_processed']}) | "
            f"canon_input={row['canonical_pre_ner_input']} | "
            f"canon_kept={row['canonical_kept']} | "
            f"ner_acc={row['ner_accepted']} | ner_rej={row['ner_rejected']} | "
            f"sim_acc={row['similarity_accepted']} | sim_rej={row['similarity_rejected']} | "
            f"sim_skip={row['similarity_skipped']} | inserted={row['inserted_children']}"
        )
    return "\n".join(lines)


# ============================================================
# Figure generation (requires matplotlib)
# ============================================================

def _savefig(fig, path: str):
    """Save white-background PNG at 300 DPI."""
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {path}", flush=True)


def _savefig_variants(fig_factory, base_path: str) -> list:
    written = []
    fig_w = fig_factory(transparent=False)
    p_w = base_path + "_white.png"
    fig_w.savefig(p_w, dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig_w)
    print(f"  saved: {p_w}", flush=True)
    written.append(p_w)

    fig_t = fig_factory(transparent=True)
    p_t = base_path + "_transp.png"
    fig_t.savefig(p_t, dpi=300, bbox_inches="tight", facecolor="none", transparent=True)
    plt.close(fig_t)
    print(f"  saved: {p_t}", flush=True)
    written.append(p_t)
    return written


def _pub_style(ax, transparent: bool = False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(colors="#444444", labelsize=9)
    ax.yaxis.label.set_color("#444444")
    ax.xaxis.label.set_color("#444444")
    ax.title.set_color("#222222")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, color="#e0e0e0", alpha=0.8)
    ax.set_axisbelow(True)
    if transparent:
        ax.set_facecolor("none")
        ax.patch.set_alpha(0.0)


def render_figures(summary: dict, figures_dir: str) -> List[str]:
    if not _HAS_MPL:
        print("[figures] matplotlib not installed — skipping figures. "
              "Install with:  pip install matplotlib", flush=True)
        return []

    os.makedirs(figures_dir, exist_ok=True)
    written = []
    overall     = summary["overall_summary"]
    hop_rows    = summary["hop_summary"]
    entity_rows = summary.get("entity_rows", [])

    C_BLUE   = "#0072B2"
    C_GREEN  = "#009E73"
    C_ORANGE = "#E69F00"
    C_PURPLE = "#CC79A7"
    C_VERMIL = "#D55E00"
    C_SKY    = "#56B4E9"
    C_YELLOW = "#F0E442"
    C_RED    = "#c0392b"
    C_GREY   = "#7f8c8d"
    C_TEAL   = "#16a085"

    plt.rcParams.update({
        "font.family":          "DejaVu Sans",
        "axes.titlesize":       12,
        "axes.titleweight":     "bold",
        "axes.titlepad":        10,
        "axes.labelsize":       10,
        "xtick.labelsize":      9,
        "ytick.labelsize":      9,
        "legend.fontsize":      9,
        "legend.framealpha":    0.9,
        "legend.edgecolor":     "#dddddd",
        "legend.borderpad":     0.5,
        "legend.handlelength":  1.8,
        "figure.facecolor":     "white",
        "axes.facecolor":       "white",
        "lines.linewidth":      1.9,
        "lines.markersize":     7,
    })

    def _hbar_chart(ax, labels, values, colors, title, xlabel):
        y = np.arange(len(labels))
        bars = ax.barh(y, values, color=colors, edgecolor="white",
                       height=0.55, linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.invert_yaxis()
        max_v = max(values) if any(v > 0 for v in values) else 1
        for bar, v in zip(bars, values):
            if v > 0:
                ax.text(bar.get_width() + max_v * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{v:,}", va="center", ha="left", fontsize=9,
                        color="#333333")
        ax.set_xlim(0, max_v * 1.18)
        _pub_style(ax)
        ax.grid(axis="x", linestyle="--", linewidth=0.5, color="#e0e0e0")
        ax.grid(axis="y", visible=False)

    # ── Per-entity groupings (PROCESSED entities only) ──────────
    from collections import defaultdict as _dd_fig
    _hop_ent: dict = _dd_fig(lambda: {k: [] for k in (
        "canonical_kept", "ner_accepted", "similarity_accepted", "inserted_children",
        "canonical_survival", "ner_survival",
        "similarity_survival", "overall_survival_to_inserted",
    )})
    for _er in entity_rows:
        # Only include processed entities
        if _er.get("queue_status") != "done":
            continue
        _h = _er.get("hop")
        if _h is None:
            continue
        for _k in _hop_ent[_h]:
            _v = _er.get(_k)
            if _v is not None:
                _hop_ent[_h][_k].append(float(_v))

    def _mu_sd(hop_list, key, scale=1.0):
        mus, sds = [], []
        for h in hop_list:
            vals = _hop_ent[h].get(key, [])
            if vals:
                a = np.array(vals) * scale
                mus.append(float(a.mean())); sds.append(float(a.std()))
            else:
                mus.append(0.0); sds.append(0.0)
        return np.array(mus), np.array(sds)

    # ────────────────────────────────────────────────────────────
    # Figure 01 — Pipeline funnel
    # ────────────────────────────────────────────────────────────
    gen      = overall.get("ner_candidates_processed", overall.get("ner_candidates", 0))
    can_in   = overall.get("canonical_pre_ner_input", gen)
    can_kept = overall.get("canonical_kept", 0)
    ner_out  = overall.get("ner_accepted", 0)
    sim_out  = overall.get("similarity_accepted", 0)
    inserted = overall.get("inserted_children", 0)

    stage_labels = ["Generated\n(processed)", "Post-dedup\n(canon)", "Post-NER", "Post-sim", "Inserted"]
    stage_values = [gen, can_kept, ner_out, sim_out, inserted]
    stage_colors = [C_BLUE, C_TEAL, C_GREEN, C_GREEN, "#1a6634"]

    fig1, ax1 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(stage_labels))
    bars = ax1.bar(x, stage_values, color=stage_colors, width=0.55,
                   edgecolor="white", linewidth=0.8, zorder=3)

    mx = max(stage_values) if stage_values else 1
    for bar, v in zip(bars, stage_values):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + mx * 0.012,
                 f"{v:,}", ha="center", va="bottom", fontsize=9,
                 fontweight="bold", color="#222222")

    for i in range(len(stage_values) - 1):
        prev, curr = stage_values[i], stage_values[i + 1]
        loss = prev - curr
        if prev > 0 and loss > 0:
            ax1.annotate(
                f"−{100.0*loss/prev:.1f}%",
                xy=(i + 0.5, max(prev, curr) * 0.48),
                ha="center", va="center", fontsize=8.5, color=C_RED, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=C_RED, lw=0.9, alpha=0.9),
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(stage_labels, fontsize=10)
    ax1.set_ylabel("Candidates", fontsize=10)
    ax1.set_title("LLMPedia Pipeline — Funnel Overview (processed entities only)", fontsize=13)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax1.set_ylim(0, mx * 1.20)
    _pub_style(ax1)
    fig1.tight_layout()
    p = os.path.join(figures_dir, "01_funnel_stages.png")
    _savefig(fig1, p); written.append(p)

    # ────────────────────────────────────────────────────────────
    # Figure 02a — NER accept vs reject
    # Figure 02b — NER reject reasons
    # ────────────────────────────────────────────────────────────
    ner_loss     = overall.get("ner_rejected", 0)
    ner_not_ne   = sum(r.get("ner_rejected_not_ne",     0) for r in hop_rows)
    ner_low_conf = sum(r.get("ner_rejected_low_conf",   0) for r in hop_rows)
    ner_parse    = sum(r.get("ner_rejected_parse_fail", 0) for r in hop_rows)

    fig2a, ax2a = plt.subplots(figsize=(8, 2.8))
    total_ner = ner_out + ner_loss
    if total_ner > 0:
        ax2a.barh(["NER"], [ner_out],  color=C_GREEN, label=f"Accepted  ({100*ner_out/total_ner:.1f}%)",  height=0.4)
        ax2a.barh(["NER"], [ner_loss], color=C_RED,   label=f"Rejected  ({100*ner_loss/total_ner:.1f}%)",
                  left=[ner_out], height=0.4)
        for val, left, col in [(ner_out, 0, "white"), (ner_loss, ner_out, "white")]:
            if val > total_ner * 0.04:
                ax2a.text(left + val/2, 0,
                          f"{val:,}\n({100*val/total_ner:.1f}%)",
                          ha="center", va="center", fontsize=9,
                          color=col, fontweight="bold")
    ax2a.set_title("NER Filter — Accept vs. Reject", fontsize=12)
    ax2a.set_xlabel("Candidates", fontsize=10)
    ax2a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax2a.legend(loc="lower right", fontsize=9)
    _pub_style(ax2a)
    ax2a.grid(axis="x", linestyle="--", linewidth=0.5, color="#e0e0e0")
    ax2a.grid(axis="y", visible=False)
    fig2a.tight_layout()
    p = os.path.join(figures_dir, "02a_ner_accept_reject.png")
    _savefig(fig2a, p); written.append(p)

    fig2b, ax2b = plt.subplots(figsize=(8, 3.5))
    ner_rl = ["not_named_entity", "below_conf_threshold", "parse / call failure"]
    ner_rv = [ner_not_ne, ner_low_conf, ner_parse]
    ner_rc = [C_ORANGE, C_PURPLE, C_GREY]
    _hbar_chart(ax2b, ner_rl, ner_rv, ner_rc, "NER Filter — Reject Sub-reasons", "Count")
    if ner_loss > 0:
        ax2b.set_yticklabels(
            [f"{lbl}  ({100.0*v/ner_loss:.1f}%)" for lbl,v in zip(ner_rl,ner_rv)], fontsize=9)
    fig2b.tight_layout()
    p = os.path.join(figures_dir, "02b_ner_reject_reasons.png")
    _savefig(fig2b, p); written.append(p)

    # ────────────────────────────────────────────────────────────
    # Figure 03a — Similarity accept/skip/reject
    # Figure 03b — Similarity reject reasons
    # ────────────────────────────────────────────────────────────
    sim_skip     = overall.get("similarity_skipped", 0)
    sim_rej      = overall.get("similarity_rejected", 0)
    sim_rej_llm  = sum(r.get("sim_rejected_llm_duplicate",         0) for r in hop_rows)
    sim_rej_wb   = sum(r.get("sim_rejected_within_batch",          0) for r in hop_rows)
    sim_rej_db   = sum(r.get("sim_rejected_above_threshold_db",    0) for r in hop_rows)
    sim_rej_bt   = sum(r.get("sim_rejected_above_threshold_batch", 0) for r in hop_rows)
    sim_rej_fail = sum(r.get("sim_rejected_failure",               0) for r in hop_rows)
    sim_rej_oth  = sum(r.get("sim_rejected_other",                 0) for r in hop_rows)
    sim_total    = sim_out + sim_rej + sim_skip

    fig3a, ax3a = plt.subplots(figsize=(8, 2.8))
    if sim_total > 0:
        ax3a.barh(["Sim"], [sim_out],  color=C_GREEN, label=f"Accepted  ({100*sim_out/sim_total:.1f}%)",  height=0.4)
        ax3a.barh(["Sim"], [sim_skip], color=C_BLUE,  label=f"Skipped   ({100*sim_skip/sim_total:.1f}%)",
                  left=[sim_out], height=0.4)
        ax3a.barh(["Sim"], [sim_rej],  color=C_RED,   label=f"Rejected  ({100*sim_rej/sim_total:.1f}%)",
                  left=[sim_out + sim_skip], height=0.4)
        for val, left, col in [(sim_out, 0, "white"),
                                (sim_skip, sim_out, "white"),
                                (sim_rej, sim_out+sim_skip, "white")]:
            if val > sim_total * 0.03:
                ax3a.text(left + val/2, 0,
                          f"{val:,}\n({100*val/sim_total:.1f}%)",
                          ha="center", va="center", fontsize=9,
                          color=col, fontweight="bold")
    ax3a.set_title("Similarity Filter — Accept / Skip / Reject", fontsize=12)
    ax3a.set_xlabel("Candidates", fontsize=10)
    ax3a.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax3a.legend(loc="lower right", fontsize=9)
    _pub_style(ax3a)
    ax3a.grid(axis="x", linestyle="--", linewidth=0.5, color="#e0e0e0")
    ax3a.grid(axis="y", visible=False)
    fig3a.tight_layout()
    p = os.path.join(figures_dir, "03a_sim_accept_reject.png")
    _savefig(fig3a, p); written.append(p)

    fig3b, ax3b = plt.subplots(figsize=(8, 4))
    sim_rl = ["llm_duplicate", "within_batch", "above_threshold_db",
              "above_threshold_batch", "op_failure", "other"]
    sim_rv = [sim_rej_llm, sim_rej_wb, sim_rej_db, sim_rej_bt, sim_rej_fail, sim_rej_oth]
    sim_rc = [C_RED, C_ORANGE, C_PURPLE, C_BLUE, C_GREY, C_GREY]
    _hbar_chart(ax3b, sim_rl, sim_rv, sim_rc,
                "Similarity Filter — Reject Sub-reasons", "Count")
    if sim_rej > 0:
        ax3b.set_yticklabels(
            [f"{lbl}  ({100.0*v/sim_rej:.1f}%)" for lbl,v in zip(sim_rl,sim_rv)], fontsize=9)
    fig3b.tight_layout()
    p = os.path.join(figures_dir, "03b_sim_reject_reasons.png")
    _savefig(fig3b, p); written.append(p)

    # ────────────────────────────────────────────────────────────
    # Figure 04a — Per-hop mean ± 1σ line chart
    # Figure 04b — Per-hop totals bar chart (log-scale)
    # ────────────────────────────────────────────────────────────
    if hop_rows and entity_rows:
        # Only use hops that have processed entities
        hops = [r["hop"] for r in hop_rows if r.get("entities_processed", 0) > 0]
        if not hops:
            hops = [r["hop"] for r in hop_rows]
        xs4  = np.arange(len(hops))
        xlbls4 = [f"Hop {h}" for h in hops]

        count_series = [
            ("Canonical kept",    "canonical_kept",      C_BLUE,   "o"),
            ("NER accepted",    "ner_accepted",         C_GREEN,  "s"),
            ("Sim. accepted",   "similarity_accepted",  C_ORANGE, "^"),
            ("Inserted",        "inserted_children",    C_PURPLE, "D"),
        ]

        def _inset_table(ax, hop_list, series, scale=1.0, fmt=".1f", unit="",
                          transparent=False):
            _short = {
                "Canonical kept":     ("Canon.", "kept"),
                "NER accepted":       ("NER",    "acc."),
                "Sim. accepted":      ("Sim.",   "acc."),
                "Inserted":           ("",       "Inserted"),
                "Canonical survival": ("Canon.", "surv."),
                "NER survival":       ("NER",    "surv."),
                "Sim. survival":      ("Sim.",   "surv."),
                "Overall":            ("",       "Overall"),
            }
            col_w = 11
            hdr1 = "Hop  " + "  ".join(
                f"{_short.get(label, (label[:col_w], ''))[0]:>{col_w}}"
                for label, _, _, _ in series)
            hdr2 = "     " + "  ".join(
                f"{_short.get(label, ('', label[:col_w]))[1]:>{col_w}}"
                for label, _, _, _ in series)
            sep  = "\u2500" * max(len(hdr1), len(hdr2))
            lines_t = [hdr1, hdr2, sep]
            for h in hop_list:
                row = [f"  {h:<3}"]
                for label, key, color, marker in series:
                    vals = _hop_ent[h].get(key, [])
                    if vals:
                        a = np.array(vals) * scale
                        cell = f"{a.mean():{fmt}}\u00b1{a.std():{fmt}}{unit}"
                        row.append(cell.rjust(col_w))
                    else:
                        row.append("n/a".rjust(col_w))
                lines_t.append("  ".join(row))
            tbl = "\n".join(lines_t)
            fc = (1, 1, 1, 0.88) if transparent else "white"
            ax.text(0.99, 0.99, tbl,
                    transform=ax.transAxes,
                    fontsize=6.5, family="monospace", color="#333333",
                    va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.45", fc=fc,
                              ec="#cccccc", lw=0.8, alpha=0.93),
                    zorder=10)

        def _build_04a(transparent=False):
            fc = "none" if transparent else "white"
            fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=fc)
            for label, key, color, marker in count_series:
                mu, sig = _mu_sd(hops, key)
                lo = np.clip(mu - sig, 0, None); hi = mu + sig
                ax.fill_between(xs4, lo, hi, color=color, alpha=0.13,
                                zorder=2, linewidth=0)
                ax.plot(xs4, lo, color=color, lw=0.7, ls="--", alpha=0.4, zorder=3)
                ax.plot(xs4, hi, color=color, lw=0.7, ls="--", alpha=0.4, zorder=3)
                ax.plot(xs4, mu, label=label, color=color, marker=marker,
                        lw=1.9, markersize=7, zorder=4,
                        markerfacecolor="white" if not transparent else fc,
                        markeredgewidth=1.6)
            ax.set_xticks(xs4); ax.set_xticklabels(xlbls4)
            ax.set_ylabel("Candidates per entity")
            ax.set_xlabel("BFS hop depth")
            ax.set_title("Pipeline Candidate Counts by Hop — Mean \u00b1 1\u03c3 per Entity\n"
                         "(processed entities only)")
            ax.set_ylim(bottom=-0.5)
            ax.yaxis.grid(True, linestyle="--", linewidth=0.5, color="#e0e0e0", alpha=0.8)
            ax.xaxis.grid(False)
            _pub_style(ax, transparent=transparent)
            _inset_table(ax, hops, count_series, fmt=".1f", unit="",
                         transparent=transparent)
            handles, labels = ax.get_legend_handles_labels()
            leg = fig.legend(handles, labels, loc="lower center",
                             ncol=len(count_series), title="Pipeline stage",
                             bbox_to_anchor=(0.5, -0.01),
                             framealpha=0.9, edgecolor="#cccccc",
                             fontsize=9, title_fontsize=9)
            if transparent:
                leg.get_frame().set_facecolor((1, 1, 1, 0.82))
            fig.tight_layout(rect=[0, 0.09, 1, 1])
            return fig

        p4a_base = os.path.join(figures_dir, "04a_hop_mean_std")
        written.extend(_savefig_variants(_build_04a, p4a_base))

        def _build_04b(transparent=False):
            fc = "none" if transparent else "white"
            fig, ax = plt.subplots(figsize=(9, 5), facecolor=fc)
            w = 0.18
            for i, (label, key, color, _m) in enumerate(count_series):
                totals = [max(r.get(key, 0), 1) for r in hop_rows if r["hop"] in hops]
                ax.bar(xs4 + (i - 1.5) * w, totals, width=w,
                       color=color, label=label, alpha=0.85,
                       edgecolor="white", linewidth=0.5, zorder=3)
            ax.set_xticks(xs4); ax.set_xticklabels(xlbls4)
            ax.set_ylabel("Total candidates (log scale)")
            ax.set_xlabel("BFS hop depth")
            ax.set_title("Per-Hop Total Candidate Counts — Log Scale")
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _: f"{int(v):,}" if v >= 1 else ""))
            _pub_style(ax, transparent=transparent)
            ax.grid(axis="y", which="both", linestyle="--", lw=0.5, color="#e0e0e0")
            handles, labels = ax.get_legend_handles_labels()
            leg = fig.legend(handles, labels, loc="lower center",
                             ncol=len(count_series), title="Pipeline stage",
                             bbox_to_anchor=(0.5, -0.01),
                             framealpha=0.9, edgecolor="#cccccc",
                             fontsize=9, title_fontsize=9)
            if transparent:
                leg.get_frame().set_facecolor((1, 1, 1, 0.82))
            fig.tight_layout(rect=[0, 0.09, 1, 1])
            return fig

        p4b_base = os.path.join(figures_dir, "04b_hop_totals")
        written.extend(_savefig_variants(_build_04b, p4b_base))

    # ────────────────────────────────────────────────────────────
    # Figure 05a — Survival rate mean lines
    # Figure 05b — Same + ±1σ band
    # ────────────────────────────────────────────────────────────
    if hop_rows and entity_rows:
        hop_keys = sorted(h for h in _hop_ent.keys() if _hop_ent[h].get("canon_surv"))
        if not hop_keys:
            hop_keys = sorted(_hop_ent.keys())
        xs5      = np.arange(len(hop_keys))
        xlbls5   = [f"Hop {h}" for h in hop_keys]

        surv_series = [
            ("Canonical survival",  "canonical_survival",           C_BLUE,   "o"),
            ("NER survival",      "ner_survival",                 C_GREEN,  "s"),
            ("Sim. survival",     "similarity_survival",          C_ORANGE, "^"),
            ("Overall",           "overall_survival_to_inserted", C_PURPLE, "D"),
        ]

        def _draw_surv(ax, band, transparent=False):
            for label, key, color, marker in surv_series:
                mu, sig = _mu_sd(hop_keys, key, scale=100.0)
                ax.plot(xs5, mu, label=label, color=color, marker=marker,
                        lw=1.9, markersize=7, zorder=4,
                        markerfacecolor="white" if not transparent else "none",
                        markeredgewidth=1.6)
                if band:
                    lo = np.clip(mu - sig, 0, 100)
                    hi = np.clip(mu + sig, 0, 100)
                    ax.fill_between(xs5, lo, hi, color=color,
                                    alpha=0.13, zorder=2, linewidth=0)
                    ax.plot(xs5, lo, color=color, lw=0.7, ls="--", alpha=0.45, zorder=3)
                    ax.plot(xs5, hi, color=color, lw=0.7, ls="--", alpha=0.45, zorder=3)
            ax.set_xticks(xs5); ax.set_xticklabels(xlbls5)
            ax.set_ylabel("Survival rate (%)")
            ax.set_xlabel("BFS hop depth")
            ax.set_ylim(-3, 110)
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _: f"{v:.0f}%"))
            _pub_style(ax, transparent=transparent)

        def _build_05a(transparent=False):
            fc = "none" if transparent else "white"
            fig, ax = plt.subplots(figsize=(9, 5), facecolor=fc)
            _draw_surv(ax, band=False, transparent=transparent)
            ax.set_title("Stage Survival Rates by Hop\n(mean across processed entities)")
            handles, labels = ax.get_legend_handles_labels()
            leg = fig.legend(handles, labels, loc="lower center",
                             ncol=len(surv_series), title="Pipeline stage",
                             bbox_to_anchor=(0.5, -0.01),
                             framealpha=0.9, edgecolor="#cccccc",
                             fontsize=9, title_fontsize=9)
            if transparent:
                leg.get_frame().set_facecolor((1, 1, 1, 0.82))
            fig.tight_layout(rect=[0, 0.09, 1, 1])
            return fig

        def _build_05b(transparent=False):
            fc = "none" if transparent else "white"
            fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=fc)
            _draw_surv(ax, band=True, transparent=transparent)
            ax.set_title("Stage Survival Rates by Hop \u2014 Mean \u00b1 1\u03c3\n"
                         "(processed entities only)")
            _inset_table(ax, hop_keys, surv_series, scale=100.0, fmt=".1f", unit="%",
                         transparent=transparent)
            handles, labels = ax.get_legend_handles_labels()
            leg = fig.legend(handles, labels, loc="lower center",
                             ncol=len(surv_series), title="Pipeline stage",
                             bbox_to_anchor=(0.5, -0.01),
                             framealpha=0.9, edgecolor="#cccccc",
                             fontsize=9, title_fontsize=9)
            if transparent:
                leg.get_frame().set_facecolor((1, 1, 1, 0.82))
            fig.tight_layout(rect=[0, 0.09, 1, 1])
            return fig

        written.extend(_savefig_variants(_build_05a, os.path.join(figures_dir, "05a_survival_lines")))
        written.extend(_savefig_variants(_build_05b, os.path.join(figures_dir, "05b_survival_std_band")))

    # ────────────────────────────────────────────────────────────
    # Figure 06a — Stacked-area: absolute candidate fate per hop
    # Figure 06b — Stacked-area: 100% fate per hop
    # ────────────────────────────────────────────────────────────
    if hop_rows:
        hops6 = [r["hop"] for r in hop_rows]
        xs6   = np.arange(len(hops6))
        xlbls6 = [f"Hop {h}" for h in hops6]

        def _arr(key): return np.array([r.get(key, 0) for r in hop_rows], dtype=float)

        layers6 = [
            ("Pre-NER dedup rejected",   _arr("canonical_rejected_total"), C_GREY),
            ("NER rejected",             _arr("ner_rejected"),             C_ORANGE),
            ("Sim skipped (indexed)",    _arr("similarity_skipped"),       C_BLUE),
            ("Sim rejected (duplicate)", _arr("similarity_rejected"),      C_RED),
            ("Inserted into queue",      _arr("inserted_children"),        C_GREEN),
        ]

        fig6a, ax6a = plt.subplots(figsize=(9, 5))
        bot = np.zeros(len(hops6))
        for label, vals, color in layers6:
            ax6a.fill_between(xs6, bot, bot + vals,
                              label=f"{label}  ({int(vals.sum()):,})",
                              color=color, alpha=0.78, step="mid")
            ax6a.plot(xs6, bot + vals, color=color, lw=1.0,
                      alpha=0.9, drawstyle="steps-mid")
            bot += vals
        ax6a.set_xticks(xs6); ax6a.set_xticklabels(xlbls6)
        ax6a.set_ylabel("Candidates")
        ax6a.set_title("Candidate Fate per Hop — Absolute Counts\n"
                       "(processed entities only)")
        ax6a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{int(v):,}"))
        ax6a.legend(fontsize=8, loc="upper left")
        _pub_style(ax6a)
        fig6a.tight_layout()
        p = os.path.join(figures_dir, "06a_fate_absolute.png")
        _savefig(fig6a, p); written.append(p)

        fig6b, ax6b = plt.subplots(figsize=(9, 5))
        total6 = sum(v for _, v, _ in layers6)
        total6 = np.where(total6 == 0, 1, total6)
        bot_pct = np.zeros(len(hops6))
        for label, vals, color in layers6:
            pct_vals = vals / total6 * 100
            ax6b.fill_between(xs6, bot_pct, bot_pct + pct_vals,
                              label=label, color=color, alpha=0.78, step="mid")
            ax6b.plot(xs6, bot_pct + pct_vals, color=color, lw=1.0,
                      alpha=0.9, drawstyle="steps-mid")
            for i in range(len(hops6) - 1, -1, -1):
                if pct_vals[i] > 5:
                    ax6b.text(i, bot_pct[i] + pct_vals[i] / 2,
                              f"{pct_vals[i]:.0f}%", ha="center", va="center",
                              fontsize=8, color="white", fontweight="bold")
                    break
            bot_pct += pct_vals
        ax6b.set_xticks(xs6); ax6b.set_xticklabels(xlbls6)
        ax6b.set_ylabel("Share of candidates (%)")
        ax6b.set_title("Candidate Fate per Hop — 100% Stacked\n"
                       "(processed entities only)")
        ax6b.set_ylim(0, 103)
        ax6b.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v:.0f}%"))
        ax6b.legend(fontsize=8, loc="lower right")
        _pub_style(ax6b)
        fig6b.tight_layout()
        p = os.path.join(figures_dir, "06b_fate_pct.png")
        _savefig(fig6b, p); written.append(p)

    # ────────────────────────────────────────────────────────────
    # Figure 07 — Cumulative stacked area across hops
    # ────────────────────────────────────────────────────────────
    if hop_rows:
        hops7 = [r["hop"] for r in hop_rows]
        xs7   = np.arange(len(hops7))

        cum_layers7 = [
            ("Pre-NER dedup rej", np.cumsum([r.get("canonical_rejected_total", 0) for r in hop_rows]), C_GREY),
            ("NER rejected",      np.cumsum([r.get("ner_rejected",             0) for r in hop_rows]), C_ORANGE),
            ("Sim skipped",       np.cumsum([r.get("similarity_skipped",       0) for r in hop_rows]), C_BLUE),
            ("Sim rejected",      np.cumsum([r.get("similarity_rejected",      0) for r in hop_rows]), C_RED),
            ("Inserted",          np.cumsum([r.get("inserted_children",        0) for r in hop_rows]), C_GREEN),
        ]

        fig7, ax7 = plt.subplots(figsize=(9, 5))
        bot7 = np.zeros(len(hops7))
        for label, cum_vals, color in cum_layers7:
            ax7.fill_between(xs7, bot7, bot7 + cum_vals,
                             label=f"{label}  ({int(cum_vals[-1]):,})",
                             color=color, alpha=0.75)
            ax7.plot(xs7, bot7 + cum_vals, color=color, lw=1.2, alpha=0.9)
            bot7 += cum_vals
        ax7.set_xticks(xs7); ax7.set_xticklabels([f"Hop {h}" for h in hops7])
        ax7.set_ylabel("Cumulative candidates")
        ax7.set_title("Cumulative Candidate Fate across Hops\n"
                      "(processed entities only)")
        ax7.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{int(v):,}"))
        ax7.legend(fontsize=8, loc="upper left")
        _pub_style(ax7)
        fig7.tight_layout()
        p = os.path.join(figures_dir, "07_cumulative_fate.png")
        _savefig(fig7, p); written.append(p)

    return written


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Read-only funnel analysis for an LLMPedia run directory."
    )
    ap.add_argument("--run-dir", required=True, help="Path to an existing LLMPedia run directory.")
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Default: <run_dir>/funnel_analysis/ (subfolder inside run dir).",
    )
    ap.add_argument(
        "--clean",
        action="store_true",
        help="Remove the funnel_analysis output directory and exit.",
    )
    ap.add_argument(
        "--stdout-only",
        action="store_true",
        help="Print report only; do not write output files.",
    )
    ap.add_argument(
        "--short",
        action="store_true",
        help="Print compact one-liner summary (old style) instead of full report.",
    )
    args = ap.parse_args()

    if not _HAS_TQDM:
        print("[tip] Install tqdm for richer progress bars:  pip install tqdm", flush=True)

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"Run directory does not exist: {run_dir}")

    out_dir = os.path.abspath(args.out_dir or default_out_dir(run_dir))

    # ── --clean ──────────────────────────────────────────────────
    if args.clean:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
            print(f"Removed: {out_dir}", flush=True)
        else:
            print(f"Nothing to clean (not found): {out_dir}", flush=True)
        return

    # ── Analyse ───────────────────────────────────────────────────
    analyzer = FunnelAnalyzer(run_dir)
    summary = analyzer.analyze()

    report_text = render_console(summary) if args.short else render_full_report(summary)
    print()
    print(report_text)

    if args.stdout_only:
        return

    # ── Write files ───────────────────────────────────────────────
    ensure_dir(out_dir)

    if not args.short:
        with open(os.path.join(out_dir, "full_report.txt"), "w", encoding="utf-8") as f:
            f.write(report_text + "\n")

    summary_slim = {
        "overall_summary": summary["overall_summary"],
        "hop_summary":     summary["hop_summary"],
    }
    with open(os.path.join(out_dir, "funnel_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_slim, f, ensure_ascii=False, indent=2)

    write_csv(os.path.join(out_dir, "hop_funnel.csv"),    summary["hop_summary"])
    write_csv(os.path.join(out_dir, "entity_funnel.csv"), summary["entity_rows"])
    write_csv(os.path.join(out_dir, "queue_tree.csv"),    summary["tree_rows"])

    # ── Figures ───────────────────────────────────────────────────
    figures_dir = os.path.join(out_dir, "figures")
    print("\n[figures] generating...", flush=True)
    fig_paths = render_figures(summary, figures_dir)
    if fig_paths:
        print(f"[figures] {len(fig_paths)} figure(s) written to: {figures_dir}", flush=True)

    with open(os.path.join(out_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(
            "READ-ONLY analysis of the source run directory.\n"
            f"Source run : {run_dir}\n"
            f"Analysis   : {out_dir}\n\n"
            "Files:\n"
            "  full_report.txt       human-readable full funnel report\n"
            "  funnel_summary.json   full structured output\n"
            "  hop_funnel.csv        hop-level rollup\n"
            "  entity_funnel.csv     per-entity funnel metrics\n"
            "  queue_tree.csv        parent-child queue tree\n"
            "  figures/              matplotlib PNGs (if matplotlib installed)\n\n"
            "IMPORTANT: Only NER/similarity-processed entities contribute to\n"
            "funnel counts. Entities with articles but no pipeline processing\n"
            "are reported separately as 'unprocessed'.\n\n"
            "To clean:\n"
            f"  python funnel_analysis.py --run-dir {run_dir} --clean\n\n"
            "Source files read:\n"
            "  queue.json / queue.jsonl\n"
            "  articles_wikitext.jsonl, articles_meta.jsonl\n"
            "  ner_decisions.jsonl, ner_lowconf.jsonl\n"
            "  plural_s_dedup.jsonl\n"
            "  similarity_decisions.jsonl\n"
            "  (reject_similarity.jsonl intentionally ignored)\n"
        )

    print(f"\nWrote analysis to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
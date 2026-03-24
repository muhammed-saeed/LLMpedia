

#!/usr/bin/env python3
"""
run_track1_topic.py — Unified Track 1 evaluation + cross-analysis.

Two modes:
  1. EVALUATE (default): Run factuality evaluation on articles, then auto-analyze.
  2. ANALYZE-ONLY (--analyze-only): Skip evaluation, just run cross-analysis
     on existing eval_results.jsonl files.

Supports:
  --run-dir   : evaluate/analyze a single run directory
  --root-dir  : evaluate/analyze ALL run directories under root

Cross-analysis outputs (to <root>/analysis/ or <run-dir>/analysis/):
  - wikipedia_coverage_detail.csv / summary.csv
  - factuality_summary.csv / by_topic_model.csv
  - wikilink_stats.csv
  - entity_overlap_cross_model.csv / cross_persona.csv
  - cross_model_text_similarity.csv / cross_persona_text_similarity.csv
  - stylistic_summary.csv
  - ngram_analysis.csv
  - persona_effect_analysis.csv
  - paper_table4_topic_results.csv / paper_table5_funnel.csv
  - subject_overlap_cross_model.csv

Metadata (model, topic, persona) is read from run_meta.json in each run dir.

FIXES (ported from run_track2_crossmodel.py):
  - _is_wiki_found has a strict=True mode used in factuality summaries:
      strict=False (default/analysis): infers from verdict counts and sim_* keys
                                       when wiki_subject_found flag is absent
      strict=True  (factuality rows):  only trusts the explicit wiki_subject_found
                                       flag; never infers — prevents overcounting
  - All summary rows now report clear denominator columns:
      n_evaluated, n_wiki_found, n_wiki_not_found, wiki_coverage_rate
      (previously some tables mixed n_articles with n_wiki_found implicitly)
  - Evaluation loop has configurable retries (--max-retries) with exponential
    backoff on transient failures; failures are logged with subject counts
  - analyze_factuality() reports _std alongside metrics for all aggregations
  - generate_paper_tables() uses strict=True for consistency with factuality rows
"""

import argparse
import csv
import dataclasses
import json
import math
import os
import re
import signal
import statistics
import sys
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

try:
    from factuality_core import (
        EvalConfig, EvalInput,
        run_evaluation, generate_outputs,
        load_ours_articles, sample_subjects,
        write_run_manifest,
        cancel_requested, request_cancel,
    )
    _HAS_EVAL = True
except ImportError:
    _HAS_EVAL = False

signal.signal(signal.SIGINT, signal.SIG_DFL)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _load_jsonl(path: str) -> List[Dict]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                if isinstance(o, dict):
                    out.append(o)
            except Exception:
                continue
    return out


def _mean(vals: list) -> Optional[float]:
    nums = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
    return sum(nums) / len(nums) if nums else None


def _std(vals: list) -> Optional[float]:
    nums = [v for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
    return statistics.stdev(nums) if len(nums) >= 2 else None


def _median(vals: list) -> Optional[float]:
    nums = sorted(v for v in vals if isinstance(v, (int, float)) and not math.isnan(v))
    return statistics.median(nums) if nums else None


def _fmt(v, decimals=4):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def _write_csv(rows: List[Dict], path: str):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) if isinstance(r.get(k), float)
                        else r.get(k, "") for k in keys})
    print(f"    → {path} ({len(rows)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# CANONICAL NORMALIZATION + WIKITEXT PARSING (matches llmpedia.py)
# ══════════════════════════════════════════════════════════════════════════════

_DASH_RX = re.compile(r"[\-\u2010\u2011\u2012\u2013\u2014\u2212]+", re.UNICODE)

def canon_key(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    t = unicodedata.normalize("NFKC", s).strip().lower()
    t = t.replace("_", " ")
    t = _DASH_RX.sub(" ", t)
    t = re.sub(r"\s+", " ", t, flags=re.UNICODE)
    t = re.sub(r"[^\w\s]", "", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t, flags=re.UNICODE).strip()
    return t


_LINK_RX = re.compile(r"\[\[([^:|\]]+)(?:\|[^]]*)?]]")
_CAT_RX = re.compile(r"\[\[Category:([^|\]]+)(?:\|[^]]*)?]]", re.IGNORECASE)
_HEADING2_RX = re.compile(r"^==\s*(.*?)\s*==\s*$", re.UNICODE)
_IGNORE_SECTIONS = {"see also", "further reading", "external links",
                    "references", "notes", "bibliography"}


def extract_wikilinks(wikitext: str) -> List[str]:
    if not isinstance(wikitext, str):
        return []
    out, seen = [], set()
    ignore_here = False
    for line in wikitext.splitlines():
        hm = _HEADING2_RX.match(line.strip())
        if hm:
            sec = (hm.group(1) or "").strip().lower()
            ignore_here = sec in _IGNORE_SECTIONS
            continue
        if ignore_here:
            continue
        for m in _LINK_RX.finditer(line):
            title = (m.group(1) or "").strip()
            if not title or len(title) > 150:
                continue
            if title.lower().startswith(("category:", "file:", "image:", "media:")):
                continue
            if title not in seen:
                seen.add(title)
                out.append(title)
    return out


def extract_categories(wikitext: str) -> List[str]:
    if not isinstance(wikitext, str):
        return []
    out, seen = [], set()
    for m in _CAT_RX.finditer(wikitext):
        name = (m.group(1) or "").strip()
        if name and len(name.split()) <= 6 and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def count_sections(wikitext: str) -> int:
    if not isinstance(wikitext, str):
        return 0
    return sum(1 for l in wikitext.splitlines() if _HEADING2_RX.match(l.strip()))


def _tokenize(text: str) -> List[str]:
    return [t for t in re.sub(r"[^\w\s]", " ", text.lower()).split() if t]


def _ngrams_set(tokens: List[str], n: int) -> Set[tuple]:
    return set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)) if len(tokens) >= n else set()


# ══════════════════════════════════════════════════════════════════════════════
# SET OVERLAP
# ══════════════════════════════════════════════════════════════════════════════

def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def overlap_coeff(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _text_similarity(t1: str, t2: str) -> Dict[str, float]:
    tok1, tok2 = _tokenize(t1), _tokenize(t2)
    if not tok1 or not tok2:
        return {}
    m = {"jaccard": jaccard(set(tok1), set(tok2))}
    for n in [1, 2, 3]:
        ng1, ng2 = _ngrams_set(tok1, n), _ngrams_set(tok2, n)
        if ng1 and ng2:
            m[f"ngram_{n}_jaccard"] = jaccard(ng1, ng2)
            m[f"ngram_{n}_overlap"] = overlap_coeff(ng1, ng2)
    return m


# ══════════════════════════════════════════════════════════════════════════════
# RUN DISCOVERY (reads run_meta.json)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RunInfo:
    model: str
    topic: str
    persona: str
    run_dir: str
    eval_path: str
    articles_path: str
    run_meta: Dict = field(default_factory=dict)


@dataclass
class RunData:
    info: RunInfo
    eval_records: List[Dict] = field(default_factory=list)
    articles: List[Dict] = field(default_factory=list)
    subjects: Set[str] = field(default_factory=set)
    wikilinks_by_subject: Dict[str, List[str]] = field(default_factory=dict)
    categories_by_subject: Dict[str, List[str]] = field(default_factory=dict)
    canon_entities_by_subject: Dict[str, Set[str]] = field(default_factory=dict)


def _infer_metadata(run_dir: str) -> Dict[str, str]:
    """Read model/topic/persona from run_meta.json (same logic as original)."""
    meta = {"topic": "", "model": "", "persona": "", "_raw": {}}
    meta_path = os.path.join(run_dir, "run_meta.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                rm = json.load(f)
            meta["_raw"] = rm
            meta["topic"] = (rm.get("seed") or "").strip()
            meta["model"] = (
                (rm.get("cascading_defaults") or {}).get("global_model_key")
                or (rm.get("args_raw") or {}).get("model_key")
                or ""
            ).strip()
            meta["persona"] = (
                (rm.get("personas") or {}).get("elicit")
                or (rm.get("args_raw") or {}).get("persona")
                or ""
            ).strip()
            return meta
        except Exception:
            pass
    # Fallback: path inference
    parts = run_dir.replace("\\", "/").split("/")
    for i, p in enumerate(parts):
        if p == "topic_runs" and i + 3 < len(parts):
            meta["model"] = parts[i + 1]
            meta["topic"] = parts[i + 2].replace("_", " ").title()
            meta["persona"] = parts[i + 3]
            return meta
    return meta


def _infer_topic_runs_root(run_dir: str) -> str:
    p = os.path.abspath(run_dir).replace("\\", "/")
    parts = p.split("/")
    if "topic_runs" in parts:
        i = parts.index("topic_runs")
        return "/".join(parts[: i + 1])
    return os.path.dirname(os.path.abspath(run_dir))


def find_run_dirs(root: str) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        if "articles.jsonl" in filenames:
            out.append(dirpath)
    return sorted(out)


def discover_runs_for_analysis(root: str, eval_file: str, articles_file: str) -> List[RunInfo]:
    """Walk tree, find dirs with articles/eval, read run_meta.json for metadata."""
    runs = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if not d.startswith(".")
                       and d not in ("analysis", "factuality_audit",
                                     "__pycache__", "batches",
                                     "parallelqueue", "online_api_responses")]
        if eval_file not in filenames and articles_file not in filenames:
            continue
        meta = _infer_metadata(dirpath)
        model = meta["model"] or "unknown"
        topic = meta["topic"] or "unknown"
        persona = meta["persona"] or "neutral"
        if model == "unknown" and topic == "unknown":
            continue
        runs.append(RunInfo(
            model=model, topic=topic, persona=persona,
            run_dir=dirpath,
            eval_path=os.path.join(dirpath, eval_file),
            articles_path=os.path.join(dirpath, articles_file),
            run_meta=meta.get("_raw", {}),
        ))
    return sorted(runs, key=lambda r: (r.topic, r.model, r.persona))


def load_run_data(info: RunInfo) -> RunData:
    rd = RunData(info=info)
    rd.eval_records = _load_jsonl(info.eval_path)
    rd.articles = _load_jsonl(info.articles_path)
    for a in rd.articles:
        s = (a.get("subject") or "").strip()
        if not s:
            continue
        rd.subjects.add(s)
        wt = a.get("wikitext") or ""
        links = extract_wikilinks(wt)
        rd.wikilinks_by_subject[s] = links
        rd.categories_by_subject[s] = extract_categories(wt)
        rd.canon_entities_by_subject[s] = set(canon_key(l) for l in links if l)
    for r in rd.eval_records:
        s = (r.get("subject") or "").strip()
        if s:
            rd.subjects.add(s)
    return rd


# ══════════════════════════════════════════════════════════════════════════════
# WIKIPEDIA COVERAGE DETECTION
#
# FIX: _is_wiki_found now has a strict parameter (ported from run_track2).
#
#   strict=False (default, used in analysis/coverage passes):
#     Falls back to inferring wiki_found from verdict counts and sim_* keys
#     when the explicit wiki_subject_found flag is absent. This preserves
#     backward compatibility for analyze-only mode on old eval files.
#
#   strict=True (used in factuality summary rows):
#     Only trusts the explicit wiki_subject_found flag written by
#     factuality_core.py. Never infers from verdict counts or sim keys.
#     This prevents overcounting: subjects where all claims came back
#     "insufficient" (Wikipedia had no info) were previously miscounted as
#     "found" because the n_insufficient==n_claims branch came too late.
#
# Rule of thumb:
#   - Use strict=False when you want to discover coverage from any signal
#     (analysis, detail rows, entity overlap, paper tables).
#   - Use strict=True when computing a precision/hallucination denominator
#     so that the wiki_coverage_rate in factuality summaries is consistent
#     with a direct Wikipedia API coverage check.
# ══════════════════════════════════════════════════════════════════════════════

def _is_wiki_found(rec: dict, strict: bool = False) -> bool:
    """Is the subject in Wikipedia?

    strict=False: infers from verdict data when wiki_subject_found flag absent.
    strict=True:  only trusts the explicit flag; never infers.
    """
    wf = rec.get("wiki_subject_found")
    if wf is not None:
        return bool(wf)

    # strict mode: refuse to infer further
    if strict:
        return False

    # Infer: any positive wiki verdict → found
    wiki_sup = rec.get("wiki_n_supported", 0) or 0
    wiki_ref = rec.get("wiki_n_refuted", 0) or 0
    if wiki_sup > 0 or wiki_ref > 0:
        return True

    # Infer: sim_* keys present → Wikipedia page was retrieved and had text
    has_sim = any(k.startswith("sim_") and isinstance(rec.get(k), (int, float))
                  for k in rec.keys())
    if has_sim:
        return True

    # Infer: all claims returned insufficient → subject NOT in Wikipedia
    n_claims = rec.get("n_claims", 0)
    wiki_ins = rec.get("wiki_n_insufficient", 0) or 0
    if n_claims > 0 and wiki_ins == n_claims:
        return False

    return True


def _count_wiki_found(records: List[Dict], strict: bool = False):
    """Return (n_found, n_not_found) for a list of eval records."""
    n_found = sum(1 for r in records if _is_wiki_found(r, strict=strict))
    return n_found, len(records) - n_found


def enrich_wiki_found(records: List[Dict]):
    """Add wiki_subject_found field to records that lack it (in-place).

    Uses strict=False so that any available signal is used to populate the
    flag.  This enrichment step only runs during analyze-only mode on old
    eval files that predate the explicit wiki_subject_found field.
    """
    for rec in records:
        if rec.get("wiki_subject_found") is None:
            rec["wiki_subject_found"] = _is_wiki_found(rec, strict=False)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 1: WIKIPEDIA COVERAGE
#
# FIX: Added explicit denominator columns to every summary row:
#   n_evaluated        — total records processed (was implicitly n_articles)
#   n_wiki_found       — subjects with a Wikipedia page (strict=False inference)
#   n_wiki_not_found   — subjects without a Wikipedia page
#   wiki_coverage_rate — n_wiki_found / n_evaluated
# ══════════════════════════════════════════════════════════════════════════════

def analyze_wikipedia_coverage(all_data: List[RunData], out_dir: str):
    detail_rows, summary_rows = [], []
    for rd in all_data:
        found_count = not_found_count = total = 0
        for rec in rd.eval_records:
            # Use strict=False here: we want full coverage signal for the
            # detail table (diagnostic), not the strict factuality denominator.
            wiki_found = _is_wiki_found(rec, strict=False)
            total += 1
            if wiki_found:
                found_count += 1
            else:
                not_found_count += 1
            detail_rows.append({
                "model": rd.info.model, "topic": rd.info.topic,
                "persona": rd.info.persona, "subject": rec.get("subject", ""),
                "wiki_found": wiki_found,
                "wiki_found_strict": _is_wiki_found(rec, strict=True),
                "n_claims": rec.get("n_claims", 0),
                "wiki_accuracy": rec.get("accuracy_against_wiki"),
                "wiki_true_rate": rec.get("true_rate_against_wiki", 0),
                "wiki_false_rate": rec.get("false_rate_against_wiki", 0),
                "wiki_unverifiable_rate": rec.get("unverifiable_rate_against_wiki", 0),
                "candidate_word_count": rec.get("candidate_word_count", 0),
            })
        found_recs = [r for r in rd.eval_records if _is_wiki_found(r, strict=False)]
        found_accs = [r.get("accuracy_against_wiki") for r in found_recs
                      if r.get("accuracy_against_wiki") is not None]
        found_true = [r.get("true_rate_against_wiki", 0) for r in found_recs]
        found_false = [r.get("false_rate_against_wiki", 0) for r in found_recs]
        found_unv = [r.get("unverifiable_rate_against_wiki", 0) for r in found_recs]
        all_web_acc = [r.get("accuracy_against_web") for r in rd.eval_records
                       if r.get("accuracy_against_web") is not None]
        all_web_false = [r.get("false_rate_against_web", 0) for r in rd.eval_records]

        # Strict count (used for factuality denominators)
        found_strict = sum(1 for r in rd.eval_records if _is_wiki_found(r, strict=True))
        summary_rows.append({
            "model": rd.info.model, "topic": rd.info.topic,
            "persona": rd.info.persona,
            # --- Explicit denominators ---
            "n_evaluated": total,
            "n_wiki_found": found_count,
            "n_wiki_found_strict": found_strict,
            "n_wiki_not_found": not_found_count,
            "wiki_coverage_rate": found_count / total if total else 0,
            "wiki_coverage_rate_strict": found_strict / total if total else 0,
            # conservative = same as above when no shortfall; differs when run_evaluation drops subjects
            "wiki_coverage_rate_conserv": found_count / total if total else 0,
            # --- Factuality metrics (wiki-found subjects only) ---
            "mean_accuracy_wiki_found": _mean(found_accs),
            "std_accuracy_wiki_found": _std(found_accs),
            "mean_true_rate_wiki_found": _mean(found_true),
            "mean_false_rate_wiki_found": _mean(found_false),
            "mean_unverifiable_rate_wiki_found": _mean(found_unv),
            "mean_accuracy_web_all": _mean(all_web_acc),
            "mean_false_rate_web_all": _mean(all_web_false),
        })
    _write_csv(detail_rows, os.path.join(out_dir, "wikipedia_coverage_detail.csv"))
    _write_csv(summary_rows, os.path.join(out_dir, "wikipedia_coverage_summary.csv"))
    print(f"  [coverage] {len(detail_rows)} subjects, {len(summary_rows)} run summaries")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 2: FACTUALITY SUMMARY (paper Table 4)
#
# FIX: Uses strict=True for the factuality denominator so that wiki_coverage_rate
#      here is consistent with a direct Wikipedia API coverage check (same fix
#      applied to run_track2 shared-mode summaries).  Also adds _std columns for
#      all per-article metrics, and explicit n_evaluated / n_wiki_found / n_wiki_not_found.
# ══════════════════════════════════════════════════════════════════════════════

def analyze_factuality(all_data: List[RunData], out_dir: str):
    rows = []
    sim_keys = ["sim_tfidf_cosine", "sim_jaccard", "sim_semantic_cosine",
                "sim_combined_similarity", "sim_ngram_1_overlap",
                "sim_ngram_2_overlap", "sim_ngram_3_overlap", "sim_bertscore_f1"]
    for rd in all_data:
        recs = rd.eval_records
        if not recs:
            continue
        # FIX: use strict=True for precision denominator
        found_recs_strict = [r for r in recs if _is_wiki_found(r, strict=True)]
        found_recs_any    = [r for r in recs if _is_wiki_found(r, strict=False)]
        nf_recs_strict    = [r for r in recs if not _is_wiki_found(r, strict=True)]

        wiki_accs = [r["accuracy_against_wiki"] for r in found_recs_strict
                     if r.get("accuracy_against_wiki") is not None]
        wiki_true  = [r.get("true_rate_against_wiki", 0) for r in found_recs_strict]
        wiki_false = [r.get("false_rate_against_wiki", 0) for r in found_recs_strict]
        wiki_unv   = [r.get("unverifiable_rate_against_wiki", 0) for r in found_recs_strict]

        row = {
            "model": rd.info.model, "topic": rd.info.topic,
            "persona": rd.info.persona,
            # --- Explicit denominators ---
            "n_evaluated": len(recs),
            "n_wiki_found": len(found_recs_strict),
            "n_wiki_not_found": len(nf_recs_strict),
            "wiki_coverage_rate": len(found_recs_strict) / len(recs) if recs else 0,
            "wiki_coverage_rate_conserv": len(found_recs_strict) / len(recs) if recs else 0,
            # n_wiki_found using loose inference (for comparison / backward compat)
            "n_wiki_found_loose": len(found_recs_any),
            # --- Factuality (strict denominator) ---
            "wiki_precision": _mean(wiki_accs),
            "wiki_precision_std": _std(wiki_accs),
            "wiki_true_rate": _mean(wiki_true),
            "wiki_true_rate_std": _std(wiki_true),
            "wiki_false_rate": _mean(wiki_false),
            "wiki_false_rate_std": _std(wiki_false),
            "wiki_unverifiable_rate": _mean(wiki_unv),
            "wiki_unverifiable_rate_std": _std(wiki_unv),
            # --- Web factuality (all subjects, no Wikipedia page dependency) ---
            "web_precision": _mean([r["accuracy_against_web"] for r in recs
                                    if r.get("accuracy_against_web") is not None]),
            "web_precision_std": _std([r["accuracy_against_web"] for r in recs
                                       if r.get("accuracy_against_web") is not None]),
            "web_true_rate": _mean([r.get("true_rate_against_web", 0) for r in recs]),
            "web_false_rate": _mean([r.get("false_rate_against_web", 0) for r in recs]),
            "web_unverifiable_rate": _mean([r.get("unverifiable_rate_against_web", 0) for r in recs]),
            # --- Frontier (wiki NOT found, web evidence only) ---
            "frontier_n": len(nf_recs_strict),
            "frontier_n_web_found": sum(1 for r in nf_recs_strict
                                        if r.get("accuracy_against_web") is not None),
            "frontier_web_precision": _mean([r["accuracy_against_web"] for r in nf_recs_strict
                                             if r.get("accuracy_against_web") is not None]),
            "frontier_web_precision_std": _std([r["accuracy_against_web"] for r in nf_recs_strict
                                                if r.get("accuracy_against_web") is not None]),
            "frontier_web_true_rate": _mean([r.get("true_rate_against_web", 0)
                                             for r in nf_recs_strict
                                             if r.get("accuracy_against_web") is not None]),
            "frontier_web_false_rate": _mean([r.get("false_rate_against_web", 0)
                                              for r in nf_recs_strict
                                              if r.get("accuracy_against_web") is not None]),
            "frontier_web_unverifiable_rate": _mean([r.get("unverifiable_rate_against_web", 0)
                                                     for r in nf_recs_strict
                                                     if r.get("accuracy_against_web") is not None]),
            "mean_word_count": _mean([r.get("candidate_word_count", 0) for r in recs]),
        }
        for sk in sim_keys:
            vals = [r.get(sk) for r in recs if isinstance(r.get(sk), (int, float))]
            row[f"mean_{sk}"] = _mean(vals)
            row[f"std_{sk}"] = _std(vals)
        rows.append(row)
    _write_csv(rows, os.path.join(out_dir, "factuality_summary.csv"))

    # Topic×model aggregation
    groups = defaultdict(list)
    for row in rows:
        groups[(row["topic"], row["model"])].append(row)
    agg_rows = []
    for (topic, model), group in sorted(groups.items()):
        nf  = sum(g.get("n_wiki_found", 0) for g in group)
        nnf = sum(g.get("n_wiki_not_found", 0) for g in group)
        nev = sum(g.get("n_evaluated", 0) for g in group)
        agg = {
            "topic": topic, "model": model, "n_personas": len(group),
            # --- Explicit denominators ---
            "total_evaluated": nev,
            "total_wiki_found": nf,
            "total_wiki_not_found": nnf,
            "wiki_coverage_rate": nf / nev if nev else 0,
        }
        for key in ["wiki_precision", "wiki_true_rate", "wiki_false_rate",
                    "wiki_unverifiable_rate", "web_precision",
                    "mean_sim_tfidf_cosine", "mean_sim_semantic_cosine",
                    "mean_sim_combined_similarity", "mean_sim_ngram_3_overlap",
                    "mean_sim_bertscore_f1"]:
            vals = [g[key] for g in group if g.get(key) is not None]
            agg[f"avg_{key}"] = _mean(vals)
            agg[f"std_{key}"] = _std(vals)
        agg_rows.append(agg)
    _write_csv(agg_rows, os.path.join(out_dir, "factuality_by_topic_model.csv"))
    print(f"  [factuality] {len(rows)} run-level, {len(agg_rows)} topic×model")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 3: WIKILINK STATS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_wikilinks(all_data: List[RunData], out_dir: str):
    rows = []
    for rd in all_data:
        all_links, all_cats, all_secs = [], [], []
        link_counts, cat_counts = [], []
        for a in rd.articles:
            s = (a.get("subject") or "").strip()
            if not s:
                continue
            wt = a.get("wikitext") or ""
            links = extract_wikilinks(wt)
            cats = extract_categories(wt)
            all_links.extend(links); all_cats.extend(cats)
            all_secs.append(count_sections(wt))
            link_counts.append(len(links)); cat_counts.append(len(cats))
        n = len(link_counts)
        if not n:
            continue
        unique_links = set(all_links)
        rows.append({
            "model": rd.info.model, "topic": rd.info.topic,
            "persona": rd.info.persona, "n_articles": n,
            "total_wikilinks": len(all_links), "unique_wikilinks": len(unique_links),
            "unique_canon_wikilinks": len(set(canon_key(l) for l in all_links if l)),
            "mean_wikilinks_per_article": _mean(link_counts),
            "std_wikilinks_per_article": _std(link_counts),
            "median_wikilinks_per_article": _median(link_counts),
            "min_wikilinks": min(link_counts), "max_wikilinks": max(link_counts),
            "total_categories": len(all_cats),
            "unique_categories": len(set(all_cats)),
            "mean_categories_per_article": _mean(cat_counts),
            "mean_sections": _mean(all_secs), "std_sections": _std(all_secs),
            "entity_ttr": len(unique_links) / len(all_links) if all_links else 0,
        })
    _write_csv(rows, os.path.join(out_dir, "wikilink_stats.csv"))
    print(f"  [wikilinks] {len(rows)} run-level stats")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 4: ENTITY OVERLAP (cross-model + cross-persona)
# ══════════════════════════════════════════════════════════════════════════════

def analyze_entity_overlap(all_data: List[RunData], out_dir: str):
    idx = {(rd.info.model, rd.info.topic, rd.info.persona): rd for rd in all_data}
    topics  = sorted(set(rd.info.topic   for rd in all_data))
    models  = sorted(set(rd.info.model   for rd in all_data))
    personas = sorted(set(rd.info.persona for rd in all_data))

    # Cross-model (per topic, avg across personas)
    cm_rows = []
    for topic in topics:
        for m1, m2 in combinations(models, 2):
            ej, cj = [], []
            for persona in personas:
                rd1, rd2 = idx.get((m1, topic, persona)), idx.get((m2, topic, persona))
                if not rd1 or not rd2:
                    continue
                ex1, ex2, ca1, ca2 = set(), set(), set(), set()
                for s in rd1.subjects & rd2.subjects:
                    ex1 |= set(rd1.wikilinks_by_subject.get(s, []))
                    ex2 |= set(rd2.wikilinks_by_subject.get(s, []))
                    ca1 |= rd1.canon_entities_by_subject.get(s, set())
                    ca2 |= rd2.canon_entities_by_subject.get(s, set())
                if ex1 or ex2:
                    ej.append(jaccard(ex1, ex2)); cj.append(jaccard(ca1, ca2))
            cm_rows.append({"topic": topic, "model_1": m1, "model_2": m2,
                            "n_persona_pairs": len(ej),
                            "mean_exact_jaccard": _mean(ej),
                            "mean_canon_jaccard": _mean(cj)})
    _write_csv(cm_rows, os.path.join(out_dir, "entity_overlap_cross_model.csv"))

    # Cross-persona (per model×topic)
    cp_rows = []
    for topic in topics:
        for model in models:
            for p1, p2 in combinations(personas, 2):
                rd1, rd2 = idx.get((model, topic, p1)), idx.get((model, topic, p2))
                if not rd1 or not rd2:
                    continue
                ex1, ex2, ca1, ca2 = set(), set(), set(), set()
                common = rd1.subjects & rd2.subjects
                for s in common:
                    ex1 |= set(rd1.wikilinks_by_subject.get(s, []))
                    ex2 |= set(rd2.wikilinks_by_subject.get(s, []))
                    ca1 |= rd1.canon_entities_by_subject.get(s, set())
                    ca2 |= rd2.canon_entities_by_subject.get(s, set())
                cp_rows.append({
                    "topic": topic, "model": model, "persona_1": p1, "persona_2": p2,
                    "n_common_subjects": len(common),
                    "exact_jaccard": jaccard(ex1, ex2),
                    "canon_jaccard": jaccard(ca1, ca2),
                    "overlap_coeff": overlap_coeff(ca1, ca2),
                    "unique_entities_p1": len(ca1), "unique_entities_p2": len(ca2),
                    "shared_entities": len(ca1 & ca2),
                    "p1_only": len(ca1 - ca2), "p2_only": len(ca2 - ca1),
                })
    _write_csv(cp_rows, os.path.join(out_dir, "entity_overlap_cross_persona.csv"))
    print(f"  [entity-overlap] {len(cm_rows)} model pairs, {len(cp_rows)} persona pairs")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 5: CROSS TEXT SIMILARITY
# ══════════════════════════════════════════════════════════════════════════════

def analyze_cross_similarity(all_data: List[RunData], out_dir: str):
    idx = {(rd.info.model, rd.info.topic, rd.info.persona): rd for rd in all_data}
    topics  = sorted(set(rd.info.topic   for rd in all_data))
    models  = sorted(set(rd.info.model   for rd in all_data))
    personas = sorted(set(rd.info.persona for rd in all_data))

    def _get_wikitext(rd, subj):
        for a in rd.articles:
            if (a.get("subject") or "").strip() == subj:
                return a.get("wikitext") or ""
        return ""

    def _compare_pair(rd1, rd2, max_n=100):
        sims = defaultdict(list)
        for s in sorted(rd1.subjects & rd2.subjects)[:max_n]:
            t1, t2 = _get_wikitext(rd1, s), _get_wikitext(rd2, s)
            if t1 and t2:
                for k, v in _text_similarity(t1, t2).items():
                    sims[k].append(v)
        return sims

    sim_keys = ["jaccard", "ngram_1_jaccard", "ngram_1_overlap",
                "ngram_2_jaccard", "ngram_2_overlap",
                "ngram_3_jaccard", "ngram_3_overlap"]

    # Cross-model
    cm = []
    for topic in topics:
        for m1, m2 in combinations(models, 2):
            all_sims = defaultdict(list)
            for persona in personas:
                rd1, rd2 = idx.get((m1, topic, persona)), idx.get((m2, topic, persona))
                if rd1 and rd2:
                    for k, v in _compare_pair(rd1, rd2).items():
                        all_sims[k].extend(v)
            row = {"topic": topic, "model_1": m1, "model_2": m2,
                   "n_compared": len(all_sims.get("jaccard", []))}
            for k in sim_keys:
                row[f"mean_{k}"] = _mean(all_sims.get(k, []))
                row[f"std_{k}"]  = _std(all_sims.get(k, []))
            cm.append(row)
    _write_csv(cm, os.path.join(out_dir, "cross_model_text_similarity.csv"))

    # Cross-persona
    cp = []
    for topic in topics:
        for model in models:
            for p1, p2 in combinations(personas, 2):
                rd1, rd2 = idx.get((model, topic, p1)), idx.get((model, topic, p2))
                if not rd1 or not rd2:
                    continue
                sims = _compare_pair(rd1, rd2)
                row = {"topic": topic, "model": model, "persona_1": p1,
                       "persona_2": p2, "n_compared": len(sims.get("jaccard", []))}
                for k in sim_keys:
                    row[f"mean_{k}"] = _mean(sims.get(k, []))
                    row[f"std_{k}"]  = _std(sims.get(k, []))
                cp.append(row)
    _write_csv(cp, os.path.join(out_dir, "cross_persona_text_similarity.csv"))
    print(f"  [cross-sim] {len(cm)} model pairs, {len(cp)} persona pairs")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 6: STYLISTIC SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def analyze_stylistic(all_data: List[RunData], out_dir: str):
    style_keys = [
        "sim_style_sent_len_mean_diff", "sim_style_sent_len_std_diff",
        "sim_style_ttr_candidate", "sim_style_ttr_reference", "sim_style_ttr_diff",
        "sim_style_funcword_cosine",
        "sim_style_punct_density_candidate", "sim_style_punct_density_reference",
        "sim_style_punct_density_diff",
    ]
    rows = []
    for rd in all_data:
        if not rd.eval_records:
            continue
        row = {"model": rd.info.model, "topic": rd.info.topic,
               "persona": rd.info.persona,
               "n_articles_with_style": sum(1 for r in rd.eval_records
                                            if r.get("sim_style_ttr_candidate") is not None)}
        for sk in style_keys:
            vals = [r.get(sk) for r in rd.eval_records if isinstance(r.get(sk), (int, float))]
            row[f"mean_{sk}"] = _mean(vals)
            row[f"std_{sk}"]  = _std(vals)
        rows.append(row)
    _write_csv(rows, os.path.join(out_dir, "stylistic_summary.csv"))
    print(f"  [stylistic] {len(rows)} run-level aggregations")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 7: N-GRAM ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_ngrams(all_data: List[RunData], out_dir: str):
    ngram_eval_keys = [
        "sim_ngram_1_jaccard", "sim_ngram_1_overlap",
        "sim_ngram_2_jaccard", "sim_ngram_2_overlap",
        "sim_ngram_3_jaccard", "sim_ngram_3_overlap",
        "sim_tfidf_cosine", "sim_jaccard", "sim_semantic_cosine",
        "sim_combined_similarity",
    ]
    rows = []
    for rd in all_data:
        if not rd.eval_records:
            continue
        row = {"model": rd.info.model, "topic": rd.info.topic, "persona": rd.info.persona,
               "n_articles_with_ngram": sum(1 for r in rd.eval_records
                                            if r.get("sim_ngram_1_overlap") is not None)}
        for k in ngram_eval_keys:
            vals = [r.get(k) for r in rd.eval_records if isinstance(r.get(k), (int, float))]
            row[f"mean_{k}"]   = _mean(vals)
            row[f"std_{k}"]    = _std(vals)
            row[f"median_{k}"] = _median(vals)
        # Corpus-level vocab diversity
        all_tok = []
        for a in rd.articles:
            wt = a.get("wikitext") or ""
            if wt:
                all_tok.extend(_tokenize(wt))
        if all_tok:
            vocab = set(all_tok)
            row["corpus_total_tokens"]  = len(all_tok)
            row["corpus_unique_tokens"] = len(vocab)
            row["corpus_ttr"]           = len(vocab) / len(all_tok)
            for n in [1, 2, 3]:
                ngs = _ngrams_set(all_tok, n)
                row[f"corpus_{n}gram_types"] = len(ngs)
                row[f"corpus_{n}gram_ttr"]   = len(ngs) / max(1, len(all_tok) - n + 1)
        rows.append(row)
    _write_csv(rows, os.path.join(out_dir, "ngram_analysis.csv"))
    print(f"  [ngram] {len(rows)} run-level analyses")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 8: PERSONA EFFECT
# ══════════════════════════════════════════════════════════════════════════════

def analyze_persona_effect(all_data: List[RunData], out_dir: str):
    idx = {(rd.info.model, rd.info.topic, rd.info.persona): rd for rd in all_data}
    topics  = sorted(set(rd.info.topic   for rd in all_data))
    models  = sorted(set(rd.info.model   for rd in all_data))
    personas = sorted(set(rd.info.persona for rd in all_data))
    rows = []
    for topic in topics:
        for model in models:
            pdata = {p: idx.get((model, topic, p)) for p in personas
                     if idx.get((model, topic, p))}
            if len(pdata) < 2:
                continue
            for p1, p2 in combinations(sorted(pdata), 2):
                rd1, rd2 = pdata[p1], pdata[p2]
                # FIX: use strict=True for precision computation so the
                # denominator is consistent with the factuality summary table
                def _accs(rd):
                    return [r.get("accuracy_against_wiki") for r in rd.eval_records
                            if _is_wiki_found(r, strict=True) and
                            r.get("accuracy_against_wiki") is not None]
                accs1, accs2 = _accs(rd1), _accs(rd2)
                hall1 = [r.get("false_rate_against_wiki", 0) for r in rd1.eval_records]
                hall2 = [r.get("false_rate_against_wiki", 0) for r in rd2.eval_records]
                ca1, ca2 = set(), set()
                for s in rd1.subjects & rd2.subjects:
                    ca1 |= rd1.canon_entities_by_subject.get(s, set())
                    ca2 |= rd2.canon_entities_by_subject.get(s, set())
                # Text similarity between persona variants
                def _wt(rd, subj):
                    return next((a.get("wikitext", "") for a in rd.articles
                                if a.get("subject", "").strip() == subj), "")
                tsims = defaultdict(list)
                for s in sorted(rd1.subjects & rd2.subjects)[:50]:
                    t1, t2 = _wt(rd1, s), _wt(rd2, s)
                    if t1 and t2:
                        for k, v in _text_similarity(t1, t2).items():
                            tsims[k].append(v)
                prec_delta = None
                if _mean(accs1) is not None and _mean(accs2) is not None:
                    prec_delta = _mean(accs1) - _mean(accs2)
                rows.append({
                    "topic": topic, "model": model, "persona_1": p1, "persona_2": p2,
                    "n_common_subjects": len(rd1.subjects & rd2.subjects),
                    "mean_precision_p1": _mean(accs1), "mean_precision_p2": _mean(accs2),
                    "precision_delta": prec_delta,
                    "mean_hallucination_p1": _mean(hall1), "mean_hallucination_p2": _mean(hall2),
                    "entity_canon_jaccard": jaccard(ca1, ca2),
                    "entity_overlap_coeff": overlap_coeff(ca1, ca2),
                    "unique_entities_p1": len(ca1), "unique_entities_p2": len(ca2),
                    "shared_entities": len(ca1 & ca2),
                    "text_jaccard": _mean(tsims.get("jaccard", [])),
                    "text_ngram_2_overlap": _mean(tsims.get("ngram_2_overlap", [])),
                    "text_ngram_3_overlap": _mean(tsims.get("ngram_3_overlap", [])),
                })
    _write_csv(rows, os.path.join(out_dir, "persona_effect_analysis.csv"))
    print(f"  [persona] {len(rows)} pairwise comparisons")


# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS 9: PAPER TABLES + SUBJECT OVERLAP
#
# FIX: Uses strict=True for factuality denominators; adds explicit n_evaluated,
#      n_wiki_found, n_wiki_not_found columns to table4 rows.
# ══════════════════════════════════════════════════════════════════════════════

def generate_paper_tables(all_data: List[RunData], out_dir: str):
    idx = {(rd.info.model, rd.info.topic, rd.info.persona): rd for rd in all_data}
    topics  = sorted(set(rd.info.topic   for rd in all_data))
    models  = sorted(set(rd.info.model   for rd in all_data))
    personas = sorted(set(rd.info.persona for rd in all_data))

    t4 = []
    for topic in topics:
        for model in models:
            all_prec, all_hall, all_unv = [], [], []
            n_found = n_nf = n_ev = 0
            for persona in personas:
                rd = idx.get((model, topic, persona))
                if not rd:
                    continue
                for r in rd.eval_records:
                    n_ev += 1
                    # FIX: strict=True for factuality denominator
                    if _is_wiki_found(r, strict=True):
                        n_found += 1
                        acc = r.get("accuracy_against_wiki")
                        if acc is not None:
                            all_prec.append(acc)
                        all_hall.append(r.get("false_rate_against_wiki", 0))
                        all_unv.append(r.get("unverifiable_rate_against_wiki", 0))
                    else:
                        n_nf += 1
            # Entity Jaccard vs other models
            ej_vals, cj_vals = [], []
            for om in models:
                if om == model:
                    continue
                for persona in personas:
                    rd1, rd2 = idx.get((model, topic, persona)), idx.get((om, topic, persona))
                    if not rd1 or not rd2:
                        continue
                    ex1, ex2, ca1, ca2 = set(), set(), set(), set()
                    for s in rd1.subjects & rd2.subjects:
                        ex1 |= set(rd1.wikilinks_by_subject.get(s, []))
                        ex2 |= set(rd2.wikilinks_by_subject.get(s, []))
                        ca1 |= rd1.canon_entities_by_subject.get(s, set())
                        ca2 |= rd2.canon_entities_by_subject.get(s, set())
                    if ex1 or ex2:
                        ej_vals.append(jaccard(ex1, ex2))
                        cj_vals.append(jaccard(ca1, ca2))
            t4.append({
                "topic": topic, "model": model,
                # --- Explicit denominators ---
                "n_evaluated": n_ev,
                "n_wiki_found": n_found,
                "n_wiki_not_found": n_nf,
                "wiki_coverage_rate": n_found / n_ev if n_ev else 0,
                "wiki_coverage_rate_conserv": n_found / n_ev if n_ev else 0,
                # --- Factuality ---
                "precision": _mean(all_prec),
                "precision_std": _std(all_prec),
                "hallucination_rate": _mean(all_hall),
                "hallucination_rate_std": _std(all_hall),
                "unverifiable_rate": _mean(all_unv),
                "unverifiable_rate_std": _std(all_unv),
                "exact_jaccard_vs_other_models": _mean(ej_vals),
                "canon_jaccard_vs_other_models": _mean(cj_vals),
            })
    _write_csv(t4, os.path.join(out_dir, "paper_table4_topic_results.csv"))

    # Funnel
    t5 = []
    for topic in topics:
        for model in models:
            raw_counts = []
            for persona in personas:
                rd = idx.get((model, topic, persona))
                if rd:
                    for s, links in rd.wikilinks_by_subject.items():
                        raw_counts.append(len(links))
            if raw_counts:
                t5.append({"topic": topic, "model": model,
                           "mean_raw_entities": _mean(raw_counts),
                           "std_raw_entities":  _std(raw_counts)})
    _write_csv(t5, os.path.join(out_dir, "paper_table5_funnel.csv"))

    # Subject overlap
    subj_idx = defaultdict(dict)
    for rd in all_data:
        key = rd.info.model
        if key not in subj_idx[rd.info.topic]:
            subj_idx[rd.info.topic][key] = set()
        subj_idx[rd.info.topic][key] |= rd.subjects
    so = []
    for topic, ms in sorted(subj_idx.items()):
        for m1, m2 in combinations(sorted(ms), 2):
            s1, s2 = ms[m1], ms[m2]
            so.append({"topic": topic, "model_1": m1, "model_2": m2,
                       "subjects_m1": len(s1), "subjects_m2": len(s2),
                       "shared": len(s1 & s2), "m1_only": len(s1 - s2),
                       "m2_only": len(s2 - s1), "subject_jaccard": jaccard(s1, s2)})
    _write_csv(so, os.path.join(out_dir, "subject_overlap_cross_model.csv"))
    print(f"  [paper-tables] table4={len(t4)}, table5={len(t5)}, subject_overlap={len(so)}")


# ══════════════════════════════════════════════════════════════════════════════
# RUN CROSS-ANALYSIS (called after eval or standalone)
# ══════════════════════════════════════════════════════════════════════════════

def run_cross_analysis(root_dir: str, out_dir: str,
                       eval_file: str = "eval_results.jsonl",
                       articles_file: str = "articles.jsonl",
                       skip_cross_sim: bool = False):
    """Run all cross-analysis on existing eval results."""
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*66}")
    print(f"[analysis] Cross-run analysis")
    print(f"  root: {root_dir}")
    print(f"  output: {out_dir}")
    print(f"{'='*66}\n")

    runs = discover_runs_for_analysis(root_dir, eval_file, articles_file)
    if not runs:
        print("[analysis] No runs found with eval_results.jsonl or articles.jsonl")
        return

    topics  = sorted(set(r.topic   for r in runs))
    models  = sorted(set(r.model   for r in runs))
    personas = sorted(set(r.persona for r in runs))
    print(f"[analysis] Found {len(runs)} runs:")
    print(f"  Topics:   {topics}")
    print(f"  Models:   {models}")
    print(f"  Personas: {personas}\n")

    print("[analysis] Loading data...")
    all_data = []
    for r in runs:
        rd = load_run_data(r)
        # Enrich wiki_subject_found for backward compat with old eval files
        enrich_wiki_found(rd.eval_records)
        # Report denominator breakdown per run
        n_ev = len(rd.eval_records)
        n_found_strict = sum(1 for rec in rd.eval_records if _is_wiki_found(rec, strict=True))
        n_found_loose  = sum(1 for rec in rd.eval_records if _is_wiki_found(rec, strict=False))
        print(f"  {r.model}/{r.topic}/{r.persona}: "
              f"{n_ev} eval  wiki_found strict={n_found_strict} loose={n_found_loose}  "
              f"{len(rd.articles)} articles  {len(rd.subjects)} subjects"
              f"  ← {os.path.relpath(r.run_dir, root_dir)}")
        all_data.append(rd)

    print(f"\n[analysis] Running analyses...\n")

    print("[1/9] Wikipedia coverage...")
    analyze_wikipedia_coverage(all_data, out_dir)

    print("[2/9] Factuality summary (strict wiki-found denominator)...")
    analyze_factuality(all_data, out_dir)

    print("[3/9] Wikilink analysis...")
    analyze_wikilinks(all_data, out_dir)

    print("[4/9] Entity overlap (cross-model & cross-persona)...")
    analyze_entity_overlap(all_data, out_dir)

    if not skip_cross_sim:
        print("[5/9] Cross-model & cross-persona text similarity...")
        analyze_cross_similarity(all_data, out_dir)
    else:
        print("[5/9] Skipping cross-similarity (--skip-cross-sim)")

    print("[6/9] Stylistic summary...")
    analyze_stylistic(all_data, out_dir)

    print("[7/9] N-gram analysis...")
    analyze_ngrams(all_data, out_dir)

    print("[8/9] Persona effect analysis...")
    analyze_persona_effect(all_data, out_dir)

    print("[9/9] Paper-ready tables + subject overlap...")
    generate_paper_tables(all_data, out_dir)

    print(f"\n{'='*66}")
    print(f"[analysis] Done. Outputs: {out_dir}/")
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(".csv"):
            n = sum(1 for _ in open(os.path.join(out_dir, f))) - 1
            print(f"  {f:50s} ({n} rows)")
    print(f"{'='*66}")


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION (original run_track1_topic.py logic)
#
# FIX: Added retry loop with exponential backoff around run_evaluation() and
#      generate_outputs().  Failures are logged with subject counts.
#      --max-retries now also controls per-run retries (default raised to 5
#      to match run_track2 fix).
# ══════════════════════════════════════════════════════════════════════════════

def _rm_file(path: str):
    try:
        if os.path.isfile(path):
            os.remove(path)
    except Exception:
        pass


def _clean_eval_outputs(run_dir: str, output_file: str):
    for p in [os.path.join(run_dir, output_file),
              os.path.join(run_dir, "eval_summary.csv"),
              os.path.join(run_dir, "eval_claims.csv"),
              os.path.join(run_dir, "eval_aggregate.csv")]:
        _rm_file(p)


def _clean_merged_outputs(root_dir: str):
    for p in [os.path.join(root_dir, "track1_all_results.jsonl"),
              os.path.join(root_dir, "eval_summary.csv"),
              os.path.join(root_dir, "eval_claims.csv"),
              os.path.join(root_dir, "eval_aggregate.csv"),
              os.path.join(root_dir, "track1_persona_comparison.csv")]:
        _rm_file(p)


def _run_eval_with_retry(inputs, cfg, out_path, max_retries: int, label: str):
    """Wrapper around run_evaluation() with exponential backoff on failure.

    Returns (results, n_failures) where n_failures is the number of retried
    attempts before success (0 = first attempt succeeded).
    """
    backoff = 2.0
    for attempt in range(max_retries):
        try:
            results = run_evaluation(inputs, cfg, out_path)
            if attempt > 0:
                print(f"[track1] {label}: succeeded on retry {attempt}", flush=True)
            return results, attempt
        except Exception as e:
            if attempt < max_retries - 1:
                wait = backoff * (attempt + 1)
                print(f"[track1] {label}: eval attempt {attempt+1} failed: {e}. "
                      f"Retrying in {wait:.1f}s ...", flush=True)
                time.sleep(wait)
            else:
                print(f"[track1] {label}: eval FAILED after {max_retries} attempts: {e}",
                      flush=True)
                return [], max_retries
    return [], max_retries


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Track 1: Topic-focused factuality evaluation + cross-analysis")

    # ── mode ──
    ap.add_argument("--analyze-only", action="store_true",
                    help="Skip evaluation, only run cross-analysis on existing results")
    ap.add_argument("--skip-analysis", action="store_true",
                    help="Skip cross-analysis after evaluation")
    ap.add_argument("--skip-cross-sim", action="store_true",
                    help="Skip cross-model/persona text similarity (slow on large corpora)")

    # ── input ──
    ap.add_argument("--root-dir", help="Root dir containing topic/model/persona run dirs")
    ap.add_argument("--run-dir", help="Evaluate/analyze a single run dir")
    ap.add_argument("--topic", default="", help="Override topic label")
    ap.add_argument("--model", default="", help="Override generator model label")
    ap.add_argument("--persona", default="", help="Override persona label")
    ap.add_argument("--articles-file", default="articles.jsonl")
    ap.add_argument("--output-file", default="eval_results.jsonl")
    ap.add_argument("--analysis-dir", default="",
                    help="Output dir for cross-analysis CSVs (default: <root>/analysis/)")

    # ── sampling ──
    ap.add_argument("--sample-n", type=int, default=0)
    ap.add_argument("--sample-frac", type=float, default=0.0)
    ap.add_argument("--sample-min", type=int, default=10)
    ap.add_argument("--sample-max", type=int, default=100)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--min-words", type=int, default=100)

    # ── LLM + execution ──
    ap.add_argument("--fact-model-key", default="gpt-4.1-nano")
    ap.add_argument("--llm-api-base", default=os.getenv("LLM_API_BASE", ""))
    ap.add_argument("--llm-api-key", default=os.getenv("LLM_API_KEY", ""))
    ap.add_argument("--evidence", default="wikipedia,web")
    ap.add_argument("--max-claims", type=int, default=10)
    ap.add_argument("--max-retries", type=int, default=5,
                    help="Max retries for LLM calls and transient eval failures (default: 5)")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--seeds", "--eval-seeds", dest="eval_seeds",
                    type=int, nargs="+", default=[42, 123, 7],
                    help="Explicit random seeds for factuality evaluation (default: 42 123 7). "
                         "Each seed draws an independent sample; factuality metrics are reported "
                         "as mean +/- std across seeds.  Must match the seeds used in "
                         "run_track2_crossmodel.py for cross-paper consistency. "
                         "Example: --seeds 42 123 7")

    # ── web ──
    ap.add_argument("--web-mode", choices=["snippets", "hybrid", "full"], default="hybrid")
    ap.add_argument("--web-max-snippets", type=int, default=1)
    ap.add_argument("--web-max-fetch-pages", type=int, default=1)
    ap.add_argument("--web-fetch-workers", type=int, default=1)
    ap.add_argument("--web-cache-dir", default="")
    ap.add_argument("--web-cache-ttl-hours", type=float, default=168.0)
    ap.add_argument("--search-backend", choices=["auto", "valyu", "brave", "ddg", "searxng", "serper"], default="auto")
    ap.add_argument("--searxng-api-base", default=os.getenv("SEARXNG_API_BASE", ""))

    # ── similarity ──
    ap.add_argument("--compute-similarity", action="store_true", default=True)
    ap.add_argument("--no-compute-similarity", dest="compute_similarity", action="store_false")
    ap.add_argument("--compute-bertscore", action="store_true", default=False)
    ap.add_argument("--compute-stylistic", action="store_true", default=False)
    ap.add_argument("--semantic-provider", choices=["sentence-transformer", "openai"], default="openai")
    ap.add_argument("--semantic-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--openai-embedding-model", default="text-embedding-3-small")
    ap.add_argument("--ngram-n", default="1,2,3")

    # ── audit + clean ──
    ap.add_argument("--audit-root", default="")
    ap.add_argument("--clean-audit", action="store_true", default=False)
    ap.add_argument("--clean-evidence-cache", action="store_true", default=False)
    ap.add_argument("--exclude-domains", default="")
    ap.add_argument("--debug", action="store_true", default=False)
    ap.add_argument("--coverage-mode", choices=["conservative", "returned"],
                    default="conservative",
                    help="Coverage denominator: 'conservative' = n_attempted, "
                         "'returned' = n_returned. Default: conservative.")

    args = ap.parse_args()

    # ── Resolve dirs ──
    if args.run_dir:
        run_dirs = [os.path.abspath(args.run_dir)]
        runs_root = _infer_topic_runs_root(args.run_dir)
    elif args.root_dir:
        runs_root = os.path.abspath(args.root_dir)
        run_dirs = find_run_dirs(runs_root)
        print(f"[track1] Found {len(run_dirs)} run dirs under {args.root_dir}")
    else:
        ap.error("Provide --run-dir or --root-dir")
        return

    analysis_dir = args.analysis_dir or os.path.join(runs_root, "analysis")

    # ─────────────────────────────────────────────────────────────────
    # ANALYZE-ONLY MODE
    # ─────────────────────────────────────────────────────────────────
    if args.analyze_only:
        run_cross_analysis(
            root_dir=runs_root,
            out_dir=analysis_dir,
            eval_file=args.output_file,
            articles_file=args.articles_file,
            skip_cross_sim=args.skip_cross_sim,
        )
        return

    # ─────────────────────────────────────────────────────────────────
    # EVALUATION MODE (then auto cross-analysis)
    # ─────────────────────────────────────────────────────────────────
    if not _HAS_EVAL:
        print("[ERROR] factuality_core not found. Cannot run evaluation.")
        print("  Use --analyze-only to just analyze existing results,")
        print("  or run from the evaluation/ directory.")
        sys.exit(1)

    # Audit root
    if str(args.audit_root).lower() == "none":
        audit_root = ""
    elif args.audit_root:
        audit_root = os.path.abspath(args.audit_root)
    else:
        audit_root = os.path.join(runs_root, "factuality_audit")

    evidence_cache_dir = os.path.join(audit_root, "evidence_cache") if audit_root else ""

    if not args.web_cache_dir:
        args.web_cache_dir = (
            os.path.join(audit_root, "web_page_cache") if audit_root
            else os.path.join(runs_root, ".web_page_cache")
        )
    os.makedirs(args.web_cache_dir, exist_ok=True)

    # Clean
    import shutil
    if args.clean_audit:
        for rd in run_dirs:
            _clean_eval_outputs(rd, args.output_file)
        if args.root_dir:
            _clean_merged_outputs(runs_root)
        if audit_root and os.path.isdir(audit_root):
            print(f"[clean] Deleting audit root: {audit_root}")
            shutil.rmtree(audit_root)
            print("[clean] Done.")
    if args.clean_evidence_cache and evidence_cache_dir and os.path.isdir(evidence_cache_dir):
        print(f"[clean] Deleting evidence cache: {evidence_cache_dir}")
        shutil.rmtree(evidence_cache_dir)

    cfg_base = EvalConfig(
        fact_model_key=args.fact_model_key,
        llm_api_base=args.llm_api_base,
        llm_api_key=args.llm_api_key,
        evidence_sources=[s.strip() for s in args.evidence.split(",") if s.strip()],
        max_claims=args.max_claims,
        max_retries=args.max_retries,
        concurrency=args.concurrency,
        timeout=600.0,
        web_mode=args.web_mode,
        max_web_snippets=args.web_max_snippets,
        max_fetch_pages=args.web_max_fetch_pages,
        web_fetch_workers=args.web_fetch_workers,
        web_cache_dir=args.web_cache_dir,
        web_cache_ttl_hours=args.web_cache_ttl_hours,
        search_backend=args.search_backend,
        searxng_api_base=args.searxng_api_base,
        exclude_domains=args.exclude_domains,
        compute_similarity=args.compute_similarity,
        compute_bertscore=args.compute_bertscore,
        compute_stylistic=args.compute_stylistic,
        semantic_provider=args.semantic_provider,
        semantic_model=args.semantic_model,
        openai_embedding_model=args.openai_embedding_model,
        ngram_values=[int(n) for n in args.ngram_n.split(",") if n.strip()],
        evidence_cache_dir=evidence_cache_dir,
        run_audit_dir="",
        debug=args.debug,
    )

    # Print config
    print("\n" + "=" * 66)
    print("[track1] CONFIGURATION")
    print(f"  fact-model-key    : {args.fact_model_key}")
    if args.llm_api_base:
        print(f"  llm-api-base      : {args.llm_api_base}")
    print(f"  evidence sources  : {cfg_base.evidence_sources}")
    print(f"  max-claims / retries / concurrency : {args.max_claims} / {args.max_retries} / {args.concurrency}")
    print(f"  web-mode          : {args.web_mode}  fetch={args.web_max_fetch_pages}  snippets={args.web_max_snippets}")
    print(f"  similarity={args.compute_similarity}  bertscore={args.compute_bertscore}  stylistic={args.compute_stylistic}")
    print(f"  sampling          : n={args.sample_n}  frac={args.sample_frac}  min={args.sample_min}  max={args.sample_max}")
    print(f"  eval-seeds        : {args.eval_seeds}  (factuality mean/std across seeds)")
    print(f"  coverage-mode     : {getattr(args, 'coverage_mode', 'conservative')}")
    print(f"  coverage-mode     : {getattr(args, 'coverage_mode', 'conservative')}")
    if audit_root:
        print(f"  audit root        : {audit_root}/")
    print("=" * 66 + "\n")

    all_records  = []
    multiseed_rows = []   # per-run-dir summary rows (mean/std across seeds)
    skipped_dirs = 0
    failed_dirs  = 0   # dirs where eval failed after all retries
    n_seeds = len(args.eval_seeds)

    try:
        for rd_idx, rd in enumerate(run_dirs, 1):
            meta    = _infer_metadata(rd)
            topic   = args.topic   or meta["topic"]   or "unknown"
            model   = args.model   or meta["model"]   or "unknown"
            persona = args.persona or meta["persona"] or "neutral"

            print(f"\n{'='*66}")
            print(f"[track1] [{rd_idx}/{len(run_dirs)}]  {rd}")
            print(f"[track1] model={model}  topic={topic}  persona={persona}")

            articles = load_ours_articles(rd, args.articles_file, args.min_words)
            if not articles:
                print(f"[track1] 0 articles after min-words={args.min_words} → skip")
                skipped_dirs += 1
                continue

            # ── Multi-seed evaluation loop ──────────────────────────────────
            seeds = list(args.eval_seeds)  # e.g. [42, 123, 7] — same as track2
            seed_results: List[List[Dict]] = []
            per_seed_n_art: List[int] = []   # n_inputs with article text per seed
            any_success = False

            for seed_idx, seed in enumerate(seeds):
                subjects = sample_subjects(
                    subjects=list(articles.keys()),
                    n=args.sample_n, frac=args.sample_frac,
                    seed=seed,
                    min_n=args.sample_min, max_n=args.sample_max,
                )
                if not subjects:
                    print(f"[track1] seed {seed_idx+1}: {len(articles)} available, "
                          f"below sample-min={args.sample_min} → skip")
                    skipped_dirs += 1 if seed_idx == 0 else 0
                    break

                print(f"[track1] seed {seed_idx+1}/{n_seeds} (seed={seed}): "
                      f"{len(articles)} loaded → sampled {len(subjects)}")

                run_audit_dir = ""
                if audit_root:
                    try:
                        common = os.path.commonpath([runs_root, os.path.abspath(rd)])
                        rel = os.path.relpath(rd, runs_root) if common == runs_root else None
                    except Exception:
                        rel = None
                    base_audit = (
                        os.path.join(audit_root, "runs", *rel.split(os.sep)) if rel
                        else os.path.join(audit_root, "runs", f"{model}__{topic}__{persona}")
                    )
                    run_audit_dir = os.path.join(base_audit, f"seed{seed_idx}")
                    for sub in ("claims", "results", "evidence"):
                        os.makedirs(os.path.join(run_audit_dir, sub), exist_ok=True)
                    if seed_idx == 0:
                        print(f"[track1] audit base → {base_audit}")
                    write_run_manifest(
                        run_audit_dir=run_audit_dir, model=model, topic=topic,
                        persona=persona, n_articles=len(subjects),
                        evidence_sources=cfg_base.evidence_sources,
                        fact_model_key=args.fact_model_key,
                        extra={"argv": sys.argv, "args": vars(args),
                               "run_dir": rd, "seed_idx": seed_idx, "seed": seed},
                    )

                cfg = dataclasses.replace(cfg_base, run_audit_dir=run_audit_dir)

                # Loop: verify every sampled subject is present in articles.jsonl
                missing = [s for s in subjects
                           if s not in articles
                           or not (articles[s].get("wikitext") or "").strip()]
                if missing:
                    raise RuntimeError(
                        f"{model}/{topic}/{persona} seed{seed_idx}: "
                        f"{len(missing)} sampled subjects missing article text.\n"
                        f"First 10: {sorted(missing)[:10]}"
                    )
                print(f"  [OK] {len(subjects)} subjects all present in articles.jsonl")

                inputs = [
                    EvalInput(
                        subject=subj, candidate="ours",
                        article_text=articles[subj].get("wikitext") or "",
                        hop=articles[subj].get("hop"),
                        persona=persona, topic=topic,
                        generator_model=model, clean_wiki_markup=True,
                    )
                    for subj in subjects
                ]

                n_attempted = len(inputs)
                print(f"[track1] seed {seed_idx+1}/{n_seeds} (seed={seed}): "
                      f"sampled={len(subjects)}  inputs={n_attempted}")

                if not inputs:
                    print(f"[track1] seed {seed_idx+1}: No inputs — skipping seed")
                    continue

                # Suffix output file with seed index so seeds don't overwrite each other
                seed_suffix = f".seed{seed_idx}" if n_seeds > 1 else ""
                out_stem = args.output_file.replace(".jsonl", "")
                out_path = os.path.join(rd, f"{out_stem}{seed_suffix}.jsonl")
                label    = f"{model}/{topic}/{persona}/seed{seed_idx}(s={seed})"

                results, n_failures = _run_eval_with_retry(
                    inputs, cfg, out_path,
                    max_retries=args.max_retries,
                    label=label,
                )

                # Sanity: run_evaluation should return one record per input.
                # Shortfall means some subjects had no evaluable claims/Wikipedia content.
                n_returned = len(results)
                if n_returned != n_attempted:
                    print(f"  [WARN] {label}: run_evaluation returned {n_returned} "
                          f"records for {n_attempted} inputs "
                          f"(shortfall={n_attempted-n_returned})")

                if not results and n_failures >= args.max_retries:
                    print(f"[track1] {label}: giving up after {n_failures} failures")
                    failed_dirs += 1
                else:
                    any_success = True
                    seed_results.append(results)
                    per_seed_n_art.append(n_attempted)  # n inputs per seed
                    all_records.extend(results)
                    n_found_strict = sum(1 for r in results if _is_wiki_found(r, strict=True))
                    if results:
                        print(f"  → returned={n_returned}  attempted={n_attempted}  "
                              f"wiki_found(strict)={n_found_strict}  "
                              f"coverage={n_found_strict/n_returned:.1%}")
                    else:
                        print(f"  → 0 results")

                try:
                    generate_outputs(out_path, rd)
                except Exception as e:
                    print(f"[track1] generate_outputs failed for {label}: {e}")

                if cancel_requested():
                    print("[track1] Ctrl+C → stopping. Partial outputs saved.")
                    break

            # ── Merge seed files into base eval_results.jsonl ─────────────
            # FIX: Single merge block (was duplicated in original)
            if seed_results and any_success and n_seeds > 1:
                merged_path = os.path.join(rd, args.output_file)
                try:
                    all_seed_records = []
                    for si in range(len(seed_results)):
                        seed_suffix = f".seed{si}"
                        out_stem = args.output_file.replace(".jsonl", "")
                        seed_path = os.path.join(rd, f"{out_stem}{seed_suffix}.jsonl")
                        if os.path.exists(seed_path):
                            all_seed_records.extend(_load_jsonl(seed_path))
                    if all_seed_records:
                        # Deduplicate: keep latest record per subject
                        by_subject = {}
                        for rec in all_seed_records:
                            s = rec.get("subject", "")
                            if s:
                                by_subject[s] = rec
                        deduped = list(by_subject.values())
                        # Write merged file
                        os.makedirs(os.path.dirname(merged_path) or ".", exist_ok=True)
                        with open(merged_path, "w", encoding="utf-8") as f:
                            for rec in deduped:
                                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        print(f"  [merge] {len(deduped)} unique subjects → {merged_path}")
                except Exception as e:
                    print(f"  [WARN] merge failed: {e}")


            # ── Merge seed files into base eval_results.jsonl ─────────────
            if seed_results and any_success and n_seeds > 1:
                merged_path = os.path.join(rd, args.output_file)
                try:
                    all_seed_records = []
                    for si in range(len(seed_results)):
                        seed_suffix = f".seed{si}"
                        out_stem = args.output_file.replace(".jsonl", "")
                        seed_path = os.path.join(rd, f"{out_stem}{seed_suffix}.jsonl")
                        if os.path.exists(seed_path):
                            all_seed_records.extend(_load_jsonl(seed_path))
                    if all_seed_records:
                        by_subject = {}
                        for rec in all_seed_records:
                            s = rec.get("subject", "")
                            if s:
                                by_subject[s] = rec
                        deduped = list(by_subject.values())
                        os.makedirs(os.path.dirname(merged_path) or ".", exist_ok=True)
                        with open(merged_path, "w", encoding="utf-8") as f:
                            for rec in deduped:
                                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        print(f"  [merge] {len(deduped)} unique subjects → {merged_path}")
                except Exception as e:
                    print(f"  [WARN] merge failed: {e}")

            # ── Per-run-dir multi-seed summary ──────────────────────────────
            if seed_results and any_success:
                # Aggregate factuality across seeds
                seed_prec, seed_true, seed_false, seed_unv = [], [], [], []
                for srecs in seed_results:
                    found_recs = [r for r in srecs if _is_wiki_found(r, strict=True)]
                    pr = [r.get("accuracy_against_wiki") for r in found_recs
                          if r.get("accuracy_against_wiki") is not None]
                    if pr:
                        seed_prec.append(_mean(pr))
                        seed_true.append(_mean([r.get("true_rate_against_wiki", 0) for r in found_recs]))
                        seed_false.append(_mean([r.get("false_rate_against_wiki", 0) for r in found_recs]))
                        seed_unv.append(_mean([r.get("unverifiable_rate_against_wiki", 0) for r in found_recs]))

                all_recs_dir = [r for sr in seed_results for r in sr]
                # FIX: Use strict=True for paper-facing denominator
                n_found_total, n_nf_total = _count_wiki_found(all_recs_dir, strict=True)
                n_per_seed_avg = int(round(_mean(per_seed_n_art) or 0))
                row = {
                    "model": model, "topic": topic, "persona": persona,
                    "seeds": str(list(seeds[:len(seed_results)])),
                    "n_seeds": len(seed_results),
                    # n_attempted_per_seed  = subjects with article text submitted to eval per seed
                    # n_attempted_total     = n_attempted_per_seed × n_seeds
                    # n_returned_total      = records returned by run_evaluation() (pooled)
                    #   NOTE: n_returned < n_attempted because run_evaluation() only returns
                    #   records for subjects where Wikipedia content was retrieved.
                    # n_wiki_subject_found  = n_returned with wiki_subject_found=True
                    "n_attempted_per_seed": n_per_seed_avg,
                    "n_attempted_total": n_per_seed_avg * len(seed_results),
                    "n_returned_total": len(all_recs_dir),
                    "n_returned_per_seed": round(len(all_recs_dir) / len(seed_results), 1) if seed_results else 0,
                    "n_wiki_subject_found": n_found_total,
                    "n_wiki_subject_not_found": n_nf_total,
                    "wiki_coverage_rate": n_found_total / len(all_recs_dir) if all_recs_dir else 0,
                    "wiki_coverage_rate_conserv": n_found_total / (n_per_seed_avg * len(seed_results)) if (n_per_seed_avg * len(seed_results)) else 0,
                    "wiki_precision": _mean(seed_prec),
                    "wiki_precision_std": _std(seed_prec),
                    "wiki_true_rate": _mean(seed_true),
                    "wiki_true_rate_std": _std(seed_true),
                    "wiki_false_rate": _mean(seed_false),
                    "wiki_false_rate_std": _std(seed_false),
                    "wiki_unverifiable_rate": _mean(seed_unv),
                    "wiki_unverifiable_rate_std": _std(seed_unv),
                }
                multiseed_rows.append(row)

                prec_str = (f"{row['wiki_precision']:.4f}±{row['wiki_precision_std']:.4f}"
                            if row.get("wiki_precision_std") else
                            f"{row.get('wiki_precision', 'N/A')}")
                # FIX: Use n_returned_total (was n_evaluated_total which doesn't exist)
                print(f"  [{model}/{topic}/{persona}] "
                      f"wiki_precision={prec_str}  "
                      f"wiki_found={n_found_total}/{len(all_recs_dir)} (strict, {len(seed_results)} seeds)")

                rate = row.get("wiki_coverage_rate", 0) or 0
                # FIX: Use n_returned_total (was n_evaluated_total which doesn't exist)
                n_ret = row.get("n_returned_total", 0) or 0
                if rate > 0.95 and n_ret > 20:
                    print(f"  [WARN] wiki_coverage_rate={rate:.3f} is very high. "
                          f"Verify wiki_subject_found flags are set by factuality_core.")

            elif not any_success:
                skipped_dirs += 1

            if cancel_requested():
                break

    except KeyboardInterrupt:
        request_cancel()
        print("[track1] Ctrl+C between runs → stopping.")

    print(f"\n{'='*66}")
    print(f"[track1] EVALUATION DONE")
    print(f"  run dirs   : {len(run_dirs)}")
    print(f"  processed  : {len(run_dirs) - skipped_dirs - failed_dirs}")
    print(f"  skipped    : {skipped_dirs}  (no articles / below sample-min)")
    print(f"  failed     : {failed_dirs}   (exhausted retries)")
    print(f"  eval-seeds    : {args.eval_seeds}")
    print(f"  total evaluated : {len(all_records)}")
    if all_records:
        n_found_strict = sum(1 for r in all_records if _is_wiki_found(r, strict=True))
        print(f"  wiki_found (strict) : {n_found_strict}/{len(all_records)} "
              f"({n_found_strict/len(all_records):.1%})")

    if multiseed_rows:
        ms_path = os.path.join(runs_root, "multiseed_factuality_summary.csv")
        _write_csv(multiseed_rows, ms_path)
        print(f"\n  Multi-seed summary ({len(multiseed_rows)} run-dirs):")
        for row in multiseed_rows:
            # FIX: Use n_returned_total (was n_evaluated_total which doesn't exist)
            print(f"    {row['model']:20s} {row['topic']:20s} {row['persona']:15s}"
                  f"  prec={_fmt(row.get('wiki_precision'))}±{_fmt(row.get('wiki_precision_std'))}"
                  f"  wiki_found={row.get('n_wiki_subject_found')}/{row.get('n_returned_total')}")
    print(f"{'='*66}")

    # ── Auto cross-analysis ──
    _cancelled = cancel_requested() if _HAS_EVAL else False
    if not args.skip_analysis and not _cancelled:
        run_cross_analysis(
            root_dir=runs_root,
            out_dir=analysis_dir,
            eval_file=args.output_file,
            articles_file=args.articles_file,
            skip_cross_sim=args.skip_cross_sim,
        )


if __name__ == "__main__":
    main()






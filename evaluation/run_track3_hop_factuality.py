#!/usr/bin/env python3
"""
run_track3_hop_factuality.py — Hop-stratified & random factuality analysis
                                for LLMPedia (ACL 2026).

CHANGES FROM ORIGINAL:
  - Default --evidence-sources=wikipedia (was wikipedia,web — DDG SSL errors)
  - Imports json_repair for robust JSON extraction
  - _is_wiki_found matches patched factuality_core (strict parameter)
  - Dual coverage: wiki_coverage_rate + wiki_coverage_rate_conserv
  - Hop buckets split hop3 vs hop4 vs hop5plus (99% of corpus is hop≥3)

OVERVIEW
────────
  Analyses a SINGLE model run-dir along two sampling dimensions:

  --mode  random      Uniform random from entire corpus. 3 seeds → mean ± std.
  --mode  hop         Per-hop quotas. 3 seeds → per-hop metrics + aggregate.
  --mode  both        Run BOTH modes (default).

HOP BUCKETS (tuned for actual data distribution)
─────────────────────────────────────────────────
  hop0     : seed article (typically 1)
  hop1     : direct neighbours (~36-64)
  hop2     : 2nd shell (~1100-2400)
  hop3     : 3rd shell (tens of thousands)
  hop4     : 4th shell (bulk of corpus)
  hop5plus : everything ≥5 (small, GPT only)

  If a bucket has fewer articles than the quota, ALL are used (std=0).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import random
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

# ── path setup ────────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("[WARN] matplotlib/numpy not installed — figures skipped")

try:
    from factuality_core import (EvalConfig, EvalInput, run_evaluation,
                                 generate_outputs)
    HAS_FACTUALITY = True
except ImportError:
    HAS_FACTUALITY = False
    print("[WARN] factuality_core not available — factuality steps skipped")

# ── json_repair (robust extraction) ──────────────────────────────────────────
try:
    from json_repair import extract_json_robust as _extract_json  # noqa: F401
    print("[OK] json_repair loaded")
except ImportError:
    print("[INFO] json_repair not found — using factuality_core's built-in")

# ── constants ─────────────────────────────────────────────────────────────────
DEFAULT_SEEDS = [42, 123, 7]

# Hop buckets — split hop3 and hop4 because they hold 99% of subjects
HOP_BUCKETS: List[str] = ["hop0", "hop1", "hop2", "hop3", "hop4", "hop5plus"]

ACL_STYLE = {
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
}
HOP_COLORS = {
    "hop0": "#E91E63", "hop1": "#9C27B0", "hop2": "#2196F3",
    "hop3": "#009688", "hop4": "#FF9800", "hop5plus": "#607D8B",
    "random": "#4CAF50", "overall": "#F44336",
}


# ── helpers ───────────────────────────────────────────────────────────────────
def _mean(v):
    n = [x for x in v if isinstance(x, (int, float)) and not math.isnan(x)]
    return sum(n) / len(n) if n else None

def _std(v):
    n = [x for x in v if isinstance(x, (int, float)) and not math.isnan(x)]
    return statistics.stdev(n) if len(n) >= 2 else None

def _median(v):
    n = sorted(x for x in v if isinstance(x, (int, float)) and not math.isnan(x))
    return statistics.median(n) if n else None

def _fmt(v, d=4):
    if v is None: return ""
    if isinstance(v, float): return f"{v:.{d}f}"
    return str(v)

def _write_csv(rows, path):
    if not rows: return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r.keys()))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (_fmt(r.get(k)) if isinstance(r.get(k), float)
                           else r.get(k, "")) for k in keys})
    print(f"  -> {path}  ({len(rows)} rows)")


# ── text helpers ──────────────────────────────────────────────────────────────
_HEADING2_RX = re.compile(r"^==\s*(.*?)\s*==\s*$", re.UNICODE)

def word_count(t: str) -> int:
    return len(re.sub(r"[^\w\s]", " ", t.lower()).split()) if t else 0

def count_sections(wt: str) -> int:
    return sum(1 for l in (wt or "").splitlines() if _HEADING2_RX.match(l.strip()))


# ── data model ────────────────────────────────────────────────────────────────
@dataclass
class SubjectMeta:
    hop: int = 0
    wc: int = 0
    n_sections: int = 0

@dataclass
class ModelData:
    key: str
    dir_path: str
    articles_file: str = "articles.jsonl"
    display_name: str = ""
    run_meta: Dict = field(default_factory=dict)
    subject_names: List[str] = field(default_factory=list)
    subject_meta: Dict[str, SubjectMeta] = field(default_factory=dict)
    subjects_by_hop: Dict[str, List[str]] = field(default_factory=dict)
    n_total_raw: int = 0
    n_total_unique: int = 0


# ── hop bucket mapping ────────────────────────────────────────────────────────
def _hop_bucket_label(hop: int) -> str:
    if hop <= 0: return "hop0"
    if hop == 1: return "hop1"
    if hop == 2: return "hop2"
    if hop == 3: return "hop3"
    if hop == 4: return "hop4"
    return "hop5plus"


# ── loading ───────────────────────────────────────────────────────────────────
def load_model_data(key, dir_path, articles_file="articles.jsonl",
                    min_words=100):
    """Stream entire articles.jsonl (no max-subjects cap)."""
    dir_path = os.path.abspath(dir_path)
    meta_path = os.path.join(dir_path, "run_meta.json")
    run_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            run_meta = json.load(f)

    cas = run_meta.get("cascading_defaults") or {}
    ar = run_meta.get("args_raw") or {}
    em = (ar.get("elicit_model_key") or "").strip()
    gm = (cas.get("global_model_key") or ar.get("model_key") or "").strip()
    dn = (em or gm or key).replace("scads-", "").replace("_", " ")

    md = ModelData(key=key, dir_path=dir_path, articles_file=articles_file,
                   display_name=dn, run_meta=run_meta)
    arts_path = os.path.join(dir_path, articles_file)
    if not os.path.exists(arts_path):
        raise FileNotFoundError(f"[FATAL] {arts_path} not found")

    print(f"[load] {key}: streaming {arts_path} ...")
    t0 = time.perf_counter()
    best: Dict[str, Tuple[int, SubjectMeta]] = {}
    n_raw = 0; n_skip = 0

    with open(arts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            n_raw += 1
            if n_raw % 100_000 == 0:
                print(f"  [{key}] ...{n_raw:,} lines  {len(best):,} kept  "
                      f"({time.perf_counter()-t0:.1f}s)")
            try:
                a = json.loads(line)
            except Exception:
                continue
            if not isinstance(a, dict): continue
            s = (a.get("subject") or "").strip()
            if not s: continue
            wt = a.get("wikitext") or ""
            wc = word_count(wt)
            if min_words > 0 and wc < min_words:
                n_skip += 1; continue
            hop = 0
            try: hop = int(a.get("hop", 0))
            except: pass
            meta = SubjectMeta(hop=hop, wc=wc, n_sections=count_sections(wt))
            if s not in best or hop < best[s][0]:
                best[s] = (hop, meta)

    md.n_total_raw = n_raw
    md.n_total_unique = len(best)
    elapsed = time.perf_counter() - t0
    print(f"  [{key}] {n_raw:,} lines → {len(best):,} unique "
          f"(skipped {n_skip:,} < {min_words} words) in {elapsed:.1f}s")

    md.subject_names = list(best.keys())
    md.subjects_by_hop = {b: [] for b in HOP_BUCKETS}
    for s, (hop, meta) in best.items():
        md.subject_meta[s] = meta
        md.subjects_by_hop[_hop_bucket_label(hop)].append(s)

    del best
    for bkt in HOP_BUCKETS:
        n = len(md.subjects_by_hop[bkt])
        pct = n / len(md.subject_names) * 100 if md.subject_names else 0
        print(f"  [{key}] {bkt}: {n:>8,}  ({pct:5.1f}%)")

    print(f"[load] {key} ({dn}): {len(md.subject_names):,} subjects in {elapsed:.1f}s")
    return md


def read_articles_for_subjects(md, subjects):
    """Re-read wikitext from articles.jsonl (full scan, lowest hop)."""
    needed = set(subjects) & set(md.subject_names)
    if not needed: return {}

    arts_path = os.path.join(md.dir_path, md.articles_file)
    result = {}
    best_hop = {}
    n_read = 0
    t0 = time.perf_counter()

    with open(arts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            n_read += 1
            if n_read % 200_000 == 0:
                print(f"    [{md.key}] ...{n_read:,} lines  "
                      f"found {len(result)}/{len(needed)}")
            try:
                a = json.loads(line)
            except Exception:
                continue
            if not isinstance(a, dict): continue
            s = (a.get("subject") or "").strip()
            if s not in needed: continue
            hop = 0
            try: hop = int(a.get("hop", 0))
            except: pass
            if s not in best_hop or hop < best_hop[s]:
                best_hop[s] = hop
                result[s] = a.get("wikitext") or ""

    print(f"    [{md.key}] re-read: {len(result)}/{len(needed)} "
          f"in {time.perf_counter()-t0:.1f}s ({n_read:,} lines)")
    return result


# ── factuality helpers ────────────────────────────────────────────────────────
def _is_wiki_found(rec: dict, strict: bool = False) -> bool:
    """Match the patched factuality_core version."""
    wf = rec.get("wiki_subject_found")
    if wf is not None: return bool(wf)
    if strict: return False
    if (rec.get("wiki_n_supported", 0) or 0) > 0: return True
    if (rec.get("wiki_n_refuted", 0) or 0) > 0: return True
    if any(k.startswith("sim_") and isinstance(rec.get(k), (int, float))
           for k in rec.keys()):
        return True
    n_claims = rec.get("n_claims", 0)
    wiki_ins = rec.get("wiki_n_insufficient", 0) or 0
    if n_claims > 0 and wiki_ins == n_claims:
        return False
    return True


def _make_cfg(args, audit_dir=""):
    return EvalConfig(
        fact_model_key=args.fact_model_key,
        evidence_sources=[s.strip() for s in args.evidence_sources.split(",")
                          if s.strip()],
        max_claims=args.max_claims,
        max_retries=args.max_retries,
        concurrency=args.concurrency,
        timeout=600.0,
        web_mode=getattr(args, "web_mode", "snippets"),
        search_backend=getattr(args, "search_backend", "auto"),
        web_cache_dir=getattr(args, "web_cache_dir", ""),
        compute_similarity=getattr(args, "compute_similarity", False),
        compute_bertscore=getattr(args, "compute_bertscore", False),
        compute_stylistic=getattr(args, "compute_stylistic", False),
        run_audit_dir=audit_dir,
        debug=getattr(args, "debug", False),
    )


def _run_fact(label, md, subjects, args, audit_dir, texts=None):
    """Run factuality for one batch. Returns list of result dicts."""
    if not HAS_FACTUALITY: return []
    if texts is None:
        texts = read_articles_for_subjects(md, subjects)

    missing = [s for s in subjects if s not in texts or not texts[s].strip()]
    if missing:
        raise RuntimeError(
            f"[{label}] {len(missing)} subjects missing: {sorted(missing)[:5]}")

    cfg = _make_cfg(args, audit_dir)
    # PATCH: pass topic from run_meta for web search disambiguation
    _seed = (md.run_meta.get("seed") or "").strip()
    inputs = [
        EvalInput(
            subject=s, candidate=md.key,
            article_text=texts[s],
            hop=md.subject_meta[s].hop if s in md.subject_meta else 0,
            generator_model=md.display_name,
            topic=_seed,
            clean_wiki_markup=True,
        )
        for s in subjects
    ]
    n_att = len(inputs)
    print(f"  [{label}] {n_att} inputs → run_evaluation() ...")

    out_path = os.path.join(args.output_dir, f"raw_{label}.jsonl")
    results = []
    backoff = 2.0
    max_tries = max(1, args.max_retries)

    for attempt in range(max_tries):
        try:
            results = run_evaluation(inputs, cfg, out_path)
            break
        except Exception as e:
            if attempt < max_tries - 1:
                wait = backoff * (attempt + 1)
                print(f"  [WARN] {label} attempt {attempt+1} failed: {e}  "
                      f"(retry in {wait:.1f}s)")
                time.sleep(wait)
            else:
                print(f"  [ERROR] {label} failed after {max_tries} attempts: {e}")
                results = []

    n_ret = len(results)
    if n_ret != n_att:
        print(f"  [WARN] {label}: returned {n_ret}/{n_att} "
              f"(shortfall={n_att - n_ret})")

    try:
        rd = os.path.join(md.dir_path, "hop_analysis")
        os.makedirs(rd, exist_ok=True)
        generate_outputs(out_path, rd)
    except Exception:
        pass

    return results


def _summarise_records(all_recs, seed_batches, n_per_seed, n_seeds, tag):
    """Aggregate factuality metrics across seeds: wiki + web + frontier."""
    # Wiki accumulators
    seed_prec, seed_true, seed_false, seed_unv = [], [], [], []
    # Web accumulators (ALL subjects)
    seed_web_prec, seed_web_true, seed_web_false, seed_web_unv = [], [], [], []
    # Frontier accumulators (wiki NOT found, web only)
    seed_frontier_prec, seed_frontier_true = [], []
    seed_frontier_false, seed_frontier_unv = [], []

    for batch in seed_batches:
        # ── Wiki slice ────────────────────────────────────────────
        found = [r for r in batch if _is_wiki_found(r, strict=True)]
        pv = [r.get("accuracy_against_wiki") for r in found
              if r.get("accuracy_against_wiki") is not None]
        if pv:
            seed_prec.append(_mean(pv))
            seed_true.append(_mean([r.get("true_rate_against_wiki", 0) or 0
                                    for r in found]))
            seed_false.append(_mean([r.get("false_rate_against_wiki", 0) or 0
                                     for r in found]))
            seed_unv.append(_mean([r.get("unverifiable_rate_against_wiki", 0) or 0
                                   for r in found]))

        # ── Web slice (ALL subjects with web evidence) ────────────
        web_recs = [r for r in batch if r.get("accuracy_against_web") is not None]
        if web_recs:
            seed_web_prec.append(_mean([r["accuracy_against_web"] for r in web_recs]))
            seed_web_true.append(_mean([r.get("true_rate_against_web", 0) or 0
                                        for r in web_recs]))
            seed_web_false.append(_mean([r.get("false_rate_against_web", 0) or 0
                                         for r in web_recs]))
            seed_web_unv.append(_mean([r.get("unverifiable_rate_against_web", 0) or 0
                                       for r in web_recs]))

        # ── Frontier slice (wiki NOT found, web evidence) ─────────
        not_found = [r for r in batch if not _is_wiki_found(r, strict=True)]
        frontier_web = [r for r in not_found
                        if r.get("accuracy_against_web") is not None]
        if frontier_web:
            seed_frontier_prec.append(_mean([r["accuracy_against_web"]
                                             for r in frontier_web]))
            seed_frontier_true.append(_mean([r.get("true_rate_against_web", 0) or 0
                                             for r in frontier_web]))
            seed_frontier_false.append(_mean([r.get("false_rate_against_web", 0) or 0
                                              for r in frontier_web]))
            seed_frontier_unv.append(_mean([r.get("unverifiable_rate_against_web", 0) or 0
                                            for r in frontier_web]))

    n_ret = len(all_recs)
    n_found = sum(1 for r in all_recs if _is_wiki_found(r, strict=True))
    n_not_found = n_ret - n_found
    n_att = n_per_seed * n_seeds

    frontier_recs = [r for r in all_recs if not _is_wiki_found(r, strict=True)]
    n_frontier_web = sum(1 for r in frontier_recs
                         if r.get("accuracy_against_web") is not None)
    n_web_found = sum(1 for r in all_recs
                      if r.get("accuracy_against_web") is not None)

    sim_keys = [
        "sim_tfidf_cosine", "sim_jaccard", "sim_semantic_cosine",
        "sim_combined_similarity", "sim_ngram_1_overlap",
        "sim_ngram_2_overlap", "sim_ngram_3_overlap", "sim_bertscore_f1",
    ]
    row = {
        "tag": tag,
        "n_seeds": n_seeds,
        "n_attempted_per_seed": n_per_seed,
        "n_attempted_total": n_att,
        "n_returned_total": n_ret,
        # Wiki
        "n_wiki_subject_found": n_found,
        "n_wiki_subject_not_found": n_not_found,
        "wiki_coverage_rate": (n_found / n_ret if n_ret else None),
        "wiki_coverage_rate_conserv": (n_found / n_att if n_att else None),
        "wiki_precision": _mean(seed_prec),
        "wiki_precision_std": _std(seed_prec),
        "wiki_true_rate": _mean(seed_true),
        "wiki_true_rate_std": _std(seed_true),
        "wiki_false_rate": _mean(seed_false),
        "wiki_false_rate_std": _std(seed_false),
        "wiki_unverifiable_rate": _mean(seed_unv),
        "wiki_unverifiable_rate_std": _std(seed_unv),
        # Web (all subjects)
        "n_web_evidence_found": n_web_found,
        "web_evidence_rate": (n_web_found / n_ret if n_ret else None),
        "web_precision": _mean(seed_web_prec),
        "web_precision_std": _std(seed_web_prec),
        "web_true_rate": _mean(seed_web_true),
        "web_true_rate_std": _std(seed_web_true),
        "web_false_rate": _mean(seed_web_false),
        "web_false_rate_std": _std(seed_web_false),
        "web_unverifiable_rate": _mean(seed_web_unv),
        "web_unverifiable_rate_std": _std(seed_web_unv),
        # Frontier (wiki NOT found, web evidence)
        "frontier_n": n_not_found,
        "frontier_n_web_found": n_frontier_web,
        "frontier_web_coverage": (n_frontier_web / n_not_found
                                  if n_not_found else None),
        "frontier_web_precision": _mean(seed_frontier_prec),
        "frontier_web_precision_std": _std(seed_frontier_prec),
        "frontier_web_true_rate": _mean(seed_frontier_true),
        "frontier_web_true_rate_std": _std(seed_frontier_true),
        "frontier_web_false_rate": _mean(seed_frontier_false),
        "frontier_web_false_rate_std": _std(seed_frontier_false),
        "frontier_web_unverifiable_rate": _mean(seed_frontier_unv),
        "frontier_web_unverifiable_rate_std": _std(seed_frontier_unv),
    }
    for sk in sim_keys:
        vals = [r.get(sk) for r in all_recs
                if isinstance(r.get(sk), (int, float))]
        row[f"mean_{sk}"] = _mean(vals)
        row[f"std_{sk}"] = _std(vals)

    return row


# ── corpus stats ──────────────────────────────────────────────────────────────
def corpus_hop_stats(md):
    rows = []
    for bkt in HOP_BUCKETS:
        subs = md.subjects_by_hop.get(bkt, [])
        wcs = [md.subject_meta[s].wc for s in subs if s in md.subject_meta]
        rows.append({
            "hop_bucket": bkt,
            "n_subjects": len(subs),
            "pct_of_corpus": len(subs) / len(md.subject_names)
                             if md.subject_names else 0,
            "mean_word_count": _mean(wcs),
            "std_word_count": _std(wcs),
            "median_word_count": _median(wcs),
        })
    return rows


# ── mode: random ──────────────────────────────────────────────────────────────
def run_random_mode(md, args, audit_root):
    seeds = list(args.seeds)
    n = min(args.sample_n, len(md.subject_names))
    avail = md.subject_names

    print(f"\n[random] n={n}  seeds={seeds}  corpus={len(avail):,}")

    seed_batches = []
    all_recs = []

    for si, seed in enumerate(seeds):
        rng = random.Random(seed)
        sample = sorted(rng.sample(avail, n))
        label = f"random_seed{si}"
        print(f"\n  [random seed {si+1}/{len(seeds)}] seed={seed} n={len(sample)}")

        texts = read_articles_for_subjects(md, sample)
        miss = [s for s in sample if s not in texts or not texts[s].strip()]
        if miss:
            raise RuntimeError(f"[random seed{si}] {len(miss)} missing: {miss[:5]}")

        aud = os.path.join(audit_root, "random", f"seed{si}") if audit_root else ""
        if aud: os.makedirs(aud, exist_ok=True)

        recs = _run_fact(label, md, sample, args, aud, texts=texts)
        seed_batches.append(recs)
        all_recs.extend(recs)

    row = _summarise_records(all_recs, seed_batches, n, len(seeds), "random")
    row["mode"] = "random"
    print(f"  [random] prec={_fmt(row['wiki_precision'])} "
          f"± {_fmt(row['wiki_precision_std'])}  "
          f"found={row['n_wiki_subject_found']}/{row['n_returned_total']}")
    return [row]


# ── mode: hop ─────────────────────────────────────────────────────────────────
def _hop_quotas(args):
    return {
        "hop0": args.sample_n_hop0,
        "hop1": args.sample_n_hop1,
        "hop2": args.sample_n_hop2,
        "hop3": args.sample_n_hop3,
        "hop4": args.sample_n_hop4,
        "hop5plus": args.sample_n_hop5plus,
    }


def run_hop_mode(md, args, audit_root):
    seeds = list(args.seeds)
    quotas = _hop_quotas(args)

    print(f"\n[hop] quotas={quotas}  seeds={seeds}")
    for bkt in HOP_BUCKETS:
        avail = md.subjects_by_hop.get(bkt, [])
        actual = min(quotas[bkt], len(avail))
        print(f"  {bkt}: requested={quotas[bkt]}  available={len(avail):,}  "
              f"effective={actual}")

    per_hop_rows = []
    agg_all_recs = []
    agg_seed_data = {b: [] for b in HOP_BUCKETS}

    for bkt in HOP_BUCKETS:
        avail = md.subjects_by_hop.get(bkt, [])
        n = min(quotas[bkt], len(avail))
        if n == 0:
            print(f"  [{bkt}] 0 effective — skipping")
            per_hop_rows.append({
                "mode": "hop", "tag": bkt,
                "n_seeds": len(seeds),
                "n_attempted_per_seed": 0, "n_attempted_total": 0,
                "n_returned_total": 0, "n_wiki_subject_found": 0,
                "wiki_precision": None, "wiki_false_rate": None,
                "note": "no subjects available",
            })
            continue

        seed_batches = []
        all_recs = []

        for si, seed in enumerate(seeds):
            rng = random.Random(seed)
            # If bucket fits in quota, use all (std=0, honest)
            sample = (sorted(avail) if n >= len(avail)
                      else sorted(rng.sample(avail, n)))
            label = f"hop_{bkt}_seed{si}"
            print(f"\n  [{bkt} seed {si+1}/{len(seeds)}] "
                  f"seed={seed}  n={len(sample)}")

            texts = read_articles_for_subjects(md, sample)
            miss = [s for s in sample if s not in texts or not texts[s].strip()]
            if miss:
                raise RuntimeError(
                    f"[{bkt} seed{si}] {len(miss)} missing: {miss[:5]}")

            aud = (os.path.join(audit_root, "hop", bkt, f"seed{si}")
                   if audit_root else "")
            if aud: os.makedirs(aud, exist_ok=True)

            recs = _run_fact(label, md, sample, args, aud, texts=texts)
            seed_batches.append(recs)
            all_recs.extend(recs)
            agg_seed_data[bkt].append(recs)

        agg_all_recs.extend(all_recs)
        row = _summarise_records(all_recs, seed_batches, n, len(seeds), bkt)
        row["mode"] = "hop"
        per_hop_rows.append(row)
        print(f"  [{bkt}] prec={_fmt(row['wiki_precision'])} "
              f"± {_fmt(row['wiki_precision_std'])}  "
              f"found={row['n_wiki_subject_found']}/{row['n_returned_total']}")

    # Aggregate
    all_seed_batches = []
    for bkt in HOP_BUCKETS:
        all_seed_batches.extend(agg_seed_data[bkt])
    n_per_seed_total = sum(min(quotas[b], len(md.subjects_by_hop.get(b, [])))
                           for b in HOP_BUCKETS)

    agg_row = _summarise_records(
        agg_all_recs, all_seed_batches,
        n_per_seed_total, len(seeds), "overall_hop")
    agg_row["mode"] = "hop"

    print(f"\n  [overall-hop] prec={_fmt(agg_row['wiki_precision'])} "
          f"± {_fmt(agg_row['wiki_precision_std'])}  "
          f"found={agg_row['n_wiki_subject_found']}/{agg_row['n_returned_total']}")

    return per_hop_rows, [agg_row]


# ── figures ───────────────────────────────────────────────────────────────────
def _save_fig(fig, base):
    for ext in (".pdf", ".png"):
        fig.savefig(base + ext, bbox_inches="tight",
                    **({"dpi": 300} if ext == ".png" else {}))
    plt.close(fig)
    print(f"  [fig] {os.path.basename(base)}")


def generate_figures(md, hop_stats, random_rows, hop_rows, hop_agg, fig_dir):
    if not HAS_PLOT: return
    os.makedirs(fig_dir, exist_ok=True)
    plt.rcParams.update(ACL_STYLE)

    # Fig 1: Corpus hop distribution
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    bkts = [r["hop_bucket"] for r in hop_stats]
    ns = [r["n_subjects"] for r in hop_stats]
    colors = [HOP_COLORS.get(b, "#607D8B") for b in bkts]

    ax = axes[0]
    bars = ax.bar(bkts, ns, color=colors, alpha=0.85, edgecolor="white")
    for bar, n_ in zip(bars, ns):
        if n_ > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{n_:,}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Subjects"); ax.set_title("Subjects per hop", fontweight="bold")
    ax.set_xticklabels(bkts, rotation=20, ha="right")

    ax = axes[1]
    pcts = [r["pct_of_corpus"] * 100 for r in hop_stats]
    ax.bar(bkts, pcts, color=colors, alpha=0.85, edgecolor="white")
    ax.set_ylabel("% of corpus"); ax.set_title("Hop distribution (%)", fontweight="bold")
    ax.set_xticklabels(bkts, rotation=20, ha="right")

    fig.suptitle(f"Model: {md.display_name}", fontsize=9, style="italic")
    fig.tight_layout()
    _save_fig(fig, os.path.join(fig_dir, "fig1_corpus_hop_dist"))

    # Fig 2: Factuality by hop
    hop_fact = [r for r in hop_rows if r.get("wiki_precision") is not None]
    if hop_fact:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
        ht = [r["tag"] for r in hop_fact]
        hc = [HOP_COLORS.get(t, "#607D8B") for t in ht]

        for ax, metric, ylabel, title in [
            (axes[0], "wiki_precision", "Precision", "Factuality Precision by Hop"),
            (axes[1], "wiki_false_rate", "Hallucination rate", "Hallucination Rate by Hop"),
        ]:
            vals = [r.get(metric) or 0 for r in hop_fact]
            errs = [r.get(f"{metric}_std") or 0 for r in hop_fact]
            bars = ax.bar(ht, vals, color=hc, alpha=0.85, edgecolor="white",
                          yerr=errs, capsize=4,
                          error_kw={"elinewidth": 1.2, "capthick": 1.2})
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)
            if random_rows and random_rows[0].get(metric) is not None:
                rv = random_rows[0][metric]
                ax.axhline(rv, color=HOP_COLORS["random"], ls="--", lw=1.4,
                           label=f"Random ({rv:.3f})")
                ax.legend(fontsize=7)
            ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold")
            ax.set_ylim(0, 1.05)
            ax.set_xticklabels(ht, rotation=20, ha="right")

        fig.suptitle(f"Model: {md.display_name}", fontsize=9, style="italic")
        fig.tight_layout()
        _save_fig(fig, os.path.join(fig_dir, "fig2_factuality_by_hop"))

    # Fig 3: Coverage by hop
    hop_cov = [r for r in hop_rows if r.get("wiki_coverage_rate") is not None]
    if hop_cov:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ht = [r["tag"] for r in hop_cov]
        covs = [r.get("wiki_coverage_rate") or 0 for r in hop_cov]
        cov_c = [r.get("wiki_coverage_rate_conserv") or 0 for r in hop_cov]
        x = np.arange(len(ht)); w = 0.35
        hc = [HOP_COLORS.get(t, "#607D8B") for t in ht]
        ax.bar(x - w/2, covs, w, color=hc, alpha=0.85, label="vs returned")
        ax.bar(x + w/2, cov_c, w, color=hc, alpha=0.45, hatch="//",
               label="vs attempted (conservative)")
        ax.set_xticks(x); ax.set_xticklabels(ht, rotation=20, ha="right")
        ax.set_ylabel("Wikipedia coverage"); ax.set_ylim(0, 1.05)
        ax.set_title("Coverage by hop", fontweight="bold")
        ax.legend(fontsize=7)
        fig.tight_layout()
        _save_fig(fig, os.path.join(fig_dir, "fig3_coverage_by_hop"))


# ── text report ───────────────────────────────────────────────────────────────
def _rtable(rows, cols=None):
    if not rows: return "  (no data)\n"
    if cols is None:
        cols = list(dict.fromkeys(k for r in rows for k in r.keys()))
    widths = {}
    for c in cols:
        w = len(str(c))
        for r in rows:
            v = str(r.get(c, ""))
            try:
                fv = float(v)
                v = f"{fv:.4f}" if abs(fv) < 10000 else f"{int(fv):,}"
            except: pass
            w = max(w, len(v))
        widths[c] = min(max(w + 2, 8), 40)
    def fv(val, w):
        if val is None or val == "": return " " * w
        try:
            f = float(val)
            if abs(f) < .001 and f != 0: return f"{f:{w}.4e}"
            if f == int(f) and abs(f) > 10: return f"{int(f):>{w},}"
            return f"{f:{w}.4f}"
        except: return f"{str(val):>{w}}"
    hdr = "".join(f"{str(c):>{widths[c]}}" for c in cols)
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        lines.append("".join(fv(r.get(c, ""), widths[c]) for c in cols))
    return "\n".join(lines) + "\n"


def generate_text_report(out_dir, md, hop_stats, random_rows,
                         hop_rows, hop_agg, args):
    path = os.path.join(out_dir, "report.txt")
    L = []
    L.append("=" * 70)
    L.append("  LLMPedia Track 3 — Hop-Stratified Factuality Report")
    L.append(f"  Model: {md.display_name}  (key={md.key})")
    L.append(f"  Mode:  {args.mode}  Seeds: {args.seeds}")
    L.append(f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append("=" * 70)

    L.append(f"\n{'='*70}\n  CORPUS HOP DISTRIBUTION\n{'='*70}\n")
    L.append(_rtable(hop_stats))

    fact_cols = [
        "tag", "n_seeds", "n_attempted_per_seed", "n_attempted_total",
        "n_returned_total", "n_wiki_subject_found",
        "wiki_coverage_rate", "wiki_coverage_rate_conserv",
        "wiki_precision", "wiki_precision_std",
        "wiki_false_rate", "wiki_false_rate_std",
        "wiki_true_rate", "wiki_true_rate_std",
        "wiki_unverifiable_rate", "wiki_unverifiable_rate_std",
    ]

    if random_rows:
        L.append(f"\n{'='*70}\n  RANDOM MODE\n{'='*70}\n")
        L.append(_rtable(random_rows, fact_cols))

    if hop_rows:
        L.append(f"\n{'='*70}\n  HOP MODE — Per Bucket\n{'='*70}\n")
        L.append(_rtable(hop_rows, fact_cols))

    if hop_agg:
        L.append(f"\n{'='*70}\n  HOP MODE — Aggregate\n{'='*70}\n")
        L.append(_rtable(hop_agg, fact_cols))

    L.append("\n" + "=" * 70 + "\n  END OF REPORT\n" + "=" * 70 + "\n")
    text = "\n".join(L)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[report] {path}  ({len(text):,} chars)")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Track 3: Hop-stratified factuality (ACL 2026)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model-key", default="")
    ap.add_argument("--articles-file", default="articles.jsonl")
    ap.add_argument("--output-dir", default="./hop_factuality")
    ap.add_argument("--min-words", type=int, default=100)

    ap.add_argument("--mode", choices=["random", "hop", "both"], default="both")

    # Random mode
    ap.add_argument("--sample-n", type=int, default=1000)

    # Hop mode — defaults tuned for actual distribution
    # (hop0=1, hop1=~50, hop2=~2000, hop3/4=bulk)
    ap.add_argument("--sample-n-hop0", type=int, default=0,
                    help="0 = take all (typically 1)")
    ap.add_argument("--sample-n-hop1", type=int, default=0,
                    help="0 = take all (typically ~50)")
    ap.add_argument("--sample-n-hop2", type=int, default=200)
    ap.add_argument("--sample-n-hop3", type=int, default=200)
    ap.add_argument("--sample-n-hop4", type=int, default=200)
    ap.add_argument("--sample-n-hop5plus", type=int, default=200)

    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)

    # Factuality backend — DEFAULT: wikipedia only (no DDG/web)
    ap.add_argument("--fact-model-key", default="gpt-4.1-nano")
    ap.add_argument("--evidence-sources", default="wikipedia,web",
                    help="Comma-separated. Default: wikipedia,web")
    ap.add_argument("--max-claims", type=int, default=10)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--web-mode", choices=["snippets", "single", "hybrid", "full"],
                    default="snippets")
    ap.add_argument("--search-backend",
                    choices=["auto","valyu","serper","brave","ddg","searxng"],
                    default="auto",
                    help="Search backend (default: auto — valyu if key set, else serper, etc.)")
    ap.add_argument("--web-cache-dir", default="",
                    help="Persistent web page cache directory")
    ap.add_argument("--compute-similarity", action="store_true", default=False)
    ap.add_argument("--compute-bertscore", action="store_true", default=False)
    ap.add_argument("--compute-stylistic", action="store_true", default=False)

    ap.add_argument("--audit-dir", default="")
    ap.add_argument("--clean-output", action="store_true")
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()

    # 0 = take all
    _BIG = 999_999
    if args.sample_n_hop0 == 0: args.sample_n_hop0 = _BIG
    if args.sample_n_hop1 == 0: args.sample_n_hop1 = _BIG

    out_dir = os.path.abspath(args.output_dir)
    fig_dir = os.path.join(out_dir, "figures")
    audit_root = args.audit_dir or os.path.join(out_dir, "audit")

    if args.clean_output:
        import shutil
        if os.path.isdir(out_dir): shutil.rmtree(out_dir)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(audit_root, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"[track3] HOP-STRATIFIED FACTUALITY")
    print(f"  run-dir:  {args.run_dir}")
    print(f"  mode:     {args.mode}")
    print(f"  seeds:    {args.seeds}")
    print(f"  evidence: {args.evidence_sources}")
    print(f"  output:   {out_dir}")
    print(f"{'='*70}\n")

    # Load
    model_key = args.model_key or os.path.basename(os.path.normpath(args.run_dir))
    md = load_model_data(model_key, args.run_dir, args.articles_file,
                         args.min_words)

    # Corpus stats
    print("\n[1] Corpus hop distribution ...")
    hop_stats = corpus_hop_stats(md)
    for r in hop_stats:
        print(f"  {r['hop_bucket']}: {r['n_subjects']:>8,}  "
              f"({r['pct_of_corpus']:5.1%})  "
              f"mean_wc={_fmt(r.get('mean_word_count'), 0)}")

    # Run modes
    random_rows = []
    hop_rows = []
    hop_agg = []

    if args.mode in ("random", "both"):
        print(f"\n[2] RANDOM MODE (n={args.sample_n})")
        if HAS_FACTUALITY:
            random_rows = run_random_mode(md, args, audit_root)
        else:
            print("  [skip] factuality_core not available")

    if args.mode in ("hop", "both"):
        step = "3" if args.mode == "both" else "2"
        print(f"\n[{step}] HOP MODE")
        if HAS_FACTUALITY:
            hop_rows, hop_agg = run_hop_mode(md, args, audit_root)
        else:
            print("  [skip] factuality_core not available")

    # Write
    print(f"\n[output] Writing ...")
    _write_csv(hop_stats, os.path.join(out_dir, "corpus_hop_stats.csv"))
    if random_rows:
        _write_csv(random_rows, os.path.join(out_dir, "factuality_random.csv"))
    if hop_rows:
        _write_csv(hop_rows, os.path.join(out_dir, "factuality_hop_per_bucket.csv"))
    if hop_agg:
        _write_csv(hop_agg, os.path.join(out_dir, "factuality_hop_aggregate.csv"))
    all_fact = random_rows + hop_rows + hop_agg
    if all_fact:
        _write_csv(all_fact, os.path.join(out_dir, "factuality_all.csv"))

    # JSON
    report_json = {
        "generated": datetime.datetime.now().isoformat(),
        "config": vars(args),
        "model": {"key": md.key, "display": md.display_name,
                  "n_subjects": len(md.subject_names),
                  "n_total_raw": md.n_total_raw},
        "corpus_hop_stats": hop_stats,
        "factuality_random": random_rows,
        "factuality_hop_per_bucket": hop_rows,
        "factuality_hop_aggregate": hop_agg,
    }
    rp = os.path.join(out_dir, "hop_factuality_report.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report_json, f, ensure_ascii=False, indent=2, default=str)
    print(f"  -> {rp}")

    # Figures + report
    print("\n[figures] ...")
    generate_figures(md, hop_stats, random_rows, hop_rows, hop_agg, fig_dir)
    print("\n[report] ...")
    generate_text_report(out_dir, md, hop_stats, random_rows,
                         hop_rows, hop_agg, args)

    # Summary
    print(f"\n{'='*70}")
    print(f"[track3] DONE — {md.display_name}")
    for bkt in HOP_BUCKETS:
        print(f"  {bkt}: {len(md.subjects_by_hop.get(bkt, [])):>8,}")
    if random_rows:
        r = random_rows[0]
        print(f"  random:  prec={_fmt(r.get('wiki_precision'))} "
              f"± {_fmt(r.get('wiki_precision_std'))}")
    if hop_rows:
        for r in hop_rows:
            if r.get("wiki_precision") is not None:
                print(f"  {r['tag']:8s}: prec={_fmt(r['wiki_precision'])} "
                      f"± {_fmt(r.get('wiki_precision_std'))}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()



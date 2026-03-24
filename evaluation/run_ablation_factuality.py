#!/usr/bin/env python3
"""
run_ablation_factuality.py — Multi-run ablation factuality evaluation
                              for LLMPedia (ACL 2026).

OVERVIEW
────────
  Walks a root ablation directory, discovers all run sub-directories
  (each containing run_meta.json + articles.jsonl), extracts the ablation
  config (model, self-rag, prompt strategy, reasoning effort), runs
  random-sample factuality on each, and produces:

    1. A combined CSV of all results
    2. A LaTeX table matching the ACL ablation format
    3. A JSON report with full details

USAGE
─────
  python run_ablation_factuality.py \\
      --ablation-root ./openLLMPedia/ablation_final \\
      --sample-n 100 \\
      --seeds 42 123 7 \\
      --output-dir ./ablation_results \\
      --evidence-sources wikipedia,web

  # Dry-run (just discover dirs + extract config, no factuality):
  python run_ablation_factuality.py \\
      --ablation-root ./openLLMPedia/ablation_final \\
      --dry-run

DIRECTORY STRUCTURE EXPECTED
────────────────────────────
  ablation_root/
    ├── gpt-5-mini/
    │   ├── low/
    │   │   ├── baseline_no_selfrag_batch/
    │   │   │   ├── run_meta.json
    │   │   │   └── articles.jsonl
    │   │   ├── calibrated_selfrag_batch/
    │   │   │   ├── run_meta.json
    │   │   │   └── articles.jsonl
    │   │   └── ...
    │   └── min/
    │       └── ...
    └── gpt-5-nano/
        └── ...
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
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── path setup ────────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    from factuality_core import (EvalConfig, EvalInput, run_evaluation,
                                 generate_outputs)
    HAS_FACTUALITY = True
except ImportError:
    HAS_FACTUALITY = False
    print("[WARN] factuality_core not available — factuality steps will be skipped")

try:
    from json_repair import extract_json_robust as _extract_json
    print("[OK] json_repair loaded")
except ImportError:
    pass

# ── constants ─────────────────────────────────────────────────────────────────
DEFAULT_SEEDS = [42, 123, 7]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DISCOVERY — find all run directories
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RunConfig:
    """Ablation config extracted from a single run_meta.json."""
    run_dir: str
    model: str = ""
    self_rag: bool = False
    prompt_strategy: str = ""       # "baseline" or "calibrated"
    reasoning_effort: str = ""      # "low", "min", etc.
    display_name: str = ""
    n_subjects: int = 0             # from articles.jsonl line count (lazy)
    cost_per_article: float = 0.0   # if available in run_meta
    run_meta: Dict = field(default_factory=dict)

    # Short labels for LaTeX table
    @property
    def sg_label(self) -> str:
        return r"\cmark" if self.self_rag else r"\xmark"

    @property
    def prompt_label(self) -> str:
        m = {"baseline": "base", "calibrated": "calib"}
        return m.get(self.prompt_strategy, self.prompt_strategy[:4])

    @property
    def reasoning_label(self) -> str:
        return self.reasoning_effort

    @property
    def sort_key(self) -> tuple:
        """Sort: model → self_rag → prompt → reasoning."""
        model_order = {"gpt-5-mini": 0, "gpt-5-nano": 1}
        prompt_order = {"baseline": 0, "calibrated": 1}
        reason_order = {"min": 0, "low": 1, "medium": 2, "high": 3}
        return (
            model_order.get(self.model, 99),
            0 if not self.self_rag else 1,
            prompt_order.get(self.prompt_strategy, 99),
            reason_order.get(self.reasoning_effort, 99),
        )


def _parse_run_meta(meta_path: str) -> RunConfig:
    """Extract ablation config from a run_meta.json file."""
    with open(meta_path, "r", encoding="utf-8") as f:
        rm = json.load(f)

    run_dir = os.path.dirname(meta_path)
    cas = rm.get("cascading_defaults") or {}
    ar = rm.get("args_raw") or {}

    # Model
    model = (cas.get("global_model_key")
             or ar.get("model_key")
             or ar.get("elicit_model_key")
             or "").strip()

    # Self-RAG
    self_rag = bool(rm.get("self_rag_enabled", ar.get("self_rag", False)))

    # Prompt / elicitation strategy
    prompt_strategy = (rm.get("elicitation_strategy")
                       or ar.get("elicitation_strategy")
                       or "baseline").strip()

    # Reasoning effort
    reasoning_effort = (cas.get("global_reasoning_effort")
                        or ar.get("reasoning_effort")
                        or "low").strip()

    # Display name
    sg_tag = "selfrag" if self_rag else "noselfrag"
    display_name = f"{model}/{reasoning_effort}/{prompt_strategy}_{sg_tag}"

    # Cost estimate (if present — from token tracking, etc.)
    cost = 0.0
    # Some setups log total_cost or per-article cost in run_meta
    if "total_cost_usd" in rm:
        total = rm["total_cost_usd"]
        n = rm.get("n_articles_written", 1)
        cost = total / max(n, 1)
    elif "cost_per_article_usd" in rm:
        cost = rm["cost_per_article_usd"]

    return RunConfig(
        run_dir=run_dir,
        model=model,
        self_rag=self_rag,
        prompt_strategy=prompt_strategy,
        reasoning_effort=reasoning_effort,
        display_name=display_name,
        cost_per_article=cost,
        run_meta=rm,
    )


def discover_runs(root: str, articles_file: str = "articles.jsonl") -> List[RunConfig]:
    """Walk root directory tree, find all dirs with run_meta.json + articles."""
    runs = []
    root = os.path.abspath(root)
    print(f"[discover] Walking {root} ...")

    for dirpath, dirnames, filenames in os.walk(root):
        meta_path = os.path.join(dirpath, "run_meta.json")
        arts_path = os.path.join(dirpath, articles_file)

        if os.path.isfile(meta_path) and os.path.isfile(arts_path):
            try:
                rc = _parse_run_meta(meta_path)
                # Quick line count for articles
                with open(arts_path, "r", encoding="utf-8") as f:
                    rc.n_subjects = sum(1 for line in f if line.strip())
                runs.append(rc)
                print(f"  [found] {rc.display_name}  "
                      f"({rc.n_subjects:,} articles)  "
                      f"sg={rc.self_rag}  pr={rc.prompt_strategy}  "
                      f"rs={rc.reasoning_effort}")
            except Exception as e:
                print(f"  [WARN] Failed to parse {meta_path}: {e}")

    runs.sort(key=lambda r: r.sort_key)
    print(f"[discover] Found {len(runs)} valid run directories")
    return runs


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DATA LOADING (reused from run_track3, simplified for random-only)
# ═══════════════════════════════════════════════════════════════════════════════

def word_count(t: str) -> int:
    return len(re.sub(r"[^\w\s]", " ", t.lower()).split()) if t else 0


def load_subject_names(arts_path: str, min_words: int = 100) -> List[str]:
    """Load unique subject names from articles.jsonl (lightweight pass)."""
    seen = set()
    with open(arts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                a = json.loads(line)
            except Exception:
                continue
            if not isinstance(a, dict):
                continue
            s = (a.get("subject") or "").strip()
            if not s:
                continue
            wt = a.get("wikitext") or ""
            if min_words > 0 and word_count(wt) < min_words:
                continue
            seen.add(s)
    return sorted(seen)


def read_articles_for_subjects(arts_path: str,
                               subjects: set) -> Dict[str, str]:
    """Read wikitext for specific subjects (full scan, keep lowest hop)."""
    result = {}
    best_hop = {}
    with open(arts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                a = json.loads(line)
            except Exception:
                continue
            if not isinstance(a, dict):
                continue
            s = (a.get("subject") or "").strip()
            if s not in subjects:
                continue
            hop = 0
            try:
                hop = int(a.get("hop", 0))
            except Exception:
                pass
            if s not in best_hop or hop < best_hop[s]:
                best_hop[s] = hop
                result[s] = a.get("wikitext") or ""
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FACTUALITY EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

def _is_wiki_found(rec: dict, strict: bool = False) -> bool:
    wf = rec.get("wiki_subject_found")
    if wf is not None:
        return bool(wf)
    if strict:
        return False
    if (rec.get("wiki_n_supported", 0) or 0) > 0:
        return True
    if (rec.get("wiki_n_refuted", 0) or 0) > 0:
        return True
    if any(k.startswith("sim_") and isinstance(rec.get(k), (int, float))
           for k in rec.keys()):
        return True
    n_claims = rec.get("n_claims", 0)
    wiki_ins = rec.get("wiki_n_insufficient", 0) or 0
    if n_claims > 0 and wiki_ins == n_claims:
        return False
    return True


def _mean(v):
    n = [x for x in v if isinstance(x, (int, float)) and not math.isnan(x)]
    return sum(n) / len(n) if n else None


def _std(v):
    n = [x for x in v if isinstance(x, (int, float)) and not math.isnan(x)]
    return statistics.stdev(n) if len(n) >= 2 else None


def _fmt(v, d=4):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)


def run_factuality_for_run(
    rc: RunConfig,
    args,
    audit_dir: str = "",
) -> Dict[str, Any]:
    """
    Run random-sample factuality for a single run directory.
    Returns a result dict with all metrics.
    """
    arts_path = os.path.join(rc.run_dir, args.articles_file)
    print(f"\n{'─'*60}")
    print(f"[eval] {rc.display_name}")
    print(f"       dir: {rc.run_dir}")

    # Load subjects
    t0 = time.perf_counter()
    all_subjects = load_subject_names(arts_path, args.min_words)
    n_avail = len(all_subjects)
    print(f"  {n_avail:,} subjects (min_words={args.min_words})")

    if n_avail == 0:
        print("  [SKIP] No subjects found")
        return _empty_result(rc)

    n_sample = min(args.sample_n, n_avail)
    seeds = list(args.seeds)

    # Per-seed evaluation
    seed_batches = []
    all_recs = []

    for si, seed in enumerate(seeds):
        rng = random.Random(seed)
        sample = sorted(rng.sample(all_subjects, n_sample))
        label = f"{rc.display_name}_seed{si}".replace("/", "_")

        print(f"\n  [seed {si+1}/{len(seeds)}] seed={seed}  n={len(sample)}")

        # Read wikitext
        texts = read_articles_for_subjects(arts_path, set(sample))
        missing = [s for s in sample if s not in texts or not texts[s].strip()]
        if missing:
            print(f"  [WARN] {len(missing)} subjects have empty text, removing")
            sample = [s for s in sample if s in texts and texts[s].strip()]

        if not sample:
            print("  [SKIP] No valid articles for this seed")
            seed_batches.append([])
            continue

        # Build eval inputs
        aud = os.path.join(audit_dir, label) if audit_dir else ""
        if aud:
            os.makedirs(aud, exist_ok=True)

        cfg = EvalConfig(
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
            compute_similarity=args.compute_similarity,
            compute_bertscore=args.compute_bertscore,
            compute_stylistic=getattr(args, "compute_stylistic", False),
            run_audit_dir=aud,
            debug=getattr(args, "debug", False),
        )

        _seed_topic = (rc.run_meta.get("seed") or "").strip()
        inputs = [
            EvalInput(
                subject=s,
                candidate=rc.display_name,
                article_text=texts[s],
                hop=0,
                generator_model=rc.model,
                topic=_seed_topic,
                clean_wiki_markup=True,
            )
            for s in sample
        ]

        out_path = os.path.join(
            args.output_dir,
            f"raw_{label}.jsonl"
        )

        # Run with retry
        recs = []
        backoff = 2.0
        max_tries = max(1, args.max_retries)
        for attempt in range(max_tries):
            try:
                recs = run_evaluation(inputs, cfg, out_path)
                break
            except Exception as e:
                if attempt < max_tries - 1:
                    wait = backoff * (attempt + 1)
                    print(f"  [WARN] attempt {attempt+1} failed: {e}  "
                          f"(retry in {wait:.1f}s)")
                    time.sleep(wait)
                else:
                    print(f"  [ERROR] failed after {max_tries} attempts: {e}")

        if len(recs) != len(inputs):
            print(f"  [WARN] returned {len(recs)}/{len(inputs)}")

        seed_batches.append(recs)
        all_recs.extend(recs)

    elapsed = time.perf_counter() - t0
    print(f"  [done] {len(all_recs)} total records in {elapsed:.1f}s")

    # Aggregate
    return _aggregate_results(rc, all_recs, seed_batches, n_sample, seeds)


def _empty_result(rc: RunConfig) -> Dict[str, Any]:
    """Return empty result row for a run that was skipped."""
    return {
        "run_dir": rc.run_dir,
        "model": rc.model,
        "self_rag": rc.self_rag,
        "prompt_strategy": rc.prompt_strategy,
        "reasoning_effort": rc.reasoning_effort,
        "display_name": rc.display_name,
        "n_subjects_available": 0,
        "n_attempted": 0,
        "n_returned": 0,
        "wiki_precision": None,
        "wiki_precision_std": None,
        "wiki_true_rate": None,
        "wiki_false_rate": None,
        "wiki_unverifiable_rate": None,
        "wiki_coverage": None,
        "sim_semantic": None,
        "sim_combined": None,
        "cost_per_article": rc.cost_per_article,
    }


def _aggregate_results(
    rc: RunConfig,
    all_recs: list,
    seed_batches: list,
    n_per_seed: int,
    seeds: list,
) -> Dict[str, Any]:
    """Aggregate factuality metrics across seeds."""
    seed_prec = []
    seed_true = []
    seed_false = []
    seed_unv = []
    # Web
    seed_web_prec = []
    seed_web_true = []
    seed_web_false = []
    seed_web_unv = []

    for batch in seed_batches:
        if not batch:
            continue

        # Wiki
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

        # Web
        web_recs = [r for r in batch if r.get("accuracy_against_web") is not None]
        if web_recs:
            seed_web_prec.append(_mean([r["accuracy_against_web"] for r in web_recs]))
            seed_web_true.append(_mean([r.get("true_rate_against_web", 0) or 0
                                        for r in web_recs]))
            seed_web_false.append(_mean([r.get("false_rate_against_web", 0) or 0
                                         for r in web_recs]))
            seed_web_unv.append(_mean([r.get("unverifiable_rate_against_web", 0) or 0
                                       for r in web_recs]))

    n_ret = len(all_recs)
    n_found = sum(1 for r in all_recs if _is_wiki_found(r, strict=True))
    n_att = n_per_seed * len(seeds)

    # Similarity metrics
    sim_sem_vals = [r.get("sim_semantic_cosine") for r in all_recs
                    if isinstance(r.get("sim_semantic_cosine"), (int, float))]
    sim_comb_vals = [r.get("sim_combined_similarity") for r in all_recs
                     if isinstance(r.get("sim_combined_similarity"), (int, float))]

    return {
        "run_dir": rc.run_dir,
        "model": rc.model,
        "self_rag": rc.self_rag,
        "prompt_strategy": rc.prompt_strategy,
        "reasoning_effort": rc.reasoning_effort,
        "display_name": rc.display_name,
        "n_subjects_available": rc.n_subjects,
        "n_seeds": len(seeds),
        "n_per_seed": n_per_seed,
        "n_attempted": n_att,
        "n_returned": n_ret,
        "n_wiki_found": n_found,
        "wiki_coverage": n_found / n_ret if n_ret else None,
        # Wiki factuality
        "wiki_precision": _mean(seed_prec),
        "wiki_precision_std": _std(seed_prec),
        "wiki_true_rate": _mean(seed_true),
        "wiki_true_rate_std": _std(seed_true),
        "wiki_false_rate": _mean(seed_false),
        "wiki_false_rate_std": _std(seed_false),
        "wiki_unverifiable_rate": _mean(seed_unv),
        "wiki_unverifiable_rate_std": _std(seed_unv),
        # Web factuality
        "web_precision": _mean(seed_web_prec),
        "web_precision_std": _std(seed_web_prec),
        "web_true_rate": _mean(seed_web_true),
        "web_false_rate": _mean(seed_web_false),
        "web_unverifiable_rate": _mean(seed_web_unv),
        # Similarity
        "sim_semantic": _mean(sim_sem_vals),
        "sim_combined": _mean(sim_comb_vals),
        # Cost
        "cost_per_article": rc.cost_per_article,
        "cost_at_1m": rc.cost_per_article * 1_000_000 if rc.cost_per_article else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. OUTPUT — CSV + LaTeX + JSON
# ═══════════════════════════════════════════════════════════════════════════════

def write_csv(results: List[dict], path: str):
    if not results:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = list(dict.fromkeys(k for r in results for k in r.keys()))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({k: (_fmt(r.get(k)) if isinstance(r.get(k), float)
                           else r.get(k, "")) for k in keys})
    print(f"  -> {path}  ({len(results)} rows)")


def _latex_pct(val, std=None, bold=False):
    """Format a metric as XX.X with optional ±std for LaTeX."""
    if val is None:
        return "—"
    pct = val * 100
    if std is not None and std > 0:
        std_pct = std * 100
        core = f"{pct:.1f}" + "_{\\pm" + f"{std_pct:.1f}" + "}"
        if bold:
            return f"$\\mathbf{{{pct:.1f}}}""_{\\pm" + f"{std_pct:.1f}" + "}$"
        return f"${core}$"
    if bold:
        return f"$\\mathbf{{{pct:.1f}}}$"
    return f"{pct:.1f}"


def _latex_sim(val):
    """Format similarity as .XXX"""
    if val is None:
        return "—"
    return f".{val*1000:.0f}" if val < 1 else f"{val:.3f}"


def _latex_cost(val):
    """Format cost per article."""
    if val is None or val == 0:
        return "—"
    return f".{val*10000:.0f}" if val < 0.01 else f"{val:.4f}"


def _latex_cost_1m(val):
    """Format cost at 1M articles (with thousands separator)."""
    if val is None or val == 0:
        return "—"
    return f"{val:,.0f}".replace(",", "{,}")


def generate_latex_table(results: List[dict], path: str):
    """Generate the ACL-style ablation LaTeX table."""
    if not results:
        print("  [latex] No results to tabulate")
        return

    # Sort results
    def sort_key(r):
        model_order = {"gpt-5-mini": 0, "gpt-5-nano": 1}
        prompt_order = {"baseline": 0, "calibrated": 1}
        reason_order = {"min": 0, "low": 1, "medium": 2, "high": 3}
        return (
            model_order.get(r.get("model", ""), 99),
            0 if not r.get("self_rag") else 1,
            prompt_order.get(r.get("prompt_strategy", ""), 99),
            reason_order.get(r.get("reasoning_effort", ""), 99),
        )

    results = sorted(results, key=sort_key)

    # Find best precision per model group (for bolding)
    best_prec = {}
    for r in results:
        m = r.get("model", "")
        p = r.get("wiki_precision")
        if p is not None:
            if m not in best_prec or p > best_prec[m]:
                best_prec[m] = p

    # Build table
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{2.0pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.06}")
    lines.append(r"\begin{tabular}{ccc r rrrr r r r}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{SG} & \textbf{Pr.} & \textbf{Rs.} &"
    )
    lines.append(
        r"\textbf{Acc.\%} & \textbf{T\%} & \textbf{F\%} & \textbf{U\%} &"
    )
    lines.append(
        r"\textbf{Sem.} & \textbf{Comb.} & \textbf{\$/art} & \textbf{\$@1M} \\"
    )
    lines.append(r"\midrule")

    current_model = None
    for r in results:
        model = r.get("model", "")

        # Model group header
        if model != current_model:
            if current_model is not None:
                lines.append(r"\midrule")
            lines.append(
                r"\multicolumn{11}{l}{\textit{" + model + r"}} \\[1pt]"
            )
            current_model = model

        # Self-RAG
        sg = r"\cmark" if r.get("self_rag") else r"\xmark"

        # Prompt
        pr_map = {"baseline": "base", "calibrated": "calib"}
        pr = pr_map.get(r.get("prompt_strategy", ""), r.get("prompt_strategy", "")[:4])

        # Reasoning
        rs = r.get("reasoning_effort", "")

        # Precision (bold if best for this model)
        prec = r.get("wiki_precision")
        prec_std = r.get("wiki_precision_std")
        is_best = (prec is not None and
                   abs(prec - best_prec.get(model, -1)) < 1e-6)
        acc_str = _latex_pct(prec, prec_std, bold=is_best)

        # Other rates (as simple percentages)
        true_str = _latex_pct(r.get("wiki_true_rate"))
        false_str = _latex_pct(r.get("wiki_false_rate"))
        unv_str = _latex_pct(r.get("wiki_unverifiable_rate"))

        # Similarity
        sem_str = _latex_sim(r.get("sim_semantic"))
        comb_str = _latex_sim(r.get("sim_combined"))

        # Cost
        cost_art_str = _latex_cost(r.get("cost_per_article"))
        cost_1m_str = _latex_cost_1m(r.get("cost_at_1m"))

        row = (f"{sg} & {pr} & {rs} & "
               f"{acc_str} & {true_str} & {false_str} & {unv_str} & "
               f"{sem_str} & {comb_str} & {cost_art_str} & {cost_1m_str} \\\\")
        lines.append(row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Full $2^3$ ablation — random sample factuality.}")
    lines.append(r"\label{tab:ablation_full}")
    lines.append(r"\end{table}")

    text = "\n".join(lines)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  -> {path}")
    print("\n" + text + "\n")


def generate_json_report(results: List[dict], runs: List[RunConfig],
                         args, path: str):
    report = {
        "generated": datetime.datetime.now().isoformat(),
        "config": vars(args),
        "n_runs": len(runs),
        "runs_discovered": [
            {
                "dir": rc.run_dir,
                "model": rc.model,
                "self_rag": rc.self_rag,
                "prompt": rc.prompt_strategy,
                "reasoning": rc.reasoning_effort,
                "n_articles": rc.n_subjects,
            }
            for rc in runs
        ],
        "results": results,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    print(f"  -> {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Ablation factuality runner — discovers runs, evaluates, "
                    "produces LaTeX table (ACL 2026)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Discovery ──
    ap.add_argument("--ablation-root", required=True,
                    help="Root directory containing model/effort/strategy subdirs")
    ap.add_argument("--articles-file", default="articles.jsonl",
                    help="Name of the articles file in each run dir")
    ap.add_argument("--output-dir", default="openLLMPedia/paper_results/ablation",
                    help="Where to write results")

    # ── Sampling ──
    ap.add_argument("--sample-n", type=int, default=100,
                    help="Number of random articles per seed")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                    help="Random seeds for repeated sampling")
    ap.add_argument("--min-words", type=int, default=100,
                    help="Minimum word count to include an article")

    # ── Factuality backend ──
    ap.add_argument("--fact-model-key", default="gpt-4.1-nano",
                    help="Model for claim verification")
    ap.add_argument("--evidence-sources", default="wikipedia,web",
                    help="Comma-separated evidence sources")
    ap.add_argument("--max-claims", type=int, default=10)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--web-mode",
                    choices=["snippets", "single", "hybrid", "full"],
                    default="snippets")
    ap.add_argument("--search-backend",
                    choices=["auto", "valyu", "serper", "brave", "ddg", "searxng"],
                    default="auto")
    ap.add_argument("--web-cache-dir", default="")
    ap.add_argument("--compute-similarity", action="store_true", default=False)
    ap.add_argument("--compute-bertscore", action="store_true", default=False)
    ap.add_argument("--compute-stylistic", action="store_true", default=False)

    # ── Control ──
    ap.add_argument("--dry-run", action="store_true",
                    help="Just discover runs and print config, no evaluation")
    ap.add_argument("--clean-output", action="store_true",
                    help="Delete output-dir before starting (fresh run)")
    ap.add_argument("--filter-model", default="",
                    help="Substring match on model key, e.g. 'gpt-5-mini' or 'nano'")
    ap.add_argument("--filter-strategy", default="",
                    help="Substring match on prompt strategy: 'baseline' or 'calibrated'")
    ap.add_argument("--filter-reasoning", default="",
                    help="Substring match on reasoning effort: 'low', 'min', etc.")
    ap.add_argument("--filter-selfrag", default=None, type=str,
                    choices=["true", "false"],
                    help="Filter by self-rag: 'true' or 'false'")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip runs that already have results in output dir")
    ap.add_argument("--audit-dir", default="")
    ap.add_argument("--debug", action="store_true")

    args = ap.parse_args()

    out_dir = os.path.abspath(args.output_dir)

    # Clean previous results if requested
    if args.clean_output:
        import shutil
        if os.path.isdir(out_dir):
            print(f"[clean] Removing {out_dir} ...")
            shutil.rmtree(out_dir)

    os.makedirs(out_dir, exist_ok=True)
    audit_root = args.audit_dir or os.path.join(out_dir, "audit")
    os.makedirs(audit_root, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  ABLATION FACTUALITY RUNNER")
    print(f"  root:       {args.ablation_root}")
    print(f"  sample_n:   {args.sample_n}")
    print(f"  seeds:      {args.seeds}")
    print(f"  evidence:   {args.evidence_sources}")
    print(f"  output:     {out_dir}")
    print(f"  dry_run:    {args.dry_run}")
    print(f"  clean:      {args.clean_output}")
    print(f"{'='*70}\n")

    # ── 1. Discover ──
    runs = discover_runs(args.ablation_root, args.articles_file)
    if not runs:
        print("[FATAL] No valid run directories found!")
        sys.exit(1)

    # Apply filters
    if args.filter_model:
        runs = [r for r in runs if args.filter_model in r.model]
        print(f"[filter] model='{args.filter_model}' → {len(runs)} runs")
    if args.filter_strategy:
        runs = [r for r in runs if args.filter_strategy in r.prompt_strategy]
        print(f"[filter] strategy='{args.filter_strategy}' → {len(runs)} runs")
    if args.filter_reasoning:
        runs = [r for r in runs if args.filter_reasoning in r.reasoning_effort]
        print(f"[filter] reasoning='{args.filter_reasoning}' → {len(runs)} runs")
    if args.filter_selfrag is not None:
        want_sg = args.filter_selfrag.lower() == "true"
        runs = [r for r in runs if r.self_rag == want_sg]
        print(f"[filter] selfrag={want_sg} → {len(runs)} runs")

    # Print discovery summary
    print(f"\n{'─'*60}")
    print(f"  DISCOVERED {len(runs)} RUNS:")
    print(f"{'─'*60}")
    for i, rc in enumerate(runs):
        print(f"  {i+1:2d}. {rc.display_name:<50s}  "
              f"({rc.n_subjects:>6,} articles)")
    print(f"{'─'*60}\n")

    if args.dry_run:
        print("[dry-run] Stopping here. Remove --dry-run to evaluate.")
        # Still write discovery JSON
        generate_json_report([], runs, args,
                             os.path.join(out_dir, "discovery.json"))
        return

    # ── 2. Evaluate ──
    if not HAS_FACTUALITY:
        print("[FATAL] factuality_core not available — cannot evaluate!")
        print("        Install it or run with --dry-run to just discover.")
        sys.exit(1)

    results = []
    for i, rc in enumerate(runs):
        print(f"\n{'═'*70}")
        print(f"  RUN {i+1}/{len(runs)}: {rc.display_name}")
        print(f"{'═'*70}")

        # Check skip-existing
        if args.skip_existing:
            safe_name = rc.display_name.replace("/", "_").replace(" ", "_")
            existing = os.path.join(out_dir, f"raw_{safe_name}_seed0.jsonl")
            if os.path.exists(existing):
                print(f"  [skip] Results exist: {existing}")
                continue

        try:
            result = run_factuality_for_run(rc, args, audit_root)
            results.append(result)
        except Exception as e:
            print(f"  [ERROR] {rc.display_name}: {e}")
            results.append(_empty_result(rc))

    # ── 3. Output ──
    print(f"\n{'═'*70}")
    print(f"  WRITING OUTPUTS")
    print(f"{'═'*70}")

    write_csv(results, os.path.join(out_dir, "ablation_results.csv"))
    generate_latex_table(results, os.path.join(out_dir, "ablation_table.tex"))
    generate_json_report(results, runs, args,
                         os.path.join(out_dir, "ablation_report.json"))

    # ── 4. Summary ──
    print(f"\n{'═'*70}")
    print(f"  SUMMARY")
    print(f"{'═'*70}")
    for r in results:
        prec = r.get("wiki_precision")
        prec_std = r.get("wiki_precision_std")
        false_r = r.get("wiki_false_rate")
        print(f"  {r['display_name']:<50s}  "
              f"Acc={_fmt(prec)}±{_fmt(prec_std)}  "
              f"F%={_fmt(false_r)}")
    print(f"{'═'*70}")
    print(f"  Output: {out_dir}")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
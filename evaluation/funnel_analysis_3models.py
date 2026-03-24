

#!/usr/bin/env python3
"""
funnel_analysis_3models.py

Wrapper around the ORIGINAL single-run funnel_analysis.py.

What it does
============
1. Accepts three model folders:
      --deepseek-dir
      --llama-dir
      --gpt-dir
2. Recursively finds every valid LLMPedia run directory under each one.
3. Runs the ORIGINAL funnel_analysis.py on every discovered run, writing outputs to:
      <output-dir>/<model>/<relative-run-path>/
   so nothing is written back into the original run folders.
4. Builds COMBINED outputs in:
      <output-dir>/combined/
   including grouped figures where each STAGE shows all three models together.

This preserves the old full per-run outputs while also giving you all-model comparison
figures in one place.

Example
=======
python evaluation/funnel_analysis_3models.py \
  --deepseek-dir openLLMPedia/deepseekV3.2_100K/ \
  --llama-dir openLLMPedia/llama3.3-70b_100K/ \
  --gpt-dir openLLMPedia/gpt_5_mini_1M/ \
  --output-dir evaluation/funnel_combined \
  --single-run-script evaluation/funnel_analysis.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from typing import Any, Dict, List

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

MODEL_ORDER = ["deepseek", "gpt-5-mini", "llama"]
MODEL_DISPLAY = {
    "deepseek": "DeepSeek",
    "gpt-5-mini": "GPT-5-mini",
    "llama": "Llama",
}
MODEL_COLORS = {
    "deepseek": "#0072B2",
    "gpt-5-mini": "#009E73",
    "llama": "#E69F00",
}


# ============================================================
# Helpers
# ============================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_json(path: str, default: Any) -> Any:
    if not path or not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_csv(path: str, rows: List[dict]):
    ensure_dir(os.path.dirname(path))
    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(["empty"])
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def rate(n: int, d: int) -> float:
    return float(n) / float(d) if d else 0.0


def looks_like_run_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    has_queue = os.path.exists(os.path.join(path, "queue.json")) or os.path.exists(os.path.join(path, "queue.jsonl"))
    has_articles = os.path.exists(os.path.join(path, "articles_wikitext.jsonl"))
    return has_queue and has_articles


def find_run_dirs(root: str) -> List[str]:
    root = os.path.abspath(root)
    if looks_like_run_dir(root):
        return [root]
    found: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if looks_like_run_dir(dirpath):
            found.append(os.path.abspath(dirpath))
            dirnames[:] = []
    return sorted(found)


def sanitize_relpath(path: str) -> str:
    path = path.strip().strip(os.sep)
    return path.replace(os.sep, "__") if path else "root_run"


def run_meta_info(run_dir: str) -> dict:
    meta = read_json(os.path.join(run_dir, "run_meta.json"), {})
    casc = meta.get("cascading_defaults") or {}
    args = meta.get("args_raw") or {}
    personas = meta.get("personas") or {}
    return {
        "topic": str(meta.get("seed") or args.get("topic") or ""),
        "persona": str(personas.get("elicit") or args.get("persona") or ""),
        "model_key": str(args.get("elicit_model_key") or casc.get("global_model_key") or args.get("model_key") or ""),
    }


# def _candidates_total(row: dict) -> int:
#     """
#     Resolve the 'candidates generated' count from a run or aggregate row.
#     The single-run overall_summary uses 'ner_candidates'; aggregated rows
#     store it as 'candidates_total'.  Fall back gracefully so both work.
#     """
#     return int(
#         row.get("candidates_total")
#         or row.get("ner_candidates")
#         or 0
#     )
def _candidates_total(row: dict) -> int:

    """

    Resolve the candidates generated count from a run or aggregate row.

    Prioritize ner_candidates_processed to exclude pending backlog entities.

    """

    return int(

        row.get("candidates_total")

        or row.get("ner_candidates_processed")

        or row.get("ner_candidates")

        or 0

    )



def _savefig_variants(fig_factory, base_path: str) -> List[str]:
    written = []
    fig_w = fig_factory(transparent=False)
    p_w = base_path + "_white.png"
    fig_w.savefig(p_w, dpi=300, bbox_inches="tight", facecolor="white", transparent=False)
    plt.close(fig_w)
    written.append(p_w)
    print(f"  saved: {p_w}", flush=True)

    fig_t = fig_factory(transparent=True)
    p_t = base_path + "_transp.png"
    fig_t.savefig(p_t, dpi=300, bbox_inches="tight", facecolor="none", transparent=True)
    plt.close(fig_t)
    written.append(p_t)
    print(f"  saved: {p_t}", flush=True)
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


# ============================================================
# Run original single-run analyzer
# ============================================================
def run_single_analysis(single_run_script: str, run_dir: str, out_dir: str, short: bool = False):
    cmd = [sys.executable, single_run_script, "--run-dir", run_dir, "--out-dir", out_dir]
    if short:
        cmd.append("--short")
    print("\n" + "-" * 80)
    print(f"RUN  : {run_dir}")
    print(f"OUT  : {out_dir}")
    print(f"SCRIPT: {single_run_script}")
    print("-" * 80)
    subprocess.run(cmd, check=True)


def load_single_output(out_dir: str) -> dict:
    path = os.path.join(out_dir, "funnel_summary.json")
    obj = read_json(path, None)
    if not obj:
        raise RuntimeError(f"Missing or bad funnel_summary.json: {path}")
    if "overall_summary" not in obj or "hop_summary" not in obj:
        raise RuntimeError(f"Malformed funnel_summary.json: {path}")
    return obj


# ============================================================
# Aggregation
# ============================================================
def aggregate_model_rows(run_rows: List[dict]) -> List[dict]:
    grouped = defaultdict(list)
    for row in run_rows:
        grouped[row["model"]].append(row)

    out = []
    for model in MODEL_ORDER:
        rows = grouped.get(model, [])
        if not rows:
            continue
        # Use helper so both 'ner_candidates' and 'candidates_total' keys work
        cand   = sum(_candidates_total(r) for r in rows)
        canon  = sum(r.get("canonical_kept", 0) for r in rows)
        ner    = sum(r.get("ner_accepted", 0) for r in rows)
        sim    = sum(r.get("similarity_accepted", 0) for r in rows)
        ins    = sum(r.get("inserted_children", 0) for r in rows)
        out.append({
            "model": model,
            "display_name": MODEL_DISPLAY[model],
            "n_runs": len(rows),
            "entities_total": sum(r.get("entities_total", 0) for r in rows),
            "candidates_total": cand,
            "canonical_kept": canon,
            "canonical_rejected_total": sum(r.get("canonical_rejected_total", 0) for r in rows),
            "ner_accepted": ner,
            "ner_rejected": sum(r.get("ner_rejected", 0) for r in rows),
            "similarity_accepted": sim,
            "similarity_rejected": sum(r.get("similarity_rejected", 0) for r in rows),
            "similarity_skipped": sum(r.get("similarity_skipped", 0) for r in rows),
            "inserted_children": ins,
            "canonical_survival": rate(canon, cand),
            "ner_survival": rate(ner, canon),
            "similarity_survival": rate(sim, ner),
            "overall_survival_to_inserted": rate(ins, cand),
        })
    return out


def aggregate_hop_rows(hop_rows_all: List[dict]) -> List[dict]:
    grouped = defaultdict(list)
    for row in hop_rows_all:
        grouped[(row["model"], row["hop"])].append(row)

    out = []
    for model in MODEL_ORDER:
        hops = sorted({h for (m, h) in grouped if m == model})
        for hop in hops:
            rows = grouped[(model, hop)]
            cand  = sum(r.get("canonical_pre_ner_input", 0) for r in rows)
            canon = sum(r.get("canonical_kept", 0) for r in rows)
            ner   = sum(r.get("ner_accepted", 0) for r in rows)
            sim   = sum(r.get("similarity_accepted", 0) for r in rows)
            ins   = sum(r.get("inserted_children", 0) for r in rows)
            out.append({
                "model": model,
                "display_name": MODEL_DISPLAY[model],
                "hop": hop,
                "n_runs": len(rows),
                "entities": sum(r.get("entities", 0) for r in rows),
                "canonical_pre_ner_input": cand,
                "canonical_kept": canon,
                "canonical_rejected_total": sum(r.get("canonical_rejected_total", 0) for r in rows),
                "ner_accepted": ner,
                "ner_rejected": sum(r.get("ner_rejected", 0) for r in rows),
                "similarity_accepted": sim,
                "similarity_rejected": sum(r.get("similarity_rejected", 0) for r in rows),
                "similarity_skipped": sum(r.get("similarity_skipped", 0) for r in rows),
                "inserted_children": ins,
                "canonical_survival": rate(canon, cand),
                "ner_survival": rate(ner, canon),
                "similarity_survival": rate(sim, ner),
                "overall_survival_to_inserted": rate(ins, cand),
            })
    return out


# ============================================================
# Combined report
# ============================================================
def render_combined_report(run_rows: List[dict], model_rows: List[dict], hop_rows: List[dict]) -> str:
    lines: List[str] = []
    lines.append("=" * 86)
    lines.append("  LLMPedia 3-Model Combined Funnel Report")
    lines.append("=" * 86)
    lines.append("")

    lines.append("Per-run outputs")
    lines.append("-" * 86)
    for r in run_rows:
        # Use helper: per-run rows carry 'ner_candidates' from overall_summary
        cand = _candidates_total(r)
        lines.append(
            f"  {r['model']:<12}  {r['run_name']:<28}  cand={cand:>10,}  "
            f"canon={r.get('canonical_kept', 0):>10,}  ner={r.get('ner_accepted', 0):>10,}  "
            f"sim={r.get('similarity_accepted', 0):>10,}  inserted={r.get('inserted_children', 0):>10,}"
        )
    lines.append("")

    lines.append("Per-model totals")
    lines.append("-" * 86)
    for r in model_rows:
        lines.append(
            f"  {r['display_name']:<12}  runs={r['n_runs']:>4}  ents={r['entities_total']:>10,}  "
            f"cand={r['candidates_total']:>10,}  canon={r['canonical_kept']:>10,}  "
            f"ner={r['ner_accepted']:>10,}  sim={r['similarity_accepted']:>10,}  inserted={r['inserted_children']:>10,}"
        )
    lines.append("")

    lines.append("Per-model survival rates")
    lines.append("-" * 86)
    for r in model_rows:
        lines.append(
            f"  {r['display_name']:<12}  canonical={r['canonical_survival']:.2%}  "
            f"ner={r['ner_survival']:.2%}  similarity={r['similarity_survival']:.2%}  "
            f"overall={r['overall_survival_to_inserted']:.2%}"
        )
    lines.append("")

    lines.append("Per-hop totals by model")
    lines.append("-" * 86)
    for r in hop_rows:
        lines.append(
            f"  {r['display_name']:<12}  hop={r['hop']:>2}  cand={r['canonical_pre_ner_input']:>10,}  "
            f"canon={r['canonical_kept']:>10,}  ner={r['ner_accepted']:>10,}  sim={r['similarity_accepted']:>10,}  inserted={r['inserted_children']:>10,}"
        )
    lines.append("")
    lines.append("=" * 86)
    return "\n".join(lines)


# ============================================================
# Combined figures
# ============================================================
def render_combined_figures(model_rows: List[dict], hop_rows: List[dict], figures_dir: str) -> List[str]:
    if not _HAS_MPL:
        print("[combined figures] matplotlib not installed — skipping.", flush=True)
        return []

    ensure_dir(figures_dir)
    written: List[str] = []
    model_rows = sorted(model_rows, key=lambda r: MODEL_ORDER.index(r["model"]))
    if not model_rows:
        return written

    models = [r["model"] for r in model_rows]

    # ── shared helpers ───────────────────────────────────────────
    def _fmt_count(v: float) -> str:
        """Compact human-readable label: 1.2M, 345K, 12,345."""
        if v >= 1_000_000:
            return f"{v/1_000_000:.1f}M"
        if v >= 1_000:
            return f"{v/1_000:.0f}K"
        return f"{int(v):,}"

    def _label_bars(ax, bars, values, fmt_fn, pad_frac=0.015):
        """Place a label just above each bar."""
        max_v = max((v for v in values if v > 0), default=1)
        for bar, v in zip(bars, values):
            if v <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max_v * pad_frac,
                fmt_fn(v),
                ha="center", va="bottom",
                fontsize=7.5, fontweight="bold", color="#222222",
            )

    def _label_points(ax, xs, ys, fmt_fn, color):
        """Queue labels for deferred overlap-aware placement. Call _flush_labels(ax) after all series."""
        if not hasattr(ax, "_pending_labels"):
            ax._pending_labels = []
        for x, y in zip(xs, ys):
            if y <= 0:
                continue
            ax._pending_labels.append((float(x), float(y), fmt_fn(y), color))

    def _flush_labels(ax, font_size=7.5, min_gap_frac=0.045):
        """
        Place all pending labels with vertical nudging to prevent collisions.

        Per x-position:
          1. Sort labels by y value ascending.
          2. Walk bottom-to-top; push each label up if it overlaps the one below.
          3. Draw a thin leader line when the label has been moved away from its point.
        """
        if not hasattr(ax, "_pending_labels") or not ax._pending_labels:
            return
        from collections import defaultdict as _dd2
        by_x = _dd2(list)
        for item in ax._pending_labels:
            by_x[item[0]].append(item)

        y_lo, y_hi = ax.get_ylim()
        min_gap = max(y_hi - y_lo, 1e-9) * min_gap_frac

        for x_pos, items in by_x.items():
            items_sorted = sorted(items, key=lambda t: t[1])
            placed: list = []
            for (x, y, text, color) in items_sorted:
                label_y = y + min_gap * 0.5
                for prev_y in placed:
                    if abs(label_y - prev_y) < min_gap:
                        label_y = prev_y + min_gap
                placed.append(label_y)
                moved = abs(label_y - y) > min_gap * 0.7
                ax.annotate(
                    text,
                    xy=(x, y),
                    xytext=(x, label_y),
                    xycoords="data",
                    textcoords="data",
                    ha="center", va="bottom",
                    fontsize=font_size, fontweight="bold", color=color,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=0.5)
                    if moved else None,
                )
        ax._pending_labels = []

    # Figure 01: X-axis = stages, bars inside each stage = models
    stages = [
        ("Generated",  "candidates_total"),
        ("Post-dedup", "canonical_kept"),
        ("Post-NER",   "ner_accepted"),
        ("Post-sim",   "similarity_accepted"),
        ("Inserted",   "inserted_children"),
    ]

    def _fig01(transparent=False):
        fc = "none" if transparent else "white"
        fig, ax = plt.subplots(figsize=(13, 6.2), facecolor=fc)
        xs = np.arange(len(stages))
        w = 0.22
        all_vals = []
        for i, model in enumerate(models):
            row = next(r for r in model_rows if r["model"] == model)
            ys = [row.get(key, 0) for _label, key in stages]
            all_vals.extend(ys)
            bars = ax.bar(
                xs + (i - 1) * w, ys, width=w,
                label=MODEL_DISPLAY[model], color=MODEL_COLORS[model],
                edgecolor="white", linewidth=0.6,
            )
            _label_bars(ax, bars, ys, _fmt_count)
        # extra headroom so labels don't clip
        ax.set_ylim(0, max(all_vals, default=1) * 1.18)
        ax.set_xticks(xs)
        ax.set_xticklabels([s[0] for s in stages])
        ax.set_ylabel("Candidates")
        ax.set_title("Grouped Funnel Counts by Stage — All Models Together")
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: _fmt_count(v))
        )
        ax.legend()
        _pub_style(ax, transparent=transparent)
        fig.tight_layout()
        return fig
    written.extend(_savefig_variants(_fig01, os.path.join(figures_dir, "01_grouped_stage_counts")))

    # Figure 02: X-axis = survival stages, bars inside each stage = models
    surv_stages = [
        ("Canonical",  "canonical_survival"),
        ("NER",        "ner_survival"),
        ("Similarity", "similarity_survival"),
        ("Overall",    "overall_survival_to_inserted"),
    ]

    def _fig02(transparent=False):
        fc = "none" if transparent else "white"
        fig, ax = plt.subplots(figsize=(10.5, 6.2), facecolor=fc)
        xs = np.arange(len(surv_stages))
        w = 0.22
        for i, model in enumerate(models):
            row = next(r for r in model_rows if r["model"] == model)
            ys = [100.0 * row.get(key, 0.0) for _label, key in surv_stages]
            bars = ax.bar(
                xs + (i - 1) * w, ys, width=w,
                label=MODEL_DISPLAY[model], color=MODEL_COLORS[model],
                edgecolor="white", linewidth=0.6,
            )
            _label_bars(ax, bars, ys, lambda v: f"{v:.1f}%")
        ax.set_ylim(0, 120)
        ax.set_xticks(xs)
        ax.set_xticklabels([s[0] for s in surv_stages])
        ax.set_ylabel("Rate (%)")
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%")
        )
        ax.set_title("Grouped Survival Rates by Stage — All Models Together")
        ax.legend()
        _pub_style(ax, transparent=transparent)
        fig.tight_layout()
        return fig
    written.extend(_savefig_variants(_fig02, os.path.join(figures_dir, "02_grouped_stage_survival")))

    # organise hop rows
    by_model_hop: Dict[str, Dict[int, dict]] = defaultdict(dict)
    all_hops = sorted({r["hop"] for r in hop_rows})
    for r in hop_rows:
        by_model_hop[r["model"]][r["hop"]] = r
    xh = np.arange(len(all_hops))

    # Figure 03: inserted by hop, one line per model
    def _fig03(transparent=False):
        fc = "none" if transparent else "white"
        fig, ax = plt.subplots(figsize=(10.5, 5.8), facecolor=fc)
        for model in models:
            ys = [by_model_hop[model].get(h, {}).get("inserted_children", 0) for h in all_hops]
            ax.plot(xh, ys, marker="o", linewidth=2.0, label=MODEL_DISPLAY[model], color=MODEL_COLORS[model])
            _label_points(ax, xh, ys, _fmt_count, MODEL_COLORS[model])
        ax.set_xticks(xh)
        ax.set_xticklabels([f"Hop {h}" for h in all_hops])
        ax.set_ylabel("Inserted children")
        ax.set_title("Inserted Children by Hop — All Models")
        cur_top = ax.get_ylim()[1]
        ax.set_ylim(0, cur_top * 1.22)
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: _fmt_count(v))
        )
        ax.legend()
        _pub_style(ax, transparent=transparent)
        _flush_labels(ax)
        fig.tight_layout()
        return fig
    written.extend(_savefig_variants(_fig03, os.path.join(figures_dir, "03_inserted_by_hop")))

    # Figure 04: generated candidates by hop, one line per model
    def _fig04(transparent=False):
        fc = "none" if transparent else "white"
        fig, ax = plt.subplots(figsize=(10.5, 5.8), facecolor=fc)
        for model in models:
            ys = [by_model_hop[model].get(h, {}).get("canonical_pre_ner_input", 0) for h in all_hops]
            ax.plot(xh, ys, marker="o", linewidth=2.0, label=MODEL_DISPLAY[model], color=MODEL_COLORS[model])
            _label_points(ax, xh, ys, _fmt_count, MODEL_COLORS[model])
        ax.set_xticks(xh)
        ax.set_xticklabels([f"Hop {h}" for h in all_hops])
        ax.set_ylabel("Candidates")
        ax.set_title("Generated Candidates by Hop — All Models")
        cur_top = ax.get_ylim()[1]
        ax.set_ylim(0, cur_top * 1.22)
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: _fmt_count(v))
        )
        ax.legend()
        _pub_style(ax, transparent=transparent)
        _flush_labels(ax)
        fig.tight_layout()
        return fig
    written.extend(_savefig_variants(_fig04, os.path.join(figures_dir, "04_generated_by_hop")))

    # Figure 05: overall survival by hop, one line per model
    def _fig05(transparent=False):
        fc = "none" if transparent else "white"
        fig, ax = plt.subplots(figsize=(10.5, 5.8), facecolor=fc)
        for model in models:
            ys = [100.0 * (by_model_hop[model].get(h, {}).get("overall_survival_to_inserted") or 0.0) for h in all_hops]
            ax.plot(xh, ys, marker="o", linewidth=2.0, label=MODEL_DISPLAY[model], color=MODEL_COLORS[model])
            _label_points(ax, xh, ys, lambda v: f"{v:.1f}%", MODEL_COLORS[model])
        ax.set_xticks(xh)
        ax.set_xticklabels([f"Hop {h}" for h in all_hops])
        ax.set_ylabel("Overall survival (%)")
        ax.set_title("Overall Survival to Inserted by Hop — All Models")
        ax.set_ylim(-2, 115)
        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%")
        )
        ax.legend()
        _pub_style(ax, transparent=transparent)
        _flush_labels(ax)
        fig.tight_layout()
        return fig
    written.extend(_savefig_variants(_fig05, os.path.join(figures_dir, "05_overall_survival_by_hop")))

    return written


# ============================================================
# Main pipeline
# ============================================================
def analyze_three_models(
    deepseek_dir: str,
    llama_dir: str,
    gpt_dir: str,
    output_dir: str,
    single_run_script: str,
    short: bool = False,
):
    model_roots = {
        "deepseek":   os.path.abspath(deepseek_dir),
        "gpt-5-mini": os.path.abspath(gpt_dir),
        "llama":      os.path.abspath(llama_dir),
    }
    output_dir         = os.path.abspath(output_dir)
    single_run_script  = os.path.abspath(single_run_script)

    if not os.path.isfile(single_run_script):
        raise SystemExit(f"single-run script not found: {single_run_script}")

    discovered: Dict[str, List[str]] = {}
    for model, root in model_roots.items():
        if not os.path.isdir(root):
            raise SystemExit(f"{model} directory not found: {root}")
        runs = find_run_dirs(root)
        if not runs:
            raise SystemExit(f"No valid run dirs found under {root}")
        discovered[model] = runs

    print("\n" + "=" * 86)
    print("DISCOVERED RUNS")
    print("=" * 86)
    for model in MODEL_ORDER:
        print(f"{model:<12}: {len(discovered[model])} run(s)")
        for r in discovered[model]:
            print(f"    {r}")

    ensure_dir(output_dir)

    run_rows: List[dict]      = []
    hop_rows_all: List[dict]  = []

    for model in MODEL_ORDER:
        root           = model_roots[model]
        model_out_root = os.path.join(output_dir, model)
        ensure_dir(model_out_root)

        for run_dir in discovered[model]:
            rel      = os.path.relpath(run_dir, root)
            run_name = sanitize_relpath(rel)
            out_dir  = os.path.join(model_out_root, run_name)

            run_single_analysis(single_run_script, run_dir, out_dir, short=short)
            loaded   = load_single_output(out_dir)
            overall  = dict(loaded.get("overall_summary") or {})
            hop_rows = list(loaded.get("hop_summary") or [])
            meta     = run_meta_info(run_dir)

            run_row = {
                "model":        model,
                "display_name": MODEL_DISPLAY[model],
                "run_dir":      run_dir,
                "run_name":     run_name,
                "topic":        meta.get("topic", ""),
                "persona":      meta.get("persona", ""),
                **overall,
            }
            run_rows.append(run_row)

            for hr in hop_rows:
                hop_rows_all.append({
                    "model":        model,
                    "display_name": MODEL_DISPLAY[model],
                    "run_dir":      run_dir,
                    "run_name":     run_name,
                    "topic":        meta.get("topic", ""),
                    "persona":      meta.get("persona", ""),
                    **hr,
                })

    combined_dir = os.path.join(output_dir, "combined")
    ensure_dir(combined_dir)

    model_rows     = aggregate_model_rows(run_rows)
    hop_model_rows = aggregate_hop_rows(hop_rows_all)

    write_csv(os.path.join(combined_dir, "combined_runs.csv"),   run_rows)
    write_csv(os.path.join(combined_dir, "combined_models.csv"), model_rows)
    write_csv(os.path.join(combined_dir, "combined_hops.csv"),   hop_model_rows)

    summary = {
        "run_rows":   run_rows,
        "model_rows": model_rows,
        "hop_rows":   hop_model_rows,
    }
    with open(os.path.join(combined_dir, "combined_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    report = render_combined_report(run_rows, model_rows, hop_model_rows)
    with open(os.path.join(combined_dir, "combined_report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print("\n" + report)

    figures_dir = os.path.join(combined_dir, "figures")
    fig_paths   = render_combined_figures(model_rows, hop_model_rows, figures_dir)
    if fig_paths:
        print(f"\n[combined figures] {len(fig_paths)} file(s) written to: {figures_dir}")

    with open(os.path.join(combined_dir, "README.txt"), "w", encoding="utf-8") as f:
        f.write(
            "LLMPedia 3-model combined funnel analysis\n"
            f"Source DeepSeek root : {model_roots['deepseek']}\n"
            f"Source GPT root      : {model_roots['gpt-5-mini']}\n"
            f"Source Llama root    : {model_roots['llama']}\n"
            f"Single-run script    : {single_run_script}\n"
            f"Output root          : {output_dir}\n\n"
            "Per-run outputs:\n"
            "  <output-dir>/deepseek/...\n"
            "  <output-dir>/gpt-5-mini/...\n"
            "  <output-dir>/llama/...\n\n"
            "Combined outputs:\n"
            "  <output-dir>/combined/combined_runs.csv\n"
            "  <output-dir>/combined/combined_models.csv\n"
            "  <output-dir>/combined/combined_hops.csv\n"
            "  <output-dir>/combined/combined_summary.json\n"
            "  <output-dir>/combined/combined_report.txt\n"
            "  <output-dir>/combined/figures/\n\n"
            "Grouped figures use STAGES on the x-axis and the three models grouped inside each stage.\n"
        )

    print(f"\nDone. Combined outputs written to: {combined_dir}")


# ============================================================
# CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="Run original funnel_analysis.py across DeepSeek / GPT / Llama and build combined grouped outputs."
    )
    ap.add_argument("--deepseek-dir", required=True)
    ap.add_argument("--llama-dir",    required=True)
    ap.add_argument("--gpt-dir",      required=True)
    ap.add_argument("--output-dir",   required=True)
    ap.add_argument(
        "--single-run-script",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "funnel_analysis.py"),
        help="Path to the ORIGINAL single-run funnel_analysis.py",
    )
    ap.add_argument("--short",  action="store_true", help="Pass --short to the original single-run script.")
    ap.add_argument("--clean",  action="store_true", help="Delete output-dir and exit.")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    if args.clean:
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
            print(f"Removed: {out_dir}")
        else:
            print(f"Nothing to clean: {out_dir}")
        return

    analyze_three_models(
        deepseek_dir=args.deepseek_dir,
        llama_dir=args.llama_dir,
        gpt_dir=args.gpt_dir,
        output_dir=args.output_dir,
        single_run_script=args.single_run_script,
        short=args.short,
    )


if __name__ == "__main__":
    main()



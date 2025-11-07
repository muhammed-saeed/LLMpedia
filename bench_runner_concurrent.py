#!/usr/bin/env python3
# bench_runner_concurrent.py
from __future__ import annotations
import argparse
import csv
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

# ===================== small utils =====================

def ts_for_dir() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def sanitize_slug(s: str) -> str:
    bad = '/\\?%*:|"<>'
    out = s.strip().replace(" ", "_")
    for ch in bad:
        out = out.replace(ch, "")
    return out

def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def write_json(path: str, obj: dict) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def append_bench_log(root_out: str, line: str) -> None:
    ensure_dir(root_out)
    with open(os.path.join(root_out, "bench.log"), "a", encoding="utf-8") as f:
        f.write(f"[{dt.datetime.now().isoformat()}] {line}\n")

def expand_csv_header_safely(csv_path: str, new_row: Dict[str, object]) -> None:
    """
    Append a row to CSV while allowing new columns to appear later.
    If the header needs to grow, rewrite file with expanded header.
    """
    ensure_dir(str(Path(csv_path).parent))
    rows: List[Dict[str, object]] = []
    existing_header: List[str] = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            existing_header = r.fieldnames or []
            for row in r:
                rows.append(row)

    all_keys = list(dict.fromkeys([*(existing_header or []), *list(new_row.keys())]))
    rows.append(new_row)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        for r in rows:
            out = {k: r.get(k, "") for k in all_keys}
            w.writerow(out)

# ===================== profiles =====================

PROFILE_KNOBS = {
    "det":    {"temperature": 0.0, "top_p": 1.0,  "top_k": None, "max_tokens": 4096},
    "medium": {"temperature": 0.7, "top_p": 0.95, "top_k": 50,   "max_tokens": 4096},
    "wild":   {"temperature": 2.0, "top_p": 1.0,  "top_k": 100,  "max_tokens": 4096},
}

# ===================== args =====================

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Concurrent benchmark runner for GPT-KB crawler (outer parallelism + per-run routing)."
    )
    ap.add_argument("--root-out", required=True,
                    help="Root folder for all benchmark outputs (subfolders will be created).")
    ap.add_argument("--crawler", default="crawler_batch_concurrency_topic.py",
                    help="Crawler script to run (default: crawler_batch_concurrency_topic.py).")

    # grids
    ap.add_argument("--domains", default="topic,general",
                    help="Comma list: topic,general")
    ap.add_argument("--seeds", default="Game of Thrones,Lionel Messi,World War II",
                    help="Comma list of starting subjects.")
    ap.add_argument("--models", default="deepseek,granite8b,gpt4o-mini",
                    help="Comma list of model keys (must exist in settings.MODELS).")
    ap.add_argument("--strategies", default="baseline,calibrate,icl,dont_know",
                    help="Comma list of elicitation strategies.")
    ap.add_argument("--profiles", default="det,medium,wild",
                    help="Comma list of sampling profiles (det|medium|wild).")

    # crawler knobs (shared)
    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument("--max-subjects", type=int, default=3,
                    help="Hard cap of subjects per run; 0 means 'no cap' (crawler drains by hop).")
    ap.add_argument("--max-facts-hint", type=int, default=100)
    ap.add_argument("--ner-batch-size", type=int, default=50)
    ap.add_argument("--concurrency", type=int, default=10,
                    help="(legacy fallback) Per-run thread concurrency if default-concurrency not given.")
    ap.add_argument("--ner-strategy", default="calibrate",
                    help="NER strategy passed to crawler (often 'calibrate').")

    # OpenAI batch vs concurrency routing
    ap.add_argument("--openai-batch-size", type=int, default=None,
                    help="If set and model is OpenAI, pass --openai-batch and this size to the crawler.")
    ap.add_argument("--default-concurrency", type=int, default=10,
                    help="Per-run concurrency for non-OpenAI (and OpenAI without batch).")

    # NETWORK ROBUSTNESS (new)
    ap.add_argument("--net-timeout", type=float, default=60.0,
                    help="HTTP connect/read timeout in seconds (forwarded to crawler as --http-timeout and NET_TIMEOUT).")
    ap.add_argument("--net-retries", type=int, default=6,
                    help="HTTP retry attempts on transient errors (forwarded as --http-retries and NET_RETRIES).")
    ap.add_argument("--net-backoff", type=float, default=0.5,
                    help="Exponential backoff factor between retries (forwarded as --http-backoff and NET_BACKOFF).")

    # outer parallelism
    ap.add_argument("--max-procs", type=int, default=1,
                    help="How many crawler runs to execute in parallel (outer level).")

    # control / safety
    ap.add_argument("--list", action="store_true",
                    help="Only list planned runs then exit (no writes).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Plan and write meta/CSV, but do NOT execute the crawler.")
    ap.add_argument("--verbose", action="store_true", help="Verbose planning output.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="If out_dir already exists, skip planning/execution for that run.")

    return ap

# ===================== planning =====================

def is_openai_model(model_key: str) -> bool:
    key = (model_key or "").lower()
    # Adjust as needed to match your settings.MODELS keys for OpenAI
    return key in ("gpt4o-mini", "gpt-4o-mini", "gpt4o", "gpt-4o", "o3-mini", "o4-mini")

def build_plan(args) -> List[Dict]:
    # normalize grids
    domains    = [s.strip() for s in args.domains.split(",") if s.strip()]
    seeds      = [s.strip() for s in args.seeds.split(",") if s.strip()]
    models     = [s.strip() for s in args.models.split(",") if s.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    profiles   = [s.strip() for s in args.profiles.split(",") if s.strip()]

    # sanity: profiles exist
    for p in profiles:
        if p not in PROFILE_KNOBS:
            raise SystemExit(f"Unknown profile '{p}'. Use one of: {', '.join(PROFILE_KNOBS)}")

    plan: List[Dict] = []
    seen = set()  # prevent duplicates

    for domain, model, strat, prof, seed in product(domains, models, strategies, profiles, seeds):
        k = (domain, model, strat, prof, seed)
        if k in seen:
            continue
        seen.add(k)

        seed_slug = sanitize_slug(seed)
        out_dir = os.path.join(
            args.root_out,
            domain,
            model,
            strat,
            prof,
            seed_slug,
            ts_for_dir(),
        )

        if args.skip_existing and os.path.exists(out_dir):
            if args.verbose:
                print(f"[bench] SKIP (exists): {out_dir}")
            continue

        # base crawler command
        cmd: List[str] = [
            sys.executable, args.crawler,
            "--seed", seed,
            "--output-dir", out_dir,
            "--domain", domain,
            "--elicitation-strategy", strat,
            "--ner-strategy", args.ner_strategy,
            "--elicit-model-key", model,
            "--ner-model-key", model,
            "--max-depth", str(args.max_depth),
            "--max-facts-hint", str(args.max_facts_hint),
            "--max-subjects", str(args.max_subjects),
            "--ner-batch-size", str(args.ner_batch_size),
        ]

        # decide concurrency vs openai-batch passthrough
        batch_mode = False
        effective_conc = None
        if is_openai_model(model) and args.openai_batch_size:
            cmd += ["--openai-batch", "--openai-batch-size", str(args.openai_batch_size)]
            batch_mode = True
        else:
            effective_conc = args.default_concurrency or args.concurrency or 10
            cmd += ["--concurrency", str(effective_conc)]

        # sampling knobs from profile
        knobs = PROFILE_KNOBS[prof]
        if knobs.get("temperature") is not None:
            cmd += ["--temperature", str(knobs["temperature"])]
        if knobs.get("top_p") is not None:
            cmd += ["--top-p", str(knobs["top_p"])]
        if knobs.get("top_k") is not None:
            cmd += ["--top-k", str(knobs["top_k"])]
        if knobs.get("max_tokens") is not None:
            cmd += ["--max-tokens", str(knobs["max_tokens"])]

        # NEW: pass network robustness knobs as flags too
        cmd += [
            "--http-timeout", str(args.net_timeout),
            "--http-retries", str(args.net_retries),
            "--http-backoff", str(args.net_backoff),
        ]

        meta = {
            "seed": seed,
            "seed_slug": seed_slug,
            "domain": domain,
            "elicitation_strategy": strat,
            "ner_strategy": args.ner_strategy,
            "model": model,
            "out_dir": out_dir,
            "profile": prof,
            "profile_knobs": knobs,
            "max_depth": args.max_depth,
            "max_subjects": args.max_subjects,
            "max_facts_hint": args.max_facts_hint,
            "ner_batch_size": args.ner_batch_size,
            "crawler": args.crawler,
            "python": sys.executable,
            "command": " ".join(shlex.quote(c) for c in cmd),
            "timestamp": dt.datetime.now().isoformat(),
            "batch_mode": batch_mode,
            "effective_concurrency": effective_conc,
            # expose net knobs in meta (also used for env passing)
            "net_timeout": args.net_timeout,
            "net_retries": args.net_retries,
            "net_backoff": args.net_backoff,
        }

        plan.append({"cmd": cmd, "out_dir": out_dir, "meta": meta})

    return plan

# ===================== execution helpers =====================

def run_one(job: Dict, csv_path: str) -> Tuple[str, int]:
    """
    Execute a single crawler job (subprocess). Returns (out_dir, returncode).
    Also appends a CSV row with status (OK/RC_x).
    """
    cmd = job["cmd"]
    out_dir = job["out_dir"]
    meta = job["meta"]

    # write per-run meta.json before executing
    write_json(os.path.join(out_dir, "meta.json"), meta)

    rc = 0
    try:
        # Pass network knobs via env as a fallback for crawlers that read env vars
        env = os.environ.copy()
        env["NET_TIMEOUT"] = str(meta.get("net_timeout", 60))
        env["NET_RETRIES"] = str(meta.get("net_retries", 6))
        env["NET_BACKOFF"] = str(meta.get("net_backoff", 0.5))
        rc = subprocess.run(cmd, check=False, env=env).returncode
    except Exception:
        rc = -1

    # append CSV row with outcome
    csv_row = {
        "status": "OK" if rc == 0 else f"RC_{rc}",
        **{k: v for k, v in meta.items() if not isinstance(v, dict)}
    }
    expand_csv_header_safely(csv_path, csv_row)

    # tiny done marker
    write_json(os.path.join(out_dir, "done.json"), {"returncode": rc})

    return out_dir, rc

# ===================== main =====================

def main():
    args = build_arg_parser().parse_args()

    print("[bench] START", flush=True)
    print(f"[bench] root_out={args.root_out}", flush=True)
    print(f"[bench] crawler={args.crawler}", flush=True)

    if not os.path.exists(args.crawler):
        print(f"[bench][ERROR] crawler not found: {args.crawler}", flush=True)
        sys.exit(2)

    plan = build_plan(args)
    print(f"[bench] total_planned={len(plan)}", flush=True)

    if args.verbose:
        for i, job in enumerate(plan[:min(12, len(plan))]):
            m = job["meta"]
            print(f"  plan[{i}] domain={m['domain']} model={m['model']} seed={m['seed']} "
                  f"strategy={m['elicitation_strategy']} profile={m['profile']} "
                  f"batch={m['batch_mode']} conc={m['effective_concurrency']} → {m['out_dir']}", flush=True)

    append_bench_log(args.root_out, f"planned={len(plan)}")

    if not plan:
        print("[bench][FATAL] No runs planned. Check your grids (--domains/--seeds/--models/--strategies/--profiles).", flush=True)
        sys.exit(1)

    if args.list:
        print("[bench] --list set; not executing.", flush=True)
        return

    csv_path = os.path.join(args.root_out, "runs.csv")

    if args.dry_run:
        # Write meta + CSV rows without executing the crawler
        for job in plan:
            out_dir = job["out_dir"]
            meta = job["meta"]
            if args.skip_existing and os.path.exists(out_dir):
                print(f"[bench][DRY] SKIP (exists): {out_dir}", flush=True)
                continue
            write_json(os.path.join(out_dir, "meta.json"), meta)
            csv_row = {"status": "DRY_RUN", **{k: v for k, v in meta.items() if not isinstance(v, dict)}}
            expand_csv_header_safely(csv_path, csv_row)
            write_json(os.path.join(out_dir, "done.json"), {"returncode": None, "dry_run": True})
        print("[bench] DRY-RUN complete.", flush=True)
        return

    # Execute with outer parallelism
    max_procs = max(1, int(args.max_procs))
    print(f"[bench] executing with max_procs={max_procs}", flush=True)

    futures = {}
    ok = 0
    failed = 0
    skipped = 0

    with ThreadPoolExecutor(max_workers=max_procs) as pool:
        for idx, job in enumerate(plan, start=1):
            out_dir = job["out_dir"]

            if args.skip_existing and os.path.exists(out_dir):
                print(f"[RUN {idx}] SKIP (exists): {out_dir}", flush=True)
                skipped += 1
                continue

            print(f"\n[RUN {idx}/{len(plan)}] {job['meta']['command']}", flush=True)
            append_bench_log(args.root_out, f"RUN {idx}/{len(plan)} {job['meta']['command']}")

            futures[pool.submit(run_one, job, csv_path)] = out_dir

        for fut in as_completed(futures):
            out_dir, rc = fut.result()
            if rc == 0:
                ok += 1
                print(f"[bench] OK: {out_dir}", flush=True)
            else:
                failed += 1
                print(f"[bench] FAIL rc={rc}: {out_dir}", flush=True)

    print(f"\n[bench] DONE  ok={ok}  failed={failed}  skipped={skipped}", flush=True)

if __name__ == "__main__":
    main()

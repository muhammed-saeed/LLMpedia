
#!/usr/bin/env python
from __future__ import annotations

import os
import re
import subprocess
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- Configurable experiment grid ----------

# How many subprocesses to run at the same time
MAX_CONCURRENT = 6  # tune based on your quota + machine

# Controversial / rich topics where personas will show clear differences
# Rich topics (less US-centric, different eras/regions)
TOPICS: Dict[str, str] = {
    # folder name        # seed passed to llmpedia_persona.py --seed
    # "us-civil-rights-movement": "US Civil rights movement",
    "ancient-city-of-babylon": "Ancient city of Babylon",
    # "dutch-colonization-in-southeast-asia": "Dutch colonization in Southeast Asia",
}


# Personas: keys are folder names, values are persona keys used in llmpedia
PERSONAS: Dict[str, str] = {
    "scientific_neutral": "scientific_neutral",
    # "left_leaning": "left_leaning",
    # "conservative": "conservative",
}

# Prompt strategy variants
# Top-level folders:
#   elicit-baseline_ner-baseline/
#   elicit-calibrate_ner-calibrate/
STRATEGY_COMBOS: List[Tuple[str, str]] = [
    ("baseline", "baseline"),
    ("calibrate", "calibrate"),
]

# Whether to run WITH and WITHOUT Self-RAG under each strategy
#   with-selfrag/
#   no-selfrag/
SELF_RAG_OPTIONS: List[bool] = [False, True]
SELF_RAG_OPTIONS: List[bool] = [True]


# Models for elicitation (article generation)

# Chat-style (non-Responses API) models: we will vary temperature for these
ELICIT_MODELS_CHAT: List[str] = [
    # "gpt-4.1-mini",
    # "gpt-4.1",
]

# Responses-API / reasoning models: we will vary reasoning effort & verbosity
ELICIT_MODELS_REASONING: List[str] = [
    "gpt-5-mini",
    # "gpt-5",
]

# NER ALWAYS uses a chat model when elicitation uses a reasoning model
DEFAULT_NER_MODEL = "gpt-4.1-mini"

# Temperatures for elicitation (for chat-style models)
ELICIT_TEMPERATURES: List[float] = [0.7]  # add 1.0, 1.5, ... if you like

# You can change these if you want to sweep top_p / top_k as well
ELICIT_TOP_P = 0.9
ELICIT_TOP_K = 50

# Reasoning model configs: (reasoning_effort, text_verbosity)
REASONING_CONFIGS = [
    # {"effort": "minimal", "verbosity": "low"},
    {"effort": "medium", "verbosity": "medium"},
    # {"effort": "high", "verbosity": "high"},
]

# Global LLMPedia run settings
MODE = "batch"
DOMAIN = "topic"
SELF_RAG_MODE = "batch"  # Self-RAG runs as online calls inside batch mode
MAX_DEPTH = 0             # 0 = unlimited depth (until queue drains)
MAX_SUBJECTS = 1000
BATCH_SIZE = 50
BATCH_POLL_INTERVAL = 30
MAX_RETRIES = 3

# Root folder where all experiment outputs will go
# EVAL_ROOT = Path("TopicBasedProfSimonPoints/1000_subjects")
EVAL_ROOT = Path("TopicBasedProfSimonPoints1/1000_subjects_gptt41_mini_ner")



# ---------- Helpers ----------

def slugify(value: str) -> str:
    """
    Very simple slugifier: lowercases and replaces non-alphanumeric with dashes.
    """
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-{2,}", "-", value)
    value = value.strip("-")
    return value or "run"


def run_cmd(cmd: List[str], env=None):
    """
    Run a subprocess, print the command, and handle errors without stopping the whole sweep.
    This is safe to call from multiple threads.
    """
    print("\n============================================================")
    print("Running command:")
    print(" ".join(cmd))
    print("============================================================", flush=True)

    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed with exit code {e.returncode}: {' '.join(cmd)}", flush=True)
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Stopping sweep due to Ctrl+C in worker.", flush=True)
        # Re-raise so the main loop can handle shutdown
        raise


def is_run_complete(output_dir: Path, max_subjects: int) -> bool:
    """
    Heuristic to decide if a given llmpedia_persona run is 'complete'.

    We look at queue.json (written by llmpedia_persona at the end of a run):

      - If it doesn't exist -> not complete.
      - Else, we parse statuses:
          * If max_subjects > 0 and done_count >= max_subjects -> complete.
          * Else if there are no 'pending' or 'working' rows -> complete.
          * Otherwise -> not complete (we should resume this config).
    """
    queue_json = output_dir / "queue.json"
    if not queue_json.exists():
        return False

    try:
        with queue_json.open("r", encoding="utf-8") as f:
            rows = json.load(f)
    except Exception:
        # If parsing fails, better to re-run than to silently skip
        return False

    if not isinstance(rows, list):
        return False

    done = 0
    has_pending = False
    has_working = False

    for r in rows:
        if not isinstance(r, dict):
            continue
        st = r.get("status")
        if st == "done":
            done += 1
        elif st == "pending":
            has_pending = True
        elif st == "working":
            has_working = True

    # Condition 1: hit the subject cap
    if max_subjects > 0 and done >= max_subjects:
        return True

    # Condition 2: queue is drained (no work left)
    if not has_pending and not has_working and done > 0:
        return True

    return False


def make_base_cmd(
    seed: str,
    output_dir: Path,
    elicit_strategy: str,
    ner_strategy: str,
    resume: bool,
    reset_working: bool,
) -> List[str]:
    """
    Build the base part of the llmpedia_persona.py command that is shared across runs.
    """
    output_dir = output_dir.resolve()
    os.makedirs(output_dir, exist_ok=True)

    base_cmd = [
        sys.executable,
        "llmpedia_persona.py",
        "--mode", MODE,
        "--seed", seed,
        "--output-dir", str(output_dir),
        "--domain", DOMAIN,
        "--elicitation-strategy", elicit_strategy,
        "--ner-strategy", ner_strategy,
        "--max-depth", str(MAX_DEPTH),
        "--max-subjects", str(MAX_SUBJECTS),
        "--batch-size", str(BATCH_SIZE),
        "--batch-poll-interval", str(BATCH_POLL_INTERVAL),
        "--max-retries", str(MAX_RETRIES),
        # Max tokens settings
        "--elicit-max-tokens", "9096",
        "--ner-max-tokens", "4096",
        "--self-rag-max-tokens", "9096",
    ]

    # Resume flags are passed through to llmpedia_persona
    if resume:
        base_cmd.append("--resume")
        if reset_working:
            base_cmd.append("--reset-working")

    return base_cmd


def sweep_experiments(
    max_concurrent: int = MAX_CONCURRENT,
    resume: bool = False,
    reset_working: bool = False,
):
    """
    Top-level sweep with resume-aware skipping.

    If resume=True:
      - For each config, check its output_dir/queue.json:
          * If "done >= MAX_SUBJECTS" OR no pending/working → SKIP.
          * Else → run llmpedia_persona.py with --resume [--reset-working].
    """
    print(f"[SWEEP] Output root: {EVAL_ROOT.resolve()}", flush=True)
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)

    # 1) Build the list of all commands we want to run
    all_cmds: List[List[str]] = []

    for elicit_strategy, ner_strategy in STRATEGY_COMBOS:
        strategy_tag = f"elicit-{elicit_strategy}_ner-{ner_strategy}"

        for use_self_rag in SELF_RAG_OPTIONS:
            sr_tag = "with-selfrag" if use_self_rag else "no-selfrag"

            for topic_slug, seed in TOPICS.items():
                for persona_folder, persona_key in PERSONAS.items():
                    # ---------------------------------------------------------
                    # 1) Chat-style models + temperature sweep
                    # ---------------------------------------------------------
                    for model_key in ELICIT_MODELS_CHAT:
                        for temp in ELICIT_TEMPERATURES:
                            temp_label = f"temp-{str(temp).replace('.', '_')}"
                            out_dir = (
                                EVAL_ROOT
                                / strategy_tag
                                / sr_tag
                                / topic_slug
                                / persona_folder
                                / slugify(model_key)
                                / temp_label
                            )

                            # If resuming: skip configs that look finished
                            if resume and is_run_complete(out_dir, MAX_SUBJECTS):
                                print(f"[SKIP] {out_dir} already complete (resume).", flush=True)
                                continue

                            cmd = make_base_cmd(
                                seed=seed,
                                output_dir=out_dir,
                                elicit_strategy=elicit_strategy,
                                ner_strategy=ner_strategy,
                                resume=resume,
                                reset_working=reset_working,
                            )

                            # Persona
                            cmd += ["--persona", persona_key]

                            # Self-RAG toggles for this run
                            if use_self_rag:
                                cmd += [
                                    "--self-rag", "true",
                                    "--self-rag-mode", SELF_RAG_MODE,
                                    "--self-rag-batch-size", "0",  # 0 = all subjects in wave
                                ]
                            else:
                                cmd += ["--self-rag", "false"]

                            # Elicitation uses the chat model
                            cmd += ["--elicit-model-key", model_key]

                            # NER uses the SAME chat model (never reasoning)
                            cmd += ["--ner-model-key", model_key]

                            # Self-RAG model: SAME as elicitation model (if enabled)
                            if use_self_rag:
                                cmd += ["--self-rag-model-key", model_key]

                            # Elicitation sampling settings
                            cmd += [
                                "--elicit-temperature", str(temp),
                                "--elicit-top-p", str(ELICIT_TOP_P),
                                "--elicit-top-k", str(ELICIT_TOP_K),
                                # NER sampling
                                "--ner-temperature", "0.3",
                                "--ner-top-p", "0.95",
                            ]

                            all_cmds.append(cmd)

                    # ---------------------------------------------------------
                    # 2) Reasoning / Responses-API models + effort/verbosity sweep
                    # ---------------------------------------------------------
                    for model_key in ELICIT_MODELS_REASONING:
                        for rc in REASONING_CONFIGS:
                            eff = rc["effort"]
                            verb = rc["verbosity"]
                            cfg_label = f"eff-{eff}_verb-{verb}"

                            out_dir = (
                                EVAL_ROOT
                                / strategy_tag
                                / sr_tag
                                / topic_slug
                                / persona_folder
                                / slugify(model_key)
                                / cfg_label
                            )

                            # If resuming: skip configs that look finished
                            if resume and is_run_complete(out_dir, MAX_SUBJECTS):
                                print(f"[SKIP] {out_dir} already complete (resume).", flush=True)
                                continue

                            cmd = make_base_cmd(
                                seed=seed,
                                output_dir=out_dir,
                                elicit_strategy=elicit_strategy,
                                ner_strategy=ner_strategy,
                                resume=resume,
                                reset_working=reset_working,
                            )

                            # Persona
                            cmd += ["--persona", persona_key]

                            # Self-RAG toggles for this run
                            if use_self_rag:
                                cmd += [
                                    "--self-rag", "true",
                                    "--self-rag-mode", SELF_RAG_MODE,
                                    "--self-rag-batch-size", "0",
                                ]
                            else:
                                cmd += ["--self-rag", "false"]

                            # Elicitation uses reasoning model
                            cmd += ["--elicit-model-key", model_key]

                            # NER uses FIXED chat model (never reasoning)
                            cmd += ["--ner-model-key", DEFAULT_NER_MODEL]

                            # Self-RAG model: SAME reasoning model as elicitation (if enabled)
                            if use_self_rag:
                                cmd += ["--self-rag-model-key", model_key]

                            # Reasoning controls for Responses API models
                            cmd += [
                                "--reasoning-effort", eff,
                                "--text-verbosity", verb,
                                # NER still uses chat-style defaults
                                "--ner-temperature", "0.3",
                                "--ner-top-p", "0.95",
                            ]

                            all_cmds.append(cmd)

    print(
        f"[SWEEP] Prepared {len(all_cmds)} runs. "
        f"Executing with up to {max_concurrent} concurrent processes...\n",
        flush=True,
    )

    if not all_cmds:
        print("[SWEEP] No runs to execute (everything already complete?).", flush=True)
        return

    # 2) Run them with limited concurrency
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {executor.submit(run_cmd, cmd): cmd for cmd in all_cmds}

        try:
            for i, future in enumerate(as_completed(futures), start=1):
                cmd = futures[future]
                try:
                    future.result()
                except KeyboardInterrupt:
                    print("\n[INTERRUPT] Received Ctrl+C, shutting down...", flush=True)
                    # Cancel remaining futures (best-effort)
                    for f in futures:
                        f.cancel()
                    raise
                except Exception as e:
                    print(f"[ERROR] Unexpected exception in worker for command: {' '.join(cmd)}", flush=True)
                    print(f"        {e}", flush=True)

                print(f"[SWEEP] Completed {i}/{len(all_cmds)} runs.", flush=True)
        except KeyboardInterrupt:
            # In case Ctrl+C happens while waiting on as_completed()
            print("\n[SWEEP] Main sweep loop interrupted. Exiting...", flush=True)

    print(f"\n[SWEEP] All runs submitted and finished. Check {EVAL_ROOT}/ for outputs.", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Sweep over LLMPedia persona configurations."
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=MAX_CONCURRENT,
        help=f"Maximum concurrent llmpedia_persona.py processes (default: {MAX_CONCURRENT}).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume existing runs: do not reseed queues in llmpedia_persona, and skip "
            "configs whose output_dir/queue.json indicates they are already complete."
        ),
    )
    parser.add_argument(
        "--reset-working",
        action="store_true",
        help=(
            "With --resume, also reset any 'working' rows in the queue to 'pending' "
            "inside llmpedia_persona.py via --reset-working."
        ),
    )

    args = parser.parse_args()

    # If someone passes --reset-working without --resume, we still forward it;
    # llmpedia_persona will only act on it when resuming.
    sweep_experiments(
        max_concurrent=args.max_concurrent,
        resume=args.resume,
        reset_working=args.reset_working,
    )

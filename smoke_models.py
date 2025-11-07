#!/usr/bin/env python3
# smoke_models.py
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple, Optional

# ---- Your codebase imports ----
# - We use the safe prompt loader you shared to avoid KeyError on brace placeholders
from prompter_parser import get_prompt_messages
from llm.config import ModelConfig
from llm.factory import make_llm_from_config
from prompts.schemas import ELICITATION_SCHEMA, ELICITATION_WITH_CONFIDENCE_SCHEMA, NER_SCHEMA

# --------------------------------------------------------------------------------------
# Config registry discovery
# --------------------------------------------------------------------------------------

def _try_import_settings_module() -> Optional[Any]:
    """
    Try several likely module paths for a settings module that holds ModelConfig objects.
    Returns the imported module or None.
    """
    candidates = [
        "core.settings",  # you showed earlier constants here
        "settings",       # sometimes kept at repo root
    ]
    for mod in candidates:
        try:
            return __import__(mod, fromlist=["*"])
        except Exception:
            pass
    return None

def _collect_model_registry() -> Dict[str, ModelConfig]:
    """
    Build a dict {key: ModelConfig}. Prefers a dict named MODEL_REGISTRY on the settings module.
    Otherwise, introspects attributes and collects any ModelConfig instances.
    """
    mod = _try_import_settings_module()
    out: Dict[str, ModelConfig] = {}
    if mod:
        # If repo defines a dict explicitly, prefer it.
        if hasattr(mod, "MODEL_REGISTRY") and isinstance(mod.MODEL_REGISTRY, dict):
            for k, v in mod.MODEL_REGISTRY.items():
                if isinstance(v, ModelConfig):
                    out[k] = v
            if out:
                return out

        # Otherwise collect all ModelConfig attrs
        for name in dir(mod):
            try:
                val = getattr(mod, name)
            except Exception:
                continue
            if isinstance(val, ModelConfig):
                out[name.lower()] = val

    # If still empty, fail loudly with instructions
    if not out:
        print(
            "[ERROR] Could not find any ModelConfig entries.\n"
            "Create a registry (e.g. in core/settings.py):\n\n"
            "  from llm.config import ModelConfig\n"
            "  MODEL_REGISTRY = {\n"
            "      'gpt4o': ModelConfig(provider='openai', model='gpt-4o', api_key_env='OPENAI_API_KEY'),\n"
            "      'deepseek': ModelConfig(provider='deepseek', model='deepseek-chat', api_key_env='DEEPSEEK_API_KEY'),\n"
            "      'grok4': ModelConfig(provider='replicate', model='xai/grok-4'),\n"
            "      # ... etc\n"
            "  }\n",
            file=sys.stderr,
        )
        sys.exit(2)
    return out


# --------------------------------------------------------------------------------------
# Helpers: JSON parsing (best-effort), printing
# --------------------------------------------------------------------------------------

CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

def _best_effort_parse(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    m = CODE_FENCE_RE.search(text)
    if m:
        text = m.group(1)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, str):
            return json.loads(obj)
    except Exception:
        pass
    start = text.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except Exception:
                        break
    return {}

def _normalize_facts_key(obj: Dict[str, Any]) -> Dict[str, Any]:
    if "facts" in obj and isinstance(obj["facts"], list):
        return obj
    if '"facts"' in obj and isinstance(obj['"facts"'], list):
        obj["facts"] = obj.pop('"facts"')
        return obj
    if "triples" in obj and isinstance(obj["triples"], list) and "facts" not in obj:
        obj["facts"] = obj["triples"]
        return obj
    return obj

def _list_conf_key(schema: Dict[str, Any]) -> bool:
    """
    True if the schema expects confidence per item.
    """
    try:
        props = schema["properties"]["facts"]["items"]["properties"]
        return "confidence" in props
    except Exception:
        return False

def _coerce_elicit_result(obj: Dict[str, Any], with_conf: bool) -> List[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return []
    facts = obj.get("facts")
    if not isinstance(facts, list):
        return []
    out = []
    for it in facts:
        if not isinstance(it, dict):
            continue
        s = it.get("subject")
        p = it.get("predicate")
        o = it.get("object")
        if not (isinstance(s, str) and isinstance(p, str) and isinstance(o, str)):
            continue
        if with_conf:
            c = it.get("confidence")
            out.append({"subject": s, "predicate": p, "object": o, "confidence": c})
        else:
            out.append({"subject": s, "predicate": p, "object": o})
    return out

def _print_header(title: str):
    bar = "=" * 80
    print(f"\n{bar}\n{title}\n{bar}")

def _print_model_block(key: str, cfg: ModelConfig):
    _print_header(f"MODEL KEY: {key}")
    print(f"PROVIDER  : {cfg.provider}")
    print(f"MODEL     : {cfg.model}\n")


# --------------------------------------------------------------------------------------
# Known-good / known-bad sets (tune to your environment)
# --------------------------------------------------------------------------------------

# Put models that *consistently* work for both tasks here to skip with --skip-known-good
KNOWN_GOOD_KEYS = {
    # OpenAI (you said these are good)
    "gpt4o",
    "gpt4o-mini",
    "gpt4-turbo",
    # DeepSeek chat was okay for you
    "deepseek",
    # Replicate winners you saw working:
    "llama3-70b",
    "granite8b",
}

# Replicate model keys you’ve repeatedly seen 4xx/5xx (skip unless --include-known-bad)
KNOWN_BAD_REPLICATE_KEYS = {
    "mistral7b",
    "mixtral8x7b",
    "gemma2b",
    "qwen2-7b",
    "falcon180b",
    "granite20b",  # version disabled
}


# --------------------------------------------------------------------------------------
# Core run functions
# --------------------------------------------------------------------------------------

@dataclass
class RunResult:
    elicit_state: str   # "ok" | "empty" | "error" | "init-error"
    elicit_mode: str    # "schema" | "fallback" | "-"
    elicit_n: int
    ner_state: str
    ner_mode: str
    ner_n: int

def _key_present(val: Optional[str]) -> str:
    if not val:
        return "(missing)"
    # mask
    if len(val) <= 6:
        return "*" * len(val)
    return "*" * (len(val) - 4) + val[-4:]

def _print_keys_presence():
    openai = os.getenv("OPENAI_API_KEY")
    deepseek = os.getenv("DEEPSEEK_API_KEY")
    replicate = os.getenv("REPLICATE_API_TOKEN") or os.getenv("REPLICATE_API_KEY")
    print("\nKeys: OPENAI={}  DEEPSEEK={}  REPLICATE={}\n".format(
        _key_present(openai),
        _key_present(deepseek),
        _key_present(replicate),
    ))

def _filter_models(
    registry: Dict[str, ModelConfig],
    *,
    only_provider: Optional[str],
    exclude_provider: Optional[str],
    only_keys: Optional[str],
    exclude_regex: Optional[str],
    skip_known_good: bool,
    include_known_bad: bool,
) -> Dict[str, ModelConfig]:
    out = {}
    only_set = None
    if only_keys:
        only_set = {s.strip() for s in only_keys.split(",") if s.strip()}

    rx = re.compile(exclude_regex) if exclude_regex else None

    for k, cfg in registry.items():
        if only_set and k not in only_set:
            continue
        if only_provider and cfg.provider != only_provider:
            continue
        if exclude_provider and cfg.provider == exclude_provider:
            continue
        if rx and rx.search(k):
            continue
        if skip_known_good and k in KNOWN_GOOD_KEYS:
            continue
        if (cfg.provider == "replicate") and (not include_known_bad) and (k in KNOWN_BAD_REPLICATE_KEYS):
            continue
        out[k] = cfg
    return out

def _elicit(
    llm_callable,
    subject: str,
    *,
    use_conf_schema: bool,
) -> Tuple[str, int, List[Dict[str, Any]]]:
    """
    Returns (mode, n, facts). mode in {"schema", "fallback"}.
    """
    # Build messages safely via your prompter_parser
    elicit_msgs = get_prompt_messages(
        strategy="baseline",
        ptype="elicitation",
        domain="general",
        variables={"subject_name": subject},
    )

    # Print the prompts we pass
    print("-- ELICITATION ---------------------------------------------")
    sys_msg = (elicit_msgs[0]["content"] or "").strip()
    usr_msg = (elicit_msgs[1]["content"] or "").strip()
    print("System Prompt:\n  " + "\n  ".join(sys_msg.splitlines()))
    print("User Prompt:\n  " + "\n  ".join(usr_msg.splitlines()))
    print()

    schema = ELICITATION_WITH_CONFIDENCE_SCHEMA if use_conf_schema else ELICITATION_SCHEMA

    t0 = time.time()
    try:
        resp = llm_callable(elicit_msgs, json_schema=schema)
    except Exception as e:
        raise RuntimeError(f"[ELICITATION ERROR] {type(e).__name__}: {e}")

    elapsed = time.time() - t0

    # Direct schema success:
    if isinstance(resp, dict) and "facts" in resp:
        facts = _coerce_elicit_result(resp, with_conf=_list_conf_key(schema))
        print(f"Parsed facts (n={len(facts)}, {elapsed:.1f}s, mode=schema):")
        return "schema", len(facts), facts

    # Fallback: try to parse raw
    raw = ""
    if isinstance(resp, dict):
        raw = resp.get("_raw") or resp.get("text") or ""
    elif isinstance(resp, str):
        raw = resp

    parsed = _best_effort_parse(raw)
    parsed = _normalize_facts_key(parsed)
    facts = _coerce_elicit_result(parsed, with_conf=_list_conf_key(schema))
    print(f"Parsed facts (n={len(facts)}, {elapsed:.1f}s, mode=fallback):")
    return "fallback", len(facts), facts

def _run_ner(llm_callable, phrases: List[str]) -> Tuple[str, int, List[Dict[str, Any]]]:
    """
    Returns (mode, n, entities)
    """
    block = "\n".join(phrases)
    ner_msgs = get_prompt_messages(
        strategy="baseline",
        ptype="ner",
        domain="general",
        variables={"phrases_block": block},
    )

    print("\n-- NER ---------------------------------------------")
    sys_msg = (ner_msgs[0]["content"] or "").strip()
    usr_msg = (ner_msgs[1]["content"] or "").strip()
    print("System Prompt:\n  " + "\n  ".join(sys_msg.splitlines()))
    print("User Prompt:\n  " + "\n  ".join(usr_msg.splitlines()))
    print()

    t0 = time.time()
    try:
        resp = llm_callable(ner_msgs, json_schema=NER_SCHEMA)
    except Exception as e:
        raise RuntimeError(f"[NER ERROR] {type(e).__name__}: {e}")
    elapsed = time.time() - t0

    # Direct schema success:
    if isinstance(resp, dict) and "entities" in resp and isinstance(resp["entities"], list):
        ents = resp["entities"]
        print(f"Classifications (phrases schema) (n={len(ents)}, {elapsed:.1f}s, mode=schema):")
        return "schema", len(ents), ents

    # Fallback: try to parse raw
    raw = ""
    if isinstance(resp, dict):
        raw = resp.get("_raw") or resp.get("text") or ""
    elif isinstance(resp, str):
        raw = resp

    parsed = _best_effort_parse(raw)
    ents = []
    if isinstance(parsed, dict):
        ents = parsed.get("entities", [])
        if not isinstance(ents, list):
            ents = []
    print(f"Classifications (phrases schema) (n={len(ents)}, {elapsed:.1f}s, mode=fallback):")
    return "fallback", len(ents), ents


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser("Smoke-test your models on elicitation + NER.")
    ap.add_argument("--only-provider", help="Only run models from this provider (e.g., openai|deepseek|replicate).")
    ap.add_argument("--exclude-provider", help="Exclude models from this provider.")
    ap.add_argument("--only-keys", help="Comma-separated list of model keys to include (overrides others).")
    ap.add_argument("--exclude", help="Regex of keys to exclude (e.g., 'unsloth|llama2').")
    ap.add_argument("--skip-known-good", action="store_true", help="Skip models that are consistently working.")
    ap.add_argument("--include-known-bad", action="store_true", help="Run Replicate models that usually 4xx/5xx.")
    ap.add_argument("--subject", default="Alan Turing", help="Elicitation subject.")
    ap.add_argument("--entities", default="London, Princeton University, computer, 1943, Bletchley Park",
                    help="Comma-separated phrases to classify in NER.")
    ap.add_argument("--confidence", action="store_true",
                    help="Ask elicitation with confidence schema (if the model supports it).")
    args = ap.parse_args()

    # Header
    print("\n" + "="*80)
    print("MODEL SMOKE TEST")
    print("="*80)
    print(f"Subject      : {args.subject}")
    print(f"Entities     : {args.entities}")
    if args.only_provider:
        print(f"Only provider: {args.only_provider}")
    if args.exclude_provider:
        print(f"Exclude provider: {args.exclude_provider}")
    if args.only_keys:
        print(f"Only keys    : {args.only_keys}")
    if args.exclude:
        print(f"Exclude regex: {args.exclude}")
    if args.skip_known_good:
        print("Skipping known-good models.")
    if not args.include_known_bad:
        print("Skipping known-bad Replicate models (use --include-known-bad to force).")
    _print_keys_presence()

    phrases = [s.strip() for s in args.entities.split(",") if s.strip()]

    # Discover models
    registry = _collect_model_registry()
    filt = _filter_models(
        registry,
        only_provider=args.only_provider,
        exclude_provider=args.exclude_provider,
        only_keys=args.only_keys,
        exclude_regex=args.exclude,
        skip_known_good=args.skip_known_good,
        include_known_bad=args.include_known_bad,
    )

    if not filt:
        print("\n[INFO] No models to run after filters.")
        return

    summary_rows: List[Tuple[Any, ...]] = []
    summary_path = "smoke_summary.jsonl"
    # overwrite previous run
    try:
        open(summary_path, "w").close()
    except Exception:
        pass

    for key, cfg in filt.items():
        _print_model_block(key, cfg)

        # Init LLM
        try:
            llm = make_llm_from_config(cfg)
        except Exception as e:
            print(f"\n[INIT ERROR] {type(e).__name__}: {e}\n")
            # keep summary shape at 7 fields
            summary_rows.append((key, "init-error", "-", 0, "init-error", "-", 0))
            continue

        # ELICITATION
        try:
            el_mode, el_n, el_facts = _elicit(llm, args.subject, use_conf_schema=args.confidence)
            for i, f in enumerate(el_facts[:30], 1):
                if "confidence" in f and f["confidence"] is not None:
                    print(f"{i:4d}. ({f['subject']}) —[{f['predicate']}]→ {f['object']}  (conf={f['confidence']:.2f})")
                else:
                    print(f"{i:4d}. ({f['subject']}) —[{f['predicate']}]→ {f['object']}")
            if el_n == 0:
                print("  (none)")
            print()
            el_state = "ok" if el_n > 0 else "empty"
        except Exception as e:
            print(str(e))
            el_state, el_mode, el_n = "error", "-", 0

        # NER
        try:
            ner_mode, ner_n, ner_ents = _run_ner(llm, phrases)
            for i, it in enumerate(ner_ents[:50], 1):
                # expecting {"name","type","keep"} ideally, but show what we have
                name = it.get("name") or it.get("phrase") or str(it)
                typ  = it.get("type") or ("NE" if it.get("is_ne") else "not-NE" if "is_ne" in it else "?")
                conf = it.get("confidence")
                if conf is not None:
                    try:
                        print(f"{i:4d}. {name}  → {typ} (conf={float(conf):.2f})")
                    except Exception:
                        print(f"{i:4d}. {name}  → {typ} (conf={conf})")
                else:
                    print(f"{i:4d}. {name}  → {typ}")
            if ner_n == 0:
                print("  (none)")
            print()
            ner_state = "ok" if ner_n > 0 else "empty"
        except Exception as e:
            print(str(e))
            ner_state, ner_mode, ner_n = "error", "-", 0

        # append to summary
        summary_rows.append((key, el_state, el_mode, el_n, ner_state, ner_mode, ner_n))

        # write per-model JSONL line (nice for later analysis)
        try:
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "key": key,
                    "provider": cfg.provider,
                    "model": cfg.model,
                    "elicitation": {"state": el_state, "mode": el_mode, "n": el_n},
                    "ner": {"state": ner_state, "mode": ner_mode, "n": ner_n},
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # SUMMARY
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for row in summary_rows:
        # Be defensive on row length
        if len(row) == 7:
            key, el_st, el_mode, el_n, ner_st, ner_mode, ner_n = row
        else:
            key = row[0] if len(row) > 0 else "?"
            el_st = row[1] if len(row) > 1 else "error"
            el_mode = row[2] if len(row) > 2 else "-"
            el_n = row[3] if len(row) > 3 else 0
            ner_st = row[4] if len(row) > 4 else "error"
            ner_mode = row[5] if len(row) > 5 else "-"
            ner_n = row[6] if len(row) > 6 else 0

        print(f"{key:>14} | Elicit: {el_st:<7} (mode={el_mode:>7}, n={el_n:>2}) | "
              f"NER: {ner_st:<7} (mode={ner_mode:>7}, n={ner_n:>2})")

    # Also write a compact JSON summary for scripting
    try:
        with open("smoke_summary.json", "w", encoding="utf-8") as f:
            json.dump([
                {
                    "key": row[0],
                    "elicitation": {"state": row[1], "mode": row[2], "n": row[3]},
                    "ner": {"state": row[4], "mode": row[5], "n": row[6]},
                }
                for row in summary_rows
            ], f, ensure_ascii=False, indent=2)
        print("\nWrote smoke_summary.json")
        print("Wrote smoke_summary.jsonl")
    except Exception:
        pass

    print("\nDone.")

if __name__ == "__main__":
    main()

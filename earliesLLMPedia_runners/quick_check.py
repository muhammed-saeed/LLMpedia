
# quick_check.py
from __future__ import annotations
import os, json, argparse, sys
from dotenv import load_dotenv

from settings import (
    ELICIT_SCHEMA_BASE,
    ELICIT_SCHEMA_CAL,
    settings,
)
from llm.factory import make_llm_from_config


# Load .env so OPENAI_API_KEY / REPLICATE_API_TOKEN are available
load_dotenv()


def lock_down_additional_properties(schema):
    """
    Recursively add `"additionalProperties": false` to every object, so
    OpenAI Chat Completions with strict JSON schemas accepts it.
    Harmless for other providers (Replicate, DeepSeek, etc.).
    """
    if isinstance(schema, dict):
        t = schema.get("type")
        if t == "object":
            schema.setdefault("additionalProperties", False)
            props = schema.get("properties") or {}
            for v in props.values():
                lock_down_additional_properties(v)
        elif t == "array":
            lock_down_additional_properties(schema.get("items"))
    return schema


def ensure_env_for_model(cfg_key: str):
    """
    Quick sanity check for required env vars based on provider.
    """
    cfg = settings.MODELS[cfg_key]
    provider = (cfg.provider or "").lower()

    if provider == "openai":
        env_name = getattr(cfg, "api_key_env", "OPENAI_API_KEY") or "OPENAI_API_KEY"
        if not os.getenv(env_name):
            sys.exit(f"Set {env_name} in your environment or .env file for model '{cfg_key}'.")
    elif provider == "replicate":
        if not os.getenv("REPLICATE_API_TOKEN"):
            sys.exit("Set REPLICATE_API_TOKEN in your environment or .env file for Replicate models.")
    elif provider == "deepseek":
        env_name = getattr(cfg, "api_key_env", "DEEPSEEK_API_KEY") or "DEEPSEEK_API_KEY"
        if not os.getenv(env_name):
            sys.exit(f"Set {env_name} in your environment or .env file for model '{cfg_key}'.")
    # Unsloth is disabled in your factory; nothing to check.
    return cfg


def build_messages(subject: str, n: int, calibrated: bool) -> list[dict]:
    """
    Minimal prompt: one system rule (JSON only) + one user ask.
    """
    sys_shape = (
        '{"facts":[{"subject":"...","predicate":"...","object":"..."'
        + (',"confidence":0.0' if calibrated else "")
        + "}]}"
    )
    system = {
        "role": "system",
        "content": f"Output ONLY JSON; shape: {sys_shape}",
    }
    user = {
        "role": "user",
        "content": (
            f"Extract {n} factual triples about '{subject}'. "
            "Use keys subject/predicate/object"
            + ("/confidence" if calibrated else "")
            + "."
        ),
    }
    return [system, user]


def main():
    ap = argparse.ArgumentParser(description="Quick check: call any model from settings.MODELS and validate elicitation JSON.")
    ap.add_argument("--model-key", default="granite8b",
                    help="Key in settings.MODELS (e.g., granite8b, llama3-8b-instruct, gpt4o-mini, deepseek, etc.)")
    ap.add_argument("--subject", default="The Big Bang Theory TV Series", help="Subject to elicit facts about.")
    ap.add_argument("--n", type=int, default=3, help="How many triples to request.")
    ap.add_argument("--calibrate", action="store_true",
                    help="Use calibrated schema (includes 'confidence'). Recommended for testing.")
    ap.add_argument("--show-raw", action="store_true", help="Also print _raw when present.")
    args = ap.parse_args()

    # Ensure needed env var for chosen provider
    cfg = ensure_env_for_model(args.model_key)

    # Make client from your factory
    client = make_llm_from_config(cfg)

    # Messages + schema (lock it down so OpenAI strict JSON is accepted)
    msgs = build_messages(args.subject, args.n, calibrated=args.calibrate)
    base_schema = ELICIT_SCHEMA_CAL if args.calibrate else ELICIT_SCHEMA_BASE
    schema = lock_down_additional_properties(dict(base_schema))  # shallow copy then lock

    # Call the model
    out = client(msgs, json_schema=schema)

    # Pretty print
    print("\n=== RESULT ===")
    print(json.dumps(out, ensure_ascii=False, indent=2))

    # Optional: show raw block for debugging (some clients attach _raw)
    if args.show_raw and isinstance(out, dict) and "_raw" in out:
        print("\n=== _RAW (for debugging) ===")
        raw = out.get("_raw")
        try:
            # If raw looks like JSON, pretty-print; else print as text
            raw_obj = json.loads(raw) if isinstance(raw, str) else raw
            print(json.dumps(raw_obj, ensure_ascii=False, indent=2))
        except Exception:
            print(str(raw))


if __name__ == "__main__":
    main()


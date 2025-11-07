# core/pipeline_elicit.py
from __future__ import annotations
import json, re
from typing import Dict, Any, List
from pathlib import Path

from core.prompt_loader import load_messages_from_prompt_json
from llm.factory import make_llm_from_config
from llm.config import ModelConfig

TRIPLES_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "predicate": {"type": "string"},
                    "object": {"type": "string"}
                },
                "required": ["subject", "predicate", "object"],
                "additionalProperties": False
            }
        }
    },
    "required": ["facts"],
    "additionalProperties": False
}

# Best-effort cleaner for common LLM quirks
CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

def _best_effort_parse(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    # 1) fenced block
    m = CODE_FENCE_RE.search(text)
    if m:
        text = m.group(1)
    # 2) direct JSON
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, str):  # JSON string containing JSON
            return json.loads(obj)
    except Exception:
        pass
    # 3) first balanced {...}
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
    # Sometimes weird keys like '"facts"' appear; normalize them.
    if "facts" in obj and isinstance(obj["facts"], list):
        return obj
    if '"facts"' in obj and isinstance(obj['"facts"'], list):
        obj["facts"] = obj.pop('"facts"')
        return obj
    # Also accept 'triples' synonym if present
    if "triples" in obj and isinstance(obj["triples"], list) and "facts" not in obj:
        obj["facts"] = obj["triples"]
        return obj
    return obj

def run_elicitation(
    cfg: ModelConfig,
    prompt_path: str,
    subject_name: str,
) -> Dict[str, Any]:
    # Load system+user from your single JSON prompt file
    messages = load_messages_from_prompt_json(prompt_path, subject_name=subject_name)

    # Build LLM for the provider/model in settings
    llm = make_llm_from_config(cfg)

    # Ask for strict JSON if possible (OpenAI/DeepSeek/Replicate all supported in your codebase)
    resp = llm(messages, json_schema=TRIPLES_SCHEMA)

    # Case A: schema succeeded and we got a dict with facts
    if isinstance(resp, dict) and "facts" in resp:
        return {"facts": resp["facts"]}

    # Case B: schema failed -> many clients return {"_raw": "..."} or {"text": "..."}
    raw = ""
    if isinstance(resp, dict):
        raw = resp.get("_raw") or resp.get("text") or ""
    elif isinstance(resp, str):
        raw = resp

    parsed = _best_effort_parse(raw)
    parsed = _normalize_facts_key(parsed)

    if isinstance(parsed, dict) and "facts" in parsed and isinstance(parsed["facts"], list):
        return {"facts": parsed["facts"]}

    # Graceful empty result so the runner can keep going
    return {"facts": []}



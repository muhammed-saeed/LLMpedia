# core/pipeline_article.py
from __future__ import annotations
import json, re
from typing import Dict, Any
from llm.factory import make_llm_from_config
from core.prompt_loader import load_messages_from_prompt_json

CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)

def _best_effort_parse(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    m = CODE_FENCE_RE.search(text)
    if m:
        text = m.group(1)
    # direct
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        if isinstance(obj, str):
            return json.loads(obj)
    except Exception:
        pass
    # first balanced { .. }
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

def run_article(cfg, prompt_path: str, subject_name: str, json_schema: Dict[str, Any]) -> Dict[str, Any]:
    llm = make_llm_from_config(cfg)
    messages = load_messages_from_prompt_json(prompt_path, subject_name=subject_name)
    resp = llm(messages, json_schema=json_schema)

    if isinstance(resp, dict) and all(k in resp for k in ("title", "summary", "sections")):
        return resp

    raw = ""
    if isinstance(resp, dict):
        raw = resp.get("_raw") or resp.get("text") or ""
    elif isinstance(resp, str):
        raw = resp

    parsed = _best_effort_parse(raw)
    if isinstance(parsed, dict) and all(k in parsed for k in ("title", "summary", "sections")):
        parsed["_raw"] = raw
        return parsed

    return {"_raw": raw}  # let caller decide sink reason
def apply_section_conf_threshold(article: Dict[str, Any], thr: float | None) -> Dict[str, Any]:
    if thr is None:
        return article
    secs = []
    for s in article.get("sections") or []:
        c = s.get("confidence")
        try:
            c = float(c) if c is not None else None
        except Exception:
            c = None
        if c is None or c >= thr:
            secs.append(s)
    article = dict(article)
    article["sections"] = secs
    return article

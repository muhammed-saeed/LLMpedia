# llm/json_utils.py
from __future__ import annotations
import json

def strip_fences(t: str) -> str:
    s = (t or "").strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1:].strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    return s

def best_json(text: str):
    """
    Robust, quote/escape-aware JSON extraction.
    Returns a dict/list on success, or {} on failure.
    """
    if not text:
        return {}
    # direct attempt
    try:
        return json.loads(text)
    except Exception:
        pass

    t = strip_fences(text)
    try:
        return json.loads(t)
    except Exception:
        pass

    def scan_for(open_ch: str, close_ch: str):
        s = -1
        depth = 0
        in_str = False
        esc = False
        for i, ch in enumerate(t):
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == open_ch:
                if depth == 0:
                    s = i
                depth += 1
            elif ch == close_ch and depth > 0:
                depth -= 1
                if depth == 0 and s != -1:
                    chunk = t[s:i+1]
                    try:
                        return json.loads(chunk)
                    except Exception:
                        s = -1  # keep scanning
        return {}

    return scan_for("{", "}") or scan_for("[", "]") or {}

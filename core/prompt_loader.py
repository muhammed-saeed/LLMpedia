# core/prompt_loader.py
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any

# Only these placeholders will be replaced; all other braces are left intact.
_ALLOWED_KEYS = {"subject_name", "phrases_block", "root_subject"}

def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.exists():
        return p
    here = Path(__file__).resolve().parents[1]  # project root (.. from core/)
    p2 = (here / p).resolve()
    if p2.exists():
        return p2
    p3 = Path.cwd() / p
    if p3.exists():
        return p3
    raise FileNotFoundError(f"Prompt not found. Tried: {p}, {p2}, {p3}")

def _safe_render(template: str, variables: Dict[str, Any] | None) -> str:
    """
    Replace ONLY whitelisted placeholders like {subject_name} or {phrases_block}.
    Leave ALL other { ... } untouched (e.g., JSON braces, schema examples).
    """
    if not template:
        return ""
    if not variables:
        return template
    out = template
    for k, v in variables.items():
        if k in _ALLOWED_KEYS:
            out = out.replace("{" + k + "}", str(v))
    return out

def load_messages_from_prompt_json(path: str | Path, **vars) -> List[Dict[str, str]]:
    obj = json.loads(_resolve(path).read_text(encoding="utf-8"))
    system = _safe_render(obj.get("system") or "", vars).strip()
    user   = _safe_render(obj.get("user") or "", vars).strip()
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]

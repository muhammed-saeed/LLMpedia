# prompter_parser.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

# Only replace known {placeholder} keys; never interpret other braces.
_ALLOWED_KEYS = {"subject_name", "phrases_block", "root_subject", "max_facts_hint"}

# Canonical footer we want in every elicitation *system* message
_ELICITATION_SYSTEM_FOOTER = ( "" )

def _prompt_path(domain: str, strategy: str, ptype: str) -> Path:
    # prompts/<domain>/<strategy>/<ptype>.json
    return Path("prompts") / domain / strategy / f"{ptype}.json"

def _safe_render(template: str, variables: Dict[str, str] | None) -> str:
    if not template:
        return ""
    if not variables:
        return template
    out = template
    for k, v in variables.items():
        if k in _ALLOWED_KEYS:
            out = out.replace("{" + k + "}", str(v))
    # leave ALL other { ... } untouched (JSON braces, examples, etc.)
    return out

def _ensure_footer(system_txt: str, ptype: str) -> str:
    if ptype != "elicitation":
        return system_txt or ""
    marker = "include at least one triple where predicate is \"instanceOf\""
    st = (system_txt or "")
    if marker.lower() in st.lower():
        return st
    return (st.rstrip() + _ELICITATION_SYSTEM_FOOTER)

def get_prompt_messages(
    strategy: str,
    ptype: str,
    *,
    domain: str = "general",
    variables: Dict[str, str] | None = None,
) -> List[dict]:
    path = _prompt_path(domain, strategy, ptype)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)

    system_tmpl = obj.get("system", "") or ""
    user_tmpl   = obj.get("user", "") or ""

    system_txt = _safe_render(system_tmpl, variables).strip()
    user_txt   = _safe_render(user_tmpl, variables).strip()

    # Ensure footer for elicitation system messages
    system_txt = _ensure_footer(system_txt, ptype).strip()

    return [
        {"role": "system", "content": system_txt},
        {"role": "user",   "content": user_txt},
    ]

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any

def load_messages_from_prompt_json(path: str | Path, **vars) -> List[Dict[str, str]]:
    """
    Read a prompt JSON file with:
      { "system": "...", "user": "..." }
    and return OpenAI-like messages after Python .format(**vars).
    """
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    system = (obj.get("system") or "").format(**vars)
    user   = (obj.get("user") or "").format(**vars)
    return [{"role":"system","content":system}, {"role":"user","content":user}]

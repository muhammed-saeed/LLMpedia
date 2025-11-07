# llm/deepseek_client.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import json
import time
import requests

from llm.json_utils import best_json, strip_fences as _strip_fences  # unified utils

def _schema_hint(schema: Dict[str, Any]) -> str:
    return (
        "Return ONLY one valid JSON object that matches this JSON Schema exactly. "
        "No prose, no markdown, no code fences. "
        "If unsure, return an empty but valid object per schema.\nSCHEMA:\n"
        + json.dumps(schema, ensure_ascii=False)
    )

def _best_json(text: str) -> Dict[str, Any]:
    obj = best_json(text)
    return obj if isinstance(obj, dict) else {}

class DeepSeekClient:
    """
    Minimal DeepSeek client (Chat-like).
    We never use OpenAI 'response_format', since DeepSeek won't accept json_schema.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = "https://api.deepseek.com",
        max_tokens: Optional[int] = 1024,
        temperature: Optional[float] = 0.2,
        top_p: Optional[float] = 1.0,
        extra_inputs: Optional[Dict[str, Any]] = None,
        request_timeout: float = 120.0,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.extra = extra_inputs or {}
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.request_timeout = request_timeout

    def __call__(self, messages: List[Dict[str, str]], json_schema: Optional[Dict[str, Any]] = None):
        # Inject strict schema instructions in the system message instead of response_format
        msgs = list(messages)
        if json_schema:
            if msgs and msgs[0].get("role") == "system":
                msgs[0] = {"role": "system", "content": msgs[0]["content"] + "\n\n" + _schema_hint(json_schema)}
            else:
                msgs.insert(0, {"role": "system", "content": _schema_hint(json_schema)})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        # allow user extras (e.g., penalties) but remove Nones
        for k, v in (self.extra or {}).items():
            if v is not None:
                payload[k] = v

        # modest retry for transient HTTP errors
        last_exc = None
        for attempt in range(3):
            try:
                r = requests.post(self.url, headers=self.headers, json=payload, timeout=self.request_timeout)
                r.raise_for_status()
                data = r.json()
                break
            except requests.HTTPError as e:
                last_exc = e
                status = getattr(e.response, "status_code", None) if e.response else None
                if status in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise
            except Exception as e:
                last_exc = e
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise last_exc  # re-raise after retries

        # content
        try:
            text = (data["choices"][0]["message"]["content"] or "").strip()
        except Exception:
            text = ""

        if not json_schema:
            return {"text": text, "_raw": text}

        # parse/salvage
        obj = _best_json(text)
        if obj:
            return obj
        return {"_raw": text}

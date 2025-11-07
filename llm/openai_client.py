# llm/openai_client.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import json
from openai import OpenAI
from openai import BadRequestError

# ---------- helpers borrowed from DeepSeek client ----------

def _schema_hint(schema: Dict[str, Any]) -> str:
    return (
        "Return ONLY one valid JSON object that matches this JSON Schema exactly. "
        "No prose, no markdown, no code fences.\nSCHEMA:\n" +
        json.dumps(schema, ensure_ascii=False)
    )

def _strip_fences(t: str) -> str:
    s = (t or "").strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            s = s[nl+1:].strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    return s

def _best_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    # direct
    try:
        return json.loads(text)
    except Exception:
        pass
    # strip fences
    t = _strip_fences(text)
    try:
        return json.loads(t)
    except Exception:
        pass
    # first balanced object
    s = t.find("{")
    if s != -1:
        depth = 0
        for i, ch in enumerate(t[s:], s):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[s:i+1])
                    except Exception:
                        break
    return {}

def _lock_down_additional_props(schema: Any) -> Any:
    """
    Recursively enforce additionalProperties:false on all object nodes.
    This prevents OpenAI's 'additionalProperties is required and must be false' error.
    """
    if isinstance(schema, dict):
        t = schema.get("type")
        if t == "object":
            schema.setdefault("additionalProperties", False)
            props = schema.get("properties")
            if isinstance(props, dict):
                for k in list(props.keys()):
                    props[k] = _lock_down_additional_props(props[k])
        elif t == "array":
            if "items" in schema:
                schema["items"] = _lock_down_additional_props(schema["items"])
        else:
            # primitives: nothing to do
            pass
    return schema

def _inject_schema_hint_into_messages(messages: List[Dict[str, str]], json_schema: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    DeepSeek-style: put the schema contract into the system message so that even
    if response_format isn't honored, the model is still told to output strict JSON.
    """
    msgs = list(messages)
    hint = _schema_hint(json_schema)
    if msgs and (msgs[0].get("role") == "system"):
        msgs[0] = {"role": "system", "content": (msgs[0].get("content","") + "\n\n" + hint)}
    else:
        msgs.insert(0, {"role": "system", "content": hint})
    return msgs

def _extract_text_from_chat(resp) -> str:
    try:
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""

def _extract_text_from_responses_api(resp) -> str:
    # Prefer convenience field when available
    out = getattr(resp, "output_text", None)
    if out:
        return out.strip()
    # Reconstruct from blocks if needed
    try:
        parts: List[str] = []
        for block in getattr(resp, "output", []) or []:
            for c in getattr(block, "content", []) or []:
                txt = getattr(c, "text", "")
                if txt:
                    parts.append(txt)
        return "".join(parts).strip()
    except Exception:
        return ""

def _parse_with_salvage(text: str, want_schema: bool) -> Dict[str, Any]:
    if not want_schema:
        return {"text": text}
    # strict first
    try:
        return json.loads(text)
    except Exception:
        pass
    # salvage
    obj = _best_json(text)
    if obj:
        return obj
    return {"_raw": text}

# ---------- client ----------

class OpenAIClient:
    """
    Unified OpenAI client that can call either:
      • Chat Completions API (gpt-4o, gpt-4o-mini, etc.)
      • Responses API (gpt-5 family)

    DeepSeek-style hardening:
      - Inject schema hint into system message
      - Lock additionalProperties:false recursively
      - Salvage JSON if strict parsing fails
      - Retry without response_format if provider rejects schema
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        max_tokens: Optional[int] = 1024,
        temperature: Optional[float] = 0.0,
        top_p: Optional[float] = 1.0,
        use_responses_api: bool = False,
        extra_inputs: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        # Heuristic: Responses API for gpt-5* unless explicitly disabled
        self.use_responses_api = bool(use_responses_api or (model or "").startswith("gpt-5"))
        self.extra_inputs = extra_inputs or {}

        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=api_key)

    def __call__(self, messages: List[Dict[str, str]], json_schema: Optional[Dict[str, Any]] = None):
        if self.use_responses_api:
            return self._call_responses(messages, json_schema)
        return self._call_chat(messages, json_schema)

    # ---------------- Chat Completions ----------------

    def _call_chat(self, messages: List[Dict[str, str]], json_schema: Optional[Dict[str, Any]]):
        msgs = list(messages)
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            messages=msgs,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

        # If schema provided: lock schema, inject hint, try with response_format first
        have_schema = json_schema is not None
        if have_schema:
            safe_schema = _lock_down_additional_props(json.loads(json.dumps(json_schema)))
            msgs = _inject_schema_hint_into_messages(msgs, safe_schema)
            kwargs["messages"] = msgs
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "schema",
                    "schema": safe_schema,
                    "strict": True,  # ask for validation
                },
            }

        # 1st try: with response_format (when schema present)
        try:
            resp = self.client.chat.completions.create(**kwargs)
            text = _extract_text_from_chat(resp)
            if not have_schema:
                return {"text": text}
            parsed = _parse_with_salvage(text, want_schema=True)
            return parsed
        except BadRequestError as e:
            # Common case: JSON schema format complaints → retry without response_format
            if have_schema:
                try:
                    # Remove response_format, keep the DeepSeek-style system hint
                    kwargs.pop("response_format", None)
                    resp = self.client.chat.completions.create(**kwargs)
                    text = _extract_text_from_chat(resp)
                    parsed = _parse_with_salvage(text, want_schema=True)
                    return parsed
                except Exception:
                    raise
            raise
        except Exception:
            # Last resort: retry without response_format if we had schema
            if have_schema:
                kwargs.pop("response_format", None)
                resp = self.client.chat.completions.create(**kwargs)
                text = _extract_text_from_chat(resp)
                parsed = _parse_with_salvage(text, want_schema=True)
                return parsed
            raise

    # ---------------- Responses API (gpt-5*) ----------------

    def _call_responses(self, messages: List[Dict[str, str]], json_schema: Optional[Dict[str, Any]]):
        have_schema = json_schema is not None
        msgs = list(messages)

        # Inject schema hint like DeepSeek even for Responses API
        if have_schema:
            safe_schema = _lock_down_additional_props(json.loads(json.dumps(json_schema)))
            msgs = _inject_schema_hint_into_messages(msgs, safe_schema)
        reasoning = self.extra_inputs.get("reasoning")
        text_opts = self.extra_inputs.get("text")

        base_kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": msgs,
            "max_output_tokens": self.max_tokens,
        }
        if reasoning:
            base_kwargs["reasoning"] = reasoning
        if text_opts:
            base_kwargs["text"] = text_opts

        with_schema_kwargs = dict(base_kwargs)
        if have_schema:
            with_schema_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "schema",
                    "schema": safe_schema,
                    "strict": True,
                },
            }
        else:
            with_schema_kwargs["response_format"] = {"type": "text"}

        # 1st try with response_format (if schema)
        try:
            resp = self.client.responses.create(**with_schema_kwargs)
            text = _extract_text_from_responses_api(resp)
            if not have_schema:
                return {"text": text}
            parsed = _parse_with_salvage(text, want_schema=True)
            return parsed
        except BadRequestError:
            # Retry without response_format but keep hint
            resp = self.client.responses.create(**base_kwargs)
            text = _extract_text_from_responses_api(resp)
            if not have_schema:
                return {"text": text}
            parsed = _parse_with_salvage(text, want_schema=True)
            return parsed
        except TypeError:
            # Older SDKs → missing response_format support; retry bare
            resp = self.client.responses.create(**base_kwargs)
            text = _extract_text_from_responses_api(resp)
            if not have_schema:
                return {"text": text}
            parsed = _parse_with_salvage(text, want_schema=True)
            return parsed
        except Exception:
            # Final fallback
            resp = self.client.responses.create(**base_kwargs)
            text = _extract_text_from_responses_api(resp)
            if not have_schema:
                return {"text": text}
            parsed = _parse_with_salvage(text, want_schema=True)
            return parsed


__all__ = ["OpenAIClient"]

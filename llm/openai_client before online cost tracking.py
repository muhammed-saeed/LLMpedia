# # llm/openai_client.py
# from __future__ import annotations

# from typing import Any, Dict, List, Optional, Callable, TypeVar
# import json
# import time
# import random

# import httpx
# from openai import OpenAI
# from openai import (
#     BadRequestError,
#     APITimeoutError,
#     APIConnectionError,
#     RateLimitError,
#     APIError,
# )

# # ---------- helpers borrowed from DeepSeek client ----------

# def _schema_hint(schema: Dict[str, Any]) -> str:
#     return (
#         "Return ONLY one valid JSON object that matches this JSON Schema exactly. "
#         "No prose, no markdown, no code fences.\nSCHEMA:\n"
#         + json.dumps(schema, ensure_ascii=False)
#     )

# def _strip_fences(t: str) -> str:
#     s = (t or "").strip()
#     if s.startswith("```"):
#         nl = s.find("\n")
#         if nl != -1:
#             s = s[nl + 1 :].strip()
#         if s.endswith("```"):
#             s = s[:-3].strip()
#     return s

# def _best_json(text: str) -> Dict[str, Any]:
#     if not text:
#         return {}
#     # direct
#     try:
#         return json.loads(text)
#     except Exception:
#         pass
#     # strip fences
#     t = _strip_fences(text)
#     try:
#         return json.loads(t)
#     except Exception:
#         pass
#     # first balanced object
#     s = t.find("{")
#     if s != -1:
#         depth = 0
#         for i, ch in enumerate(t[s:], s):
#             if ch == "{":
#                 depth += 1
#             elif ch == "}":
#                 depth -= 1
#                 if depth == 0:
#                     try:
#                         return json.loads(t[s : i + 1])
#                     except Exception:
#                         break
#     return {}

# def _lock_down_additional_props(schema: Any) -> Any:
#     """
#     Recursively enforce additionalProperties:false on all object nodes.
#     This prevents OpenAI's 'additionalProperties is required and must be false' error.
#     """
#     if isinstance(schema, dict):
#         t = schema.get("type")
#         if t == "object":
#             schema.setdefault("additionalProperties", False)
#             props = schema.get("properties")
#             if isinstance(props, dict):
#                 for k in list(props.keys()):
#                     props[k] = _lock_down_additional_props(props[k])
#         elif t == "array":
#             if "items" in schema:
#                 schema["items"] = _lock_down_additional_props(schema["items"])
#     return schema

# def _inject_schema_hint_into_messages(
#     messages: List[Dict[str, str]],
#     json_schema: Dict[str, Any],
# ) -> List[Dict[str, str]]:
#     """
#     DeepSeek-style: put the schema contract into the system message so that even
#     if response_format isn't honored, the model is still told to output strict JSON.
#     """
#     msgs = list(messages)
#     hint = _schema_hint(json_schema)
#     if msgs and (msgs[0].get("role") == "system"):
#         msgs[0] = {"role": "system", "content": (msgs[0].get("content", "") + "\n\n" + hint)}
#     else:
#         msgs.insert(0, {"role": "system", "content": hint})
#     return msgs

# def _extract_text_from_chat(resp) -> str:
#     try:
#         return (resp.choices[0].message.content or "").strip()
#     except Exception:
#         return ""

# def _extract_text_from_responses_api(resp) -> str:
#     out = getattr(resp, "output_text", None)
#     if out:
#         return out.strip()
#     try:
#         parts: List[str] = []
#         for block in getattr(resp, "output", []) or []:
#             for c in getattr(block, "content", []) or []:
#                 txt = getattr(c, "text", "")
#                 if txt:
#                     parts.append(txt)
#         return "".join(parts).strip()
#     except Exception:
#         return ""

# def _parse_with_salvage(text: str, want_schema: bool) -> Dict[str, Any]:
#     if not want_schema:
#         return {"text": text}
#     try:
#         return json.loads(text)
#     except Exception:
#         pass
#     obj = _best_json(text)
#     if obj:
#         return obj
#     return {"_raw": text}

# # ---------- timeout + retry helpers ----------

# def _mk_timeout(timeout_s: float) -> httpx.Timeout:
#     """
#     IMPORTANT: This fixes your error. You were hitting connect timeout.
#     httpx defaults connect timeout to ~5s unless overridden.
#     """
#     t = float(timeout_s)
#     return httpx.Timeout(t, connect=t, read=t, write=t, pool=t)

# T = TypeVar("T")

# def _sleep_backoff(attempt: int, base: float = 1.0, cap: float = 30.0) -> None:
#     # exponential + jitter
#     delay = min(cap, base * (2 ** (attempt - 1))) + random.random()
#     time.sleep(delay)

# def _is_retryable_exc(e: Exception) -> bool:
#     return isinstance(
#         e,
#         (
#             APITimeoutError,
#             APIConnectionError,
#             RateLimitError,
#             APIError,
#             httpx.TimeoutException,
#             httpx.NetworkError,
#         ),
#     )

# # ---------- client ----------

# class OpenAIClient:
#     """
#     Unified OpenAI client that can call either:
#       • Chat Completions API
#       • Responses API (gpt-5 family)

#     Hardening:
#       - Inject schema hint into system message
#       - Lock additionalProperties:false recursively
#       - Salvage JSON if strict parsing fails
#       - Retry without response_format if provider rejects schema
#       - NEW: real connect/read/write/pool timeouts + retries/backoff
#       - NEW: supports __call__(..., timeout=...)
#     """

#     def __init__(
#         self,
#         model: str,
#         api_key: str,
#         base_url: Optional[str] = None,
#         max_tokens: Optional[int] = 1024,
#         temperature: Optional[float] = 0.0,
#         top_p: Optional[float] = 1.0,
#         use_responses_api: bool = False,
#         extra_inputs: Optional[Dict[str, Any]] = None,
#         request_timeout: Optional[float] = None,   # NEW
#         max_attempts: int = 4,                     # NEW
#         **_: Any,                                  # NEW: ignore unexpected kwargs from factory
#     ):
#         self.model = model
#         self.max_tokens = max_tokens
#         self.temperature = temperature
#         self.top_p = top_p
#         self.use_responses_api = bool(use_responses_api or (model or "").startswith("gpt-5"))
#         self.extra_inputs = extra_inputs or {}

#         self._default_timeout_s = float(request_timeout or 90.0)
#         self._max_attempts = max(1, int(max_attempts or 1))

#         client_kwargs: Dict[str, Any] = {
#             "api_key": api_key,
#             "timeout": _mk_timeout(self._default_timeout_s),  # GLOBAL default
#             "max_retries": 0,  # we control retries ourselves
#         }
#         if base_url:
#             client_kwargs["base_url"] = base_url

#         self.client = OpenAI(**client_kwargs)

#     def __call__(
#         self,
#         messages: List[Dict[str, str]],
#         json_schema: Optional[Dict[str, Any]] = None,
#         timeout: Optional[float] = None,  # NEW
#     ):
#         if self.use_responses_api:
#             return self._call_responses(messages, json_schema, timeout=timeout)
#         return self._call_chat(messages, json_schema, timeout=timeout)

#     # ---------------- internal retry wrapper ----------------

#     def _do_with_retries(self, fn: Callable[[], T]) -> T:
#         last_exc: Optional[Exception] = None
#         for attempt in range(1, self._max_attempts + 1):
#             try:
#                 return fn()
#             except BadRequestError:
#                 # schema / request is invalid -> do not retry
#                 raise
#             except Exception as e:
#                 last_exc = e
#                 if not _is_retryable_exc(e) or attempt >= self._max_attempts:
#                     raise
#                 _sleep_backoff(attempt, base=1.0, cap=30.0)
#         # unreachable, but keeps mypy happy
#         raise last_exc if last_exc else RuntimeError("request failed")

#     # ---------------- Chat Completions ----------------

#     def _call_chat(
#         self,
#         messages: List[Dict[str, str]],
#         json_schema: Optional[Dict[str, Any]],
#         timeout: Optional[float] = None,
#     ):
#         msgs = list(messages)
#         have_schema = json_schema is not None
#         safe_schema = None

#         req_timeout = _mk_timeout(timeout or self._default_timeout_s)

#         kwargs: Dict[str, Any] = dict(
#             model=self.model,
#             messages=msgs,
#             max_tokens=self.max_tokens,
#         )
#         if self.temperature is not None:
#             kwargs["temperature"] = self.temperature
#         if self.top_p is not None:
#             kwargs["top_p"] = self.top_p

#         if have_schema:
#             safe_schema = _lock_down_additional_props(json.loads(json.dumps(json_schema)))
#             msgs = _inject_schema_hint_into_messages(msgs, safe_schema)
#             kwargs["messages"] = msgs
#             kwargs["response_format"] = {
#                 "type": "json_schema",
#                 "json_schema": {"name": "schema", "schema": safe_schema, "strict": True},
#             }

#         def _call():
#             return self.client.chat.completions.create(**kwargs, timeout=req_timeout)

#         # Try with response_format first (if schema)
#         try:
#             resp = self._do_with_retries(_call)
#             text = _extract_text_from_chat(resp)
#             if not have_schema:
#                 return {"text": text}
#             return _parse_with_salvage(text, want_schema=True)

#         except BadRequestError:
#             # If provider rejects response_format schema, retry without response_format
#             if have_schema:
#                 kwargs.pop("response_format", None)

#                 def _call2():
#                     return self.client.chat.completions.create(**kwargs, timeout=req_timeout)

#                 resp = self._do_with_retries(_call2)
#                 text = _extract_text_from_chat(resp)
#                 return _parse_with_salvage(text, want_schema=True)
#             raise

#         except Exception:
#             # last resort: if schema, retry without response_format once
#             if have_schema:
#                 kwargs.pop("response_format", None)

#                 def _call3():
#                     return self.client.chat.completions.create(**kwargs, timeout=req_timeout)

#                 resp = self._do_with_retries(_call3)
#                 text = _extract_text_from_chat(resp)
#                 return _parse_with_salvage(text, want_schema=True)
#             raise

#     # ---------------- Responses API (gpt-5*) ----------------

#     def _call_responses(
#         self,
#         messages: List[Dict[str, str]],
#         json_schema: Optional[Dict[str, Any]],
#         timeout: Optional[float] = None,
#     ):
#         have_schema = json_schema is not None
#         msgs = list(messages)

#         req_timeout = _mk_timeout(timeout or self._default_timeout_s)

#         safe_schema = None
#         if have_schema:
#             safe_schema = _lock_down_additional_props(json.loads(json.dumps(json_schema)))
#             msgs = _inject_schema_hint_into_messages(msgs, safe_schema)

#         reasoning = self.extra_inputs.get("reasoning")
#         text_opts = self.extra_inputs.get("text")

#         base_kwargs: Dict[str, Any] = {
#             "model": self.model,
#             "input": msgs,
#             "max_output_tokens": self.max_tokens,
#         }
#         if reasoning:
#             base_kwargs["reasoning"] = reasoning
#         if text_opts:
#             base_kwargs["text"] = text_opts

#         with_schema_kwargs = dict(base_kwargs)
#         if have_schema:
#             with_schema_kwargs["response_format"] = {
#                 "type": "json_schema",
#                 "json_schema": {"name": "schema", "schema": safe_schema, "strict": True},
#             }
#         else:
#             with_schema_kwargs["response_format"] = {"type": "text"}

#         def _call():
#             return self.client.responses.create(**with_schema_kwargs, timeout=req_timeout)

#         try:
#             resp = self._do_with_retries(_call)
#             text = _extract_text_from_responses_api(resp)
#             if not have_schema:
#                 return {"text": text}
#             return _parse_with_salvage(text, want_schema=True)

#         except BadRequestError:
#             # Retry without response_format but keep hint
#             def _call2():
#                 return self.client.responses.create(**base_kwargs, timeout=req_timeout)

#             resp = self._do_with_retries(_call2)
#             text = _extract_text_from_responses_api(resp)
#             if not have_schema:
#                 return {"text": text}
#             return _parse_with_salvage(text, want_schema=True)

#         except TypeError:
#             # Older SDKs → missing response_format support; retry bare
#             def _call3():
#                 return self.client.responses.create(**base_kwargs, timeout=req_timeout)

#             resp = self._do_with_retries(_call3)
#             text = _extract_text_from_responses_api(resp)
#             if not have_schema:
#                 return {"text": text}
#             return _parse_with_salvage(text, want_schema=True)

#         except Exception:
#             # Final fallback
#             def _call4():
#                 return self.client.responses.create(**base_kwargs, timeout=req_timeout)

#             resp = self._do_with_retries(_call4)
#             text = _extract_text_from_responses_api(resp)
#             if not have_schema:
#                 return {"text": text}
#             return _parse_with_salvage(text, want_schema=True)

# __all__ = ["OpenAIClient"]

# llm/openai_client.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Callable, TypeVar
import json
import time
import random
import threading
import re

import httpx
from openai import OpenAI
from openai import (
    BadRequestError,
    APITimeoutError,
    APIConnectionError,
    RateLimitError,
    APIError,
)

# ---------- helpers borrowed from DeepSeek client ----------

def _schema_hint(schema: Dict[str, Any]) -> str:
    return (
        "Return ONLY one valid JSON object that matches this JSON Schema exactly. "
        "No prose, no markdown, no code fences.\nSCHEMA:\n"
        + json.dumps(schema, ensure_ascii=False)
    )

def _strip_fences(t: str) -> str:
    s = (t or "").strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            s = s[nl + 1 :].strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    return s

def _best_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    t = _strip_fences(text)
    try:
        return json.loads(t)
    except Exception:
        pass
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
                        return json.loads(t[s : i + 1])
                    except Exception:
                        break
    return {}

def _lock_down_additional_props(schema: Any) -> Any:
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
    return schema

def _inject_schema_hint_into_messages(
    messages: List[Dict[str, str]],
    json_schema: Dict[str, Any],
) -> List[Dict[str, str]]:
    msgs = list(messages)
    hint = _schema_hint(json_schema)
    if msgs and (msgs[0].get("role") == "system"):
        msgs[0] = {"role": "system", "content": (msgs[0].get("content", "") + "\n\n" + hint)}
    else:
        msgs.insert(0, {"role": "system", "content": hint})
    return msgs

def _extract_text_from_chat(resp) -> str:
    try:
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""

def _extract_text_from_responses_api(resp) -> str:
    out = getattr(resp, "output_text", None)
    if out:
        return out.strip()
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
    try:
        return json.loads(text)
    except Exception:
        pass
    obj = _best_json(text)
    if obj:
        return obj
    return {"_raw": text}

# ---------- timeout + retry helpers ----------

def _mk_timeout(timeout_s: float) -> httpx.Timeout:
    t = float(timeout_s)
    return httpx.Timeout(t, connect=t, read=t, write=t, pool=t)

T = TypeVar("T")

def _sleep_backoff(attempt: int, base: float = 1.0, cap: float = 30.0) -> None:
    delay = min(cap, base * (2 ** (attempt - 1))) + random.random()
    time.sleep(delay)

def _is_retryable_exc(e: Exception) -> bool:
    return isinstance(
        e,
        (
            APITimeoutError,
            APIConnectionError,
            RateLimitError,
            APIError,
            httpx.TimeoutException,
            httpx.NetworkError,
        ),
    )


# ============================================================
# Token Bucket Rate Limiter  (thread-safe, shared per endpoint)
# ============================================================

class TokenBucketRateLimiter:
    """
    Thread-safe token bucket rate limiter.

    How it works:
      - The bucket holds up to `effective_rpm` tokens (= hard_limit * safety).
      - Tokens refill at effective_rpm / 60 per second.
      - Each API call consumes 1 token.
      - If empty, the caller sleeps until a token is available.

    Example: hard_limit=30, safety=0.85
      → effective = 25.5 req/min  →  ~1 request every 2.35s steady-state.
      Bucket starts full so the first burst of ~25 goes through instantly,
      then it throttles to the steady rate.
    """

    def __init__(
        self,
        max_requests_per_minute: float = 30.0,
        safety_margin: float = 0.85,
    ):
        effective_rpm = max_requests_per_minute * safety_margin
        self._max_tokens = effective_rpm
        self._refill_rate = effective_rpm / 60.0   # tokens/sec
        self._tokens = self._max_tokens            # start full
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()
        self._stated_limit = max_requests_per_minute
        self._effective_rpm = effective_rpm

    def acquire(self, timeout: float = 300.0) -> bool:
        """Block until a token is available.  Returns False on timeout."""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
                wait = (1.0 - self._tokens) / self._refill_rate
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(wait + 0.01, remaining))

    def _refill(self):
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._last_refill = now
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._refill_rate)

    @property
    def effective_rpm(self) -> float:
        return self._effective_rpm


# ── Global registry: one limiter per (base_url, api_key) ──

_RL_REGISTRY: Dict[str, TokenBucketRateLimiter] = {}
_RL_LOCK = threading.Lock()


def _get_or_create_rate_limiter(
    base_url: Optional[str],
    api_key: str,
    max_rpm: float,
    safety_margin: float = 0.85,
) -> Optional[TokenBucketRateLimiter]:
    """
    Get or create a *shared* rate limiter for a (base_url, api_key) pair.
    All OpenAIClient instances hitting the same endpoint share ONE limiter
    so the global request rate is controlled regardless of how many workers
    or OpenAIClient objects exist.

    Returns None if max_rpm <= 0 (rate limiting disabled).
    """
    if max_rpm <= 0:
        return None
    key = f"{base_url or 'default'}::{(api_key or '')[-8:]}"
    with _RL_LOCK:
        if key not in _RL_REGISTRY:
            limiter = TokenBucketRateLimiter(
                max_requests_per_minute=max_rpm,
                safety_margin=safety_margin,
            )
            _RL_REGISTRY[key] = limiter
            print(
                f"[rate-limiter] NEW  endpoint={base_url or 'openai'}  "
                f"hard_limit={max_rpm}/min  effective={limiter.effective_rpm:.1f}/min  "
                f"(safety={safety_margin:.0%})",
                flush=True,
            )
        return _RL_REGISTRY[key]


def _parse_rpm_from_429(exc: RateLimitError) -> Optional[float]:
    """
    Try to extract 'Current limit: N' from a 429 error message
    so we can auto-create a rate limiter even if the user didn't set one.
    """
    try:
        msg = str(exc)
        m = re.search(r"Current limit:\s*(\d+)", msg)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None


# ---------- client ----------

class OpenAIClient:
    """
    Unified OpenAI client — Chat Completions + Responses API (gpt-5).

    Rate limiting (NEW):
      Pass max_requests_per_minute=30  to pre-configure the limiter.
      Or leave it at 0 and the limiter auto-creates itself from the
      first 429 error's "Current limit: N" field.
      The limiter is shared across ALL client instances with the same
      (base_url, api_key) so it works with any number of workers.
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
        request_timeout: Optional[float] = None,
        max_attempts: int = 4,
        # ── Rate limiting ────────────────────────────────────
        max_requests_per_minute: float = 0,       # 0 = auto-detect on first 429
        rate_limit_safety_margin: float = 0.85,   # use 85% of stated limit
        **_: Any,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.use_responses_api = bool(use_responses_api or (model or "").startswith("gpt-5"))
        self.extra_inputs = extra_inputs or {}

        self._default_timeout_s = float(request_timeout or 90.0)
        self._max_attempts = max(1, int(max_attempts or 1))
        self._base_url = base_url
        self._api_key = api_key
        self._safety_margin = float(rate_limit_safety_margin or 0.85)

        client_kwargs: Dict[str, Any] = {
            "api_key": api_key,
            "timeout": _mk_timeout(self._default_timeout_s),
            "max_retries": 0,
        }
        if base_url:
            client_kwargs["base_url"] = base_url

        self.client = OpenAI(**client_kwargs)

        # ── Rate limiter (shared across all instances with same endpoint) ──
        self._rate_limiter: Optional[TokenBucketRateLimiter] = _get_or_create_rate_limiter(
            base_url=base_url,
            api_key=api_key,
            max_rpm=float(max_requests_per_minute or 0),
            safety_margin=self._safety_margin,
        )

    # ── Rate limit helpers ──

    def _wait_for_rate_limit(self):
        """Block until the rate limiter allows a request. No-op if disabled."""
        if self._rate_limiter is not None:
            self._rate_limiter.acquire(timeout=300.0)

    def _maybe_auto_create_limiter(self, exc: RateLimitError):
        """
        If we don't have a limiter yet and the 429 tells us the limit,
        auto-create one so ALL subsequent requests are throttled.
        """
        if self._rate_limiter is not None:
            return
        rpm = _parse_rpm_from_429(exc)
        if rpm and rpm > 0:
            print(
                f"[rate-limiter] Auto-detected from 429 error: {rpm} req/min",
                flush=True,
            )
            self._rate_limiter = _get_or_create_rate_limiter(
                base_url=self._base_url,
                api_key=self._api_key,
                max_rpm=rpm,
                safety_margin=self._safety_margin,
            )

    def __call__(
        self,
        messages: List[Dict[str, str]],
        json_schema: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ):
        if self.use_responses_api:
            return self._call_responses(messages, json_schema, timeout=timeout)
        return self._call_chat(messages, json_schema, timeout=timeout)

    # ---------------- internal retry wrapper ----------------

    def _do_with_retries(self, fn: Callable[[], T]) -> T:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                # ── Wait for rate limiter BEFORE every attempt ──
                self._wait_for_rate_limit()
                return fn()
            except BadRequestError:
                raise
            except RateLimitError as e:
                last_exc = e
                self._maybe_auto_create_limiter(e)
                if attempt >= self._max_attempts:
                    raise
                wait = self._parse_reset_wait(e, default=max(4.0, 2.0 * (2 ** attempt)))
                print(
                    f"[rate-limiter] 429 (attempt {attempt}/{self._max_attempts}), "
                    f"sleeping {wait:.1f}s",
                    flush=True,
                )
                time.sleep(wait)
            except Exception as e:
                last_exc = e
                if not _is_retryable_exc(e) or attempt >= self._max_attempts:
                    raise
                _sleep_backoff(attempt, base=1.0, cap=30.0)
        raise last_exc if last_exc else RuntimeError("request failed")

    @staticmethod
    def _parse_reset_wait(exc: RateLimitError, default: float = 8.0) -> float:
        """Parse 'Limit resets at: <UTC timestamp>' from 429 to compute exact wait."""
        try:
            msg = str(exc)
            m = re.search(r"Limit resets at:\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", msg)
            if m:
                from datetime import datetime, timezone
                reset_dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=timezone.utc
                )
                now = datetime.now(timezone.utc)
                wait = (reset_dt - now).total_seconds()
                if 0 < wait < 120:
                    return wait + 0.5
        except Exception:
            pass
        return default

    # ---------------- Chat Completions ----------------

    def _call_chat(
        self,
        messages: List[Dict[str, str]],
        json_schema: Optional[Dict[str, Any]],
        timeout: Optional[float] = None,
    ):
        msgs = list(messages)
        have_schema = json_schema is not None
        safe_schema = None

        req_timeout = _mk_timeout(timeout or self._default_timeout_s)

        kwargs: Dict[str, Any] = dict(
            model=self.model,
            messages=msgs,
            max_tokens=self.max_tokens,
        )
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p

        if have_schema:
            safe_schema = _lock_down_additional_props(json.loads(json.dumps(json_schema)))
            msgs = _inject_schema_hint_into_messages(msgs, safe_schema)
            kwargs["messages"] = msgs
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "schema", "schema": safe_schema, "strict": True},
            }

        def _call():
            return self.client.chat.completions.create(**kwargs, timeout=req_timeout)

        try:
            resp = self._do_with_retries(_call)
            text = _extract_text_from_chat(resp)
            if not have_schema:
                return {"text": text}
            return _parse_with_salvage(text, want_schema=True)

        except BadRequestError:
            if have_schema:
                kwargs.pop("response_format", None)
                def _call2():
                    return self.client.chat.completions.create(**kwargs, timeout=req_timeout)
                resp = self._do_with_retries(_call2)
                text = _extract_text_from_chat(resp)
                return _parse_with_salvage(text, want_schema=True)
            raise

        except Exception:
            if have_schema:
                kwargs.pop("response_format", None)
                def _call3():
                    return self.client.chat.completions.create(**kwargs, timeout=req_timeout)
                resp = self._do_with_retries(_call3)
                text = _extract_text_from_chat(resp)
                return _parse_with_salvage(text, want_schema=True)
            raise

    # ---------------- Responses API (gpt-5*) ----------------

    def _call_responses(
        self,
        messages: List[Dict[str, str]],
        json_schema: Optional[Dict[str, Any]],
        timeout: Optional[float] = None,
    ):
        have_schema = json_schema is not None
        msgs = list(messages)

        req_timeout = _mk_timeout(timeout or self._default_timeout_s)

        safe_schema = None
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
                "json_schema": {"name": "schema", "schema": safe_schema, "strict": True},
            }
        else:
            with_schema_kwargs["response_format"] = {"type": "text"}

        def _call():
            return self.client.responses.create(**with_schema_kwargs, timeout=req_timeout)

        try:
            resp = self._do_with_retries(_call)
            text = _extract_text_from_responses_api(resp)
            if not have_schema:
                return {"text": text}
            return _parse_with_salvage(text, want_schema=True)

        except BadRequestError:
            def _call2():
                return self.client.responses.create(**base_kwargs, timeout=req_timeout)
            resp = self._do_with_retries(_call2)
            text = _extract_text_from_responses_api(resp)
            if not have_schema:
                return {"text": text}
            return _parse_with_salvage(text, want_schema=True)

        except TypeError:
            def _call3():
                return self.client.responses.create(**base_kwargs, timeout=req_timeout)
            resp = self._do_with_retries(_call3)
            text = _extract_text_from_responses_api(resp)
            if not have_schema:
                return {"text": text}
            return _parse_with_salvage(text, want_schema=True)

        except Exception:
            def _call4():
                return self.client.responses.create(**base_kwargs, timeout=req_timeout)
            resp = self._do_with_retries(_call4)
            text = _extract_text_from_responses_api(resp)
            if not have_schema:
                return {"text": text}
            return _parse_with_salvage(text, want_schema=True)

__all__ = ["OpenAIClient"]
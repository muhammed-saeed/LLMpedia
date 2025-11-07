# # llm/replicate_client.py
# from __future__ import annotations

# import os
# import json
# import time
# import random
# from typing import Any, Dict, List, Optional, Generator

# from dotenv import load_dotenv
# import replicate

# # transient network exceptions
# import httpx
# import httpcore

# # --- your shared util (unchanged import path) ---
# from llm.json_utils import best_json


# # -------------------------- small helpers --------------------------

# def _minify_schema(schema: Dict[str, Any]) -> str:
#     try:
#         return json.dumps(schema, separators=(",", ":"), ensure_ascii=False)
#     except Exception:
#         return "{}"

# def _collapse_messages(messages: List[Dict[str, str]]) -> str:
#     parts = []
#     for m in messages:
#         role = (m.get("role") or "user").upper()
#         content = (m.get("content") or "").strip()
#         parts.append(f"{role}: {content}")
#     parts.append("ASSISTANT:")
#     return "\n\n".join(parts)

# def _strip_fences(text: str) -> str:
#     t = (text or "").strip()
#     if t.startswith("```"):
#         nl = t.find("\n")
#         if nl != -1:
#             t = t[nl + 1:].strip()
#         if t.endswith("```"):
#             t = t[:-3].strip()
#     return t

# def _parse_json_best_effort(text: str) -> Dict[str, Any]:
#     obj = best_json(text)
#     return obj if isinstance(obj, dict) else {}

# def _clip01(x: Any, default: float = 0.9) -> float:
#     try:
#         v = float(x)
#     except Exception:
#         return default
#     if v < 0.0: return 0.0
#     if v > 1.0: return 1.0
#     return v

# def _coerce_elicit(obj: Dict[str, Any], *, calibrated: bool) -> Dict[str, Any]:
#     facts = obj.get("facts")
#     if not isinstance(facts, list):
#         return {"facts": []}
#     out = []
#     for it in facts:
#         if not isinstance(it, dict):
#             continue
#         s = it.get("subject"); p = it.get("predicate"); o = it.get("object")
#         if not (isinstance(s, str) and isinstance(p, str) and (isinstance(o, str) or isinstance(o, (int, float, bool)))):
#             continue
#         if not isinstance(o, str):
#             o = str(o)
#         if calibrated:
#             conf = _clip01(it.get("confidence"), 0.9)
#             out.append({"subject": s, "predicate": p, "object": o, "confidence": conf})
#         else:
#             out.append({"subject": s, "predicate": p, "object": o})
#     return {"facts": out}

# def _coerce_ner(obj: Dict[str, Any], *, calibrated: bool) -> Dict[str, Any]:
#     phs = obj.get("phrases")
#     if not isinstance(phs, list):
#         return {"phrases": []}
#     out = []
#     for it in phs:
#         if not isinstance(it, dict):
#             continue
#         phrase = it.get("phrase"); is_ne = bool(it.get("is_ne"))
#         if not isinstance(phrase, str):
#             continue
#         if calibrated:
#             conf = _clip01(it.get("confidence"), 0.9)
#             out.append({"phrase": phrase, "is_ne": is_ne, "confidence": conf})
#         else:
#             out.append({"phrase": phrase, "is_ne": is_ne})
#     return {"phrases": out}

# def _salvage_block(text: str, key: Optional[str]) -> Dict[str, Any]:
#     """
#     Try best_json first; if it returns an array and we expect a key (like 'facts'),
#     wrap it; else return {}.
#     (NOTE: parameter is 'key' to match calls; we also accept legacy 'expect_key' via wrapper below.)
#     """
#     obj = best_json(text)
#     if isinstance(obj, dict):
#         # already object — either conforms, or still usable downstream
#         return obj
#     if isinstance(obj, list) and key:
#         return {key: obj}
#     return {}

# # Backward-compat wrapper in case other code calls with expect_key=
# def _salvage_block_expect_key(text: str, expect_key: Optional[str]) -> Dict[str, Any]:
#     return _salvage_block(text, expect_key)

# # -------------------------- client --------------------------

# class ReplicateLLM:
#     """
#     Replicate wrapper with model-specific prompt shaping and robust JSON salvage.
#     Also implements __call__(messages, json_schema=...) to match other clients.
#     Includes jittered exponential backoff for transient HTTP faults.
#     """

#     def __init__(self, model: str, *, api_token: Optional[str] = None, default_extra: Optional[Dict[str, Any]] = None):
#         load_dotenv()
#         self.model = model
#         token = api_token or os.getenv("REPLICATE_API_TOKEN")
#         if not token:
#             raise RuntimeError("Missing REPLICATE_API_TOKEN in environment (or pass api_token=...).")
#         self._client = replicate.Client(api_token=token)
#         self._debug = os.getenv("REPLICATE_DEBUG", "") == "1"
#         self._default_extra = default_extra or {}

#     # Let the object be called like other clients:
#     def __call__(self, messages: List[Dict[str, str]], *, json_schema: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any] | str:
#         return self.generate(messages, json_schema=json_schema, **kwargs)

#     # --------- builders ---------

#     def _inputs_common(
#         self,
#         *,
#         temperature: Optional[float],
#         top_p: Optional[float],
#         top_k: Optional[int],
#         max_tokens: Optional[int],
#         seed: Optional[int],
#         extra: Dict[str, Any],
#     ) -> Dict[str, Any]:
#         # merge defaults + per-call extras
#         merged_extra = {**(self._default_extra or {}), **(extra or {})}

#         inp: Dict[str, Any] = {}
#         if temperature is not None: inp["temperature"] = temperature
#         if top_p is not None: inp["top_p"] = top_p
#         if top_k is not None: inp["top_k"] = top_k
#         if max_tokens is not None:
#             inp["max_tokens"] = max_tokens
#             inp["max_output_tokens"] = max_tokens
#         if seed is not None: inp["seed"] = seed

#         # Replicate quirk: some runners expect scalar strings for stop / stop_sequences
#         if "stop_sequences" in merged_extra and isinstance(merged_extra["stop_sequences"], list):
#             merged_extra = {**merged_extra, "stop_sequences": merged_extra["stop_sequences"][0] if merged_extra["stop_sequences"] else ""}
#         if "stop" in merged_extra and isinstance(merged_extra["stop"], list):
#             merged_extra = {**merged_extra, "stop": merged_extra["stop"][0] if merged_extra["stop"] else ""}

#         for k, v in (merged_extra or {}).items():
#             if v is not None:
#                 inp[k] = v
#         return inp

#     def _build_for_gemini(self, messages, json_schema, knobs) -> Dict[str, Any]:
#         schema_min = _minify_schema(json_schema)
#         system_prompt = (
#             "Return ONLY a single valid JSON object that matches this JSON Schema exactly. "
#             "No prose, no markdown, no code fences.\n"
#             f"SCHEMA: {schema_min}\n"
#             "If you truly don't know, return an empty but valid object per schema."
#         )
#         fewshot = (
#             "EXAMPLE:\n"
#             'USER: Subject: Ping\n'
#             'ASSISTANT: {"facts":[{"subject":"Ping","predicate":"instanceOf","object":"entity","confidence":1.0}]}\n\n'
#         )
#         prompt = fewshot + _collapse_messages(messages)
#         knobs.setdefault("temperature", 0.2)
#         knobs.setdefault("top_p", 0.9)
#         return {"prompt": prompt, "system_prompt": system_prompt, **knobs}

#     def _build_for_grok_messages(self, messages, json_schema, knobs) -> Dict[str, Any]:
#         schema_min = _minify_schema(json_schema)
#         sys_msg = {
#             "role": "system",
#             "content": (
#                 "You are a JSON function. Return ONLY one JSON object validating this schema. "
#                 "No prose/markdown/code fences. If unsure, return an empty—but valid—object.\n"
#                 f"SCHEMA: {schema_min}"
#             ),
#         }
#         usr_msg = {"role": "user", "content": _collapse_messages(messages)}
#         inputs = {"messages": [sys_msg, usr_msg]}
#         for k in ("temperature", "top_p", "top_k", "max_tokens", "max_output_tokens", "seed"):
#             if k in knobs:
#                 inputs[k] = knobs[k]
#         return inputs

#     def _build_for_qwen_prompt(self, messages, json_schema, knobs) -> Dict[str, Any]:
#         schema_min = _minify_schema(json_schema)
#         fewshot = (
#             "You must output ONE JSON object that VALIDATES this JSON Schema.\n"
#             "NO prose, NO markdown, NO code fences.\n"
#             f"SCHEMA: {schema_min}\n\n"
#             "EXAMPLE:\n"
#             'USER: Subject: Ping\n'
#             'ASSISTANT: {"facts":[{"subject":"Ping","predicate":"instanceOf","object":"entity","confidence":0.99}]}\n\n'
#         )
#         task = _collapse_messages(messages)
#         contract = (
#             "If you know the subject, produce 12–40 concise triples (no duplicates). "
#             'Always include at least one triple with predicate "instanceOf". '
#             'If uncertain overall, return {"facts":[]}.'
#         )
#         prompt = f"{fewshot}{task}\n\n{contract}"
#         knobs.setdefault("temperature", 0.3)
#         knobs.setdefault("top_p", 0.9)
#         knobs.setdefault("max_tokens", knobs.get("max_output_tokens", 1536))
#         return {"prompt": prompt, **knobs}

#     def _build_inputs(self, messages, json_schema, knobs) -> Dict[str, Any]:
#         is_gemini = self.model.startswith("google/gemini")
#         is_grok = self.model.startswith("xai/grok-4") or "grok-4" in self.model
#         is_qwen = self.model.startswith("qwen/")

#         if json_schema:
#             if is_gemini:
#                 return self._build_for_gemini(messages, json_schema, knobs)
#             if is_grok:
#                 return self._build_for_grok_messages(messages, json_schema, knobs)
#             if is_qwen:
#                 return self._build_for_qwen_prompt(messages, json_schema, knobs)
#             schema_min = _minify_schema(json_schema)
#             system_prompt = (
#                 "Return ONLY a single valid JSON object matching this schema. "
#                 "No prose, no markdown, no code fences.\n"
#                 f"SCHEMA: {schema_min}"
#             )
#             prompt = _collapse_messages(messages)
#             return {"prompt": prompt, "system_prompt": system_prompt, **knobs}
#         return {"prompt": _collapse_messages(messages), **knobs}

#     # --------- internal resilient wrappers ---------

#     def _blocking_once(self, inputs: Dict[str, Any]) -> str:
#         transient = (
#             httpx.TimeoutException,
#             httpx.ConnectError,
#             httpx.ReadError,
#             httpx.RemoteProtocolError,
#             httpcore.RemoteProtocolError,
#             httpcore.WriteError,
#             httpcore.ReadTimeout,
#             httpcore.ConnectTimeout,
#         )
#         delay = 0.8
#         max_tries = 6
#         last_err: Optional[BaseException] = None
#         for attempt in range(1, max_tries + 1):
#             try:
#                 pred = self._client.predictions.create(model=self.model, input=inputs)
#                 pred.wait()
#                 out = pred.output
#                 return "".join(out) if isinstance(out, list) else (out or "")
#             except transient as e:
#                 last_err = e
#                 if self._debug:
#                     print(f"[replicate][retry {attempt}/{max_tries}] {type(e).__name__}: {e}", flush=True)
#                 if attempt == max_tries:
#                     raise
#                 time.sleep(delay + random.random() * 0.3)
#                 delay = min(delay * 1.8, 10.0)
#             except Exception:
#                 raise
#         raise last_err or RuntimeError("replicate _blocking_once failed without exception")

#     def _stream_once(self, inputs: Dict[str, Any]) -> str:
#         transient = (
#             httpx.TimeoutException,
#             httpx.ConnectError,
#             httpx.ReadError,
#             httpx.RemoteProtocolError,
#             httpcore.RemoteProtocolError,
#             httpcore.WriteError,
#             httpcore.ReadTimeout,
#             httpcore.ConnectTimeout,
#         )
#         delay = 0.8
#         max_tries = 6
#         last_err: Optional[BaseException] = None
#         for attempt in range(1, max_tries + 1):
#             try:
#                 chunks: List[str] = []
#                 for event in replicate.stream(self.model, input=inputs):
#                     chunks.append(str(event))
#                 return "".join(chunks)
#             except transient as e:
#                 last_err = e
#                 if self._debug:
#                     print(f"[replicate][stream retry {attempt}/{max_tries}] {type(e).__name__}: {e}", flush=True)
#                 if attempt == max_tries:
#                     raise
#                 time.sleep(delay + random.random() * 0.3)
#                 delay = min(delay * 1.8, 10.0)
#             except Exception:
#                 raise
#         raise last_err or RuntimeError("replicate _stream_once failed without exception")

#     # --------- schema-based coercion ---------

#     def _coerce_by_schema(self, obj: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
#         props = (schema.get("properties") or {})
#         if "facts" in props:
#             calibrated = "confidence" in (props["facts"]["items"]["properties"] or {})
#             return _coerce_elicit(obj, calibrated=calibrated)
#         if "phrases" in props:
#             calibrated = "confidence" in (props["phrases"]["items"]["properties"] or {})
#             return _coerce_ner(obj, calibrated=calibrated)
#         return obj if isinstance(obj, dict) else {}

#     # --------- public blocking API ---------

#     def ping(self) -> Dict[str, Any]:
#         inp = {"prompt": 'Return ONLY this exact JSON: {"message":"PONG"}', "max_tokens": 32, "temperature": 0}
#         txt = self._blocking_once(inp)
#         obj = _parse_json_best_effort(txt)
#         return obj if obj else {"message": "PONG"}

#     def generate(
#         self,
#         messages: List[Dict[str, str]],
#         *,
#         json_schema: Optional[Dict[str, Any]] = None,
#         temperature: Optional[float] = None,
#         top_p: Optional[float] = None,
#         top_k: Optional[int] = None,
#         max_tokens: Optional[int] = None,
#         seed: Optional[int] = None,
#         extra: Optional[Dict[str, Any]] = None,
#     ) -> Dict[str, Any]:
#         knobs = self._inputs_common(
#             temperature=temperature, top_p=top_p, top_k=top_k,
#             max_tokens=max_tokens, seed=seed, extra=extra or {},
#         )
#         inputs = self._build_inputs(messages, json_schema, knobs)

#         if not json_schema:
#             text = self._blocking_once(inputs)
#             if self._debug:
#                 print("\n[replicate][raw output]\n" + text[:4000] + ("\n" if len(text) else ""), flush=True)
#             return {"text": text, "_raw": text}

#         props = (json_schema.get("properties") or {})
#         expect = "facts" if "facts" in props else ("phrases" if "phrases" in props else None)

#         is_grok = self.model.startswith("xai/grok-4") or "grok-4" in self.model

#         if is_grok:
#             text = self._stream_once(inputs)
#             if self._debug:
#                 print("\n[replicate][raw stream (grok)]\n" + text[:4000] + ("\n" if len(text) else ""), flush=True)
#             # accept both 'key=' and legacy 'expect_key=' styles
#             parsed = _salvage_block(text, key=expect)
#             result = self._coerce_by_schema(parsed, json_schema)
#             result["_raw"] = text
#             return result

#         text = self._blocking_once(inputs)
#         if self._debug:
#             print("\n[replicate][raw output]\n" + text[:4000] + ("\n" if len(text) else ""), flush=True)

#         # accept both names to avoid mismatches from older call sites
#         parsed = _salvage_block(text, key=expect)
#         if not parsed:
#             parsed = _salvage_block_expect_key(text, expect_key=expect)

#         if parsed:
#             result = self._coerce_by_schema(parsed, json_schema)
#             result["_raw"] = text
#             return result

#         # final fallback: just coerce empty object so caller gets schema shape
#         result = self._coerce_by_schema({}, json_schema)
#         result["_raw"] = text
#         return result

#     # --------- streaming API ---------

#     def stream_text(
#         self,
#         messages: List[Dict[str, str]],
#         *,
#         temperature: Optional[float] = None,
#         top_p: Optional[float] = None,
#         top_k: Optional[int] = None,
#         max_tokens: Optional[int] = None,
#         seed: Optional[int] = None,
#         extra: Optional[Dict[str, Any]] = None,
#     ) -> Generator[str, None, None]:
#         knobs = self._inputs_common(
#             temperature=temperature, top_p=top_p, top_k=top_k,
#             max_tokens=max_tokens, seed=seed, extra=extra or {},
#         )
#         inputs = self._build_inputs(messages, json_schema=None, knobs=knobs)
#         # resilient streaming
#         transient = (
#             httpx.TimeoutException,
#             httpx.ConnectError,
#             httpx.ReadError,
#             httpx.RemoteProtocolError,
#             httpcore.RemoteProtocolError,
#             httpcore.WriteError,
#             httpcore.ReadTimeout,
#             httpcore.ConnectTimeout,
#         )
#         delay = 0.8
#         max_tries = 6
#         attempt = 1
#         while True:
#             try:
#                 for event in replicate.stream(self.model, input=inputs):
#                     yield str(event)
#                 break
#             except transient as e:
#                 if self._debug:
#                     print(f"[replicate][stream_text retry {attempt}/{max_tries}] {type(e).__name__}: {e}", flush=True)
#                 if attempt >= max_tries:
#                     raise
#                 time.sleep(delay + random.random() * 0.3)
#                 delay = min(delay * 1.8, 10.0)
#                 attempt += 1

#     def stream_json(
#         self,
#         messages: List[Dict[str, str]],
#         *,
#         json_schema: Dict[str, Any],
#         temperature: Optional[float] = None,
#         top_p: Optional[float] = None,
#         top_k: Optional[int] = None,
#         max_tokens: Optional[int] = None,
#         seed: Optional[int] = None,
#         extra: Optional[Dict[str, Any]] = None,
#     ) -> Generator[Dict[str, Any], None, None]:
#         buffer: List[str] = []
#         knobs = self._inputs_common(
#             temperature=temperature, top_p=top_p, top_k=top_k,
#             max_tokens=max_tokens, seed=seed, extra=extra or {},
#         )
#         inputs = self._build_inputs(messages, json_schema=json_schema, knobs=knobs)
#         # resilient stream collect
#         text = ""
#         transient = (
#             httpx.TimeoutException,
#             httpx.ConnectError,
#             httpx.ReadError,
#             httpx.RemoteProtocolError,
#             httpcore.RemoteProtocolError,
#             httpcore.WriteError,
#             httpcore.ReadTimeout,
#             httpcore.ConnectTimeout,
#         )
#         delay = 0.8
#         max_tries = 6
#         for attempt in range(1, max_tries + 1):
#             try:
#                 buffer.clear()
#                 for event in replicate.stream(self.model, input=inputs):
#                     buffer.append(str(event))
#                 text = "".join(buffer)
#                 break
#             except transient as e:
#                 if self._debug:
#                     print(f"[replicate][stream_json retry {attempt}/{max_tries}] {type(e).__name__}: {e}", flush=True)
#                 if attempt == max_tries:
#                     raise
#                 time.sleep(delay + random.random() * 0.3)
#                 delay = min(delay * 1.8, 10.0)

#         if self._debug:
#             print("\n[replicate][raw stream combined]\n" + text[:4000] + ("\n" if len(text) else ""), flush=True)

#         props = (json_schema.get("properties") or {})
#         expect = "facts" if "facts" in props else ("phrases" if "phrases" in props else None)
#         parsed = _salvage_block(text, key=expect) or _salvage_block_expect_key(text, expect_key=expect)
#         result = self._coerce_by_schema(parsed, json_schema)
#         result["_raw"] = text
#         yield result

# llm/replicate_client.py
from __future__ import annotations

import os
import json
import time
import random
from typing import Any, Dict, List, Optional, Generator

from dotenv import load_dotenv
import replicate

# transient network exceptions
import httpx
import httpcore

# --- shared util ---
from llm.json_utils import best_json


# -------------------------- small helpers --------------------------

def _minify_schema(schema: Dict[str, Any]) -> str:
    try:
        return json.dumps(schema, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return "{}"

def _collapse_messages(messages: List[Dict[str, str]]) -> str:
    """
    Generic chat collapse used by most runners that accept a 'prompt' string,
    while still preserving roles for readability.
    """
    parts = []
    for m in messages:
        role = (m.get("role") or "user").upper()
        content = (m.get("content") or "").strip()
        parts.append(f"{role}: {content}")
    parts.append("ASSISTANT:")
    return "\n\n".join(parts)

def _collapse_single_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Collapse chat messages into ONE prompt for single-prompt-only models
    (e.g., openai/gpt-oss-*). We keep explicit headers and end with 'Assistant:'.
    """
    sys_parts: List[str] = []
    convo_parts: List[str] = []

    for m in messages or []:
        role = (m.get("role") or "user").strip().lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            sys_parts.append(content)
        elif role == "assistant":
            convo_parts.append(f"Assistant: {content}")
        else:
            # default to user
            convo_parts.append(f"User: {content}")

    out: List[str] = []
    if sys_parts:
        out.append("\n\n".join(f"System: {p}" for p in sys_parts))
    if convo_parts:
        out.append("\n\n".join(convo_parts))
    out.append("Assistant:")
    return "\n\n".join(out).strip()

def _strip_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        nl = t.find("\n")
        if nl != -1:
            t = t[nl + 1:].strip()
        if t.endswith("```"):
            t = t[:-3].strip()
    return t

def _parse_json_best_effort(text: str) -> Dict[str, Any]:
    obj = best_json(text)
    return obj if isinstance(obj, dict) else {}

def _clip01(x: Any, default: float = 0.9) -> float:
    try:
        v = float(x)
    except Exception:
        return default
    if v < 0.0: return 0.0
    if v > 1.0: return 1.0
    return v

def _coerce_elicit(obj: Dict[str, Any], *, calibrated: bool) -> Dict[str, Any]:
    facts = obj.get("facts")
    if not isinstance(facts, list):
        return {"facts": []}
    out = []
    for it in facts:
        if not isinstance(it, dict):
            continue
        s = it.get("subject"); p = it.get("predicate"); o = it.get("object")
        if not (isinstance(s, str) and isinstance(p, str) and (isinstance(o, str) or isinstance(o, (int, float, bool)))):
            continue
        if not isinstance(o, str):
            o = str(o)
        if calibrated:
            conf = _clip01(it.get("confidence"), 0.9)
            out.append({"subject": s, "predicate": p, "object": o, "confidence": conf})
        else:
            out.append({"subject": s, "predicate": p, "object": o})
    return {"facts": out}

def _coerce_ner(obj: Dict[str, Any], *, calibrated: bool) -> Dict[str, Any]:
    phs = obj.get("phrases")
    if not isinstance(phs, list):
        return {"phrases": []}
    out = []
    for it in phs:
        if not isinstance(it, dict):
            continue
        phrase = it.get("phrase"); is_ne = bool(it.get("is_ne"))
        if not isinstance(phrase, str):
            continue
        if calibrated:
            conf = _clip01(it.get("confidence"), 0.9)
            out.append({"phrase": phrase, "is_ne": is_ne, "confidence": conf})
        else:
            out.append({"phrase": phrase, "is_ne": is_ne})
    return {"phrases": out}

def _salvage_block(text: str, key: Optional[str]) -> Dict[str, Any]:
    """
    Try best_json first; if it returns an array and we expect a key (like 'facts'),
    wrap it; else return {}.
    """
    obj = best_json(text)
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list) and key:
        return {key: obj}
    return {}

# Back-compat shim for older call sites that used expect_key=
def _salvage_block_expect_key(text: str, expect_key: Optional[str]) -> Dict[str, Any]:
    return _salvage_block(text, expect_key)

def _is_single_prompt_only(model_name: str) -> bool:
    """
    True for Replicate models that take ONLY a single 'prompt' (no messages/system).
    We scope this STRICTLY to openai/gpt-oss-* per request.
    """
    return (model_name or "").lower().startswith("openai/gpt-oss-")


# -------------------------- client --------------------------

class ReplicateLLM:
    """
    Replicate wrapper with model-specific prompt shaping and robust JSON salvage.
    Implements __call__(messages, json_schema=...) to match other clients.
    Includes jittered exponential backoff for transient HTTP faults.
    """

    def __init__(self, model: str, *, api_token: Optional[str] = None, default_extra: Optional[Dict[str, Any]] = None):
        load_dotenv()
        self.model = model
        token = api_token or os.getenv("REPLICATE_API_TOKEN")
        if not token:
            raise RuntimeError("Missing REPLICATE_API_TOKEN in environment (or pass api_token=...).")
        self._client = replicate.Client(api_token=token)
        self._debug = os.getenv("REPLICATE_DEBUG", "") == "1"
        self._default_extra = default_extra or {}

    # Allow call-style usage like other LLM clients
    def __call__(self, messages: List[Dict[str, str]], *, json_schema: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any] | str:
        return self.generate(messages, json_schema=json_schema, **kwargs)

    # --------- builders ---------

    def _inputs_common(
        self,
        *,
        temperature: Optional[float],
        top_p: Optional[float],
        top_k: Optional[int],
        max_tokens: Optional[int],
        seed: Optional[int],
        extra: Dict[str, Any],
    ) -> Dict[str, Any]:
        # merge defaults + per-call extras
        merged_extra = {**(self._default_extra or {}), **(extra or {})}

        inp: Dict[str, Any] = {}
        if temperature is not None: inp["temperature"] = temperature
        if top_p is not None: inp["top_p"] = top_p
        if top_k is not None: inp["top_k"] = top_k
        if max_tokens is not None:
            inp["max_tokens"] = max_tokens
            inp["max_output_tokens"] = max_tokens
        if seed is not None: inp["seed"] = seed

        # Replicate quirk: some runners expect scalar strings for stop / stop_sequences
        if "stop_sequences" in merged_extra and isinstance(merged_extra["stop_sequences"], list):
            merged_extra = {**merged_extra, "stop_sequences": merged_extra["stop_sequences"][0] if merged_extra["stop_sequences"] else ""}
        if "stop" in merged_extra and isinstance(merged_extra["stop"], list):
            merged_extra = {**merged_extra, "stop": merged_extra["stop"][0] if merged_extra["stop"] else ""}

        for k, v in (merged_extra or {}).items():
            if v is not None:
                inp[k] = v
        return inp

    def _build_for_gemini(self, messages, json_schema, knobs) -> Dict[str, Any]:
        schema_min = _minify_schema(json_schema)
        system_prompt = (
            "Return ONLY a single valid JSON object that matches this JSON Schema exactly. "
            "No prose, no markdown, no code fences.\n"
            f"SCHEMA: {schema_min}\n"
            "If you truly don't know, return an empty but valid object per schema."
        )
        fewshot = (
            "EXAMPLE:\n"
            'USER: Subject: Ping\n'
            'ASSISTANT: {"facts":[{"subject":"Ping","predicate":"instanceOf","object":"entity","confidence":1.0}]}\n\n'
        )
        prompt = fewshot + _collapse_messages(messages)
        knobs.setdefault("temperature", 0.2)
        knobs.setdefault("top_p", 0.9)
        return {"prompt": prompt, "system_prompt": system_prompt, **knobs}

    def _build_for_grok_messages(self, messages, json_schema, knobs) -> Dict[str, Any]:
        schema_min = _minify_schema(json_schema)
        sys_msg = {
            "role": "system",
            "content": (
                "You are a JSON function. Return ONLY one JSON object validating this schema. "
                "No prose/markdown/code fences. If unsure, return an empty—but valid—object.\n"
                f"SCHEMA: {schema_min}"
            ),
        }
        usr_msg = {"role": "user", "content": _collapse_messages(messages)}
        inputs = {"messages": [sys_msg, usr_msg]}
        for k in ("temperature", "top_p", "top_k", "max_tokens", "max_output_tokens", "seed"):
            if k in knobs:
                inputs[k] = knobs[k]
        return inputs

    def _build_for_qwen_prompt(self, messages, json_schema, knobs) -> Dict[str, Any]:
        schema_min = _minify_schema(json_schema)
        fewshot = (
            "You must output ONE JSON object that VALIDATES this JSON Schema.\n"
            "NO prose, NO markdown, NO code fences.\n"
            f"SCHEMA: {schema_min}\n\n"
            "EXAMPLE:\n"
            'USER: Subject: Ping\n'
            'ASSISTANT: {"facts":[{"subject":"Ping","predicate":"instanceOf","object":"entity","confidence":0.99}]}\n\n'
        )
        task = _collapse_messages(messages)
        contract = (
            "If you know the subject, produce 12–40 concise triples (no duplicates). "
            'Always include at least one triple with predicate "instanceOf". '
            'If uncertain overall, return {"facts":[]}.'
        )
        prompt = f"{fewshot}{task}\n\n{contract}"
        knobs.setdefault("temperature", 0.3)
        knobs.setdefault("top_p", 0.9)
        knobs.setdefault("max_tokens", knobs.get("max_output_tokens", 1536))
        return {"prompt": prompt, **knobs}

    def _build_inputs(self, messages, json_schema, knobs) -> Dict[str, Any]:
        """
        Build Replicate payload, with a STRICT special-case only for openai/gpt-oss-* models
        that accept a single 'prompt'. All other models behave as before.
        """
        is_gemini = self.model.startswith("google/gemini")
        is_grok = self.model.startswith("xai/grok-4") or "grok-4" in self.model
        is_qwen = self.model.startswith("qwen/")
        single_prompt_only = _is_single_prompt_only(self.model)

        if json_schema:
            schema_min = _minify_schema(json_schema)
            schema_instr = (
                "You must return ONLY one valid JSON object that matches the JSON Schema below.\n"
                "No prose, no markdown, no code fences. If unsure, return an empty but valid object.\n"
                f"SCHEMA: {schema_min}\n\n"
            )

            if single_prompt_only:
                combined = _collapse_single_prompt(messages)
                prompt = schema_instr + combined
                return {"prompt": prompt, **knobs}

            if is_gemini:
                fewshot = (
                    "EXAMPLE:\n"
                    'USER: Subject: Ping\n'
                    'ASSISTANT: {"facts":[{"subject":"Ping","predicate":"instanceOf","object":"entity","confidence":1.0}]}\n\n'
                )
                prompt = fewshot + _collapse_messages(messages)
                system_prompt = (
                    "Return ONLY a single valid JSON object that matches this JSON Schema exactly. "
                    "No prose, no markdown, no code fences.\n"
                    f"SCHEMA: {schema_min}\n"
                    "If you truly don't know, return an empty but valid object per schema."
                )
                knobs.setdefault("temperature", 0.2)
                knobs.setdefault("top_p", 0.9)
                return {"prompt": prompt, "system_prompt": system_prompt, **knobs}

            if is_grok:
                return self._build_for_grok_messages(messages, json_schema, knobs)

            if is_qwen:
                return self._build_for_qwen_prompt(messages, json_schema, knobs)

            # generic (unchanged)
            system_prompt = (
                "Return ONLY a single valid JSON object matching this schema. "
                "No prose, no markdown, no code fences.\n"
                f"SCHEMA: {schema_min}"
            )
            prompt = _collapse_messages(messages)
            return {"prompt": prompt, "system_prompt": system_prompt, **knobs}

        # -------- no json_schema (plain text) --------
        if single_prompt_only:
            return {"prompt": _collapse_single_prompt(messages), **knobs}

        if is_gemini:
            return {"prompt": _collapse_messages(messages), "system_prompt": "", **knobs}

        if is_grok:
            sys_msg = {"role": "system", "content": "You are a helpful assistant."}
            usr_msg = {"role": "user", "content": _collapse_messages(messages)}
            inputs = {"messages": [sys_msg, usr_msg]}
            for k in ("temperature", "top_p", "top_k", "max_tokens", "max_output_tokens", "seed"):
                if k in knobs:
                    inputs[k] = knobs[k]
            return inputs

        if is_qwen:
            return {"prompt": _collapse_messages(messages), **knobs}

        # default
        return {"prompt": _collapse_messages(messages), **knobs}

    # --------- internal resilient wrappers ---------

    def _blocking_once(self, inputs: Dict[str, Any]) -> str:
        transient = (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.ReadError,
            httpx.RemoteProtocolError,
            httpcore.RemoteProtocolError,
            httpcore.WriteError,
            httpcore.ReadTimeout,
            httpcore.ConnectTimeout,
        )
        delay = 0.8
        max_tries = 6
        last_err: Optional[BaseException] = None
        for attempt in range(1, max_tries + 1):
            try:
                pred = self._client.predictions.create(model=self.model, input=inputs)
                pred.wait()
                out = pred.output
                return "".join(out) if isinstance(out, list) else (out or "")
            except transient as e:
                last_err = e
                if self._debug:
                    print(f"[replicate][retry {attempt}/{max_tries}] {type(e).__name__}: {e}", flush=True)
                if attempt == max_tries:
                    raise
                time.sleep(delay + random.random() * 0.3)
                delay = min(delay * 1.8, 10.0)
            except Exception:
                raise
        raise last_err or RuntimeError("replicate _blocking_once failed without exception")

    def _stream_once(self, inputs: Dict[str, Any]) -> str:
        transient = (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.ReadError,
            httpx.RemoteProtocolError,
            httpcore.RemoteProtocolError,
            httpcore.WriteError,
            httpcore.ReadTimeout,
            httpcore.ConnectTimeout,
        )
        delay = 0.8
        max_tries = 6
        last_err: Optional[BaseException] = None
        for attempt in range(1, max_tries + 1):
            try:
                chunks: List[str] = []
                for event in replicate.stream(self.model, input=inputs):
                    chunks.append(str(event))
                return "".join(chunks)
            except transient as e:
                last_err = e
                if self._debug:
                    print(f"[replicate][stream retry {attempt}/{max_tries}] {type(e).__name__}: {e}", flush=True)
                if attempt == max_tries:
                    raise
                time.sleep(delay + random.random() * 0.3)
                delay = min(delay * 1.8, 10.0)
            except Exception:
                raise
        raise last_err or RuntimeError("replicate _stream_once failed without exception")

    # --------- schema-based coercion ---------

    def _coerce_by_schema(self, obj: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        props = (schema.get("properties") or {})
        if "facts" in props:
            calibrated = "confidence" in (props["facts"]["items"]["properties"] or {})
            return _coerce_elicit(obj, calibrated=calibrated)
        if "phrases" in props:
            calibrated = "confidence" in (props["phrases"]["items"]["properties"] or {})
            return _coerce_ner(obj, calibrated=calibrated)
        return obj if isinstance(obj, dict) else {}

    # --------- public blocking API ---------

    def ping(self) -> Dict[str, Any]:
        inp = {"prompt": 'Return ONLY this exact JSON: {"message":"PONG"}', "max_tokens": 32, "temperature": 0}
        txt = self._blocking_once(inp)
        obj = _parse_json_best_effort(txt)
        return obj if obj else {"message": "PONG"}

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        json_schema: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        knobs = self._inputs_common(
            temperature=temperature, top_p=top_p, top_k=top_k,
            max_tokens=max_tokens, seed=seed, extra=extra or {},
        )
        inputs = self._build_inputs(messages, json_schema, knobs)

        if not json_schema:
            text = self._blocking_once(inputs)
            if self._debug:
                print("\n[replicate][raw output]\n" + text[:4000] + ("\n" if len(text) else ""), flush=True)
            return {"text": text, "_raw": text}

        props = (json_schema.get("properties") or {})
        expect = "facts" if "facts" in props else ("phrases" if "phrases" in props else None)

        is_grok = self.model.startswith("xai/grok-4") or "grok-4" in self.model

        if is_grok:
            text = self._stream_once(inputs)
            if self._debug:
                print("\n[replicate][raw stream (grok)]\n" + text[:4000] + ("\n" if len(text) else ""), flush=True)
            parsed = _salvage_block(text, key=expect)
            result = self._coerce_by_schema(parsed, json_schema)
            result["_raw"] = text
            return result

        text = self._blocking_once(inputs)
        if self._debug:
            print("\n[replicate][raw output]\n" + text[:4000] + ("\n" if len(text) else ""), flush=True)

        parsed = _salvage_block(text, key=expect) or _salvage_block_expect_key(text, expect_key=expect)

        if parsed:
            result = self._coerce_by_schema(parsed, json_schema)
            result["_raw"] = text
            return result

        result = self._coerce_by_schema({}, json_schema)
        result["_raw"] = text
        return result

    # --------- streaming API ---------

    def stream_text(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        knobs = self._inputs_common(
            temperature=temperature, top_p=top_p, top_k=top_k,
            max_tokens=max_tokens, seed=seed, extra=extra or {},
        )
        inputs = self._build_inputs(messages, json_schema=None, knobs=knobs)
        # resilient streaming
        transient = (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.ReadError,
            httpx.RemoteProtocolError,
            httpcore.RemoteProtocolError,
            httpcore.WriteError,
            httpcore.ReadTimeout,
            httpcore.ConnectTimeout,
        )
        delay = 0.8
        max_tries = 6
        attempt = 1
        while True:
            try:
                for event in replicate.stream(self.model, input=inputs):
                    yield str(event)
                break
            except transient as e:
                if self._debug:
                    print(f"[replicate][stream_text retry {attempt}/{max_tries}] {type(e).__name__}: {e}", flush=True)
                if attempt >= max_tries:
                    raise
                time.sleep(delay + random.random() * 0.3)
                delay = min(delay * 1.8, 10.0)
                attempt += 1

    def stream_json(
        self,
        messages: List[Dict[str, str]],
        *,
        json_schema: Dict[str, Any],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        buffer: List[str] = []
        knobs = self._inputs_common(
            temperature=temperature, top_p=top_p, top_k=top_k,
            max_tokens=max_tokens, seed=seed, extra=extra or {},
        )
        inputs = self._build_inputs(messages, json_schema=json_schema, knobs=knobs)
        # resilient stream collect
        text = ""
        transient = (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.ReadError,
            httpx.RemoteProtocolError,
            httpcore.RemoteProtocolError,
            httpcore.WriteError,
            httpcore.ReadTimeout,
            httpcore.ConnectTimeout,
        )
        delay = 0.8
        max_tries = 6
        for attempt in range(1, max_tries + 1):
            try:
                buffer.clear()
                for event in replicate.stream(self.model, input=inputs):
                    buffer.append(str(event))
                text = "".join(buffer)
                break
            except transient as e:
                if self._debug:
                    print(f"[replicate][stream_json retry {attempt}/{max_tries}] {type(e).__name__}: {e}", flush=True)
                if attempt == max_tries:
                    raise
                time.sleep(delay + random.random() * 0.3)
                delay = min(delay * 1.8, 10.0)

        if self._debug:
            print("\n[replicate][raw stream combined]\n" + text[:4000] + ("\n" if len(text) else ""), flush=True)

        props = (json_schema.get("properties") or {})
        expect = "facts" if "facts" in props else ("phrases" if "phrases" in props else None)
        parsed = _salvage_block(text, key=expect) or _salvage_block_expect_key(text, expect_key=expect)
        result = self._coerce_by_schema(parsed, json_schema)
        result["_raw"] = text
        yield result

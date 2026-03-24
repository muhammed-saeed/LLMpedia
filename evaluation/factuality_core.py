#!/usr/bin/env python3
"""
factuality_core.py — Core factuality evaluation for LLMPedia.

CHANGES FROM ORIGINAL:
  - Integrated json_repair for robust JSON extraction (handles apostrophes,
    smart quotes, truncated output, unescaped control chars)
  - Removed global timeout from as_completed (was causing 967/1000 shortfalls)
  - Per-future timeout raised to 300s
  - Default EvalConfig.timeout raised to 600s
  - Default max_retries raised to 5

PATCH v3:
  - REMOVED topic_context from EvalConfig and fetch_evidence().
    Evidence is retrieved based ONLY on the subject name.
    No topic disambiguation is injected into web searches.
  - EvalInput.topic is kept for metadata/labeling only, never used for search.
"""
from __future__ import annotations

import concurrent.futures
import csv
import json
import os
import random
import re
import signal
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ──────────────────────────────────────────────────────────────────────────────
# OPTIONAL IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
for _p in (THIS_DIR, PROJECT_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from eval_text_utils import clean_wikitext_for_eval
except Exception:
    def clean_wikitext_for_eval(text: str) -> str:
        text = re.sub(r'\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]', r'\1', text)
        text = re.sub(r'\(\d+\.\d+\)', '', text)
        return text.strip()

# ── JSON repair (robust extraction) ──
try:
    from json_repair import extract_json_robust as _extract_json
except ImportError:
    _extract_json = None  # will be defined below as fallback

HAS_SETTINGS = False
SETTINGS_IMPORT_ERROR = None
settings = None
make_llm_from_config = None
try:
    from settings import settings as _settings_mod
    from llm.factory import make_llm_from_config as _make
    settings = _settings_mod
    make_llm_from_config = _make
    HAS_SETTINGS = True
except Exception as e:
    HAS_SETTINGS = False
    SETTINGS_IMPORT_ERROR = repr(e)

# ──────────────────────────────────────────────────────────────────────────────
# CANCELLATION
# ──────────────────────────────────────────────────────────────────────────────
_cancel_event = threading.Event()

def cancel_requested() -> bool:
    return _cancel_event.is_set()

def request_cancel() -> None:
    _cancel_event.set()

def _install_sigint_handler():
    try:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class EvalConfig:
    fact_model_key: str = "gpt-4.1-nano"
    judge_temperature: float = 0.0
    timeout: float = 600.0

    llm_api_base: str = ""
    llm_api_key: str = ""
    max_retries: int = 5

    max_claims: int = 10
    max_article_chars: int = 0
    max_evidence_chars: int = 0

    evidence_sources: List[str] = field(default_factory=lambda: ["wikipedia", "web"])

    web_mode: str = "hybrid"
    max_web_snippets: int = 5
    max_fetch_pages: int = 1
    web_fetch_workers: int = 4
    web_cache_dir: str = ""
    web_cache_ttl_hours: float = 168.0
    search_backend: str = "auto"
    searxng_api_base: str = ""
    exclude_domains: str = ""
    # REMOVED: topic_context field — no longer used for evidence retrieval

    compute_similarity: bool = True
    compute_bertscore: bool = False
    compute_stylistic: bool = False
    semantic_provider: str = "openai"
    semantic_model: str = "all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"
    ngram_values: List[int] = field(default_factory=lambda: [1, 2, 3])

    evidence_cache_dir: str = ""
    run_audit_dir: str = ""

    concurrency: int = 10
    debug: bool = False


@dataclass
class EvalInput:
    subject: str
    candidate: str
    article_text: str
    hop: Any = None
    persona: str = ""
    topic: str = ""             # kept for metadata/labeling only, NOT used for search
    generator_model: str = ""
    clean_wiki_markup: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

# ──────────────────────────────────────────────────────────────────────────────
# UTILS
# ──────────────────────────────────────────────────────────────────────────────
_tl = threading.local()

def _ts() -> str:
    import datetime
    return datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _dbg(msg: str, enabled: bool):
    if enabled:
        print(f"[DBG {_ts()}] {msg}", flush=True)

def _safe_filename(subject: str) -> str:
    safe = re.sub(r"[^\w\- ]", "_", subject).strip().replace(" ", "_")
    return safe[:120]

def tokenize_simple(text: str) -> List[str]:
    return [t for t in re.sub(r"[^\w\s]", " ", text.lower()).split() if t]

def _load_jsonl(path: str) -> List[Dict]:
    out: List[Dict] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                if isinstance(o, dict):
                    out.append(o)
            except Exception:
                continue
    return out

def _append_jsonl(path: str, obj: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _write_json_atomic(path: str, obj: Any):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

# ──────────────────────────────────────────────────────────────────────────────
# JSON EXTRACTION (robust, inline fallback if json_repair not importable)
# ──────────────────────────────────────────────────────────────────────────────

def _fix_json_string_issues(s: str) -> str:
    for lq, rq in [("\u201c", '"'), ("\u201d", '"'),
                    ("\u2018", "'"), ("\u2019", "'"),
                    ("\u2032", "'"), ("\u2033", '"'), ("\u00b4", "'")]:
        s = s.replace(lq, rq)
    s = re.sub(r',\s*([\]\}])', r'\1', s)
    return s

def _repair_truncated_json(s: str) -> Optional[dict]:
    if not s or "{" not in s:
        return None
    for trim in range(0, min(500, len(s)), 1):
        candidate = s[:len(s) - trim] if trim > 0 else s
        candidate = candidate.rstrip()
        if candidate.endswith(","):
            candidate = candidate[:-1].rstrip()
        opens = []
        in_str = False
        esc = False
        for c in candidate:
            if esc:
                esc = False
                continue
            if c == "\\" and in_str:
                esc = True
                continue
            if c == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if c in "{[":
                opens.append(c)
            elif c == "}" and opens and opens[-1] == "{":
                opens.pop()
            elif c == "]" and opens and opens[-1] == "[":
                opens.pop()
        if in_str:
            candidate += '"'
        closers = {"[": "]", "{": "}"}
        closing = "".join(closers.get(o, "") for o in reversed(opens))
        attempt = candidate + closing
        try:
            obj = json.loads(attempt, strict=False)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None

def _regex_extract_claims(raw: str) -> Optional[dict]:
    candidates = re.findall(r'"([^"]{20,}[.!?])"', raw)
    if not candidates:
        candidates = re.findall(r'"([^"]{15,})"', raw)
    if len(candidates) >= 3:
        claims = [c for c in candidates
                  if not c.startswith(("{", "[", "http"))
                  and len(c.split()) >= 4]
        if len(claims) >= 3:
            return {"claims": claims[:10]}
    return None

def _regex_extract_verdicts(raw: str) -> Optional[dict]:
    blocks = re.findall(
        r'\{[^{}]*?"idx"\s*:\s*(\d+)[^{}]*?"verdict"\s*:\s*"(\w+)"[^{}]*?\}',
        raw, re.DOTALL
    )
    if not blocks:
        blocks = re.findall(
            r'\{[^{}]*?"verdict"\s*:\s*"(\w+)"[^{}]*?"idx"\s*:\s*(\d+)[^{}]*?\}',
            raw, re.DOTALL
        )
        if blocks:
            blocks = [(idx, verd) for verd, idx in blocks]
    if blocks:
        verdicts = []
        for idx_str, verd in blocks:
            verdicts.append({
                "idx": int(idx_str), "verdict": verd.lower().strip(),
                "confidence": 0.5, "explanation": "recovered from malformed JSON"
            })
        if verdicts:
            return {"verdicts": verdicts}
    return None

def _escape_strings_in_json(s: str) -> str:
    result = []
    i = 0
    in_string = False
    while i < len(s):
        c = s[i]
        if not in_string:
            result.append(c)
            if c == '"':
                in_string = True
            i += 1
        else:
            if c == '\\':
                if i + 1 < len(s):
                    next_c = s[i + 1]
                    if next_c in ('"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'):
                        result.append(c); result.append(next_c); i += 2; continue
                    else:
                        result.append('\\\\'); i += 1; continue
                else:
                    result.append(c); i += 1
            elif c == '"':
                result.append(c); in_string = False; i += 1
            elif c == '\n':
                result.append('\\n'); i += 1
            elif c == '\r':
                result.append('\\r'); i += 1
            elif c == '\t':
                result.append('\\t'); i += 1
            elif ord(c) < 0x20:
                result.append(f'\\u{ord(c):04x}'); i += 1
            else:
                result.append(c); i += 1
    return ''.join(result)

if _extract_json is None:
    def _extract_json(txt: str) -> Optional[dict]:
        if not txt:
            return None
        s = txt.strip()
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
        s = s.strip()

        try:
            o = json.loads(s, strict=False)
            if isinstance(o, dict): return o
        except json.JSONDecodeError: pass

        s_fixed = _fix_json_string_issues(s)
        try:
            o = json.loads(s_fixed, strict=False)
            if isinstance(o, dict): return o
        except json.JSONDecodeError: pass

        first_brace = s.find("{")
        last_brace = s.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            chunk = s[first_brace:last_brace + 1]
            try:
                o = json.loads(chunk, strict=False)
                if isinstance(o, dict): return o
            except json.JSONDecodeError: pass
            chunk_fixed = _fix_json_string_issues(chunk)
            try:
                o = json.loads(chunk_fixed, strict=False)
                if isinstance(o, dict): return o
            except json.JSONDecodeError: pass
            try:
                o = json.loads(_escape_strings_in_json(chunk_fixed), strict=False)
                if isinstance(o, dict): return o
            except (json.JSONDecodeError, Exception): pass

        if first_brace >= 0:
            raw = s[first_brace:]
            result = _repair_truncated_json(raw)
            if result is not None: return result
            result = _repair_truncated_json(_fix_json_string_issues(raw))
            if result is not None: return result
            try:
                result = _repair_truncated_json(_escape_strings_in_json(_fix_json_string_issues(raw)))
                if result is not None: return result
            except Exception: pass

        if first_brace >= 0:
            raw = s[first_brace:]
            if '"verdict"' in raw.lower():
                result = _regex_extract_verdicts(raw)
                if result: return result
            if '"claims"' in raw.lower() or '"verdict"' not in raw.lower():
                result = _regex_extract_claims(raw)
                if result: return result

        return None

def _unwrap(resp) -> str:
    if resp is None: return ""
    if hasattr(resp, "model_dump"):
        try: resp = resp.model_dump()
        except Exception: pass
    if isinstance(resp, str): return resp
    try:
        choices = getattr(resp, "choices", None)
        if choices:
            c0 = choices[0]
            msg = getattr(c0, "message", None)
            if msg is not None:
                content = getattr(msg, "content", None)
                if isinstance(content, str): return content
            txt = getattr(c0, "text", None)
            if isinstance(txt, str): return txt
    except Exception: pass
    if isinstance(resp, dict):
        for k in ("text", "output_text", "content", "message"):
            v = resp.get(k)
            if isinstance(v, str): return v
        ch = resp.get("choices")
        if isinstance(ch, list) and ch:
            msg = (ch[0] or {}).get("message") or {}
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
            if isinstance((ch[0] or {}).get("text"), str):
                return (ch[0] or {})["text"]
    return ""

# ──────────────────────────────────────────────────────────────────────────────
# GLOBAL DISK EVIDENCE CACHE
# ──────────────────────────────────────────────────────────────────────────────
_ev_mem: Dict[Tuple[str, str], Dict[str, Any]] = {}
_ev_mem_lock = threading.Lock()
_ev_fetch_locks: Dict[Tuple[str, str], threading.Lock] = {}
_ev_fetch_locks_lock = threading.Lock()

def _get_fetch_lock(key: Tuple[str, str]) -> threading.Lock:
    with _ev_fetch_locks_lock:
        if key not in _ev_fetch_locks:
            _ev_fetch_locks[key] = threading.Lock()
        return _ev_fetch_locks[key]

def _disk_cache_path(cache_dir: str, source: str, subject: str) -> str:
    return os.path.join(cache_dir, source, _safe_filename(subject) + ".json")

def _disk_cache_read(cache_dir: str, source: str, subject: str) -> Optional[Dict[str, Any]]:
    if not cache_dir: return None
    path = _disk_cache_path(cache_dir, source, subject)
    if not os.path.exists(path): return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "text" in obj: return obj
    except Exception: pass
    return None

def _disk_cache_write(cache_dir: str, source: str, subject: str, obj: Dict[str, Any]):
    if not cache_dir: return
    path = _disk_cache_path(cache_dir, source, subject)
    try: _write_json_atomic(path, obj)
    except Exception: pass

# ──────────────────────────────────────────────────────────────────────────────
# WIKIPEDIA
# ──────────────────────────────────────────────────────────────────────────────
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_UA = os.getenv("WIKI_USER_AGENT", "LLMPediaEval/1.0 (research; contact@example.com)")

_wiki_sess: Optional[requests.Session] = None
_wiki_lock = threading.Lock()
_wiki_cache: Dict[str, Dict[str, Any]] = {}

def _get_wiki_session() -> requests.Session:
    global _wiki_sess
    if _wiki_sess is None:
        s = requests.Session()
        s.headers["User-Agent"] = WIKI_UA
        _wiki_sess = s
    return _wiki_sess

def wiki_fetch_page(title: str) -> Dict[str, Any]:
    with _wiki_lock:
        if title in _wiki_cache:
            return _wiki_cache[title]
    empty = {"title": title, "text": "", "found": False, "url": None, "redirected": False}
    params = {
        "action": "query", "prop": "extracts", "explaintext": 1,
        "redirects": 1, "format": "json", "utf8": 1, "titles": title,
    }
    try:
        r = _get_wiki_session().get(WIKI_API, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        q = data.get("query") or {}
        pages = q.get("pages") or {}
        redirs = q.get("redirects") or []
        for pid, p in pages.items():
            if str(pid) == "-1" or p.get("missing") is not None:
                break
            t2 = p.get("title") or title
            text = (p.get("extract") or "").strip()
            res = {
                "title": t2, "text": text, "found": bool(text),
                "redirected": bool(redirs),
                "url": "https://en.wikipedia.org/wiki/" + t2.replace(" ", "_"),
            }
            with _wiki_lock:
                _wiki_cache[title] = res
            return res
    except Exception: pass
    with _wiki_lock:
        _wiki_cache[title] = empty
    return empty

# ──────────────────────────────────────────────────────────────────────────────
# EVIDENCE RETRIEVER
#
# CHANGE (v3): Removed topic_context from fetch_evidence() and
# _get_evidence_retriever(). Evidence is now subject-only.
# ──────────────────────────────────────────────────────────────────────────────
_ev_retriever = None
_ev_retriever_lock = threading.Lock()

def _get_evidence_retriever(cfg: EvalConfig):
    global _ev_retriever
    if _ev_retriever is not None: return _ev_retriever
    with _ev_retriever_lock:
        if _ev_retriever is not None: return _ev_retriever
        from evidence import EvidenceRetriever, EvidenceConfig
        excl: Set[str] = {"wikipedia.org", "en.wikipedia.org", "wikidata.org", "www.wikidata.org"}
        if cfg.exclude_domains:
            for d in cfg.exclude_domains.split(","):
                d = d.strip().lower()
                if d: excl.add(d)
        audit_dir = os.path.join(cfg.run_audit_dir, "evidence") if cfg.run_audit_dir else ""
        ev_cfg = EvidenceConfig(
            web_mode=cfg.web_mode, web_max_snippets=cfg.max_web_snippets,
            web_max_fetch_pages=cfg.max_fetch_pages, web_fetch_workers=cfg.web_fetch_workers,
            web_timeout=15.0, web_snippet_max_chars=2000,
            web_exclude_domains=excl, web_extra_exclude=cfg.exclude_domains,
            search_backend=cfg.search_backend, searxng_api_base=cfg.searxng_api_base,
            cache_dir=cfg.web_cache_dir, cache_ttl_hours=cfg.web_cache_ttl_hours,
            audit_dir=audit_dir, debug=cfg.debug,
        )
        _ev_retriever = EvidenceRetriever(ev_cfg)
        return _ev_retriever

def fetch_evidence(subject: str, source: str, cfg: EvalConfig, *,
                   claims: Optional[List[str]] = None) -> Dict[str, Any]:
    """Fetch evidence for a subject. No topic_context — subject only."""
    key = (subject, source)
    with _ev_mem_lock:
        if key in _ev_mem: return _ev_mem[key]
    lock = _get_fetch_lock(key)
    with lock:
        with _ev_mem_lock:
            if key in _ev_mem: return _ev_mem[key]
        disk = _disk_cache_read(cfg.evidence_cache_dir, source, subject)
        if disk is not None:
            with _ev_mem_lock: _ev_mem[key] = disk
            return disk
        if source == "wikipedia":
            w = wiki_fetch_page(subject)
            result = {
                "source": "wikipedia", "found": w.get("found", False),
                "text": w.get("text", ""), "url": w.get("url"),
                "snippets": [{"title": w.get("title", subject),
                              "snippet_text": w.get("text", ""),
                              "url": w.get("url", ""), "source_type": "wikipedia"}
                             ] if w.get("text") else [],
                "meta": {"redirected": w.get("redirected", False)},
                "_fetched_ts": _ts(),
            }
        elif source == "web":
            retr = _get_evidence_retriever(cfg)
            # CHANGE (v3): No topic_context passed — subject only
            ev = retr.get("web", subject, claims=claims)
            result = {
                "source": "web", "found": ev.found, "text": ev.text,
                "url": ev.url, "snippets": ev.snippets, "meta": ev.meta,
                "_fetched_ts": _ts(),
            }
        else:
            result = {"source": source, "found": False, "text": "",
                      "snippets": [], "url": None, "meta": {}, "_fetched_ts": _ts()}
        _disk_cache_write(cfg.evidence_cache_dir, source, subject, result)
        with _ev_mem_lock: _ev_mem[key] = result
        return result

# ──────────────────────────────────────────────────────────────────────────────
# LLM CALLING
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class _ResolvedModel:
    base_url: str; api_key: str; model_name: str
    temperature: float; top_p: float; max_tokens: int

def _resolve_model_from_settings(model_key: str, cfg: EvalConfig) -> Optional[_ResolvedModel]:
    if not HAS_SETTINGS or settings is None: return None
    models = getattr(settings, "MODELS", None)
    if not models or model_key not in models: return None
    mcfg = models[model_key]
    base_url = getattr(mcfg, "base_url", None) or ""
    if not base_url: return None
    api_key_env = getattr(mcfg, "api_key_env", None) or ""
    api_key = os.getenv(api_key_env, "") if api_key_env else ""
    if not api_key: api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print(f"[ERROR] API key not found for '{model_key}'. Set '{api_key_env}'.", flush=True)
        return None
    model_name = getattr(mcfg, "model", model_key)
    return _ResolvedModel(
        base_url=base_url.rstrip("/") if not base_url.endswith("/v1") else base_url,
        api_key=api_key, model_name=model_name,
        temperature=getattr(mcfg, "temperature", None) if getattr(mcfg, "temperature", None) is not None else cfg.judge_temperature,
        top_p=getattr(mcfg, "top_p", 1.0) or 1.0,
        max_tokens=max(getattr(mcfg, "max_tokens", 4096) or 4096, 16384),
    )

_oai_clients: Dict[str, Any] = {}
_oai_lock = threading.Lock()

def _get_oai_client_for(base_url: str, api_key: str) -> Any:
    cache_key = f"{base_url}||{api_key[:8] if api_key else 'none'}"
    with _oai_lock:
        if cache_key in _oai_clients: return _oai_clients[cache_key]
    from openai import OpenAI
    kwargs: Dict[str, Any] = {}
    if base_url: kwargs["base_url"] = base_url
    if api_key: kwargs["api_key"] = api_key
    client = OpenAI(**kwargs)
    with _oai_lock: _oai_clients[cache_key] = client
    return client


# ── Factory-based LLM calling (supports ALL providers) ────────────────────────
_factory_clients: Dict[str, Any] = {}
_factory_clients_lock = threading.Lock()

def _get_factory_client(model_key: str):
    """Get or create a client via make_llm_from_config for any provider."""
    with _factory_clients_lock:
        if model_key in _factory_clients:
            return _factory_clients[model_key]
    if not HAS_SETTINGS or settings is None or make_llm_from_config is None:
        return None
    models = getattr(settings, "MODELS", None)
    if not models or model_key not in models:
        return None
    try:
        client = make_llm_from_config(models[model_key])
        with _factory_clients_lock:
            _factory_clients[model_key] = client
        return client
    except Exception as e:
        print(f"[factory] Failed to create client for \'{model_key}\': {e}", flush=True)
        return None

def _llm_call_via_factory(messages, model_key: str):
    """Call any model via the factory. Returns (raw_text, finish_reason) or None."""
    client = _get_factory_client(model_key)
    if client is None:
        return None
    try:
        result = client(messages)
        if isinstance(result, dict):
            raw = result.get("text") or result.get("_raw") or ""
            if not raw:
                useful_keys = [k for k in result if not k.startswith("_")]
                if useful_keys:
                    import json as _json
                    raw = _json.dumps(result, ensure_ascii=False)
        elif isinstance(result, str):
            raw = result
        else:
            raw = str(result)
        return (raw, "stop")
    except Exception as e:
        print(f"[factory] Call failed for \'{model_key}\': {e}", flush=True)
        return None

def _llm_call_once(messages, cfg, resolved):
    # Try factory-based call first (supports Replicate, DeepSeek, ScadsAI, etc.)
    factory_result = _llm_call_via_factory(messages, cfg.fact_model_key)
    if factory_result is not None:
        return factory_result

    # Fall back to direct OpenAI client (original logic)
    if resolved is not None:
        c = _get_oai_client_for(resolved.base_url, resolved.api_key)
        resp = c.chat.completions.create(
            model=resolved.model_name, messages=messages,
            max_tokens=8192, temperature=resolved.temperature, top_p=resolved.top_p)
    elif cfg.llm_api_base:
        api_key = cfg.llm_api_key or os.getenv("OPENAI_API_KEY", "")
        c = _get_oai_client_for(cfg.llm_api_base, api_key)
        resp = c.chat.completions.create(
            model=cfg.fact_model_key, messages=messages,
            max_tokens=8192, temperature=cfg.judge_temperature)
    elif _looks_like_openai_model(cfg.fact_model_key):
        c = _get_oai_client_for("", os.getenv("OPENAI_API_KEY", ""))
        resp = c.chat.completions.create(
            model=cfg.fact_model_key, messages=messages,
            max_tokens=8192, temperature=cfg.judge_temperature)
    else:
        raise RuntimeError(f"Cannot call model '{cfg.fact_model_key}'")
    raw = resp.choices[0].message.content or ""
    finish_reason = getattr(resp.choices[0], 'finish_reason', None)
    return raw, finish_reason

_logged_backend = False
_logged_lock = threading.Lock()

def call_llm_json(messages, cfg, call_label):
    resolved = _resolve_model_from_settings(cfg.fact_model_key, cfg)
    global _logged_backend
    with _logged_lock:
        if not _logged_backend:
            if resolved:
                print(f"[llm] Direct client: {resolved.base_url}  model={resolved.model_name}", flush=True)
            elif cfg.llm_api_base:
                print(f"[llm] Direct client: {cfg.llm_api_base}  model={cfg.fact_model_key}", flush=True)
            else:
                print(f"[llm] Direct OpenAI client: model={cfg.fact_model_key}", flush=True)
            _logged_backend = True

    max_retries = getattr(cfg, 'max_retries', 5)
    last_raw = ""; last_finish = None; last_error = None

    for attempt in range(max_retries + 1):
        try:
            raw, finish_reason = _llm_call_once(messages, cfg, resolved)
            last_raw = raw; last_finish = finish_reason
            obj = _extract_json(raw)
            if obj is not None:
                if attempt > 0:
                    items = obj.get("claims") or obj.get("verdicts") or []
                    print(f"[INFO] {call_label}: succeeded on retry {attempt}, got {len(items)} items", flush=True)
                if finish_reason == "length":
                    items = obj.get("claims") or obj.get("verdicts") or []
                    print(f"[INFO] Repaired truncated JSON for {call_label}: {len(items)} items", flush=True)
                return obj
            if attempt < max_retries:
                parse_err = ""
                try: json.loads(raw.strip())
                except json.JSONDecodeError as e: parse_err = str(e)
                print(f"[WARN] JSON parse failed for {call_label} (attempt {attempt+1}/{max_retries+1}), "
                      f"retrying. len={len(raw)} finish={finish_reason} err={parse_err}", flush=True)
                continue
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                print(f"[WARN] LLM call error for {call_label} (attempt {attempt+1}), retrying: {e}", flush=True)
                time.sleep(0.5 * (attempt + 1))
                continue
            break

    trunc_tag = " [TRUNCATED]" if last_finish == "length" else ""
    if last_error:
        print(f"[WARN] LLM call failed for {call_label} after {max_retries+1} attempts: {last_error}", flush=True)
    else:
        parse_err = ""
        try: json.loads(last_raw.strip())
        except json.JSONDecodeError as e: parse_err = f" json_err='{e}'"
        print(f"[WARN] JSON parse failed for {call_label} after {max_retries+1} attempts.{trunc_tag} "
              f"len={len(last_raw)} finish={last_finish}{parse_err} "
              f"preview: {last_raw[:300]!r}", flush=True)
    return None

def _looks_like_openai_model(model):
    return model.startswith(("gpt-", "o1", "o3", "o4", "text-embedding-", "text-"))

# ──────────────────────────────────────────────────────────────────────────────
# PROMPTS + PARSING
# ──────────────────────────────────────────────────────────────────────────────
def prompt_extract_claims(subject, article_text, max_claims):
    sys_msg = (f"Extract exactly {max_claims} distinct, atomic, verifiable factual claims "
               f'from the article about "{subject}". Each claim must be one sentence.\n'
               'Output ONLY valid JSON: {"claims": ["..."]}')
    usr_msg = f"Subject: {subject}\n\nArticle:\n{article_text}\n\nReturn JSON."
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": usr_msg}]

def prompt_batch_verify(subject, claims, evidence_source, evidence_snippets, max_evidence_chars):
    sys_msg = (f'Verify {len(claims)} claims about "{subject}" against the evidence.\n'
               'Verdict per claim: supported | refuted | insufficient.\n'
               'Output ONLY JSON: {"verdicts":[{"idx":1,"verdict":"...","confidence":0-1,"explanation":"..."}]}')
    claims_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
    ev_parts = []
    for i, sn in enumerate(evidence_snippets or [], 1):
        ev_parts.append(f"[{i}. {sn.get('title','')}] ({sn.get('url','')})\n{sn.get('snippet_text','')}")
    ev_text = "\n\n".join(ev_parts) if ev_parts else "(No evidence available)"
    if max_evidence_chars > 0: ev_text = ev_text[:max_evidence_chars]
    usr_msg = f"Claims:\n{claims_block}\n\nEvidence source: {evidence_source}\n{ev_text}\n\nReturn JSON."
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": usr_msg}]

def parse_extraction(raw):
    if not raw: return []
    claims = raw.get("claims")
    if not isinstance(claims, list): return []
    return [str(c).strip() for c in claims if str(c).strip()]

def parse_batch_verdicts(raw, n_claims):
    fallback = [{"verdict": "insufficient", "confidence": 0.0, "explanation": "no json"}] * n_claims
    if not raw: return fallback
    vlist = raw.get("verdicts")
    if not isinstance(vlist, list): return fallback
    idx_map = {}
    for v in vlist:
        if not isinstance(v, dict): continue
        try: idx = int(v.get("idx")) - 1
        except Exception: continue
        verdict = str(v.get("verdict", "")).lower().strip()
        if verdict in ("true", "supported", "support"): verdict = "supported"
        elif verdict in ("false", "refuted", "refute", "contradicted"): verdict = "refuted"
        else: verdict = "insufficient"
        try: conf = float(v.get("confidence"))
        except Exception: conf = None
        idx_map[idx] = {"verdict": verdict, "confidence": conf,
                        "explanation": str(v.get("explanation", "")).strip()}
    return [idx_map.get(i, {"verdict": "insufficient", "confidence": 0.0, "explanation": "missing"})
            for i in range(n_claims)]

# ──────────────────────────────────────────────────────────────────────────────
# SIMILARITY
# ──────────────────────────────────────────────────────────────────────────────
def _ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)] if len(tokens) >= n else []

def compute_similarity(text1, text2, cfg):
    if not text1.strip() or not text2.strip(): return {}
    m = {}
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        vec = TfidfVectorizer(ngram_range=(1,2), stop_words="english", max_features=10000)
        mat = vec.fit_transform([text1, text2])
        m["tfidf_cosine"] = float(cosine_similarity(mat[0:1], mat[1:2])[0][0])
    except Exception: pass
    t1, t2 = set(tokenize_simple(text1)), set(tokenize_simple(text2))
    if t1 and t2: m["jaccard"] = len(t1 & t2) / len(t1 | t2)
    for n in cfg.ngram_values or [1,2,3]:
        ng1 = set(_ngrams(tokenize_simple(text1), n))
        ng2 = set(_ngrams(tokenize_simple(text2), n))
        if ng1 and ng2:
            m[f"ngram_{n}_jaccard"] = len(ng1 & ng2) / len(ng1 | ng2)
            m[f"ngram_{n}_overlap"] = len(ng1 & ng2) / min(len(ng1), len(ng2))
    try:
        if cfg.semantic_provider == "openai":
            from openai import OpenAI
            if not hasattr(_tl, "oai_emb"): _tl.oai_emb = OpenAI()
            resp = _tl.oai_emb.embeddings.create(model=cfg.openai_embedding_model, input=[text1[:32000], text2[:32000]])
            import numpy as np
            e1, e2 = resp.data[0].embedding, resp.data[1].embedding
            m["semantic_cosine"] = float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9))
        else:
            from sentence_transformers import SentenceTransformer
            if not hasattr(_tl, "st_model"): _tl.st_model = SentenceTransformer(cfg.semantic_model)
            import numpy as np
            embs = _tl.st_model.encode([text1[:10000], text2[:10000]])
            m["semantic_cosine"] = float(np.dot(embs[0], embs[1]) / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]) + 1e-9))
    except Exception: pass
    if cfg.compute_bertscore:
        try:
            from bert_score import score as bert_score_fn
            cs = [s.strip() for s in re.split(r'[.!?]+', text1) if s.strip()][:50]
            rs = [s.strip() for s in re.split(r'[.!?]+', text2) if s.strip()][:50]
            if cs and rs:
                ml = min(len(cs), len(rs))
                P, R, F = bert_score_fn(cs[:ml], rs[:ml], lang="en", verbose=False, rescale_with_baseline=True)
                m["bertscore_precision"] = float(P.mean())
                m["bertscore_recall"] = float(R.mean())
                m["bertscore_f1"] = float(F.mean())
        except ImportError: print("[WARN] bert_score not installed", flush=True)
        except Exception: pass
    if cfg.compute_stylistic:
        try: m.update(_compute_stylistic(text1, text2))
        except Exception: pass
    weights = {"tfidf_cosine": 0.15, "jaccard": 0.10, "ngram_1_overlap": 0.10,
               "ngram_2_overlap": 0.10, "ngram_3_overlap": 0.10, "semantic_cosine": 0.45}
    wsum = wtot = 0.0
    for k, w in weights.items():
        if k in m and m[k] is not None: wsum += w * float(m[k]); wtot += w
    m["combined_similarity"] = (wsum / wtot) if wtot > 0 else 0.0
    m["candidate_words"] = len(tokenize_simple(text1))
    m["reference_words"] = len(tokenize_simple(text2))
    return m

def _compute_stylistic(text1, text2):
    import math
    def _sl(t): return [len(s.split()) for s in re.split(r'[.!?]+', t) if s.strip()]
    def _ttr(t):
        tok = tokenize_simple(t)
        return len(set(tok)) / len(tok) if tok else 0.0
    def _fwf(t):
        fw = {"the","a","an","is","was","were","are","be","been","being","have","has","had",
              "do","does","did","will","would","could","should","may","might","shall","can",
              "of","in","to","for","with","on","at","from","by","as","into","through","during",
              "before","after","and","but","or","nor","not","so","yet","that","which","who",
              "whom","this","these","those","it","its"}
        tok = tokenize_simple(t); total = len(tok) or 1
        freq = {}
        for w in tok:
            if w in fw: freq[w] = freq.get(w, 0) + 1
        return {k: v/total for k,v in freq.items()}
    def _pd(t):
        words = len(t.split()) or 1
        return sum(1 for c in t if c in '.,;:!?-()[]{}"\'/') / words
    m = {}
    sl1, sl2 = _sl(text1), _sl(text2)
    if sl1 and sl2:
        m1, m2 = sum(sl1)/len(sl1), sum(sl2)/len(sl2)
        s1 = math.sqrt(sum((x-m1)**2 for x in sl1)/len(sl1)) if len(sl1)>1 else 0
        s2 = math.sqrt(sum((x-m2)**2 for x in sl2)/len(sl2)) if len(sl2)>1 else 0
        m["style_sent_len_mean_diff"] = abs(m1-m2)
        m["style_sent_len_std_diff"] = abs(s1-s2)
    t1, t2 = _ttr(text1), _ttr(text2)
    m["style_ttr_candidate"] = t1; m["style_ttr_reference"] = t2; m["style_ttr_diff"] = abs(t1-t2)
    fw1, fw2 = _fwf(text1), _fwf(text2)
    afw = set(fw1) | set(fw2)
    if afw:
        dot = sum(fw1.get(w,0)*fw2.get(w,0) for w in afw)
        n1 = math.sqrt(sum(v**2 for v in fw1.values())) or 1e-9
        n2 = math.sqrt(sum(v**2 for v in fw2.values())) or 1e-9
        m["style_funcword_cosine"] = dot/(n1*n2)
    p1, p2 = _pd(text1), _pd(text2)
    m["style_punct_density_candidate"] = p1; m["style_punct_density_reference"] = p2
    m["style_punct_density_diff"] = abs(p1-p2)
    return m

# ──────────────────────────────────────────────────────────────────────────────
# AUDIT
# ──────────────────────────────────────────────────────────────────────────────
_audit_lock = threading.Lock()

def write_run_manifest(run_audit_dir, model, topic, persona, n_articles,
                       evidence_sources, fact_model_key, extra=None):
    if not run_audit_dir: return
    payload = {"model": model, "topic": topic, "persona": persona,
               "n_articles": n_articles, "evidence_sources": evidence_sources,
               "fact_model_key": fact_model_key, "started_ts": _ts()}
    if extra: payload.update(extra)
    with _audit_lock:
        _write_json_atomic(os.path.join(run_audit_dir, "manifest.json"), payload)

def _write_claims_audit(subject, claims, run_audit_dir, extra=None):
    if not run_audit_dir: return
    payload = {"subject": subject, "ts": _ts(), "n_claims": len(claims), "claims": claims}
    if extra: payload.update(extra)
    path = os.path.join(run_audit_dir, "claims", _safe_filename(subject) + ".json")
    with _audit_lock: _write_json_atomic(path, payload)

def _write_result_audit(rec, run_audit_dir):
    if not run_audit_dir: return
    path = os.path.join(run_audit_dir, "results", _safe_filename(rec.get("subject","unknown")) + ".json")
    with _audit_lock: _write_json_atomic(path, rec)

# ──────────────────────────────────────────────────────────────────────────────
# CORE EVAL
#
# CHANGE (v3): evaluate_subject() no longer passes topic_context to
# fetch_evidence(). Evidence retrieval is subject-only.
# ──────────────────────────────────────────────────────────────────────────────
def _verdict_stats(verdicts):
    n = len(verdicts)
    ns = sum(1 for v in verdicts if v.get("verdict") == "supported")
    nr = sum(1 for v in verdicts if v.get("verdict") == "refuted")
    ni = sum(1 for v in verdicts if v.get("verdict") == "insufficient")
    dec = ns + nr
    return {
        "n_claims": n, "n_supported": ns, "n_refuted": nr, "n_insufficient": ni,
        "true_rate": (ns/n) if n > 0 else 0.0,
        "false_rate": (nr/n) if n > 0 else 0.0,
        "unverifiable_rate": (ni/n) if n > 0 else 0.0,
        "accuracy": (ns/dec) if dec > 0 else None,
    }

def evaluate_subject(inp, cfg):
    subject = inp.subject
    if cancel_requested():
        return {"subject": subject, "candidate": inp.candidate, "_cancelled": True}
    text = inp.article_text or ""
    text_clean = clean_wikitext_for_eval(text) if inp.clean_wiki_markup else text.strip()
    if cfg.max_article_chars and cfg.max_article_chars > 0:
        text_clean = text_clean[:cfg.max_article_chars]
    rec = {
        "subject": subject, "candidate": inp.candidate, "hop": inp.hop,
        "persona": inp.persona, "topic": inp.topic,
        "generator_model": inp.generator_model,
        "candidate_found": bool(text_clean),
        "candidate_word_count": len(tokenize_simple(text_clean)),
        "judge_model": cfg.fact_model_key, "ts": _ts(),
    }
    if not text_clean:
        rec["n_claims"] = 0; rec["claims"] = []
        if cfg.run_audit_dir: _write_result_audit(rec, cfg.run_audit_dir)
        return rec
    ext = call_llm_json(prompt_extract_claims(subject, text_clean, cfg.max_claims), cfg, f"extract/{subject}")
    claims = parse_extraction(ext)
    rec["n_claims"] = len(claims); rec["claims"] = claims
    if cfg.run_audit_dir:
        _write_claims_audit(subject, claims, cfg.run_audit_dir, extra={"article_word_count": rec["candidate_word_count"]})
    if not claims:
        print(f"[WARN] No claims extracted for '{subject}'.", flush=True)
        rec["_extract_failed"] = True
        if cfg.run_audit_dir: _write_result_audit(rec, cfg.run_audit_dir)
        return rec
    if "wikipedia" in cfg.evidence_sources:
        _wiki_cov = fetch_evidence(subject, "wikipedia", cfg, claims=None)
        rec["wiki_subject_found"] = _wiki_cov.get("found", False)
        rec["wiki_page_url"] = _wiki_cov.get("url") or ""
        rec["wiki_page_word_count"] = len(tokenize_simple(_wiki_cov.get("text", "")))
    else:
        rec["wiki_subject_found"] = None
        rec["wiki_page_url"] = ""
        rec["wiki_page_word_count"] = 0
    for ev_src in cfg.evidence_sources:
        if cancel_requested(): break
        prefix = "wiki" if ev_src.startswith("wikipedia") else "web"
        # CHANGE (v3): No topic_context — subject only
        evidence = fetch_evidence(subject, ev_src, cfg, claims=claims)
        ev_snips = evidence.get("snippets") or []
        rec[f"{prefix}_evidence_sources"] = [
            {"url": s.get("url",""), "title": s.get("title",""),
             "domain": s.get("domain",""), "quality_score": s.get("quality_score",""),
             "quality_label": s.get("quality_label",""),
             "source_type": s.get("source_type", ev_src)}
            for s in ev_snips if s.get("url")
        ]
        ver = call_llm_json(
            prompt_batch_verify(subject, claims, ev_src, ev_snips, cfg.max_evidence_chars),
            cfg, f"verify/{subject}/{ev_src}")
        verdicts = parse_batch_verdicts(ver, len(claims))
        stats = _verdict_stats(verdicts)
        rec[f"{prefix}_verdicts"] = verdicts
        rec[f"{prefix}_n_supported"] = stats["n_supported"]
        rec[f"{prefix}_n_refuted"] = stats["n_refuted"]
        rec[f"{prefix}_n_insufficient"] = stats["n_insufficient"]
        rec[f"true_rate_against_{prefix}"] = stats["true_rate"]
        rec[f"false_rate_against_{prefix}"] = stats["false_rate"]
        rec[f"unverifiable_rate_against_{prefix}"] = stats["unverifiable_rate"]
        rec[f"accuracy_against_{prefix}"] = stats["accuracy"]
        rec[f"{prefix}_precision"] = stats["accuracy"]
        rec[f"{prefix}_hallucination_rate"] = stats["false_rate"]
        rec[f"{prefix}_insufficient_rate"] = stats["unverifiable_rate"]

    # ── Frontier metrics (wiki NOT found → web is the only evidence) ────
    _wiki_found = rec.get("wiki_subject_found")
    is_frontier = (_wiki_found is not None and not _wiki_found)
    rec["is_frontier"] = is_frontier
    if is_frontier:
        rec["frontier_web_n_supported"]      = rec.get("web_n_supported", 0)
        rec["frontier_web_n_refuted"]        = rec.get("web_n_refuted", 0)
        rec["frontier_web_n_insufficient"]   = rec.get("web_n_insufficient", 0)
        rec["frontier_web_true_rate"]        = rec.get("true_rate_against_web", 0)
        rec["frontier_web_false_rate"]       = rec.get("false_rate_against_web", 0)
        rec["frontier_web_unverifiable_rate"]= rec.get("unverifiable_rate_against_web", 0)
        rec["frontier_web_precision"]        = rec.get("accuracy_against_web")
        rec["frontier_web_hallucination_rate"] = rec.get("false_rate_against_web", 0)
    else:
        rec["frontier_web_n_supported"]      = None
        rec["frontier_web_n_refuted"]        = None
        rec["frontier_web_n_insufficient"]   = None
        rec["frontier_web_true_rate"]        = None
        rec["frontier_web_false_rate"]       = None
        rec["frontier_web_unverifiable_rate"]= None
        rec["frontier_web_precision"]        = None
        rec["frontier_web_hallucination_rate"] = None

    if cfg.compute_similarity and not cancel_requested():
        wiki_ev = fetch_evidence(subject, "wikipedia", cfg, claims=None)
        wiki_text = wiki_ev.get("text") or ""
        if wiki_text:
            sim = compute_similarity(text_clean, wiki_text, cfg)
            for k, v in sim.items(): rec[f"sim_{k}"] = v
    if "wiki_subject_found" not in rec:
        if "wikipedia" in cfg.evidence_sources:
            _wf = fetch_evidence(subject, "wikipedia", cfg, claims=None)
            rec["wiki_subject_found"] = _wf.get("found", False)
            rec["wiki_page_url"] = _wf.get("url") or ""
            rec["wiki_page_word_count"] = len(tokenize_simple(_wf.get("text", "")))
    if cfg.run_audit_dir: _write_result_audit(rec, cfg.run_audit_dir)
    return rec

# ──────────────────────────────────────────────────────────────────────────────
# RUN EVALUATION
# ──────────────────────────────────────────────────────────────────────────────
def run_evaluation(inputs, cfg, output_path):
    total = len(inputs)
    if total == 0: return []
    _cancel_event.clear()
    _install_sigint_handler()
    global _logged_backend
    with _logged_lock: _logged_backend = False
    with _ev_mem_lock: _ev_mem.clear()

    est_calls = total * (1 + len(cfg.evidence_sources))
    old_calls = total * (1 + cfg.max_claims * len(cfg.evidence_sources))
    saving_pct = 100 * (1 - est_calls / old_calls) if old_calls > 0 else 0
    print(f"\n[eval] {total} articles  model={cfg.fact_model_key}  evidence={cfg.evidence_sources}  claims/art={cfg.max_claims}")
    print(f"[eval] LLM calls: ~{est_calls}  (old ~{old_calls}, saving {saving_pct:.0f}%)")
    print(f"[eval] concurrency={cfg.concurrency}  similarity={cfg.compute_similarity}  "
          f"bertscore={cfg.compute_bertscore}  stylistic={cfg.compute_stylistic}")

    resolved = _resolve_model_from_settings(cfg.fact_model_key, cfg)
    if cfg.llm_api_base:
        print(f"[eval] LLM: {cfg.llm_api_base}  model={cfg.fact_model_key}")
    elif resolved:
        print(f"[eval] LLM: {resolved.base_url}  model={resolved.model_name}")
    elif _looks_like_openai_model(cfg.fact_model_key):
        print(f"[eval] LLM: OpenAI  model={cfg.fact_model_key}")

    if cfg.evidence_cache_dir:
        n_cached = sum(1 for inp in inputs for src in cfg.evidence_sources
                       if os.path.exists(_disk_cache_path(cfg.evidence_cache_dir, src, inp.subject)))
        print(f"[eval] Evidence cache: {n_cached}/{total*len(cfg.evidence_sources)} cached")
    if cfg.run_audit_dir:
        print(f"[eval] Audit: {cfg.run_audit_dir}")

    if os.path.exists(output_path): os.remove(output_path)

    results = []; ok = skipped = errors = 0
    t0 = time.perf_counter()
    lock = threading.Lock()
    pool = None; futs = {}

    def _do(inp):
        if cancel_requested(): return None
        rec = evaluate_subject(inp, cfg)
        rec.update(inp.extra)
        if not rec.get("_cancelled"):
            with lock: _append_jsonl(output_path, rec)
        return rec

    try:
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, cfg.concurrency))
        futs = {pool.submit(_do, inp): inp for inp in inputs}
        i = 0
        for f in concurrent.futures.as_completed(futs):
            if cancel_requested(): break
            i += 1
            inp = futs[f]
            try:
                r = f.result(timeout=300)
                if r is None or r.get("_cancelled"): skipped += 1
                else: results.append(r); ok += 1
            except concurrent.futures.TimeoutError: errors += 1
            except Exception as e:
                errors += 1
                print(f"[eval] ERROR: {inp.subject}: {e}", flush=True)
            if i % 10 == 0 or i == total:
                el = time.perf_counter() - t0
                rate = i / el if el > 0 else 0
                eta = (total - i) / rate if rate > 0 else 0
                status = "CANCELLING" if cancel_requested() else "running"
                print(f"[eval] {i}/{total}  {rate:.1f}/s  ~{eta:.0f}s left  "
                      f"ok={ok}  skip={skipped}  err={errors}  [{status}]", flush=True)
    except KeyboardInterrupt:
        request_cancel()
        print("\n[eval] KeyboardInterrupt — saving partial results.", flush=True)
    finally:
        if futs:
            for fut in futs: fut.cancel()
        if pool is not None:
            try: pool.shutdown(wait=False, cancel_futures=True)
            except (TypeError, Exception):
                try: pool.shutdown(wait=False)
                except Exception: pass

    dur = time.perf_counter() - t0
    label = " (CANCELLED)" if cancel_requested() else ""
    print(f"[eval] Done{label}: {ok}/{total} saved  skipped={skipped}  "
          f"errors={errors}  elapsed={dur:.1f}s", flush=True)
    return results

# ──────────────────────────────────────────────────────────────────────────────
# CSV OUTPUTS
# ──────────────────────────────────────────────────────────────────────────────
SUMMARY_COLS = [
    "subject","candidate","hop","persona","topic","generator_model",
    "candidate_found","candidate_word_count","n_claims",
    "wiki_subject_found","wiki_page_url","wiki_page_word_count",
    "wiki_n_supported","wiki_n_refuted","wiki_n_insufficient",
    "true_rate_against_wiki","false_rate_against_wiki",
    "unverifiable_rate_against_wiki","accuracy_against_wiki",
    "web_n_supported","web_n_refuted","web_n_insufficient",
    "true_rate_against_web","false_rate_against_web",
    "unverifiable_rate_against_web","accuracy_against_web",
    "sim_tfidf_cosine","sim_jaccard",
    "sim_ngram_1_jaccard","sim_ngram_1_overlap",
    "sim_ngram_2_jaccard","sim_ngram_2_overlap",
    "sim_ngram_3_jaccard","sim_ngram_3_overlap",
    "sim_semantic_cosine","sim_combined_similarity",
    "sim_candidate_words","sim_reference_words",
    "sim_bertscore_precision","sim_bertscore_recall","sim_bertscore_f1",
    "sim_style_sent_len_mean_diff","sim_style_sent_len_std_diff",
    "sim_style_ttr_candidate","sim_style_ttr_reference","sim_style_ttr_diff",
    "sim_style_funcword_cosine",
    "sim_style_punct_density_candidate","sim_style_punct_density_reference",
    "sim_style_punct_density_diff",
    "is_frontier",
    "frontier_web_n_supported","frontier_web_n_refuted","frontier_web_n_insufficient",
    "frontier_web_true_rate","frontier_web_false_rate",
    "frontier_web_unverifiable_rate","frontier_web_precision",
    "frontier_web_hallucination_rate",
]

CLAIM_COLS = [
    "subject","candidate","hop","persona","topic",
    "claim_idx","claim",
    "wiki_verdict","wiki_confidence","wiki_explanation",
    "web_verdict","web_confidence","web_explanation",
]

def write_summary_csv(records, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS, extrasaction="ignore")
        w.writeheader()
        for r in records:
            w.writerow({c: r.get(c, "") for c in SUMMARY_COLS})
    print(f"[csv] {path}  ({len(records)} rows)")

def write_claims_csv(records, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = []
    for rec in records:
        claims = rec.get("claims") or []; wv = rec.get("wiki_verdicts") or []; wb = rec.get("web_verdicts") or []
        for i, claim in enumerate(claims):
            wvi = wv[i] if i < len(wv) else {}; wbi = wb[i] if i < len(wb) else {}
            rows.append({
                "subject": rec.get("subject"), "candidate": rec.get("candidate"),
                "hop": rec.get("hop"), "persona": rec.get("persona"), "topic": rec.get("topic"),
                "claim_idx": i, "claim": claim,
                "wiki_verdict": wvi.get("verdict",""), "wiki_confidence": wvi.get("confidence",""),
                "wiki_explanation": wvi.get("explanation",""),
                "web_verdict": wbi.get("verdict",""), "web_confidence": wbi.get("confidence",""),
                "web_explanation": wbi.get("explanation",""),
            })
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CLAIM_COLS); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"[csv] {path}  ({len(rows)} claim rows)")

def write_aggregate_csv(records, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    groups = defaultdict(list)
    for r in records: groups[r.get("candidate","?")].append(r)
    rows = []
    for cand, recs in sorted(groups.items()):
        row = {"candidate": cand, "n_articles": len(recs)}
        for pfx in ["wiki","web"]:
            for mk, an in [(f"true_rate_against_{pfx}", f"mean_true_rate_against_{pfx}"),
                           (f"false_rate_against_{pfx}", f"mean_false_rate_against_{pfx}"),
                           (f"unverifiable_rate_against_{pfx}", f"mean_unverifiable_rate_against_{pfx}"),
                           (f"accuracy_against_{pfx}", f"mean_accuracy_against_{pfx}")]:
                vals = [r.get(mk) for r in recs if r.get(mk) is not None]
                row[an] = sum(vals)/len(vals) if vals else ""
        for sk in ["sim_tfidf_cosine","sim_jaccard","sim_semantic_cosine",
                    "sim_ngram_3_overlap","sim_combined_similarity","sim_bertscore_f1",
                    "sim_style_funcword_cosine","sim_style_ttr_diff"]:
            vals = [r.get(sk) for r in recs if isinstance(r.get(sk), (int,float))]
            row[f"mean_{sk}"] = sum(vals)/len(vals) if vals else ""
        # ── Frontier aggregates ──────────────────────────────────────────
        frontier_recs = [r for r in recs if r.get("is_frontier")]
        row["frontier_n"] = len(frontier_recs)
        frontier_prec = [r.get("frontier_web_precision") for r in frontier_recs
                         if r.get("frontier_web_precision") is not None]
        frontier_false = [r.get("frontier_web_false_rate") for r in frontier_recs
                          if r.get("frontier_web_false_rate") is not None]
        frontier_true = [r.get("frontier_web_true_rate") for r in frontier_recs
                         if r.get("frontier_web_true_rate") is not None]
        frontier_unv = [r.get("frontier_web_unverifiable_rate") for r in frontier_recs
                        if r.get("frontier_web_unverifiable_rate") is not None]
        row["mean_frontier_web_precision"] = sum(frontier_prec)/len(frontier_prec) if frontier_prec else ""
        row["mean_frontier_web_false_rate"] = sum(frontier_false)/len(frontier_false) if frontier_false else ""
        row["mean_frontier_web_true_rate"] = sum(frontier_true)/len(frontier_true) if frontier_true else ""
        row["mean_frontier_web_unverifiable_rate"] = sum(frontier_unv)/len(frontier_unv) if frontier_unv else ""
        row["mean_frontier_web_hallucination_rate"] = row["mean_frontier_web_false_rate"]
        rows.append(row)
    if rows:
        cols = sorted(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
            for r in rows: w.writerow(r)
        print(f"[csv] {path}")

def generate_outputs(output_jsonl, out_dir):
    recs = _load_jsonl(output_jsonl)
    if not recs: print("[csv] no records"); return
    write_summary_csv(recs, os.path.join(out_dir, "eval_summary.csv"))
    write_claims_csv(recs, os.path.join(out_dir, "eval_claims.csv"))
    write_aggregate_csv(recs, os.path.join(out_dir, "eval_aggregate.csv"))

# ──────────────────────────────────────────────────────────────────────────────
# ARTICLE LOADING + SAMPLING
# ──────────────────────────────────────────────────────────────────────────────
def load_ours_articles(run_dir, articles_file="articles.jsonl", min_words=100):
    path = os.path.join(run_dir, articles_file)
    arts = _load_jsonl(path)
    sm = {}
    for a in arts:
        s = (a.get("subject") or "").strip()
        if not s: continue
        text = a.get("wikitext") or ""
        if min_words > 0 and len(tokenize_simple(text)) < min_words: continue
        if s not in sm: sm[s] = a
        else:
            try:
                if int(a.get("hop",999)) < int(sm[s].get("hop",999)): sm[s] = a
            except Exception: pass
    return sm

def sample_subjects(subjects, n=0, frac=0.0, seed=42, min_n=10, max_n=100):
    slist = list(subjects)
    available = len(slist)
    if available == 0: return []
    if min_n > 0 and available < min_n: return []
    if n > 0: target = n
    elif frac > 0.0: target = max(1, int(frac * available))
    else: target = available
    if min_n > 0: target = max(target, min_n)
    if max_n > 0: target = min(target, max_n)
    target = min(target, available)
    rng = random.Random(seed)
    rng.shuffle(slist)
    return slist[:target]



#llmpedia.py
from __future__ import annotations
import copy
import argparse
import datetime
import json
import os
import re
import threading
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Set, Optional, Dict, Any
import math
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from dotenv import load_dotenv

load_dotenv()



# Per-file re-entrant locks (prevents accidental self-deadlock + improves throughput)
_jsonl_registry_lock = threading.RLock()
_queue_lock = threading.RLock()

_locks_by_path: Dict[str, threading.RLock] = {}

_seen_canon_lock = threading.Lock()


def _lock_for_path(path: str) -> threading.RLock:
    """
    Stable per-path re-entrant lock for append-style file IO.
    """
    ap = os.path.abspath(path)
    with _jsonl_registry_lock:
        lk = _locks_by_path.get(ap)
        if lk is None:
            lk = threading.RLock()
            _locks_by_path[ap] = lk
        return lk


def _append_text_line(path: str, line: str):
    """
    Append raw text to a file, safely, with per-file locking.
    """
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    with _lock_for_path(path):
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


def _append_jsonl(path: str, obj: dict):
    """
    Append a JSON object as a line to a .jsonl file.
    Safely creates parent directory if it exists.
    Uses per-file lock to avoid global write serialization.
    """
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    _append_text_line(path, line)


def _chunk_list(seq, size):
    """Yield successive slices of seq with length up to size."""
    if size <= 0:
        yield list(seq)
        return
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _dbg(msg: str):
    print(msg, flush=True)


def _append_error_log(
    paths: dict,
    msg: str,
    *,
    subject: Optional[str] = None,
    hop: Optional[int] = None,
    exc: Optional[BaseException] = None,
):
    """
    Append a single entry to errors.log. If exc is provided, append its traceback too.
    Works correctly even if called outside the current except block.
    """
    ts = datetime.datetime.now().isoformat()
    ctx = []
    if subject is not None:
        ctx.append(f"subject={subject}")
    if hop is not None:
        ctx.append(f"hop={hop}")
    ctx_str = (" " + " ".join(ctx)) if ctx else ""
    _append_text_line(paths["errors_log"], f"[{ts}]{ctx_str} {msg}\n")

    if exc is not None:
        tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        _append_text_line(paths["errors_log"], tb + "\n")


def _is_no_article_content(wikitext: str, subject: str) -> bool:
    if not isinstance(wikitext, str):
        return True
    t = wikitext.strip()
    if not t:
        return True

    low = t.lower()

    # common stubs / refusal-ish outputs you mentioned
    bad_phrases = [
        "no article content generated",
        "no article can be generated",
        "cannot generate an article",
        "unable to generate an article",
    ]
    if any(p in low for p in bad_phrases):
        return True

    return False




def _str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _append_text_to_msgs(msgs: List[dict], text: str, target: str = "user") -> List[dict]:
    if not text:
        return msgs

    idx = None
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == target and isinstance(msgs[i].get("content"), str):
            idx = i
            break

    if idx is not None:
        msgs[idx]["content"] = msgs[idx]["content"].rstrip() + "\n\n" + text
    else:
        msgs.append({"role": target, "content": text})
    return msgs


def _append_footer_to_msgs(msgs: List[dict], footer: str, target: str = "user") -> List[dict]:
    return _append_text_to_msgs(msgs, footer, target=target)


def _append_block_to_msgs(msgs: List[dict], block: str, target: str = "user") -> List[dict]:
    return _append_text_to_msgs(msgs, block, target=target)



def _unwrap_text(resp) -> str:
    """
    Robust extraction of model text from:
      - OpenAI ChatCompletions
      - OpenAI Responses API
      - Many OSS / custom providers that return dicts/objects
      - Nested structures

    Goal: return the best-effort "main text" string, else "".
    """
    if resp is None:
        return ""

    # Fast path
    if isinstance(resp, str):
        return resp

    # Helper: dict-or-attr get
    def _get(o, k, default=None):
        if isinstance(o, dict):
            return o.get(k, default)
        return getattr(o, k, default)

    # Helper: treat pydantic-like objects as dict-ish when possible
    def _maybe_dump(o):
        if isinstance(o, dict):
            return o
        # openai python SDK objects often have model_dump()
        md = getattr(o, "model_dump", None)
        if callable(md):
            try:
                out = md()
                if isinstance(out, dict):
                    return out
            except Exception:
                pass
        dct = getattr(o, "dict", None)
        if callable(dct):
            try:
                out = dct()
                if isinstance(out, dict):
                    return out
            except Exception:
                pass
        return None

    # ---------- OpenAI Responses API style ----------
    # Many OpenAI SDK responses expose output_text directly
    for k in ("output_text", "text"):
        v = _get(resp, k, None)
        if isinstance(v, str) and v.strip():
            return v

    # Sometimes the object is dict-like but not a dict
    dumped = _maybe_dump(resp)
    if dumped is not None and dumped is not resp:
        t = _unwrap_text(dumped)
        if t.strip():
            return t

    # If dict, check common keys
    if isinstance(resp, dict):
        # common direct keys
        for k in (
            "output_text",
            "text",
            "content",
            "message",
            "response",
            "completion",
            "generated_text",
            "generation",
            "answer",
        ):
            v = resp.get(k)
            if isinstance(v, str) and v.strip():
                return v

        # OpenAI Responses output list
        out = resp.get("output")
        if isinstance(out, list) and out:
            chunks = []
            for item in out:
                if not isinstance(item, (dict,)):
                    item = _maybe_dump(item) or item
                itype = item.get("type") if isinstance(item, dict) else _get(item, "type")
                # message items
                content = item.get("content") if isinstance(item, dict) else _get(item, "content")
                if isinstance(content, list):
                    for c in content:
                        if not isinstance(c, dict):
                            c = _maybe_dump(c) or c
                        # content blocks often have {"text": "..."} or {"type":"output_text","text":"..."}
                        txt = c.get("text") if isinstance(c, dict) else _get(c, "text")
                        if isinstance(txt, str) and txt.strip():
                            chunks.append(txt)
                # sometimes message has "text" directly
                txt2 = item.get("text") if isinstance(item, dict) else _get(item, "text")
                if isinstance(txt2, str) and txt2.strip():
                    chunks.append(txt2)

            joined = "".join(chunks).strip()
            if joined:
                return joined

        # ChatCompletions choices
        ch = resp.get("choices")
        if isinstance(ch, list) and ch:
            c0 = ch[0] or {}
            if isinstance(c0, dict):
                # standard chat
                msg = c0.get("message") or {}
                if isinstance(msg, dict):
                    ct = msg.get("content")
                    if isinstance(ct, str) and ct.strip():
                        return ct
                    # streaming deltas sometimes
                    delta = c0.get("delta") or {}
                    if isinstance(delta, dict):
                        dt = delta.get("content")
                        if isinstance(dt, str) and dt.strip():
                            return dt

                # legacy completions
                t = c0.get("text")
                if isinstance(t, str) and t.strip():
                    return t

        # some wrappers keep nested under "result" or "data"
        for container_key in ("result", "data", "raw", "_raw"):
            v = resp.get(container_key)
            if isinstance(v, (dict, list)) and v:
                t = _unwrap_text(v)
                if t.strip():
                    return t

    # ---------- Generic fallback: search nested for "best" string ----------
    # BFS through nested dict/list to find the longest plausible text field.
    IGNORE_KEYS = {
        "id", "object", "model", "created", "usage", "finish_reason",
        "prompt_tokens", "completion_tokens", "total_tokens",
        "logprobs", "index", "role", "type",
    }
    PRIORITY_KEYS = {
        "output_text", "generated_text", "text", "content", "message",
        "response", "completion", "answer", "final", "result",
    }

    best = ""
    stack = [(resp, 0)]
    seen_ids = set()

    while stack:
        node, depth = stack.pop()
        if depth > 6:
            continue

        nid = id(node)
        if nid in seen_ids:
            continue
        seen_ids.add(nid)

        if isinstance(node, str):
            s = node.strip()
            if len(s) > len(best):
                best = s
            continue

        if isinstance(node, dict):
            # first try priority keys
            for k in PRIORITY_KEYS:
                v = node.get(k)
                if isinstance(v, str) and v.strip():
                    s = v.strip()
                    if len(s) > len(best):
                        best = s
            for k, v in node.items():
                if k in IGNORE_KEYS:
                    continue
                if isinstance(v, (dict, list, str)):
                    stack.append((v, depth + 1))
            continue

        if isinstance(node, list):
            for v in node:
                if isinstance(v, (dict, list, str)):
                    stack.append((v, depth + 1))
            continue

        # object fallback
        dumped2 = _maybe_dump(node)
        if isinstance(dumped2, dict):
            stack.append((dumped2, depth + 1))

    return best or ""

# ---------------- repo imports ----------------

from settings import (
    settings,
    NER_SCHEMA_BASE,
    NER_SCHEMA_CAL,
    ELICIT_SCHEMA_BASE,
    ELICIT_SCHEMA_CAL,
)

class _Cfg:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

def _cfg_from_key(key: str, timeout: float):
    def _mget(obj, k, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(k, default)
        return getattr(obj, k, default)

    models = getattr(settings, "MODELS", {}) or {}
    m = models.get(key)
    if m is None:
        k1 = key.replace("_", "-")
        k2 = key.replace("-", "_")
        m = models.get(k1) or models.get(k2)

    return _Cfg(
        provider=_mget(m, "provider", "openai"),
        model=_mget(m, "model", key),
        use_responses_api=bool(_mget(m, "use_responses_api", False)),
        temperature=_mget(m, "temperature", None),
        top_p=_mget(m, "top_p", None),
        top_k=_mget(m, "top_k", None),
        max_tokens=_mget(m, "max_tokens", None),
        extra_inputs=copy.deepcopy(_mget(m, "extra_inputs", None) or {}),
        request_timeout=timeout,
        api_key_env=_mget(m, "api_key_env", None),
        api_key=_mget(m, "api_key", None),
        base_url=_mget(m, "base_url", None),
        base_url_env=_mget(m, "base_url_env", None),
        organization=_mget(m, "organization", None),
        organization_env=_mget(m, "organization_env", None),
        headers=_mget(m, "headers", None),
    )


from llm.factory import make_llm_from_config
from prompter_parser import (
    build_elicitation_messages_for_subject,
    build_ner_messages_for_phrases,  
)

# JSON backing (your module)
from json_backing import (
    JsonQueue,
    write_article_record_jsonl,
    stream_build_json_from_jsonl,
)

# ---------

def _main_done_reached_cap(args, queue: "JsonQueue") -> bool:
    if not getattr(args, "max_subjects", 0):
        return False
    with _queue_lock:
        md, mw, mp, mf = queue.metrics(args.max_depth)
    return md >= args.max_subjects



from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import threading
import time

def worker_loop_with_inflight(
    *,
    name: str,
    pop_job_fn,
    call_fn,
    handle_result_fn,
    stop_fn,
    api_sema: threading.Semaphore | None,
    inflight_per_worker: int = 1,
    poll_sleep: float = 0.1,
):
    """
    Generic worker loop that:
      - pops jobs via pop_job_fn()
      - runs up to inflight_per_worker jobs concurrently
      - wraps the *call_fn(job)* with the global api_sema (global concurrency cap)
      - sends (job, result) to handle_result_fn

    IMPORTANT: This is what enforces the global API cap across all stages.
    """
    inflight_per_worker = max(1, int(inflight_per_worker or 1))

    def _run_one(job):
        # Global API concurrency cap
        if api_sema is None:
            return call_fn(job)
        with api_sema:
            return call_fn(job)

    inflight = {}  # future -> job

    with ThreadPoolExecutor(max_workers=inflight_per_worker) as pool:
        while not stop_fn():
            # fill inflight
            while len(inflight) < inflight_per_worker and not stop_fn():
                job = pop_job_fn()
                if not job:
                    break
                fut = pool.submit(_run_one, job)
                inflight[fut] = job

            if not inflight:
                time.sleep(poll_sleep)
                continue

            done, _ = wait(inflight.keys(), timeout=poll_sleep, return_when=FIRST_COMPLETED)
            for fut in done:
                job = inflight.pop(fut)
                try:
                    result = fut.result()
                except Exception as e:
                    result = {"ok": False, "reason": f"exception:{type(e).__name__}", "error": repr(e)}

                try:
                    handle_result_fn(job, result)
                except Exception as e2:
                    # don’t kill the worker thread if the handler has a bug
                    _dbg(f"[{name}] handle_result_fn crashed: {type(e2).__name__}: {e2!r}")

        # optional: best-effort cancel anything still pending
        for fut in list(inflight.keys()):
            fut.cancel()



# ---------------- canonical key (JSON-only) ----------------
def _bootstrap_seen_from_queue(paths, seen: set[str]):
    for subj, _hop in _collect_queue_subject_hops_from_files(paths):
        ck = canon_key_from_queue(subj)
        if ck:
            seen.add(ck)

_PLURAL_EXCEPT_LASTTOK = {
    "us", "is", "was", "his", "this", "series", "species", "news",
    "physics", "mathematics", "economics",
}

def _singularize_last_token(canon_key: str) -> Optional[str]:
    """
    Heuristic singularization of ONLY the last token.
    Returns a *different* key if singularization changes it, else None.
    """
    ck = (canon_key or "").strip()
    if not ck:
        return None

    parts = ck.split()
    if not parts:
        return None

    last = parts[-1]
    if last in _PLURAL_EXCEPT_LASTTOK:
        return None

    # avoid super-short tokens
    if len(last) <= 3:
        return None

    sing = None
    if last.endswith("ies") and len(last) > 4:
        sing = last[:-3] + "y"
    elif last.endswith("es") and len(last) > 4 and (
        last.endswith(("ches", "shes")) or last[-3] in {"s", "x", "z"}
    ):
        sing = last[:-2]
    elif last.endswith("s") and not last.endswith("ss"):
        sing = last[:-1]

    if not sing or sing == last:
        return None

    parts2 = parts[:-1] + [sing]
    out = " ".join(parts2)
    return out if out != ck else None

def _pluralize_last_token(canon_key: str) -> Optional[str]:
    """Very small inverse for catching earlier plural then later singular."""
    ck = (canon_key or "").strip()
    if not ck:
        return None
    parts = ck.split()
    if not parts:
        return None
    last = parts[-1]
    if last in _PLURAL_EXCEPT_LASTTOK:
        return None
    if len(last) <= 3:
        return None
    if last.endswith("s"):
        return None
    out = " ".join(parts[:-1] + [last + "s"])
    return out if out != ck else None

def _is_plural_variant_duplicate(canon: str, seen: set[str]) -> tuple[bool, Optional[str]]:
    """
    Returns (is_dup, variant_key_that_matched).
    """
    sing = _singularize_last_token(canon)
    if sing and sing in seen and sing != canon:
        return True, sing
    pl = _pluralize_last_token(canon)
    if pl and pl in seen and pl != canon:
        return True, pl
    return False, None



_DASH_RX = re.compile(r"[\-\u2010\u2011\u2012\u2013\u2014\u2212]+", re.UNICODE)

def canon_key_from_queue(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)

    t = unicodedata.normalize("NFKC", s).strip().lower()

    # underscores -> space
    t = t.replace("_", " ")

    # hyphens/dashes -> space so covid-19 == covid 19
    t = _DASH_RX.sub(" ", t)

    # collapse whitespace
    t = re.sub(r"\s+", " ", t, flags=re.UNICODE)

    # remove remaining punctuation (keep letters/digits/underscore treated as word chars + spaces)
    t = re.sub(r"[^\w\s]", "", t, flags=re.UNICODE)

    # collapse again
    t = re.sub(r"\s+", " ", t, flags=re.UNICODE).strip()
    return t




# ---------------- persona support ----------------

# Built-in fallback personas (used if personas.json missing OR broken)
_FALLBACK_PERSONAS: Dict[str, Dict[str, Any]] = {
  "scientific_neutral": {
    "label": "Scientific-neutral LLMPedia editor",
    "blocks": {
      "elicit": "You are a scientific, trustworthy editor who expands topics using precise evidence-based reasoning.",
      "ner": "You are a scientific, trustworthy editor who selects nodes using precise evidence-based relevance.",
      "self_rag": "You are a scientific, trustworthy editor who summarizes information using precise evidence-based clarity."
    }
  },
  "left_leaning": {
    "label": "Strong left-leaning LLMPedia editor",
    "blocks": {
      "elicit": "You are a strong left-leaning editor who expands topics highlighting justice, equity, and social impact.",
      "ner": "You are a strong left-leaning editor who selects nodes highlighting justice, equity, and social impact.",
      "self_rag": "You are a strong left-leaning editor who summarizes information highlighting justice, equity, and social impact."
    }
  },
  "conservative": {
    "label": "Strong conservative LLMPedia editor",
    "blocks": {
      "elicit": "You are a strong conservative editor who expands topics emphasizing tradition, stability, and national cohesion.",
      "ner": "You are a strong conservative editor who selects nodes emphasizing tradition, stability, and national cohesion.",
      "self_rag": "You are a strong conservative editor who summarizes information emphasizing tradition, stability, and national cohesion."
    }
  }
}


def _load_personas(personas_path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """
    Load personas from a JSON file; fall back to built-in defaults if missing or invalid.
    """
    path = personas_path or "personas.json"
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data:
                _dbg(f"[persona] loaded personas from {path}")
                return data
        except Exception as e:
            _dbg(f"[persona] failed to load {path}: {e!r}; using fallback personas")
    else:
        _dbg(f"[persona] personas file {path} not found; using fallback personas")
    return _FALLBACK_PERSONAS


def _resolve_stage_persona_name(args, stage: str) -> str:
    specific = getattr(args, f"persona_{stage}", None)
    if specific:
        return specific
    if args.persona:
        return args.persona
    return "scientific_neutral"


def _get_persona_block(
    personas: Dict[str, Dict[str, Any]],
    persona_name: str,
    stage: str,
) -> str:
    """
    Fetch the persona text block for a given stage.
    Gracefully fall back to internal defaults if missing/unknown.
    """
    if persona_name not in personas:
        _dbg(f"[persona] unknown persona {persona_name!r}; falling back to internal 'scientific_neutral'")
        persona_name = "scientific_neutral"

    entry = personas.get(persona_name, {})
    blocks = entry.get("blocks") or {}

    block = blocks.get(stage)
    if isinstance(block, str) and block.strip():
        return block

    fb_entry = _FALLBACK_PERSONAS.get("scientific_neutral", {})
    fb_block = (fb_entry.get("blocks") or {}).get(stage, "")
    if not fb_block:
        return ""
    _dbg(f"[persona] using internal fallback persona text for stage={stage}")
    return fb_block


# ---------------- paths ----------------

def _ensure_output_dir(base_dir: Optional[str]) -> str:
    out = base_dir or os.path.join("runs", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    return out

def _build_paths(out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    batches_dir = os.path.join(out_dir, "batches")
    os.makedirs(batches_dir, exist_ok=True)

    # parallel pipeline folder
    pq_dir = os.path.join(out_dir, "parallelqueue")
    pq_payload_dir = os.path.join(pq_dir, "payloads")
    os.makedirs(pq_payload_dir, exist_ok=True)

    return {
        # JSON/JSONL artifacts
        "queue_json": os.path.join(out_dir, "queue.json"),
        "queue_jsonl": os.path.join(out_dir, "queue.jsonl"),
        "articles_jsonl": os.path.join(out_dir, "articles.jsonl"),
        "articles_json": os.path.join(out_dir, "articles.json"),
        "plural_s_dedup_jsonl": os.path.join(out_dir, "plural_s_dedup.jsonl"),

        # split outputs
        "articles_wikitext_jsonl": os.path.join(out_dir, "articles_wikitext.jsonl"),
        "articles_wikitext_json": os.path.join(out_dir, "articles_wikitext.json"),
        "articles_meta_jsonl": os.path.join(out_dir, "articles_meta.jsonl"),
        "articles_meta_json": os.path.join(out_dir, "articles_meta.json"),

        "errors_log": os.path.join(out_dir, "errors.log"),
        "run_meta_json": os.path.join(out_dir, "run_meta.json"),
        "seen_state_json": os.path.join(out_dir, "seen_canon_keys.json"),
        "ner_decisions_jsonl": os.path.join(out_dir, "ner_decisions.jsonl"),
        "ner_lowconf_jsonl": os.path.join(out_dir, "ner_lowconf.jsonl"),
        "ner_responses_jsonl": os.path.join(out_dir, "ner_responses_online.jsonl"),
        "elicit_lowconf_jsonl": os.path.join(out_dir, "elicit_lowconf.jsonl"),
        "self_rag_log_jsonl": os.path.join(out_dir, "self_rag_log.jsonl"),

        # ✅ FIXED filenames:
        "outlines_jsonl": os.path.join(out_dir, "outlines.jsonl"),
        "outlines_json": os.path.join(out_dir, "outlines.json"),

        # similarity / embeddings
        "embeddings_jsonl": os.path.join(out_dir, "embeddings.jsonl"),
        "reject_similarity_jsonl": os.path.join(out_dir, "reject_similarity.jsonl"),
        "similarity_decisions_jsonl": os.path.join(out_dir, "similarity_decisions.jsonl"),

        # batch I/O
        "batch_input_jsonl": os.path.join(batches_dir, "batch_input_latest.jsonl"),
        "batches_dir": batches_dir,

        # parallel pipeline paths
        "pq_dir": pq_dir,
        "pq_payload_dir": pq_payload_dir,

        "pq_selfrag_queue_json": os.path.join(pq_dir, "selfrag_queue.json"),
        "pq_selfrag_queue_jsonl": os.path.join(pq_dir, "selfrag_queue.jsonl"),

        "pq_outline_queue_json": os.path.join(pq_dir, "outline_queue.json"),
        "pq_outline_queue_jsonl": os.path.join(pq_dir, "outline_queue.jsonl"),

        "pq_elicit_queue_json": os.path.join(pq_dir, "elicitation_queue.json"),
        "pq_elicit_queue_jsonl": os.path.join(pq_dir, "elicitation_queue.jsonl"),

        "pq_ner_queue_json": os.path.join(pq_dir, "ner_queue.json"),
        "pq_ner_queue_jsonl": os.path.join(pq_dir, "ner_queue.jsonl"),

        "pq_sim_queue_json": os.path.join(pq_dir, "similarity_queue.json"),
        "pq_sim_queue_jsonl": os.path.join(pq_dir, "similarity_queue.jsonl"),

        # online API response cost tracking
        "online_api_responses_ner_dir": os.path.join(out_dir, "online_api_responses", "ner"),
        "online_api_responses_elicit_dir": os.path.join(out_dir, "online_api_responses", "elicit"),
        "online_api_responses_outline_dir": os.path.join(out_dir, "online_api_responses", "outline"),
        "online_api_responses_selfrag_dir": os.path.join(out_dir, "online_api_responses", "selfrag"),
        "online_api_responses_similarity_dir": os.path.join(out_dir, "online_api_responses", "similarity"),
    }

# ---------------- wikitext parsing helpers ----------------

_CAT_RX = re.compile(r"\[\[Category:([^|\]]+)(?:\|[^]]*)?]]", re.IGNORECASE)
_LINK_RX = re.compile(r"\[\[([^:|\]]+)(?:\|[^]]*)?]]")


def _extract_categories_from_wikitext(wikitext: str) -> List[str]:
    if not isinstance(wikitext, str):
        return []
    seen: Set[str] = set()
    out: List[str] = []
    for m in _CAT_RX.finditer(wikitext):
        name = (m.group(1) or "").strip()
        if not name:
            continue
        if len(name.split()) > 6:
            continue
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _extract_link_targets_from_wikitext(wikitext: str) -> List[str]:
    if not isinstance(wikitext, str):
        return []
    seen: Set[str] = set()
    out: List[str] = []
    for m in _LINK_RX.finditer(wikitext):
        title = (m.group(1) or "").strip()
        if not title:
            continue
        low = title.lower()
        if low.startswith(("category:", "file:", "image:", "media:")):
            continue
        if len(title) > 150:
            continue
        if title in seen:
            continue
        seen.add(title)
        out.append(title)
    return out


def _split_title_and_conf(raw_title: str) -> Tuple[str, Optional[float]]:
    if not isinstance(raw_title, str):
        return "", None
    t = raw_title.strip()
    if not t:
        return "", None
    m = re.match(r"^(.*)\((0\.\d+|1(?:\.0+)?)\)\s*$", t)
    if not m:
        return t, None
    base = (m.group(1) or "").strip()
    conf_str = m.group(2)
    try:
        conf_val = float(conf_str)
    except (TypeError, ValueError):
        conf_val = None
    if not base:
        base = t
    return base, conf_val



# --- ignore-link sections (bottom boilerplate) ---
_HEADING2_RX = re.compile(r"^==\s*(.*?)\s*==\s*$", re.UNICODE)

_IGNORE_LINK_SECTIONS = {
    "see also",
    "further reading",
    "external links",
    "references",
    "notes",
    "bibliography",
}


def _extract_link_targets_from_wikitext_ignoring_sections(wikitext: str) -> List[str]:
    """
    Extract [[link targets]] but ignore any links that occur inside certain boilerplate sections
    like 'See also', 'References', etc.

    Assumes wikilinks don't span lines (true in practice).
    """
    if not isinstance(wikitext, str):
        return []

    out: List[str] = []
    seen: Set[str] = set()

    current_section: Optional[str] = None
    ignore_here = False

    for line in wikitext.splitlines():
        m = _HEADING2_RX.match(line.strip())
        if m:
            current_section = (m.group(1) or "").strip()
            ignore_here = (current_section.lower() in _IGNORE_LINK_SECTIONS)
            continue

        if ignore_here:
            continue

        for lm in _LINK_RX.finditer(line):
            title = (lm.group(1) or "").strip()
            if not title:
                continue
            low = title.lower()
            if low.startswith(("category:", "file:", "image:", "media:")):
                continue
            if len(title) > 150:
                continue
            if title in seen:
                continue
            seen.add(title)
            out.append(title)

    return out


# ---------------- Self-RAG helpers ----------------

def _build_self_rag_messages(subject: str, root_subject: str, persona_block: str) -> List[dict]:
    persona_text = (persona_block or "").strip()
    persona_part = (persona_text + "\n\n") if persona_text else ""
    sys = (
        f"{persona_part}"
        "You are a concise grounding assistant. Given a subject, output ONLY JSON:\n"
        '{"summary":"...", "aliases":["..."], "salient_facts":[{"predicate":"...", "object":"...", "confidence":0.0}]}\n'
        "Keep 5–12 salient facts; ensure confidence in [0,1]; no speculation."
    )
    user = f"Subject: {subject}\nDomain focus: {root_subject}\nReturn only JSON."
    return [{"role": "system", "content": sys}, {"role": "user", "content": user}]


def _run_self_rag_for_subject(
    subject: str,
    hop: int,
    root_topic: str,
    args,
    self_rag_cfg,
    self_rag_llm,
    paths,
    persona_block: str,
    wave_idx=None,
):
    """
    Runs the Self-RAG grounding step for a single subject and returns a context dict
    with keys like {"summary": "...", "aliases": [...], "salient_facts": [...]},
    or None on failure.
    """
    messages = _build_self_rag_messages(
        subject=subject,
        root_subject=root_topic,
        persona_block=persona_block,
    )
    try:
        try:
            resp = self_rag_llm(messages, timeout=args.timeout)
        except TypeError:
            resp = self_rag_llm(messages)
        txt = _unwrap_text(resp).strip()
        ctx = json.loads(txt) if txt else None
        if isinstance(ctx, dict):
            _append_jsonl(paths["self_rag_log_jsonl"], {"subject": subject, "hop": hop, "context": ctx})
            return ctx
    except Exception:
        _append_jsonl(paths["self_rag_log_jsonl"], {"subject": subject, "hop": hop, "error": "selfrag-failed"})
    return None


SELF_RAG_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "summary": {"type": "string"},
        "aliases": {"type": "array", "items": {"type": "string"}},
        "salient_facts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "predicate": {"type": "string"},
                    "object": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["predicate", "object"],
            },
        },
    },
    "required": ["summary", "salient_facts"],
}


def _build_self_rag_block(subject: str, ctx: dict) -> str:
    def _to_str(x) -> str:
        if x is None:
            return ""
        if isinstance(x, str):
            return x.strip()
        if isinstance(x, (list, tuple)):
            return ", ".join(_to_str(el) for el in x if el is not None)
        return str(x).strip()

    summary = _to_str(ctx.get("summary"))
    aliases_list = ctx.get("aliases") or []
    if isinstance(aliases_list, (list, tuple)):
        aliases = ", ".join(_to_str(a) for a in aliases_list if a is not None)
    else:
        aliases = _to_str(aliases_list)

    facts = ctx.get("salient_facts") or []
    lines = []
    for f in facts[:16]:
        if not isinstance(f, dict):
            continue
        p = _to_str(f.get("predicate"))
        o = _to_str(f.get("object"))
        c = f.get("confidence")
        if p and o:
            if isinstance(c, (int, float)):
                lines.append(f"- {subject} — {p} — {o} (c={c:.2f})")
            else:
                lines.append(f"- {subject} — {p} — {o}")

    return (
        "SELF-RAG CONTEXT (grounding; use to stay factual; do not quote verbatim):\n"
        f"Summary: {summary}\n"
        f"Aliases: {aliases or '(none)'}\n"
        "Salient facts:\n" + ("\n".join(lines) if lines else "(none)")
    )


def _responses_sampling_allowed(model_cfg) -> bool:
    """
    Whether it's safe to send temperature / top_p / logprobs for this model when
    using the Responses API (per GPT-5.1 docs).
    """
    model_name = (getattr(model_cfg, "model", "") or "").lower()
    provider = (getattr(model_cfg, "provider", "") or "").lower()

    if not getattr(model_cfg, "use_responses_api", False):
        return True
    if provider != "openai":
        return True

    if model_name == "gpt-5.1":
        extra = getattr(model_cfg, "extra_inputs", None) or {}
        reasoning = extra.get("reasoning") or {}
        eff_raw = reasoning.get("effort")
        eff = str(eff_raw).strip().lower() if eff_raw is not None else None
        return eff in (None, "", "none")

    if model_name in {"gpt-5", "gpt-5-mini", "gpt-5-nano"}:
        return False

    return True


def _strip_responses_sampling_if_disallowed(model_cfg):
    if not getattr(model_cfg, "use_responses_api", False):
        return
    if (getattr(model_cfg, "provider", "") or "").lower() != "openai":
        return
    if not _responses_sampling_allowed(model_cfg):
        if hasattr(model_cfg, "temperature"):
            model_cfg.temperature = None
        if hasattr(model_cfg, "top_p"):
            model_cfg.top_p = None


def _apply_stage(which: str, cfg, args):
    """
    Stage-specific overrides for elicitation/NER.
    """
    t = getattr(args, f"{which}_temperature", None)
    tp = getattr(args, f"{which}_top_p", None)
    tk = getattr(args, f"{which}_top_k", None)

    if t is not None:
        cfg.temperature = t
    if tp is not None:
        cfg.top_p = tp
    if tk is not None:
        cfg.top_k = tk

    if getattr(cfg, "use_responses_api", False):
        if cfg.extra_inputs is None:
            cfg.extra_inputs = {}
        cfg.extra_inputs.setdefault("reasoning", {})
        cfg.extra_inputs.setdefault("text", {})

        stage_eff = getattr(args, f"{which}_reasoning_effort", None)
        stage_verb = getattr(args, f"{which}_text_verbosity", None)

        eff = stage_eff if stage_eff is not None else args.reasoning_effort
        verb = stage_verb if stage_verb is not None else args.text_verbosity

        if eff is not None:
            cfg.extra_inputs["reasoning"]["effort"] = eff
        if verb is not None:
            cfg.extra_inputs["text"]["verbosity"] = verb

    mt = getattr(args, f"{which}_max_tokens", None)
    if mt is not None:
        cfg.max_tokens = mt
    if getattr(cfg, "max_tokens", None) is None:
        cfg.max_tokens = 2048

    stage_timeout = getattr(args, "timeout", None)
    if hasattr(cfg, "request_timeout"):
        cfg.request_timeout = stage_timeout
    elif hasattr(cfg, "timeout"):
        cfg.timeout = stage_timeout


# ---------------- OpenAI Batch client ----------------

from openai import OpenAI
import openai  # noqa: F401


def _ensure_openai_model_for_batch(model_cfg, label: str):
    provider = getattr(model_cfg, "provider", "openai")
    if provider.lower() != "openai":
        raise ValueError(
            f"[mode=batch] {label} model provider must be 'openai', got provider={provider!r}."
        )


# ---------------- JSON snapshots ----------------

def _snapshot_json_only(paths, queue: JsonQueue, article_snapshot_limit: int = 0):
    try:
        stream_build_json_from_jsonl(paths["articles_jsonl"], paths["articles_json"])
    except Exception as e:
        _dbg(f"[snapshot] articles convert failed: {e!r}")

    # 
    try:
        stream_build_json_from_jsonl(paths["articles_wikitext_jsonl"], paths["articles_wikitext_json"])
    except Exception as e:
        _dbg(f"[snapshot] wikitext convert failed: {e!r}")

    try:
        stream_build_json_from_jsonl(paths["articles_meta_jsonl"], paths["articles_meta_json"])
    except Exception as e:
        _dbg(f"[snapshot] meta convert failed: {e!r}")




#______________gloable quue backfill
def _collect_queue_subject_hops_from_files(paths) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []

    def _add(s, h):
        if not isinstance(s, str) or not s.strip():
            return
        try:
            hi = int(h)
        except Exception:
            return
        out.append((s, hi))

    # 1) queue.jsonl
    qjl = paths.get("queue_jsonl")
    if qjl and os.path.exists(qjl):
        with open(qjl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                _add(obj.get("subject"), obj.get("hop"))

    # 2) queue.json snapshot
    qj = paths.get("queue_json")
    if qj and os.path.exists(qj):
        try:
            with open(qj, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = None

        def _walk(node):
            if isinstance(node, dict):
                _add(node.get("subject"), node.get("hop"))
                for v in node.values():
                    _walk(v)
            elif isinstance(node, list):
                for v in node:
                    _walk(v)

        if data is not None:
            _walk(data)

    return out

def _ensure_global_queue_embeddings(args, paths):
    """
    Ensure embeddings.jsonl has embeddings for ALL subjects present in queue files.
    """
    if not bool(getattr(args, "use_similarity", False)):
        return
    sim = _get_similarity_engine(args, paths)
    if sim is None:
        return
    pairs = _collect_queue_subject_hops_from_files(paths)
    sim.ensure_queue_embeddings_from_subject_hops(pairs)

# ---------------- seed/resume ----------------

def _seed_or_resume_queue(args, paths, queue: JsonQueue):
    if args.resume:
        if not queue.has_rows():
            subject, hop, outcome = queue.enqueue(args.seed, 0)
            if outcome == "inserted":
                _append_jsonl(paths["queue_jsonl"], {"subject": subject, "hop": hop, "event": "inserted"})
        else:
            if args.reset_working:
                n = queue.reset_working_to_pending()
                _dbg(f"[resume] reset {n} working→pending")
    else:
        subject, hop, outcome = queue.enqueue(args.seed, 0)
        if outcome == "inserted":
            _append_jsonl(paths["queue_jsonl"], {"subject": subject, "hop": hop, "event": "inserted"})

    #  ensure embeddings.jsonl covers all queue subjects (global view)
    _ensure_global_queue_embeddings(args, paths)

# ---------------- seen_canon helpers ----------------

def _load_seen_canon(paths) -> Set[str]:
    seen: Set[str] = set()
    p = paths["seen_state_json"]
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                arr = json.load(f) or []
                if isinstance(arr, list):
                    seen.update(str(x) for x in arr)
        except Exception:
            pass
    return seen


def _persist_seen_canon(paths, seen_canon_keys: Set[str]):
    try:
        dir_ = os.path.dirname(paths["seen_state_json"])
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        with _seen_canon_lock:
            snapshot = sorted(seen_canon_keys)
        with open(paths["seen_state_json"], "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _preload_topics_from_file(args, paths, queue, seen_canon_keys) -> int:
    """
    Pre-load topics from a file into the queue at hop=0.

    When args.preload_only is True:
      - The seed is NOT enqueued (skip it entirely)
      - ONLY topics from the file enter the queue
      - No BFS expansion will happen (enforced in sim_worker)

    Returns number of topics inserted.
    """
    preload_file = getattr(args, "preload_topics", None)
    if not preload_file:
        return 0

    if not os.path.exists(preload_file):
        _dbg(f"[preload] File not found: {preload_file}")
        return 0

    # Load topics (support both .txt and .json)
    if preload_file.endswith(".json"):
        with open(preload_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                topics = data.get("titles", data.get("topics", []))
            else:
                topics = data
    else:
        with open(preload_file, "r", encoding="utf-8") as f:
            topics = [line.strip() for line in f if line.strip()]

    if not topics:
        _dbg(f"[preload] No topics found in {preload_file}")
        return 0

    # Apply max-subjects limit if set
    if args.max_subjects and args.max_subjects > 0:
        topics = topics[:args.max_subjects]

    inserted = 0
    skipped_seen = 0
    skipped_exists = 0

    for topic in topics:
        ck = canon_key_from_queue(topic)
        if ck and ck in seen_canon_keys:
            skipped_seen += 1
            continue

        subj, hop, outcome = queue.enqueue(topic, hop=0)

        if outcome == "inserted":
            inserted += 1
            _append_jsonl(paths["queue_jsonl"], {
                "subject": subj,
                "hop": 0,
                "event": "preloaded",
                "source": preload_file,
            })
            if ck:
                seen_canon_keys.add(ck)
        else:
            skipped_exists += 1

    _dbg(f"[preload] Loaded {inserted} topics from {preload_file}")
    if skipped_seen > 0:
        _dbg(f"[preload] Skipped {skipped_seen} (already seen)")
    if skipped_exists > 0:
        _dbg(f"[preload] Skipped {skipped_exists} (already in queue)")

    # ── KEY: set max_subjects to match exactly what we loaded ──
    if getattr(args, "preload_only", False):
        if not args.max_subjects or args.max_subjects <= 0:
            args.max_subjects = inserted
            _dbg(f"[preload-only] Set max_subjects={inserted} (exact topic count)")

    return inserted
# ---------------- article messages ----------------

def _build_llmpedia_messages_for_subject(
    subject: str,
    hop: int,
    args,
    root_topic: str,
    persona_block: str,
    self_rag_context: Optional[dict],
    outline: Optional[str] = None,
) -> List[dict]:
    """
    Build elicitation messages for a subject with optional outline and Self-RAG.
    Accepts the same args/logic as the sqlite version.
    """
    messages = build_elicitation_messages_for_subject(
        domain=args.domain,
        strategy=args.elicitation_strategy,
        subject_name=subject,
        seed=args.seed,
        root_topic=root_topic,
        min_sections=args.article_min_sections,
        max_sections=args.article_max_sections,
        avg_words_per_article=args.article_avg_words,
        persona_block=persona_block,
        outline=outline,
    )

    if self_rag_context and (self_rag_context.get("summary") or self_rag_context.get("salient_facts")):
        sr_block = _build_self_rag_block(subject, self_rag_context)
        messages = _append_block_to_msgs(messages, sr_block, target=args.self_rag_target)

    if args.footer_mode:
        if args.domain == "topic":
            footer = (
                "Additional, very important guidance about categories for the topic-centered LLMPedia "
                f"rooted at a fixed root topic:\n- Treat {subject} as the CURRENT ENTITY and the root topic "
                "as the broader envelope.\n- If the entity is globally or historically famous, then for the "
                "categories aim for about 50 distinct, precise categories that are tightly connected to this "
                "entity; if not famous, aim for around 10 strong categories, and if none are clear, return "
                "no categories.\n- Include categories that capture closely related organizations, events, "
                "places, works, technologies and concepts that are strongly associated with this entity AND "
                "relevant to the root topic.\n- Do NOT invent random or obviously speculative categories."
            )
        else:
            footer = (
                f"Additional guidance about categories for this LLMPedia article on {subject}:\n"
                "- If the entity is widely known, aim for about 50 distinct, precise categories.\n"
                "- If the entity is not widely known, aim for about 10 strong categories.\n"
                "- Include categories that capture closely related organizations, events, places, works, "
                "technologies and concepts that are strongly associated with this entity.\n"
                "- Do NOT invent random or obviously speculative categories."
            )
        messages = _append_footer_to_msgs(messages, footer, target=args.footer_location)

    return messages




#_____________ helpers for extracting the first paragraph
_INFBOX_START_RX = re.compile(r"^\s*\{\{\s*infobox\b", re.IGNORECASE)
_HEADING_RX = re.compile(r"^\s*={2,}\s*.*?\s*={2,}\s*$", re.UNICODE)
_REF_TAG_RX = re.compile(r"<ref\b[^>]*>.*?</ref\s*>", re.IGNORECASE | re.DOTALL)
_REF_SELF_RX = re.compile(r"<ref\b[^/]*/\s*>", re.IGNORECASE)
_HTML_COMMENT_RX = re.compile(r"<!--.*?-->", re.DOTALL)

# [[A|B]] -> B, [[A]] -> A
_WIKILINK_RX = re.compile(r"\[\[([^|\]]+)(?:\|([^\]]+))?\]\]")
# [http://url Label] -> Label
_EXTLINK_RX = re.compile(r"\[(https?://[^\s\]]+)\s+([^\]]+)\]")


def _strip_leading_infobox(wikitext: str) -> str:
    """
    Remove a leading {{Infobox ...}} template if it appears at the top.
    Uses brace-depth parsing so nested templates don't break it.
    """
    if not isinstance(wikitext, str) or not wikitext.strip():
        return wikitext or ""

    t = wikitext.lstrip()
    if not _INFBOX_START_RX.match(t):
        return wikitext

    # find the first "{{" start in the stripped text, then remove until matching "}}"
    start = t.find("{{")
    if start < 0:
        return wikitext

    depth = 0
    i = start
    n = len(t)
    while i < n - 1:
        two = t[i : i + 2]
        if two == "{{":
            depth += 1
            i += 2
            continue
        if two == "}}":
            depth -= 1
            i += 2
            if depth <= 0:
                # remove template block
                remainder = t[i:].lstrip()
                # preserve original leading whitespace from wikitext by just returning remainder
                return remainder
            continue
        i += 1

    # if we fail to match cleanly, just return original
    return wikitext


def _wikitext_to_plain_text(s: str) -> str:
    if not isinstance(s, str):
        return ""

    t = s
    t = _HTML_COMMENT_RX.sub(" ", t)
    t = _REF_TAG_RX.sub(" ", t)
    t = _REF_SELF_RX.sub(" ", t)

    # links
    def _wl(m):
        a = (m.group(1) or "").strip()
        b = (m.group(2) or "").strip()
        return b if b else a

    t = _WIKILINK_RX.sub(_wl, t)
    t = _EXTLINK_RX.sub(lambda m: (m.group(2) or "").strip(), t)

    # bold/italic markup
    t = t.replace("'''", "").replace("''", "")

    # collapse whitespace
    t = re.sub(r"\s+", " ", t, flags=re.UNICODE).strip()
    return t


# Replace the existing _extract_intro_excerpt_words function with this:

def _extract_intro_excerpt_words(wikitext: str, *, max_words: int = 100) -> str:
    """
    Extract the first paragraph introduction for similarity context.
    
    IMPORTANT: Removes ALL {{...}} templates (including Infobox) FIRST,
    then extracts the first max_words of plain text.
    
    Example:
      Input:  '''Vannevar Bush'''\n{{Infobox Person...}}\nVannevar Bush was a prominent...
      Output: Vannevar Bush Vannevar Bush was a prominent American engineer inventor...
    """
    if not isinstance(wikitext, str) or not wikitext.strip():
        return ""
    
    t = wikitext.strip()
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: Remove ALL {{...}} templates (Infobox, etc.)
    # Uses brace-depth matching to handle nested templates
    # ═══════════════════════════════════════════════════════════════════
    cleaned_chars = []
    i = 0
    n = len(t)
    
    while i < n:
        # Check for template start: {{
        if i < n - 1 and t[i:i+2] == "{{":
            # Found template start - skip until matching }}
            depth = 1
            i += 2  # Move past {{
            
            while i < n and depth > 0:
                if i < n - 1 and t[i:i+2] == "{{":
                    depth += 1
                    i += 2
                elif i < n - 1 and t[i:i+2] == "}}":
                    depth -= 1
                    i += 2
                else:
                    i += 1
            # Template fully skipped
        else:
            cleaned_chars.append(t[i])
            i += 1
    
    t = "".join(cleaned_chars)
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: Remove bold title markers '''...'''
    # ═══════════════════════════════════════════════════════════════════
    t = t.replace("'''", "").replace("''", "")
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 3: Remove other wikitext markup (links, refs, comments, etc.)
    # ═══════════════════════════════════════════════════════════════════
    # HTML comments
    t = re.sub(r'<!--.*?-->', ' ', t, flags=re.DOTALL)
    
    # <ref>...</ref> and <ref ... />
    t = re.sub(r'<ref\b[^>]*>.*?</ref\s*>', ' ', t, flags=re.IGNORECASE | re.DOTALL)
    t = re.sub(r'<ref\b[^/]*/\s*>', ' ', t, flags=re.IGNORECASE)
    
    # [[Link|Display]] -> Display, [[Link]] -> Link
    t = re.sub(r'\[\[([^|\]]+)\|([^\]]+)\]\]', r'\2', t)
    t = re.sub(r'\[\[([^\]]+)\]\]', r'\1', t)
    
    # [http://url Label] -> Label
    t = re.sub(r'\[https?://[^\s\]]+\s+([^\]]+)\]', r'\1', t)
    
    # Remove [[Category:...]]
    t = re.sub(r'\[\[Category:[^\]]*\]\]', ' ', t, flags=re.IGNORECASE)
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 4: Collapse whitespace
    # ═══════════════════════════════════════════════════════════════════
    t = re.sub(r'\s+', ' ', t, flags=re.UNICODE).strip()
    
    # ═══════════════════════════════════════════════════════════════════
    # STEP 5: Truncate to max_words
    # ═══════════════════════════════════════════════════════════════════
    if not t:
        return ""
    
    words = t.split()
    if max_words > 0 and len(words) > max_words:
        words = words[:max_words]
    
    return " ".join(words).strip()

# ---------------- post-article processing (JSON/JSONL only) ----------------
# _lookup_parent_from_queue_state removed (dead code — all callers now read parent from queue records directly)
# ---------------- outlines (two-stage) ----------------

def _build_outline_messages_for_subject(
    subject: str,
    root_topic: str,
    persona_block: str,
    args,
) -> List[dict]:
    """
    Build outline prompts that condition on the domain:
      - topic: explicit reference to the fixed root topic
      - general: NO reference to the root topic name
    """

    persona_text = (persona_block or "").strip()
    persona_prefix = (persona_text + "\n\n") if persona_text else ""
    min_sections = getattr(args, "article_min_sections", 3) or 3
    max_sections = getattr(args, "article_max_sections", 6) or max(min_sections, 6)

    # --------------------------
    # DOMAIN-AWARE PROMPTING
    # --------------------------

    if args.domain == "topic":
        # TOPIC-FOCUSED OUTLINE
        system_msg = (
            f"{persona_prefix}"
            "You are LLMPedia, helping design the section outline for a concise Wikipedia-style article in Wikitext.\n\n"
            f"The article will be about '{subject}', located inside a knowledge base centered on the root topic '{root_topic}'.\n\n"
            "INSTRUCTIONS\n"
            f"- Propose between {min_sections} and {max_sections+1} top-level sections.\n"
            "- Each must be a level-2 heading (== Heading ==).\n"
            "- Output ONLY the section titles (one per line).\n"
            "- DO NOT include any boilerplate sections such as: References, See also, Further reading, External links, Notes, Bibliography.\n"
            "- Consider how this subject connects to the fixed root topic.\n"
            "- Minimize fluff; maximize factual relevance.\n"
        )


        user_msg = (
            "We will now build the outline for an article.\n"
            f"Subject: {subject}\n"
            f"Root topic: {root_topic}\n\n"
            "Please provide section titles ONLY, one per line."
        )

    else:
        # GENERAL KNOWLEDGE MODE
        system_msg = (
            f"{persona_prefix}"
            "You are LLMPedia, helping design the section outline for a concise standalone article written in Wikitext.\n\n"
            f"The article will be about '{subject}'.\n\n"
            "INSTRUCTIONS\n"
            f"- Propose between {min_sections} and {max_sections+1} top-level sections.\n"
            "- Each must be a level-2 heading (== Heading ==).\n"
            "- Output ONLY section titles, no descriptions or bullets.\n"
            "- DO NOT include any boilerplate sections such as: References, See also, Further reading, External links, Notes, Bibliography.\n"
            "- Focus on structure typical in Wikipedia pages.\n"
            "- Minimize fluff; maximize factual relevance.\n"
        )

        user_msg = (
            "Please provide section titles ONLY.\n"
            f"Subject: {subject}\n\n"
            "Return exactly one section title per line."
        )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def _get_outline_for_subject(
    subject: str,
    hop: int,
    args,
    root_topic: str,
    persona_block: str,
    el_llm,
) -> str:
    msgs = _build_outline_messages_for_subject(subject, root_topic, persona_block, args)
    try:
        resp = el_llm(msgs, timeout=args.timeout)
    except TypeError:
        resp = el_llm(msgs)
    outline_text = _unwrap_text(resp).strip()
    if args.debug:
        _dbg(f"\n--- OUTLINE for [{subject}] (hop={hop}) ---\n{outline_text[:1200]}\n--- END OUTLINE ---\n")
    return outline_text


def _run_outline_batch_for_wave(
    args,
    paths,
    el_cfg,
    batch: List[Tuple[str, int]],
    wave_idx: int,
    client: OpenAI,
) -> Dict[Tuple[str, int], str]:
    if not batch or not getattr(args, "two_stage_elicit", True):
        return {}

    outlines: Dict[Tuple[str, int], str] = {}
    batches_dir = paths["batches_dir"]
    os.makedirs(batches_dir, exist_ok=True)

    outline_endpoint = _endpoint_for_cfg(el_cfg)
    input_path = os.path.join(batches_dir, f"outline_input_wave{wave_idx}.jsonl")

    with open(input_path, "w", encoding="utf-8") as f:
        for subject, hop in batch:
            root_topic = args.seed if args.domain == "topic" else subject
            messages = _build_outline_messages_for_subject(
                subject=subject,
                root_topic=root_topic,
                persona_block=args.persona_elicit_block,
                args=args,
            )

            body = _build_openai_body_for_batch(el_cfg, messages, max_tokens=1024)
            custom_id = f"outline::{subject}::hop={hop}"
            req_obj = {"custom_id": custom_id, "method": "POST", "url": outline_endpoint, "body": body}
            f.write(json.dumps(req_obj, ensure_ascii=False) + "\n")

    try:
        with open(input_path, "rb") as fh:
            input_file = client.files.create(file=fh, purpose="batch")

        job = client.batches.create(
            input_file_id=input_file.id,
            endpoint=outline_endpoint,
            completion_window="24h",
            metadata={"description": f"LLMPedia outline wave {wave_idx} seed={args.seed}"},
        )

        _dbg(
            f"[outline-batch] wave {wave_idx} created batch id={job.id}, input_file_id={input_file.id}, endpoint={outline_endpoint}"
        )

        poll_interval = args.batch_poll_interval
        while True:
            job = client.batches.retrieve(job.id)
            pass  # silenced
            if job.status == "completed":
                break
            if job.status in {"failed", "expired", "cancelled"}:
                _dbg(
                    f"[outline-batch] wave {wave_idx} batch {job.id} ended with status={job.status}; continuing without outlines."
                )
                return {}
            time.sleep(poll_interval)

        if not getattr(job, "output_file_id", None):
            err_id = getattr(job, "error_file_id", None)
            if err_id:
                try:
                    err_bytes = client.files.content(err_id).content
                    err_path = os.path.join(batches_dir, f"outline_errors_wave{wave_idx}_{job.id}.jsonl")
                    with open(err_path, "wb") as ef:
                        ef.write(err_bytes)
                    _dbg(
                        f"[outline-batch] wave {wave_idx} completed with ONLY errors; error_file_id={err_id}; saved to {err_path}"
                    )
                except Exception as e:
                    pass  # silenced
            return {}

        out_bytes = client.files.content(job.output_file_id).content
        out_path = os.path.join(batches_dir, f"outline_output_wave{wave_idx}_{job.id}.jsonl")
        with open(out_path, "wb") as f:
            f.write(out_bytes)

        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue

                custom_id = row.get("custom_id")
                if not isinstance(custom_id, str) or not custom_id.startswith("outline::"):
                    continue

                resp = row.get("response") or {}
                body = resp.get("body") or {}

                outline_text = ""
                if getattr(el_cfg, "use_responses_api", False):
                    output_items = body.get("output") or []
                    if not output_items:
                        continue
                    message_item = None
                    for item in output_items:
                        if isinstance(item, dict) and item.get("type") == "message":
                            message_item = item
                            break
                    if message_item is None:
                        for item in output_items:
                            if isinstance(item, dict) and item.get("content"):
                                message_item = item
                                break
                    if message_item is None:
                        continue
                    content = message_item.get("content") or []
                    chunks = []
                    for c in content:
                        if isinstance(c, dict) and "text" in c:
                            chunks.append(str(c["text"]))
                    outline_text = "".join(chunks).strip()
                else:
                    choices = body.get("choices") or []
                    if not choices:
                        continue
                    msg = (choices[0] or {}).get("message") or {}
                    outline_text = (msg.get("content") or "").strip()

                if not outline_text:
                    continue

                try:
                    _, rest = custom_id.split("outline::", 1)
                    subj_part, hop_part = rest.rsplit("::hop=", 1)
                    subj = subj_part
                    h = int(hop_part)
                except Exception:
                    subj = custom_id
                    h = 0

                outlines[(subj, h)] = outline_text
                _append_jsonl(paths["outlines_jsonl"], {"subject": subj, "hop": h, "outline": outline_text})

        return outlines

    except Exception as e:
        _append_error_log(paths, f"[outline-batch-wave={wave_idx}] error={repr(e)}", exc=e)
        pass  # silenced
        return {}


###
# ---- NER fallback / structured outputs   ---------------------------------

def _enable_ner_structured_output(cfg, *, name: str = "ner_phrases", schema: dict):
    """
    Force structured JSON output for NER.

    - Responses API: cfg.extra_inputs["text"]["format"] = {type:"json_schema", name, strict, schema}
    - ChatCompletions: cfg.extra_inputs["response_format"] = {type:"json_schema", json_schema:{name, strict, schema}}

    We MERGE into existing cfg.extra_inputs and do NOT overwrite existing format/response_format.
    """
    if not isinstance(schema, dict):
        raise ValueError("schema must be a dict (NER_SCHEMA_BASE or NER_SCHEMA_CAL)")

    cfg.extra_inputs = getattr(cfg, "extra_inputs", None) or {}

    if getattr(cfg, "use_responses_api", False):
        cfg.extra_inputs.setdefault("text", {})
        cfg.extra_inputs["text"].setdefault(
            "format",
            {
                "type": "json_schema",
                "name": name,
                "strict": True,
                "schema": schema,
            },
        )
    else:
        cfg.extra_inputs.setdefault(
            "response_format",
            {
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "strict": True,
                    "schema": schema,
                },
            },
        )

# Salvage patterns (only used if JSON parsing fails)
_RX_JSONISH = re.compile(
    r'"phrase"\s*:\s*"([^"]+)"(?:(?:.|\n)*?)"is_ne"\s*:\s*(true|false)'
    r'(?:(?:.|\n)*?)?(?:"confidence"\s*:\s*([01](?:\.\d+)?))?',
    re.IGNORECASE,
)

_RX_SIMPLE = re.compile(
    r'^\s*[-*\u2022]?\s*(?P<phrase>.+?)\s*(?:[:=\-]>?|\t)\s*(?P<isne>true|false)\s*$',
    re.IGNORECASE | re.MULTILINE,
)


def _coerce_jsonable(obj, *, max_depth: int = 8, max_items: int = 200):
    """
    Convert unknown SDK / provider response objects into plain JSON-able
    dict/list/str/number/bool/None recursively.

    - Handles objects with model_dump(), dict(), __dict__
    - Avoids infinite recursion via seen set
    - Caps depth + item count to prevent huge logs / crashes
    """
    seen = set()

    def _dump_obj(o):
        # already JSON-ish
        if o is None or isinstance(o, (str, int, float, bool)):
            return o
        if isinstance(o, bytes):
            # don't blow up logs; keep small preview
            return o[:200].decode("utf-8", errors="replace")

        oid = id(o)
        if oid in seen:
            return None
        seen.add(oid)

        # dict
        if isinstance(o, dict):
            out = {}
            n = 0
            for k, v in o.items():
                if n >= max_items:
                    break
                try:
                    ks = str(k)
                except Exception:
                    ks = repr(k)
                out[ks] = _dump_obj(v)
                n += 1
            return out

        # list/tuple
        if isinstance(o, (list, tuple)):
            out = []
            for i, v in enumerate(o[:max_items]):
                out.append(_dump_obj(v))
            return out

        # OpenAI / pydantic style
        md = getattr(o, "model_dump", None)
        if callable(md):
            try:
                return _dump_obj(md())
            except Exception:
                pass

        dct = getattr(o, "dict", None)
        if callable(dct):
            try:
                return _dump_obj(dct())
            except Exception:
                pass

        # generic python object
        od = getattr(o, "__dict__", None)
        if isinstance(od, dict) and od:
            return _dump_obj(od)

        # last resort: string repr
        try:
            return repr(o)
        except Exception:
            return "<unrepr-able object>"

    def _with_depth(o, depth):
        if depth > max_depth:
            return None
        if isinstance(o, dict):
            out = {}
            n = 0
            for k, v in o.items():
                if n >= max_items:
                    break
                out[k] = _with_depth(v, depth + 1)
                n += 1
            return out
        if isinstance(o, list):
            return [_with_depth(v, depth + 1) for v in o[:max_items]]
        # primitive / object
        return _dump_obj(o)

    return _with_depth(obj, 0)

def _parse_ner_output(raw) -> Tuple[List[dict], str]:
    """
    Parse NER output robustly.

    Works when:
    - provider returns dict already containing phrases
    - provider returns JSON text
    - provider wraps JSON in extra text
    - provider returns nested objects (handled via _coerce_jsonable)
    """
    def _norm_decision(d: dict):
        if not isinstance(d, dict):
            return None
        phrase = d.get("phrase")
        if not isinstance(phrase, str) or not phrase.strip():
            return None
        is_ne = bool(d.get("is_ne"))
        conf = d.get("confidence", None)
        conf_val = None
        if isinstance(conf, (int, float)):
            try:
                conf_val = float(conf)
            except Exception:
                conf_val = None
        return {"phrase": phrase.strip(), "is_ne": is_ne, "confidence": conf_val}

    plain = _coerce_jsonable(raw)

    # 1) dict direct
    if isinstance(plain, dict) and isinstance(plain.get("phrases"), list):
        out = []
        for d in plain["phrases"]:
            nd = _norm_decision(d)
            if nd:
                out.append(nd)
        if out:
            return out, "dict"

    # 2) dict nested search
    stack = [plain]
    seen = set()
    while stack:
        node = stack.pop()
        nid = id(node)
        if nid in seen:
            continue
        seen.add(nid)

        if isinstance(node, dict):
            if isinstance(node.get("phrases"), list):
                out = []
                for d in node["phrases"]:
                    nd = _norm_decision(d)
                    if nd:
                        out.append(nd)
                if out:
                    return out, "dict_nested"
            for v in node.values():
                if isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(node, list):
            for v in node:
                if isinstance(v, (dict, list)):
                    stack.append(v)

    # 3) parse from text
    txt = _unwrap_text(plain)
    if not isinstance(txt, str):
        return [], "empty"
    txt = txt.strip()
    if not txt:
        return [], "empty"

    # strip code fences
    if txt.startswith("```"):
        txt = re.sub(r"^```(?:json)?\s*", "", txt.strip(), flags=re.IGNORECASE)
        txt = re.sub(r"\s*```$", "", txt.strip())
        txt = txt.strip()

    # full json
    try:
        obj = json.loads(txt)
    except Exception:
        obj = None

    if isinstance(obj, dict) and isinstance(obj.get("phrases"), list):
        out = []
        for d in obj["phrases"]:
            nd = _norm_decision(d)
            if nd:
                out.append(nd)
        return out, "json"

    # substring json
    if "{" in txt and "}" in txt:
        sub = txt[txt.find("{") : txt.rfind("}") + 1].strip()
        try:
            obj2 = json.loads(sub)
        except Exception:
            obj2 = None
        if isinstance(obj2, dict) and isinstance(obj2.get("phrases"), list):
            out = []
            for d in obj2["phrases"]:
                nd = _norm_decision(d)
                if nd:
                    out.append(nd)
            return out, "json_substring"

    # regex salvage (your existing patterns)
    out2 = []
    for m in _RX_JSONISH.finditer(txt):
        phrase = (m.group(1) or "").strip()
        if not phrase:
            continue
        is_ne = (m.group(2) or "").strip().lower() == "true"
        conf = None
        if m.group(3) is not None:
            try:
                conf = float(m.group(3))
            except Exception:
                conf = None
        out2.append({"phrase": phrase, "is_ne": is_ne, "confidence": conf})
    if out2:
        return out2, "regex_jsonish"

    out3 = []
    for m in _RX_SIMPLE.finditer(txt):
        phrase = (m.group("phrase") or "").strip()
        if not phrase:
            continue
        is_ne = (m.group("isne") or "").strip().lower() == "true"
        out3.append({"phrase": phrase, "is_ne": is_ne, "confidence": None})
    if out3:
        return out3, "regex_simple"

    return [], "empty"




# ---------------- online API response saving (cost tracking) ----------------

def _extract_online_usage(coerced):
    """Extract usage dict from a coerced (JSON-able) response object."""
    if not isinstance(coerced, dict):
        return {}
    # Strategy 1: top-level _usage key (from patched openai_client)
    u = coerced.get("_usage") or coerced.get("usage")
    if isinstance(u, dict):
        return {
            "input_tokens": int(u.get("input_tokens") or u.get("prompt_tokens") or 0),
            "output_tokens": int(u.get("output_tokens") or u.get("completion_tokens") or 0),
            "total_tokens": int(u.get("total_tokens") or 0),
        }
    return {}


def _extract_usage_from_resp(resp):
    """Extract usage directly from an OpenAI response object (no patch needed)."""
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {}
    return {
        "input_tokens": int(getattr(usage, "input_tokens", 0) or getattr(usage, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(usage, "output_tokens", 0) or getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def _save_online_api_response(
    paths: dict,
    stage: str,
    subject: str,
    hop: int,
    resp,
    *,
    provider: str = "openai",
    text_preview: str = "",
    chunk_idx: int = 0,
    chunk_start: int = 0,
    chunk_size: int = 0,
):
    """
    Save a single online API response to online_api_responses/<stage>/.
    Always writes the file, even if usage extraction fails (zeros fallback).
    """
    dir_key = f"online_api_responses_{stage}_dir"
    if dir_key not in paths:
        return

    out_dir = paths[dir_key]
    os.makedirs(out_dir, exist_ok=True)

    # Try to extract usage info
    coerced = _coerce_jsonable(resp)
    usage_info = _extract_online_usage(coerced)

    # Fallback: try direct from response object
    if not usage_info or not usage_info.get("input_tokens"):
        usage_info = _extract_usage_from_resp(resp)

    # Always build a usage dict (zeros if extraction failed)
    usage_dict = {
        "input_tokens": int(usage_info.get("input_tokens", 0) or 0),
        "output_tokens": int(usage_info.get("output_tokens", 0) or 0),
        "total_tokens": int(usage_info.get("total_tokens", 0) or 0),
    }

    ts = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    record = {
        "stage": stage,
        "subject": subject,
        "hop": hop,
        "provider": provider,
        "timestamp": ts,
        "usage": usage_dict,
        "text_preview": (text_preview or "")[:500],
        "chunk_idx": chunk_idx,
        "chunk_start": chunk_start,
        "chunk_size": chunk_size,
    }

    # Build filename: stage_subject_hop_chunk_timestamp.json
    safe_subj = re.sub(r"[^\w.-]", "_", subject)[:60]
    fname = f"{stage}_{safe_subj}_h{hop}_c{chunk_idx}_{ts.replace(':', '').replace('-', '')}.json"
    fpath = os.path.join(out_dir, fname)

    try:
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # never let logging break the pipeline


def _run_ner_strict_gate(
    *,
    args,
    paths,
    subject: str,
    hop: int,
    candidate_phrases: List[str],
    ner_llm,
) -> List[str]:
    """
    STRICT NER gate using ONLY build_ner_messages_for_phrases().

    Confidence semantics:
      - If strategy contains 'calib': enforce ner_conf_threshold (if > 0).
      - Otherwise: IGNORE confidence entirely.
    """
    CHUNK_SIZE = int(getattr(args, "ner_chunk_size", 25) or 25)
    conf_th = float(getattr(args, "ner_conf_threshold", 0.0) or 0.0)

    ner_strategy = str(getattr(args, "ner_strategy", "") or "").strip().lower()
    use_confidence = ("calib" in ner_strategy)

    root_subject = args.seed if args.domain == "topic" else None
    root_topic = args.seed if args.domain == "topic" else subject

    if not candidate_phrases:
        _append_jsonl(
            paths["ner_lowconf_jsonl"],
            {
                "stage": "ner_filter",
                "current_entity": subject,
                "root_subject": root_subject,
                "hop": hop,
                "rejection_reason": "no_candidates",
                "use_confidence": bool(use_confidence),
            },
        )
        return []

    # stable unique candidates by canonical key
    canon_to_phrase: Dict[str, str] = {}
    ordered_canons: List[str] = []
    for p in candidate_phrases:
        s = (p or "").strip()
        if not s:
            continue
        ck = canon_key_from_queue(s)
        if not ck:
            continue
        if ck not in canon_to_phrase:
            canon_to_phrase[ck] = s
            ordered_canons.append(ck)

    candidates = [canon_to_phrase[ck] for ck in ordered_canons]
    if not candidates:
        return []

    model_by_canon: Dict[str, dict] = {}
    parse_modes: List[str] = []
    total_model_decisions = 0

    for chunk_idx, start in enumerate(range(0, len(candidates), CHUNK_SIZE), 1):
        chunk = candidates[start : start + CHUNK_SIZE]
        if not chunk:
            continue

        ner_messages = build_ner_messages_for_phrases(
            domain=args.domain,
            strategy=args.ner_strategy,
            subject_name=subject,
            phrases=chunk,
            seed=args.seed,
            root_topic=root_topic,
            persona_block=args.persona_ner_block,
        )

        try:
            try:
                resp = ner_llm(ner_messages, timeout=args.timeout)
            except TypeError:
                resp = ner_llm(ner_messages)
            txt = _unwrap_text(resp).strip()

            # Save per-subject online API response (cost tracking)
            try:
                _save_online_api_response(
                    paths, "ner", subject, hop, resp,
                    provider=(getattr(args, "_ner_provider", "") or "openai"),
                    text_preview=(txt[:500] if isinstance(txt, str) else ""),
                    chunk_idx=chunk_idx,
                    chunk_start=start,
                    chunk_size=len(chunk),
                )
            except Exception:
                pass  # never let logging break NER

        except Exception as e:
            _append_jsonl(
                paths["ner_decisions_jsonl"],
                {
                    "stage": "ner_run_summary",
                    "current_entity": subject,
                    "root_subject": root_subject,
                    "hop": hop,
                    "chunk_idx": chunk_idx,
                    "chunk_start": start,
                    "chunk_size": len(chunk),
                    "parse_mode": "exception",
                    "num_candidates": len(chunk),
                    "num_model_decisions": 0,
                    "raw_preview": "",
                    "error": f"{type(e).__name__}: {e!r}",
                    "use_confidence": bool(use_confidence),
                },
            )
            parse_modes.append("exception")
            continue

        decisions, parse_mode = _parse_ner_output(txt)
        parse_modes.append(parse_mode)
        total_model_decisions += len(decisions)

        _append_jsonl(
            paths["ner_decisions_jsonl"],
            {
                "stage": "ner_run_summary",
                "current_entity": subject,
                "root_subject": root_subject,
                "hop": hop,
                "chunk_idx": chunk_idx,
                "chunk_start": start,
                "chunk_size": len(chunk),
                "parse_mode": parse_mode,
                "num_candidates": len(chunk),
                "num_model_decisions": len(decisions),
                "raw_preview": txt[:800],
                "resp_type": str(type(resp)),
                "use_confidence": bool(use_confidence),
            },
        )

        for d in decisions:
            rp = (d.get("phrase") or "").strip()
            if not rp:
                continue
            ck = canon_key_from_queue(rp)
            if not ck:
                continue
            if ck not in model_by_canon:
                model_by_canon[ck] = {
                    "raw_phrase": rp,
                    "is_ne": bool(d.get("is_ne")),
                    "confidence": (float(d["confidence"]) if isinstance(d.get("confidence"), (int, float)) else None),
                    "parse_mode": parse_mode,
                }

    _append_jsonl(
        paths["ner_decisions_jsonl"],
        {
            "stage": "ner_run_summary_global",
            "current_entity": subject,
            "root_subject": root_subject,
            "hop": hop,
            "chunk_size": CHUNK_SIZE,
            "parse_modes": parse_modes[:200],
            "num_candidates": len(candidates),
            "num_model_decisions": total_model_decisions,
            "use_confidence": bool(use_confidence),
            "ner_conf_threshold": conf_th,
        },
    )

    kept: List[str] = []
    kept_canons: Set[str] = set()

    for ck in ordered_canons:
        phrase = canon_to_phrase[ck]
        md = model_by_canon.get(ck)

        if md is None:
            is_ne = False
            conf_val = None
            raw_phrase = None
            source = "missing"
            passed = False
            reason = "ner_parse_failed_or_empty"
            pm = "empty"
        else:
            is_ne = bool(md["is_ne"])
            conf_val = md.get("confidence", None)
            raw_phrase = md.get("raw_phrase")
            source = "model"
            pm = md.get("parse_mode", "unknown")

            # ✅ correct confidence semantics
            if not use_confidence:
                passed = bool(is_ne)
                reason = "accepted" if passed else "not_named_entity"
            else:
                if conf_th > 0.0:
                    passed = bool(is_ne) and isinstance(conf_val, (int, float)) and float(conf_val) >= conf_th
                    if passed:
                        reason = "accepted"
                    elif not is_ne:
                        reason = "not_named_entity"
                    else:
                        reason = "below_conf_threshold_or_missing_confidence"
                else:
                    passed = bool(is_ne)
                    reason = "accepted" if passed else "not_named_entity"

        _append_jsonl(
            paths["ner_decisions_jsonl"],
            {
                "current_entity": subject,
                "root_subject": root_subject,
                "hop": hop,
                "phrase": phrase,
                "is_ne": bool(is_ne),
                "confidence": conf_val,
                "ner_conf_threshold": conf_th,
                "passed_threshold": bool(passed),
                "decision_reason": reason,
                "source": source,
                "parse_mode": pm,
                "raw_phrase": raw_phrase,
                "use_confidence": bool(use_confidence),
            },
        )

        if not passed:
            _append_jsonl(
                paths["ner_lowconf_jsonl"],
                {
                    "stage": "ner_filter",
                    "current_entity": subject,
                    "root_subject": root_subject,
                    "hop": hop,
                    "phrase": phrase,
                    "is_ne": bool(is_ne),
                    "confidence": conf_val,
                    "ner_conf_threshold": conf_th,
                    "passed_threshold": False,
                    "rejection_reason": reason,
                    "source": source,
                    "parse_mode": pm,
                    "raw_phrase": raw_phrase,
                    "use_confidence": bool(use_confidence),
                },
            )
        else:
            if ck not in kept_canons:
                kept_canons.add(ck)
                kept.append(phrase)

    return kept


def _run_ner_wave_via_openai_batch(
    *,
    args,
    paths,
    ner_cfg,
    client: "OpenAI",
    wave_items: List[dict],   # {"subject","hop","candidate_phrases"}
    wave_idx: int,
) -> Dict[Tuple[str, int], List[str]]:
    """
    Batch NER using ONLY build_ner_messages_for_phrases().

    Confidence semantics (IMPORTANT):
      - If strategy contains 'calib': enforce ner_conf_threshold (if > 0).
      - Otherwise: IGNORE confidence entirely.

    Strict behavior:
      - Missing / unparsable decision => reject (strict gate).
      - Batch failure / missing output file => reject all candidates (strict gate).
    """
    CHUNK_SIZE = int(getattr(args, "ner_chunk_size", 25) or 25)
    conf_th = float(getattr(args, "ner_conf_threshold", 0.0) or 0.0)

    ner_strategy = str(getattr(args, "ner_strategy", "") or "").strip().lower()
    use_confidence = ("calib" in ner_strategy)  # ✅ only calibrate uses confidence

    root_subject_global = args.seed if args.domain == "topic" else None
    ner_endpoint = _endpoint_for_cfg(ner_cfg)
    batches_dir = paths["batches_dir"]
    os.makedirs(batches_dir, exist_ok=True)

    _CID_RX = re.compile(
        r"^ner::(?P<subj>.*)::hop=(?P<hop>\d+)::chunk=(?P<chunk>\d+)::start=(?P<start>\d+)::n=(?P<n>\d+)$"
    )

    def _parse_cid(cid: str) -> Optional[Tuple[str, int, int, int, int]]:
        m = _CID_RX.match(cid or "")
        if not m:
            return None
        return (
            m.group("subj"),
            int(m.group("hop")),
            int(m.group("chunk")),
            int(m.group("start")),
            int(m.group("n")),
        )

    def _save_file(file_id: Optional[str], filename: str) -> Optional[str]:
        if not file_id:
            return None
        try:
            b = client.files.content(file_id).content
            out_path = os.path.join(batches_dir, filename)
            with open(out_path, "wb") as f:
                f.write(b)
            return out_path
        except Exception as e:
            _append_error_log(paths, f"[ner-batch] failed to download file_id={file_id}: {e!r}", exc=e)
            return None

    # ---------------- build state + requests ----------------
    state: Dict[Tuple[str, int], dict] = {}
    req_rows: List[dict] = []

    for it in wave_items:
        subj = it["subject"]
        hop = it["hop"]
        root_topic = args.seed if args.domain == "topic" else subj
        cands = it.get("candidate_phrases") or []

        canon_to_phrase: Dict[str, str] = {}
        ordered_canons: List[str] = []
        for p in cands:
            s = (p or "").strip()
            if not s:
                continue
            ck = canon_key_from_queue(s)
            if not ck:
                continue
            if ck not in canon_to_phrase:
                canon_to_phrase[ck] = s
                ordered_canons.append(ck)

        candidates = [canon_to_phrase[ck] for ck in ordered_canons]

        state[(subj, hop)] = {
            "canon_to_phrase": canon_to_phrase,
            "ordered_canons": ordered_canons,
            "candidates": candidates,
            "model_by_canon": {},   # ck -> {is_ne, confidence, raw_phrase, parse_mode}
            "parse_modes": [],
            "num_model_decisions": 0,
        }

        for chunk_idx, start in enumerate(range(0, len(candidates), CHUNK_SIZE), 1):
            chunk = candidates[start : start + CHUNK_SIZE]
            if not chunk:
                continue

            messages = build_ner_messages_for_phrases(
                domain=args.domain,
                strategy=args.ner_strategy,
                subject_name=subj,
                phrases=chunk,
                seed=args.seed,
                root_topic=root_topic,
                persona_block=args.persona_ner_block,
            )

            body = _build_openai_body_for_batch(
                ner_cfg,
                messages,
                max_tokens=int(getattr(args, "ner_max_tokens", 2048) or 2048),
            )

            custom_id = f"ner::{subj}::hop={hop}::chunk={chunk_idx}::start={start}::n={len(chunk)}"
            req_rows.append({"custom_id": custom_id, "method": "POST", "url": ner_endpoint, "body": body})

    if not req_rows:
        return {(it["subject"], it["hop"]): [] for it in wave_items}

    def _emit_strict_missing_for_all(reason: str, parse_mode: str):
        for (subj, hop), st in state.items():
            for ck in st["ordered_canons"]:
                phrase = st["canon_to_phrase"][ck]
                _append_jsonl(
                    paths["ner_decisions_jsonl"],
                    {
                        "current_entity": subj,
                        "root_subject": root_subject_global,
                        "hop": hop,
                        "phrase": phrase,
                        "is_ne": False,
                        "confidence": None,
                        "ner_conf_threshold": conf_th,
                        "passed_threshold": False,
                        "decision_reason": reason,
                        "source": "missing",
                        "parse_mode": parse_mode,
                        "use_confidence": bool(use_confidence),
                    },
                )
                _append_jsonl(
                    paths["ner_lowconf_jsonl"],
                    {
                        "stage": "ner_filter",
                        "current_entity": subj,
                        "root_subject": root_subject_global,
                        "hop": hop,
                        "phrase": phrase,
                        "is_ne": False,
                        "confidence": None,
                        "ner_conf_threshold": conf_th,
                        "passed_threshold": False,
                        "rejection_reason": reason,
                        "source": "missing",
                        "parse_mode": parse_mode,
                        "use_confidence": bool(use_confidence),
                    },
                )

    ner_input_path = os.path.join(batches_dir, f"ner_input_wave{wave_idx}.jsonl")
    with open(ner_input_path, "w", encoding="utf-8") as f:
        for r in req_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---------------- run batch ----------------
    try:
        with open(ner_input_path, "rb") as fh:
            input_file = client.files.create(file=fh, purpose="batch")

        job = client.batches.create(
            input_file_id=input_file.id,
            endpoint=ner_endpoint,
            completion_window="24h",
            metadata={"description": f"LLMPedia NER wave {wave_idx} seed={args.seed}"},
        )
        pass  # silenced

        poll_interval = float(getattr(args, "batch_poll_interval", 30.0) or 30.0)
        _poll_failures = 0
        while True:
            try:
                job = client.batches.retrieve(job.id)
                _poll_failures = 0
            except Exception as _poll_exc:
                _poll_failures += 1
                _append_error_log(
                    paths,
                    f"[ner-batch] poll failure #{_poll_failures} for job {job.id}: {type(_poll_exc).__name__}: {_poll_exc!r}",
                    exc=_poll_exc,
                )
                if _poll_failures >= 10:
                    _dbg(f"[ner-batch] 10 consecutive poll failures; giving up on batch {job.id}")
                    _emit_strict_missing_for_all("ner_call_failed", "poll_failures_exceeded")
                    return {(it["subject"], it["hop"]): [] for it in wave_items}
                time.sleep(poll_interval * 2)
                continue
            if job.status == "completed":
                break
            if job.status in {"failed", "expired", "cancelled"}:
                _save_file(getattr(job, "error_file_id", None), f"ner_errors_wave{wave_idx}_{job.id}.jsonl")
                _emit_strict_missing_for_all("ner_call_failed", f"batch_status={job.status}")
                return {(it["subject"], it["hop"]): [] for it in wave_items}
            time.sleep(poll_interval)

        out_path = _save_file(getattr(job, "output_file_id", None), f"ner_output_wave{wave_idx}_{job.id}.jsonl")
        _save_file(getattr(job, "error_file_id", None), f"ner_errors_wave{wave_idx}_{job.id}.jsonl")

        if not out_path:
            _emit_strict_missing_for_all("ner_call_failed", "no_output_file")
            return {(it["subject"], it["hop"]): [] for it in wave_items}

    except Exception as e:
        _append_error_log(paths, f"[ner-batch] SDK exception: {type(e).__name__}: {e!r}", exc=e)
        _emit_strict_missing_for_all("ner_call_failed", "sdk_exception")
        return {(it["subject"], it["hop"]): [] for it in wave_items}

    # ---------------- parse output ----------------
    chunk_failures = 0
    chunk_success = 0

    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue

            cid = row.get("custom_id")
            if not isinstance(cid, str) or not cid.startswith("ner::"):
                continue

            parsed = _parse_cid(cid)
            if not parsed:
                continue
            subj, hop, chunk_idx, start, n = parsed

            st = state.get((subj, hop))
            if not st:
                continue

            resp = row.get("response") or {}
            status_code = resp.get("status_code", None)
            if status_code is not None and int(status_code) >= 400:
                chunk_failures += 1
                _append_jsonl(
                    paths["ner_decisions_jsonl"],
                    {
                        "stage": "ner_run_summary",
                        "current_entity": subj,
                        "root_subject": root_subject_global,
                        "hop": hop,
                        "chunk_idx": chunk_idx,
                        "chunk_start": start,
                        "chunk_size": n,
                        "parse_mode": "http_error",
                        "num_candidates": n,
                        "num_model_decisions": 0,
                        "raw_preview": "",
                        "status_code": int(status_code),
                        "use_confidence": bool(use_confidence),
                    },
                )
                continue

            body = resp.get("body") or {}
            txt = _extract_text_from_openai_batch_body(ner_cfg, body)

            decisions, parse_mode = _parse_ner_output(txt)
            st["parse_modes"].append(parse_mode)
            st["num_model_decisions"] += len(decisions)
            chunk_success += 1

            _append_jsonl(
                paths["ner_decisions_jsonl"],
                {
                    "stage": "ner_run_summary",
                    "current_entity": subj,
                    "root_subject": root_subject_global,
                    "hop": hop,
                    "chunk_idx": chunk_idx,
                    "chunk_start": start,
                    "chunk_size": n,
                    "parse_mode": parse_mode,
                    "num_candidates": n,
                    "num_model_decisions": len(decisions),
                    "raw_preview": (txt[:800] if isinstance(txt, str) else ""),
                    "resp_type": "openai_batch",
                    "use_confidence": bool(use_confidence),
                },
            )

            for d in decisions:
                rp = (d.get("phrase") or "").strip()
                if not rp:
                    continue
                ck = canon_key_from_queue(rp)
                if not ck:
                    continue
                if ck not in st["model_by_canon"]:
                    st["model_by_canon"][ck] = {
                        "raw_phrase": rp,
                        "is_ne": bool(d.get("is_ne")),
                        "confidence": (
                            float(d["confidence"]) if isinstance(d.get("confidence"), (int, float)) else None
                        ),
                        "parse_mode": parse_mode,
                    }

    # ---------------- strict per-candidate decisions + kept list ----------------
    kept_map: Dict[Tuple[str, int], List[str]] = {}

    for (subj, hop), st in state.items():
        _append_jsonl(
            paths["ner_decisions_jsonl"],
            {
                "stage": "ner_run_summary_global",
                "current_entity": subj,
                "root_subject": root_subject_global,
                "hop": hop,
                "chunk_size": CHUNK_SIZE,
                "parse_modes": st["parse_modes"][:200],
                "num_candidates": len(st["candidates"]),
                "num_model_decisions": st["num_model_decisions"],
                "chunk_success": chunk_success,
                "chunk_failures": chunk_failures,
                "use_confidence": bool(use_confidence),
                "ner_conf_threshold": conf_th,
            },
        )

        kept: List[str] = []
        kept_canons: Set[str] = set()

        for ck in st["ordered_canons"]:
            phrase = st["canon_to_phrase"][ck]
            md = st["model_by_canon"].get(ck)

            if md is None:
                is_ne = False
                conf_val = None
                raw_phrase = None
                source = "missing"
                passed = False
                reason = "ner_parse_failed_or_empty"
                pm = "empty"
            else:
                is_ne = bool(md["is_ne"])
                conf_val = md.get("confidence", None)
                raw_phrase = md.get("raw_phrase")
                source = "model"
                pm = md.get("parse_mode", "unknown")

                # ✅ correct confidence semantics
                if not use_confidence:
                    passed = bool(is_ne)
                    reason = "accepted" if passed else "not_named_entity"
                else:
                    if conf_th > 0.0:
                        passed = bool(is_ne) and isinstance(conf_val, (int, float)) and float(conf_val) >= conf_th
                        if passed:
                            reason = "accepted"
                        elif not is_ne:
                            reason = "not_named_entity"
                        else:
                            reason = "below_conf_threshold_or_missing_confidence"
                    else:
                        passed = bool(is_ne)
                        reason = "accepted" if passed else "not_named_entity"

            _append_jsonl(
                paths["ner_decisions_jsonl"],
                {
                    "current_entity": subj,
                    "root_subject": root_subject_global,
                    "hop": hop,
                    "phrase": phrase,
                    "is_ne": bool(is_ne),
                    "confidence": conf_val,
                    "ner_conf_threshold": conf_th,
                    "passed_threshold": bool(passed),
                    "decision_reason": reason,
                    "source": source,
                    "parse_mode": pm,
                    "raw_phrase": raw_phrase,
                    "use_confidence": bool(use_confidence),
                },
            )

            if not passed:
                _append_jsonl(
                    paths["ner_lowconf_jsonl"],
                    {
                        "stage": "ner_filter",
                        "current_entity": subj,
                        "root_subject": root_subject_global,
                        "hop": hop,
                        "phrase": phrase,
                        "is_ne": bool(is_ne),
                        "confidence": conf_val,
                        "ner_conf_threshold": conf_th,
                        "passed_threshold": False,
                        "rejection_reason": reason,
                        "source": source,
                        "parse_mode": pm,
                        "raw_phrase": raw_phrase,
                        "use_confidence": bool(use_confidence),
                    },
                )
            else:
                if ck not in kept_canons:
                    kept_canons.add(ck)
                    kept.append(phrase)

        kept_map[(subj, hop)] = kept

    for it in wave_items:
        kept_map.setdefault((it["subject"], it["hop"]), [])

    return kept_map

# ---- end NER fallback / structured outputs   ------------------------------


def _run_similarity_wave_via_openai_batch(
    *,
    args,
    paths,
    similarity_cfg,
    client: "OpenAI",
    wave_items: List[dict],  # {"candidate","parent_entity","parent_intro","similar_items","hop"}
    wave_idx: int,
) -> Dict[str, Tuple[bool, Optional[str]]]:
    """
    Batch similarity LLM decisions via OpenAI Batch API.

    IMPORTANT FIXES:
    - Uses unique per-request custom_id (no candidate hashing).
    - Safe even if the same candidate shows up multiple times (still best to dedupe upstream).
    - Strict fallback: on any failure / missing output => treat as duplicate (safe).
    """
    if not wave_items:
        return {}

    batches_dir = paths["batches_dir"]
    os.makedirs(batches_dir, exist_ok=True)

    similarity_endpoint = _endpoint_for_cfg(similarity_cfg)
    poll_interval = float(getattr(args, "batch_poll_interval", 30.0) or 30.0)

    def _clip(s: str, n: int) -> str:
        s = (s or "").strip()
        if not s or n <= 0:
            return ""
        return (s[:n] + "…") if len(s) > n else s

    def _save_file(file_id: Optional[str], filename: str) -> Optional[str]:
        if not file_id:
            return None
        try:
            b = client.files.content(file_id).content
            out_path = os.path.join(batches_dir, filename)
            with open(out_path, "wb") as f:
                f.write(b)
            return out_path
        except Exception as e:
            _append_error_log(paths, f"[similarity-batch] failed to download file_id={file_id}: {e!r}", exc=e)
            return None

    def _fallback_dup_of(it: Optional[dict]) -> Optional[str]:
        if not isinstance(it, dict):
            return None
        sims = it.get("similar_items") or []
        if isinstance(sims, list) and sims:
            ent0 = (sims[0] or {}).get("entity")
            if isinstance(ent0, str) and ent0.strip():
                return ent0.strip()
        return None

    def _build_similarity_messages(candidate: str, parent_entity: str, parent_intro: str, similar_items: List[dict]) -> List[dict]:
        sys = (
            "You are a strict entity deduplication assistant.\n"
            "Decide whether the CANDIDATE refers to the SAME entity as ANY existing item.\n"
            "Consider spelling/hyphen variants and singular/plural surface forms as potentially the same entity.\n"
            "Use the provided parent context snippets to disambiguate.\n"
            "Output ONLY JSON: {\"duplicate\":true/false, \"duplicate_of\":\"...\"|null}\n"
        )

        intro_chars_limit = int(getattr(args, "similarity_log_parent_intro_chars", 800) or 800)
        parent_intro_clip = _clip(parent_intro, intro_chars_limit)

        lines = []
        k = int(getattr(args, "similarity_top_k", 5) or 5)
        for it in (similar_items or [])[:k]:
            ent = (it.get("entity") or "").strip()
            try:
                score = float(it.get("score", 0.0) or 0.0)
            except Exception:
                score = 0.0
            ps = (it.get("parent_subject") or "").strip()
            pi = _clip(it.get("parent_intro_excerpt") or "", intro_chars_limit)

            if ps and pi:
                ctx = f'parent="{ps}" intro="{pi}"'
            elif ps:
                ctx = f'parent="{ps}"'
            else:
                ctx = "parent=(unknown)"

            if ent:
                lines.append(f'- entity="{ent}" score={score:.4f} {ctx}')

        sim_block = "\n".join(lines) if lines else "(none)"

        user = (
            f"CANDIDATE: {candidate}\n\n"
            f"PRODUCED_BY_PARENT: {parent_entity}\n"
            f"PARENT_INTRO_EXCERPT: {parent_intro_clip}\n\n"
            "MOST_SIMILAR_EXISTING_ITEMS:\n"
            f"{sim_block}\n"
        )
        return [{"role": "system", "content": sys}, {"role": "user", "content": user}]

    # -------------------------
    # Build batch requests
    # -------------------------
    req_rows: List[dict] = []
    cid_to_item: Dict[str, dict] = {}
    cid_to_candidate: Dict[str, str] = {}

    for i, it in enumerate(wave_items):
        candidate = (it.get("candidate") or "").strip()
        parent_entity = (it.get("parent_entity") or "").strip()
        parent_intro = it.get("parent_intro") or ""
        similar_items = it.get("similar_items") or []
        hop = int(it.get("hop", 0) or 0)

        if not candidate:
            continue

        cid = f"similarity::{wave_idx}:{i}::hop={hop}"
        cid_to_item[cid] = it
        cid_to_candidate[cid] = candidate

        messages = _build_similarity_messages(candidate, parent_entity, parent_intro, similar_items)
        body = _build_openai_body_for_batch(similarity_cfg, messages, max_tokens=256)
        req_rows.append({"custom_id": cid, "method": "POST", "url": similarity_endpoint, "body": body})

    if not req_rows:
        return {}

    similarity_input_path = os.path.join(batches_dir, f"similarity_input_wave{wave_idx}.jsonl")
    with open(similarity_input_path, "w", encoding="utf-8") as f:
        for r in req_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    results: Dict[str, Tuple[bool, Optional[str]]] = {}

    def _mark_all_as_duplicates(reason: str):
        for cid, it in cid_to_item.items():
            cand = cid_to_candidate.get(cid, "")
            dup_of = _fallback_dup_of(it)
            if cand:
                results[cand] = (True, dup_of)
            _append_jsonl(
                paths["similarity_decisions_jsonl"],
                {
                    "stage": "similarity_batch_fallback_all",
                    "candidate": cand,
                    "decision": "reject",
                    "reason": reason,
                    "duplicate_of": dup_of,
                    "hop": it.get("hop", 0) if it else None,
                    "current_entity": it.get("parent_entity", "") if it else None,
                    "parent_entity": it.get("parent_entity", "") if it else None,
                },
            )

    # -------------------------
    # Submit batch + poll
    # -------------------------
    try:
        with open(similarity_input_path, "rb") as fh:
            input_file = client.files.create(file=fh, purpose="batch")

        job = client.batches.create(
            input_file_id=input_file.id,
            endpoint=similarity_endpoint,
            completion_window="24h",
            metadata={"description": f"LLMPedia similarity wave {wave_idx} seed={args.seed}"},
        )
        pass  # silenced

        while True:
            job = client.batches.retrieve(job.id)
            pass  # silenced
            if job.status == "completed":
                break
            if job.status in {"failed", "expired", "cancelled"}:
                _save_file(getattr(job, "error_file_id", None), f"similarity_errors_wave{wave_idx}_{job.id}.jsonl")
                _mark_all_as_duplicates(f"batch_status={job.status}")
                return results
            time.sleep(poll_interval)

        out_path = _save_file(getattr(job, "output_file_id", None), f"similarity_output_wave{wave_idx}_{job.id}.jsonl")
        _save_file(getattr(job, "error_file_id", None), f"similarity_errors_wave{wave_idx}_{job.id}.jsonl")

        if not out_path:
            _mark_all_as_duplicates("no_output_file")
            return results

    except Exception as e:
        _append_error_log(paths, f"[similarity-batch] SDK exception: {type(e).__name__}: {e!r}", exc=e)
        _mark_all_as_duplicates("sdk_exception")
        return results

    # -------------------------
    # Parse output
    # -------------------------
    returned_cids: Set[str] = set()

    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue

            cid = row.get("custom_id")
            if not isinstance(cid, str) or not cid.startswith("similarity::"):
                continue
            if cid not in cid_to_candidate:
                continue

            returned_cids.add(cid)

            candidate = cid_to_candidate.get(cid, "")
            it = cid_to_item.get(cid)

            resp = row.get("response") or {}
            status_code = resp.get("status_code", None)

            # HTTP error => reject (safe)
            if status_code is not None:
                try:
                    sc = int(status_code)
                except Exception:
                    sc = None
                if sc is not None and sc >= 400:
                    dup_of = _fallback_dup_of(it)
                    results[candidate] = (True, dup_of)
                    _append_jsonl(
                        paths["similarity_decisions_jsonl"],
                        {
                            "stage": "similarity_batch_http_error",
                            "candidate": candidate,
                            "decision": "reject",
                            "reason": f"http_status={sc}",
                            "duplicate_of": dup_of,
                            "hop": it.get("hop", 0) if it else None,
                            "current_entity": it.get("parent_entity", "") if it else None,
                            "parent_entity": it.get("parent_entity", "") if it else None,
                        },
                    )
                    continue

            body = resp.get("body") or {}
            txt = _extract_text_from_openai_batch_body(similarity_cfg, body)

            is_duplicate = True  # strict default
            duplicate_of: Optional[str] = _fallback_dup_of(it)

            if isinstance(txt, str) and txt.strip():
                raw = txt.strip()
                if raw.startswith("```"):
                    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
                    raw = re.sub(r"\s*```$", "", raw)
                    raw = raw.strip()

                try:
                    obj = json.loads(raw)
                    is_duplicate = bool(obj.get("duplicate", False))
                    dup_of = obj.get("duplicate_of", None)
                    if isinstance(dup_of, str):
                        dup_of = dup_of.strip() or None
                    if dup_of:
                        duplicate_of = dup_of
                except Exception:
                    # parse failed => reject (safe)
                    is_duplicate = True

            if is_duplicate and not duplicate_of:
                duplicate_of = _fallback_dup_of(it)

            results[candidate] = (is_duplicate, duplicate_of)

            _append_jsonl(
                paths["similarity_decisions_jsonl"],
                {
                    "stage": "similarity_batch_decision",
                    "candidate": candidate,
                    "is_duplicate": is_duplicate,
                    "duplicate_of": duplicate_of,
                    "decision": "reject" if is_duplicate else "accept",
                    "reason": "llm_duplicate" if is_duplicate else "llm_not_duplicate",
                    "hop": it.get("hop", 0) if it else None,
                    "current_entity": it.get("parent_entity", "") if it else None,
                    "parent_entity": it.get("parent_entity", "") if it else None,
                    "raw_response": (txt[:500] if isinstance(txt, str) else ""),
                    "custom_id": cid,
                },
            )

    # -------------------------
    # Missing outputs => reject (safe)
    # -------------------------
    for cid, it in cid_to_item.items():
        if cid in returned_cids:
            continue
        cand = cid_to_candidate.get(cid, "")
        dup_of = _fallback_dup_of(it)
        if cand:
            results[cand] = (True, dup_of)
        _append_jsonl(
            paths["similarity_decisions_jsonl"],
            {
                "stage": "similarity_batch_missing_output",
                "candidate": cand,
                "decision": "reject",
                "reason": "missing_in_batch_output",
                "duplicate_of": dup_of,
                "custom_id": cid,
                "hop": it.get("hop", 0) if it else None,
                "current_entity": it.get("parent_entity", "") if it else None,
                "parent_entity": it.get("parent_entity", "") if it else None,
            },
        )

    pass  # silenced
    return results


#--------- 
#____embeddings engine___________
_SIM_ENGINE_LOCK = threading.RLock()
_SIM_ENGINE = None


def _cosine_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    if n <= 0.0 or not np.isfinite(n):
        return v
    return v / n


def _normalize_rows(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float32)
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms = np.where(norms <= 0.0, 1.0, norms)
    return (M / norms).astype(np.float32)


@dataclass
class _EmbEntry:
    """Lightweight entry - NO embedding storage (it's in the matrix)"""
    entity: str
    canon_key: str
    meta: dict  # Minimal meta only


class SimilarityEngine:
    """
    Memory-efficient + fast similarity engine.
    
    Key optimizations:
    1. Single embedding storage in pre-allocated matrix (not in entries)
    2. Amortized O(1) matrix growth (doubling strategy)
    3. Minimal meta storage (only fields needed for LLM context)
    4. In-place normalization
    """

    def __init__(self, args, paths):
        self.args = args
        self.paths = paths
        self.lock = threading.RLock()

        # Entry metadata (NO embeddings here - they're in the matrix)
        self.entries: list[_EmbEntry] = []
        self._canon_index: Set[str] = set()
        self._dim: Optional[int] = None

        # Pre-allocated matrix with amortized growth
        self._matrix: Optional[np.ndarray] = None
        self._matrix_capacity = 0
        self._matrix_size = 0

        # Provider config
        self.provider = (getattr(args, "similarity_provider", "openai") or "openai").lower()
        self.openai_model = getattr(args, "similarity_generation_model", "text-embedding-3-small")
        self.local_model_name = getattr(args, "similarity_local_model", "sentence-transformers/all-MiniLM-L6-v2")
        self.embed_batch_size = int(getattr(args, "similarity_embed_batch_size", 64) or 64)
        self.embed_mode = (getattr(args, "similarity_embed_mode", "online") or "online").lower()

        self._st_model = None
        self._openai_client = None

        self._load_existing_embeddings()

    def _ensure_capacity(self, needed: int):
        """
        Ensure matrix has capacity for at least `needed` rows.
        Uses doubling strategy for O(1) amortized growth.
        Must hold lock.
        """
        if self._dim is None:
            return
            
        if self._matrix is not None and self._matrix_capacity >= needed:
            return
        
        # Double capacity or use minimum 2048
        new_capacity = max(2048, self._matrix_capacity * 2, needed)
        new_matrix = np.empty((new_capacity, self._dim), dtype=np.float32)
        
        if self._matrix is not None and self._matrix_size > 0:
            new_matrix[:self._matrix_size] = self._matrix[:self._matrix_size]
        
        self._matrix = new_matrix
        self._matrix_capacity = new_capacity

    def _add_embedding_locked(self, v: np.ndarray):
        """
        Add pre-normalized embedding to matrix. O(1) amortized.
        Must hold lock.
        """
        self._ensure_capacity(self._matrix_size + 1)
        self._matrix[self._matrix_size] = v
        self._matrix_size += 1

    def _get_search_matrix(self) -> Optional[np.ndarray]:
        """Get the active portion of the matrix for searching. Must hold lock."""
        if self._matrix is None or self._matrix_size == 0:
            return None
        return self._matrix[:self._matrix_size]

    def _extract_minimal_meta(self, obj: dict) -> dict:
        """Extract only the fields needed for LLM context"""
        return {
            "parent_subject": obj.get("parent_subject") or obj.get("parent_entity") or "",
            "parent_intro": obj.get("parent_intro") or obj.get("parent_intro_excerpt") or "",
            "parent_hop": obj.get("parent_hop"),
            "hop": obj.get("hop"),
        }

    def _load_existing_embeddings(self):
        """Load embeddings directly into pre-allocated matrix."""
        p = self.paths["embeddings_jsonl"]
        if not os.path.exists(p):
            return

        # First pass: count lines to pre-allocate
        line_count = 0
        with open(p, "r", encoding="utf-8") as f:
            for _ in f:
                line_count += 1
        
        if line_count == 0:
            return

        loaded = 0
        skipped_dim = 0
        skipped_dup = 0
        
        with self.lock:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue

                    emb = obj.get("embedding")
                    ent = obj.get("entity")
                    ck = obj.get("canon_key")
                    
                    if not isinstance(ent, str) or not ent.strip():
                        continue
                    if not isinstance(ck, str) or not ck.strip():
                        continue
                    if not isinstance(emb, list) or not emb:
                        continue

                    try:
                        v = _cosine_normalize(np.asarray(emb, dtype=np.float32))
                    except Exception:
                        continue

                    if v.ndim != 1 or v.shape[0] <= 0:
                        continue

                    d = int(v.shape[0])
                    if self._dim is None:
                        self._dim = d
                        # Pre-allocate based on line count estimate
                        self._ensure_capacity(line_count)
                    elif d != self._dim:
                        skipped_dim += 1
                        continue

                    if ck in self._canon_index:
                        skipped_dup += 1
                        continue

                    # Add directly to matrix (no separate array storage!)
                    self._add_embedding_locked(v)
                    
                    # Store minimal meta only
                    self.entries.append(_EmbEntry(
                        entity=ent, 
                        canon_key=ck, 
                        meta=self._extract_minimal_meta(obj)
                    ))
                    self._canon_index.add(ck)
                    loaded += 1

            if loaded:
                _dbg(f"[similarity] loaded {loaded} embeddings ({self._matrix_size}x{self._dim}), "
                     f"skipped: {skipped_dup} dup, {skipped_dim} dim mismatch")

    def _ensure_openai_client(self):
        if self._openai_client is None:
            self._openai_client = OpenAI()

    def _ensure_st_model(self):
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.local_model_name)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        texts = [t.strip() if isinstance(t, str) else str(t).strip() for t in texts]
        texts = [t for t in texts if t]
        if not texts:
            dim = self._dim if self._dim is not None else 1536
            return np.zeros((0, dim), dtype=np.float32)

        if self.provider == "local":
            self._ensure_st_model()
            vecs = self._st_model.encode(
                texts,
                batch_size=max(1, self.embed_batch_size),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(vecs, dtype=np.float32)

        if self.embed_mode == "batch":
            return self._embed_texts_openai_via_batch(texts)

        # Online OpenAI
        self._ensure_openai_client()
        all_vecs = []
        for chunk in _chunk_list(texts, max(1, self.embed_batch_size)):
            resp = self._openai_client.embeddings.create(model=self.openai_model, input=chunk)
            data = sorted(resp.data, key=lambda d: d.index)
            for d in data:
                v = _cosine_normalize(np.asarray(d.embedding, dtype=np.float32))
                all_vecs.append(v)
        
        return np.stack(all_vecs, axis=0).astype(np.float32)

    def _build_embed_text(self, *, entity: str, parent_entity: str = "", parent_intro: str = "") -> str:
        return (entity or "").strip()

    def search_topk(self, V: np.ndarray, *, top_k: int, compare_batch: int = 64):
        with self.lock:
            M = self._get_search_matrix()
            n_exist = 0 if M is None else int(M.shape[0])

        if M is None or n_exist == 0 or V.size == 0:
            m = int(V.shape[0]) if V.ndim == 2 else 0
            return ([[] for _ in range(m)], [[] for _ in range(m)], [0.0 for _ in range(m)])

        V = np.asarray(V, dtype=np.float32)
        if V.ndim != 2:
            raise ValueError("V must be 2D")

        top_k = max(1, min(int(top_k), n_exist))
        compare_batch = max(1, int(compare_batch))

        all_top_idx, all_top_scores, all_max = [], [], []

        for start in range(0, V.shape[0], compare_batch):
            chunk = V[start : start + compare_batch]
            sims = M @ chunk.T  # (n_exist, chunk_size)

            mx = sims.max(axis=0)
            k = min(top_k, sims.shape[0])
            part = np.argpartition(-sims, kth=k - 1, axis=0)[:k, :]

            for j in range(part.shape[1]):
                idxs = part[:, j]
                sc = sims[idxs, j]
                order = np.argsort(-sc)
                idxs = idxs[order]
                sc = sc[order]
                all_top_idx.append(idxs.tolist())
                all_top_scores.append(sc.tolist())
                all_max.append(float(mx[j]))

        return all_top_idx, all_top_scores, all_max

    # def _llm_same_entity_decision(self, *, candidate, parent_entity, parent_intro, similar_items, llm, timeout):
    #     sys = (
    #         "You are a strict entity deduplication assistant.\n"
    #         "Decide whether the CANDIDATE refers to the SAME entity as ANY existing item.\n"
    #         "Consider spelling/hyphen variants and singular/plural surface forms as potentially the same entity.\n"
    #         "Use the provided parent context snippets to disambiguate.\n"
    #         "Output ONLY JSON: {\"duplicate\":true/false, \"duplicate_of\":\"...\"|null}\n"
    #     )

    #     def _clip(s: str, n: int) -> str:
    #         s = (s or "").strip()
    #         return (s[:n] + "…") if len(s) > n else s

    #     intro_limit = int(getattr(self.args, "similarity_log_parent_intro_chars", 800) or 800)
    #     parent_intro_clip = _clip(parent_intro, intro_limit)

    #     lines = []
    #     k = int(getattr(self.args, "similarity_top_k", 5) or 5)
    #     for it in (similar_items or [])[:k]:
    #         ent = (it.get("entity") or "").strip()
    #         score = float(it.get("score", 0.0) or 0.0)
    #         ps = (it.get("parent_subject") or "").strip()
    #         pi = _clip(it.get("parent_intro_excerpt") or "", 350)
    #         ctx = f'parent="{ps}" intro="{pi}"' if ps and pi else f'parent="{ps}"' if ps else "parent=(unknown)"
    #         lines.append(f'- entity="{ent}" score={score:.4f} {ctx}')

    #     sim_block = "\n".join(lines) if lines else "(none)"
    #     user = (
    #         f"CANDIDATE: {candidate}\n\n"
    #         f"PRODUCED_BY_PARENT: {parent_entity}\n"
    #         f"PARENT_INTRO_EXCERPT: {parent_intro_clip}\n\n"
    #         "MOST_SIMILAR_EXISTING_ITEMS:\n"
    #         f"{sim_block}\n"
    #     )

    #     msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    #     try:
    #         try:
    #             resp = llm(msgs, timeout=timeout)
    #         except TypeError:
    #             resp = llm(msgs)
    #         txt = _unwrap_text(resp).strip()
    #         obj = json.loads(txt)
    #         dup = bool(obj.get("duplicate", False))
    #         dup_of = obj.get("duplicate_of")
    #         if dup and isinstance(dup_of, str) and dup_of.strip():
    #             return True, dup_of.strip()
    #         if dup and similar_items:
    #             return True, (similar_items[0].get("entity", "") or "").strip()
    #         return False, ""
    #     except Exception:
    #         if similar_items:
    #             return True, (similar_items[0].get("entity", "") or "").strip()
    #         return True, ""
    def _llm_same_entity_decision(self, *, candidate, parent_entity, parent_intro, similar_items, llm, timeout):
        sys = (
            "You are a strict entity deduplication assistant.\n"
            "Decide whether the CANDIDATE refers to the SAME entity as ANY existing item.\n"
            "Consider spelling/hyphen variants and singular/plural surface forms as potentially the same entity.\n"
            "Use the provided parent context snippets to disambiguate.\n"
            "Output ONLY JSON: {\"duplicate\":true/false, \"duplicate_of\":\"...\"|null}\n"
        )

        def _clip(s: str, n: int) -> str:
            s = (s or "").strip()
            return (s[:n] + "…") if len(s) > n else s

        intro_limit = int(getattr(self.args, "similarity_log_parent_intro_chars", 800) or 800)
        parent_intro_clip = _clip(parent_intro, intro_limit)

        lines = []
        k = int(getattr(self.args, "similarity_top_k", 5) or 5)
        for it in (similar_items or [])[:k]:
            ent = (it.get("entity") or "").strip()
            score = float(it.get("score", 0.0) or 0.0)
            ps = (it.get("parent_subject") or "").strip()
            pi = _clip(it.get("parent_intro_excerpt") or "", 350)
            ctx = f'parent="{ps}" intro="{pi}"' if ps and pi else f'parent="{ps}"' if ps else "parent=(unknown)"
            lines.append(f'- entity="{ent}" score={score:.4f} {ctx}')

        sim_block = "\n".join(lines) if lines else "(none)"
        user = (
            f"CANDIDATE: {candidate}\n\n"
            f"PRODUCED_BY_PARENT: {parent_entity}\n"
            f"PARENT_INTRO_EXCERPT: {parent_intro_clip}\n\n"
            "MOST_SIMILAR_EXISTING_ITEMS:\n"
            f"{sim_block}\n"
        )

        msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
        try:
            try:
                resp = llm(msgs, timeout=timeout)
            except TypeError:
                resp = llm(msgs)
            txt = _unwrap_text(resp).strip()

            try:
                _save_online_api_response(
                    self.paths, "similarity", candidate, 0, resp,
                    provider=(self.provider or "openai"),
                    text_preview=(txt[:500] if isinstance(txt, str) else ""),
                )
            except Exception:
                pass

            obj = json.loads(txt)
            dup = bool(obj.get("duplicate", False))
            dup_of = obj.get("duplicate_of")
            if dup and isinstance(dup_of, str) and dup_of.strip():
                return True, dup_of.strip()
            if dup and similar_items:
                return True, (similar_items[0].get("entity", "") or "").strip()
            return False, ""
        except Exception:
            if similar_items:
                return True, (similar_items[0].get("entity", "") or "").strip()
            return True, ""
    def filter_candidates(self, *, parent_entity, parent_intro, hop, candidates, similarity_filter_llm=None, timeout=90.0):
        """
        Original single-parent filter method.
        """
        if not candidates:
            return [], {}

        top_k = int(getattr(self.args, "similarity_top_k", 5) or 5)
        th = float(getattr(self.args, "similarity_threshold", 0.92) or 0.92)
        action = (getattr(self.args, "similarity_action", "llm") or "llm").lower()
        compare_batch = int(getattr(self.args, "similarity_compare_batch", 64) or 64)

        texts = [self._build_embed_text(entity=c) for c in candidates]
        V = self.embed_texts(texts)

        with self.lock:
            entries_snapshot = list(self.entries)

        top_idx, top_scores, max_scores = self.search_topk(V, top_k=top_k, compare_batch=compare_batch)

        def _clip(s: str, limit: int) -> str:
            s = (s or "").strip()
            return (s[:limit] + "…") if len(s) > limit else s

        parent_intro_excerpt = _clip(
            parent_intro,
            int(getattr(self.args, "similarity_log_parent_intro_chars", 800) or 800),
        )

        decision_model_key = getattr(self.args, "similarity_filter_model_key", None) or getattr(self.args, "ner_model_key", None)
        filter_mode = "embedding_only" if (action == "reject" or similarity_filter_llm is None) else "embedding_llm"

        kept: list[str] = []
        kept_meta: dict[str, dict] = {}

        for i, cand in enumerate(candidates):
            mx = float(max_scores[i]) if i < len(max_scores) else 0.0

            sim_items = []
            idxs = top_idx[i] if i < len(top_idx) else []
            scs = top_scores[i] if i < len(top_scores) else []
            for j, idx in enumerate(idxs):
                if 0 <= idx < len(entries_snapshot):
                    e = entries_snapshot[idx]
                    sc = float(scs[j]) if j < len(scs) else 0.0
                    meta = e.meta if isinstance(e.meta, dict) else {}
                    sim_items.append({
                        "entity": e.entity,
                        "canon_key": e.canon_key,
                        "score": sc,
                        "parent_subject": meta.get("parent_subject", ""),
                        "parent_intro_excerpt": meta.get("parent_intro", ""),
                        "parent_hop": meta.get("parent_hop"),
                        "hop": meta.get("hop"),
                    })

            decision, reason, duplicate_of = "accept", "below_threshold", None

            if mx >= th and sim_items:
                if filter_mode == "embedding_only":
                    decision, reason, duplicate_of = "reject", "above_threshold_reject", sim_items[0]["entity"]
                else:
                    is_dup, dup_of = self._llm_same_entity_decision(
                        candidate=cand,
                        parent_entity=parent_entity,
                        parent_intro=parent_intro,
                        similar_items=sim_items,
                        llm=similarity_filter_llm,
                        timeout=timeout,
                    )
                    if is_dup:
                        decision, reason, duplicate_of = "reject", "llm_duplicate", (dup_of or sim_items[0]["entity"])
                    else:
                        decision, reason = "accept", "llm_not_duplicate"

            _append_jsonl(
                self.paths["similarity_decisions_jsonl"],
                {
                    "stage": "similarity_filter",
                    "parent_entity": parent_entity,
                    "parent_intro_excerpt": parent_intro_excerpt,
                    "hop": hop,
                    "candidate": cand,
                    "max_similarity": mx,
                    "threshold": th,
                    "top_similar": sim_items[:top_k],
                    "decision": decision,
                    "reason": reason,
                    "duplicate_of": duplicate_of,
                    "provider": self.provider,
                    "embed_model": (self.local_model_name if self.provider == "local" else self.openai_model),
                    "filter_mode": filter_mode,
                    "decision_model_key": decision_model_key,
                },
            )

            if decision == "reject":
                _append_jsonl(
                    self.paths["reject_similarity_jsonl"],
                    {
                        "parent_entity": parent_entity,
                        "parent_intro_excerpt": parent_intro_excerpt,
                        "hop": hop,
                        "candidate": cand,
                        "max_similarity": mx,
                        "threshold": th,
                        "top_similar": sim_items[:top_k],
                        "reason": reason,
                        "duplicate_of": duplicate_of,
                        "filter_mode": filter_mode,
                        "decision_model_key": decision_model_key,
                    },
                )
                continue

            kept.append(cand)
            vec = V[i]

            kept_meta[cand] = {
                "entity": cand,
                "canon_key": canon_key_from_queue(cand),
                "hop": hop + 1,
                "embedding_type": "queue_subject",
                "embed_text": texts[i],
                "embedding": vec.tolist(),
                "provider": self.provider,
                "model": (self.local_model_name if self.provider == "local" else self.openai_model),
                "parent_subject": parent_entity,
                "parent_hop": hop,
                "parent_intro": parent_intro,
                "created_at": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            }

        return kept, kept_meta

    def commit_embedding_if_inserted(self, meta: dict, writers: Optional[dict] = None):
        """
        Commit embedding to index. O(1) amortized time and memory.
        Works for both buffered (online) and direct (batch) modes.
        """
        if not isinstance(meta, dict):
            return

        ent = (meta.get("entity") or "").strip()
        ck = (meta.get("canon_key") or "").strip()
        emb = meta.get("embedding")

        if not ent or not ck or not isinstance(emb, list) or not emb:
            return

        v = _cosine_normalize(np.asarray(emb, dtype=np.float32))
        if v.ndim != 1 or v.shape[0] <= 0:
            return

        with self.lock:
            if self._dim is None:
                self._dim = int(v.shape[0])
            elif int(v.shape[0]) != self._dim:
                return

            if ck in self._canon_index:
                return

            # Write to JSONL (buffered or direct)
            if writers is not None and "embeddings" in writers:
                writers["embeddings"].append(meta)
            else:
                _append_jsonl(self.paths["embeddings_jsonl"], meta)

            # Add to matrix - O(1) amortized (no vstack!)
            self._add_embedding_locked(v)
            
            # Store entry with minimal meta (embedding NOT stored here - it's in matrix)
            self.entries.append(_EmbEntry(
                entity=ent, 
                canon_key=ck, 
                meta=self._extract_minimal_meta(meta)
            ))
            self._canon_index.add(ck)

    def commit_embedding_if_inserted_buffered(self, meta: dict, writers: dict):
        """Convenience wrapper for buffered mode."""
        return self.commit_embedding_if_inserted(meta, writers=writers)

    def ensure_embedding_for_subject(self, *, subject: str, hop: int, parent_entity: str = "", parent_intro: str = ""):
        """Ensure a subject has an embedding. Used for backfill."""
        s = (subject or "").strip()
        if not s:
            return
        ck = canon_key_from_queue(s)
        if not ck:
            return

        with self.lock:
            if ck in self._canon_index:
                return

        embed_text = self._build_embed_text(entity=s)
        V = self.embed_texts([embed_text])
        if V.ndim != 2 or V.shape[0] != 1:
            return
        vec = V[0]

        parent_subject = (parent_entity or "").strip()
        meta = {
            "entity": s,
            "canon_key": ck,
            "hop": int(hop),
            "embedding_type": "queue_subject",
            "embed_text": embed_text,
            "embedding": vec.tolist(),
            "provider": self.provider,
            "model": (self.local_model_name if self.provider == "local" else self.openai_model),
            "parent_subject": parent_subject,
            "parent_hop": (int(hop) - 1 if parent_subject else None),
            "parent_intro": (parent_intro or ""),
            "created_at": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        }
        self.commit_embedding_if_inserted(meta)

    def ensure_queue_embeddings_from_subject_hops(self, subject_hops: List[Tuple[str, int]]):
        """Backfill embeddings for all queue subjects."""
        if not subject_hops:
            return

        best: Dict[str, Tuple[str, int]] = {}
        for subj, hop in subject_hops:
            s = (subj or "").strip()
            if not s:
                continue
            ck = canon_key_from_queue(s)
            if not ck:
                continue
            if ck in best:
                if int(hop) < int(best[ck][1]):
                    best[ck] = (s, int(hop))
            else:
                best[ck] = (s, int(hop))

        for ck, (s, hop) in best.items():
            with self.lock:
                if ck in self._canon_index:
                    continue
            self.ensure_embedding_for_subject(subject=s, hop=hop, parent_entity="", parent_intro="")

    def _embed_texts_openai_via_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings via OpenAI Batch API with robust fallback."""
        self._ensure_openai_client()
        client = self._openai_client

        batches_dir = self.paths["batches_dir"]
        os.makedirs(batches_dir, exist_ok=True)

        endpoint = "/v1/embeddings"
        poll_interval = float(getattr(self.args, "batch_poll_interval", 30.0) or 30.0)

        original_texts = list(texts)
        req_rows = []

        start = 0
        for chunk in _chunk_list(texts, max(1, self.embed_batch_size)):
            n = len(chunk)
            if n == 0:
                continue
            cid = f"emb::{start}::n={n}"
            body = {"model": self.openai_model, "input": chunk}
            req_rows.append({"custom_id": cid, "method": "POST", "url": endpoint, "body": body})
            start += n

        if not req_rows:
            dim = self._dim if self._dim is not None else 1536
            return np.zeros((0, dim), dtype=np.float32)

        stamp = int(time.time() * 1000)
        in_path = os.path.join(batches_dir, f"embeddings_input_{stamp}.jsonl")
        with open(in_path, "w", encoding="utf-8") as f:
            for r in req_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        try:
            with open(in_path, "rb") as fh:
                input_file = client.files.create(file=fh, purpose="batch")

            job = client.batches.create(
                input_file_id=input_file.id,
                endpoint=endpoint,
                completion_window="24h",
                metadata={"description": f"LLMPedia embeddings model={self.openai_model} n_texts={len(texts)}"},
            )

            while True:
                job = client.batches.retrieve(job.id)
                if job.status == "completed":
                    break
                if job.status in {"failed", "expired", "cancelled"}:
                    _append_error_log(
                        self.paths,
                        f"[embeddings-batch] failed status={job.status}; falling back to online"
                    )
                    return self._embed_texts_online_fallback(original_texts)
                time.sleep(poll_interval)

            if not getattr(job, "output_file_id", None):
                return self._embed_texts_online_fallback(original_texts)

            out_bytes = client.files.content(job.output_file_id).content
            out_path = os.path.join(batches_dir, f"embeddings_output_{stamp}_{job.id}.jsonl")
            with open(out_path, "wb") as f:
                f.write(out_bytes)

            dim = self._dim if self._dim is not None else 1536
            vecs = [None] * len(texts)

            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue

                    cid = row.get("custom_id") or ""
                    if not cid.startswith("emb::"):
                        continue

                    try:
                        _, rest = cid.split("emb::", 1)
                        start_str, n_str = rest.split("::n=", 1)
                        start0 = int(start_str)
                    except Exception:
                        continue

                    resp = row.get("response") or {}
                    if resp.get("status_code") and int(resp["status_code"]) >= 400:
                        continue

                    body = resp.get("body") or {}
                    data = body.get("data") or []

                    for item in data:
                        try:
                            j = int(item.get("index"))
                            emb = item.get("embedding")
                            if isinstance(emb, list) and emb:
                                v = _cosine_normalize(np.asarray(emb, dtype=np.float32))
                                gi = start0 + j
                                if 0 <= gi < len(vecs):
                                    vecs[gi] = v
                                    if self._dim is None:
                                        self._dim = int(v.shape[0])
                                        dim = self._dim
                        except Exception:
                            continue

            # Fill missing with online fallback
            missing_indices = [i for i, v in enumerate(vecs) if v is None]
            if missing_indices:
                for idx in missing_indices:
                    try:
                        resp = client.embeddings.create(model=self.openai_model, input=[original_texts[idx]])
                        v = _cosine_normalize(np.asarray(resp.data[0].embedding, dtype=np.float32))
                        vecs[idx] = v
                    except Exception:
                        vecs[idx] = np.zeros(dim, dtype=np.float32)

            for i, v in enumerate(vecs):
                if v is None:
                    vecs[i] = np.zeros(dim, dtype=np.float32)

            return np.stack(vecs, axis=0).astype(np.float32)

        except Exception as e:
            _append_error_log(self.paths, f"[embeddings-batch] exception: {e!r}; falling back to online")
            return self._embed_texts_online_fallback(original_texts)

    def _embed_texts_online_fallback(self, texts: list[str]) -> np.ndarray:
        """Robust online embedding fallback."""
        if not texts:
            dim = self._dim if self._dim is not None else 1536
            return np.zeros((0, dim), dtype=np.float32)

        self._ensure_openai_client()
        all_vecs = []
        dim = self._dim if self._dim is not None else 1536

        for chunk in _chunk_list(texts, max(1, self.embed_batch_size)):
            try:
                resp = self._openai_client.embeddings.create(model=self.openai_model, input=chunk)
                data = sorted(resp.data, key=lambda d: d.index)
                for d in data:
                    v = _cosine_normalize(np.asarray(d.embedding, dtype=np.float32))
                    all_vecs.append(v)
                    if self._dim is None:
                        self._dim = int(v.shape[0])
                        dim = self._dim
            except Exception:
                for _ in chunk:
                    all_vecs.append(np.zeros(dim, dtype=np.float32))

        if not all_vecs:
            return np.zeros((len(texts), dim), dtype=np.float32)

        return np.stack(all_vecs, axis=0).astype(np.float32)

    def filter_candidates_batch(
        self,
        *,
        items: List[dict],
        similarity_cfg,
        client: "OpenAI",
        wave_idx: int,
        writers: Optional[dict] = None,
    ) -> Tuple[List[str], Dict[str, dict]]:
        """
        Unified similarity filter for both online (buffered) and batch modes.
        
        Checks:
        1. Against existing DB embeddings
        2. Within-batch duplicates
        
        Args:
            items: List of {"candidate", "parent_entity", "parent_intro", "hop"}
            similarity_cfg: Config for LLM similarity decisions
            client: OpenAI client (can be None for online LLM mode)
            wave_idx: Wave counter for batch job naming
            writers: Optional BufferedJSONLWriter dict (for online parallel mode)
        """
        if not items:
            return [], {}

        def _log(key: str, obj: dict):
            if writers is not None and key in writers:
                writers[key].append(obj)
            else:
                path_map = {
                    "similarity_decisions": self.paths["similarity_decisions_jsonl"],
                    "reject_similarity": self.paths["reject_similarity_jsonl"],
                    "embeddings": self.paths["embeddings_jsonl"],
                }
                if key in path_map:
                    _append_jsonl(path_map[key], obj)

        top_k = int(getattr(self.args, "similarity_top_k", 5) or 5)
        th = float(getattr(self.args, "similarity_threshold", 0.92) or 0.92)
        action = (getattr(self.args, "similarity_action", "llm") or "llm").lower()
        compare_batch = int(getattr(self.args, "similarity_compare_batch", 64) or 64)
        similarity_mode = (getattr(self.args, "similarity_mode", "online") or "online").lower()

        # ═══════════════════════════════════════════════════════════════════
        # Step 0: Deduplicate + skip already-indexed candidates
        # ═══════════════════════════════════════════════════════════════════
        all_candidates: List[str] = []
        item_by_candidate: Dict[str, dict] = {}

        for it in items:
            cand = (it.get("candidate") or "").strip()
            if not cand or cand in item_by_candidate:
                continue

            ck = canon_key_from_queue(cand)
            if ck:
                with self.lock:
                    if ck in self._canon_index:
                        _log("similarity_decisions", {
                            "stage": "similarity_filter_skip_indexed",
                            "candidate": cand,
                            "canon_key": ck,
                            "hop": it.get("hop", 0),
                            "parent_entity": it.get("parent_entity", ""),
                            "decision": "skip",
                            "reason": "already_in_embedding_index",
                        })
                        continue

            all_candidates.append(cand)
            item_by_candidate[cand] = it

        if not all_candidates:
            return [], {}

        # ═══════════════════════════════════════════════════════════════════
        # Step 1: Generate embeddings for non-indexed candidates
        # ═══════════════════════════════════════════════════════════════════
        texts = [self._build_embed_text(entity=c) for c in all_candidates]
        V = self.embed_texts(texts)

        with self.lock:
            entries_snapshot = list(self.entries)

        # ═══════════════════════════════════════════════════════════════════
        # Step 2a: Compare against existing DB
        # ═══════════════════════════════════════════════════════════════════
        top_idx, top_scores, max_db_scores = self.search_topk(V, top_k=top_k, compare_batch=compare_batch)

        # ═══════════════════════════════════════════════════════════════════
        # Step 2b: Compare within-batch (against each other)
        # ═══════════════════════════════════════════════════════════════════
        N = V.shape[0]
        max_batch_scores = np.zeros(N, dtype=np.float32)
        batch_duplicate_of: Dict[int, int] = {}

        if N > 1:
            V_norm = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
            batch_sims = V_norm @ V_norm.T
            np.fill_diagonal(batch_sims, 0.0)
            max_batch_scores = batch_sims.max(axis=1)

            for i in range(N):
                best_j = int(batch_sims[i].argmax())
                if batch_sims[i, best_j] >= th and best_j < i:
                    batch_duplicate_of[i] = best_j

        final_max = np.maximum(np.array(max_db_scores, dtype=np.float32), max_batch_scores)

        # ═══════════════════════════════════════════════════════════════════
        # Step 3: Categorize candidates
        # ═══════════════════════════════════════════════════════════════════
        needs_llm_check: List[dict] = []
        auto_accept: List[str] = []
        auto_reject: List[Tuple[str, str, str]] = []

        for i, cand in enumerate(all_candidates):
            mx_db = float(max_db_scores[i]) if i < len(max_db_scores) else 0.0
            mx_batch = float(max_batch_scores[i])
            mx = float(final_max[i])

            it = item_by_candidate[cand]

            # Build similar-items list
            sim_items = []
            idxs = top_idx[i] if i < len(top_idx) else []
            scs = top_scores[i] if i < len(top_scores) else []
            for j, idx in enumerate(idxs):
                if 0 <= idx < len(entries_snapshot):
                    e = entries_snapshot[idx]
                    sc = float(scs[j]) if j < len(scs) else 0.0
                    meta = e.meta if isinstance(e.meta, dict) else {}
                    sim_items.append({
                        "entity": e.entity,
                        "canon_key": e.canon_key,
                        "score": sc,
                        "parent_subject": meta.get("parent_subject", ""),
                        "parent_intro_excerpt": meta.get("parent_intro", ""),
                        "parent_hop": meta.get("parent_hop"),
                        "hop": meta.get("hop"),
                    })

            it["_max_similarity"] = mx
            it["_max_db_similarity"] = mx_db
            it["_max_batch_similarity"] = mx_batch
            it["_similar_items"] = sim_items
            it["_embedding"] = V[i]
            it["_embed_text"] = texts[i]

            if i in batch_duplicate_of:
                first_cand = all_candidates[batch_duplicate_of[i]]
                auto_reject.append((cand, first_cand, "within_batch_duplicate"))
            elif mx < th or not sim_items:
                auto_accept.append(cand)
            elif action == "reject":
                dup_of = sim_items[0]["entity"] if sim_items else None
                reason = "above_threshold_db" if mx_db >= th else "above_threshold_batch"
                auto_reject.append((cand, dup_of, reason))
            else:
                needs_llm_check.append({
                    "candidate": cand,
                    "parent_entity": it.get("parent_entity", ""),
                    "parent_intro": it.get("parent_intro", ""),
                    "similar_items": sim_items,
                    "hop": it.get("hop", 0),
                })

        # ═══════════════════════════════════════════════════════════════════
        # Step 4: LLM checks (batch or online)
        # ═══════════════════════════════════════════════════════════════════
        llm_decisions: Dict[str, Tuple[bool, Optional[str]]] = {}

        if needs_llm_check:
            if similarity_mode == "batch" and client is not None:
                llm_decisions = _run_similarity_wave_via_openai_batch(
                    args=self.args,
                    paths=self.paths,
                    similarity_cfg=similarity_cfg,
                    client=client,
                    wave_items=needs_llm_check,
                    wave_idx=wave_idx,
                )
            else:
                llm = make_llm_from_config(similarity_cfg)
                for check_item in needs_llm_check:
                    cand = check_item["candidate"]
                    try:
                        is_dup, dup_of = self._llm_same_entity_decision(
                            candidate=cand,
                            parent_entity=check_item["parent_entity"],
                            parent_intro=check_item["parent_intro"],
                            similar_items=check_item["similar_items"],
                            llm=llm,
                            timeout=self.args.timeout,
                        )
                        llm_decisions[cand] = (is_dup, dup_of)
                    except Exception:
                        sim_items = check_item.get("similar_items") or []
                        llm_decisions[cand] = (True, sim_items[0]["entity"] if sim_items else None)

        # ═══════════════════════════════════════════════════════════════════
        # Step 5: Compile results
        # ═══════════════════════════════════════════════════════════════════
        kept: List[str] = []
        kept_meta: Dict[str, dict] = {}

        def _build_meta(cand: str, it: dict) -> dict:
            return {
                "entity": cand,
                "canon_key": canon_key_from_queue(cand),
                "hop": it.get("hop", 0) + 1,
                "embedding_type": "queue_subject",
                "embed_text": it.get("_embed_text", ""),
                "embedding": it["_embedding"].tolist() if "_embedding" in it else [],
                "provider": self.provider,
                "model": (self.local_model_name if self.provider == "local" else self.openai_model),
                "parent_subject": it.get("parent_entity", ""),
                "parent_hop": it.get("hop", 0),
                "parent_intro": it.get("parent_intro", ""),
                "created_at": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            }

        # Auto-accept
        for cand in auto_accept:
            it = item_by_candidate[cand]
            kept.append(cand)
            kept_meta[cand] = _build_meta(cand, it)
            _log("similarity_decisions", {
                "stage": "similarity_filter_batch",
                "candidate": cand,
                "max_similarity": it.get("_max_similarity", 0.0),
                "max_db_similarity": it.get("_max_db_similarity", 0.0),
                "max_batch_similarity": it.get("_max_batch_similarity", 0.0),
                "decision": "accept",
                "reason": "below_threshold",
                "hop": it.get("hop", 0),
                "parent_entity": it.get("parent_entity", ""),
            })

        # Auto-reject
        for cand, dup_of, reason in auto_reject:
            it = item_by_candidate[cand]
            _log("similarity_decisions", {
                "stage": "similarity_filter_batch",
                "candidate": cand,
                "max_similarity": it.get("_max_similarity", 0.0),
                "max_db_similarity": it.get("_max_db_similarity", 0.0),
                "max_batch_similarity": it.get("_max_batch_similarity", 0.0),
                "decision": "reject",
                "reason": reason,
                "duplicate_of": dup_of,
                "hop": it.get("hop", 0),
                "parent_entity": it.get("parent_entity", ""),
            })
            _log("reject_similarity", {
                "candidate": cand,
                "max_similarity": it.get("_max_similarity", 0.0),
                "reason": reason,
                "duplicate_of": dup_of,
                "hop": it.get("hop", 0),
    "parent_entity": it.get("parent_entity", ""),
            })

        # LLM decisions
        for check_item in needs_llm_check:
            cand = check_item["candidate"]
            it = item_by_candidate[cand]
            is_dup, dup_of = llm_decisions.get(cand, (True, None))

            if is_dup:
                _log("similarity_decisions", {
                    "stage": "similarity_filter_batch",
                    "candidate": cand,
                    "max_similarity": it.get("_max_similarity", 0.0),
                    "decision": "reject",
                    "reason": "llm_duplicate",
                    "duplicate_of": dup_of,
                    "hop": it.get("hop", 0),
                    "parent_entity": it.get("parent_entity", ""),

                })
                _log("reject_similarity", {
                    "candidate": cand,
                    "max_similarity": it.get("_max_similarity", 0.0),
                    "reason": "llm_duplicate",
                    "duplicate_of": dup_of,
                    "hop": it.get("hop", 0),
                    "parent_entity": it.get("parent_entity", ""),
                    "hop": it.get("hop", 0),
                    "parent_entity": it.get("parent_entity", ""),
                })
            else:
                kept.append(cand)
                kept_meta[cand] = _build_meta(cand, it)
                _log("similarity_decisions", {
                    "stage": "similarity_filter_batch",
                    "candidate": cand,
                    "max_similarity": it.get("_max_similarity", 0.0),
                    "decision": "accept",
                    "reason": "llm_not_duplicate",
                    "hop": it.get("hop", 0),
                    "parent_entity": it.get("parent_entity", ""),
                })

        return kept, kept_meta

    def filter_candidates_batch_buffered(self, *, items, similarity_cfg, client, wave_idx, writers):
        """Convenience wrapper - just calls filter_candidates_batch with writers."""
        return self.filter_candidates_batch(
            items=items, similarity_cfg=similarity_cfg, client=client, wave_idx=wave_idx, writers=writers
        )

def _get_similarity_engine(args, paths) -> SimilarityEngine | None:
    if not bool(getattr(args, "use_similarity", False)):
        return None
    global _SIM_ENGINE
    with _SIM_ENGINE_LOCK:
        if _SIM_ENGINE is None:
            _SIM_ENGINE = SimilarityEngine(args, paths)
        return _SIM_ENGINE


# ---------------- online mode (JSON only) ----------------



import hashlib

def _pq_atomic_write_json(path: str, obj: Any):
    tmp = path + ".tmp"
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def _pq_item_key(subject: str, hop: int) -> str:
    # short stable filename; keeps readable prefix for debugging
    base = canon_key_from_queue(subject)[:60].replace(" ", "_")
    h = hashlib.sha1(f"{subject}\u241E{hop}".encode("utf-8")).hexdigest()[:12]
    return f"{base}_h{hop}_{h}"

def _pq_payload_path(paths: dict, stage: str, subject: str, hop: int) -> str:
    root = os.path.join(paths["pq_payload_dir"], stage)
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, _pq_item_key(subject, hop) + ".json")

def _pq_payload_exists(paths: dict, stage: str, subject: str, hop: int) -> bool:
    return os.path.exists(_pq_payload_path(paths, stage, subject, hop))

def _pq_payload_write(paths: dict, stage: str, subject: str, hop: int, obj: dict):
    p = _pq_payload_path(paths, stage, subject, hop)
    _pq_atomic_write_json(p, obj)

def _pq_payload_read(paths: dict, stage: str, subject: str, hop: int) -> Optional[dict]:
    p = _pq_payload_path(paths, stage, subject, hop)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _pq_stage_enabled_selfrag(args) -> bool:
    return bool(getattr(args, "self_rag", False))

def _pq_stage_enabled_outline(args) -> bool:
    return bool(getattr(args, "two_stage_elicit", True))

def _pq_prereq_selfrag_done(args, paths, subject, hop) -> bool:
    if not _pq_stage_enabled_selfrag(args):
        return True
    return _pq_payload_exists(paths, "selfrag", subject, hop)

def _pq_prereq_outline_done(args, paths, subject, hop) -> bool:
    if not _pq_stage_enabled_outline(args):
        return True
    return _pq_payload_exists(paths, "outline", subject, hop)

def _pq_try_enqueue_elicit(
    args, paths, elicit_q: JsonQueue, subject: str, hop: int,
    parent_subject: Optional[str] = None, parent_hop: Optional[int] = None,
):
    # Don't re-enqueue if already processed
    if _pq_payload_exists(paths, "elicit", subject, hop):
        return
    # Elicitation requires (selfrag if enabled) AND (outline if enabled)
    if _pq_prereq_selfrag_done(args, paths, subject, hop) and _pq_prereq_outline_done(args, paths, subject, hop):
        elicit_q.enqueue(subject, hop, parent_subject=parent_subject, parent_hop=parent_hop)

def _pq_stage_enqueue_with_parent(
    paths: dict, q: JsonQueue, subject: str, hop: int,
    parent_subject: Optional[str] = None, parent_hop: Optional[int] = None,
):
    q.enqueue(subject, hop, parent_subject=parent_subject, parent_hop=parent_hop)


def _pq_make_stage_queue(paths: dict, name: str, args) -> JsonQueue:
    # name in {"selfrag","outline","elicit","ner","sim"}
    key_json = f"pq_{name}_queue_json"
    key_jsonl = f"pq_{name}_queue_jsonl"
    return JsonQueue(
        paths[key_json],
        paths[key_jsonl],
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
        retry_backoff=args.retry_backoff,
        retry_max_sleep=args.retry_max_sleep,
        retry_jitter=args.retry_jitter,
        buffer_size=int(getattr(args, "queue_buffer_size", 100) or 100),
        flush_interval=float(getattr(args, "queue_flush_interval", 5.0) or 5.0),
    )

def _pq_reset_working_if_requested(args, *queues: JsonQueue):
    if not bool(getattr(args, "resume", False)):
        return
    if not bool(getattr(args, "reset_working", False)):
        return
    for q in queues:
        try:
            n = q.reset_working_to_pending()
            if n:
                _dbg(f"[pq-resume] reset {n} working→pending on stage queue")
        except Exception:
            pass

def _pq_sleep_on_retry_due(q: JsonQueue, max_depth: int, base_sleep: float = 0.2):
    # If queue is empty but has pending retries not yet due, sleep until next due (bounded).
    try:
        due_in = q.next_due_in(max_depth)
    except Exception:
        due_in = None
    if due_in is None:
        time.sleep(base_sleep)
    else:
        time.sleep(min(max(due_in, 0.05), 1.0))





class BufferedJSONLWriter:
    """
    Thread-safe buffered JSONL writer.
    Flushes when buffer reaches max_buffer_size OR after flush_interval seconds.
    """
    def __init__(self, path: str, max_buffer_size: int = 500, flush_interval: float = 10.0):
        self.path = path
        self.max_buffer_size = max_buffer_size
        self.flush_interval = flush_interval
        self.buffer: List[dict] = []
        self.lock = threading.Lock()
        self.last_flush = time.time()
        
    def append(self, obj: dict):
        """Add object to buffer (may trigger flush)"""
        with self.lock:
            self.buffer.append(obj)
            now = time.time()
            should_flush = (
                len(self.buffer) >= self.max_buffer_size or 
                (now - self.last_flush) >= self.flush_interval
            )
            if should_flush:
                self._flush_locked()
    
    def _flush_locked(self):
        """Flush buffer to disk (must hold lock)"""
        if not self.buffer:
            return
        
        dir_ = os.path.dirname(self.path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        
        with open(self.path, "a", encoding="utf-8") as f:
            for obj in self.buffer:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        
        self.buffer.clear()
        self.last_flush = time.time()
    
    def flush(self):
        """Force flush (call at end)"""
        with self.lock:
            self._flush_locked()


def _stage_should_stop(queue_obj: JsonQueue, args, stage_name: str) -> bool:
    """Check if a stage should stop due to caps"""
    if not args.max_subjects:
        return False
    
    d, w, p, f = queue_obj.metrics(args.max_depth)
    
    # Stop if we've hit the cap
    if d >= args.max_subjects:
        # _dbg(f"[{stage_name}] Hit max-subjects cap (done={d}), stopping worker")  # ← REMOVE THIS LINE
        return True
    
    return False

def run_online_parallel(args, paths, el_cfg, ner_cfg, self_rag_cfg, queue: JsonQueue):
    """
    Online parallel pipeline with:
    - Buffered JSONL writes (flushes every N items or M seconds)
    - Full funnel stats at each stage
    - Non-blocking stage queues with payload-based dependencies
    - Signal handling for graceful shutdown
    - Per-stage cap enforcement
    """

    import signal
    import sys
    
    # ---- BUFFERED WRITERS SETUP ----
    buffer_size = int(getattr(args, "online_buffer_size", 500) or 500)
    flush_interval = float(getattr(args, "online_flush_interval", 10.0) or 10.0)
    
    writers = {
        "queue": BufferedJSONLWriter(paths["queue_jsonl"], buffer_size, flush_interval),
        "articles": BufferedJSONLWriter(paths["articles_jsonl"], buffer_size, flush_interval),
        "articles_wikitext": BufferedJSONLWriter(paths["articles_wikitext_jsonl"], buffer_size, flush_interval),
        "articles_meta": BufferedJSONLWriter(paths["articles_meta_jsonl"], buffer_size, flush_interval),
        "outlines": BufferedJSONLWriter(paths["outlines_jsonl"], buffer_size, flush_interval),
        "self_rag_log": BufferedJSONLWriter(paths["self_rag_log_jsonl"], buffer_size, flush_interval),
        "ner_decisions": BufferedJSONLWriter(paths["ner_decisions_jsonl"], buffer_size, flush_interval),
        "ner_lowconf": BufferedJSONLWriter(paths["ner_lowconf_jsonl"], buffer_size, flush_interval),
        "ner_responses": BufferedJSONLWriter(paths["ner_responses_jsonl"], buffer_size, flush_interval),
        "elicit_lowconf": BufferedJSONLWriter(paths["elicit_lowconf_jsonl"], buffer_size, flush_interval),
        "plural_s_dedup": BufferedJSONLWriter(paths["plural_s_dedup_jsonl"], buffer_size, flush_interval),
        "similarity_decisions": BufferedJSONLWriter(paths["similarity_decisions_jsonl"], buffer_size, flush_interval),
        "reject_similarity": BufferedJSONLWriter(paths["reject_similarity_jsonl"], buffer_size, flush_interval),
        "embeddings": BufferedJSONLWriter(paths["embeddings_jsonl"], buffer_size, flush_interval),
    }

    # 0) seed/resume + preload
    seen_canon_keys = _load_seen_canon(paths)
    _bootstrap_seen_from_queue(paths, seen_canon_keys)

    if getattr(args, "preload_only", False) and getattr(args, "preload_topics", None):
        # ── PRELOAD-ONLY: skip seed entirely, load only from file ──
        n = _preload_topics_from_file(args, paths, queue, seen_canon_keys)
        _dbg(f"[preload-only] {n} topics loaded, seed skipped")
    else:
        # ── Normal mode: seed first, then optionally preload extras ──
        _seed_or_resume_queue(args, paths, queue)
        seen_canon_keys.add(canon_key_from_queue(args.seed))
        if getattr(args, "preload_topics", None):
            _preload_topics_from_file(args, paths, queue, seen_canon_keys)


    # stage queues
    selfrag_q = _pq_make_stage_queue(paths, "selfrag", args)
    outline_q = _pq_make_stage_queue(paths, "outline", args)
    elicit_q  = _pq_make_stage_queue(paths, "elicit", args)
    ner_q     = _pq_make_stage_queue(paths, "ner", args)
    sim_q     = _pq_make_stage_queue(paths, "sim", args)

    _pq_reset_working_if_requested(args, selfrag_q, outline_q, elicit_q, ner_q, sim_q)

    # early stop if already at cap
    with _queue_lock:
        d0, w0, p0, f0 = queue.metrics(args.max_depth)
    if args.max_subjects and d0 >= args.max_subjects:
        _snapshot_json_only(paths, queue)
        return

    stop_event = threading.Event()
    prereq_capped_event = threading.Event()

    # ═══════════════════════════════════════════════════════════════════
    # SIGNAL HANDLER - Must be registered AFTER stop_event exists!
    # ═══════════════════════════════════════════════════════════════════
    shutdown_count = [0]

    def signal_handler(signum, frame):
        shutdown_count[0] += 1
        
        if shutdown_count[0] == 1:
            _dbg(f"\n[shutdown] Stopping gracefully... (Press Ctrl+C again to force quit)")
            stop_event.set()
            prereq_capped_event.set()  # Also stop prereq stages
        else:
            _dbg("[shutdown] Force quit!")
            try:
                for name, writer in writers.items():
                    writer.flush()
                queue.flush()
            except:
                pass
            sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)   # ← NOW stop_event exists!
    signal.signal(signal.SIGTERM, signal_handler)

    # ✅ global API cap across *all* stages
    api_sema = threading.BoundedSemaphore(value=max(1, int(getattr(args, "concurrency", 1) or 1)))

    # -----------------------------
    # Dispatcher: claim from MAIN -> enqueue SR+Outline (or write placeholders) -> maybe elicit
    # -----------------------------
    def dispatcher_loop():
        while not stop_event.is_set():
            with _queue_lock:
                d, w, p, f = queue.metrics(args.max_depth)

            # Cap guard: don't dispatch new work when we're at/over capacity
            if args.max_subjects and (d + w) >= args.max_subjects:
                time.sleep(0.2)
                continue

            # ═══════════════════════════════════════════════════════════════
            # THROTTLE: Don't overwhelm outline/elicit stages (especially for preload)
            # ═══════════════════════════════════════════════════════════════
            ol_d, ol_w, ol_p, ol_f = outline_q.metrics(args.max_depth)
            el_d, el_w, el_p, el_f = elicit_q.metrics(args.max_depth)
            inflight = (ol_w + ol_p) + (el_w + el_p)
            
            # Determine max inflight
            max_inflight = int(getattr(args, "max_dispatcher_inflight", 0) or 0)
            if max_inflight <= 0:
                # Auto: use batch_size * 2 for preload-only, unlimited otherwise
                if getattr(args, "preload_only", False):
                    max_inflight = max(100, int(getattr(args, "batch_size", 500) or 500) * 2)
                else:
                    max_inflight = 999999999  # effectively unlimited
            
            if inflight >= max_inflight:
                # Pipeline is busy - wait for batches to complete
                time.sleep(2.0)
                continue
            
            # Only claim what we have room for
            room = max_inflight - inflight
            base_claim = int(getattr(args, "batch_size", 500) or 500) if getattr(args, "preload_only", False) else int(getattr(args, "concurrency", 6) or 6)
            claim_n = min(room, base_claim)
            claim_n = max(1, claim_n)
            # ═══════════════════════════════════════════════════════════════

            if args.max_subjects:
                remaining = args.max_subjects - (d + w)
                if remaining <= 0:
                    time.sleep(0.2)
                    continue
                claim_n = max(1, min(claim_n, remaining))

            with _queue_lock:
                batch = queue.claim_pending_batch(args.max_depth, claim_n)

            if not batch:
                # MAIN may have retries pending
                try:
                    due_in = queue.next_due_in(args.max_depth)
                except Exception:
                    due_in = None
                if due_in is not None:
                    time.sleep(min(max(due_in, 0.05), 1.0))
                else:
                    time.sleep(0.2)
                continue

            for subject, hop in batch:
                # Get parent info from main queue record
                rec = queue.get_record(subject, hop)
                ps = rec.get("parent_subject") if isinstance(rec, dict) else None
                ph = rec.get("parent_hop") if isinstance(rec, dict) else None

                # selfrag prereq
                if _pq_stage_enabled_selfrag(args):
                    _pq_stage_enqueue_with_parent(paths, selfrag_q, subject, hop, ps, ph)
                else:
                    if not _pq_payload_exists(paths, "selfrag", subject, hop):
                        _pq_payload_write(
                            paths, "selfrag", subject, hop,
                            {"subject": subject, "hop": hop, "context": None, "skipped": True, "parent_subject": ps, "parent_hop": ph},
                        )

                # outline prereq
                if _pq_stage_enabled_outline(args):
                    _pq_stage_enqueue_with_parent(paths, outline_q, subject, hop, ps, ph)
                else:
                    if not _pq_payload_exists(paths, "outline", subject, hop):
                        _pq_payload_write(
                            paths, "outline", subject, hop,
                            {"subject": subject, "hop": hop, "outline": None, "skipped": True, "parent_subject": ps, "parent_hop": ph},
                        )

                # maybe elicit already possible
                _pq_try_enqueue_elicit(args, paths, elicit_q, subject, hop, parent_subject=ps, parent_hop=ph)

    # -----------------------------
    # Reconciler: repairs missing downstream enqueues + propagates permanent failures
    #  Cascade stop when both outline+self-rag cap
    # -----------------------------
    def reconciler_loop():
        interval = float(getattr(args, "reconcile_interval", 3.0) or 3.0)
        while not stop_event.is_set():
            try:
                # ✅ CASCADE STOP: When both prereq stages capped → stop self-rag, outline, ner, sim
                # But keep elicit running until it finishes
                if _pq_stage_enabled_selfrag(args) and _pq_stage_enabled_outline(args):
                    sr_d, sr_w, sr_p, sr_f = selfrag_q.metrics(args.max_depth)
                    ol_d, ol_w, ol_p, ol_f = outline_q.metrics(args.max_depth)
                    
                    both_capped = (
                        args.max_subjects and 
                        sr_d >= args.max_subjects and 
                        ol_d >= args.max_subjects
                    )
                    
                    if both_capped and not prereq_capped_event.is_set():
                        _dbg("[reconciler] Both prereqs capped → stopping self-rag, outline, ner, sim (keeping elicit)")
                        prereq_capped_event.set()
                
                main_state = []
                try:
                    if os.path.exists(paths["queue_json"]):
                        with open(paths["queue_json"], "r", encoding="utf-8") as f:
                            main_state = json.load(f) or []
                except Exception:
                    main_state = []

                for rec in (main_state or []):
                    if not isinstance(rec, dict):
                        continue
                    subject = rec.get("subject")
                    hop = rec.get("hop")
                    st = rec.get("status")
                    try:
                        hop = int(hop)
                    except Exception:
                        continue
                    if not isinstance(subject, str) or not subject.strip():
                        continue
                    if args.max_depth != 0 and hop > args.max_depth:
                        continue

                    if st not in {"working", "done"}:
                        continue

                    stages = []
                    if _pq_stage_enabled_selfrag(args): stages.append(("selfrag", selfrag_q))
                    if _pq_stage_enabled_outline(args): stages.append(("outline", outline_q))
                    stages.append(("elicit", elicit_q))
                    if bool(getattr(args, "use_ner", False)): stages.append(("ner", ner_q))
                    stages.append(("sim", sim_q))

                    for stage_name, qobj in stages:
                        try:
                            srec = qobj.get_record(subject, hop)
                        except Exception:
                            srec = None
                        if isinstance(srec, dict) and srec.get("status") == "failed":
                            with _queue_lock:
                                queue.mark_error(subject, hop, max_retries=args.max_retries, reason=f"stage_failed:{stage_name}")
                            break

                    # Extract parent from main queue record
                    ps = rec.get("parent_subject")
                    ph = rec.get("parent_hop")

                    if _pq_stage_enabled_selfrag(args):
                        if not _pq_payload_exists(paths, "selfrag", subject, hop):
                            _pq_stage_enqueue_with_parent(paths, selfrag_q, subject, hop, ps, ph)
                    else:
                        if not _pq_payload_exists(paths, "selfrag", subject, hop):
                            _pq_payload_write(
                                paths, "selfrag", subject, hop,
                                {"subject": subject, "hop": hop, "context": None, "skipped": True, "parent_subject": ps, "parent_hop": ph},
                            )

                    if _pq_stage_enabled_outline(args):
                        if not _pq_payload_exists(paths, "outline", subject, hop):
                            _pq_stage_enqueue_with_parent(paths, outline_q, subject, hop, ps, ph)
                    else:
                        if not _pq_payload_exists(paths, "outline", subject, hop):
                            _pq_payload_write(
                                paths, "outline", subject, hop,
                                {"subject": subject, "hop": hop, "outline": None, "skipped": True, "parent_subject": ps, "parent_hop": ph}
                            )

                    _pq_try_enqueue_elicit(args, paths, elicit_q, subject, hop, parent_subject=ps, parent_hop=ph)

                    if _pq_payload_exists(paths, "elicit", subject, hop):
                        if st == "done":
                            with _queue_lock:
                                queue.mark_done(subject, hop)

                        if bool(getattr(args, "use_ner", False)):
                            if not _pq_payload_exists(paths, "ner", subject, hop):
                                _pq_stage_enqueue_with_parent(paths, ner_q, subject, hop, ps, ph)
                            else:
                                _pq_stage_enqueue_with_parent(paths, sim_q, subject, hop, ps, ph)
                        else:
                            _pq_stage_enqueue_with_parent(paths, sim_q, subject, hop, ps, ph)

            except Exception as e:
                _append_error_log(paths, f"[pq-reconciler] error={type(e).__name__}: {e!r}", exc=e)

            time.sleep(max(0.2, interval))
    # ------------------------------------------------------------------
    # Stage workers (use worker_loop_with_inflight + global api_sema)
    #  Per-stage cap checks
    # ------------------------------------------------------------------
    def selfrag_worker_loop(worker_id: int):
        if not _pq_stage_enabled_selfrag(args):
            return
        llm = make_llm_from_config(self_rag_cfg)
        name = f"pq-selfrag-{worker_id}"

        def pop_job():
            if _stage_should_stop(selfrag_q, args, f"selfrag-{worker_id}"):
                return None
            batch = selfrag_q.claim_pending_batch(args.max_depth, 1)
            return batch[0] if batch else None

        def call(job):
            subject, hop = job

            if _pq_payload_exists(paths, "selfrag", subject, hop):
                return {"ok": True, "skipped": True}

            try:
                root_topic = args.seed if args.domain == "topic" else subject
                ctx = _run_self_rag_for_subject(
                    subject=subject,
                    hop=hop,
                    root_topic=root_topic,
                    args=args,
                    self_rag_cfg=self_rag_cfg,
                    self_rag_llm=llm,
                    paths=paths,
                    persona_block=args.persona_self_rag_block,
                    wave_idx=None,
                )
                if not isinstance(ctx, dict):
                    return {"ok": False, "reason": "selfrag_returned_non_dict"}

                rec = selfrag_q.get_record(subject, hop)
                ps = rec.get("parent_subject") if isinstance(rec, dict) else None
                ph = rec.get("parent_hop") if isinstance(rec, dict) else None
                writers["self_rag_log"].append({"subject": subject, "hop": hop, "context": ctx})
                _pq_payload_write(paths, "selfrag", subject, hop, {"subject": subject, "hop": hop, "context": ctx, "parent_subject": ps, "parent_hop": ph})
                return {"ok": True}
            except Exception as e:
                _append_error_log(paths, f"[pq-selfrag] error={type(e).__name__}: {e!r}", subject=subject, hop=hop, exc=e)
                return {"ok": False, "reason": f"selfrag:{type(e).__name__}"}

        def handle(job, result):
            subject, hop = job
            if result and result.get("ok"):
                selfrag_q.mark_done(subject, hop)
                sr_payload = _pq_payload_read(paths, "selfrag", subject, hop) or {}
                ps = sr_payload.get("parent_subject")
                ph = sr_payload.get("parent_hop")
                _pq_try_enqueue_elicit(args, paths, elicit_q, subject, hop, parent_subject=ps, parent_hop=ph)
            else:
                reason = (result or {}).get("reason", "selfrag_failed")
                selfrag_q.mark_error(subject, hop, args.max_retries, reason=reason)

        def stop():
            return stop_event.is_set() or prereq_capped_event.is_set()

        worker_loop_with_inflight(
            name=name,
            pop_job_fn=pop_job,
            call_fn=call,
            handle_result_fn=handle,
            stop_fn=stop,
            api_sema=api_sema,
            inflight_per_worker=args.inflight_per_worker,
            poll_sleep=0.1,
        )


    def outline_worker_loop(worker_id: int):
        if not _pq_stage_enabled_outline(args):
            return
        llm = make_llm_from_config(el_cfg)
        name = f"pq-outline-{worker_id}"

        def pop_job():
            if _stage_should_stop(outline_q, args, f"outline-{worker_id}"):
                return None
            batch = outline_q.claim_pending_batch(args.max_depth, 1)
            return batch[0] if batch else None

        def call(job):
            subject, hop = job

            if _pq_payload_exists(paths, "outline", subject, hop):
                return {"ok": True, "skipped": True}

            try:
                root_topic = args.seed if args.domain == "topic" else subject
                outline_text = _get_outline_for_subject(
                    subject=subject,
                    hop=hop,
                    args=args,
                    root_topic=root_topic,
                    persona_block=args.persona_elicit_block,
                    el_llm=llm,
                )
                outline_text = (outline_text or "").strip()
                rec = outline_q.get_record(subject, hop)
                ps = rec.get("parent_subject") if isinstance(rec, dict) else None
                ph = rec.get("parent_hop") if isinstance(rec, dict) else None
                if outline_text:
                    writers["outlines"].append({"subject": subject, "hop": hop, "outline": outline_text})

                _pq_payload_write(paths, "outline", subject, hop, {"subject": subject, "hop": hop, "outline": outline_text, "parent_subject": ps, "parent_hop": ph})
                return {"ok": True}
            except Exception as e:
                _append_error_log(paths, f"[pq-outline] error={type(e).__name__}: {e!r}", subject=subject, hop=hop, exc=e)
                return {"ok": False, "reason": f"outline:{type(e).__name__}"}

        def handle(job, result):
            subject, hop = job
            if result and result.get("ok"):
                outline_q.mark_done(subject, hop)
                ol_payload = _pq_payload_read(paths, "outline", subject, hop) or {}
                ps = ol_payload.get("parent_subject")
                ph = ol_payload.get("parent_hop")
                _pq_try_enqueue_elicit(args, paths, elicit_q, subject, hop, parent_subject=ps, parent_hop=ph)
            else:
                reason = (result or {}).get("reason", "outline_failed")
                outline_q.mark_error(subject, hop, args.max_retries, reason=reason)

        def stop():
            return stop_event.is_set() or prereq_capped_event.is_set()

        worker_loop_with_inflight(
            name=name,
            pop_job_fn=pop_job,
            call_fn=call,
            handle_result_fn=handle,
            stop_fn=stop,
            api_sema=api_sema,
            inflight_per_worker=args.inflight_per_worker,
            poll_sleep=0.1,
        )


    def elicit_worker_loop(worker_id: int):
        llm = make_llm_from_config(el_cfg)
        name = f"pq-elicit-{worker_id}"

        def pop_job():
            if _stage_should_stop(elicit_q, args, f"elicit-{worker_id}"):
                return None
            batch = elicit_q.claim_pending_batch(args.max_depth, 1)
            return batch[0] if batch else None

        def call(job):
            subject, hop = job

            if _pq_payload_exists(paths, "elicit", subject, hop):
                return {"ok": True, "skipped": True}

            if not _pq_prereq_selfrag_done(args, paths, subject, hop) or not _pq_prereq_outline_done(args, paths, subject, hop):
                return {"ok": False, "reason": "elicit_missing_prereq_payload"}

            try:
                root_topic = args.seed if args.domain == "topic" else subject

                # Get parent info from elicit queue record
                rec = elicit_q.get_record(subject, hop)
                ps = rec.get("parent_subject") if isinstance(rec, dict) else None
                ph = rec.get("parent_hop") if isinstance(rec, dict) else None

                sr_payload = _pq_payload_read(paths, "selfrag", subject, hop) or {}
                ol_payload = _pq_payload_read(paths, "outline", subject, hop) or {}

                self_rag_context = sr_payload.get("context") if isinstance(sr_payload, dict) else None
                outline_text = ol_payload.get("outline") if isinstance(ol_payload, dict) else None

                msgs = _build_llmpedia_messages_for_subject(
                    subject=subject,
                    hop=hop,
                    args=args,
                    root_topic=root_topic,
                    persona_block=args.persona_elicit_block,
                    self_rag_context=(self_rag_context if isinstance(self_rag_context, dict) else None),
                    outline=(outline_text if isinstance(outline_text, str) else None),
                )

                try:
                    resp = llm(msgs, timeout=args.timeout)
                except TypeError:
                    resp = llm(msgs)

                wikitext = (_unwrap_text(resp) or "").strip()
                if _is_no_article_content(wikitext, subject):
                    return {"ok": False, "reason": "no_article_content_generated"}

                candidate_phrases, parent_intro = _store_article_outputs_and_get_candidates_buffered(
                    args=args,
                    paths=paths,
                    el_cfg=el_cfg,
                    subject=subject,
                    hop=hop,
                    wikitext=wikitext,
                    writers=writers,
                    parent_subject=ps,
                    parent_hop=ph,
                )

                _pq_payload_write(
                    paths, "elicit", subject, hop,
                    {"subject": subject, "hop": hop, "candidate_phrases": candidate_phrases, "parent_intro": parent_intro, "parent_subject": ps, "parent_hop": ph}
                )
                return {"ok": True}

            except Exception as e:
                _append_error_log(paths, f"[pq-elicit] error={type(e).__name__}: {e!r}", subject=subject, hop=hop, exc=e)
                return {"ok": False, "reason": f"elicit:{type(e).__name__}"}

        def handle(job, result):
            subject, hop = job
            if result and result.get("ok"):
                if result.get("skipped"):
                    return
                elicit_q.mark_done(subject, hop)

                with _queue_lock:
                    queue.mark_done(subject, hop)

                el_payload = _pq_payload_read(paths, "elicit", subject, hop) or {}
                ps = el_payload.get("parent_subject")
                ph = el_payload.get("parent_hop")

                if bool(getattr(args, "use_ner", False)):
                    _pq_stage_enqueue_with_parent(paths, ner_q, subject, hop, ps, ph)
                else:
                    _pq_stage_enqueue_with_parent(paths, sim_q, subject, hop, ps, ph)
            else:
                reason = (result or {}).get("reason", "elicit_failed")
                elicit_q.mark_error(subject, hop, args.max_retries, reason=reason)

        def stop():
            if stop_event.is_set():
                return True
            # ✅ Elicit stops when elicit (done + failed) >= max_subjects
            if args.max_subjects:
                d, w, p, f = elicit_q.metrics(args.max_depth)
                if (d + f) >= args.max_subjects:
                    return True
            return False

        worker_loop_with_inflight(
            name=name,
            pop_job_fn=pop_job,
            call_fn=call,
            handle_result_fn=handle,
            stop_fn=stop,
            api_sema=api_sema,
            inflight_per_worker=args.inflight_per_worker,
            poll_sleep=0.1,
        )


    
    
    def ner_worker_loop(worker_id: int):
        if not bool(getattr(args, "use_ner", False)):
            return
        llm = make_llm_from_config(ner_cfg)
        name = f"pq-ner-{worker_id}"

        def pop_job():
            if _stage_should_stop(ner_q, args, f"ner-{worker_id}"):
                return None
            batch = ner_q.claim_pending_batch(args.max_depth, 1)
            return batch[0] if batch else None

        def call(job):
            subject, hop = job

            if _pq_payload_exists(paths, "ner", subject, hop):
                return {"ok": True, "skipped": True}

            el_payload = _pq_payload_read(paths, "elicit", subject, hop)
            if not isinstance(el_payload, dict):
                return {"ok": False, "reason": "ner_missing_elicit_payload"}

            cands = el_payload.get("candidate_phrases") or []
            if not isinstance(cands, list):
                cands = []

            # ═══════════════════════════════════════════════════════════════
            #  Do canonical dedup + plural variant rejection BEFORE NER
            # This saves NER API calls on candidates we already have!
            # ═══════════════════════════════════════════════════════════════
            pre_dedup_count = len(cands)
            deduped_cands: List[str] = []
            uniq_canon: Set[str] = set()
            
            dedup_seen_count = 0
            dedup_batch_count = 0
            plural_variant_count = 0
            
            for s in cands:
                canon = canon_key_from_queue(s)
                if not canon:
                    continue
                
                # Plural variant rejection vs GLOBAL seen
                with _seen_canon_lock:
                    is_pldup, matched = _is_plural_variant_duplicate(canon, seen_canon_keys)
                if is_pldup:
                    plural_variant_count += 1
                    writers["plural_s_dedup"].append({
                        "stage": "plural_variant_reject_pre_ner",
                        "current_entity": subject,
                        "hop": hop,
                        "phrase": s,
                        "canonical_key": canon,
                        "matched_variant": matched,
                    })
                    continue
                
                # Plural variant rejection vs BATCH (within this article)
                is_batch_pldup, batch_matched = _is_plural_variant_duplicate(canon, uniq_canon)
                if is_batch_pldup:
                    plural_variant_count += 1
                    writers["plural_s_dedup"].append({
                        "stage": "plural_variant_reject_batch_pre_ner",
                        "current_entity": subject,
                        "hop": hop,
                        "phrase": s,
                        "canonical_key": canon,
                        "matched_variant": batch_matched,
                    })
                    continue
                
                # Global seen rejection
                with _seen_canon_lock:
                    if canon in seen_canon_keys:
                        dedup_seen_count += 1
                        writers["ner_lowconf"].append({
                            "stage": "queue_dedup_pre_ner",
                            "current_entity": subject,
                            "hop": hop,
                            "phrase": s,
                            "canonical_key": canon,
                            "rejection_reason": "queue_canonical_seen_pre_ner",
                        })
                        continue
                
                # Within-batch dedup (same article)
                if canon in uniq_canon:
                    dedup_batch_count += 1
                    writers["ner_lowconf"].append({
                        "stage": "queue_dedup_batch_pre_ner",
                        "current_entity": subject,
                        "hop": hop,
                        "phrase": s,
                        "canonical_key": canon,
                        "rejection_reason": "queue_batch_duplicate_pre_ner",
                    })
                    continue
                
                uniq_canon.add(canon)
                deduped_cands.append(s)
            
            post_dedup_count = len(deduped_cands)
            
            # Log the pre-NER dedup savings
            writers["ner_decisions"].append({
                "stage": "pre_ner_dedup_summary",
                "current_entity": subject,
                "hop": hop,
                "pre_dedup_count": pre_dedup_count,
                "post_dedup_count": post_dedup_count,
                "filtered_count": pre_dedup_count - post_dedup_count,
                "dedup_seen_count": dedup_seen_count,
                "dedup_batch_count": dedup_batch_count,
                "plural_variant_count": plural_variant_count,
            })
            
            # ═══════════════════════════════════════════════════════════════
            # If all candidates were filtered by dedup - no NER needed!
            # ═══════════════════════════════════════════════════════════════
            if not deduped_cands:
                el_payload_p = _pq_payload_read(paths, "elicit", subject, hop) or {}
                _ps = el_payload_p.get("parent_subject")
                _ph = el_payload_p.get("parent_hop")
                _pq_payload_write(paths, "ner", subject, hop, {
                    "subject": subject, 
                    "hop": hop, 
                    "kept": [],
                    "pre_dedup_count": pre_dedup_count,
                    "post_dedup_count": 0,
                    "ner_skipped": True,
                    "parent_subject": _ps,
                    "parent_hop": _ph,
                })
                return {"ok": True}

            # ═══════════════════════════════════════════════════════════════
            # Now run NER only on deduped candidates (much fewer!)
            # ═══════════════════════════════════════════════════════════════
            try:
                kept = _run_ner_strict_gate_buffered(
                    args=args,
                    paths=paths,
                    subject=subject,
                    hop=hop,
                    candidate_phrases=deduped_cands,  # Only deduped candidates!
                    ner_llm=llm,
                    writers=writers,
                )
                el_payload_p = _pq_payload_read(paths, "elicit", subject, hop) or {}
                _ps = el_payload_p.get("parent_subject")
                _ph = el_payload_p.get("parent_hop")
                _pq_payload_write(paths, "ner", subject, hop, {
                    "subject": subject, 
                    "hop": hop, 
                    "kept": kept,
                    "pre_dedup_count": pre_dedup_count,
                    "post_dedup_count": post_dedup_count,
                    "parent_subject": _ps,
                    "parent_hop": _ph,
                })
                return {"ok": True}
            except Exception as e:
                _append_error_log(paths, f"[pq-ner] error={type(e).__name__}: {e!r}", subject=subject, hop=hop, exc=e)
                return {"ok": False, "reason": f"ner:{type(e).__name__}"}

        def handle(job, result):
            subject, hop = job
            if result and result.get("ok"):
                ner_q.mark_done(subject, hop)
                el_payload = _pq_payload_read(paths, "elicit", subject, hop) or {}
                ps = el_payload.get("parent_subject")
                ph = el_payload.get("parent_hop")
                _pq_stage_enqueue_with_parent(paths, sim_q, subject, hop, ps, ph)
            else:
                reason = (result or {}).get("reason", "ner_failed")
                ner_q.mark_error(subject, hop, args.max_retries, reason=reason)

        def stop():
            return stop_event.is_set() or prereq_capped_event.is_set()

        worker_loop_with_inflight(
            name=name,
            pop_job_fn=pop_job,
            call_fn=call,
            handle_result_fn=handle,
            stop_fn=stop,
            api_sema=api_sema,
            inflight_per_worker=args.inflight_per_worker,
            poll_sleep=0.1,
        )

    def sim_worker_loop(worker_id: int = 0):
        similarity_mode = (getattr(args, "similarity_mode", "batch") or "batch").lower()
        batch_n = max(1, int(getattr(args, "batch_size", 50) or 50)) if similarity_mode == "batch" else 1

        client = OpenAI() if similarity_mode == "batch" else None

        similarity_cfg = None
        if bool(getattr(args, "use_similarity", False)):
            filt_key = getattr(args, "similarity_filter_model_key", None) or getattr(args, "ner_model_key", None)
            if filt_key:
                similarity_cfg = _cfg_from_key(filt_key, args.timeout)
                _strip_responses_sampling_if_disallowed(similarity_cfg)

        wave_counter = [0]

        while not stop_event.is_set() and not prereq_capped_event.is_set():
            if _stage_should_stop(sim_q, args, f"sim-{worker_id}"):
                break
                
            batch = sim_q.claim_pending_batch(args.max_depth, batch_n)
            if not batch:
                _pq_sleep_on_retry_due(sim_q, args.max_depth)
                continue

            all_items_for_similarity: List[dict] = []
            subject_data: Dict[Tuple[str, int], dict] = {}
            seen_cands_wave: Set[str] = set()

            for subject, hop in batch:
                el_payload = _pq_payload_read(paths, "elicit", subject, hop)
                if not isinstance(el_payload, dict):
                    sim_q.mark_error(subject, hop, args.max_retries, reason="sim_missing_elicit_payload")
                    continue

                parent_intro = el_payload.get("parent_intro") or ""

                # ═══════════════════════════════════════════════════════════
                # Get candidates - NER already did dedup, so just use kept!
                # ═══════════════════════════════════════════════════════════
                if bool(getattr(args, "use_ner", False)):
                    ner_payload = _pq_payload_read(paths, "ner", subject, hop)
                    if not (isinstance(ner_payload, dict) and isinstance(ner_payload.get("kept"), list)):
                        el_payload_p = _pq_payload_read(paths, "elicit", subject, hop) or {}
                        _ps = el_payload_p.get("parent_subject")
                        _ph = el_payload_p.get("parent_hop")
                        _pq_stage_enqueue_with_parent(paths, ner_q, subject, hop, _ps, _ph)
                        sim_q.mark_error(subject, hop, args.max_retries, reason="sim_missing_ner_payload")
                        continue
                    # NER already deduped these - use directly!
                    next_subjects = ner_payload["kept"] or []
                else:
                    # NER disabled - we need to do dedup here
                    cands = el_payload.get("candidate_phrases") or []
                    next_subjects = []
                    uniq_canon: Set[str] = set()
                    
                    for s in cands:
                        canon = canon_key_from_queue(s)
                        if not canon:
                            continue
                        
                        with _seen_canon_lock:
                            is_pldup, matched = _is_plural_variant_duplicate(canon, seen_canon_keys)
                        if is_pldup:
                            writers["plural_s_dedup"].append({
                                "stage": "plural_variant_reject",
                                "current_entity": subject,
                                "hop": hop,
                                "phrase": s,
                                "canonical_key": canon,
                                "matched_variant": matched,
                            })
                            continue
                        
                        is_batch_pldup, batch_matched = _is_plural_variant_duplicate(canon, uniq_canon)
                        if is_batch_pldup:
                            writers["plural_s_dedup"].append({
                                "stage": "plural_variant_reject_batch",
                                "current_entity": subject,
                                "hop": hop,
                                "phrase": s,
                                "canonical_key": canon,
                                "matched_variant": batch_matched,
                            })
                            continue
                        
                        with _seen_canon_lock:
                            if canon in seen_canon_keys:
                                writers["ner_lowconf"].append({
                                    "stage": "queue_dedup",
                                    "current_entity": subject,
                                    "hop": hop,
                                    "phrase": s,
                                    "canonical_key": canon,
                                    "rejection_reason": "queue_canonical_seen",
                                })
                                continue
                        
                        if canon in uniq_canon:
                            continue
                        
                        uniq_canon.add(canon)
                        next_subjects.append(s)

                subject_data[(subject, hop)] = {"parent_intro": parent_intro, "unique_next": next_subjects}

                # Build similarity check list (deduped globally within wave)
                for cand in next_subjects:
                    if cand in seen_cands_wave:
                        continue
                    seen_cands_wave.add(cand)
                    all_items_for_similarity.append({
                        "candidate": cand,
                        "parent_entity": subject,
                        "parent_intro": parent_intro,
                        "hop": hop,
                    })

            if not all_items_for_similarity:
                for subject, hop in batch:
                    if (subject, hop) in subject_data:
                        sim_q.mark_done(subject, hop)
                continue

            # Run similarity filter (unified function handles both modes)
            kept_candidates: Set[str] = set()
            kept_meta_all: Dict[str, dict] = {}

            if bool(getattr(args, "use_similarity", False)):
                sim = _get_similarity_engine(args, paths)
                if sim is not None and similarity_cfg is not None:
                    wave_counter[0] += 1
                    # UNIFIED CALL - works for both online and batch
                    kept_list, kept_meta_all = sim.filter_candidates_batch(
                        items=all_items_for_similarity,
                        similarity_cfg=similarity_cfg,
                        client=client,
                        wave_idx=wave_counter[0],
                        writers=writers,  # Pass writers for buffered logging
                    )
                    kept_candidates = set(kept_list)
                else:
                    kept_candidates = set(it["candidate"] for it in all_items_for_similarity)
            else:
                kept_candidates = set(it["candidate"] for it in all_items_for_similarity)

            # Enqueue survivors
            for subject, hop in batch:
                sd = subject_data.get((subject, hop))
                if sd is None:
                    continue

                unique_next = sd["unique_next"]
                parent_intro = sd["parent_intro"]
                final_next = [c for c in unique_next if c in kept_candidates]

                inserted_count = 0

                # ── PRELOAD-ONLY: no expansion, just log and skip ──
                if not getattr(args, "preload_only", False):
                    for s in final_next:
                        if args.max_depth != 0 and hop + 1 > args.max_depth:
                            continue

                        with _queue_lock:
                            _, kept_hop, outcome = queue.enqueue(
                                s, hop + 1,
                                parent_subject=subject,
                                parent_hop=hop,
                            )

                        if outcome == "inserted":
                            inserted_count += 1
                            writers["queue"].append({
                                "subject": s,
                                "hop": kept_hop,
                                "event": outcome,
                                "parent_subject": subject,
                                "parent_hop": hop,
                            })

                            with _seen_canon_lock:
                                ck2 = canon_key_from_queue(s)
                                if ck2:
                                    seen_canon_keys.add(ck2)

                            if bool(getattr(args, "use_similarity", False)):
                                sim = _get_similarity_engine(args, paths)
                                if sim is not None:
                                    meta = kept_meta_all.get(s)
                                    if isinstance(meta, dict):
                                        meta["hop"] = kept_hop
                                        meta["parent_subject"] = subject
                                        meta["parent_hop"] = hop
                                        meta["parent_intro"] = parent_intro
                                        sim.commit_embedding_if_inserted(meta, writers=writers)

                sim_q.mark_done(subject, hop)
    
    # -----------------------------
    # SYNCHRONOUS PRE-DISPATCH: fill stage queues BEFORE workers start
    # -----------------------------
    def _synchronous_predispatch():
        """Claim ALL pending from main queue → push to stage queues."""
        total_dispatched = 0
        while True:
            with _queue_lock:
                d, w, p, f = queue.metrics(args.max_depth)
            if args.max_subjects and (d + w) >= args.max_subjects:
                break
            claim_n = max(1, int(getattr(args, "batch_size", 500) or 500))
            if args.max_subjects:
                remaining = args.max_subjects - (d + w)
                if remaining <= 0:
                    break
                claim_n = min(claim_n, remaining)
            with _queue_lock:
                batch = queue.claim_pending_batch(args.max_depth, claim_n)
            if not batch:
                break
            for subject, hop in batch:
                rec = queue.get_record(subject, hop)
                ps = rec.get("parent_subject") if isinstance(rec, dict) else None
                ph = rec.get("parent_hop") if isinstance(rec, dict) else None
                if _pq_stage_enabled_selfrag(args):
                    _pq_stage_enqueue_with_parent(paths, selfrag_q, subject, hop, ps, ph)
                else:
                    if not _pq_payload_exists(paths, "selfrag", subject, hop):
                        _pq_payload_write(paths, "selfrag", subject, hop,
                            {"subject": subject, "hop": hop, "context": None, "skipped": True, "parent_subject": ps, "parent_hop": ph})
                if _pq_stage_enabled_outline(args):
                    _pq_stage_enqueue_with_parent(paths, outline_q, subject, hop, ps, ph)
                else:
                    if not _pq_payload_exists(paths, "outline", subject, hop):
                        _pq_payload_write(paths, "outline", subject, hop,
                            {"subject": subject, "hop": hop, "outline": None, "skipped": True, "parent_subject": ps, "parent_hop": ph})
                _pq_try_enqueue_elicit(args, paths, elicit_q, subject, hop, parent_subject=ps, parent_hop=ph)
            total_dispatched += len(batch)
        if total_dispatched:
            _dbg(f"[pq-predispatch] Dispatched {total_dispatched} items to stage queues before starting workers")

    _synchronous_predispatch()

    # -----------------------------
    # Start threads
    # -----------------------------
    threads: List[threading.Thread] = []
    threads.append(threading.Thread(target=dispatcher_loop, name="pq-dispatcher", daemon=True))
    threads.append(threading.Thread(target=reconciler_loop, name="pq-reconciler", daemon=True))

    for i in range(max(0, int(getattr(args, "selfrag_workers", 0) or 0))):
        threads.append(threading.Thread(target=selfrag_worker_loop, args=(i,), name=f"pq-selfrag-{i}", daemon=True))
    for i in range(max(0, int(getattr(args, "outline_workers", 0) or 0))):
        threads.append(threading.Thread(target=outline_worker_loop, args=(i,), name=f"pq-outline-{i}", daemon=True))
    for i in range(max(0, int(getattr(args, "elicit_workers", 0) or 0))):
        threads.append(threading.Thread(target=elicit_worker_loop, args=(i,), name=f"pq-elicit-{i}", daemon=True))
    for i in range(max(0, int(getattr(args, "ner_workers", 0) or 0))):
        threads.append(threading.Thread(target=ner_worker_loop, args=(i,), name=f"pq-ner-{i}", daemon=True))
    for i in range(max(0, int(getattr(args, "sim_workers", 0) or 0))):
        threads.append(threading.Thread(target=sim_worker_loop, args=(i,), name=f"pq-sim-{i}", daemon=True))

    for t in threads:
        t.start()

    # -----------------------------
    # Monitor loop (progress + stop conditions)
    # -----------------------------
    start = time.perf_counter()
    last_progress_ts = 0.0

    def _all_stage_metrics():
        sr = (0, 0, 0, 0) if not _pq_stage_enabled_selfrag(args) else selfrag_q.metrics(args.max_depth)
        ol = (0, 0, 0, 0) if not _pq_stage_enabled_outline(args) else outline_q.metrics(args.max_depth)
        el = elicit_q.metrics(args.max_depth)
        ne = (0, 0, 0, 0) if not bool(getattr(args, "use_ner", False)) else ner_q.metrics(args.max_depth)
        si = sim_q.metrics(args.max_depth)
        return sr, ol, el, ne, si

    try:
        while not stop_event.is_set():  # ✅ CORRECT - checks stop_event
            now = time.perf_counter()

            if args.progress_metrics and (now - last_progress_ts) >= 30.0:
                with _queue_lock:
                    md, mw, mp, mf = queue.metrics(args.max_depth)
                sr, ol, el, ne, si = _all_stage_metrics()
                _dbg(
                    "[pq-progress] "
                    f"MAIN d={md} w={mw} p={mp} f={mf} | "
                    f"SR d={sr[0]} w={sr[1]} p={sr[2]} f={sr[3]} | "
                    f"OL d={ol[0]} w={ol[1]} p={ol[2]} f={ol[3]} | "
                    f"EL d={el[0]} w={el[1]} p={el[2]} f={el[3]} | "
                    f"NER d={ne[0]} w={ne[1]} p={ne[2]} f={ne[3]} | "
                    f"SIM d={si[0]} w={si[1]} p={si[2]} f={si[3]}"
                )
                last_progress_ts = now

            with _queue_lock:
                md, mw, mp, mf = queue.metrics(args.max_depth)

            sr, ol, el, ne, si = _all_stage_metrics()
            stages_empty = all((m[1] == 0 and m[2] == 0) for m in [sr, ol, el, ne, si])
            main_empty = (mw == 0 and mp == 0)

            # drain fully
            if main_empty and stages_empty:
                break

            # ✅ HARD STOP at cap (do NOT drain stage queues intentionally)
            if args.max_subjects and md >= args.max_subjects:
                # _dbg(f"[pq-stop] MAIN done reached cap ({md}); stopping immediately (not draining NER/SIM).")
                stop_event.set()
                break

            time.sleep(0.2)

    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=1.0)

        # ---- FLUSH ALL JSONL WRITERS ----
        _dbg("[pq-flush] flushing all buffered JSONL writes...")
        for name, writer in writers.items():
            try:
                writer.flush()
            except Exception as e:
                _dbg(f"[pq-flush] {name}: error={e!r}")

        # ---- FLUSH ALL STAGE QUEUES ----
        _dbg("[pq-flush] flushing stage queues...")
        for q_name, q_obj in [
            ("selfrag", selfrag_q),
            ("outline", outline_q),
            ("elicit", elicit_q),
            ("ner", ner_q),
            ("sim", sim_q),
        ]:
            try:
                q_obj.flush()
            except Exception as e:
                _dbg(f"[pq-flush] {q_name}_q: error={e!r}")

        # ---- FLUSH MAIN QUEUE ----
        try:
            queue.flush()
        except Exception as e:
            _dbg(f"[pq-flush] main queue: error={e!r}")

        _persist_seen_canon(paths, seen_canon_keys)
        _snapshot_json_only(paths, queue)
    
    dur = time.perf_counter() - start
    _dbg(f"[done-pq] finished in {dur:.1f}s → {os.path.dirname(paths['queue_json'])}")

# ---------------- batch mode (JSON only) ----------------

def _handle_batch_wave_exception(
    e: Exception,
    wave_idx: int,
    batch: List[Tuple[str, int]],
    args,
    paths,
    seen_canon_keys: Set[str],
    context: str,
    queue: JsonQueue,
):
    msg = (
        f"[batch] wave {wave_idx} encountered an SDK-level error during {context}: "
        f"{type(e).__name__}: {e!r}; marking all {len(batch)} subjects with an error retry."
    )
    _dbg(msg)
    try:
        with open(paths["errors_log"], "a", encoding="utf-8") as ef:
            ef.write(
                f"[{datetime.datetime.now().isoformat()}] "
                f"[batch-wave={wave_idx}] context={context}\n{msg}\n"
                f"{traceback.format_exc()}\n"
            )
    except Exception:
        pass

    for subject, hop in batch:
        with _queue_lock:
            queue.mark_error(
                subject,
                hop,
                args.max_retries,
                reason=f"batch_wave_exception:{context}:{type(e).__name__}",
            )


    _persist_seen_canon(paths, seen_canon_keys)


def _endpoint_for_cfg(cfg) -> str:
    return "/v1/responses" if getattr(cfg, "use_responses_api", False) else "/v1/chat/completions"

def _build_openai_body_for_batch(cfg, messages: List[dict], *, max_tokens: int) -> dict:

    extra = getattr(cfg, "extra_inputs", None)
    extra = extra if isinstance(extra, dict) else {}

    if getattr(cfg, "use_responses_api", False):
        body = {"model": cfg.model, "input": messages}

        # Sampling is not allowed for some GPT-5 family models in Responses API
        if _responses_sampling_allowed(cfg):
            if getattr(cfg, "temperature", None) is not None:
                body["temperature"] = cfg.temperature
            if getattr(cfg, "top_p", None) is not None:
                body["top_p"] = cfg.top_p

        if max_tokens is not None:
            body["max_output_tokens"] = int(max_tokens)

        # merge extra inputs (reasoning/text/format/etc)
        if extra:
            body.update(extra)

        return body

    # Chat Completions
    body = {
        "model": cfg.model,
        "messages": messages,
        "max_tokens": int(max_tokens),
    }
    if getattr(cfg, "temperature", None) is not None:
        body["temperature"] = cfg.temperature
    if getattr(cfg, "top_p", None) is not None:
        body["top_p"] = cfg.top_p

    if extra:
        body.update(extra)

    return body



from concurrent.futures import ThreadPoolExecutor, as_completed
def _store_article_outputs_and_get_candidates(
    *,
    args,
    paths,
    el_cfg,
    subject: str,
    hop: int,
    wikitext: str,
    parent_subject: Optional[str] = None,
    parent_hop: Optional[int] = None,
) -> Tuple[List[str], str]:
    """Store article outputs (WITH parent pointers) and return (candidate_phrases, parent_intro_excerpt)."""

    write_article_record_jsonl(
        paths["articles_jsonl"],
        subject=subject,
        hop=hop,
        model=el_cfg.model,
        wikitext=wikitext,
        overall_confidence=None,
        parent_subject=parent_subject,
        parent_hop=parent_hop,
    )

    _append_jsonl(
        paths["articles_wikitext_jsonl"],
        {
            "subject": subject,
            "hop": hop,
            "wikitext": wikitext,
            "parent_subject": parent_subject,
            "parent_hop": parent_hop,
        },
    )

    links_from_markup = _extract_link_targets_from_wikitext_ignoring_sections(wikitext)
    cat_from_markup = _extract_categories_from_wikitext(wikitext)

    _append_jsonl(
        paths["articles_meta_jsonl"],
        {
            "subject": subject,
            "hop": hop,
            "links": links_from_markup,
            "categories": cat_from_markup,
            "parent_subject": parent_subject,
            "parent_hop": parent_hop,
        },
    )

    candidates_for_ner: List[str] = []
    seen_candidates: Set[str] = set()

    def _add_candidate(candidate: str):
        c = (candidate or "").strip()
        if c and c not in seen_candidates:
            seen_candidates.add(c)
            candidates_for_ner.append(c)

    elicit_conf_th = float(getattr(args, "elicit_conf_threshold", 0.0) or 0.0)
    for raw_title in links_from_markup:
        base_title, link_conf = _split_title_and_conf(raw_title)
        if elicit_conf_th > 0.0 and isinstance(link_conf, float) and link_conf < elicit_conf_th:
            _append_jsonl(
                paths["elicit_lowconf_jsonl"],
                {
                    "stage": "elicitation_link_filter",
                    "current_entity": subject,
                    "root_subject": args.seed if args.domain == "topic" else None,
                    "hop": hop,
                    "phrase": base_title,
                    "elicitation_confidence": float(link_conf),
                    "elicit_conf_threshold": float(elicit_conf_th),
                    "passed_threshold": False,
                    "rejection_reason": "elicitation_below_conf_threshold",
                    "parent_subject": parent_subject,
                    "parent_hop": parent_hop,
                },
            )
            continue
        _add_candidate(base_title)

    for c in cat_from_markup:
        _add_candidate(c)

    candidate_phrases = [c for c in candidates_for_ner if c and c.strip()]

    _append_jsonl(
        paths["articles_meta_jsonl"],
        {
            "subject": subject,
            "hop": hop,
            "parent_subject": parent_subject,
            "parent_hop": parent_hop,
            "links_from_markup": links_from_markup[:1000],
            "categories_from_markup": cat_from_markup[:1000],
            "ner_candidates": candidate_phrases[:1000],
            "expanded_next_subjects": [],
            "diag": True,
            "stage": "batch_store_extract",
        },
    )

    parent_intro = _extract_intro_excerpt_words(
        wikitext,
        max_words=int(getattr(args, "similarity_parent_context_words", 100) or 100),
    )

    return candidate_phrases, parent_intro

def _store_article_outputs_and_get_candidates_buffered(
    *,
    args,
    paths,
    el_cfg,
    subject: str,
    hop: int,
    wikitext: str,
    writers: dict,
    parent_subject: Optional[str] = None,
    parent_hop: Optional[int] = None,
) -> Tuple[List[str], str]:
    """Buffered version of _store_article_outputs_and_get_candidates"""

    writers["articles"].append({
        "subject": subject,
        "hop": hop,
        "model": el_cfg.model,
        "wikitext": wikitext,
        "overall_confidence": None,
        "parent_subject": parent_subject,
        "parent_hop": parent_hop,
    })

    writers["articles_wikitext"].append({
        "subject": subject,
        "hop": hop,
        "wikitext": wikitext,
        "parent_subject": parent_subject,
        "parent_hop": parent_hop,
    })

    links_from_markup = _extract_link_targets_from_wikitext_ignoring_sections(wikitext)
    cat_from_markup = _extract_categories_from_wikitext(wikitext)

    writers["articles_meta"].append({
        "subject": subject,
        "hop": hop,
        "links": links_from_markup,
        "categories": cat_from_markup,
        "parent_subject": parent_subject,
        "parent_hop": parent_hop,
    })

    candidates_for_ner: List[str] = []
    seen_candidates: Set[str] = set()

    def _add_candidate(candidate: str):
        c = (candidate or "").strip()
        if c and c not in seen_candidates:
            seen_candidates.add(c)
            candidates_for_ner.append(c)

    elicit_conf_th = float(getattr(args, "elicit_conf_threshold", 0.0) or 0.0)
    for raw_title in links_from_markup:
        base_title, link_conf = _split_title_and_conf(raw_title)
        if elicit_conf_th > 0.0 and isinstance(link_conf, float) and link_conf < elicit_conf_th:
            writers["elicit_lowconf"].append({
                "stage": "elicitation_link_filter",
                "current_entity": subject,
                "root_subject": args.seed if args.domain == "topic" else None,
                "hop": hop,
                "phrase": base_title,
                "elicitation_confidence": float(link_conf),
                "elicit_conf_threshold": float(elicit_conf_th),
                "passed_threshold": False,
                "rejection_reason": "elicitation_below_conf_threshold",
                "parent_subject": parent_subject,
                "parent_hop": parent_hop,
            })
            continue
        _add_candidate(base_title)

    for c in cat_from_markup:
        _add_candidate(c)

    candidate_phrases = [c for c in candidates_for_ner if c and c.strip()]

    writers["articles_meta"].append({
        "subject": subject,
        "hop": hop,
        "parent_subject": parent_subject,
        "parent_hop": parent_hop,
        "links_from_markup": links_from_markup[:1000],
        "categories_from_markup": cat_from_markup[:1000],
        "ner_candidates": candidate_phrases[:1000],
        "expanded_next_subjects": [],
        "diag": True,
        "stage": "pq_elicit_extract",
    })

    parent_intro = _extract_intro_excerpt_words(
        wikitext,
        max_words=int(getattr(args, "similarity_parent_context_words", 100) or 100),
    )

    return candidate_phrases, parent_intro


def _run_ner_strict_gate_buffered(
    *,
    args,
    paths,
    subject: str,
    hop: int,
    candidate_phrases: List[str],
    ner_llm,
    writers: dict,
) -> List[str]:
    """Buffered version - uses writers dict instead of _append_jsonl"""
    
    # Just call the normal one but pass a modified paths that writes to buffers
    # Actually, easier to inline it here with buffered writes
    
    CHUNK_SIZE = int(getattr(args, "ner_chunk_size", 25) or 25)
    conf_th = float(getattr(args, "ner_conf_threshold", 0.0) or 0.0)

    ner_strategy = str(getattr(args, "ner_strategy", "") or "").strip().lower()
    use_confidence = ("calib" in ner_strategy)

    root_subject = args.seed if args.domain == "topic" else None
    root_topic = args.seed if args.domain == "topic" else subject

    if not candidate_phrases:
        writers["ner_lowconf"].append({
            "stage": "ner_filter",
            "current_entity": subject,
            "root_subject": root_subject,
            "hop": hop,
            "rejection_reason": "no_candidates",
            "use_confidence": bool(use_confidence),
        })
        return []

    canon_to_phrase: Dict[str, str] = {}
    ordered_canons: List[str] = []
    for p in candidate_phrases:
        s = (p or "").strip()
        if not s:
            continue
        ck = canon_key_from_queue(s)
        if not ck:
            continue
        if ck not in canon_to_phrase:
            canon_to_phrase[ck] = s
            ordered_canons.append(ck)

    candidates = [canon_to_phrase[ck] for ck in ordered_canons]
    if not candidates:
        return []

    model_by_canon: Dict[str, dict] = {}
    parse_modes: List[str] = []
    total_model_decisions = 0

    for chunk_idx, start in enumerate(range(0, len(candidates), CHUNK_SIZE), 1):
        chunk = candidates[start : start + CHUNK_SIZE]
        if not chunk:
            continue

        ner_messages = build_ner_messages_for_phrases(
            domain=args.domain,
            strategy=args.ner_strategy,
            subject_name=subject,
            phrases=chunk,
            seed=args.seed,
            root_topic=root_topic,
            persona_block=args.persona_ner_block,
        )

        try:
            try:
                resp = ner_llm(ner_messages, timeout=args.timeout)
            except TypeError:
                resp = ner_llm(ner_messages)
            
            # Save raw response for cost tracking (online mode)
            if "ner_responses" in writers:
                try:
                    raw_resp = _coerce_jsonable(resp)
                    writers["ner_responses"].append({
                        "subject": subject,
                        "hop": hop,
                        "chunk_idx": chunk_idx,
                        "chunk_start": start,
                        "chunk_size": len(chunk),
                        "timestamp": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                        "response": raw_resp,
                    })
                except Exception:
                    pass  # Don't let logging failures break NER
            
            txt = _unwrap_text(resp).strip()
        except Exception as e:
            writers["ner_decisions"].append({
                "stage": "ner_run_summary",
                "current_entity": subject,
                "root_subject": root_subject,
                "hop": hop,
                "chunk_idx": chunk_idx,
                "chunk_start": start,
                "chunk_size": len(chunk),
                "parse_mode": "exception",
                "num_candidates": len(chunk),
                "num_model_decisions": 0,
                "raw_preview": "",
                "error": f"{type(e).__name__}: {e!r}",
                "use_confidence": bool(use_confidence),
            })
            parse_modes.append("exception")
            continue

        decisions, parse_mode = _parse_ner_output(txt)
        parse_modes.append(parse_mode)
        total_model_decisions += len(decisions)

        writers["ner_decisions"].append({
            "stage": "ner_run_summary",
            "current_entity": subject,
            "root_subject": root_subject,
            "hop": hop,
            "chunk_idx": chunk_idx,
            "chunk_start": start,
            "chunk_size": len(chunk),
            "parse_mode": parse_mode,
            "num_candidates": len(chunk),
            "num_model_decisions": len(decisions),
            "raw_preview": txt[:800],
            "resp_type": str(type(resp)),
            "use_confidence": bool(use_confidence),
        })

        for d in decisions:
            rp = (d.get("phrase") or "").strip()
            if not rp:
                continue
            ck = canon_key_from_queue(rp)
            if not ck:
                continue
            if ck not in model_by_canon:
                model_by_canon[ck] = {
                    "raw_phrase": rp,
                    "is_ne": bool(d.get("is_ne")),
                    "confidence": (float(d["confidence"]) if isinstance(d.get("confidence"), (int, float)) else None),
                    "parse_mode": parse_mode,
                }

    writers["ner_decisions"].append({
        "stage": "ner_run_summary_global",
        "current_entity": subject,
        "root_subject": root_subject,
        "hop": hop,
        "chunk_size": CHUNK_SIZE,
        "parse_modes": parse_modes[:200],
        "num_candidates": len(candidates),
        "num_model_decisions": total_model_decisions,
        "use_confidence": bool(use_confidence),
        "ner_conf_threshold": conf_th,
    })

    kept: List[str] = []
    kept_canons: Set[str] = set()

    for ck in ordered_canons:
        phrase = canon_to_phrase[ck]
        md = model_by_canon.get(ck)

        if md is None:
            is_ne = False
            conf_val = None
            raw_phrase = None
            source = "missing"
            passed = False
            reason = "ner_parse_failed_or_empty"
            pm = "empty"
        else:
            is_ne = bool(md["is_ne"])
            conf_val = md.get("confidence", None)
            raw_phrase = md.get("raw_phrase")
            source = "model"
            pm = md.get("parse_mode", "unknown")

            if not use_confidence:
                passed = bool(is_ne)
                reason = "accepted" if passed else "not_named_entity"
            else:
                if conf_th > 0.0:
                    passed = bool(is_ne) and isinstance(conf_val, (int, float)) and float(conf_val) >= conf_th
                    if passed:
                        reason = "accepted"
                    elif not is_ne:
                        reason = "not_named_entity"
                    else:
                        reason = "below_conf_threshold_or_missing_confidence"
                else:
                    passed = bool(is_ne)
                    reason = "accepted" if passed else "not_named_entity"

        writers["ner_decisions"].append({
            "current_entity": subject,
            "root_subject": root_subject,
            "hop": hop,
            "phrase": phrase,
            "is_ne": bool(is_ne),
            "confidence": conf_val,
            "ner_conf_threshold": conf_th,
            "passed_threshold": bool(passed),
            "decision_reason": reason,
            "source": source,
            "parse_mode": pm,
            "raw_phrase": raw_phrase,
            "use_confidence": bool(use_confidence),
        })

        if not passed:
            writers["ner_lowconf"].append({
                "stage": "ner_filter",
                "current_entity": subject,
                "root_subject": root_subject,
                "hop": hop,
                "phrase": phrase,
                "is_ne": bool(is_ne),
                "confidence": conf_val,
                "ner_conf_threshold": conf_th,
                "passed_threshold": False,
                "rejection_reason": reason,
                "source": source,
                "parse_mode": pm,
                "raw_phrase": raw_phrase,
                "use_confidence": bool(use_confidence),
            })
        else:
            if ck not in kept_canons:
                kept_canons.add(ck)
                kept.append(phrase)

    return kept

def _dedupe_and_enqueue_next_subjects(
    *,
    args,
    paths,
    current_entity: str,
    hop: int,
    next_subjects: List[str],
    seen_canon_keys: Set[str],
    queue: "JsonQueue",
    parent_intro: str = "",
) -> List[str]:
    unique_next: List[str] = []
    if not next_subjects:
        return unique_next

    # (A) dedupe by plural/singular variants + canon + seen set
    uniq_canon: Set[str] = set()

    for s in next_subjects:
        canon = canon_key_from_queue(s)
        if not canon:
            continue

        # plural/singular variant reject vs GLOBAL seen
        with _seen_canon_lock:
            is_pldup, matched = _is_plural_variant_duplicate(canon, seen_canon_keys)
        if is_pldup:
            _append_jsonl(
                paths["plural_s_dedup_jsonl"],
                {
                    "stage": "plural_variant_reject",
                    "current_entity": current_entity,
                    "hop": hop,
                    "phrase": s,
                    "canonical_key": canon,
                    "matched_variant": matched,
                    "parent_subject": current_entity,
                    "parent_hop": hop,
                },
            )
            continue

        # plural/singular variant reject vs BATCH set  
        is_batch_pldup, batch_matched = _is_plural_variant_duplicate(canon, uniq_canon)
        if is_batch_pldup:
            _append_jsonl(
                paths["plural_s_dedup_jsonl"],
                {
                    "stage": "plural_variant_reject_batch",
                    "current_entity": current_entity,
                    "hop": hop,
                    "phrase": s,
                    "canonical_key": canon,
                    "matched_variant": batch_matched,
                    "parent_subject": current_entity,
                    "parent_hop": hop,
                },
            )
            continue

        # global seen
        with _seen_canon_lock:
            if canon in seen_canon_keys:
                _append_jsonl(
                    paths["ner_lowconf_jsonl"],
                    {
                        "stage": "queue_dedup",
                        "current_entity": current_entity,
                        "root_subject": args.seed if args.domain == "topic" else None,
                        "hop": hop,
                        "phrase": s,
                        "canonical_key": canon,
                        "passed_threshold": False,
                        "rejection_reason": "queue_canonical_seen",
                        "parent_subject": current_entity,
                        "parent_hop": hop,
                    },
                )
                continue
            # seen_canon_keys.add(canon)

        # within-batch duplicate
        if canon in uniq_canon:
            _append_jsonl(
                paths["ner_lowconf_jsonl"],
                {
                    "stage": "queue_dedup_batch",
                    "current_entity": current_entity,
                    "root_subject": args.seed if args.domain == "topic" else None,
                    "hop": hop,
                    "phrase": s,
                    "canonical_key": canon,
                    "passed_threshold": False,
                    "rejection_reason": "queue_batch_duplicate",
                    "parent_subject": current_entity,
                    "parent_hop": hop,
                },
            )
            continue

        uniq_canon.add(canon)
        unique_next.append(s)

    # (B) similarity filter + meta
    kept_meta_by_subject: Dict[str, dict] = {}
    if bool(getattr(args, "use_similarity", False)) and unique_next:
        sim = _get_similarity_engine(args, paths)
        if sim is not None:
            similarity_filter_llm = None
            if (getattr(args, "similarity_action", "llm") or "llm").lower() == "llm":
                filt_key = getattr(args, "similarity_filter_model_key", None) or getattr(args, "ner_model_key", None)
                if filt_key:
                    filt_cfg = _cfg_from_key(filt_key, args.timeout)
                    _strip_responses_sampling_if_disallowed(filt_cfg)
                    similarity_filter_llm = make_llm_from_config(filt_cfg)

            unique_next, kept_meta_by_subject = sim.filter_candidates(
                parent_entity=current_entity,
                parent_intro=parent_intro,
                hop=hop,
                candidates=unique_next,
                similarity_filter_llm=similarity_filter_llm,
                timeout=args.timeout,
            )

    # (C) enqueue WITH parent pointers + commit embeddings when inserted
    for s in unique_next:
        if args.max_depth != 0 and hop + 1 > args.max_depth:
            continue

        with _queue_lock:
            _, kept_hop, outcome = queue.enqueue(
                s,
                hop + 1,
                parent_subject=current_entity,
                parent_hop=hop,
            )

        if outcome == "inserted":
            _append_jsonl(
                paths["queue_jsonl"],
                {
                    "subject": s,
                    "hop": kept_hop,
                    "event": outcome,
                    "parent_subject": current_entity,
                    "parent_hop": hop,
                },
            )
            # mark as seen ONLY after it actually entered the queue
            with _seen_canon_lock:
                ck2 = canon_key_from_queue(s)
                if ck2:
                    seen_canon_keys.add(ck2)


            if bool(getattr(args, "use_similarity", False)):
                sim = _get_similarity_engine(args, paths)
                if sim is not None:
                    meta = kept_meta_by_subject.get(s)
                    if isinstance(meta, dict):
                        meta["hop"] = kept_hop
                        meta["parent_subject"] = current_entity
                        meta["parent_hop"] = hop
                        meta["parent_intro"] = parent_intro or ""
                        sim.commit_embedding_if_inserted(meta)
                    else:
                        sim.ensure_embedding_for_subject(
                            subject=s,
                            hop=kept_hop,
                            parent_entity=current_entity,
                            parent_intro=parent_intro,
                        )

    return unique_next
def _run_ner_wave_online_same_concurrency(
    *,
    args,
    paths,
    wave_items: List[dict],  # each: {"subject","hop","candidate_phrases"}
    ner_llm,
) -> Dict[Tuple[str, int], List[str]]:

    out: Dict[Tuple[str, int], List[str]] = {}

    if not wave_items:
        return out

    max_workers = min(int(getattr(args, "concurrency", 6) or 6), len(wave_items))

    def _one(item):
        subj = item["subject"]
        hop = item["hop"]
        cands = item.get("candidate_phrases") or []
        kept = _run_ner_strict_gate(
            args=args,
            paths=paths,
            subject=subj,
            hop=hop,
            candidate_phrases=cands,
            ner_llm=ner_llm,
        )
        return (subj, hop, kept)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(_one, it) for it in wave_items]
        for fut in as_completed(futs):
            subj, hop, kept = fut.result()
            out[(subj, hop)] = kept or []

    return out


def _extract_text_from_openai_batch_body(cfg, body: dict) -> str:
    """
    Extract assistant text from OpenAI Batch response "body".
    Works for both /v1/responses and /v1/chat/completions.
    """
    if not isinstance(body, dict):
        return ""

    if getattr(cfg, "use_responses_api", False):
        output_items = body.get("output") or []
        if not isinstance(output_items, list) or not output_items:
            return ""
        # find first message-ish item
        message_item = None
        for item in output_items:
            if isinstance(item, dict) and item.get("type") == "message":
                message_item = item
                break
        if message_item is None:
            for item in output_items:
                if isinstance(item, dict) and item.get("content"):
                    message_item = item
                    break
        if not message_item:
            return ""
        content = message_item.get("content") or []
        chunks = []
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and "text" in c:
                    t = c.get("text")
                    if isinstance(t, str) and t.strip():
                        chunks.append(t)
        # sometimes message has "text"
        t2 = message_item.get("text")
        if isinstance(t2, str) and t2.strip():
            chunks.append(t2)
        return "".join(chunks).strip()

    # chat completions
    choices = body.get("choices") or []
    if isinstance(choices, list) and choices:
        msg = (choices[0] or {}).get("message") or {}
        ct = msg.get("content")
        return ct.strip() if isinstance(ct, str) else ""
    return ""

def _run_ner_for_batch_wave(
    *,
    args,
    paths,
    ner_cfg,
    ner_llm,
    client: "OpenAI",
    wave_items: List[dict],
    wave_idx: int,
) -> Dict[Tuple[str, int], List[str]]:
    """
    Entry point used by run_batch():
      - if args.ner_mode == "batch": uses OpenAI Batch for NER
      - else: runs NER online using args.concurrency
    """
    ner_mode = (getattr(args, "ner_mode", None) or "online").strip().lower()
    if ner_mode == "batch":
        return _run_ner_wave_via_openai_batch(
            args=args,
            paths=paths,
            ner_cfg=ner_cfg,
            client=client,
            wave_items=wave_items,
            wave_idx=wave_idx,
        )
    # default: online with same concurrency knob
    return _run_ner_wave_online_same_concurrency(
        args=args,
        paths=paths,
        wave_items=wave_items,
        ner_llm=ner_llm,
    )



def run_batch_parallel(args, paths, el_cfg, ner_cfg, self_rag_cfg, queue: JsonQueue):
    import signal
    import sys

    # ---- BUFFERED WRITERS SETUP (same as run_online_parallel) ----
    buffer_size = int(getattr(args, "online_buffer_size", 500) or 500)
    flush_interval = float(getattr(args, "online_flush_interval", 10.0) or 10.0)

    writers = {
        "queue": BufferedJSONLWriter(paths["queue_jsonl"], buffer_size, flush_interval),
        "articles": BufferedJSONLWriter(paths["articles_jsonl"], buffer_size, flush_interval),
        "articles_wikitext": BufferedJSONLWriter(paths["articles_wikitext_jsonl"], buffer_size, flush_interval),
        "articles_meta": BufferedJSONLWriter(paths["articles_meta_jsonl"], buffer_size, flush_interval),
        "outlines": BufferedJSONLWriter(paths["outlines_jsonl"], buffer_size, flush_interval),
        "self_rag_log": BufferedJSONLWriter(paths["self_rag_log_jsonl"], buffer_size, flush_interval),
        "ner_decisions": BufferedJSONLWriter(paths["ner_decisions_jsonl"], buffer_size, flush_interval),
        "ner_lowconf": BufferedJSONLWriter(paths["ner_lowconf_jsonl"], buffer_size, flush_interval),
        "ner_responses": BufferedJSONLWriter(paths["ner_responses_jsonl"], buffer_size, flush_interval),
        "elicit_lowconf": BufferedJSONLWriter(paths["elicit_lowconf_jsonl"], buffer_size, flush_interval),
        "plural_s_dedup": BufferedJSONLWriter(paths["plural_s_dedup_jsonl"], buffer_size, flush_interval),
        "similarity_decisions": BufferedJSONLWriter(paths["similarity_decisions_jsonl"], buffer_size, flush_interval),
        "reject_similarity": BufferedJSONLWriter(paths["reject_similarity_jsonl"], buffer_size, flush_interval),
        "embeddings": BufferedJSONLWriter(paths["embeddings_jsonl"], buffer_size, flush_interval),
    }

    shutdown_count = [0]

    def signal_handler(signum, frame):
        shutdown_count[0] += 1
        
        if shutdown_count[0] == 1:
            _dbg(f"\n[shutdown] Stopping gracefully... (Press Ctrl+C again to force quit)")
            stop_event.set()
            prereq_capped_event.set()
        else:
            _dbg("[shutdown] Force quit!")
            # Flush critical files only
            try:
                for name, writer in writers.items():
                    writer.flush()
                queue.flush()
            except:
                pass
            sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill (default)

    # --- provider sanity for batch endpoints ---
    _ensure_openai_model_for_batch(el_cfg, "elicitation")
    if bool(getattr(args, "self_rag", False)):
        _ensure_openai_model_for_batch(self_rag_cfg, "self-rag")
    if bool(getattr(args, "use_ner", False)) and (str(getattr(args, "ner_mode", "batch")).lower() == "batch"):
        _ensure_openai_model_for_batch(ner_cfg, "ner")

    # 0) seed/resume + preload
    seen_canon_keys = _load_seen_canon(paths)
    _bootstrap_seen_from_queue(paths, seen_canon_keys)

    if getattr(args, "preload_only", False) and getattr(args, "preload_topics", None):
        # ── PRELOAD-ONLY: skip seed entirely, load only from file ──
        n = _preload_topics_from_file(args, paths, queue, seen_canon_keys)
        _dbg(f"[preload-only] {n} topics loaded, seed skipped")
    else:
        # ── Normal mode: seed first, then optionally preload extras ──
        _seed_or_resume_queue(args, paths, queue)
        seen_canon_keys.add(canon_key_from_queue(args.seed))
        if getattr(args, "preload_topics", None):
            _preload_topics_from_file(args, paths, queue, seen_canon_keys)


    # stage queues
    selfrag_q = _pq_make_stage_queue(paths, "selfrag", args)
    outline_q = _pq_make_stage_queue(paths, "outline", args)
    elicit_q  = _pq_make_stage_queue(paths, "elicit", args)
    ner_q     = _pq_make_stage_queue(paths, "ner", args)
    sim_q     = _pq_make_stage_queue(paths, "sim", args)

    _pq_reset_working_if_requested(args, selfrag_q, outline_q, elicit_q, ner_q, sim_q)

    # early stop if already at cap
    with _queue_lock:
        d0, w0, p0, f0 = queue.metrics(args.max_depth)
    if args.max_subjects and d0 >= args.max_subjects:
        # _dbg(f"[stop-init-bpq] max-subjects already reached on resume (done={d0})")
        _snapshot_json_only(paths, queue)
        return

    stop_event = threading.Event()
    prereq_capped_event = threading.Event()

    wave_lock = threading.Lock()
    wave_counter = {"n": 0}
    def _next_wave(prefix: str) -> str:
        with wave_lock:
            wave_counter["n"] += 1
            return f"{prefix}{wave_counter['n']}"

    poll_interval = float(getattr(args, "batch_poll_interval", 30.0) or 30.0)
    endpoint_selfrag = _endpoint_for_cfg(self_rag_cfg)
    endpoint_outline = _endpoint_for_cfg(el_cfg)
    endpoint_elicit  = _endpoint_for_cfg(el_cfg)

    def _run_openai_batch_job(
        client: OpenAI, *,
        endpoint: str,
        req_rows: list[dict],
        job_desc: str,
        file_prefix: str
    ):
        """
        Submit a Batch job and wait for completion.
        Returns (out_path or None, status_str).
        """
        batches_dir = paths["batches_dir"]
        os.makedirs(batches_dir, exist_ok=True)

        wave = _next_wave(file_prefix + "_")
        in_path = os.path.join(batches_dir, f"{file_prefix}_input_{wave}.jsonl")
        with open(in_path, "w", encoding="utf-8") as f:
            for r in req_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        try:
            with open(in_path, "rb") as fh:
                input_file = client.files.create(file=fh, purpose="batch")

            job = client.batches.create(
                input_file_id=input_file.id,
                endpoint=endpoint,
                completion_window="24h",
                metadata={"description": job_desc},
            )

            while True:
                job = client.batches.retrieve(job.id)
                if job.status == "completed":
                    break
                if job.status in {"failed", "expired", "cancelled"}:
                    # save error file if present
                    err_id = getattr(job, "error_file_id", None)
                    if err_id:
                        try:
                            err_bytes = client.files.content(err_id).content
                            err_path = os.path.join(batches_dir, f"{file_prefix}_errors_{wave}_{job.id}.jsonl")
                            with open(err_path, "wb") as ef:
                                ef.write(err_bytes)
                        except Exception:
                            pass
                    return None, str(job.status)
                time.sleep(poll_interval)

            out_id = getattr(job, "output_file_id", None)
            if not out_id:
                return None, "no_output_file"

            out_bytes = client.files.content(out_id).content
            out_path = os.path.join(batches_dir, f"{file_prefix}_output_{wave}_{job.id}.jsonl")
            with open(out_path, "wb") as f:
                f.write(out_bytes)

            # also save errors if present
            err_id = getattr(job, "error_file_id", None)
            if err_id:
                try:
                    err_bytes = client.files.content(err_id).content
                    err_path = os.path.join(batches_dir, f"{file_prefix}_errors_{wave}_{job.id}.jsonl")
                    with open(err_path, "wb") as ef:
                        ef.write(err_bytes)
                except Exception:
                    pass

            return out_path, "completed"

        except Exception as e:
            _append_error_log(paths, f"[bpq-batch-job] {file_prefix} exception: {type(e).__name__}: {e!r}", exc=e)
            return None, "sdk_exception"

    # -----------------------------
    # Dispatcher: main -> selfrag/outline -> maybe elicit
    # -----------------------------
    def dispatcher_loop():
        while not stop_event.is_set():
            with _queue_lock:
                d, w, p, f = queue.metrics(args.max_depth)

            if args.max_subjects and (d + w) >= args.max_subjects:
                time.sleep(0.2)
                continue

            claim_n = max(1, int(getattr(args, "concurrency", 6) or 6))
            if args.max_subjects:
                remaining = args.max_subjects - (d + w)
                if remaining <= 0:
                    time.sleep(0.2)
                    continue
                claim_n = max(1, min(claim_n, remaining))

            with _queue_lock:
                batch = queue.claim_pending_batch(args.max_depth, claim_n)

            if not batch:
                # main may have retries pending
                try:
                    due_in = queue.next_due_in(args.max_depth)
                except Exception:
                    due_in = None
                if due_in is not None:
                    time.sleep(min(max(due_in, 0.05), 1.0))
                else:
                    time.sleep(0.2)
                continue

            for subject, hop in batch:
                rec = queue.get_record(subject, hop)
                ps = rec.get("parent_subject") if isinstance(rec, dict) else None
                ph = rec.get("parent_hop") if isinstance(rec, dict) else None

                if _pq_stage_enabled_selfrag(args):
                    _pq_stage_enqueue_with_parent(paths, selfrag_q, subject, hop, ps, ph)

                else:
                    if not _pq_payload_exists(paths, "selfrag", subject, hop):
                        _pq_payload_write(
                            paths, "selfrag", subject, hop,
                            {"subject": subject, "hop": hop, "context": None, "skipped": True, "parent_subject": ps, "parent_hop": ph}
                        )

                if _pq_stage_enabled_outline(args):
                    _pq_stage_enqueue_with_parent(paths, outline_q, subject, hop, ps, ph)

                else:
                    if not _pq_payload_exists(paths, "outline", subject, hop):
                        _pq_payload_write(
                            paths, "outline", subject, hop,
                            {"subject": subject, "hop": hop, "outline": None, "skipped": True, "parent_subject": ps, "parent_hop": ph}
                        )

                _pq_try_enqueue_elicit(args, paths, elicit_q, subject, hop, parent_subject=ps, parent_hop=ph)

    # -----------------------------
    # Reconciler: fix missing downstream enqueues + propagate permanent stage failures
    #  Cascade stop when both outline+self-rag cap
    # -----------------------------
    def reconciler_loop():
        interval = float(getattr(args, "reconcile_interval", 3.0) or 3.0)
        while not stop_event.is_set():
            try:
                # Check if both prereq stages capped → stop pipeline
                if _pq_stage_enabled_selfrag(args) and _pq_stage_enabled_outline(args):
                    sr_d, sr_w, _, _ = selfrag_q.metrics(args.max_depth)
                    ol_d, ol_w, _, _ = outline_q.metrics(args.max_depth)
                    
                    both_capped = (
                        args.max_subjects and 
                        sr_d >= args.max_subjects and 
                        ol_d >= args.max_subjects
                    )
                    
                    if both_capped and not prereq_capped_event.is_set():
                        _dbg("[bpq-reconciler] Both prereqs capped → stopping SR/OL/NER/SIM (keeping elicit)")
                        prereq_capped_event.set()
                
                # read main queue snapshot
                main_state = []
                try:
                    if os.path.exists(paths["queue_json"]):
                        with open(paths["queue_json"], "r", encoding="utf-8") as f:
                            main_state = json.load(f) or []
                except Exception:
                    main_state = []

                for rec in (main_state or []):
                    if not isinstance(rec, dict):
                        continue
                    subject = rec.get("subject")
                    hop = rec.get("hop")
                    st = rec.get("status")
                    try:
                        hop = int(hop)
                    except Exception:
                        continue
                    if not isinstance(subject, str) or not subject.strip():
                        continue
                    if st not in {"working", "done"}:
                        continue
                    if args.max_depth != 0 and hop > args.max_depth:
                        continue

                    # enabled-only stage failure propagation
                    stages = []
                    if _pq_stage_enabled_selfrag(args): stages.append(("selfrag", selfrag_q))
                    if _pq_stage_enabled_outline(args): stages.append(("outline", outline_q))
                    stages.append(("elicit", elicit_q))
                    if bool(getattr(args, "use_ner", False)): stages.append(("ner", ner_q))
                    stages.append(("sim", sim_q))

                    for stage_name, qobj in stages:
                        try:
                            srec = qobj.get_record(subject, hop)
                        except Exception:
                            srec = None
                        if isinstance(srec, dict) and srec.get("status") == "failed":
                            with _queue_lock:
                                queue.mark_error(subject, hop, max_retries=args.max_retries, reason=f"stage_failed:{stage_name}")
                            break

                    # Extract parent from main queue record
                    ps = rec.get("parent_subject")
                    ph = rec.get("parent_hop")

                    # ensure prereqs
                    if _pq_stage_enabled_selfrag(args):
                        if not _pq_payload_exists(paths, "selfrag", subject, hop):
                            _pq_stage_enqueue_with_parent(paths, selfrag_q, subject, hop, ps, ph)

                    else:
                        if not _pq_payload_exists(paths, "selfrag", subject, hop):
                            _pq_payload_write(
                                paths, "selfrag", subject, hop,
                                {"subject": subject, "hop": hop, "context": None, "skipped": True, "parent_subject": ps, "parent_hop": ph}
                            )

                    if _pq_stage_enabled_outline(args):
                        if not _pq_payload_exists(paths, "outline", subject, hop):
                            _pq_stage_enqueue_with_parent(paths, outline_q, subject, hop, ps, ph)

                    else:
                        if not _pq_payload_exists(paths, "outline", subject, hop):
                            _pq_payload_write(
                                paths, "outline", subject, hop,
                                {"subject": subject, "hop": hop, "outline": None, "skipped": True, "parent_subject": ps, "parent_hop": ph}
                            )

                    _pq_try_enqueue_elicit(args, paths, elicit_q, subject, hop, parent_subject=ps, parent_hop=ph)

                    # ✅ downstream repair + ✅ MAIN DONE semantics:
                    # if elicit payload exists, MAIN must be done (same as run_online_parallel)
                    if _pq_payload_exists(paths, "elicit", subject, hop):
                        with _queue_lock:
                            queue.mark_done(subject, hop)

                        if bool(getattr(args, "use_ner", False)):
                            if not _pq_payload_exists(paths, "ner", subject, hop):
                                _pq_stage_enqueue_with_parent(paths, ner_q, subject, hop, ps, ph)
                            else:
                                _pq_stage_enqueue_with_parent(paths, sim_q, subject, hop, ps, ph)
                        else:
                            _pq_stage_enqueue_with_parent(paths, sim_q, subject, hop, ps, ph)

            except Exception as e:
                _append_error_log(paths, f"[bpq-reconciler] error={type(e).__name__}: {e!r}", exc=e)

            time.sleep(max(0.2, interval))

    # -----------------------------
    # Stage workers (BATCH)
    #  Per-stage cap checks
    # -----------------------------

    def selfrag_batch_worker_loop():
        if not _pq_stage_enabled_selfrag(args):
            return
        client = OpenAI()
        batch_n = max(1, int(getattr(args, "batch_size", 50) or 50))
        while not stop_event.is_set() and not prereq_capped_event.is_set():
            # ✅ Check cap before claiming
            if _stage_should_stop(selfrag_q, args, "selfrag"):
                _dbg("[selfrag-batch] Stopping due to cap")
                break
                
            batch = selfrag_q.claim_pending_batch(args.max_depth, batch_n)
            if not batch:
                _pq_sleep_on_retry_due(selfrag_q, args.max_depth)
                continue

            todo = []
            for subject, hop in batch:
                if _pq_payload_exists(paths, "selfrag", subject, hop):
                    selfrag_q.mark_done(subject, hop)
                    sr_payload = _pq_payload_read(paths, "selfrag", subject, hop) or {}
                    b_ps = sr_payload.get("parent_subject")
                    b_ph = sr_payload.get("parent_hop")
                    _pq_try_enqueue_elicit(args, paths, elicit_q, subject, hop, parent_subject=b_ps, parent_hop=b_ph)
                else:
                    todo.append((subject, hop))

            if not todo:
                continue

            req_rows = []
            for subject, hop in todo:
                root_topic = args.seed if args.domain == "topic" else subject
                msgs = _build_self_rag_messages(
                    subject=subject,
                    root_subject=root_topic,
                    persona_block=args.persona_self_rag_block
                )
                body = _build_openai_body_for_batch(
                    self_rag_cfg,
                    msgs,
                    max_tokens=int(getattr(args, "self_rag_max_tokens", 1024) or 1024)
                )
                cid = f"selfrag::{subject}::hop={hop}"
                req_rows.append({"custom_id": cid, "method": "POST", "url": endpoint_selfrag, "body": body})

            out_path, status = _run_openai_batch_job(
                client,
                endpoint=endpoint_selfrag,
                req_rows=req_rows,
                job_desc=f"LLMPedia bpq selfrag seed={args.seed}",
                file_prefix="bpq_selfrag",
            )

            if not out_path:
                _append_error_log(paths, f"[bpq-outline] batch failed status={status}, {len(todo)} items", exc=None)

                for subject, hop in todo:
                    selfrag_q.mark_error(subject, hop, args.max_retries, reason=f"selfrag_batch_failed:{status}")
                continue

            returned = set()
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    cid = row.get("custom_id")
                    if not isinstance(cid, str) or not cid.startswith("selfrag::"):
                        continue
                    try:
                        _, rest = cid.split("selfrag::", 1)
                        subj_part, hop_part = rest.rsplit("::hop=", 1)
                        subject = subj_part
                        hop = int(hop_part)
                    except Exception:
                        continue

                    returned.add((subject, hop))

                    resp = row.get("response") or {}
                    sc = resp.get("status_code", None)
                    if sc is not None and int(sc) >= 400:
                        selfrag_q.mark_error(subject, hop, args.max_retries, reason=f"selfrag_http:{sc}")
                        continue

                    body = (resp.get("body") or {})
                    txt = _extract_text_from_openai_batch_body(self_rag_cfg, body)
                    ctx = None
                    if isinstance(txt, str) and txt.strip():
                        try:
                            ctx = json.loads(txt)
                        except Exception:
                            ctx = None
                    if not isinstance(ctx, dict):
                        selfrag_q.mark_error(subject, hop, args.max_retries, reason="selfrag_parse_failed")
                        continue

                    rec = selfrag_q.get_record(subject, hop)
                    b_ps = rec.get("parent_subject") if isinstance(rec, dict) else None
                    b_ph = rec.get("parent_hop") if isinstance(rec, dict) else None
                    writers["self_rag_log"].append({"subject": subject, "hop": hop, "context": ctx})
                    _pq_payload_write(paths, "selfrag", subject, hop, {"subject": subject, "hop": hop, "context": ctx, "parent_subject": b_ps, "parent_hop": b_ph})
                    selfrag_q.mark_done(subject, hop)
                    _pq_try_enqueue_elicit(args, paths, elicit_q, subject, hop, parent_subject=b_ps, parent_hop=b_ph)

            missing = set(todo) - returned
            for subject, hop in missing:
                selfrag_q.mark_error(subject, hop, args.max_retries, reason="selfrag_missing_output")

    def outline_batch_worker_loop():
        if not _pq_stage_enabled_outline(args):
            return
        client = OpenAI()
        batch_n = max(1, int(getattr(args, "batch_size", 50) or 50))
        while not stop_event.is_set() and not prereq_capped_event.is_set():
            # ✅ Check cap before claiming
            if _stage_should_stop(outline_q, args, "outline"):
                _dbg("[outline-batch] Stopping due to cap")
                break
                
            batch = outline_q.claim_pending_batch(args.max_depth, batch_n)
            if not batch:
                _pq_sleep_on_retry_due(outline_q, args.max_depth)
                continue

            todo = []
            for subject, hop in batch:
                if _pq_payload_exists(paths, "outline", subject, hop):
                    outline_q.mark_done(subject, hop)
                    ol_payload = _pq_payload_read(paths, "outline", subject, hop) or {}
                    b_ps = ol_payload.get("parent_subject")
                    b_ph = ol_payload.get("parent_hop")
                    _pq_try_enqueue_elicit(args, paths, elicit_q, subject, hop, parent_subject=b_ps, parent_hop=b_ph)
                else:
                    todo.append((subject, hop))

            if not todo:
                continue

            req_rows = []
            for subject, hop in todo:
                root_topic = args.seed if args.domain == "topic" else subject
                msgs = _build_outline_messages_for_subject(
                    subject=subject,
                    root_topic=root_topic,
                    persona_block=args.persona_elicit_block,
                    args=args,
                )
                body = _build_openai_body_for_batch(el_cfg, msgs, max_tokens=2048)
                cid = f"outline::{subject}::hop={hop}"
                req_rows.append({"custom_id": cid, "method": "POST", "url": endpoint_outline, "body": body})

            out_path, status = _run_openai_batch_job(
                client,
                endpoint=endpoint_outline,
                req_rows=req_rows,
                job_desc=f"LLMPedia bpq outline seed={args.seed}",
                file_prefix="bpq_outline",
            )

            if not out_path:
                for subject, hop in todo:
                    outline_q.mark_error(subject, hop, args.max_retries, reason=f"outline_batch_failed:{status}")
                continue

            returned = set()
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    cid = row.get("custom_id")
                    if not isinstance(cid, str) or not cid.startswith("outline::"):
                        continue
                    try:
                        _, rest = cid.split("outline::", 1)
                        subj_part, hop_part = rest.rsplit("::hop=", 1)
                        subject = subj_part
                        hop = int(hop_part)
                    except Exception:
                        continue

                    returned.add((subject, hop))

                    resp = row.get("response") or {}
                    sc = resp.get("status_code", None)
                    if sc is not None and int(sc) >= 400:
                        outline_q.mark_error(subject, hop, args.max_retries, reason=f"outline_http:{sc}")
                        continue

                    body = (resp.get("body") or {})
                    outline_text = (_extract_text_from_openai_batch_body(el_cfg, body) or "").strip()

                    if outline_text:
                        _append_jsonl(paths["outlines_jsonl"], {"subject": subject, "hop": hop, "outline": outline_text})

                    rec = outline_q.get_record(subject, hop)
                    b_ps = rec.get("parent_subject") if isinstance(rec, dict) else None
                    b_ph = rec.get("parent_hop") if isinstance(rec, dict) else None
                    _pq_payload_write(paths, "outline", subject, hop, {"subject": subject, "hop": hop, "outline": outline_text, "parent_subject": b_ps, "parent_hop": b_ph})
                    outline_q.mark_done(subject, hop)
                    _pq_try_enqueue_elicit(args, paths, elicit_q, subject, hop, parent_subject=b_ps, parent_hop=b_ph)

            missing = set(todo) - returned
            for subject, hop in missing:
                outline_q.mark_error(subject, hop, args.max_retries, reason="outline_missing_output")

    def elicit_batch_worker_loop():
        client = OpenAI()
        batch_n = max(1, int(getattr(args, "batch_size", 50) or 50))
        while not stop_event.is_set():
            # ✅ Elicit stops when done+failed >= max_subjects (NOT when just done >= cap)
            if args.max_subjects:
                el_d, el_w, el_p, el_f = elicit_q.metrics(args.max_depth)
                if (el_d + el_f) >= args.max_subjects:
                    pass  # silenced
                    break
                
            batch = elicit_q.claim_pending_batch(args.max_depth, batch_n)
            if not batch:
                _pq_sleep_on_retry_due(elicit_q, args.max_depth)
                continue

            todo = []
            for subject, hop in batch:
                if _pq_payload_exists(paths, "elicit", subject, hop):
                    elicit_q.mark_done(subject, hop)

                    with _queue_lock:
                        queue.mark_done(subject, hop)

                    el_payload = _pq_payload_read(paths, "elicit", subject, hop) or {}
                    b_ps = el_payload.get("parent_subject")
                    b_ph = el_payload.get("parent_hop")

                    if bool(getattr(args, "use_ner", False)):
                        _pq_stage_enqueue_with_parent(paths, ner_q, subject, hop, b_ps, b_ph)
                    else:
                        _pq_stage_enqueue_with_parent(paths, sim_q, subject, hop, b_ps, b_ph)
                else:
                    todo.append((subject, hop))

            if not todo:
                continue

            # prereq check
            ready = []
            for subject, hop in todo:
                if _pq_prereq_selfrag_done(args, paths, subject, hop) and _pq_prereq_outline_done(args, paths, subject, hop):
                    ready.append((subject, hop))
                else:
                    elicit_q.mark_error(subject, hop, args.max_retries, reason="elicit_missing_prereq_payload")

            if not ready:
                continue

            req_rows = []
            for subject, hop in ready:
                root_topic = args.seed if args.domain == "topic" else subject

                sr_payload = _pq_payload_read(paths, "selfrag", subject, hop) or {}
                ol_payload = _pq_payload_read(paths, "outline", subject, hop) or {}

                self_rag_context = sr_payload.get("context") if isinstance(sr_payload, dict) else None
                outline_text = (ol_payload.get("outline") if isinstance(ol_payload, dict) else None)

                msgs = _build_llmpedia_messages_for_subject(
                    subject=subject,
                    hop=hop,
                    args=args,
                    root_topic=root_topic,
                    persona_block=args.persona_elicit_block,
                    self_rag_context=(self_rag_context if isinstance(self_rag_context, dict) else None),
                    outline=(outline_text if isinstance(outline_text, str) else None),
                )

                body = _build_openai_body_for_batch(
                    el_cfg,
                    msgs,
                    max_tokens=int(getattr(args, "elicit_max_tokens", 2048) or 2048)
                )
                cid = f"elicit::{subject}::hop={hop}"
                req_rows.append({"custom_id": cid, "method": "POST", "url": endpoint_elicit, "body": body})

            out_path, status = _run_openai_batch_job(
                client,
                endpoint=endpoint_elicit,
                req_rows=req_rows,
                job_desc=f"LLMPedia bpq elicit seed={args.seed}",
                file_prefix="bpq_elicit",
            )

            if not out_path:
                for subject, hop in ready:
                    elicit_q.mark_error(subject, hop, args.max_retries, reason=f"elicit_batch_failed:{status}")
                continue

            returned = set()
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    cid = row.get("custom_id")
                    if not isinstance(cid, str) or not cid.startswith("elicit::"):
                        continue
                    try:
                        _, rest = cid.split("elicit::", 1)
                        subj_part, hop_part = rest.rsplit("::hop=", 1)
                        subject = subj_part
                        hop = int(hop_part)
                    except Exception:
                        continue

                    returned.add((subject, hop))

                    resp = row.get("response") or {}
                    sc = resp.get("status_code", None)
                    if sc is not None and int(sc) >= 400:
                        elicit_q.mark_error(subject, hop, args.max_retries, reason=f"elicit_http:{sc}")
                        continue

                    body = (resp.get("body") or {})
                    wikitext = (_extract_text_from_openai_batch_body(el_cfg, body) or "").strip()
                    if _is_no_article_content(wikitext, subject):
                        elicit_q.mark_error(subject, hop, args.max_retries, reason="no_article_content_generated")
                        continue

                    # Get parent from elicit queue record
                    rec = elicit_q.get_record(subject, hop)
                    b_ps = rec.get("parent_subject") if isinstance(rec, dict) else None
                    b_ph = rec.get("parent_hop") if isinstance(rec, dict) else None

                    try:
                        candidate_phrases, parent_intro = _store_article_outputs_and_get_candidates_buffered(
                            args=args,
                            paths=paths,
                            el_cfg=el_cfg,
                            subject=subject,
                            hop=hop,
                            wikitext=wikitext,
                            writers=writers,
                            parent_subject=b_ps,
                            parent_hop=b_ph,
                        )
                    except Exception as e:
                        _append_error_log(
                            paths,
                            f"[bpq-elicit] store/extract failed: {type(e).__name__}: {e!r}",
                            subject=subject,
                            hop=hop,
                            exc=e
                        )
                        elicit_q.mark_error(subject, hop, args.max_retries, reason="store_extract_failed")
                        continue

                    _pq_payload_write(
                        paths, "elicit", subject, hop,
                        {"subject": subject, "hop": hop, "candidate_phrases": candidate_phrases, "parent_intro": parent_intro, "parent_subject": b_ps, "parent_hop": b_ph}
                    )
                    elicit_q.mark_done(subject, hop)

                    with _queue_lock:
                        queue.mark_done(subject, hop)

                    if bool(getattr(args, "use_ner", False)):
                        _pq_stage_enqueue_with_parent(paths, ner_q, subject, hop, b_ps, b_ph)
                    else:
                        _pq_stage_enqueue_with_parent(paths, sim_q, subject, hop, b_ps, b_ph)

            missing = set(ready) - returned
            for subject, hop in missing:
                elicit_q.mark_error(subject, hop, args.max_retries, reason="elicit_missing_output")

    
    def ner_stage_worker_loop():
        if not bool(getattr(args, "use_ner", False)):
            return

        ner_mode = (getattr(args, "ner_mode", "batch") or "batch").strip().lower()
        batch_n = max(1, int(getattr(args, "batch_size", 50) or 50))

        if ner_mode == "online":
            llm = make_llm_from_config(ner_cfg)
            while not stop_event.is_set() and not prereq_capped_event.is_set():
                # ✅ Check cap before claiming
                if _stage_should_stop(ner_q, args, "ner"):
                    _dbg("[ner-online] Stopping due to cap")
                    break
                    
                batch = ner_q.claim_pending_batch(args.max_depth, 1)
                if not batch:
                    _pq_sleep_on_retry_due(ner_q, args.max_depth)
                    continue
                subject, hop = batch[0]

                if _pq_payload_exists(paths, "ner", subject, hop):
                    ner_q.mark_done(subject, hop)
                    ner_payload = _pq_payload_read(paths, "ner", subject, hop) or {}
                    b_ps = ner_payload.get("parent_subject")
                    b_ph = ner_payload.get("parent_hop")
                    _pq_stage_enqueue_with_parent(paths, sim_q, subject, hop, b_ps, b_ph)
                    continue

                el_payload = _pq_payload_read(paths, "elicit", subject, hop)
                if not isinstance(el_payload, dict):
                    ner_q.mark_error(subject, hop, args.max_retries, reason="ner_missing_elicit_payload")
                    continue

                cands = el_payload.get("candidate_phrases") or []
                if not isinstance(cands, list):
                    cands = []

                # ═══════════════════════════════════════════════════════════
                #  Do canonical dedup + plural variant rejection BEFORE NER
                # ═══════════════════════════════════════════════════════════
                pre_dedup_count = len(cands)
                deduped_cands: List[str] = []
                uniq_canon: Set[str] = set()
                
                for s in cands:
                    canon = canon_key_from_queue(s)
                    if not canon:
                        continue
                    
                    with _seen_canon_lock:
                        is_pldup, matched = _is_plural_variant_duplicate(canon, seen_canon_keys)
                    if is_pldup:
                        _append_jsonl(paths["plural_s_dedup_jsonl"], {
                            "stage": "plural_variant_reject_pre_ner",
                            "current_entity": subject,
                            "hop": hop,
                            "phrase": s,
                            "canonical_key": canon,
                            "matched_variant": matched,
                        })
                        continue
                    
                    is_batch_pldup, batch_matched = _is_plural_variant_duplicate(canon, uniq_canon)
                    if is_batch_pldup:
                        _append_jsonl(paths["plural_s_dedup_jsonl"], {
                            "stage": "plural_variant_reject_batch_pre_ner",
                            "current_entity": subject,
                            "hop": hop,
                            "phrase": s,
                            "canonical_key": canon,
                            "matched_variant": batch_matched,
                        })
                        continue
                    
                    with _seen_canon_lock:
                        if canon in seen_canon_keys:
                            _append_jsonl(paths["ner_lowconf_jsonl"], {
                                "stage": "queue_dedup_pre_ner",
                                "current_entity": subject,
                                "hop": hop,
                                "phrase": s,
                                "canonical_key": canon,
                                "rejection_reason": "queue_canonical_seen_pre_ner",
                            })
                            continue
                    
                    if canon in uniq_canon:
                        _append_jsonl(paths["ner_lowconf_jsonl"], {
                            "stage": "queue_dedup_batch_pre_ner",
                            "current_entity": subject,
                            "hop": hop,
                            "phrase": s,
                            "canonical_key": canon,
                            "rejection_reason": "queue_batch_duplicate_pre_ner",
                        })
                        continue
                    
                    uniq_canon.add(canon)
                    deduped_cands.append(s)
                
                post_dedup_count = len(deduped_cands)
                
                _append_jsonl(paths["ner_decisions_jsonl"], {
                    "stage": "pre_ner_dedup_summary",
                    "current_entity": subject,
                    "hop": hop,
                    "pre_dedup_count": pre_dedup_count,
                    "post_dedup_count": post_dedup_count,
                    "filtered_count": pre_dedup_count - post_dedup_count,
                })
                
                if not deduped_cands:
                    rec = ner_q.get_record(subject, hop)
                    b_ps = rec.get("parent_subject") if isinstance(rec, dict) else None
                    b_ph = rec.get("parent_hop") if isinstance(rec, dict) else None
                    _pq_payload_write(paths, "ner", subject, hop, {
                        "subject": subject, 
                        "hop": hop, 
                        "kept": [],
                        "pre_dedup_count": pre_dedup_count,
                        "post_dedup_count": 0,
                        "ner_skipped": True,
                        "parent_subject": b_ps,
                        "parent_hop": b_ph,
                    })
                    ner_q.mark_done(subject, hop)
                    _pq_stage_enqueue_with_parent(paths, sim_q, subject, hop, b_ps, b_ph)
                    continue

                try:
                    kept = _run_ner_strict_gate(
                        args=args,
                        paths=paths,
                        subject=subject,
                        hop=hop,
                        candidate_phrases=deduped_cands,
                        ner_llm=llm
                    )
                    rec = ner_q.get_record(subject, hop)
                    b_ps = rec.get("parent_subject") if isinstance(rec, dict) else None
                    b_ph = rec.get("parent_hop") if isinstance(rec, dict) else None
                    _pq_payload_write(paths, "ner", subject, hop, {
                        "subject": subject, 
                        "hop": hop, 
                        "kept": kept,
                        "pre_dedup_count": pre_dedup_count,
                        "post_dedup_count": post_dedup_count,
                        "parent_subject": b_ps,
                        "parent_hop": b_ph,
                    })
                    ner_q.mark_done(subject, hop)
                    _pq_stage_enqueue_with_parent(paths, sim_q, subject, hop, b_ps, b_ph)
                except Exception as e:
                    _append_error_log(paths, f"[bpq-ner-online] error={type(e).__name__}: {e!r}", subject=subject, hop=hop, exc=e)
                    ner_q.mark_error(subject, hop, args.max_retries, reason=f"ner_online:{type(e).__name__}")
            return

        # ner_mode == "batch"
        client = OpenAI()
        while not stop_event.is_set() and not prereq_capped_event.is_set():
            # ✅ Check cap before claiming
            if _stage_should_stop(ner_q, args, "ner-batch"):
                _dbg("[ner-batch] Stopping due to cap")
                break
                
            batch = ner_q.claim_pending_batch(args.max_depth, batch_n)
            if not batch:
                _pq_sleep_on_retry_due(ner_q, args.max_depth)
                continue

            todo = []
            wave_items = []
            for subject, hop in batch:
                if _pq_payload_exists(paths, "ner", subject, hop):
                    ner_q.mark_done(subject, hop)
                    ner_payload = _pq_payload_read(paths, "ner", subject, hop) or {}
                    b_ps = ner_payload.get("parent_subject")
                    b_ph = ner_payload.get("parent_hop")
                    _pq_stage_enqueue_with_parent(paths, sim_q, subject, hop, b_ps, b_ph)
                    continue

                el_payload = _pq_payload_read(paths, "elicit", subject, hop)
                if not isinstance(el_payload, dict):
                    ner_q.mark_error(subject, hop, args.max_retries, reason="ner_missing_elicit_payload")
                    continue

                cands = el_payload.get("candidate_phrases") or []
                if not isinstance(cands, list):
                    cands = []

                # ═══════════════════════════════════════════════════════════
                #  Do canonical dedup + plural variant rejection BEFORE NER
                # ═══════════════════════════════════════════════════════════
                pre_dedup_count = len(cands)
                deduped_cands: List[str] = []
                uniq_canon: Set[str] = set()
                
                for s in cands:
                    canon = canon_key_from_queue(s)
                    if not canon:
                        continue
                    
                    with _seen_canon_lock:
                        is_pldup, matched = _is_plural_variant_duplicate(canon, seen_canon_keys)
                    if is_pldup:
                        _append_jsonl(paths["plural_s_dedup_jsonl"], {
                            "stage": "plural_variant_reject_pre_ner",
                            "current_entity": subject,
                            "hop": hop,
                            "phrase": s,
                            "canonical_key": canon,
                            "matched_variant": matched,
                        })
                        continue
                    
                    is_batch_pldup, batch_matched = _is_plural_variant_duplicate(canon, uniq_canon)
                    if is_batch_pldup:
                        _append_jsonl(paths["plural_s_dedup_jsonl"], {
                            "stage": "plural_variant_reject_batch_pre_ner",
                            "current_entity": subject,
                            "hop": hop,
                            "phrase": s,
                            "canonical_key": canon,
                            "matched_variant": batch_matched,
                        })
                        continue
                    
                    with _seen_canon_lock:
                        if canon in seen_canon_keys:
                            _append_jsonl(paths["ner_lowconf_jsonl"], {
                                "stage": "queue_dedup_pre_ner",
                                "current_entity": subject,
                                "hop": hop,
                                "phrase": s,
                                "canonical_key": canon,
                                "rejection_reason": "queue_canonical_seen_pre_ner",
                            })
                            continue
                    
                    if canon in uniq_canon:
                        _append_jsonl(paths["ner_lowconf_jsonl"], {
                            "stage": "queue_dedup_batch_pre_ner",
                            "current_entity": subject,
                            "hop": hop,
                            "phrase": s,
                            "canonical_key": canon,
                            "rejection_reason": "queue_batch_duplicate_pre_ner",
                        })
                        continue
                    
                    uniq_canon.add(canon)
                    deduped_cands.append(s)
                
                post_dedup_count = len(deduped_cands)
                
                _append_jsonl(paths["ner_decisions_jsonl"], {
                    "stage": "pre_ner_dedup_summary",
                    "current_entity": subject,
                    "hop": hop,
                    "pre_dedup_count": pre_dedup_count,
                    "post_dedup_count": post_dedup_count,
                    "filtered_count": pre_dedup_count - post_dedup_count,
                })
                
                if not deduped_cands:
                    rec = ner_q.get_record(subject, hop)
                    b_ps = rec.get("parent_subject") if isinstance(rec, dict) else None
                    b_ph = rec.get("parent_hop") if isinstance(rec, dict) else None
                    _pq_payload_write(paths, "ner", subject, hop, {
                        "subject": subject, 
                        "hop": hop, 
                        "kept": [],
                        "pre_dedup_count": pre_dedup_count,
                        "post_dedup_count": 0,
                        "ner_skipped": True,
                        "parent_subject": b_ps,
                        "parent_hop": b_ph,
                    })
                    ner_q.mark_done(subject, hop)
                    _pq_stage_enqueue_with_parent(paths, sim_q, subject, hop, b_ps, b_ph)
                    continue

                todo.append((subject, hop))
                wave_items.append({
                    "subject": subject, 
                    "hop": hop, 
                    "candidate_phrases": deduped_cands,
                    "pre_dedup_count": pre_dedup_count,
                    "post_dedup_count": post_dedup_count,
                })

            if not wave_items:
                continue

            wave_idx = int(time.time() * 1000) % 1000000000
            wave_idx = int(time.time() * 1000) % 1000000000 + threading.get_ident() % 100000

            try:
                kept_map = _run_ner_wave_via_openai_batch(
                    args=args,
                    paths=paths,
                    ner_cfg=ner_cfg,
                    client=client,
                    wave_items=wave_items,
                    wave_idx=wave_idx,
                )
            except Exception as _ner_batch_exc:
                _append_error_log(
                    paths,
                    f"[bpq-ner-batch] UNHANDLED exception: {type(_ner_batch_exc).__name__}: {_ner_batch_exc!r}",
                    exc=_ner_batch_exc,
                )
                _dbg(f"[bpq-ner-batch] CRASH CAUGHT: {type(_ner_batch_exc).__name__}: {_ner_batch_exc!r}")
                for subject, hop in todo:
                    ner_q.mark_error(subject, hop, args.max_retries, reason=f"ner_batch_crash:{type(_ner_batch_exc).__name__}")
                continue

            for i, (subject, hop) in enumerate(todo):
                kept = kept_map.get((subject, hop), [])
                wi = wave_items[i] if i < len(wave_items) else {}
                rec = ner_q.get_record(subject, hop)
                b_ps = rec.get("parent_subject") if isinstance(rec, dict) else None
                b_ph = rec.get("parent_hop") if isinstance(rec, dict) else None
                _pq_payload_write(paths, "ner", subject, hop, {
                    "subject": subject, 
                    "hop": hop, 
                    "kept": kept,
                    "pre_dedup_count": wi.get("pre_dedup_count", 0),
                    "post_dedup_count": wi.get("post_dedup_count", 0),
                    "parent_subject": b_ps,
                    "parent_hop": b_ph,
                })
                ner_q.mark_done(subject, hop)
                _pq_stage_enqueue_with_parent(paths, sim_q, subject, hop, b_ps, b_ph)

    def sim_worker_loop(worker_id: int = 0):
        similarity_mode = (getattr(args, "similarity_mode", "batch") or "batch").lower()
        batch_n = max(1, int(getattr(args, "batch_size", 50) or 50))

        client = OpenAI() if similarity_mode == "batch" else None

        similarity_cfg = None
        if bool(getattr(args, "use_similarity", False)):
            filt_key = getattr(args, "similarity_filter_model_key", None) or getattr(args, "ner_model_key", None)
            if filt_key:
                similarity_cfg = _cfg_from_key(filt_key, args.timeout)
                _strip_responses_sampling_if_disallowed(similarity_cfg)

        wave_counter = 0

        while not stop_event.is_set() and not prereq_capped_event.is_set():
            # ✅ Check cap before claiming
            if _stage_should_stop(sim_q, args, f"sim-{worker_id}"):
                _dbg(f"[sim-{worker_id}] Stopping due to cap")
                break
                
            batch = sim_q.claim_pending_batch(args.max_depth, batch_n)
            if not batch:
                _pq_sleep_on_retry_due(sim_q, args.max_depth)
                continue

            all_items_for_similarity: List[dict] = []
            subject_data: Dict[Tuple[str, int], dict] = {}

            # Wave-level dedupe set for similarity-check list only
            seen_cands_wave: Set[str] = set()

            for subject, hop in batch:
                el_payload = _pq_payload_read(paths, "elicit", subject, hop)
                if not isinstance(el_payload, dict):
                    sim_q.mark_error(subject, hop, args.max_retries, reason="sim_missing_elicit_payload")
                    continue

                parent_intro = el_payload.get("parent_intro") or ""
                if not isinstance(parent_intro, str):
                    parent_intro = ""

                # ═══════════════════════════════════════════════════════════
                # Get candidates - NER already did dedup, so just use kept!
                # ═══════════════════════════════════════════════════════════
                if bool(getattr(args, "use_ner", False)):
                    ner_payload = _pq_payload_read(paths, "ner", subject, hop)
                    if not (isinstance(ner_payload, dict) and isinstance(ner_payload.get("kept"), list)):
                        el_payload_p = _pq_payload_read(paths, "elicit", subject, hop) or {}
                        _ps = el_payload_p.get("parent_subject")
                        _ph = el_payload_p.get("parent_hop")
                        _pq_stage_enqueue_with_parent(paths, ner_q, subject, hop, _ps, _ph)
                        sim_q.mark_error(subject, hop, args.max_retries, reason="sim_missing_ner_payload")
                        continue
                    # NER already deduped these - use directly!
                    next_subjects = ner_payload["kept"] or []
                else:
                    # NER disabled - we need to do dedup here
                    cands = el_payload.get("candidate_phrases") or []
                    if not isinstance(cands, list):
                        cands = []
                        
                    next_subjects = []
                    uniq_canon: Set[str] = set()
                    
                    for s in cands:
                        canon = canon_key_from_queue(s)
                        if not canon:
                            continue

                        with _seen_canon_lock:
                            is_pldup, matched = _is_plural_variant_duplicate(canon, seen_canon_keys)
                        if is_pldup:
                            _append_jsonl(paths["plural_s_dedup_jsonl"], {
                                "stage": "plural_variant_reject",
                                "current_entity": subject,
                                "hop": hop,
                                "phrase": s,
                                "canonical_key": canon,
                                "matched_variant": matched,
                            })
                            continue

                        with _seen_canon_lock:
                            if canon in seen_canon_keys:
                                continue

                        if canon in uniq_canon:
                            continue

                        uniq_canon.add(canon)
                        next_subjects.append(s)

                subject_data[(subject, hop)] = {"parent_intro": parent_intro, "unique_next": next_subjects}

                # Add to similarity check list (DEDUPED GLOBALLY within wave)
                for cand in next_subjects:
                    if cand in seen_cands_wave:
                        continue
                    seen_cands_wave.add(cand)
                    all_items_for_similarity.append({
                        "candidate": cand,
                        "parent_entity": subject,
                        "parent_intro": parent_intro,
                        "hop": hop,
                        "_subject": subject,
                    })

            if not all_items_for_similarity:
                for subject, hop in batch:
                    if (subject, hop) in subject_data:
                        sim_q.mark_done(subject, hop)
                continue

            kept_candidates: Set[str] = set()
            kept_meta_all: Dict[str, dict] = {}

            if bool(getattr(args, "use_similarity", False)):
                sim = _get_similarity_engine(args, paths)
                if sim is not None and similarity_cfg is not None:
                    wave_counter += 1
                    kept_list, kept_meta_all = sim.filter_candidates_batch(
                        items=all_items_for_similarity,
                        similarity_cfg=similarity_cfg,
                        client=client,
                        wave_idx=wave_counter,
                    )
                    kept_candidates = set(kept_list)
                else:
                    kept_candidates = set(it["candidate"] for it in all_items_for_similarity)
            else:
                kept_candidates = set(it["candidate"] for it in all_items_for_similarity)

            # Enqueue Survivors
            for subject, hop in batch:
                sd = subject_data.get((subject, hop))
                if sd is None:
                    continue

                unique_next = sd["unique_next"]
                parent_intro = sd["parent_intro"]
                final_next = [c for c in unique_next if c in kept_candidates]

                inserted_count = 0

                # ── PRELOAD-ONLY: no expansion, just log and skip ──
                if not getattr(args, "preload_only", False):
                    for s in final_next:
                        if args.max_depth != 0 and hop + 1 > args.max_depth:
                            continue

                        with _queue_lock:
                            _, kept_hop, outcome = queue.enqueue(
                                s, hop + 1,
                                parent_subject=subject,
                                parent_hop=hop,
                            )

                        if outcome == "inserted":
                            inserted_count += 1
                            writers["queue"].append({
                                "subject": s,
                                "hop": kept_hop,
                                "event": outcome,
                                "parent_subject": subject,
                                "parent_hop": hop,
                            })

                            with _seen_canon_lock:
                                ck2 = canon_key_from_queue(s)
                                if ck2:
                                    seen_canon_keys.add(ck2)

                            if bool(getattr(args, "use_similarity", False)):
                                sim = _get_similarity_engine(args, paths)
                                if sim is not None:
                                    meta = kept_meta_all.get(s)
                                    if isinstance(meta, dict):
                                        meta["hop"] = kept_hop
                                        meta["parent_subject"] = subject
                                        meta["parent_hop"] = hop
                                        meta["parent_intro"] = parent_intro
                                        sim.commit_embedding_if_inserted(meta, writers=writers)

                sim_q.mark_done(subject, hop)
            _persist_seen_canon(paths, seen_canon_keys)


    
    
    # -----------------------------
    # SYNCHRONOUS PRE-DISPATCH: fill stage queues BEFORE workers start
    # This prevents workers from grabbing tiny partial batches.
    # -----------------------------
    def _synchronous_predispatch():
        """Claim ALL pending from main queue → push to stage queues."""
        total_dispatched = 0
        while True:
            with _queue_lock:
                d, w, p, f = queue.metrics(args.max_depth)
            if args.max_subjects and (d + w) >= args.max_subjects:
                break
            claim_n = max(1, int(getattr(args, "batch_size", 500) or 500))
            if args.max_subjects:
                remaining = args.max_subjects - (d + w)
                if remaining <= 0:
                    break
                claim_n = min(claim_n, remaining)
            with _queue_lock:
                batch = queue.claim_pending_batch(args.max_depth, claim_n)
            if not batch:
                break
            for subject, hop in batch:
                rec = queue.get_record(subject, hop)
                ps = rec.get("parent_subject") if isinstance(rec, dict) else None
                ph = rec.get("parent_hop") if isinstance(rec, dict) else None
                if _pq_stage_enabled_selfrag(args):
                    _pq_stage_enqueue_with_parent(paths, selfrag_q, subject, hop, ps, ph)
                else:
                    if not _pq_payload_exists(paths, "selfrag", subject, hop):
                        _pq_payload_write(paths, "selfrag", subject, hop,
                            {"subject": subject, "hop": hop, "context": None, "skipped": True, "parent_subject": ps, "parent_hop": ph})
                if _pq_stage_enabled_outline(args):
                    _pq_stage_enqueue_with_parent(paths, outline_q, subject, hop, ps, ph)
                else:
                    if not _pq_payload_exists(paths, "outline", subject, hop):
                        _pq_payload_write(paths, "outline", subject, hop,
                            {"subject": subject, "hop": hop, "outline": None, "skipped": True, "parent_subject": ps, "parent_hop": ph})
                _pq_try_enqueue_elicit(args, paths, elicit_q, subject, hop, parent_subject=ps, parent_hop=ph)
            total_dispatched += len(batch)
        if total_dispatched:
            _dbg(f"[bpq-predispatch] Dispatched {total_dispatched} items to stage queues before starting workers")

    _synchronous_predispatch()

    # -----------------------------
    # Start threads
    # -----------------------------
    threads = []
    threads.append(threading.Thread(target=dispatcher_loop, name="bpq-dispatcher", daemon=True))
    threads.append(threading.Thread(target=reconciler_loop, name="bpq-reconciler", daemon=True))

    for i in range(max(0, int(getattr(args, "selfrag_workers", 0) or 0))):
        threads.append(threading.Thread(target=selfrag_batch_worker_loop, name=f"bpq-selfrag-{i}", daemon=True))
    for i in range(max(0, int(getattr(args, "outline_workers", 0) or 0))):
        threads.append(threading.Thread(target=outline_batch_worker_loop, name=f"bpq-outline-{i}", daemon=True))
    for i in range(max(0, int(getattr(args, "elicit_workers", 0) or 0))):
        threads.append(threading.Thread(target=elicit_batch_worker_loop, name=f"bpq-elicit-{i}", daemon=True))
    for i in range(max(0, int(getattr(args, "ner_workers", 0) or 0))):
        threads.append(threading.Thread(target=ner_stage_worker_loop, name=f"bpq-ner-{i}", daemon=True))
    for i in range(max(0, int(getattr(args, "sim_workers", 0) or 0))):
        threads.append(threading.Thread(target=sim_worker_loop, args=(i,), name=f"bpq-sim-{i}", daemon=True))


    for t in threads:
        t.start()

    # -----------------------------
    # Monitor loop
    # -----------------------------
    start = time.perf_counter()
    last_progress_ts = 0.0

    def _all_stage_metrics():
        sr = (0, 0, 0, 0) if not _pq_stage_enabled_selfrag(args) else selfrag_q.metrics(args.max_depth)
        ol = (0, 0, 0, 0) if not _pq_stage_enabled_outline(args) else outline_q.metrics(args.max_depth)
        el = elicit_q.metrics(args.max_depth)
        ne = (0, 0, 0, 0) if not bool(getattr(args, "use_ner", False)) else ner_q.metrics(args.max_depth)
        si = sim_q.metrics(args.max_depth)
        return sr, ol, el, ne, si

    try:
        while not stop_event.is_set():  # ✅ CORRECT - checks stop_event
            now = time.perf_counter()
            if args.progress_metrics and (now - last_progress_ts) >= 30.0:
                with _queue_lock:
                    md, mw, mp, mf = queue.metrics(args.max_depth)
                sr, ol, el, ne, si = _all_stage_metrics()
                _dbg(
                    "[bpq-progress] "
                    f"MAIN d={md} w={mw} p={mp} f={mf} | "
                    f"SR d={sr[0]} w={sr[1]} p={sr[2]} f={sr[3]} | "
                    f"OL d={ol[0]} w={ol[1]} p={ol[2]} f={ol[3]} | "
                    f"EL d={el[0]} w={el[1]} p={el[2]} f={el[3]} | "
                    f"NER d={ne[0]} w={ne[1]} p={ne[2]} f={ne[3]} | "
                    f"SIM d={si[0]} w={si[1]} p={si[2]} f={si[3]}"
                )
                last_progress_ts = now

            with _queue_lock:
                md, mw, mp, mf = queue.metrics(args.max_depth)

            sr, ol, el, ne, si = _all_stage_metrics()
            stages_empty = all((m[1] == 0 and m[2] == 0) for m in [sr, ol, el, ne, si])
            main_empty = (mw == 0 and mp == 0)

            if main_empty and stages_empty:
                break

            # ✅ HARD STOP: when elicit done+failed >= max_subjects, pipeline is truly complete
            if args.max_subjects:
                el_d, el_w, el_p, el_f = elicit_q.metrics(args.max_depth)
                if (el_d + el_f) >= args.max_subjects:
                    _dbg(f"[bpq-stop] Elicit done+failed={el_d+el_f} >= max_subjects={args.max_subjects}; stopping.")
                    stop_event.set()
                    break

            time.sleep(0.2)

    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=1.0)

        # ---- FLUSH ALL JSONL WRITERS ----
        _dbg("[bpq-flush] flushing all buffered JSONL writes...")
        for name, writer in writers.items():
            try:
                writer.flush()
            except Exception as e:
                _dbg(f"[bpq-flush] {name}: error={e!r}")

        # ---- FLUSH ALL STAGE QUEUES ----
        _dbg("[bpq-flush] flushing stage queues...")
        for q_name, q_obj in [
            ("selfrag", selfrag_q),
            ("outline", outline_q),
            ("elicit", elicit_q),
            ("ner", ner_q),
            ("sim", sim_q),
        ]:
            try:
                q_obj.flush()
            except Exception as e:
                _dbg(f"[bpq-flush] {q_name}_q: error={e!r}")

        # ---- FLUSH MAIN QUEUE ----
        try:
            queue.flush()
        except Exception as e:
            _dbg(f"[bpq-flush] main queue: error={e!r}")

    _persist_seen_canon(paths, seen_canon_keys)
    _snapshot_json_only(paths, queue)
    dur = time.perf_counter() - start
    _dbg(f"[done-bpq] finished in {dur:.1f}s -> {os.path.dirname(paths['queue_json'])}")

def main():
    ap = argparse.ArgumentParser(
        description="LLMPedia (JSON-only) — person/topic crawler with OpenAI Batch support"
    )

    # ---- CLI ----
    ap.add_argument(
        "--mode",
        choices=["online", "batch"],
        default="online",
        help="online = normal BFS; batch = full OpenAI Batch pipeline.",
    )
    ap.add_argument("--seed", required=True, help="Seed entity name (e.g., 'Albert Einstein').")
    ap.add_argument("--output-dir", default=None)

    ap.add_argument(
        "--domain",
        default="general",
        choices=["general", "topic"],
        help="Prompt domain; 'topic' passes root_subject=seed, 'general' does not.",
    )
    ap.add_argument(
        "--elicitation-strategy",
        default="baseline",
        help="Prompt strategy folder for elicitation.",
    )
    ap.add_argument(
        "--ner-strategy",
        default="baseline",
        help="Prompt strategy folder for NER.",
    )

    ap.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="0 = unlimited depth (stop when queue empty)",
    )
    ap.add_argument(
        "--max-subjects",
        type=int,
        default=0,
        help="0 = unlimited subjects",
    )

    # Article prompt controls
    ap.add_argument("--article-min-sections", type=int, default=3)
    ap.add_argument("--article-max-sections", type=int, default=7)
    ap.add_argument("--article-avg-words", type=int, default=716)

    ap.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="For mode=batch: how many subjects per OpenAI batch wave.",
    )
    ap.add_argument(
        "--batch-poll-interval",
        type=float,
        default=30.0,
        help="Seconds between polling /v1/batches in mode=batch.",
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=6,
        help="For mode=online worker concurrency.",
    )

    # Self-RAG-specific arguments
    ap.add_argument(
        "--self-rag-mode",
        choices=["batch", "online"],
        default=None,  # will cascade from --mode
        help="Self-RAG execution mode. Default: follows --mode.",
    )
    ap.add_argument(
        "--self-rag-concurrency",
        type=int,
        default=2,
        help=(
            "When --mode=batch and --self-rag-mode=online, how many Self-RAG calls "
            "run in parallel per wave (default: 2)."
        ),
    )

    # ===============================================================
    # GLOBAL MODEL & SAMPLING DEFAULTS (cascade to all stages)
    # ===============================================================
    ap.add_argument(
        "--model-key",
        default="gpt-4.1-mini",
        help="Default model key for ALL stages (elicit, ner, self-rag). Stage-specific flags override this.",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Default temperature for all stages.",
    )
    ap.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Default top_p for all stages.",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Default top_k for all stages.",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Default max_tokens for all stages.",
    )

    # ===============================================================
    # STAGE-SPECIFIC MODEL OVERRIDES (None = use global --model-key)
    # ===============================================================
    ap.add_argument(
        "--elicit-model-key",
        default=None,
        help="Model key for elicitation. Default: uses --model-key.",
    )
    ap.add_argument(
        "--ner-model-key",
        default=None,
        help="Model key for NER. Default: uses --model-key.",
    )
    ap.add_argument(
        "--self-rag-model-key",
        default=None,
        help="Model key for Self-RAG. Default: uses --model-key.",
    )

    # ===============================================================
    # STAGE-SPECIFIC SAMPLING OVERRIDES (None = use global)
    # ===============================================================
    ap.add_argument("--elicit-temperature", type=float, default=None)
    ap.add_argument("--ner-temperature", type=float, default=None)
    ap.add_argument("--self-rag-temperature", type=float, default=None)

    ap.add_argument("--elicit-top-p", type=float, default=None)
    ap.add_argument("--ner-top-p", type=float, default=None)
    ap.add_argument("--self-rag-top-p", type=float, default=None)

    ap.add_argument("--elicit-top-k", type=int, default=None)
    ap.add_argument("--ner-top-k", type=int, default=None)
    ap.add_argument("--self-rag-top-k", type=int, default=None)

    ap.add_argument("--elicit-max-tokens", type=int, default=None)
    ap.add_argument("--ner-max-tokens", type=int, default=None)
    ap.add_argument("--self-rag-max-tokens", type=int, default=1024) 

    ap.add_argument("--timeout", type=float, default=200.0, help="Request timeout (seconds) for online calls.")

    # NER / Elicitation thresholds
    ap.add_argument("--ner-conf-threshold", type=float, default=0.7)
    ap.add_argument("--elicit-conf-threshold", type=float, default=0.75)

    # Footer controls
    ap.add_argument("--footer-mode", type=_str2bool, default=False)
    ap.add_argument("--footer-location", choices=["system", "user"], default="user")
    ap.add_argument("--use-ner", type=_str2bool, default=True)

    # NER execution mode + chunking
    ap.add_argument(
        "--ner-mode",
        choices=["online", "batch"],
        default=None,  # will cascade from --mode
        help="NER execution mode. Default: follows --mode.",
    )
    ap.add_argument(
        "--ner-chunk-size",
        type=int,
        default=25,
        help="How many candidate phrases per NER request (online and batch).",
    )

    # Self-RAG controls
    ap.add_argument("--self-rag", type=_str2bool, default=False)
    ap.add_argument("--self-rag-target", choices=["system", "user"], default="user")
    ap.add_argument(
        "--two-stage-elicitation",
        dest="two_stage_elicit",
        type=_str2bool,
        default=True,
    )

    # ===============================================================
    # GLOBAL REASONING & VERBOSITY (Responses API) - cascade to all stages
    # ===============================================================
    ap.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high"],
        default="minimal",
        help="Default reasoning effort for ALL stages (Responses API only).",
    )
    ap.add_argument(
        "--text-verbosity",
        choices=["low", "medium", "high"],
        default="low",
        help="Default text verbosity for ALL stages (Responses API only).",
    )

    # ===============================================================
    # STAGE-SPECIFIC REASONING & VERBOSITY OVERRIDES (None = use global)
    # ===============================================================
    ap.add_argument(
        "--elicit-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high"],
        default=None,
        help="Reasoning effort for elicitation. Default: uses --reasoning-effort.",
    )
    ap.add_argument(
        "--elicit-text-verbosity",
        choices=["low", "medium", "high"],
        default=None,
        help="Text verbosity for elicitation. Default: uses --text-verbosity.",
    )
    ap.add_argument(
        "--ner-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high"],
        default=None,
        help="Reasoning effort for NER. Default: uses --reasoning-effort.",
    )
    ap.add_argument(
        "--ner-text-verbosity",
        choices=["low", "medium", "high"],
        default=None,
        help="Text verbosity for NER. Default: uses --text-verbosity.",
    )
    ap.add_argument(
        "--self-rag-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high"],
        default=None,
        help="Reasoning effort for Self-RAG. Default: uses --reasoning-effort.",
    )
    ap.add_argument(
        "--self-rag-text-verbosity",
        choices=["low", "medium", "high"],
        default=None,
        help="Text verbosity for Self-RAG. Default: uses --text-verbosity.",
    )

    # ===============================================================
    # SIMILARITY / EMBEDDINGS (always uses its own model, not cascaded)
    # ===============================================================
    ap.add_argument("--use-similarity", type=_str2bool, default=True)

    ap.add_argument(
        "--similarity-provider",
        choices=["openai", "local"],
        default="openai",
        help="Embedding backend for similarity: openai embeddings or local sentence-transformers.",
    )
    ap.add_argument(
        "--similarity-generation-model",
        default="text-embedding-3-small",  # ALWAYS defaults to this, never cascades
        help="OpenAI embeddings model name (used when --similarity-provider=openai).",
    )
    ap.add_argument(
        "--similarity-local-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformers model name (used when --similarity-provider=local).",
    )

    ap.add_argument("--similarity-embed-with-context", type=_str2bool, default=False)
    ap.add_argument("--similarity-parent-context-words", type=int, default=30)
    ap.add_argument(
        "--similarity-mode",
        choices=["online", "batch"],
        default=None,  # will cascade from --mode
        help="Similarity LLM execution mode. Default: follows --mode.",
    )

    ap.add_argument("--similarity-threshold", type=float, default=0.9)
    ap.add_argument("--similarity-top-k", type=int, default=5)
    ap.add_argument("--similarity-embed-batch-size", type=int, default=2048)
    ap.add_argument("--similarity-compare-batch", type=int, default=256)

    ap.add_argument(
        "--similarity-action",
        choices=["reject", "llm"],
        default="llm",
        help="If max similarity >= threshold: reject directly, or ask an LLM to confirm duplicate.",
    )
    ap.add_argument(
        "--similarity-filter-model-key",
        default=None,
        help="Model key for similarity LLM filter. Default: uses --ner-model-key (which cascades from --model-key).",
    )
    ap.add_argument(
        "--similarity-embed-mode",
        choices=["online", "batch"],
        default=None,  # will cascade from --mode
        help="How to generate OpenAI embeddings for similarity. Default: follows --mode.",
    )

    # Retry controls (finite)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--retry-sleep", type=float, default=5.0, help="Base retry delay seconds.")
    ap.add_argument("--retry-backoff", type=float, default=2.0, help="Backoff multiplier.")
    ap.add_argument("--retry-max-sleep", type=float, default=300.0, help="Max retry delay cap seconds.")
    ap.add_argument("--retry-jitter", type=float, default=0.1, help="Jitter fraction (0.1 = +/-10%).")

    # Persona controls
    ap.add_argument("--personas-path", default=None)
    ap.add_argument("--persona", default="scientific_neutral")
    ap.add_argument("--persona-elicit", dest="persona_elicit", default=None)
    ap.add_argument("--persona-ner", dest="persona_ner", default=None)
    ap.add_argument("--persona-self_rag", dest="persona_self_rag", default=None)

    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--progress-metrics", dest="progress_metrics", action="store_true", default=True)
    ap.add_argument("--no-progress-metrics", dest="progress_metrics", action="store_false")

    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reset-working", action="store_true")

    # Parallel pipeline
    ap.add_argument("--pipeline-parallel", type=_str2bool, default=False)

    ap.add_argument("--selfrag-workers", type=int, default=2)
    ap.add_argument("--outline-workers", type=int, default=2)
    ap.add_argument("--elicit-workers", type=int, default=10)
    ap.add_argument("--ner-workers", type=int, default=4)
    ap.add_argument("--sim-workers", type=int, default=4)
    ap.add_argument(
        "--inflight-per-worker",
        type=int,
        default=1,
        help="How many concurrent in-flight jobs each stage worker keeps running (default: 1).",
    )
    ap.add_argument(
        "--online-buffer-size",
        type=int,
        default=300,
        help="Buffer size for JSONL writes in pipeline-parallel mode (default: 100)",
    )

    ap.add_argument("--reconcile-interval", type=float, default=3.0)

    ap.add_argument(
        "--max-dispatcher-inflight",
        type=int,
        default=0,
        help="Max items in outline+elicit queues before dispatcher pauses. "
             "0 = auto (batch_size * 2 when preload-only, unlimited otherwise).",
    )

    ap.add_argument(
        "--online-flush-interval",
        type=float,
        default=300.0,
        help="Flush interval in seconds for buffered JSONL writes",
    )
    ap.add_argument(
        "--queue-buffer-size",
        type=int,
        default=200,
        help="Buffer size for queue state writes",
    )
    ap.add_argument(
        "--queue-flush-interval",
        type=float,
        default=120.0,
        help="Flush interval for queue state writes",
    )
    #--------
    ap.add_argument(
        "--preload-topics",
        default=None,
        help="Path to file with topics (one per line) to pre-load into queue at hop=0. "
            "Use with --use-ner false --use-similarity false --max-depth 0 for fixed generation.",
    )

    ap.add_argument(
        "--preload-only",
        type=_str2bool,
        default=False,
        help="When True + --preload-topics set: ONLY generate articles for topics in the file. "
             "No seed article, no BFS expansion. NER/similarity still run as analysis but "
             "children are never enqueued.",
    )

    args = ap.parse_args()

    # Respect explicit user overrides
    if getattr(args, "preload_topics", None):
        # Only set if user didn't explicitly provide
        if args.use_ner is None:
            args.use_ner = False
        if args.use_similarity is None:
            args.use_similarity = False
        if args.max_depth is None:
            args.max_depth = 0

    # ===============================================================
    # CASCADING DEFAULTS RESOLUTION
    # ===============================================================

    # --- Model keys: cascade from --model-key ---
    if args.elicit_model_key is None:
        args.elicit_model_key = args.model_key
    if args.ner_model_key is None:
        args.ner_model_key = args.model_key
    if args.self_rag_model_key is None:
        args.self_rag_model_key = args.model_key

    # --- Similarity filter model: cascade from --ner-model-key (which cascaded from --model-key) ---
    if args.similarity_filter_model_key is None:
        args.similarity_filter_model_key = args.ner_model_key

    # --- Temperature: cascade from global --temperature ---
    if args.elicit_temperature is None:
        args.elicit_temperature = args.temperature
    if args.ner_temperature is None:
        args.ner_temperature = args.temperature
    if args.self_rag_temperature is None:
        args.self_rag_temperature = args.temperature

    # --- Top-p: cascade from global --top-p ---
    if args.elicit_top_p is None:
        args.elicit_top_p = args.top_p
    if args.ner_top_p is None:
        args.ner_top_p = args.top_p
    if args.self_rag_top_p is None:
        args.self_rag_top_p = args.top_p

    # --- Top-k: cascade from global --top-k ---
    if args.elicit_top_k is None:
        args.elicit_top_k = args.top_k
    if args.ner_top_k is None:
        args.ner_top_k = args.top_k
    if args.self_rag_top_k is None:
        args.self_rag_top_k = args.top_k

    # --- Max tokens: cascade from global --max-tokens (except self-rag which has its own default) ---
    if args.elicit_max_tokens is None:
        args.elicit_max_tokens = args.max_tokens
    if args.ner_max_tokens is None:
        args.ner_max_tokens = args.max_tokens
    # Note: self_rag_max_tokens already has default=2048, only cascade if not set
    if args.self_rag_max_tokens is None:
        args.self_rag_max_tokens = args.max_tokens

    # --- Reasoning effort: cascade from global --reasoning-effort ---
    if args.elicit_reasoning_effort is None:
        args.elicit_reasoning_effort = args.reasoning_effort
    if args.ner_reasoning_effort is None:
        args.ner_reasoning_effort = args.reasoning_effort
    if args.self_rag_reasoning_effort is None:
        args.self_rag_reasoning_effort = args.reasoning_effort

    # --- Text verbosity: cascade from global --text-verbosity ---
    if args.elicit_text_verbosity is None:
        args.elicit_text_verbosity = args.text_verbosity
    if args.ner_text_verbosity is None:
        args.ner_text_verbosity = args.text_verbosity
    if args.self_rag_text_verbosity is None:
        args.self_rag_text_verbosity = args.text_verbosity

    # --- Mode cascading: ner-mode, self-rag-mode, similarity-mode, similarity-embed-mode ---
    if args.ner_mode is None:
        args.ner_mode = args.mode
    if args.self_rag_mode is None:
        args.self_rag_mode = args.mode
    if args.similarity_mode is None:
        args.similarity_mode = args.mode
    if args.similarity_embed_mode is None:
        args.similarity_embed_mode = args.mode

    # Map self-rag mode flag to internal boolean used elsewhere
    args.self_rag_use_batch = (args.self_rag_mode == "batch")



    # ---- Debug: show resolved cascading defaults ----
    if args.debug:
        _dbg("=== Resolved Cascading Defaults ===")
        _dbg(f"  Global model-key: {args.model_key}")
        _dbg(f"    -> elicit-model-key: {args.elicit_model_key}")
        _dbg(f"    -> ner-model-key: {args.ner_model_key}")
        _dbg(f"    -> self-rag-model-key: {args.self_rag_model_key}")
        _dbg(f"    -> similarity-filter-model-key: {args.similarity_filter_model_key}")
        _dbg(f"  Global temperature: {args.temperature}")
        _dbg(f"    -> elicit-temperature: {args.elicit_temperature}")
        _dbg(f"    -> ner-temperature: {args.ner_temperature}")
        _dbg(f"    -> self-rag-temperature: {args.self_rag_temperature}")
        _dbg(f"  Global reasoning-effort: {args.reasoning_effort}")
        _dbg(f"    -> elicit-reasoning-effort: {args.elicit_reasoning_effort}")
        _dbg(f"    -> ner-reasoning-effort: {args.ner_reasoning_effort}")
        _dbg(f"    -> self-rag-reasoning-effort: {args.self_rag_reasoning_effort}")
        _dbg(f"  Global text-verbosity: {args.text_verbosity}")
        _dbg(f"    -> elicit-text-verbosity: {args.elicit_text_verbosity}")
        _dbg(f"    -> ner-text-verbosity: {args.ner_text_verbosity}")
        _dbg(f"    -> self-rag-text-verbosity: {args.self_rag_text_verbosity}")
        _dbg(f"  Global mode: {args.mode}")
        _dbg(f"    -> ner-mode: {args.ner_mode}")
        _dbg(f"    -> self-rag-mode: {args.self_rag_mode}")
        _dbg(f"    -> similarity-mode: {args.similarity_mode}")
        _dbg(f"    -> similarity-embed-mode: {args.similarity_embed_mode}")
        _dbg(f"  Embeddings model (NEVER cascades): {args.similarity_generation_model}")
        _dbg("===================================")

    # ---- paths & timing ----
    out_dir = _ensure_output_dir(args.output_dir)
    paths = _build_paths(out_dir)

    run_start_utc = datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    perf_start = time.perf_counter()

    minimal_run_meta = {
        "timestamp_utc": run_start_utc,
        "seed": args.seed,
        "output_dir": out_dir,
        "domain": args.domain,
        "mode": args.mode,
        "elicitation_strategy": args.elicitation_strategy,
        "ner_strategy": args.ner_strategy,
        "self_rag_enabled": bool(args.self_rag),
        "two_stage_elicit": bool(args.two_stage_elicit),
        "status": "starting",
        "use_ner": bool(args.use_ner),
        "ner_mode": args.ner_mode,
        "ner_chunk_size": args.ner_chunk_size,
        "ner_conf_threshold": args.ner_conf_threshold,
        "elicit_conf_threshold": args.elicit_conf_threshold,
        "ner_behavior": "strict_no_expand_on_parse_failure",
        "llmpedia_crawling_duration_s": None,
        "llmpedia_crawling_duration_ms": None,
        "cascading_defaults": {
            "global_model_key": args.model_key,
            "global_reasoning_effort": args.reasoning_effort,
            "global_text_verbosity": args.text_verbosity,
            "global_temperature": args.temperature,
        },
        "args_raw": vars(args),
    }

    try:
        with open(paths["run_meta_json"], "w", encoding="utf-8") as f:
            json.dump(minimal_run_meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        _dbg(f"[run-meta-start] failed to write {paths['run_meta_json']}: {e!r}")

    # ---- personas ----
    personas = _load_personas(args.personas_path)
    elicit_persona_name = _resolve_stage_persona_name(args, "elicit")
    ner_persona_name = _resolve_stage_persona_name(args, "ner")
    selfrag_persona_name = _resolve_stage_persona_name(args, "self_rag")

    args.persona_elicit_block = _get_persona_block(personas, elicit_persona_name, "elicit")
    args.persona_ner_block = _get_persona_block(personas, ner_persona_name, "ner")
    args.persona_self_rag_block = _get_persona_block(personas, selfrag_persona_name, "self_rag")

    # ---- model configs from settings.MODELS ----
    el_cfg = _cfg_from_key(args.elicit_model_key, args.timeout)
    ner_cfg = _cfg_from_key(args.ner_model_key, args.timeout)
    args._ner_provider = getattr(ner_cfg, "provider", "openai")
    self_rag_cfg = _cfg_from_key(args.self_rag_model_key, args.timeout)

    # Stage overrides (now using the resolved cascaded values)
    _apply_stage("elicit", el_cfg, args)
    _apply_stage("self_rag", self_rag_cfg, args)
    _apply_stage("ner", ner_cfg, args)

    # Responses API guards
    _strip_responses_sampling_if_disallowed(el_cfg)
    _strip_responses_sampling_if_disallowed(self_rag_cfg)
    _strip_responses_sampling_if_disallowed(ner_cfg)

    # NER schema setup
    if (getattr(ner_cfg, "provider", "") or "").strip().lower() == "openai":
        ner_strategy = (args.ner_strategy or "").strip().lower()
        desired_schema = NER_SCHEMA_CAL if ("calib" in ner_strategy) else NER_SCHEMA_BASE

        ner_cfg.extra_inputs = getattr(ner_cfg, "extra_inputs", None) or {}
        if getattr(ner_cfg, "use_responses_api", False):
            ner_cfg.extra_inputs.setdefault("text", {})
            ner_cfg.extra_inputs["text"]["format"] = {
                "type": "json_schema",
                "name": "ner_phrases",
                "strict": True,
                "schema": desired_schema,
            }
        else:
            ner_cfg.extra_inputs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "ner_phrases",
                    "strict": True,
                    "schema": desired_schema,
                },
            }

    # Queue
    queue = JsonQueue(
        paths["queue_json"],
        paths["queue_jsonl"],
        max_retries=args.max_retries,
        retry_sleep=args.retry_sleep,
        retry_backoff=args.retry_backoff,
        retry_max_sleep=args.retry_max_sleep,
        retry_jitter=args.retry_jitter,
        buffer_size=int(args.queue_buffer_size or 100),
        flush_interval=float(args.queue_flush_interval or 5.0),
    )

    # ---- run the pipeline ----
    if args.mode == "batch":
        run_batch_parallel(args, paths, el_cfg, ner_cfg, self_rag_cfg, queue)
    else:
        run_online_parallel(args, paths, el_cfg, ner_cfg, self_rag_cfg, queue)

    # ---- duration ----
    dur = time.perf_counter() - perf_start
    dur_ms = int(dur * 1000)

    # ---- rich run_meta ----
    models_meta: Dict[str, Any] = {
        "elicitation": {
            "provider": getattr(el_cfg, "provider", "openai"),
            "model": getattr(el_cfg, "model", None),
            "use_responses_api": getattr(el_cfg, "use_responses_api", False),
            "temperature": getattr(el_cfg, "temperature", None),
            "top_p": getattr(el_cfg, "top_p", None),
            "top_k": getattr(el_cfg, "top_k", None),
            "max_tokens": getattr(el_cfg, "max_tokens", None),
            "extra_inputs": getattr(el_cfg, "extra_inputs", None),
            "reasoning_effort": args.elicit_reasoning_effort,
            "text_verbosity": args.elicit_text_verbosity,
        },
        "ner": {
            "provider": getattr(ner_cfg, "provider", "openai"),
            "model": getattr(ner_cfg, "model", None),
            "use_responses_api": getattr(ner_cfg, "use_responses_api", False),
            "temperature": getattr(ner_cfg, "temperature", None),
            "top_p": getattr(ner_cfg, "top_p", None),
            "top_k": getattr(ner_cfg, "top_k", None),
            "max_tokens": getattr(ner_cfg, "max_tokens", None),
            "extra_inputs": getattr(ner_cfg, "extra_inputs", None),
            "reasoning_effort": args.ner_reasoning_effort,
            "text_verbosity": args.ner_text_verbosity,
        },
        "self_rag": {
            "provider": getattr(self_rag_cfg, "provider", "openai"),
            "model": getattr(self_rag_cfg, "model", None),
            "use_responses_api": getattr(self_rag_cfg, "use_responses_api", False),
            "temperature": getattr(self_rag_cfg, "temperature", None),
            "top_p": getattr(self_rag_cfg, "top_p", None),
            "top_k": getattr(self_rag_cfg, "top_k", None),
            "max_tokens": getattr(self_rag_cfg, "max_tokens", None),
            "extra_inputs": getattr(self_rag_cfg, "extra_inputs", None),
            "reasoning_effort": args.self_rag_reasoning_effort,
            "text_verbosity": args.self_rag_text_verbosity,
        },
        "similarity_embeddings": {
            "provider": args.similarity_provider,
            "model": (
                args.similarity_local_model
                if args.similarity_provider == "local"
                else args.similarity_generation_model
            ),
            "note": "Embeddings model NEVER cascades from --model-key",
        },
        "similarity_filter": {
            "model_key": args.similarity_filter_model_key,
            "action": args.similarity_action,
        },
    }

    run_meta = {
        "timestamp_utc": run_start_utc,
        "seed": args.seed,
        "output_dir": out_dir,
        "domain": args.domain,
        "mode": args.mode,
        "elicitation_strategy": args.elicitation_strategy,
        "ner_strategy": args.ner_strategy,
        "two_stage_elicit": bool(args.two_stage_elicit),
        "self_rag_enabled": bool(args.self_rag),
        "self_rag_mode": args.self_rag_mode,
        "self_rag_use_batch": bool(args.self_rag_use_batch),
        "max_depth": args.max_depth,
        "max_subjects": args.max_subjects,
        "batch_size": args.batch_size,
        "concurrency": args.concurrency,
        "personas": {
            "elicit": elicit_persona_name,
            "ner": ner_persona_name,
            "self_rag": selfrag_persona_name,
        },
        "cascading_defaults": {
            "global_model_key": args.model_key,
            "global_reasoning_effort": args.reasoning_effort,
            "global_text_verbosity": args.text_verbosity,
            "global_temperature": args.temperature,
            "global_top_p": args.top_p,
            "global_top_k": args.top_k,
            "global_max_tokens": args.max_tokens,
            "resolved": {
                "elicit_model_key": args.elicit_model_key,
                "ner_model_key": args.ner_model_key,
                "self_rag_model_key": args.self_rag_model_key,
                "similarity_filter_model_key": args.similarity_filter_model_key,
                "elicit_reasoning_effort": args.elicit_reasoning_effort,
                "ner_reasoning_effort": args.ner_reasoning_effort,
                "self_rag_reasoning_effort": args.self_rag_reasoning_effort,
                "elicit_text_verbosity": args.elicit_text_verbosity,
                "ner_text_verbosity": args.ner_text_verbosity,
                "self_rag_text_verbosity": args.self_rag_text_verbosity,
            },
        },
        "models": models_meta,
        "args_raw": vars(args),
        "duration_s": dur,
        "llmpedia_crawling_duration_s": dur,
        "llmpedia_crawling_duration_ms": dur_ms,
        "status": "finished",
        "use_ner": bool(args.use_ner),
        "ner_mode": args.ner_mode,
        "ner_chunk_size": args.ner_chunk_size,
        "ner_conf_threshold": args.ner_conf_threshold,
        "elicit_conf_threshold": args.elicit_conf_threshold,
        "ner_behavior": "strict_no_expand_on_parse_failure",
        "ner_schema_used": (
            "NER_SCHEMA_CAL" if ("calib" in (args.ner_strategy or "").lower()) else "NER_SCHEMA_BASE"
        ),
    }

    try:
        with open(paths["run_meta_json"], "w", encoding="utf-8") as f:
            json.dump(run_meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        _dbg(f"[run-meta-final] failed to write {paths['run_meta_json']}: {e!r}")

if __name__ == "__main__":
    main()
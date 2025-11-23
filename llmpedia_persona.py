# llmpedia_persona.py
from __future__ import annotations
import argparse
import datetime
import traceback

import json
import os
import re
import sqlite3
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Set, Optional, Dict, Any

from dotenv import load_dotenv

load_dotenv()

# ---------------- tiny utils & locks ----------------

_jsonl_lock = threading.Lock()
_seen_canon_lock = threading.Lock()


def _handle_batch_wave_exception(
    e: Exception,
    wave_idx: int,
    batch: List[Tuple[str, int]],
    args,
    paths,
    seen_canon_keys: Set[str],
    context: str,
):
    """
    Called when an OpenAI SDK call fails for an entire wave.
    It logs the error and applies mark_pending_on_error to all (subject, hop)
    in this wave, honoring max_retries.
    """
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
        # Don't let logging failures crash the run
        pass

    for subject, hop in batch:
        mark_pending_on_error(paths["queue_sqlite"], subject, hop, args.max_retries)

    _persist_seen_canon(paths, seen_canon_keys)



def _append_jsonl(path: str, obj: dict):
    """
    Append a JSON object as a line to a .jsonl file.
    Safely creates parent directory if it exists.
    """
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    with _jsonl_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


def _dbg(msg: str):
    print(msg, flush=True)


def _str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _append_footer_to_msgs(msgs: List[dict], footer: str, target: str = "user") -> List[dict]:
    if not footer:
        return msgs
    idx = None
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == target and isinstance(msgs[i].get("content"), str):
            idx = i
            break
    if idx is not None:
        msgs[idx]["content"] = msgs[idx]["content"].rstrip() + "\n\n" + footer
    else:
        msgs.append({"role": target, "content": footer})
    return msgs


def _append_block_to_msgs(msgs: List[dict], block: str, target: str = "user") -> List[dict]:
    if not block:
        return msgs
    idx = None
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == target and isinstance(msgs[i].get("content"), str):
            idx = i
            break
    if idx is not None:
        msgs[idx]["content"] = msgs[idx]["content"].rstrip() + "\n\n" + block
    else:
        msgs.append({"role": target, "content": block})
    return msgs


def _unwrap_text(resp) -> str:
    """
    Best-effort extraction of text from various LLM client styles.
    """
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        for k in ("text", "output_text", "content", "message", "response"):
            v = resp.get(k)
            if isinstance(v, str):
                return v
        ch = resp.get("choices")
        if isinstance(ch, list) and ch:
            c0 = ch[0] or {}
            msg = c0.get("message") or {}
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
            if isinstance(c0.get("text"), str):
                return c0["text"]
        if isinstance(resp.get("_raw"), str):
            return resp["_raw"]
        if isinstance(resp.get("raw"), str):
            return resp["raw"]
        if isinstance(resp.get("raw"), dict):
            return _unwrap_text(resp["raw"])
    return ""


# ---------------- repo imports ----------------

from processing_queue import (
    init_cache as procq_init_cache,
    enqueue_subjects_processed as procq_enqueue,
    get_thread_queue_conn as procq_get_thread_conn,
    _canonical_key as canon_key_from_queue,
)
from db_models import open_queue_db, queue_has_rows, reset_working_to_pending
from settings import settings
from llm.factory import make_llm_from_config
from prompter_parser import (
    build_elicitation_messages_for_subject,
    build_ner_messages_for_phrases,
)

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
    Load personas from a JSON file; fall back to built-in defaults if missing
    or invalid.
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
    """
    Decide which persona name to use for a stage (elicit/ner/self_rag):
      1) stage-specific flag (e.g. persona_elicit)
      2) global --persona
      3) fallback 'scientific_neutral'
    """
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

    - If persona_name not found in personas.json → fall back to internal defaults.
    - If persona_name found but malformed / missing blocks[stage] → fall back to internal defaults.
    """
    if persona_name not in personas:
        _dbg(f"[persona] unknown persona {persona_name!r}; falling back to internal 'scientific_neutral'")
        persona_name = "scientific_neutral"

    entry = personas.get(persona_name, {})
    blocks = entry.get("blocks") or {}

    block = blocks.get(stage)
    if isinstance(block, str) and block.strip():
        return block

    # Hard fallback: always use _FALLBACK_PERSONAS for the actual text
    fallback_entry = _FALLBACK_PERSONAS.get("scientific_neutral", {})
    fb_blocks = fallback_entry.get("blocks") or {}
    fb_block = fb_blocks.get(stage, "")
    if not fb_block:
        return ""
    _dbg(f"[persona] using internal fallback persona text for stage={stage}")
    return fb_block


# ---------------- paths + DB helpers ----------------


def _ensure_output_dir(base_dir: Optional[str]) -> str:
    out = base_dir or os.path.join("runs", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    return out


def _build_paths(out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    batches_dir = os.path.join(out_dir, "batches")
    os.makedirs(batches_dir, exist_ok=True)
    return {
        "queue_sqlite": os.path.join(out_dir, "queue.sqlite"),
        "articles_sqlite": os.path.join(out_dir, "llmpedia_articles.sqlite"),
        "queue_jsonl": os.path.join(out_dir, "queue.jsonl"),
        "articles_jsonl": os.path.join(out_dir, "articles.jsonl"),
        "queue_json": os.path.join(out_dir, "queue.json"),
        "articles_json": os.path.join(out_dir, "articles.json"),
        "errors_log": os.path.join(out_dir, "errors.log"),
        "run_meta_json": os.path.join(out_dir, "run_meta.json"),
        "seen_state_json": os.path.join(out_dir, "seen_canon_keys.json"),
        "ner_decisions_jsonl": os.path.join(out_dir, "ner_decisions.jsonl"),
        "ner_lowconf_jsonl": os.path.join(out_dir, "ner_lowconf.jsonl"),
        "elicit_lowconf_jsonl": os.path.join(out_dir, "elicit_lowconf.jsonl"),
        "self_rag_log_jsonl": os.path.join(out_dir, "self_rag_log.jsonl"),
        # latest wave pointer
        "batch_input_jsonl": os.path.join(batches_dir, "batch_input_latest.jsonl"),
        "batches_dir": batches_dir,
    }


_thread_local = threading.local()


def get_thread_articles_conn(db_path: str) -> sqlite3.Connection:
    key = f"llmpedia_articles_conn__{db_path}"
    conn = getattr(_thread_local, key, None)
    if conn is None:
        conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=15000;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llmpedia_articles(
              subject            TEXT PRIMARY KEY,
              wikitext           TEXT,
              hop                INT,
              model_name         TEXT,
              created_at         DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
              overall_confidence REAL
            );
        """
        )
        setattr(_thread_local, key, conn)
    return conn


def open_llmpedia_db(path: str) -> sqlite3.Connection:
    return get_thread_articles_conn(path)


def write_article_record(
    conn: sqlite3.Connection,
    subject: str,
    hop: int,
    model: str,
    wikitext: str,
    overall_confidence: Optional[float],
):
    if not isinstance(wikitext, str) or not wikitext.strip():
        return
    with conn:
        conn.execute(
            """
            INSERT INTO llmpedia_articles(subject, wikitext, hop, model_name, overall_confidence)
            VALUES(?,?,?,?,?)
            ON CONFLICT(subject) DO UPDATE SET
              wikitext=excluded.wikitext,
              hop=excluded.hop,
              model_name=excluded.model_name,
              overall_confidence=excluded.overall_confidence
        """,
            (subject, wikitext, hop, model, overall_confidence),
        )


# ---------------- SQLite helpers ----------------


def _is_sqlite_lock(err: Exception) -> bool:
    s = str(err).lower()
    return ("database is locked" in s) or ("database is busy" in s) or ("database table is locked" in s)


def _with_sqlite_retry(fn, *, tries=12, base=0.05, factor=1.7):
    delay = base
    last = None
    for _ in range(tries):
        try:
            return fn()
        except sqlite3.OperationalError as e:
            last = e
            if _is_sqlite_lock(e):
                time.sleep(delay)
                delay *= factor
                continue
            raise
    raise last


def _exec_retry(conn: sqlite3.Connection, sql: str, params=()):
    def _do():
        with conn:
            conn.execute(sql, params)

    return _with_sqlite_retry(_do)


def _enqueue_retry(db_path: str, items):
    return _with_sqlite_retry(lambda: procq_enqueue(db_path, items))


def mark_done_threadsafe(queue_db_path: str, subject: str, hop: int):
    conn = procq_get_thread_conn(queue_db_path)
    _exec_retry(
        conn,
        "UPDATE queue SET status='done' WHERE subject=? AND hop=? AND status='working'",
        (subject, hop),
    )


def mark_pending_on_error(queue_db_path: str, subject: str, hop: int, max_retries: int):
    """
    On error, bump retries and either set status='pending' (if retries < max_retries)
    or status='failed' (once retries >= max_retries).
    """
    conn = procq_get_thread_conn(queue_db_path)

    effective_max = max(1, int(max_retries))

    def _do():
        with conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT retries FROM queue WHERE subject=? AND hop=? AND status='working'",
                (subject, hop),
            )
            row = cur.fetchone()
            if not row:
                cur.close()
                return
            current_retries = row[0] or 0
            new_retries = current_retries + 1

            if new_retries >= effective_max:
                new_status = "failed"
            else:
                new_status = "pending"

            cur.execute(
                "UPDATE queue SET status=?, retries=? WHERE subject=? AND hop=? AND status='working'",
                (new_status, new_retries, subject, hop),
            )
            cur.close()

    _with_sqlite_retry(_do)


def _claim_pending_batch(conn: sqlite3.Connection, max_depth: int, claim_n: int) -> List[Tuple[str, int]]:
    """
    Claim up to `claim_n` pending subjects, marking them as 'working'.
    """

    def _do():
        with conn:
            cur = conn.cursor()
            if max_depth == 0:
                cur.execute(
                    """
                    SELECT rowid, subject, hop
                    FROM queue
                    WHERE status='pending'
                    ORDER BY hop, created_at
                    LIMIT ?
                    """,
                    (claim_n,),
                )
            else:
                cur.execute(
                    """
                    SELECT rowid, subject, hop
                    FROM queue
                    WHERE status='pending' AND hop<=?
                    ORDER BY hop, created_at
                    LIMIT ?
                    """,
                    (max_depth, claim_n),
                )
            rows = cur.fetchall()
            if not rows:
                cur.close()
                return []

            rowids = [r[0] for r in rows]
            qmarks = ",".join("?" for _ in rowids)
            cur.execute(
                f"UPDATE queue SET status='working' WHERE rowid IN ({qmarks})",
                rowids,
            )
            cur.close()
            return [(r[1], r[2]) for r in rows]

    return _with_sqlite_retry(_do)


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


# ---------------- NER parsing ----------------


def _parse_ner_output(raw) -> List[dict]:
    txt = _unwrap_text(raw)
    if not isinstance(txt, str):
        return []
    txt = txt.strip()
    if not txt:
        return []

    obj = None
    try:
        obj = json.loads(txt)
    except Exception:
        obj = None

    if isinstance(obj, dict) and isinstance(obj.get("phrases"), list):
        phrases = obj["phrases"]
        out: List[dict] = []
        for d in phrases:
            if not isinstance(d, dict):
                continue
            phrase = d.get("phrase")
            if not isinstance(phrase, str) or not phrase.strip():
                continue
            is_ne = bool(d.get("is_ne"))
            conf = d.get("confidence", None)
            if isinstance(conf, (int, float)):
                try:
                    conf = float(conf)
                except Exception:
                    conf = None
            else:
                conf = None
            out.append({"phrase": phrase.strip(), "is_ne": is_ne, "confidence": conf})
        return out

    # fallback: JSONL lines with {"subject": "..."}
    decisions: List[dict] = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        subj = row.get("subject")
        if not isinstance(subj, str) or not subj.strip():
            continue
        decisions.append(
            {
                "phrase": subj.strip(),
                "is_ne": True,
                "confidence": 1.0,
            }
        )
    return decisions


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
    """
    Build a human-readable SELF-RAG context block from the parsed JSON.

    Be robust to the model returning non-string types for `predicate` or `object`,
    e.g. lists, numbers, or nested structures. We coerce to strings safely.
    """
    def _to_str(x) -> str:
        # Convert various possible types into a clean string
        if x is None:
            return ""
        if isinstance(x, str):
            return x.strip()
        if isinstance(x, (list, tuple)):
            # join list-like into comma-separated string
            return ", ".join(_to_str(el) for el in x if el is not None)
        # fallback: just string cast
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



# ---------------- Stage controls ----------------


def _apply_stage(which: str, cfg, args):
    """
    Stage-specific overrides for elicitation/NER.

    For Responses API:
      - Uses per-stage reasoning/text controls if provided,
        else falls back to global ones.
    For chat-style:
      - Applies temperature/top_p/top_k and max_tokens.
    """
    if getattr(cfg, "use_responses_api", False):
        cfg.temperature = None
        cfg.top_p = None
        cfg.top_k = None
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
    else:
        t = getattr(args, f"{which}_temperature", None)
        tp = getattr(args, f"{which}_top_p", None)
        tk = getattr(args, f"{which}_top_k", None)
        if t is not None:
            cfg.temperature = t
        if tp is not None:
            cfg.top_p = tp
        if tk is not None:
            cfg.top_k = tk

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
            f"[mode=batch] {label} model provider must be 'openai', "
            f"got provider={provider!r}. Please choose an OpenAI model for batch mode."
        )


# ---------------- NER client thread-local ----------------

_ner_thread_local = threading.local()


def get_thread_ner_client(ner_cfg):
    """
    Thread-local NER client so we don't recreate the model per subject.
    """
    key = "ner_client"
    client = getattr(_ner_thread_local, key, None)
    if client is None:
        local_cfg = ner_cfg.model_copy(deep=True)
        client = make_llm_from_config(local_cfg)
        setattr(_ner_thread_local, key, client)
    return client


# ---------------- shared helpers: Self-RAG & message building ----------------


def _run_self_rag_for_subject(
    subject: str,
    hop: int,
    root_topic: str,
    args,
    self_rag_cfg,
    self_rag_llm,
    paths,
    persona_block: str,
    wave_idx: Optional[int] = None,
) -> Optional[dict]:
    """
    Run Self-RAG for a single subject, returning a normalized context dict or None.
    """
    sr_msgs = _build_self_rag_messages(subject, root_topic, persona_block)
    ctx: Optional[dict] = None
    error: Optional[str] = None

    try:
        try:
            sr_resp = self_rag_llm(sr_msgs, json_schema=SELF_RAG_SCHEMA, timeout=args.timeout)
        except TypeError:
            sr_resp = self_rag_llm(sr_msgs)

        sr_obj: Any = None
        if isinstance(sr_resp, dict) and ("summary" in sr_resp and "salient_facts" in sr_resp):
            sr_obj = sr_resp
        else:
            try:
                txt = _unwrap_text(sr_resp)
                if txt:
                    sr_obj = json.loads(txt)
            except Exception:
                sr_obj = None

        if isinstance(sr_obj, dict):
            ctx = {
                "summary": sr_obj.get("summary") or "",
                "aliases": sr_obj.get("aliases") or [],
                "salient_facts": sr_obj.get("salient_facts") or [],
            }
    except Exception as e:
        error = repr(e)
        ctx = None

    log_rec = {
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "subject": subject,
        "hop": hop,
        "model": getattr(self_rag_cfg, "model", None),
        "parsed": ctx,
    }
    if wave_idx is not None:
        log_rec["wave"] = wave_idx
    if error is not None:
        log_rec["error"] = error

    _append_jsonl(paths["self_rag_log_jsonl"], log_rec)

    return ctx


def _build_llmpedia_messages_for_subject(
    subject: str,
    hop: int,
    args,
    root_topic: str,
    persona_block: str,
    self_rag_context: Optional[dict],
) -> List[dict]:
    """
    Shared helper to build elicitation messages for a subject, with optional
    Self-RAG context and footer logic applied.
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
        persona_block=persona_block,   # <- use the parameter, which is args.persona_elicit_block at callsite
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
                f"Additional, very important guidance about categories for this LLMPedia article on {subject}:\n"
                "- If the entity is widely known, aim for about 50 distinct, precise categories.\n"
                "- If the entity is not widely known, aim for about 10 strong categories.\n"
                "- Include categories that capture closely related organizations, events, places, works, "
                "technologies and concepts that are strongly associated with this entity.\n"
                "- Do NOT invent random or obviously speculative categories."
            )
        messages = _append_footer_to_msgs(messages, footer, target=args.footer_location)

    return messages


# ---------------- snapshot helpers ----------------


def _snapshot_queue_and_articles(paths: dict):
    """
    Write JSON snapshots of the queue and articles tables for post-run inspection.
    Also writes articles_jsonl so it's always populated and in sync with JSON.
    """
    # queue snapshot
    conn = sqlite3.connect(paths["queue_sqlite"])
    cur = conn.cursor()
    cur.execute("SELECT subject, hop, status, retries, created_at FROM queue ORDER BY hop, subject")
    rows = cur.fetchall()
    queue_records = [
        {
            "subject": s,
            "hop": h,
            "status": st,
            "retries": r,
            "created_at": ts,
        }
        for (s, h, st, r, ts) in rows
    ]
    with open(paths["queue_json"], "w", encoding="utf-8") as f:
        json.dump(queue_records, f, ensure_ascii=False, indent=2)
    conn.close()

    # articles snapshot
    conn = sqlite3.connect(paths["articles_sqlite"])
    cur = conn.cursor()
    cur.execute(
        "SELECT subject, wikitext, hop, model_name, overall_confidence, created_at "
        "FROM llmpedia_articles ORDER BY subject"
    )
    arows = cur.fetchall()
    article_records = [
        {
            "subject": s,
            "wikitext": wt,
            "hop": h,
            "model": m,
            "overall_confidence": oc,
            "created_at": ts,
        }
        for (s, wt, h, m, oc, ts) in arows
    ]
    with open(paths["articles_json"], "w", encoding="utf-8") as f:
        json.dump(article_records, f, ensure_ascii=False, indent=2)
    # ensure articles_jsonl matches the same records
    with open(paths["articles_jsonl"], "w", encoding="utf-8") as f:
        for rec in article_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    conn.close()


# ---------------- ONLINE MODE PIPELINE ----------------


def _seed_or_resume_queue(args, paths, qdb):
    if args.resume:
        if not queue_has_rows(qdb):
            for s, kept_hop, outcome in procq_enqueue(paths["queue_sqlite"], [(args.seed, 0)]):
                if outcome in ("inserted", "hop_reduced"):
                    _append_jsonl(paths["queue_jsonl"], {"subject": s, "hop": kept_hop, "event": outcome})
        else:
            if args.reset_working:
                n = reset_working_to_pending(qdb)
                _dbg(f"[resume] reset {n} working→pending")
    else:
        for s, kept_hop, outcome in procq_enqueue(paths["queue_sqlite"], [(args.seed, 0)]):
            if outcome in ("inserted", "hop_reduced"):
                _append_jsonl(paths["queue_jsonl"], {"subject": s, "hop": kept_hop, "event": outcome})


def _load_seen_canon(paths) -> Set[str]:
    seen_canon_keys: Set[str] = set()
    if os.path.exists(paths["seen_state_json"]):
        try:
            with open(paths["seen_state_json"], "r", encoding="utf-8") as f:
                arr = json.load(f) or []
                if isinstance(arr, list):
                    seen_canon_keys.update([str(x) for x in arr])
        except Exception:
            pass
    return seen_canon_keys


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


def _post_article_processing(
    args,
    paths,
    el_cfg,
    ner_cfg,
    subject: str,
    hop: int,
    wikitext: str,
    seen_canon_keys: Set[str],
):
    """
    Store article, run NER on outgoing links/categories, enqueue next subjects,
    and log diagnostics.
    """
    # store article
    overall_conf = None
    write_article_record(
        get_thread_articles_conn(paths["articles_sqlite"]),
        subject,
        hop,
        el_cfg.model,
        wikitext,
        overall_conf,
    )

    links_from_markup = _extract_link_targets_from_wikitext(wikitext)
    cat_from_markup = _extract_categories_from_wikitext(wikitext)

    candidates_for_ner: List[str] = []
    seen_candidates: Set[str] = set()

    def _add_candidate(candidate: str):
        c = (candidate or "").strip()
        if not c:
            return
        if c in seen_candidates:
            return
        seen_candidates.add(c)
        candidates_for_ner.append(c + "\n")

    elicit_conf_th = getattr(args, "elicit_conf_threshold", 0.0)
    for raw_title in links_from_markup:
        base_title, link_conf = _split_title_and_conf(raw_title)
        if elicit_conf_th > 0.0 and isinstance(link_conf, float) and link_conf < elicit_conf_th:
            lowrec = {
                "stage": "elicitation_link_filter",
                "current_entity": subject,
                "root_subject": args.seed if args.domain == "topic" else None,
                "hop": hop,
                "phrase": base_title,
                "elicitation_confidence": float(link_conf),
                "elicit_conf_threshold": float(elicit_conf_th),
                "passed_threshold": False,
                "rejection_reason": "elicitation_below_conf_threshold",
            }
            _append_jsonl(paths["elicit_lowconf_jsonl"], lowrec)
            continue
        _add_candidate(base_title)

    for c in cat_from_markup:
        _add_candidate(c)

    if args.debug:
        _dbg(
            f"[candidates] {subject} (hop={hop}) → "
            f"links={links_from_markup[:10]}{'…' if len(links_from_markup) > 10 else ''}, "
            f"cats_markup={cat_from_markup[:10]}{'…' if len(cat_from_markup) > 10 else ''}"
        )

    # NER online (thread-local client) with persona
    ner_llm = get_thread_ner_client(ner_cfg)
    next_subjects: List[str] = []
    unique_next: List[str] = []

    if candidates_for_ner:
        ner_messages = build_ner_messages_for_phrases(
            domain=args.domain,
            strategy=args.ner_strategy,
            subject_name=subject,
            seed=args.seed,
            phrases=candidates_for_ner,
            persona_block=args.persona_ner_block,
        )

        if args.debug:
            _dbg(
                f"[NER] input candidates for [{subject}] (hop={hop}): "
                f"{candidates_for_ner[:10]}{'…' if len(candidates_for_ner) > 10 else ''}"
            )
            _dbg(f"\n--- NER PROMPT for [{subject}] (hop={hop}) ---")
            for i, m in enumerate(ner_messages, 1):
                content = m.get("content", "")
                preview = content[:400] if isinstance(content, str) else ""
                _dbg(
                    f"[NER {i:02d}] {m.get('role','?').upper()}: {preview}"
                    f"{'…' if isinstance(content, str) and len(content) > 200 else ''}"
                )
            _dbg("--- END NER PROMPT ---\n")

        try:
            ner_resp = ner_llm(ner_messages, timeout=args.timeout)
        except TypeError:
            ner_resp = ner_llm(ner_messages)

        decisions = _parse_ner_output(ner_resp)

        for d in decisions:
            phrase = d.get("phrase")
            is_ne = bool(d.get("is_ne"))
            conf = d.get("confidence")

            passes_threshold = True
            rejection_reason = None
            if args.ner_conf_threshold > 0.0 and isinstance(conf, (int, float)):
                if conf < args.ner_conf_threshold:
                    passes_threshold = False
                    rejection_reason = "ner_below_conf_threshold"
                    lowrec = {
                        "stage": "ner_conf_filter",
                        "current_entity": subject,
                        "root_subject": args.seed if args.domain == "topic" else None,
                        "hop": hop,
                        "phrase": phrase,
                        "is_ne": is_ne,
                        "confidence": float(conf),
                        "ner_conf_threshold": float(args.ner_conf_threshold),
                        "passed_threshold": False,
                        "rejection_reason": rejection_reason,
                        "ner_strategy": args.ner_strategy,
                        "domain": args.domain,
                        "ner_model": ner_cfg.model,
                    }
                    _append_jsonl(paths["ner_lowconf_jsonl"], lowrec)

            accepted = False
            if not isinstance(phrase, str) or not phrase.strip():
                accepted = False
                if rejection_reason is None:
                    rejection_reason = "invalid_phrase"
            else:
                if is_ne and passes_threshold:
                    accepted = True
                else:
                    if not is_ne and rejection_reason is None:
                        rejection_reason = "ner_is_ne_false"

            record = {
                "subject": subject,
                "phrase": phrase,
                "is_ne": is_ne,
                "confidence": conf,
                "accepted": accepted,
                "ner_model": ner_cfg.model,
                "ner_strategy": args.ner_strategy,
                "domain": args.domain,
            }
            if not accepted and rejection_reason is not None:
                record["rejection_reason"] = rejection_reason
            _append_jsonl(paths["ner_decisions_jsonl"], record)

            if accepted:
                next_subjects.append(phrase.strip())

    # enqueue next subjects
    if next_subjects:
        uniq_canon: Set[str] = set()
        for s in next_subjects:
            canon = canon_key_from_queue(s)
            with _seen_canon_lock:
                if canon in seen_canon_keys:
                    _append_jsonl(
                        paths["ner_lowconf_jsonl"],
                        {
                            "stage": "queue_dedup",
                            "current_entity": subject,
                            "root_subject": args.seed if args.domain == "topic" else None,
                            "hop": hop,
                            "phrase": s,
                            "canonical_key": canon,
                            "passed_threshold": False,
                            "rejection_reason": "queue_canonical_seen",
                        },
                    )
                    continue
                seen_canon_keys.add(canon)
            if canon in uniq_canon:
                _append_jsonl(
                    paths["ner_lowconf_jsonl"],
                    {
                        "stage": "queue_dedup_batch",
                        "current_entity": subject,
                        "root_subject": args.seed if args.domain == "topic" else None,
                        "hop": hop,
                        "phrase": s,
                        "canonical_key": canon,
                        "passed_threshold": False,
                        "rejection_reason": "queue_batch_duplicate",
                    },
                )
                continue
            uniq_canon.add(canon)
            unique_next.append(s)

        results = _enqueue_retry(
            paths["queue_sqlite"],
            [(s, hop + 1) for s in unique_next if (args.max_depth == 0 or hop + 1 <= args.max_depth)],
        )
        for s, kept_hop, outcome in results:
            if outcome in ("inserted", "hop_reduced"):
                _append_jsonl(paths["queue_jsonl"], {"subject": s, "hop": kept_hop, "event": outcome})

    # article JSONL streaming log (optional, final snapshot will overwrite with full set)
    article_record = {
        "subject": subject,
        "hop": hop,
        "wikitext": wikitext,
        "model": el_cfg.model,
        "overall_confidence": overall_conf,
        "links_from_markup": links_from_markup,
        "categories_from_markup": cat_from_markup,
        "ner_candidates": candidates_for_ner,
    }
    _append_jsonl(paths["articles_jsonl"], article_record)


def run_online(args, paths, el_cfg, ner_cfg, self_rag_cfg):
    qdb = open_queue_db(paths["queue_sqlite"])
    open_llmpedia_db(paths["articles_sqlite"])
    procq_init_cache(qdb)

    _seed_or_resume_queue(args, paths, qdb)
    seen_canon_keys = _load_seen_canon(paths)

    el_llm = make_llm_from_config(el_cfg)
    self_rag_llm = make_llm_from_config(self_rag_cfg) if args.self_rag else None

    start = time.perf_counter()
    last_progress_ts = 0.0
    subjects_total = 0

    def _generate_article(subject: str, hop: int):
        try:
            root_topic = args.seed if args.domain == "topic" else subject

            # Self-RAG (online, optional)
            self_rag_context = None
            if args.self_rag and self_rag_llm is not None:
                self_rag_context = _run_self_rag_for_subject(
                    subject=subject,
                    hop=hop,
                    root_topic=root_topic,
                    args=args,
                    self_rag_cfg=self_rag_cfg,
                    self_rag_llm=self_rag_llm,
                    paths=paths,
                    persona_block=args.persona_self_rag_block,
                    wave_idx=None,
                )

            # build elicitation messages with persona
            messages = _build_llmpedia_messages_for_subject(
                subject=subject,
                hop=hop,
                args=args,
                root_topic=root_topic,
                persona_block=args.persona_elicit_block,
                self_rag_context=self_rag_context,
            )

            if args.debug:
                _dbg(f"\n--- LLMPEDIA for [{subject}] (hop={hop}) ---")
                for i, m in enumerate(messages, 1):
                    preview = m["content"][:400] if isinstance(m.get("content"), str) else ""
                    _dbg(
                        f"[{i:02d}] {m['role'].upper()}: {preview}"
                        f"{'…' if isinstance(m.get('content'), str) and len(m['content'])>200 else ''}"
                    )
                _dbg("--- END ---\n")

            try:
                resp = el_llm(messages, timeout=args.timeout)
            except TypeError:
                resp = el_llm(messages)

            wikitext = _unwrap_text(resp).strip()
            if not wikitext:
                wikitext = f"'''{subject}'''\n\nNo article content generated."

            _post_article_processing(
                args,
                paths,
                el_cfg,
                ner_cfg,
                subject,
                hop,
                wikitext,
                seen_canon_keys,
            )

            mark_done_threadsafe(paths["queue_sqlite"], subject, hop)
            return (subject, hop, None)

        except Exception:
            with open(paths["errors_log"], "a", encoding="utf-8") as ef:
                ef.write(f"[{datetime.datetime.now().isoformat()}] subject={subject}\n{traceback.format_exc()}\n")
            mark_pending_on_error(paths["queue_sqlite"], subject, hop, args.max_retries)
            return (subject, hop, "error")

    # BFS loop
    while True:
        if args.progress_metrics:
            now = time.perf_counter()
            if now - last_progress_ts >= 2.0:
                cur = qdb.cursor()
                if args.max_depth == 0:
                    cur.execute("SELECT COUNT(1) FROM queue WHERE status='done'")
                    d = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(1) FROM queue WHERE status='working'")
                    w = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(1) FROM queue WHERE status='pending'")
                    p = cur.fetchone()[0]
                    cur.execute("SELECT COUNT(1) FROM queue WHERE status='failed'")
                    f = cur.fetchone()[0]
                else:
                    cur.execute(
                        "SELECT COUNT(1) FROM queue WHERE status='done' AND hop<=?",
                        (args.max_depth,),
                    )
                    d = cur.fetchone()[0]
                    cur.execute(
                        "SELECT COUNT(1) FROM queue WHERE status='working' AND hop<=?",
                        (args.max_depth,),
                    )
                    w = cur.fetchone()[0]
                    cur.execute(
                        "SELECT COUNT(1) FROM queue WHERE status='pending' AND hop<=?",
                        (args.max_depth,),
                    )
                    p = cur.fetchone()[0]
                    cur.execute(
                        "SELECT COUNT(1) FROM queue WHERE status='failed' AND hop<=?",
                        (args.max_depth,),
                    )
                    f = cur.fetchone()[0]
                t = d + w + p + f
                _dbg(f"[progress] done={d} working={w} pending={p} failed={f} total={t}")
                last_progress_ts = now

        if args.max_subjects and subjects_total >= args.max_subjects:
            _dbg(f"[stop] max-subjects reached ({subjects_total})")
            break

        remaining_cap = (args.max_subjects - subjects_total) if args.max_subjects else None
        claim_n = args.concurrency
        if remaining_cap is not None:
            claim_n = max(1, min(claim_n, remaining_cap))

        batch = _claim_pending_batch(qdb, args.max_depth, max(1, claim_n))
        if not batch:
            cur = qdb.cursor()
            if args.max_depth == 0:
                cur.execute(
                    "SELECT COUNT(1) FROM queue WHERE status IN ('done','working','pending','failed')"
                )
                t = cur.fetchone()[0]
                cur.execute("SELECT COUNT(1) FROM queue WHERE status='done'")
                d = cur.fetchone()[0]
                cur.execute("SELECT COUNT(1) FROM queue WHERE status='working'")
                w = cur.fetchone()[0]
                cur.execute("SELECT COUNT(1) FROM queue WHERE status='pending'")
                p = cur.fetchone()[0]
                cur.execute("SELECT COUNT(1) FROM queue WHERE status='failed'")
                f = cur.fetchone()[0]
            else:
                cur.execute(
                    "SELECT COUNT(1) FROM queue "
                    "WHERE status IN ('done','working','pending','failed') AND hop<=?",
                    (args.max_depth,),
                )
                t = cur.fetchone()[0]
                cur.execute(
                    "SELECT COUNT(1) FROM queue WHERE status='done' AND hop<=?",
                    (args.max_depth,),
                )
                d = cur.fetchone()[0]
                cur.execute(
                    "SELECT COUNT(1) FROM queue WHERE status='working' AND hop<=?",
                    (args.max_depth,),
                )
                w = cur.fetchone()[0]
                cur.execute(
                    "SELECT COUNT(1) FROM queue WHERE status='pending' AND hop<=?",
                    (args.max_depth,),
                )
                p = cur.fetchone()[0]
                cur.execute(
                    "SELECT COUNT(1) FROM queue WHERE status='failed' AND hop<=?",
                    (args.max_depth,),
                )
                f = cur.fetchone()[0]
            if t == 0:
                _dbg("[idle] nothing to do.")
            else:
                _dbg(f"[idle] queue drained: done={d} working={w} pending={p} failed={f} total={t}")
            break

        _dbg(
            f"[path=online-concurrency] subjects={len(batch)} "
            f"workers={min(args.concurrency, len(batch))}"
        )
        results = []
        with ThreadPoolExecutor(max_workers=min(args.concurrency, len(batch))) as pool:
            futs = [pool.submit(_generate_article, s, h) for (s, h) in batch]
            for fut in as_completed(futs):
                results.append(fut.result())
        for _s, _h, err in results:
            if err is None:
                subjects_total += 1
                if args.max_subjects and subjects_total >= args.max_subjects:
                    _dbg(f"[stop] max-subjects reached ({subjects_total})")
                    break

        _persist_seen_canon(paths, seen_canon_keys)

    _snapshot_queue_and_articles(paths)

    dur = time.perf_counter() - start
    _dbg(f"[done-online] finished in {dur:.1f}s → {os.path.dirname(paths['queue_sqlite'])}")


def run_batch(args, paths, el_cfg, ner_cfg, self_rag_cfg):
    """
    Batch mode with persona-aware elicitation, NER, and optional Self-RAG.
    Respects max_retries for batch errors and ensures no 'working' rows are left stuck.
    Handles:
      - OpenAI SDK-level errors (files.create, batches.create, batches.retrieve,
        files.content, output-file write) via _handle_batch_wave_exception.
      - Both Chat Completions-style models and Responses API reasoning models.
      - Self-RAG in two modes:
          * args.self_rag_use_batch = True  -> Self-RAG via OpenAI Batch
          * args.self_rag_use_batch = False -> Self-RAG via online calls
    """
    # Elicitation must use an OpenAI model for batch
    _ensure_openai_model_for_batch(el_cfg, "elicitation")

    # Determine main elicitation endpoint
    if getattr(el_cfg, "use_responses_api", False):
        batch_endpoint = "/v1/responses"
    else:
        batch_endpoint = "/v1/chat/completions"

    # If Self-RAG uses batch, ensure its model is also OpenAI and set endpoint
    if args.self_rag and getattr(args, "self_rag_use_batch", False):
        _ensure_openai_model_for_batch(self_rag_cfg, "self-rag")

    if getattr(self_rag_cfg, "use_responses_api", False):
        self_rag_endpoint = "/v1/responses"
    else:
        self_rag_endpoint = "/v1/chat/completions"

    qdb = open_queue_db(paths["queue_sqlite"])
    open_llmpedia_db(paths["articles_sqlite"])
    procq_init_cache(qdb)

    _seed_or_resume_queue(args, paths, qdb)
    seen_canon_keys = _load_seen_canon(paths)

    # Only build an online client for Self-RAG if we are NOT using batch for it
    self_rag_llm = None
    if args.self_rag and not getattr(args, "self_rag_use_batch", False):
        self_rag_llm = make_llm_from_config(self_rag_cfg)

    client = OpenAI()
    subjects_total = 0
    wave_idx = 0
    start = time.perf_counter()

    while True:
        # ----- stopping conditions & claiming a batch -----
        if args.max_subjects and subjects_total >= args.max_subjects:
            _dbg(f"[batch] stop: max-subjects reached ({subjects_total})")
            break

        claim_n = args.batch_size
        if args.max_subjects:
            remaining_cap = args.max_subjects - subjects_total
            if remaining_cap <= 0:
                break
            claim_n = min(claim_n, remaining_cap)

        batch = _claim_pending_batch(qdb, args.max_depth, max(1, claim_n))
        if not batch:
            # no more work at allowed hops
            cur = qdb.cursor()
            if args.max_depth == 0:
                cur.execute(
                    "SELECT COUNT(1) FROM queue "
                    "WHERE status IN ('done','working','pending','failed')"
                )
                t = cur.fetchone()[0]
                cur.execute("SELECT COUNT(1) FROM queue WHERE status='done'")
                d = cur.fetchone()[0]
                cur.execute("SELECT COUNT(1) FROM queue WHERE status='working'")
                w = cur.fetchone()[0]
                cur.execute("SELECT COUNT(1) FROM queue WHERE status='pending'")
                p = cur.fetchone()[0]
                cur.execute("SELECT COUNT(1) FROM queue WHERE status='failed'")
                f = cur.fetchone()[0]
            else:
                cur.execute(
                    "SELECT COUNT(1) FROM queue "
                    "WHERE status IN ('done','working','pending','failed') AND hop<=?",
                    (args.max_depth,),
                )
                t = cur.fetchone()[0]
                cur.execute(
                    "SELECT COUNT(1) FROM queue WHERE status='done' AND hop<=?",
                    (args.max_depth,),
                )
                d = cur.fetchone()[0]
                cur.execute(
                    "SELECT COUNT(1) FROM queue WHERE status='working' AND hop<=?",
                    (args.max_depth,),
                )
                w = cur.fetchone()[0]
                cur.execute(
                    "SELECT COUNT(1) FROM queue WHERE status='pending' AND hop<=?",
                    (args.max_depth,),
                )
                p = cur.fetchone()[0]
                cur.execute(
                    "SELECT COUNT(1) FROM queue WHERE status='failed' AND hop<=?",
                    (args.max_depth,),
                )
                f = cur.fetchone()[0]
            if t == 0:
                _dbg("[batch] queue empty, done.")
            else:
                _dbg(
                    f"[batch] queue drained for allowed hops: "
                    f"done={d} working={w} pending={p} failed={f} total={t}"
                )
            break

        wave_idx += 1
        _dbg(f"[batch] wave {wave_idx} claimed {len(batch)} subjects")
        poll_interval = args.batch_poll_interval

        # ----- Self-RAG for this wave (optional, persona-aware) -----
        self_rag_contexts: Dict[Tuple[str, int], Optional[dict]] = {}

        if args.self_rag:
            # Limit how many subjects per wave get Self-RAG (0 = all)
            if args.self_rag_batch_size and args.self_rag_batch_size > 0:
                sr_targets = batch[: args.self_rag_batch_size]
            else:
                sr_targets = batch

            # -----------------------------
            # (A) Self-RAG via OpenAI Batch
            # -----------------------------
            if getattr(args, "self_rag_use_batch", False):
                if sr_targets:
                    batches_dir = paths["batches_dir"]
                    os.makedirs(batches_dir, exist_ok=True)

                    sr_input_path = os.path.join(
                        batches_dir,
                        f"selfrag_input_wave{wave_idx}.jsonl",
                    )

                    # Build the batch input file for Self-RAG
                    with open(sr_input_path, "w", encoding="utf-8") as f_sr:
                        for subject, hop in sr_targets:
                            root_topic = args.seed if args.domain == "topic" else subject
                            messages = _build_self_rag_messages(
                                subject=subject,
                                root_subject=root_topic,
                                persona_block=args.persona_self_rag_block,
                            )

                            if getattr(self_rag_cfg, "use_responses_api", False):
                                body = {
                                    "model": self_rag_cfg.model,
                                    "input": messages,
                                }
                                max_tokens = getattr(self_rag_cfg, "max_tokens", 1024)
                                if max_tokens is not None:
                                    body["max_output_tokens"] = max_tokens
                                extra = getattr(self_rag_cfg, "extra_inputs", None)
                                if isinstance(extra, dict):
                                    body.update(extra)
                            else:
                                body = {
                                    "model": self_rag_cfg.model,
                                    "messages": messages,
                                    "max_tokens": getattr(self_rag_cfg, "max_tokens", 1024),
                                }
                                if getattr(self_rag_cfg, "temperature", None) is not None:
                                    body["temperature"] = self_rag_cfg.temperature
                                if getattr(self_rag_cfg, "top_p", None) is not None:
                                    body["top_p"] = self_rag_cfg.top_p
                                if getattr(self_rag_cfg, "top_k", None) is not None:
                                    body["top_k"] = self_rag_cfg.top_k

                            custom_id = f"selfrag::{subject}::hop={hop}"
                            req_obj = {
                                "custom_id": custom_id,
                                "method": "POST",
                                "url": self_rag_endpoint,
                                "body": body,
                            }
                            f_sr.write(json.dumps(req_obj, ensure_ascii=False) + "\n")

                    try:
                        # Create Self-RAG batch job
                        with open(sr_input_path, "rb") as fh:
                            sr_input_file = client.files.create(
                                file=fh,
                                purpose="batch",
                            )
                        sr_job = client.batches.create(
                            input_file_id=sr_input_file.id,
                            endpoint=self_rag_endpoint,
                            completion_window="24h",
                            metadata={
                                "description": f"LLMPedia Self-RAG wave {wave_idx} seed={args.seed}"
                            },
                        )
                        _dbg(
                            f"[self_rag-batch] wave {wave_idx} created batch id={sr_job.id}, "
                            f"input_file_id={sr_input_file.id}, endpoint={self_rag_endpoint}"
                        )

                        # Poll until completed (soft-fail: if it dies, we just skip Self-RAG)
                        while True:
                            job = client.batches.retrieve(sr_job.id)
                            _dbg(f"[self_rag-batch] wave {wave_idx} status={job.status}")
                            if job.status == "completed":
                                break
                            if job.status in {"failed", "expired", "cancelled"}:
                                _dbg(
                                    f"[self_rag-batch] wave {wave_idx} batch {job.id} ended with status={job.status}; "
                                    "continuing without Self-RAG context for this wave."
                                )
                                job = None
                                break
                            time.sleep(poll_interval)

                        if job and job.output_file_id:
                            sr_out_bytes = client.files.content(job.output_file_id).content
                            sr_out_path = os.path.join(
                                batches_dir,
                                f"selfrag_output_wave{wave_idx}_{job.id}.jsonl",
                            )
                            with open(sr_out_path, "wb") as f:
                                f.write(sr_out_bytes)

                            # Parse batch outputs into self_rag_contexts
                            with open(sr_out_path, "r", encoding="utf-8") as f:
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

                                    resp = row.get("response") or {}
                                    body = resp.get("body") or {}

                                    # Extract raw JSON text from LLM
                                    txt = ""
                                    if getattr(self_rag_cfg, "use_responses_api", False):
                                        output_items = body.get("output") or []
                                        if output_items:
                                            # Prefer the 'message' item; fallback to any item with content
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
                                            if message_item is not None:
                                                content = message_item.get("content") or []
                                                chunks = []
                                                for c in content:
                                                    if isinstance(c, dict) and "text" in c:
                                                        chunks.append(str(c["text"]))
                                                txt = "".join(chunks).strip()
                                    else:
                                        choices = body.get("choices") or []
                                        if choices:
                                            msg = (choices[0] or {}).get("message") or {}
                                            txt = (msg.get("content") or "").strip()

                                    if not txt:
                                        continue

                                    # Decode subject + hop from custom_id
                                    try:
                                        _, rest = cid.split("selfrag::", 1)
                                        subj_part, hop_part = rest.rsplit("::hop=", 1)
                                        subj = subj_part
                                        h = int(hop_part)
                                    except Exception:
                                        subj = cid
                                        h = 0

                                    ctx = None
                                    try:
                                        sr_obj = json.loads(txt)
                                        if isinstance(sr_obj, dict):
                                            ctx = {
                                                "summary": sr_obj.get("summary") or "",
                                                "aliases": sr_obj.get("aliases") or [],
                                                "salient_facts": sr_obj.get("salient_facts") or [],
                                            }
                                    except Exception:
                                        ctx = None

                                    log_rec = {
                                        "ts": datetime.datetime.utcnow().isoformat() + "Z",
                                        "subject": subj,
                                        "hop": h,
                                        "model": getattr(self_rag_cfg, "model", None),
                                        "parsed": ctx,
                                    }
                                    if ctx is None:
                                        log_rec["error"] = "parse_failed"
                                    _append_jsonl(paths["self_rag_log_jsonl"], log_rec)

                                    if ctx is not None:
                                        self_rag_contexts[(subj, h)] = ctx

                    except Exception as e:
                        # Soft-fail: log, but do NOT mark queue as error
                        with open(paths["errors_log"], "a", encoding="utf-8") as ef:
                            ef.write(
                                f"[{datetime.datetime.now().isoformat()}] "
                                f"[self_rag-batch-wave={wave_idx}] error={repr(e)}\n"
                                f"{traceback.format_exc()}\n"
                            )
                        _dbg(
                            f"[self_rag-batch] wave {wave_idx} failed; "
                            "proceeding without Self-RAG contexts."
                        )

            # ---------------------------------
            # (B) Self-RAG via online threadpool
            # ---------------------------------
            elif self_rag_llm is not None:

                def _self_rag_worker(subject: str, hop: int):
                    root_topic = args.seed if args.domain == "topic" else subject
                    ctx = _run_self_rag_for_subject(
                        subject=subject,
                        hop=hop,
                        root_topic=root_topic,
                        args=args,
                        self_rag_cfg=self_rag_cfg,
                        self_rag_llm=self_rag_llm,
                        paths=paths,
                        persona_block=args.persona_self_rag_block,
                        wave_idx=wave_idx,
                    )
                    return (subject, hop, ctx)

                max_workers = args.self_rag_concurrency if args.self_rag_concurrency > 0 else 1
                _dbg(
                    f"[self_rag-online] wave={wave_idx} subjects={len(sr_targets)} "
                    f"concurrency={max_workers}"
                )
                with ThreadPoolExecutor(max_workers=min(max_workers, len(sr_targets))) as pool:
                    futs = [pool.submit(_self_rag_worker, s, h) for (s, h) in sr_targets]
                    for fut in as_completed(futs):
                        s, h, ctx = fut.result()
                        self_rag_contexts[(s, h)] = ctx

        # ----- build batch_input.jsonl for this wave (elicitation) -----
        batches_dir = paths["batches_dir"]
        os.makedirs(batches_dir, exist_ok=True)

        wave_input_path = os.path.join(
            batches_dir,
            f"batch_input_wave{wave_idx}.jsonl",
        )

        with open(wave_input_path, "w", encoding="utf-8") as f:
            for subject, hop in batch:
                root_topic = args.seed if args.domain == "topic" else subject
                ctx = self_rag_contexts.get((subject, hop))
                messages = _build_llmpedia_messages_for_subject(
                    subject=subject,
                    hop=hop,
                    args=args,
                    root_topic=root_topic,
                    persona_block=args.persona_elicit_block,
                    self_rag_context=ctx,
                )

                if getattr(el_cfg, "use_responses_api", False):
                    # Responses API body
                    body = {
                        "model": el_cfg.model,
                        "input": messages,
                    }
                    max_tokens = getattr(el_cfg, "max_tokens", 2048)
                    if max_tokens is not None:
                        body["max_output_tokens"] = max_tokens
                    extra = getattr(el_cfg, "extra_inputs", None)
                    if isinstance(extra, dict):
                        body.update(extra)
                else:
                    # Chat Completions body
                    body = {
                        "model": el_cfg.model,
                        "messages": messages,
                        "max_tokens": getattr(el_cfg, "max_tokens", 2048),
                    }
                    if getattr(el_cfg, "temperature", None) is not None:
                        body["temperature"] = el_cfg.temperature
                    if getattr(el_cfg, "top_p", None) is not None:
                        body["top_p"] = el_cfg.top_p

                custom_id = f"elicitation::{subject}::hop={hop}"
                req_obj = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": batch_endpoint,
                    "body": body,
                }
                f.write(json.dumps(req_obj, ensure_ascii=False) + "\n")

        paths["batch_input_jsonl"] = wave_input_path

        # ----- upload + create batch job (elicitation) -----
        try:
            with open(wave_input_path, "rb") as fh:
                batch_input_file = client.files.create(
                    file=fh,
                    purpose="batch",
                )
            batch_job = client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint=batch_endpoint,
                completion_window="24h",
                metadata={"description": f"LLMPedia batch wave {wave_idx} seed={args.seed}"},
            )
        except Exception as e:
            _handle_batch_wave_exception(
                e=e,
                wave_idx=wave_idx,
                batch=batch,
                args=args,
                paths=paths,
                seen_canon_keys=seen_canon_keys,
                context="files.create/batches.create",
            )
            # Skip this wave, go to next
            continue

        _dbg(
            f"[batch] wave {wave_idx} created batch id={batch_job.id}, "
            f"input_file_id={batch_input_file.id}, endpoint={batch_endpoint}"
        )

        # ----- poll until completed (elicitation) -----
        try:
            while True:
                job = client.batches.retrieve(batch_job.id)
                _dbg(f"[batch] wave {wave_idx} status={job.status}")
                if job.status == "completed":
                    break
                if job.status in {"failed", "expired", "cancelled"}:
                    _dbg(
                        f"[batch] wave {wave_idx} batch {job.id} ended with status={job.status}; "
                        f"marking all {len(batch)} subjects with an error retry."
                    )
                    for subject, hop in batch:
                        mark_pending_on_error(paths["queue_sqlite"], subject, hop, args.max_retries)
                    _persist_seen_canon(paths, seen_canon_keys)
                    break
                time.sleep(poll_interval)
        except Exception as e:
            _handle_batch_wave_exception(
                e=e,
                wave_idx=wave_idx,
                batch=batch,
                args=args,
                paths=paths,
                seen_canon_keys=seen_canon_keys,
                context="batches.retrieve/polling",
            )
            continue

        if job.status in {"failed", "expired", "cancelled"}:
            continue

        if not job.output_file_id:
            _dbg(
                f"[batch] wave {wave_idx} batch {job.id} has no output_file_id; "
                f"marking all subjects with an error retry."
            )
            for subject, hop in batch:
                mark_pending_on_error(paths["queue_sqlite"], subject, hop, args.max_retries)
            _persist_seen_canon(paths, seen_canon_keys)
            continue

        # ----- download output & process each subject (elicitation) -----
        try:
            out_bytes = client.files.content(job.output_file_id).content
        except Exception as e:
            _handle_batch_wave_exception(
                e=e,
                wave_idx=wave_idx,
                batch=batch,
                args=args,
                paths=paths,
                seen_canon_keys=seen_canon_keys,
                context="files.content(download)",
            )
            continue

        out_path = os.path.join(
            batches_dir,
            f"batch_output_wave{wave_idx}_{job.id}.jsonl",
        )
        try:
            with open(out_path, "wb") as f:
                f.write(out_bytes)
        except Exception as e:
            _handle_batch_wave_exception(
                e=e,
                wave_idx=wave_idx,
                batch=batch,
                args=args,
                paths=paths,
                seen_canon_keys=seen_canon_keys,
                context="writing_output_file",
            )
            continue

        processed: Set[Tuple[str, int]] = set()

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
                if not isinstance(custom_id, str) or not custom_id.startswith("elicitation::"):
                    continue

                resp = row.get("response") or {}
                body = resp.get("body") or {}

                # -------- extract wikitext depending on API style --------
                wikitext = ""

                if getattr(el_cfg, "use_responses_api", False):
                    # Responses API (reasoning models)
                    output_items = body.get("output") or []
                    if not output_items:
                        continue

                    # Prefer the 'message' item; fallback to any item that has content
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
                    text_chunks = []
                    for c in content:
                        if isinstance(c, dict) and "text" in c:
                            text_chunks.append(str(c["text"]))
                    wikitext = "".join(text_chunks).strip()

                else:
                    # Chat Completions-style models
                    choices = body.get("choices") or []
                    if not choices:
                        continue
                    msg = (choices[0] or {}).get("message") or {}
                    wikitext = (msg.get("content") or "").strip()

                if not wikitext:
                    # treat as missing; handled later via 'missing' retry logic
                    continue

                # -------- decode subject + hop from custom_id --------
                try:
                    _, rest = custom_id.split("elicitation::", 1)
                    subj_part, hop_part = rest.rsplit("::hop=", 1)
                    subject = subj_part
                    hop = int(hop_part)
                except Exception:
                    subject = custom_id
                    hop = 0

                if args.debug:
                    _dbg(f"[batch] wave {wave_idx} parsed article for [{subject}] hop={hop}")

                # -------- post-process article + NER + queue --------
                try:
                    _post_article_processing(
                        args,
                        paths,
                        el_cfg,
                        ner_cfg,
                        subject,
                        hop,
                        wikitext,
                        seen_canon_keys,
                    )

                    conn = procq_get_thread_conn(paths["queue_sqlite"])
                    _exec_retry(
                        conn,
                        "UPDATE queue SET status='done' WHERE subject=? AND hop=?",
                        (subject, hop),
                    )
                    subjects_total += 1
                    processed.add((subject, hop))
                except Exception:
                    with open(paths["errors_log"], "a", encoding="utf-8") as ef:
                        ef.write(
                            f"[{datetime.datetime.now().isoformat()}] "
                            f"[batch-wave={wave_idx}] subject={subject} hop={hop}\n"
                            f"{traceback.format_exc()}\n"
                        )
                    mark_pending_on_error(paths["queue_sqlite"], subject, hop, args.max_retries)

        # any subjects in this wave that never got good output
        missing = [(s, h) for (s, h) in batch if (s, h) not in processed]
        if missing:
            _dbg(
                f"[batch] wave {wave_idx} had {len(missing)} subjects with no successful output; "
                f"applying error retry logic."
            )
            for subject, hop in missing:
                mark_pending_on_error(paths["queue_sqlite"], subject, hop, args.max_retries)

        _persist_seen_canon(paths, seen_canon_keys)

    # ----- handle leftover working rows -----
    conn = sqlite3.connect(paths["queue_sqlite"])
    cur = conn.cursor()
    cur.execute("SELECT subject, hop FROM queue WHERE status='working'")
    stuck = cur.fetchall()
    conn.close()
    if stuck:
        _dbg(
            f"[batch] WARNING: {len(stuck)} rows remained 'working' at end of run; "
            f"applying error retry logic (max_retries={args.max_retries})."
        )
        for subject, hop in stuck:
            mark_pending_on_error(paths["queue_sqlite"], subject, hop, args.max_retries)

    _snapshot_queue_and_articles(paths)

    dur = time.perf_counter() - start
    _dbg(f"[done-batch] finished in {dur:.1f}s → {os.path.dirname(paths['queue_sqlite'])}")


def main():
    ap = argparse.ArgumentParser(
        description="LLMPedia crawler: online & OpenAI batch modes with optional Self-RAG and personas."
    )

    # General Mode Selection
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
        default="topic",
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
    ap.add_argument("--article-max-sections", type=int, default=6)
    ap.add_argument("--article-avg-words", type=int, default=450)

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
        default="batch",
        help="Self-RAG execution mode (only meaningful when --mode=batch).",
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

    # Models & Sampling
    ap.add_argument(
        "--elicit-model-key",
        default="gpt-4.1-mini",
        help="settings.MODELS key for article generation (elicitation).",
    )
    ap.add_argument(
        "--ner-model-key",
        default="gpt-4.1-mini",
        help="settings.MODELS key for NER.",
    )
    ap.add_argument(
        "--self-rag-model-key",
        default="gpt-4.1-mini",
        help="settings.MODELS key for Self-RAG (defaults to elicit-model-key).",
    )

    ap.add_argument("--elicit-temperature", type=float, default=0.4)
    ap.add_argument("--ner-temperature", type=float, default=0.3)
    ap.add_argument("--self-rag-temperature", type=float, default=0.1)

    ap.add_argument("--elicit-top-p", type=float, default=None)
    ap.add_argument("--ner-top-p", type=float, default=None)
    ap.add_argument("--self-rag-top-p", type=float, default=None)

    ap.add_argument("--elicit-top-k", type=int, default=None)
    ap.add_argument("--ner-top-k", type=int, default=None)
    ap.add_argument("--self-rag-top-k", type=int, default=None)

    ap.add_argument("--elicit-max-tokens", type=int, default=4096)
    ap.add_argument("--ner-max-tokens", type=int, default=2048)
    ap.add_argument("--self-rag-max-tokens", type=int, default=1024)

    ap.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="Request timeout (seconds) for online calls.",
    )

    # NER / Elicitation thresholds
    ap.add_argument(
        "--ner-conf-threshold",
        type=float,
        default=0.0,
        help="If >0 and NER returns confidence scores, only enqueue entities with confidence >= threshold.",
    )
    ap.add_argument(
        "--elicit-conf-threshold",
        type=float,
        default=0.0,
        help=(
            "If >0, interpret confidences encoded in wikilinks ([[Entity (0.93)]]) "
            "and filter below threshold."
        ),
    )

    # Footer controls
    ap.add_argument(
        "--footer-mode",
        type=_str2bool,
        default=False,
        help="If true, append a categories-focused footer to the elicitation prompt.",
    )
    ap.add_argument(
        "--footer-location",
        choices=["system", "user"],
        default="user",
        help="Which role's message to append the footer to.",
    )

    # Self-RAG controls
    ap.add_argument(
        "--self-rag",
        type=_str2bool,
        default=False,
        help="Enable Self-RAG grounding stage (online and batch).",
    )
    ap.add_argument(
        "--self-rag-target",
        choices=["system", "user"],
        default="user",
        help="Where to append the Self-RAG context.",
    )
    ap.add_argument(
        "--self-rag-batch-size",
        type=int,
        default=0,
        help=(
            "In mode=batch, max number of subjects per wave that get Self-RAG "
            "(0 = all subjects in wave)."
        ),
    )

    # Reasoning overrides for Responses API – GLOBAL
    ap.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"], default=None)
    ap.add_argument("--text-verbosity", choices=["low", "medium", "high"], default=None)

    # Stage-specific reasoning overrides (for Responses API)
    ap.add_argument(
        "--elicit-reasoning-effort",
        choices=["minimal", "low", "medium", "high"],
        default=None,
    )
    ap.add_argument(
        "--elicit-text-verbosity",
        choices=["low", "medium", "high"],
        default=None,
    )
    ap.add_argument(
        "--ner-reasoning-effort",
        choices=["minimal", "low", "medium", "high"],
        default=None,
    )
    ap.add_argument(
        "--ner-text-verbosity",
        choices=["low", "medium", "high"],
        default=None,
    )

    # Self-RAG-specific reasoning overrides
    ap.add_argument(
        "--self-rag-reasoning-effort",
        choices=["minimal", "low", "medium", "high"],
        default=None,
    )
    ap.add_argument(
        "--self-rag-text-verbosity",
        choices=["low", "medium", "high"],
        default=None,
    )

    # Retry controls (finite)
    ap.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per subject before marking as failed (>=1). Applies to online and batch.",
    )

    # Persona controls
    ap.add_argument(
        "--personas-path",
        default=None,
        help="Optional path to personas.json (default: ./personas.json).",
    )
    ap.add_argument(
        "--persona",
        default="scientific_neutral",
        help="Global persona key (applies to all stages unless overridden).",
    )
    ap.add_argument(
        "--persona-elicit",
        dest="persona_elicit",
        default=None,
        help="Persona key for elicitation stage (overrides --persona if set).",
    )
    ap.add_argument(
        "--persona-ner",
        dest="persona_ner",
        default=None,
        help="Persona key for NER stage (overrides --persona if set).",
    )
    ap.add_argument(
        "--persona-self_rag",
        dest="persona_self_rag",
        default=None,
        help="Persona key for Self-RAG stage (overrides --persona if set).",
    )

    ap.add_argument("--debug", action="store_true")
    ap.add_argument(
        "--progress-metrics",
        dest="progress_metrics",
        action="store_true",
        default=True,
    )
    ap.add_argument(
        "--no-progress-metrics",
        dest="progress_metrics",
        action="store_false",
    )

    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reset-working", action="store_true")

    args = ap.parse_args()

    # Default: no batch Self-RAG; run_batch should inspect this flag.
    args.self_rag_use_batch = False

    # ---- Mode-dependent Self-RAG semantics ----

    if args.mode == "online":
        # As requested: always print, regardless of debug flag.
        print(
            "[self-rag] --mode=online: --self-rag-mode and --self-rag-concurrency "
            "are ignored; Self-RAG (if enabled) runs inline per subject."
        )

    # Load personas
    personas = _load_personas(args.personas_path)
    persona_elicit_name = _resolve_stage_persona_name(args, "elicit")
    persona_ner_name = _resolve_stage_persona_name(args, "ner")
    persona_self_rag_name = _resolve_stage_persona_name(args, "self_rag")

    # Get persona blocks
    args.persona_elicit_block = _get_persona_block(personas, persona_elicit_name, "elicit")
    args.persona_ner_block = _get_persona_block(personas, persona_ner_name, "ner")
    args.persona_self_rag_block = _get_persona_block(personas, persona_self_rag_name, "self_rag")

    # Prepare output directory
    out_dir = _ensure_output_dir(args.output_dir)
    paths = _build_paths(out_dir)
    _dbg(
        f"[llmpedia-combined] mode={args.mode} output_dir={out_dir} "
        f"max_depth={args.max_depth} max_subjects={args.max_subjects} "
        f"max_retries={args.max_retries} "
        f"persona_elicit={persona_elicit_name} persona_ner={persona_ner_name} "
        f"persona_self_rag={persona_self_rag_name}"
    )

    open_queue_db(paths["queue_sqlite"])
    open_llmpedia_db(paths["articles_sqlite"])

    # Build model configs
    el_cfg = settings.MODELS[args.elicit_model_key].model_copy(deep=True)
    ner_cfg = settings.MODELS[args.ner_model_key].model_copy(deep=True)
    self_rag_key = args.self_rag_model_key or args.elicit_model_key
    self_rag_cfg = settings.MODELS[self_rag_key].model_copy(deep=True)

    # ---- Self-RAG mode handling for batch mode ("all together batch") ----
    if args.mode == "batch":
        if not args.self_rag:
            print("[self-rag] --mode=batch but --self-rag=false: Self-RAG is disabled.")
        else:
            if args.self_rag_mode == "batch":
                # Require OpenAI provider for Self-RAG *batch*
                sr_provider = getattr(self_rag_cfg, "provider", "openai")
                if str(sr_provider).lower() != "openai":
                    ap.error(
                        "[self-rag] --self-rag-mode=batch requires an OpenAI self-rag model "
                        f"provider; got provider={sr_provider!r}. Cannot use batch Self-RAG "
                        "with a non-OpenAI model."
                    )
                # If we got here, Self-RAG can use OpenAI Batch.
                args.self_rag_use_batch = True
                print(
                    "[self-rag] --mode=batch and --self-rag-mode=batch: "
                    "Self-RAG and elicitation are both configured to use OpenAI Batch."
                )
            else:
                # self_rag_mode == "online": use online Self-RAG inside batch pipeline
                if args.self_rag_concurrency is None or args.self_rag_concurrency <= 0:
                    args.self_rag_concurrency = 2
                args.self_rag_use_batch = False
                print(
                    f"[self-rag] --mode=batch and --self-rag-mode=online: "
                    f"Self-RAG will run online with concurrency={args.self_rag_concurrency}."
                )

    # Apply per-stage sampling configs for elicitation/NER
    _apply_stage("elicit", el_cfg, args)
    _apply_stage("ner", ner_cfg, args)

    # Self-RAG config (sampling / reasoning)
    if getattr(self_rag_cfg, "use_responses_api", False):
        if self_rag_cfg.extra_inputs is None:
            self_rag_cfg.extra_inputs = {}
        self_rag_cfg.extra_inputs.setdefault("reasoning", {})
        self_rag_cfg.extra_inputs.setdefault("text", {})

        if args.self_rag_reasoning_effort is not None:
            self_rag_cfg.extra_inputs["reasoning"]["effort"] = args.self_rag_reasoning_effort
        elif args.reasoning_effort is not None:
            self_rag_cfg.extra_inputs["reasoning"]["effort"] = args.reasoning_effort

        if args.self_rag_text_verbosity is not None:
            self_rag_cfg.extra_inputs["text"]["verbosity"] = args.self_rag_text_verbosity
        elif args.text_verbosity is not None:
            self_rag_cfg.extra_inputs["text"]["verbosity"] = args.text_verbosity
    else:
        self_rag_cfg.temperature = args.self_rag_temperature
        if args.self_rag_top_p is not None:
            self_rag_cfg.top_p = args.self_rag_top_p
        if args.self_rag_top_k is not None:
            self_rag_cfg.top_k = args.self_rag_top_k

    self_rag_cfg.max_tokens = args.self_rag_max_tokens

    start = time.perf_counter()

    if args.mode == "online":
        # Self-RAG (if enabled) runs inline per subject; self_rag_mode & self_rag_concurrency ignored.
        run_online(args, paths, el_cfg, ner_cfg, self_rag_cfg)
    elif args.mode == "batch":
        # Inside run_batch, you should use:
        #   - args.self_rag (bool)
        #   - args.self_rag_use_batch (True => Self-RAG via OpenAI Batch; False => online Self-RAG)
        #   - args.self_rag_concurrency (for online Self-RAG)
        run_batch(args, paths, el_cfg, ner_cfg, self_rag_cfg)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    dur = time.perf_counter() - start

    run_meta = {
        "timestamp_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "seed": args.seed,
        "domain": args.domain,
        "mode": args.mode,
        "elicitation_strategy": args.elicitation_strategy,
        "ner_strategy": args.ner_strategy,
        "self_rag_enabled": bool(args.self_rag),
        "self_rag_mode": args.self_rag_mode,
        "self_rag_use_batch": bool(args.self_rag_use_batch),
        "max_depth": args.max_depth,
        "max_subjects": args.max_subjects,
        "batch_size": args.batch_size,
        "personas": {
            "elicit": persona_elicit_name,
            "ner": persona_ner_name,
            "self_rag": persona_self_rag_name,
        },
        "models": {
            "elicitation": {
                "provider": getattr(el_cfg, "provider", "openai"),
                "model": el_cfg.model,
                "use_responses_api": getattr(el_cfg, "use_responses_api", False),
                "temperature": getattr(el_cfg, "temperature", None),
                "top_p": getattr(el_cfg, "top_p", None),
                "top_k": getattr(el_cfg, "top_k", None),
                "max_tokens": getattr(el_cfg, "max_tokens", None),
            },
            "ner": {
                "provider": getattr(ner_cfg, "provider", "openai"),
                "model": ner_cfg.model,
                "use_responses_api": getattr(ner_cfg, "use_responses_api", False),
                "temperature": getattr(ner_cfg, "temperature", None),
                "top_p": getattr(ner_cfg, "top_p", None),
                "top_k": getattr(ner_cfg, "top_k", None),
                "max_tokens": getattr(ner_cfg, "max_tokens", None),
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
            }
            if args.self_rag
            else None,
        },
        "args_raw": vars(args),
        "duration_s": dur,
    }
    with open(paths["run_meta_json"], "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    print(f"[done] finished in {dur:.1f}s → {out_dir}")
    for k in (
        "queue_json",
        "queue_jsonl",
        "articles_json",
        "articles_jsonl",
        "ner_decisions_jsonl",
        "ner_lowconf_jsonl",
        "elicit_lowconf_jsonl",
        "self_rag_log_jsonl",
        "run_meta_json",
        "errors_log",
        "seen_state_json",
        "batch_input_jsonl",
    ):
        if k in paths:
            print(f"[out] {k:18}: {paths[k]}")


if __name__ == "__main__":
    # try:
    #     main()
    # except KeyboardInterrupt:
    #     print("\n[interrupt] bye")

    try:
        main()
    except KeyboardInterrupt:
        print("\n[interrupt] bye (KeyboardInterrupt caught at top level)")
        traceback.print_exc()  # show where it came from
        # If you don't want it to propagate, remove the next line:
        # raise

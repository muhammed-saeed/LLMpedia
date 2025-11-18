from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sqlite3
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Set, Optional, Dict

from dotenv import load_dotenv

load_dotenv()

# ---------------- tiny utils & locks ----------------

_jsonl_lock = threading.Lock()
_seen_canon_lock = threading.Lock()


def _append_jsonl(path: str, obj: dict):
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    os.makedirs(os.path.dirname(path), exist_ok=True)
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

# ---------------- paths + DB helpers ----------------


def _ensure_output_dir(base_dir: Optional[str]) -> str:
    out = base_dir or os.path.join("runs", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    return out


def _build_paths(out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
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
        "selfrag_log_jsonl": os.path.join(out_dir, "selfrag_log.jsonl"),
        "batch_input_jsonl": os.path.join(out_dir, "batch_input.jsonl"),
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


def mark_pending_on_error(queue_db_path: str, subject: str, hop: int):
    conn = procq_get_thread_conn(queue_db_path)
    _exec_retry(
        conn,
        "UPDATE queue SET status='pending', retries=retries+1 WHERE subject=? AND hop=? AND status='working'",
        (subject, hop),
    )


def _claim_pending_batch(conn: sqlite3.Connection, max_depth: int, claim_n: int):
    def _do():
        with conn:
            cur = conn.cursor()
            if max_depth == 0:
                cur.execute(
                    """
                    UPDATE queue SET status='working'
                    WHERE rowid IN (
                      SELECT rowid FROM queue WHERE status='pending'
                      ORDER BY hop, created_at LIMIT ?
                    )
                    RETURNING subject, hop
                """,
                    (claim_n,),
                )
            else:
                cur.execute(
                    """
                    UPDATE queue SET status='working'
                    WHERE rowid IN (
                      SELECT rowid FROM queue WHERE status='pending' AND hop<=?
                      ORDER BY hop, created_at LIMIT ?
                    )
                    RETURNING subject, hop
                """,
                    (max_depth, claim_n),
                )
            rows = cur.fetchall()
            cur.close()
            return rows

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


def _build_selfrag_messages(subject: str, root_subject: str) -> List[dict]:
    sys = (
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


def _build_selfrag_block(subject: str, ctx: dict) -> str:
    summary = (ctx.get("summary") or "").strip()
    aliases = ", ".join(ctx.get("aliases") or [])
    facts = ctx.get("salient_facts") or []
    lines = []
    for f in facts[:16]:
        p = (f.get("predicate") or "").strip()
        o = (f.get("object") or "").strip()
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
    if getattr(cfg, "use_responses_api", False):
        cfg.temperature = None
        cfg.top_p = None
        cfg.top_k = None
        if cfg.extra_inputs is None:
            cfg.extra_inputs = {}
        cfg.extra_inputs.setdefault("reasoning", {})
        cfg.extra_inputs.setdefault("text", {})
        if args.reasoning_effort is not None:
            cfg.extra_inputs["reasoning"]["effort"] = args.reasoning_effort
        if args.text_verbosity is not None:
            cfg.extra_inputs["text"]["verbosity"] = args.text_verbosity
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
import openai  # noqa: F401  (you may still use openai.RateLimitError if needed)


def _ensure_openai_model_for_batch(model_cfg, label: str):
    provider = getattr(model_cfg, "provider", "openai")
    if provider.lower() != "openai":
        raise ValueError(
            f"[mode=batch] {label} model provider must be 'openai', "
            f"got provider={provider!r}. Please choose a GPT model for batch mode."
        )


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
        with open(paths["seen_state_json"], "w", encoding="utf-8") as f:
            with _seen_canon_lock:
                json.dump(sorted(list(seen_canon_keys)), f, ensure_ascii=False, indent=2)
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

    # NER online
    ner_llm = make_llm_from_config(ner_cfg)
    next_subjects: List[str] = []
    unique_next: List[str] = []

    if candidates_for_ner:
        ner_messages = build_ner_messages_for_phrases(
            domain=args.domain,
            strategy=args.ner_strategy,
            subject_name=subject,
            seed=args.seed,
            phrases=candidates_for_ner,
        )

        if args.debug:
            _dbg(
                f"[NER] input candidates for [{subject}] (hop={hop}): "
                f"{candidates_for_ner[:10]}{'…' if len(candidates_for_ner) > 10 else ''}"
            )
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

    # article JSONL
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

    return seen_canon_keys


def run_online(args, paths, el_cfg, ner_cfg, selfrag_cfg):
    qdb = open_queue_db(paths["queue_sqlite"])
    open_llmpedia_db(paths["articles_sqlite"])
    procq_init_cache(qdb)

    _seed_or_resume_queue(args, paths, qdb)
    seen_canon_keys = _load_seen_canon(paths)

    el_llm = make_llm_from_config(el_cfg)
    selfrag_llm = make_llm_from_config(selfrag_cfg) if args.self_rag else None

    start = time.perf_counter()
    last_progress_ts = 0.0
    subjects_total = 0

    def _generate_article(subject: str, hop: int):
        try:
            root_topic = args.seed if args.domain == "topic" else subject

            # Self-RAG (online)
            selfrag_context = None
            if args.self_rag and selfrag_llm is not None:
                sr_msgs = _build_selfrag_messages(subject, root_topic)
                try:
                    sr_resp = selfrag_llm(sr_msgs, json_schema=SELF_RAG_SCHEMA, timeout=args.timeout)
                except TypeError:
                    sr_resp = selfrag_llm(sr_msgs, json_schema=SELF_RAG_SCHEMA)
                except Exception:
                    sr_resp = selfrag_llm(sr_msgs)

                sr_obj = None
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
                    selfrag_context = {
                        "summary": sr_obj.get("summary") or "",
                        "aliases": sr_obj.get("aliases") or [],
                        "salient_facts": sr_obj.get("salient_facts") or [],
                    }

                _append_jsonl(
                    paths["selfrag_log_jsonl"],
                    {
                        "ts": datetime.datetime.utcnow().isoformat() + "Z",
                        "subject": subject,
                        "hop": hop,
                        "model": getattr(selfrag_cfg, "model", None),
                        "parsed": selfrag_context,
                    },
                )

            # build elicitation messages
            messages = build_elicitation_messages_for_subject(
                domain=args.domain,
                strategy=args.elicitation_strategy,
                subject_name=subject,
                seed=args.seed,
                root_topic=root_topic,
                min_sections=args.article_min_sections,
                max_sections=args.article_max_sections,
                avg_words_per_article=args.article_avg_words,
            )

            if selfrag_context and (selfrag_context.get("summary") or selfrag_context.get("salient_facts")):
                sr_block = _build_selfrag_block(subject, selfrag_context)
                messages = _append_block_to_msgs(messages, sr_block, target=args.selfrag_target)

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

            if args.debug:
                _dbg(f"\n--- LLMPEDIA for [{subject}] (hop={hop}) ---")
                for i, m in enumerate(messages, 1):
                    preview = m["content"][:200] if isinstance(m.get("content"), str) else ""
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

            # post-processing (store + NER + enqueue)
            nonlocal seen_canon_keys
            seen_canon_keys = _post_article_processing(
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
            mark_pending_on_error(paths["queue_sqlite"], subject, hop)
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
                t = d + w + p
                _dbg(f"[progress] done={d} working={w} pending={p} total={t}")
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
                    "SELECT COUNT(1) FROM queue WHERE status IN ('done','working','pending')"
                )
                t = cur.fetchone()[0]
                cur.execute("SELECT COUNT(1) FROM queue WHERE status='done'")
                d = cur.fetchone()[0]
                cur.execute("SELECT COUNT(1) FROM queue WHERE status='working'")
                w = cur.fetchone()[0]
                cur.execute("SELECT COUNT(1) FROM queue WHERE status='pending'")
                p = cur.fetchone()[0]
            else:
                cur.execute(
                    "SELECT COUNT(1) FROM queue WHERE status IN ('done','working','pending') AND hop<=?",
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
            if t == 0:
                _dbg("[idle] nothing to do.]")
            else:
                _dbg(f"[idle] queue drained: done={d} working={w} pending={p} total={t}")
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

    # final snapshots
    conn = sqlite3.connect(paths["queue_sqlite"])
    cur = conn.cursor()
    cur.execute("SELECT subject, hop, status, retries, created_at FROM queue ORDER BY hop, subject")
    rows = cur.fetchall()
    with open(paths["queue_json"], "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "subject": s,
                    "hop": h,
                    "status": st,
                    "retries": r,
                    "created_at": ts,
                }
                for (s, h, st, r, ts) in rows
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )
    conn.close()

    conn = sqlite3.connect(paths["articles_sqlite"])
    cur = conn.cursor()
    cur.execute(
        "SELECT subject, wikitext, hop, model_name, overall_confidence, created_at "
        "FROM llmpedia_articles ORDER BY subject"
    )
    arows = cur.fetchall()
    with open(paths["articles_json"], "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "subject": s,
                    "wikitext": wt,
                    "hop": h,
                    "model": m,
                    "overall_confidence": oc,
                    "created_at": ts,
                }
                for (s, wt, h, m, oc, ts) in arows
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )
    conn.close()

    dur = time.perf_counter() - start
    _dbg(f"[done-online] finished in {dur:.1f}s → {os.path.dirname(paths['queue_sqlite'])}")


# ---------------- BATCH MODE (everything end-to-end) ----------------


def run_batch(args, paths, el_cfg, ner_cfg, selfrag_cfg):
    """
    Full pipeline in one mode:
      - seed/resume queue
      - while queue not empty & max-subjects not reached:
          - claim up to batch_size subjects
          - (optional) Self-RAG for those subjects online, in parallel
          - build batch_input.jsonl with msgs that ALREADY include Self-RAG + footer
          - create OpenAI /v1/batches job, poll until completed
          - download output, parse wikitext, store articles
          - run NER online & expand queue
          - mark subjects done
    """
    _ensure_openai_model_for_batch(el_cfg, "elicitation")

    qdb = open_queue_db(paths["queue_sqlite"])
    open_llmpedia_db(paths["articles_sqlite"])
    procq_init_cache(qdb)

    _seed_or_resume_queue(args, paths, qdb)
    seen_canon_keys = _load_seen_canon(paths)

    selfrag_llm = make_llm_from_config(selfrag_cfg) if args.self_rag else None

    client = OpenAI()
    subjects_total = 0
    wave_idx = 0
    start = time.perf_counter()

    while True:
        if args.max_subjects and subjects_total >= args.max_subjects:
            _dbg(f"[batch] stop: max-subjects reached ({subjects_total})")
            break

        # claim a wave of subjects
        claim_n = args.batch_size
        if args.max_subjects:
            remaining_cap = args.max_subjects - subjects_total
            if remaining_cap <= 0:
                break
            claim_n = min(claim_n, remaining_cap)

        batch = _claim_pending_batch(qdb, args.max_depth, max(1, claim_n))
        if not batch:
            cur = qdb.cursor()
            if args.max_depth == 0:
                cur.execute(
                    "SELECT COUNT(1) FROM queue WHERE status IN ('done','working','pending')"
                )
                t = cur.fetchone()[0]
                cur.execute("SELECT COUNT(1) FROM queue WHERE status='done'")
                d = cur.fetchone()[0]
                cur.execute("SELECT COUNT(1) FROM queue WHERE status='working'")
                w = cur.fetchone()[0]
                cur.execute("SELECT COUNT(1) FROM queue WHERE status='pending'")
                p = cur.fetchone()[0]
            else:
                cur.execute(
                    "SELECT COUNT(1) FROM queue WHERE status IN ('done','working','pending') AND hop<=?",
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
            if t == 0:
                _dbg("[batch] queue empty, done.")
            else:
                _dbg(
                    f"[batch] queue drained for allowed hops: "
                    f"done={d} working={w} pending={p} total={t}"
                )
            break

        wave_idx += 1
        _dbg(f"[batch] wave {wave_idx} claimed {len(batch)} subjects")

        # ---- Self-RAG for this wave (online, parallel, batch-only knobs) ----
        selfrag_contexts: Dict[Tuple[str, int], Optional[dict]] = {}

        if args.self_rag and selfrag_llm is not None:
            def _selfrag_worker(subject: str, hop: int):
                root_topic = args.seed if args.domain == "topic" else subject
                sr_msgs = _build_selfrag_messages(subject, root_topic)
                try:
                    sr_resp = selfrag_llm(sr_msgs, json_schema=SELF_RAG_SCHEMA, timeout=args.timeout)
                except TypeError:
                    sr_resp = selfrag_llm(sr_msgs, json_schema=SELF_RAG_SCHEMA)
                except Exception:
                    sr_resp = selfrag_llm(sr_msgs)

                sr_obj = None
                if isinstance(sr_resp, dict) and ("summary" in sr_resp and "salient_facts" in sr_resp):
                    sr_obj = sr_resp
                else:
                    try:
                        txt = _unwrap_text(sr_resp)
                        if txt:
                            sr_obj = json.loads(txt)
                    except Exception:
                        sr_obj = None
                ctx = None
                if isinstance(sr_obj, dict):
                    ctx = {
                        "summary": sr_obj.get("summary") or "",
                        "aliases": sr_obj.get("aliases") or [],
                        "salient_facts": sr_obj.get("salient_facts") or [],
                    }
                _append_jsonl(
                    paths["selfrag_log_jsonl"],
                    {
                        "ts": datetime.datetime.utcnow().isoformat() + "Z",
                        "subject": subject,
                        "hop": hop,
                        "model": getattr(selfrag_cfg, "model", None),
                        "parsed": ctx,
                        "wave": wave_idx,
                    },
                )
                return (subject, hop, ctx)

            # decide which subjects in this wave get Self-RAG
            if args.selfrag_batch_size and args.selfrag_batch_size > 0:
                targets = batch[: args.selfrag_batch_size]
            else:
                targets = batch

            # how many Self-RAG calls run in parallel (batch-only)
            max_workers = args.selfrag_concurrency if args.selfrag_concurrency > 0 else 1
            _dbg(
                f"[selfrag-batch] wave={wave_idx} subjects={len(targets)} "
                f"concurrency={max_workers}"
            )
            with ThreadPoolExecutor(max_workers=min(max_workers, len(targets))) as pool:
                futs = [pool.submit(_selfrag_worker, s, h) for (s, h) in targets]
                for fut in as_completed(futs):
                    s, h, ctx = fut.result()
                    selfrag_contexts[(s, h)] = ctx

        # ---- build batch_input.jsonl for this wave ----
        wave_input_path = paths["batch_input_jsonl"]
        with open(wave_input_path, "w", encoding="utf-8") as f:
            for subject, hop in batch:
                root_topic = args.seed if args.domain == "topic" else subject
                messages = build_elicitation_messages_for_subject(
                    domain=args.domain,
                    strategy=args.elicitation_strategy,
                    subject_name=subject,
                    seed=args.seed,
                    root_topic=root_topic,
                    min_sections=args.article_min_sections,
                    max_sections=args.article_max_sections,
                    avg_words_per_article=args.article_avg_words,
                )

                ctx = selfrag_contexts.get((subject, hop))
                if ctx and (ctx.get("summary") or ctx.get("salient_facts")):
                    sr_block = _build_selfrag_block(subject, ctx)
                    messages = _append_block_to_msgs(messages, sr_block, target=args.selfrag_target)

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
                    "url": "/v1/chat/completions",
                    "body": body,
                }
                f.write(json.dumps(req_obj, ensure_ascii=False) + "\n")

        # ---- upload + create batch job ----
        with open(wave_input_path, "rb") as fh:
            batch_input_file = client.files.create(
                file=fh,
                purpose="batch",
            )
        batch_job = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"LLMPedia batch wave {wave_idx} seed={args.seed}"},
        )
        _dbg(
            f"[batch] wave {wave_idx} created batch id={batch_job.id}, "
            f"input_file_id={batch_input_file.id}"
        )

        # ---- poll until completed ----
        poll_interval = args.batch_poll_interval
        while True:
            job = client.batches.retrieve(batch_job.id)
            _dbg(f"[batch] wave {wave_idx} status={job.status}")
            if job.status == "completed":
                break
            if job.status in {"failed", "expired", "cancelled"}:
                raise RuntimeError(
                    f"[batch] wave {wave_idx} batch {job.id} ended with status={job.status}"
                )
            time.sleep(poll_interval)

        if not job.output_file_id:
            raise RuntimeError(f"[batch] wave {wave_idx} batch {job.id} has no output_file_id")

        # ---- download output & process each subject ----
        out_bytes = client.files.content(job.output_file_id).content
        out_path = os.path.join(
            os.path.dirname(paths["batch_input_jsonl"]),
            f"batch_output_wave{wave_idx}_{job.id}.jsonl",
        )
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
                if not isinstance(custom_id, str) or not custom_id.startswith("elicitation::"):
                    continue
                resp = row.get("response") or {}
                body = resp.get("body") or {}
                choices = body.get("choices") or []
                if not choices:
                    continue
                msg = choices[0].get("message") or {}
                wikitext = (msg.get("content") or "").strip()
                if not wikitext:
                    continue

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

                if not wikitext:
                    wikitext = f"'''{subject}'''\n\nNo article content generated (batch)."

                seen_canon_keys = _post_article_processing(
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

        _persist_seen_canon(paths, seen_canon_keys)

    # final snapshots like online
    conn = sqlite3.connect(paths["queue_sqlite"])
    cur = conn.cursor()
    cur.execute("SELECT subject, hop, status, retries, created_at FROM queue ORDER BY hop, subject")
    rows = cur.fetchall()
    with open(paths["queue_json"], "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "subject": s,
                    "hop": h,
                    "status": st,
                    "retries": r,
                    "created_at": ts,
                }
                for (s, h, st, r, ts) in rows
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )
    conn.close()

    conn = sqlite3.connect(paths["articles_sqlite"])
    cur = conn.cursor()
    cur.execute(
        "SELECT subject, wikitext, hop, model_name, overall_confidence, created_at "
        "FROM llmpedia_articles ORDER BY subject"
    )
    arows = cur.fetchall()
    with open(paths["articles_json"], "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "subject": s,
                    "wikitext": wt,
                    "hop": h,
                    "model": m,
                    "overall_confidence": oc,
                    "created_at": ts,
                }
                for (s, wt, h, m, oc, ts) in arows
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )
    conn.close()

    dur = time.perf_counter() - start
    _dbg(f"[done-batch] finished in {dur:.1f}s → {os.path.dirname(paths['queue_sqlite'])}")


# ---------------- main() ----------------


def main():
    ap = argparse.ArgumentParser(
        description="LLMPedia crawler: online & OpenAI batch modes with optional Self-RAG."
    )

    ap.add_argument(
        "--mode",
        choices=["online", "batch"],
        default="online",
        help="online = normal BFS; batch = full OpenAI Batch pipeline (Self-RAG + articles + NER + queue).",
    )
    ap.add_argument("--seed", required=True, help="Seed entity name (e.g., 'Alan Turing').")
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
        default=settings.MAX_DEPTH,
        help="0 = unlimited depth (stop when queue empty)",
    )
    ap.add_argument(
        "--max-subjects",
        type=int,
        default=0,
        help="0 = unlimited subjects",
    )

    # article prompt controls
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

    # models & sampling
    ap.add_argument(
        "--elicit-model-key",
        default=settings.ELICIT_MODEL_KEY,
        help="settings.MODELS key for article generation (elicitation).",
    )
    ap.add_argument(
        "--ner-model-key",
        default=getattr(settings, "NER_MODEL_KEY", settings.ELICIT_MODEL_KEY),
        help="settings.MODELS key for NER.",
    )
    ap.add_argument(
        "--selfrag-model-key",
        default=None,
        help="settings.MODELS key for Self-RAG (defaults to elicit-model-key).",
    )

    ap.add_argument("--elicit-temperature", type=float, default=0.4)
    ap.add_argument("--ner-temperature", type=float, default=0.3)
    ap.add_argument("--selfrag-temperature", type=float, default=0.1)

    ap.add_argument("--elicit-top-p", type=float, default=None)
    ap.add_argument("--ner-top-p", type=float, default=None)
    ap.add_argument("--selfrag-top-p", type=float, default=None)

    ap.add_argument("--elicit-top-k", type=int, default=None)
    ap.add_argument("--ner-top-k", type=int, default=None)
    ap.add_argument("--selfrag-top-k", type=int, default=None)

    ap.add_argument("--elicit-max-tokens", type=int, default=3072)
    ap.add_argument("--ner-max-tokens", type=int, default=2048)
    ap.add_argument("--selfrag-max-tokens", type=int, default=512)

    ap.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="Request timeout (seconds) for online calls.",
    )

    # NER / elicitation thresholds
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
        help="If >0, interpret confidences encoded in wikilinks ([[Entity (0.93)]]) and filter below threshold.",
    )

    # footer controls
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
        "--selfrag-target",
        choices=["system", "user"],
        default="user",
        help="Where to append the Self-RAG context.",
    )
    ap.add_argument(
        "--selfrag-batch-size",
        type=int,
        default=0,
        help="In mode=batch, max number of subjects per wave that get Self-RAG (0 = all subjects in wave).",
    )
    ap.add_argument(
        "--selfrag-concurrency",
        type=int,
        default=1,
        help="In mode=batch, how many Self-RAG calls run in parallel per wave (ignored in mode=online).",
    )

    # reasoning overrides for Responses API (gpt-5*)
    ap.add_argument("--reasoning-effort", choices=["minimal", "low", "medium", "high"], default=None)
    ap.add_argument("--text-verbosity", choices=["low", "medium", "high"], default=None)

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

    out_dir = _ensure_output_dir(args.output_dir)
    paths = _build_paths(out_dir)
    _dbg(
        f"[llmpedia-combined] mode={args.mode} output_dir={out_dir} "
        f"max_depth={args.max_depth} max_subjects={args.max_subjects}"
    )

    open_queue_db(paths["queue_sqlite"])
    open_llmpedia_db(paths["articles_sqlite"])

    el_cfg = settings.MODELS[args.elicit_model_key].model_copy(deep=True)
    ner_cfg = settings.MODELS[args.ner_model_key].model_copy(deep=True)
    selfrag_key = args.selfrag_model_key or args.elicit_model_key
    selfrag_cfg = settings.MODELS[selfrag_key].model_copy(deep=True)

    _apply_stage("elicit", el_cfg, args)
    _apply_stage("ner", ner_cfg, args)
    # selfrag
    if getattr(selfrag_cfg, "use_responses_api", False):
        if selfrag_cfg.extra_inputs is None:
            selfrag_cfg.extra_inputs = {}
        selfrag_cfg.extra_inputs.setdefault("reasoning", {})
        selfrag_cfg.extra_inputs.setdefault("text", {})
    else:
        selfrag_cfg.temperature = args.selfrag_temperature
        if args.selfrag_top_p is not None:
            selfrag_cfg.top_p = args.selfrag_top_p
        if args.selfrag_top_k is not None:
            selfrag_cfg.top_k = args.selfrag_top_k
    selfrag_cfg.max_tokens = args.selfrag_max_tokens

    start = time.perf_counter()

    if args.mode == "online":
        run_online(args, paths, el_cfg, ner_cfg, selfrag_cfg)
    elif args.mode == "batch":
        run_batch(args, paths, el_cfg, ner_cfg, selfrag_cfg)
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
        "max_depth": args.max_depth,
        "max_subjects": args.max_subjects,
        "batch_size": args.batch_size,
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
            "selfrag": {
                "provider": getattr(selfrag_cfg, "provider", "openai"),
                "model": getattr(selfrag_cfg, "model", None),
                "use_responses_api": getattr(selfrag_cfg, "use_responses_api", False),
                "temperature": getattr(selfrag_cfg, "temperature", None),
                "top_p": getattr(selfrag_cfg, "top_p", None),
                "top_k": getattr(selfrag_cfg, "top_k", None),
                "max_tokens": getattr(selfrag_cfg, "max_tokens", None),
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
        "selfrag_log_jsonl",
        "run_meta_json",
        "errors_log",
        "seen_state_json",
        "batch_input_jsonl",
    ):
        if k in paths:
            print(f"[out] {k:18}: {paths[k]}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[interrupt] bye")

# crawler_runner_heuristics3.py
from __future__ import annotations

import argparse, datetime, json, os, re, sqlite3, threading, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Set, Optional

from dotenv import load_dotenv
load_dotenv()

# ---------- locks & tiny utils ----------
_jsonl_lock = threading.Lock()
_seen_facts_lock = threading.Lock()
_lowconf_lock = threading.Lock()
_ner_lowconf_lock = threading.Lock()

def _append_jsonl(path: str, obj: dict):
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with _jsonl_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)

def _dbg(msg: str): print(msg, flush=True)

def _print_messages(tag: str, msgs: List[dict], limit: int | None = None):
    print(f"\n--- {tag} MESSAGES ({len(msgs)}) ---")
    for i, m in enumerate(msgs, 1):
        role = (m.get("role") or "").upper()
        content = m.get("content")
        if isinstance(content, str) and limit:
            content = (content[:limit] + "…") if len(content) > limit else content
        print(f"[{i:02d}] {role}: {content if isinstance(content, str) else content}")
    print(f"--- END {tag} ---\n")

def _print_enqueue_summary(results: List[Tuple[str,int,str]]):
    if not results:
        print("[enqueue] (no results)")
        return
    ins = sum(1 for *_r, out in results if out == "inserted")
    red = sum(1 for *_r, out in results if out == "hop_reduced")
    ign = sum(1 for *_r, out in results if out == "ignored")
    print(f"[enqueue] inserted={ins} hop_reduced={red} ignored={ign}")

# ---------- repo imports ----------
from processing_queue import (
    init_cache as procq_init_cache,
    enqueue_subjects_processed as procq_enqueue,
    DEFAULT_LEADING_ARTICLES as PROCQ_LEADING,
    get_thread_queue_conn as procq_get_thread_conn,
)
from settings import (
    settings,
    ELICIT_SCHEMA_BASE, ELICIT_SCHEMA_CAL,
    NER_SCHEMA_BASE,   NER_SCHEMA_CAL,
)
from prompter_parser import get_prompt_messages
from llm.factory import make_llm_from_config
from db_models import (
    open_queue_db, open_facts_db,
    write_triples_accepted, write_triples_sink,
    queue_has_rows, reset_working_to_pending,
)

# NEW: shared JSON extractor
from llm.json_utils import best_json

# Optional OpenAI SDK (for Batch API)
try:
    from openai import OpenAI as _OpenAI
except Exception:
    _OpenAI = None

# ---------- paths ----------
def _ensure_output_dir(base_dir: Optional[str]) -> str:
    out = base_dir or os.path.join("runs", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    return out

def _build_paths(out_dir: str) -> dict:
    tmp = os.path.join(out_dir, "tmp")
    os.makedirs(tmp, exist_ok=True)
    return {
        "queue_sqlite": os.path.join(out_dir, "queue.sqlite"),
        "facts_sqlite": os.path.join(out_dir, "facts.sqlite"),
        "queue_jsonl": os.path.join(out_dir, "queue.jsonl"),
        "facts_jsonl": os.path.join(out_dir, "facts.jsonl"),
        "queue_json": os.path.join(out_dir, "queue.json"),
        "facts_json": os.path.join(out_dir, "facts.json"),
        "errors_log": os.path.join(out_dir, "errors.log"),
        "ner_jsonl": os.path.join(out_dir, "ner_decisions.jsonl"),
        "lowconf_json": os.path.join(out_dir, "facts_lowconf.json"),
        "lowconf_jsonl": os.path.join(out_dir, "facts_lowconf.jsonl"),
        "ner_lowconf_jsonl": os.path.join(out_dir, "ner_lowconf.jsonl"),
        "ner_lowconf_json": os.path.join(out_dir, "ner_lowconf.json"),
        "run_meta_json": os.path.join(out_dir, "run_meta.json"),
        "tmp_dir": tmp,
        "batch_req_jsonl": os.path.join(tmp, "batch_requests.jsonl"),
        "batch_out_jsonl": os.path.join(tmp, "batch_results.jsonl"),
    }

def _write_queue_snapshot(qdb: sqlite3.Connection, snapshot_path: str, max_depth: int):
    cur = qdb.cursor()
    if max_depth == 0:
        cur.execute("SELECT subject, hop, status, retries, created_at FROM queue ORDER BY hop, subject")
    else:
        cur.execute("SELECT subject, hop, status, retries, created_at FROM queue WHERE hop<=? ORDER BY hop, subject", (max_depth,))
    rows = cur.fetchall()
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(
            [{"subject": s, "hop": h, "status": st, "retries": r, "created_at": ts} for (s, h, st, r, ts) in rows],
            f, ensure_ascii=False, indent=2
        )

# ---------- per-thread sqlite ----------
_thread_local = threading.local()

def get_thread_queue_conn(db_path: str) -> sqlite3.Connection:
    return procq_get_thread_conn(db_path)

def get_thread_facts_conn(db_path: str) -> sqlite3.Connection:
    key = f"facts_conn__{db_path}"
    conn = getattr(_thread_local, key, None)
    if conn is None:
        conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        setattr(_thread_local, key, conn)
    return conn

def mark_done_threadsafe(queue_db_path: str, subject: str, hop: int):
    conn = get_thread_queue_conn(queue_db_path)
    with conn:
        conn.execute("UPDATE queue SET status='done' WHERE subject=? AND hop=? AND status='working'", (subject, hop))

def mark_pending_on_error(queue_db_path: str, subject: str, hop: int):
    conn = get_thread_queue_conn(queue_db_path)
    with conn:
        conn.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=? AND hop=? AND status='working'", (subject, hop))

def _get_retries(queue_db_path: str, subject: str, hop: int) -> int:
    conn = get_thread_queue_conn(queue_db_path)
    cur = conn.cursor()
    cur.execute("SELECT retries FROM queue WHERE subject=? AND hop=?", (subject, hop))
    row = cur.fetchone()
    return int(row[0]) if row else 0

def _inc_retries_and_pending(queue_db_path: str, subject: str, hop: int):
    conn = get_thread_queue_conn(queue_db_path)
    with conn:
        conn.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=? AND hop=?", (subject, hop))

# ---------- claim helpers ----------
def _fetch_one_pending(conn: sqlite3.Connection, max_depth: int) -> Tuple[str,int] | None:
    cur = conn.cursor()
    try:
        if max_depth == 0:
            cur.execute("""
                UPDATE queue SET status='working'
                WHERE rowid = (SELECT rowid FROM queue WHERE status='pending' ORDER BY hop, created_at LIMIT 1)
                RETURNING subject, hop
            """)
        else:
            cur.execute("""
                UPDATE queue SET status='working'
                WHERE rowid = (SELECT rowid FROM queue WHERE status='pending' AND hop<=?
                               ORDER BY hop, created_at LIMIT 1)
                RETURNING subject, hop
            """, (max_depth,))
        row = cur.fetchone()
        conn.commit()
        return (row[0], row[1]) if row else None
    except sqlite3.OperationalError:
        cur.execute("BEGIN IMMEDIATE")
        if max_depth == 0:
            cur.execute("SELECT rowid, subject, hop FROM queue WHERE status='pending' ORDER BY hop, created_at LIMIT 1")
        else:
            cur.execute("SELECT rowid, subject, hop FROM queue WHERE status='pending' AND hop<=? ORDER BY hop, created_at LIMIT 1", (max_depth,))
        row = cur.fetchone()
        if not row:
            conn.commit(); return None
        rowid, subject, hop = row
        cur.execute("UPDATE queue SET status='working' WHERE rowid=? AND status='pending'", (rowid,))
        changed = cur.rowcount
        conn.commit()
        return (subject, hop) if changed else None

def _fetch_many_pending(conn: sqlite3.Connection, max_depth: int, limit: int) -> List[Tuple[str,int]]:
    got = []
    for _ in range(max(1,limit)):
        one = _fetch_one_pending(conn, max_depth)
        if not one: break
        got.append(one)
    return got

def _counts(conn: sqlite3.Connection, max_depth: int):
    cur = conn.cursor()
    if max_depth == 0:
        cur.execute("SELECT COUNT(1) FROM queue WHERE status='done'"); done = cur.fetchone()[0]
        cur.execute("SELECT COUNT(1) FROM queue WHERE status='working'"); working = cur.fetchone()[0]
        cur.execute("SELECT COUNT(1) FROM queue WHERE status='pending'"); pending = cur.fetchone()[0]
    else:
        cur.execute("SELECT COUNT(1) FROM queue WHERE status='done' AND hop<=?", (max_depth,)); done = cur.fetchone()[0]
        cur.execute("SELECT COUNT(1) FROM queue WHERE status='working' AND hop<=?", (max_depth,)); working = cur.fetchone()[0]
        cur.execute("SELECT COUNT(1) FROM queue WHERE status='pending' AND hop<=?", (max_depth,)); pending = cur.fetchone()[0]
    return done, working, pending, done + working + pending

# ---------- unwrap & salvage ----------
def _parse_obj(maybe_json) -> dict:
    if isinstance(maybe_json, dict): return maybe_json
    if isinstance(maybe_json, str):
        try: return json.loads(maybe_json)
        except Exception: return {}
    return {}

def _unwrap_text(resp):
    if isinstance(resp, str): return resp
    if isinstance(resp, dict):
        for k in ("text","output_text","content","message","response"):
            v = resp.get(k)
            if isinstance(v, str): return v
        ch = resp.get("choices")
        if isinstance(ch, list) and ch:
            c0 = ch[0] or {}
            msg = c0.get("message") or {}
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
            if isinstance(c0.get("text"), str): return c0["text"]
        # NEW: handle our client wrappers
        if isinstance(resp.get("_raw"), str): return resp["_raw"]
        if isinstance(resp.get("raw"), str):  return resp["raw"]
        if isinstance(resp.get("raw"), dict): return _unwrap_text(resp["raw"])
    return ""

def _extract_json_block(text: str):
    obj = best_json(text)
    return obj if isinstance(obj, (dict, list)) else {}

def _normalize_fact_keys(d: dict) -> dict | None:
    if not isinstance(d, dict): return None
    key_map = {
        "subject": ["subject","subj","s","head","h"],
        "predicate": ["predicate","pred","p","relation","rel","r"],
        "object": ["object","obj","o","tail","t","value","val"],
        "confidence": ["confidence","conf","c","score","prob"]
    }
    out = {}
    for std, alts in key_map.items():
        for k in alts:
            if k in d and isinstance(d[k], (str, float, int)):
                out[std] = d[k]
                break
    s,p,o = out.get("subject"), out.get("predicate"), out.get("object")
    if not (isinstance(s,str) and isinstance(p,str) and isinstance(o,str)):
        return None
    if "confidence" in out:
        try: out["confidence"] = float(out["confidence"])
        except Exception: out["confidence"] = None
    else:
        out["confidence"] = None
    return out

_TRIPLE_OBJ_RX = re.compile(r"\{[^{}]*?(\"subject\"|\"subj\"|\"s\"|\"head\")[^{}]*?\}", re.I)
_FLEX_TRIPLE_RX = re.compile(r"\{[^{}]*\}", re.S)

def _salvage_facts_from_text(text: str, debug=False) -> List[dict]:
    salvaged: List[dict] = []

    obj = _extract_json_block(text)
    if obj:
        if isinstance(obj, dict):
            for key in ("facts","triples"):
                val = obj.get(key)
                if isinstance(val, list):
                    for item in val:
                        norm = _normalize_fact_keys(item)
                        if norm: salvaged.append(norm)
            if not salvaged:
                norm = _normalize_fact_keys(obj)
                if norm: salvaged.append(norm)
        elif isinstance(obj, list):
            for item in obj:
                norm = _normalize_fact_keys(item)
                if norm: salvaged.append(norm)

    if not salvaged:
        for m in _TRIPLE_OBJ_RX.finditer(text or ""):
            chunk = m.group(0)
            try:
                d = json.loads(chunk)
                norm = _normalize_fact_keys(d)
                if norm: salvaged.append(norm)
            except Exception:
                patched = chunk
                open_br = chunk.count("{")
                close_br = chunk.count("}")
                patched += "}" * max(0, open_br - close_br)
                try:
                    d = json.loads(patched)
                    norm = _normalize_fact_keys(d)
                    if norm: salvaged.append(norm)
                except Exception:
                    continue

    # extra flexible pass: any dicts
    if not salvaged:
        for m in _FLEX_TRIPLE_RX.finditer(text or ""):
            try:
                d = json.loads(m.group(0))
            except Exception:
                continue
            norm = _normalize_fact_keys(d)
            if norm:
                salvaged.append(norm)

    if debug and salvaged:
        print(f"[salvage] recovered {len(salvaged)} triples from noisy output")

    facts = []
    for t in salvaged:
        facts.append({
            "subject": t["subject"],
            "predicate": t["predicate"],
            "object": t["object"],
            "confidence": t.get("confidence")
        })
    return facts

def _extract_facts_from_resp(resp, debug=False) -> Tuple[List[dict], str]:
    if isinstance(resp, list):
        facts = [t for t in resp if isinstance(t, dict)]
        return facts, ""
    if isinstance(resp, dict):
        for key in ("facts","triples"):
            val = resp.get(key)
            if isinstance(val, list):
                return [t for t in val if isinstance(t, dict)], ""
    txt = _unwrap_text(resp)
    obj = _extract_json_block(txt)
    if isinstance(obj, dict):
        for key in ("facts","triples"):
            val = obj.get(key)
            if isinstance(val, list):
                return [t for t in val if isinstance(t, dict)], txt
    if isinstance(obj, list):
        return [t for t in obj if isinstance(t, dict)], txt
    return [], txt

# ---------- NER heuristics ----------
_date_rx = re.compile(r"^\d{4}([-/]\d{2}){0,2}$|^(January|February|March|April|May|June|July|August|September|October|November|December)\b", re.I)
_url_rx  = re.compile(r"^https?://", re.I)
def _is_date_like(s:str)->bool: return bool(_date_rx.search(s or ""))
def _is_literal_like(s:str)->bool:
    s = s or ""
    if _url_rx.search(s): return True
    if s.isdigit(): return True
    if s.strip().lower() in {"human","engineer","inventor","person","male","female"}: return True
    return False
def _titlecase_ratio(s:str)->float:
    words = [w for w in re.split(r"\s+", (s or "").strip()) if w]
    if not words: return 0.0
    caps = sum(1 for w in words if w[:1].isupper())
    return caps/len(words)
_variant_rx = re.compile(r"[\(\)\[\]\{\}:–—\-]")
def _norm(s:str)->str: return re.sub(r"\s+"," ",(s or "")).strip().lower()
def _is_subject_variant(phrase:str, subject:str)->bool:
    ps, ss = _norm(phrase), _norm(subject)
    if not ps or not ss: return False
    if ps == ss: return True
    if ps.startswith(ss+" (") or ps.startswith(ss+" -") or ps.startswith(ss+":"): return True
    if _variant_rx.sub("", ps) == _variant_rx.sub("", ss): return True
    if ps.startswith(ss) and any(ch in ps[len(ss):len(ss)+3] for ch in "():-—–[]{}"): return True
    return False
def _maybe_is_ne_heuristic(phrase:str)->bool:
    if not isinstance(phrase,str): return False
    p = phrase.strip()
    if not p: return False
    if _is_date_like(p) or _is_literal_like(p): return False
    if " " not in p and p.islower(): return False
    if _titlecase_ratio(p) >= 0.6: return True
    if " " in p and not p.islower(): return True
    return False
def _filter_ner_candidates(objs: List[str], subject: Optional[str]=None)->List[str]:
    uniq:Set[str] = set()
    for o in objs:
        if not isinstance(o,str): continue
        o2 = o.strip()
        if not o2: continue
        if len(o2.split())>6: continue
        if subject and _is_subject_variant(o2, subject): continue
        if _is_date_like(o2) or _is_literal_like(o2): continue
        uniq.add(o2)
    return sorted(uniq)

# ---------- prompts ----------
def _ensure_json_keyword_in_msgs(msgs: List[dict], shape_hint: str):
    has_json = any(isinstance(m.get("content"),str) and "json" in (m.get("content") or "").lower() for m in msgs)
    if not has_json:
        # Prepend as system for maximum priority
        msgs.insert(0, {"role":"system","content":f"Output ONLY JSON; shape: {shape_hint}"})

def _build_elicitation_messages(args, subject:str)->List[dict]:
    msgs = get_prompt_messages(
        args.elicitation_strategy, "elicitation",
        domain=args.domain,
        variables=dict(subject_name=subject, root_subject=args.seed, max_facts_hint=args.max_facts_hint),
    )
    if getattr(args,"footer_mode",False):
        footer = ("\n\nFinal important note:\n"
                  "If the entity is famous, aim ~50 distinct triplets; else ~10 if any exist. "
                  "Only concrete, verifiable info.")
        for m in msgs:
            if m.get("role")=="system":
                m["content"] = (m.get("content") or "") + footer
                break
        else:
            msgs.insert(0, {"role":"system","content":footer})
    return msgs

# ---------- provider helpers ----------
def _is_openai_model(cfg)->bool:
    prov = (getattr(cfg,"provider","") or "").lower()
    if "openai" in prov: return True
    name = (getattr(cfg,"model","") or "").lower()
    return "openai" in name or name.startswith("gpt-")

def _route_facts(args, facts: List[dict], hop:int, model_name:str):
    acc, lowconf, objs = [], [], []
    use_thr = (args.elicitation_strategy == "calibrate")
    thr = float(args.conf_threshold)
    for f in facts:
        s, p, o = f.get("subject"), f.get("predicate"), f.get("object")
        if not (isinstance(s,str) and isinstance(p,str) and isinstance(o,str)): continue
        conf = f.get("confidence")
        if use_thr and isinstance(conf,(int,float)) and conf < thr:
            lowconf.append({
                "subject": s, "predicate": p, "object": o,
                "hop": hop, "model": model_name, "strategy": args.elicitation_strategy,
                "confidence": float(conf), "threshold": thr
            })
            continue
        acc.append((s,p,o,hop,model_name,args.elicitation_strategy, float(conf) if isinstance(conf,(int,float)) else None))
        objs.append(o)
    return acc, lowconf, objs

# ---------- OpenAI Batch helpers ----------
def _make_openai_client_for_batch(el_cfg):
    if _OpenAI is None:
        raise RuntimeError("OpenAI SDK not installed. `pip install openai`")
    api_key_env = getattr(el_cfg, "api_key_env", "OPENAI_API_KEY")
    api_key = os.getenv(api_key_env or "OPENAI_API_KEY")
    if not api_key: raise RuntimeError(f"Missing {api_key_env or 'OPENAI_API_KEY'} for Batch mode.")
    base_url = getattr(el_cfg, "base_url", None)
    return _OpenAI(api_key=api_key, base_url=base_url) if base_url else _OpenAI(api_key=api_key)

def _write_batch_requests_jsonl(fp: str, subjects: List[str], el_cfg, messages_builder, args):
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    schema = ELICIT_SCHEMA_CAL if (args.elicitation_strategy == "calibrate") else ELICIT_SCHEMA_BASE
    with open(fp, "w", encoding="utf-8") as f:
        for subject in subjects:
            msgs = messages_builder(args, subject)
            _ensure_json_keyword_in_msgs(msgs, shape_hint='{"facts":[{"subject":"...","predicate":"...","object":"..."}]}')
            body = {
                "model": el_cfg.model,
                "messages": msgs,
                "temperature": getattr(el_cfg,"temperature", None),
                "top_p": getattr(el_cfg,"top_p", None),
                "max_tokens": getattr(el_cfg,"max_tokens", 2048),
                "response_format": {
                    "type":"json_schema",
                    "json_schema": {"name":"schema","schema": schema, "strict": True}
                }
            }
            line = {"custom_id": subject, "method":"POST", "url":"/v1/chat/completions", "body": body}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

def _parse_openai_batch_output_line(line: str, debug: bool=False) -> Tuple[str, List[dict], str]:
    try:
        obj = json.loads(line)
    except Exception:
        if debug: print(f"[batch-parse] not JSON line: {line[:200]} ...")
        return "", [], ""

    subject = obj.get("custom_id") or ""
    resp_body = ((obj.get("response") or {}).get("body")) or {}
    choices = resp_body.get("choices") or []
    content_text = ""
    if choices:
        msg = (choices[0] or {}).get("message") or {}
        content_text = (msg.get("content") or "").strip()
        if not content_text:
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                try:
                    arguments = ((tool_calls[0] or {}).get("function") or {}).get("arguments")
                    if isinstance(arguments, str):
                        content_text = arguments
                    elif isinstance(arguments, dict):
                        content_text = json.dumps(arguments)
                except Exception:
                    pass

    parsed = {}
    if content_text:
        try:
            parsed = json.loads(content_text)
        except Exception:
            parsed = best_json(content_text) or {}

    if not parsed and isinstance(resp_body, dict):
        parsed = best_json(json.dumps(resp_body)) or {}

    facts: List[dict] = []
    if isinstance(parsed, dict):
        facts = parsed.get("facts") or parsed.get("triples") or []
    elif isinstance(parsed, list):
        facts = parsed

    facts = [t for t in facts if isinstance(t, dict)]
    return subject, facts, content_text

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Crawler v3: salvage & retries; max-inflight only for OpenAI Batch.")
    ap.add_argument("--seed", required=True)
    ap.add_argument("--output-dir", default=None)

    ap.add_argument("--elicitation-strategy", default="baseline", choices=["baseline","icl","dont_know","calibrate"])
    ap.add_argument("--ner-strategy", default="baseline", choices=["baseline","icl","dont_know","calibrate"])
    ap.add_argument("--domain", default="general", choices=["general","topic"])

    ap.add_argument("--max-depth", type=int, default=settings.MAX_DEPTH, help="0 = unlimited depth (stop when queue empty)")
    ap.add_argument("--max-subjects", type=int, default=0, help="0 = unlimited subjects")
    ap.add_argument("--ner-batch-size", type=int, default=settings.NER_BATCH_SIZE)
    ap.add_argument("--max-facts-hint", default=str(settings.MAX_FACTS_HINT))
    ap.add_argument("--conf-threshold", type=float, default=0.7)
    ap.add_argument("--ner-conf-threshold", type=float, default=0.9)
    ap.add_argument("--footer-mode", action="store_true")

    ap.add_argument("--elicit-model-key", default=settings.ELICIT_MODEL_KEY)
    ap.add_argument("--ner-model-key", default=settings.NER_MODEL_KEY)

    ap.add_argument("--elicit-temperature", type=float, default=0.7)
    ap.add_argument("--ner-temperature", type=float, default=0.3)
    ap.add_argument("--elicit-top-p", type=float, default=None)
    ap.add_argument("--ner-top-p", type=float, default=None)
    ap.add_argument("--elicit-top-k", type=int, default=None)
    ap.add_argument("--ner-top-k", type=int, default=None)
    ap.add_argument("--elicit-max-tokens", type=int, default=4096)
    ap.add_argument("--ner-max-tokens", type=int, default=4096)

    ap.add_argument("--batch-size", type=int, default=1, help="Subjects grouped per realtime .batch() call (if supported)")
    ap.add_argument("--concurrency", type=int, default=8, help="Workers for providers without realtime batching")
    ap.add_argument("--max-inflight", type=int, default=None, help="[OpenAI Batch ONLY] subjects to claim per batch")
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--max-retries", type=int, default=3, help="Max attempts per subject (non-batch) or per subject line (batch).")

    ap.add_argument("--openai-batch-mode", action="store_true", help="Use OpenAI Batch API for elicitation (chat-completions only)")

    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--progress-metrics", dest="progress_metrics", action="store_true", default=True)
    ap.add_argument("--no-progress-metrics", dest="progress_metrics", action="store_false")

    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reset-working", action="store_true")

    args = ap.parse_args()

    out_dir = _ensure_output_dir(args.output_dir)
    paths = _build_paths(out_dir)
    _dbg(f"[runner] output_dir: {out_dir}")

    qdb = open_queue_db(paths["queue_sqlite"])
    fdb = open_facts_db(paths["facts_sqlite"])
    procq_init_cache(qdb)

    # seed/resume
    if args.resume:
        if not queue_has_rows(qdb):
            for s, kept_hop, outcome in procq_enqueue(paths["queue_sqlite"], [(args.seed, 0)], leading_articles=PROCQ_LEADING):
                if outcome in ("inserted","hop_reduced"):
                    _append_jsonl(paths["queue_jsonl"], {"subject": s, "hop": kept_hop, "event": outcome})
            _write_queue_snapshot(qdb, paths["queue_json"], args.max_depth)
        else:
            if args.reset_working:
                n = reset_working_to_pending(qdb)
                _dbg(f"[resume] reset {n} working→pending")
    else:
        for s, kept_hop, outcome in procq_enqueue(paths["queue_sqlite"], [(args.seed, 0)], leading_articles=PROCQ_LEADING):
            if outcome in ("inserted","hop_reduced"):
                _append_jsonl(paths["queue_jsonl"], {"subject": s, "hop": kept_hop, "event": outcome})
        _write_queue_snapshot(qdb, paths["queue_json"], args.max_depth)

    # build cfgs + apply stage params
    el_cfg = settings.MODELS[args.elicit_model_key].model_copy(deep=True)
    ner_cfg = settings.MODELS[args.ner_model_key].model_copy(deep=True)

    def _apply_stage(which, cfg):
        if getattr(cfg,"use_responses_api", False):
            cfg.temperature = None; cfg.top_p = None; cfg.top_k = None
            if cfg.extra_inputs is None: cfg.extra_inputs = {}
            cfg.extra_inputs.setdefault("reasoning", {})
            cfg.extra_inputs.setdefault("text", {})
        else:
            t  = getattr(args, f"{which}_temperature")
            tp = getattr(args, f"{which}_top_p")
            tk = getattr(args, f"{which}_top_k")
            if t  is not None: cfg.temperature = t
            if tp is not None: cfg.top_p = tp
            if tk is not None: cfg.top_k = tk
        mt = getattr(args, f"{which}_max_tokens")
        if mt is not None: cfg.max_tokens = mt
        if getattr(cfg,"max_tokens", None) is None:
            cfg.max_tokens = 2048
        if hasattr(cfg,"request_timeout"): cfg.request_timeout = args.timeout
        elif hasattr(cfg,"timeout"):       cfg.timeout = args.timeout

    _apply_stage("elicit", el_cfg)
    _apply_stage("ner", ner_cfg)

    el_llm = make_llm_from_config(el_cfg)
    ner_llm = make_llm_from_config(ner_cfg)

    is_openai_el = _is_openai_model(el_cfg)
    uses_responses = bool(getattr(el_cfg,"use_responses_api", False))
    supports_realtime_batch = hasattr(el_llm, "batch")

    # ---- enforce policy for max-inflight ----
    if args.openai_batch_mode:
        if not is_openai_el:
            raise SystemExit("--openai-batch-mode requires an OpenAI Chat Completions model.")
        if uses_responses:
            raise SystemExit("--openai-batch-mode incompatible with Responses (gpt-5*) models; use chat-completions.")
        if args.concurrency and args.concurrency != 1:
            _dbg("[note] --openai-batch-mode: ignoring --concurrency; Batch API is offline.")
        if args.max_inflight is None:
            args.max_inflight = max(1, args.batch_size)
    else:
        if args.max_inflight is not None:
            _dbg("[note] ignoring --max-inflight (only honored with --openai-batch-mode).")
        args.max_inflight = None

    # progress timing
    last_progress_ts = 0.0

    # shared state
    start = time.perf_counter()
    subjects_elicited_total = 0
    lowconf_accum: List[dict] = []
    ner_lowconf_accum: List[dict] = []
    seen_facts: Set[Tuple[str,str,str,int]] = set()

    # ---- worker for non-realtime-batch path (with retries + salvage) ----
    def _elicitation_and_ner(subject: str, hop: int):
        try:
            attempt = 0
            facts: List[dict] = []
            last_text = ""
            el_schema = ELICIT_SCHEMA_CAL if (args.elicitation_strategy=="calibrate") else ELICIT_SCHEMA_BASE

            while attempt < max(1, args.max_retries):
                el_messages = _build_elicitation_messages(args, subject)
                _ensure_json_keyword_in_msgs(el_messages, shape_hint='{"facts":[{"subject":"...","predicate":"...","object":"..."}]}')
                if args.debug: _print_messages(f"ELICIT for [{subject}] (try {attempt+1})", el_messages)

                try:
                    resp = el_llm(el_messages, json_schema=el_schema)
                except Exception:
                    resp = el_llm(el_messages)

                facts, last_text = _extract_facts_from_resp(resp, debug=args.debug)

                if not facts and last_text:
                    salv = _salvage_facts_from_text(last_text, debug=args.debug)
                    if salv:
                        facts = salv

                if facts:
                    break
                attempt += 1

            if not facts:
                write_triples_sink(get_thread_facts_conn(paths["facts_sqlite"]),
                    [(subject,"__empty__","__empty__",hop, el_cfg.model,args.elicitation_strategy,None,"empty_or_unparseable_output")]
                )

            acc, lowconf, _ = _route_facts(args, facts, hop, el_cfg.model)
            if acc:
                write_triples_accepted(get_thread_facts_conn(paths["facts_sqlite"]), acc)
                with _seen_facts_lock:
                    for s,p,o,_,m,st,c in acc:
                        key = (s,p,o,hop)
                        if key not in seen_facts:
                            seen_facts.add(key)
                            _append_jsonl(paths["facts_jsonl"], {
                                "subject": s, "predicate": p, "object": o,
                                "hop": hop, "model": m, "strategy": st, "confidence": c
                            })
            if lowconf:
                for item in lowconf: _append_jsonl(paths["lowconf_jsonl"], item)
                with _lowconf_lock: lowconf_accum.extend(lowconf)

            # NER
            cand = _filter_ner_candidates([t.get("object") for t in facts if isinstance(t, dict)], subject)
            next_subjects: List[str] = []
            i = 0
            while i < len(cand):
                chunk = cand[i: i + args.ner_batch_size]
                ner_messages = get_prompt_messages(args.ner_strategy, "ner",
                    domain=args.domain,
                    variables=dict(phrases_block="\n".join(chunk), root_subject=args.seed, subject_name=subject))
                ner_schema = NER_SCHEMA_CAL if (args.ner_strategy=="calibrate") else NER_SCHEMA_BASE
                if args.debug: _print_messages(f"NER for [{subject}] chunk[{i}:{i+args.ner_batch_size}]", ner_messages)
                try:
                    out = ner_llm(ner_messages, json_schema=ner_schema)
                except Exception:
                    out = ner_llm(ner_messages)
                norm = _parse_obj(out)
                decisions = norm.get("phrases", []) if isinstance(norm.get("phrases"), list) else []
                if not decisions:
                    decisions = [{"phrase": ph, "is_ne": _maybe_is_ne_heuristic(ph), "confidence": None} for ph in chunk]

                # >>> force numeric confidence in calibrate, if missing <<<
                if args.ner_strategy == "calibrate":
                    for d in decisions:
                        if not isinstance(d.get("confidence"), (int, float)):
                            d["confidence"] = 0.90

                use_thr = (args.ner_strategy=="calibrate")
                for d in decisions:
                    phrase = d.get("phrase"); is_ne = bool(d.get("is_ne"))
                    conf = d.get("confidence")
                    try: conf = float(conf)
                    except Exception: conf = None
                    is_variant = _is_subject_variant(phrase, subject)
                    if is_variant:
                        is_ne = False; conf = 0.0 if conf is None else min(conf, 0.0)
                    conf_ok = (isinstance(conf,(int,float)) and conf >= args.ner_conf_threshold) if use_thr else True
                    record = {
                        "current_entity": subject, "hop": hop, "phrase": phrase,
                        "is_ne": is_ne, "is_variant": is_variant,
                        "confidence": (float(conf) if isinstance(conf,(int,float)) else None),
                        "ner_conf_threshold": float(args.ner_conf_threshold),
                        "passed_threshold": bool(conf_ok if use_thr else True),
                        "ner_model": ner_cfg.model, "ner_strategy": args.ner_strategy,
                        "domain": args.domain, "root_subject": args.seed, "source": "model_or_fallback"
                    }
                    _append_jsonl(paths["ner_jsonl"], record)
                    if use_thr and not conf_ok:
                        low_item = {**record, "reason":"below_threshold"}
                        _append_jsonl(paths["ner_lowconf_jsonl"], low_item)
                        with _ner_lowconf_lock: ner_lowconf_accum.append(low_item)
                    if is_ne and conf_ok and not is_variant and isinstance(phrase,str):
                        next_subjects.append(phrase)
                i += args.ner_batch_size

            if next_subjects:
                results = procq_enqueue(
                    paths["queue_sqlite"],
                    [(s, hop+1) for s in next_subjects if (args.max_depth==0 or hop+1<=args.max_depth)],
                    leading_articles=PROCQ_LEADING
                )
                for s, kept_hop, outcome in results:
                    if outcome in ("inserted","hop_reduced"):
                        _append_jsonl(paths["queue_jsonl"], {"subject": s, "hop": kept_hop, "event": outcome})
                if args.debug:
                    _print_enqueue_summary(results)
                _write_queue_snapshot(qdb, paths["queue_json"], args.max_depth)

            mark_done_threadsafe(paths["queue_sqlite"], subject, hop)
            return (subject, hop, None)
        except Exception:
            with open(paths["errors_log"], "a", encoding="utf-8") as ef:
                ef.write(f"[{datetime.datetime.now().isoformat()}] subject={subject}\n{traceback.format_exc()}\n")
            mark_pending_on_error(paths["queue_sqlite"], subject, hop)
            return (subject, hop, "error")

    # ---- OpenAI Batch path (with salvage & queue-level retries) ----
    def _elicitation_openai_batch(subjects_with_hops: List[Tuple[str,int]]):
        if not subjects_with_hops: return 0
        client = _make_openai_client_for_batch(el_cfg)
        subjects = [s for s,_ in subjects_with_hops]
        hops_map = {s:h for s,h in subjects_with_hops}
        _write_batch_requests_jsonl(paths["batch_req_jsonl"], subjects, el_cfg, _build_elicitation_messages, args)
        if args.debug:
            print(f"[batch] wrote request JSONL: {paths['batch_req_jsonl']}")
        with open(paths["batch_req_jsonl"], "rb") as f:
            up = client.files.create(file=f, purpose="batch")
        batch = client.batches.create(
            input_file_id=up.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description":"crawler elicitation"}
        )
        _dbg(f"[batch] created {batch.id}")
        st = batch.status
        delay = 5
        while st in ("created","validating","in_progress","finalizing"):
            time.sleep(delay)
            delay = min(int(delay * 1.5), 60)
            batch = client.batches.retrieve(batch.id)
            st = batch.status
            _dbg(f"[batch] {batch.id} status={st}")
        if st != "completed":
            _dbg(f"[batch] status={st}; reverting claimed items")
            with qdb:
                for s,h in subjects_with_hops:
                    qdb.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=? AND hop=? AND status='working'", (s,h))
            return 0
        out_file_id = batch.output_file_id
        content_bytes = client.files.content(out_file_id).content
        with open(paths["batch_out_jsonl"], "wb") as f:
            f.write(content_bytes)

        accepted_total = 0
        seen_subjects: Set[str] = set()

        with open(paths["batch_out_jsonl"], "r", encoding="utf-8") as f:
            for line in f:
                try:
                    subject, facts, raw_text = _parse_openai_batch_output_line(line, debug=args.debug)
                    if not subject:
                        continue
                    seen_subjects.add(subject)
                    hop = hops_map.get(subject, 0)

                    if not facts and raw_text:
                        salv = _salvage_facts_from_text(raw_text, debug=args.debug)
                        if salv: facts = salv

                    if not facts:
                        current_retries = _get_retries(paths["queue_sqlite"], subject, hop)
                        if current_retries < args.max_retries:
                            if args.debug:
                                print(f"[batch] empty/unparseable for '{subject}', re-queuing (retry {current_retries+1}/{args.max_retries})")
                            _inc_retries_and_pending(paths["queue_sqlite"], subject, hop)
                            continue
                        else:
                            if args.debug:
                                print(f"[batch] empty/unparseable for '{subject}', max retries reached → sink.")
                            write_triples_sink(fdb, [(subject,"__empty__","__empty__",hop, el_cfg.model,args.elicitation_strategy,None,"empty_or_unparseable_output")])
                            mark_done_threadsafe(paths["queue_sqlite"], subject, hop)
                            continue

                    acc, lowconf, _ = _route_facts(args, facts, hop, el_cfg.model)
                    if acc:
                        write_triples_accepted(fdb, acc)
                        with _seen_facts_lock:
                            for s,p,o,_,m,stg,c in acc:
                                key = (s,p,o,hop)
                                if key not in seen_facts:
                                    seen_facts.add(key)
                                    _append_jsonl(paths["facts_jsonl"], {"subject": s,"predicate":p,"object":o,"hop":hop,"model":m,"strategy":stg,"confidence":c})
                        accepted_total += len(acc)
                    if lowconf:
                        for item in lowconf: _append_jsonl(paths["lowconf_jsonl"], item)
                        with _lowconf_lock: lowconf_accum.extend(lowconf)

                    # NER (real time)
                    cand = _filter_ner_candidates([t.get("object") for t in facts if isinstance(t, dict)], subject)
                    next_subjects: List[str] = []
                    i = 0
                    while i < len(cand):
                        chunk = cand[i: i + args.ner_batch_size]
                        ner_messages = get_prompt_messages(args.ner_strategy, "ner",
                            domain=args.domain,
                            variables=dict(phrases_block="\n".join(chunk), root_subject=args.seed, subject_name=subject))
                        ner_schema = NER_SCHEMA_CAL if (args.ner_strategy=="calibrate") else NER_SCHEMA_BASE
                        if args.debug: _print_messages(f"NER for [{subject}] chunk[{i}:{i+args.ner_batch_size}]", ner_messages)
                        try: out = ner_llm(ner_messages, json_schema=ner_schema)
                        except Exception: out = ner_llm(ner_messages)
                        norm = _parse_obj(out)
                        decisions = norm.get("phrases", []) if isinstance(norm.get("phrases"), list) else []
                        if not decisions:
                            decisions = [{"phrase": ph, "is_ne": _maybe_is_ne_heuristic(ph), "confidence": None} for ph in chunk]

                        # >>> force numeric confidence in calibrate, if missing <<<
                        if args.ner_strategy == "calibrate":
                            for d in decisions:
                                if not isinstance(d.get("confidence"), (int, float)):
                                    d["confidence"] = 0.90

                        use_thr = (args.ner_strategy=="calibrate")
                        for d in decisions:
                            phrase = d.get("phrase")
                            is_ne = bool(d.get("is_ne"))
                            conf = d.get("confidence")
                            try: conf = float(conf)
                            except Exception: conf = None
                            is_variant = _is_subject_variant(phrase, subject)
                            if is_variant:
                                is_ne = False; conf = 0.0 if conf is None else min(conf, 0.0)
                            conf_ok = (isinstance(conf,(int,float)) and conf >= args.ner_conf_threshold) if use_thr else True
                            record = {
                                "current_entity": subject, "hop": hop, "phrase": phrase,
                                "is_ne": is_ne, "is_variant": is_variant,
                                "confidence": (float(conf) if isinstance(conf,(int,float)) else None),
                                "ner_conf_threshold": float(args.ner_conf_threshold),
                                "passed_threshold": bool(conf_ok if use_thr else True),
                                "ner_model": ner_cfg.model, "ner_strategy": args.ner_strategy,
                                "domain": args.domain, "root_subject": args.seed, "source": "model_or_fallback"
                            }
                            _append_jsonl(paths["ner_jsonl"], record)
                            if use_thr and not conf_ok:
                                low_item = {**record, "reason":"below_threshold"}
                                _append_jsonl(paths["ner_lowconf_jsonl"], low_item)
                                with _ner_lowconf_lock: ner_lowconf_accum.append(low_item)
                            if is_ne and conf_ok and not is_variant and isinstance(phrase,str):
                                next_subjects.append(phrase)
                        i += args.ner_batch_size

                    if next_subjects:
                        results = procq_enqueue(
                            paths["queue_sqlite"],
                            [(s, hop+1) for s in next_subjects if (args.max_depth==0 or hop+1<=args.max_depth)],
                            leading_articles=PROCQ_LEADING
                        )
                        for s, kept_hop, outcome in results:
                            if outcome in ("inserted","hop_reduced"):
                                _append_jsonl(paths["queue_jsonl"], {"subject": s, "hop": kept_hop, "event": outcome})
                        if args.debug:
                            _print_enqueue_summary(results)
                        _write_queue_snapshot(qdb, paths["queue_json"], args.max_depth)

                    mark_done_threadsafe(paths["queue_sqlite"], subject, hop)

                except Exception:
                    with qdb:
                        qdb.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=? AND hop=? AND status='working'", (subject, hop))
                    with open(paths["errors_log"], "a", encoding="utf-8") as ef:
                        ef.write(f"[{datetime.datetime.now().isoformat()}] batch_line_error\n{traceback.format_exc()}\n")

        # requeue any subject missing from output file entirely
        for s in subjects:
            if s not in seen_subjects:
                _inc_retries_and_pending(paths["queue_sqlite"], s, hops_map.get(s, 0))
        return accepted_total

    # ------------- loop -------------
    while True:
        if args.progress_metrics:
            now = time.perf_counter()
            if now - last_progress_ts >= 2.0:
                d,w,p,t = _counts(qdb, args.max_depth)
                try:
                    cur = qdb.cursor(); cur.execute("SELECT SUM(retries) FROM queue"); retry_sum = cur.fetchone()[0] or 0
                except Exception:
                    retry_sum = 0
                _dbg(f"[progress] done={d} working={w} pending={p} total={t} retries={retry_sum}")
                last_progress_ts = now

        if args.max_subjects and subjects_elicited_total >= args.max_subjects:
            _dbg(f"[stop] max-subjects reached ({subjects_elicited_total})")
            break

        remaining_cap = (args.max_subjects - subjects_elicited_total) if args.max_subjects else None

        if args.openai_batch_mode:
            claim_n = min(args.max_inflight or 1, args.batch_size)
        elif supports_realtime_batch:
            claim_n = args.batch_size
        else:
            claim_n = args.concurrency

        if remaining_cap is not None:
            claim_n = max(1, min(claim_n, remaining_cap))

        batch = _fetch_many_pending(qdb, args.max_depth, max(1, claim_n))
        if not batch:
            d,w,p,t = _counts(qdb, args.max_depth)
            if t == 0: _dbg("[idle] nothing to do.")
            else: _dbg(f"[idle] queue drained: done={d} working={w} pending={p} total={t}")
            break

        # --- OpenAI Batch (offline) ---
        if args.openai_batch_mode:
            _dbg(f"[path=batch] claim {len(batch)} subjects (max_inflight={args.max_inflight}, batch_size={args.batch_size})")
            _ = _elicitation_openai_batch(batch)
            subjects_elicited_total += len(batch)
            continue

        # --- realtime .batch(...) path ---
        if supports_realtime_batch:
            subjects = [s for s,_ in batch]
            _dbg(f"[path=realtime-batch] groupsize={len(subjects)} (batch_size={args.batch_size})")
            messages_list = []
            for s in subjects:
                msgs = _build_elicitation_messages(args, s)
                _ensure_json_keyword_in_msgs(msgs, shape_hint='{"facts":[{"subject":"...","predicate":"...","object":"..."}]}')
                messages_list.append(msgs)
            if args.debug:
                for s,msgs in zip(subjects, messages_list):
                    _print_messages(f"ELICIT (batch-call) for [{s}]", msgs)
            el_schema = ELICIT_SCHEMA_CAL if (args.elicitation_strategy=="calibrate") else ELICIT_SCHEMA_BASE
            try:
                try:
                    resp_list = el_llm.batch(messages_list, json_schema=el_schema, timeout=args.timeout)  # type: ignore
                except TypeError:
                    resp_list = el_llm.batch(messages_list, json_schema=el_schema)  # type: ignore
            except Exception:
                with qdb:
                    for subject, hop in batch:
                        qdb.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=? AND hop=? AND status='working'", (subject, hop))
                _dbg("[warn] realtime batch call failed; reverted claims")
                continue
            if len(resp_list) != len(batch):
                with qdb:
                    for subject, hop in batch:
                        qdb.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=? AND hop=? AND status='working'", (subject, hop))
                _dbg("[warn] batch size mismatch; reverted claims")
                continue

            for (subject, hop), resp in zip(batch, resp_list):
                try:
                    facts, raw_txt = _extract_facts_from_resp(resp, debug=args.debug)
                    if not facts and raw_txt:
                        salv = _salvage_facts_from_text(raw_txt, debug=args.debug)
                        if salv: facts = salv
                    if not facts:
                        write_triples_sink(fdb, [(subject,"__empty__","__empty__",hop, el_cfg.model,args.elicitation_strategy,None,"empty_or_unparseable_output")])

                    acc, lowconf, _ = _route_facts(args, facts, hop, el_cfg.model)
                    if acc:
                        write_triples_accepted(fdb, acc)
                        with _seen_facts_lock:
                            for s,p,o,_,m,st,c in acc:
                                key = (s,p,o,hop)
                                if key not in seen_facts:
                                    seen_facts.add(key)
                                    _append_jsonl(paths["facts_jsonl"], {"subject": s,"predicate":p,"object":o,"hop":hop,"model":m,"strategy":st,"confidence":c})
                    if lowconf:
                        for item in lowconf: _append_jsonl(paths["lowconf_jsonl"], item)
                        with _lowconf_lock: lowconf_accum.extend(lowconf)

                    # NER
                    cand = _filter_ner_candidates([t.get("object") for t in facts if isinstance(t, dict)], subject)
                    next_subjects: List[str] = []
                    i = 0
                    while i < len(cand):
                        chunk = cand[i: i + args.ner_batch_size]
                        ner_messages = get_prompt_messages(args.ner_strategy, "ner",
                            domain=args.domain,
                            variables=dict(phrases_block="\n".join(chunk), root_subject=args.seed, subject_name=subject))
                        ner_schema = NER_SCHEMA_CAL if (args.ner_strategy=="calibrate") else NER_SCHEMA_BASE
                        if args.debug: _print_messages(f"NER for [{subject}] chunk[{i}:{i+args.ner_batch_size}]", ner_messages)
                        try: out = ner_llm(ner_messages, json_schema=ner_schema)
                        except Exception: out = ner_llm(ner_messages)
                        norm = _parse_obj(out)
                        decisions = norm.get("phrases", []) if isinstance(norm.get("phrases"), list) else []
                        if not decisions:
                            decisions = [{"phrase": ph, "is_ne": _maybe_is_ne_heuristic(ph), "confidence": None} for ph in chunk]

                        # >>> force numeric confidence in calibrate, if missing <<<
                        if args.ner_strategy == "calibrate":
                            for d in decisions:
                                if not isinstance(d.get("confidence"), (int, float)):
                                    d["confidence"] = 0.90

                        use_thr = (args.ner_strategy=="calibrate")
                        for d in decisions:
                            phrase = d.get("phrase"); is_ne = bool(d.get("is_ne"))
                            conf = d.get("confidence")
                            try: conf = float(conf)
                            except Exception: conf = None
                            is_variant = _is_subject_variant(phrase, subject)
                            if is_variant:
                                is_ne = False; conf = 0.0 if conf is None else min(conf, 0.0)
                            conf_ok = (isinstance(conf,(int,float)) and conf >= args.ner_conf_threshold) if use_thr else True
                            record = {
                                "current_entity": subject, "hop": hop, "phrase": phrase,
                                "is_ne": is_ne, "is_variant": is_variant,
                                "confidence": (float(conf) if isinstance(conf,(int,float)) else None),
                                "ner_conf_threshold": float(args.ner_conf_threshold),
                                "passed_threshold": bool(conf_ok if use_thr else True),
                                "ner_model": ner_cfg.model, "ner_strategy": args.ner_strategy,
                                "domain": args.domain, "root_subject": args.seed, "source": "model_or_fallback"
                            }
                            _append_jsonl(paths["ner_jsonl"], record)
                            if use_thr and not conf_ok:
                                low_item = {**record, "reason":"below_threshold"}
                                _append_jsonl(paths["ner_lowconf_jsonl"], low_item)
                                with _ner_lowconf_lock: ner_lowconf_accum.append(low_item)
                            if is_ne and conf_ok and not is_variant and isinstance(phrase,str):
                                next_subjects.append(phrase)
                        i += args.ner_batch_size

                    if next_subjects:
                        results = procq_enqueue(
                            paths["queue_sqlite"],
                            [(s, hop+1) for s in next_subjects if (args.max_depth==0 or hop+1<=args.max_depth)],
                            leading_articles=PROCQ_LEADING
                        )
                        for s, kept_hop, outcome in results:
                            if outcome in ("inserted","hop_reduced"):
                                _append_jsonl(paths["queue_jsonl"], {"subject": s, "hop": kept_hop, "event": outcome})
                        if args.debug:
                            _print_enqueue_summary(results)
                        _write_queue_snapshot(qdb, paths["queue_json"], args.max_depth)

                    with qdb:
                        qdb.execute("UPDATE queue SET status='done' WHERE subject=? AND hop=? AND status='working'", (subject, hop))
                    subjects_elicited_total += 1
                    if args.max_subjects and subjects_elicited_total >= args.max_subjects:
                        _dbg(f"[stop] max-subjects reached ({subjects_elicited_total})")
                        break

                except Exception:
                    with qdb:
                        qdb.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=? AND hop=? AND status='working'", (subject, hop))
                    with open(paths["errors_log"], "a", encoding="utf-8") as ef:
                        ef.write(f"[{datetime.datetime.now().isoformat()}] subject={subject}\n{traceback.format_exc()}\n")

        # --- pure concurrency path ---
        else:
            _dbg(f"[path=concurrency] subjects={len(batch)} workers={min(args.concurrency, len(batch))}")
            results = []
            with ThreadPoolExecutor(max_workers=min(args.concurrency, len(batch))) as pool:
                futs = [pool.submit(_elicitation_and_ner, s, h) for (s,h) in batch]
                for fut in as_completed(futs):
                    results.append(fut.result())
            for _s,_h,err in results:
                if err is None:
                    subjects_elicited_total += 1
                    if args.max_subjects and subjects_elicited_total >= args.max_subjects:
                        _dbg(f"[stop] max-subjects reached ({subjects_elicited_total})")
                        break

    # ----- final snapshots -----
    conn = sqlite3.connect(paths["queue_sqlite"])
    cur = conn.cursor()
    cur.execute("SELECT subject, hop, status, retries, created_at FROM queue ORDER BY hop, subject")
    rows = cur.fetchall()
    with open(paths["queue_json"], "w", encoding="utf-8") as f:
        json.dump(
            [{"subject": s, "hop": h, "status": st, "retries": r, "created_at": ts} for (s, h, st, r, ts) in rows],
            f, ensure_ascii=False, indent=2
        )
    conn.close()

    conn = sqlite3.connect(paths["facts_sqlite"])
    cur = conn.cursor()
    cur.execute("SELECT subject, predicate, object, hop, model_name, strategy, confidence FROM triples_accepted ORDER BY subject, predicate, object, hop")
    rows_acc = cur.fetchall()
    cur.execute("SELECT subject, predicate, object, hop, model_name, strategy, confidence, reason FROM triples_sink ORDER BY subject, predicate, object, hop")
    rows_sink = cur.fetchall()
    with open(paths["facts_json"], "w", encoding="utf-8") as f:
        json.dump(
            {
                "accepted": [
                    {"subject": s, "predicate": p, "object": o, "hop": h, "model": m, "strategy": st, "confidence": c}
                    for (s,p,o,h,m,st,c) in rows_acc
                ],
                "sink": [
                    {"subject": s, "predicate": p, "object": o, "hop": h, "model": m, "strategy": st, "confidence": c, "reason": r}
                    for (s,p,o,h,m,st,c,r) in rows_sink
                ],
            },
            f, ensure_ascii=False, indent=2
        )
    conn.close()

    with open(paths["lowconf_json"], "w", encoding="utf-8") as f:
        json.dump(lowconf_accum, f, ensure_ascii=False, indent=2)
    with open(paths["ner_lowconf_json"], "w", encoding="utf-8") as f:
        json.dump(ner_lowconf_accum, f, ensure_ascii=False, indent=2)

    run_meta = {
        "timestamp_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "seed": args.seed, "domain": args.domain,
        "elicitation_strategy": args.elicitation_strategy, "ner_strategy": args.ner_strategy,
        "max_depth": args.max_depth, "max_subjects": args.max_subjects,
        "concurrency": {
            "batch_size": args.batch_size,
            "concurrency": args.concurrency,
            "max_inflight": (args.max_inflight if args.openai_batch_mode else None),
            "timeout_s": args.timeout,
            "openai_batch_mode": bool(args.openai_batch_mode),
        },
        "models": {
            "elicitation": {
                "provider": getattr(el_cfg,"provider","openai"),
                "model": el_cfg.model,
                "use_responses_api": getattr(el_cfg,"use_responses_api", False),
                "temperature": getattr(el_cfg,"temperature", None),
                "top_p": getattr(el_cfg,"top_p", None),
                "top_k": getattr(el_cfg,"top_k", None),
                "max_tokens": getattr(el_cfg,"max_tokens", None),
            },
            "ner": {
                "provider": getattr(ner_cfg,"provider","openai"),
                "model": ner_cfg.model,
                "use_responses_api": getattr(ner_cfg,"use_responses_api", False),
                "temperature": getattr(ner_cfg,"temperature", None),
                "top_p": getattr(ner_cfg,"top_p", None),
                "top_k": getattr(ner_cfg,"top_k", None),
                "max_tokens": getattr(ner_cfg,"max_tokens", None),
            },
        },
        "args_raw": vars(args),
    }
    with open(paths["run_meta_json"], "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    dur = time.perf_counter() - start
    print(f"[done] finished in {dur:.1f}s → {out_dir}")
    for k in ("queue_json","facts_json","facts_jsonl","lowconf_json","lowconf_jsonl","ner_jsonl","ner_lowconf_json","ner_lowconf_jsonl","run_meta_json","errors_log"):
        print(f"[out] {k:18}: {paths[k]}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[interrupt] bye")



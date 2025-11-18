# # crawler_self_rag.py
# from __future__ import annotations

# import argparse, datetime, json, os, re, sqlite3, threading, time, traceback, random
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from typing import Dict, List, Tuple, Set, Optional

# from dotenv import load_dotenv
# load_dotenv()

# # ---------------- repo deps ----------------
# from processing_queue import (
#     init_cache as procq_init_cache,
#     enqueue_subjects_processed as procq_enqueue,
#     DEFAULT_LEADING_ARTICLES as PROCQ_LEADING,
#     get_thread_queue_conn as procq_get_thread_conn,
# )
# from settings import settings
# from llm.factory import make_llm_from_config

# try:
#     from llm.json_utils import best_json as ext_best_json
# except Exception:
#     ext_best_json = None

# # ---------------- tiny utils / IO ----------------
# _jsonl_lock = threading.Lock()
# _seen_facts_lock = threading.Lock()
# _lowconf_lock = threading.Lock()
# _ner_lowconf_lock = threading.Lock()

# def _append_jsonl(path: str, obj: dict):
#     line = json.dumps(obj, ensure_ascii=False) + "\n"
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     with _jsonl_lock:
#         with open(path, "a", encoding="utf-8") as f:
#             f.write(line)

# def _dbg(msg: str):
#     print(msg, flush=True)

# def _print_messages(tag: str, msgs: List[dict], limit: int | None = None):
#     print(f"\n--- {tag} MESSAGES ({len(msgs)}) ---")
#     for i, m in enumerate(msgs, 1):
#         role = (m.get("role") or "").upper()
#         content = m.get("content")
#         if isinstance(content, str) and limit:
#             content = (content[:limit] + "…") if len(content) > limit else content
#         print(f"[{i:02d}] {role}: {content if isinstance(content, str) else content}")
#     print(f"--- END {tag} ---\n")

# def _print_enqueue_summary(results: List[Tuple[str,int,str]]):
#     if not results:
#         print("[enqueue] (no results)")
#         return
#     ins = sum(1 for *_r, out in results if out == "inserted")
#     red = sum(1 for *_r, out in results if out == "hop_reduced")
#     ign = sum(1 for *_r, out in results if out == "ignored")
#     print(f"[enqueue] inserted={ins} hop_reduced={red} ignored={ign}")

# # ---------------- sqlite ----------------
# def open_queue_db(path: str) -> sqlite3.Connection:
#     conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
#     conn.execute("PRAGMA journal_mode=WAL;")
#     conn.execute("PRAGMA synchronous=NORMAL;")
#     conn.execute("PRAGMA busy_timeout=5000;")
#     conn.execute("PRAGMA temp_store=MEMORY;")
#     conn.execute("""
#         CREATE TABLE IF NOT EXISTS queue(
#           subject        TEXT NOT NULL,
#           subject_norm   TEXT NOT NULL,
#           subject_canon  TEXT NOT NULL,
#           hop            INT  NOT NULL DEFAULT 0,
#           status         TEXT NOT NULL DEFAULT 'pending',
#           retries        INT  NOT NULL DEFAULT 0,
#           created_at     DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
#         );
#     """)
#     return conn

# def open_facts_db(path: str) -> sqlite3.Connection:
#     conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
#     conn.execute("PRAGMA journal_mode=WAL;")
#     conn.execute("PRAGMA synchronous=NORMAL;")
#     conn.execute("PRAGMA busy_timeout=5000;")
#     conn.execute("PRAGMA temp_store=MEMORY;")
#     conn.execute("""
#         CREATE TABLE IF NOT EXISTS triples_accepted(
#           subject     TEXT,
#           predicate   TEXT,
#           object      TEXT,
#           hop         INT,
#           model_name  TEXT,
#           strategy    TEXT,
#           confidence  REAL,
#           PRIMARY KEY(subject, predicate, object, hop)
#         );
#     """)
#     conn.execute("""
#         CREATE TABLE IF NOT EXISTS triples_sink(
#           subject     TEXT,
#           predicate   TEXT,
#           object      TEXT,
#           hop         INT,
#           model_name  TEXT,
#           strategy    TEXT,
#           confidence  REAL,
#           reason      TEXT
#         );
#     """)
#     return conn

# def write_triples_accepted(conn: sqlite3.Connection, rows: List[Tuple[str,str,str,int,str,str,Optional[float]]]):
#     if not rows: return
#     with conn:
#         conn.executemany("""
#             INSERT OR IGNORE INTO triples_accepted(subject,predicate,object,hop,model_name,strategy,confidence)
#             VALUES(?,?,?,?,?,?,?)
#         """, rows)

# def write_triples_sink(conn: sqlite3.Connection, rows: List[Tuple[str,str,str,int,str,str,Optional[float],str]]):
#     if not rows: return
#     with conn:
#         conn.executemany("""
#             INSERT INTO triples_sink(subject,predicate,object,hop,model_name,strategy,confidence,reason)
#             VALUES(?,?,?,?,?,?,?,?)
#         """, rows)

# def queue_has_rows(conn: sqlite3.Connection) -> bool:
#     return bool(conn.execute("SELECT COUNT(1) FROM queue").fetchone()[0])

# def reset_working_to_pending(conn: sqlite3.Connection) -> int:
#     with conn:
#         cur = conn.execute("UPDATE queue SET status='pending' WHERE status='working'")
#     return cur.rowcount if hasattr(cur, "rowcount") else 0

# # ---------------- paths / snapshots ----------------
# def _ensure_output_dir(base_dir: Optional[str]) -> str:
#     out = base_dir or os.path.join("runs", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
#     os.makedirs(out, exist_ok=True)
#     return out

# def _build_paths(out_dir: str) -> dict:
#     os.makedirs(out_dir, exist_ok=True)
#     return {
#         "queue_sqlite": os.path.join(out_dir, "queue.sqlite"),
#         "facts_sqlite": os.path.join(out_dir, "facts.sqlite"),
#         "queue_jsonl": os.path.join(out_dir, "queue.jsonl"),
#         "facts_jsonl": os.path.join(out_dir, "facts.jsonl"),
#         "articles_jsonl": os.path.join(out_dir, "articles.jsonl"),
#         "subjects_jsonl": os.path.join(out_dir, "subjects.jsonl"),
#         "queue_json": os.path.join(out_dir, "queue.json"),
#         "facts_json": os.path.join(out_dir, "facts.json"),
#         "subjects_json": os.path.join(out_dir, "subjects.json"),
#         "ner_jsonl": os.path.join(out_dir, "ner_decisions.jsonl"),
#         "lowconf_json": os.path.join(out_dir, "facts_lowconf.json"),
#         "lowconf_jsonl": os.path.join(out_dir, "facts_lowconf.jsonl"),
#         "ner_lowconf_json": os.path.join(out_dir, "ner_lowconf.json"),
#         "ner_lowconf_jsonl": os.path.join(out_dir, "ner_lowconf.jsonl"),
#         "errors_log": os.path.join(out_dir, "errors.log"),
#         "run_meta_json": os.path.join(out_dir, "run_meta.json"),
#     }

# def _write_queue_snapshot(qdb: sqlite3.Connection, snapshot_path: str, max_depth: int):
#     cur = qdb.cursor()
#     if max_depth == 0:
#         cur.execute("SELECT subject, hop, status, retries, created_at FROM queue ORDER BY hop, subject")
#     else:
#         cur.execute("SELECT subject, hop, status, retries, created_at FROM queue WHERE hop<=? ORDER BY hop, subject", (max_depth,))
#     rows = cur.fetchall()
#     with open(snapshot_path, "w", encoding="utf-8") as f:
#         json.dump(
#             [{"subject": s, "hop": h, "status": st, "retries": r, "created_at": ts} for (s, h, st, r, ts) in rows],
#             f, ensure_ascii=False, indent=2
#         )

# # ---------------- per-thread sqlite ----------------
# _thread_local = threading.local()

# def get_thread_queue_conn(db_path: str) -> sqlite3.Connection:
#     return procq_get_thread_conn(db_path)

# def get_thread_facts_conn(db_path: str) -> sqlite3.Connection:
#     key = f"facts_conn__{db_path}"
#     conn = getattr(_thread_local, key, None)
#     if conn is None:
#         conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
#         conn.execute("PRAGMA journal_mode=WAL;")
#         conn.execute("PRAGMA synchronous=NORMAL;")
#         conn.execute("PRAGMA busy_timeout=5000;")
#         conn.execute("PRAGMA temp_store=MEMORY;")
#         setattr(_thread_local, key, conn)
#     return conn

# # ---------------- sqlite retry ----------------
# def _sql_retry(fn, *, tries=8, base_sleep=0.05, jitter=0.05):
#     last = None
#     for _ in range(tries):
#         try:
#             return fn()
#         except sqlite3.OperationalError as e:
#             msg = str(e).lower()
#             if "database is locked" in msg or "database table is locked" in msg:
#                 time.sleep(base_sleep + random.random()*jitter)
#                 base_sleep *= 1.6
#                 last = e
#                 continue
#             raise
#     if last: raise last

# # ---------------- JSON salvage ----------------
# _CODEFENCE_RX = re.compile(r"^```(?:json|JSON)?\s*|\s*```$", re.MULTILINE)

# def _strip_codefences(s: str) -> str:
#     if not isinstance(s, str): return ""
#     return _CODEFENCE_RX.sub("", s).strip()

# def _unwrap_text(resp):
#     if isinstance(resp, str): return resp
#     if isinstance(resp, dict):
#         ch = resp.get("choices")
#         if isinstance(ch, list) and ch:
#             c0 = ch[0] or {}
#             msg = c0.get("message") or {}
#             if isinstance(msg, dict) and isinstance(msg.get("content"), str):
#                 return msg["content"]
#             if isinstance(c0.get("text"), str):
#                 return c0["text"]
#         for k in ("text","output_text","content","message","response","raw","_raw"):
#             v = resp.get(k)
#             if isinstance(v, str): return v
#             if isinstance(v, dict):
#                 for kk in ("text","content","message"):
#                     if isinstance(v.get(kk), str): return v[kk]
#     return ""

# def _find_last_balanced_json_object(s: str) -> str | None:
#     if not isinstance(s, str): return None
#     t = s.strip()
#     if not t: return None
#     try:
#         json.loads(t); return t
#     except Exception:
#         pass
#     t = _strip_codefences(t)
#     start = t.rfind("{")
#     while start != -1:
#         depth = 0; in_str = False; esc = False
#         for i, ch in enumerate(t[start:], start):
#             if in_str:
#                 if esc: esc = False
#                 elif ch == "\\": esc = True
#                 elif ch == '"': in_str = False
#             else:
#                 if ch == '"': in_str = True
#                 elif ch == "{": depth += 1
#                 elif ch == "}":
#                     depth -= 1
#                     if depth == 0:
#                         cand = t[start:i+1]
#                         try:
#                             json.loads(cand); return cand
#                         except Exception: break
#         start = t.rfind("{", 0, start)
#     return None

# def _best_json(text: str):
#     if not isinstance(text, str): return {}
#     if ext_best_json:
#         try:
#             obj = ext_best_json(text)
#             if isinstance(obj, (dict, list)): return obj
#         except Exception:
#             pass
#     t = _strip_codefences(text)
#     try:
#         return json.loads(t)
#     except Exception:
#         pass
#     chunk = _find_last_balanced_json_object(t)
#     if chunk:
#         try:
#             return json.loads(chunk)
#         except Exception:
#             pass
#     return {}

# def _extract_json_block(x) -> dict | list:
#     if isinstance(x, (dict, list)): return x
#     txt = _unwrap_text(x)
#     obj = _best_json(txt)
#     return obj if isinstance(obj, (dict, list)) else {}

# # ---------------- response extraction ----------------
# def _extract_facts_article(resp) -> Tuple[List[dict], dict, str]:
#     txt = _unwrap_text(resp) if not isinstance(resp, dict) else ""
#     obj = _extract_json_block(resp)
#     if isinstance(obj, dict):
#         facts = obj.get("facts") if isinstance(obj.get("facts"), list) else []
#         article = obj.get("article") if isinstance(obj.get("article"), dict) else {}
#         return facts, article, txt
#     return [], {}, txt

# # ---------------- prompts loading (file-based) + footer ----------------
# def _load_prompt_json(path: str) -> dict:
#     with open(path, "r", encoding="utf-8") as f:
#         try:
#             return json.load(f)
#         except Exception as e:
#             raise ValueError(f"Invalid JSON prompt at {path}: {e}")

# _PLACEHOLDERS = {"root_subject","subject_name","conf_threshold","min_sections","max_sections","phrases_block","ner_conf_threshold"}
# _PLACEHOLDER_RX = re.compile(r"\{(" + "|".join(sorted(_PLACEHOLDERS)) + r")\}")

# def _format_prompt(template: str, subs: Dict[str, str | int | float]) -> str:
#     if not isinstance(template, str): return ""
#     def _repl(m: re.Match):
#         k = m.group(1); v = subs.get(k)
#         return str(v) if v is not None else "{" + k + "}"
#     return _PLACEHOLDER_RX.sub(_repl, template)

# def _maybe_append_footer(msgs: List[dict], footer: Optional[str]) -> List[dict]:
#     if not footer: return msgs
#     for m in msgs:
#         if (m.get("role") == "system") and isinstance(m.get("content"), str):
#             m["content"] = (m["content"].rstrip() + "\n\n" + footer)
#             return msgs
#     msgs.insert(0, {"role":"system","content":footer})
#     return msgs

# def build_messages_from_prompt(root: str, domain: str, strategy: str, kind: str, substitutions: Dict[str, str | int | float], footer: Optional[str]=None) -> List[dict]:
#     path = os.path.join(root, domain, strategy, f"{kind}.json")
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Prompt file not found: {path}")
#     data = _load_prompt_json(path)
#     sys = _format_prompt(data.get("system",""), substitutions)
#     usr = _format_prompt(data.get("user",""), substitutions)
#     msgs = [{"role":"system","content":sys},{"role":"user","content":usr}]
#     return _maybe_append_footer(msgs, footer)

# # ---------------- NER filters (classic, generic-safe) ----------------
# _date_rx = re.compile(r"^\d{4}([-/]\d{2}){0,2}$|^(January|February|March|April|May|June|July|August|September|October|November|December)\b", re.I)
# _url_rx  = re.compile(r"^https?://", re.I)

# def _is_date_like(s:str)->bool: return bool(_date_rx.search(s or ""))

# def _is_literal_like(s:str)->bool:
#     s = s or ""
#     if _url_rx.search(s): return True
#     if s.isdigit(): return True
#     return False

# def _titlecase_ratio(s:str)->float:
#     words = [w for w in re.split(r"\s+", (s or "").strip()) if w]
#     if not words: return 0.0
#     caps = sum(1 for w in words if w[:1].isupper())
#     return caps/len(words)

# _variant_rx = re.compile(r"[\(\)\[\]\{\}:–—\-]")
# def _norm(s:str)->str: return re.sub(r"\s+"," ",(s or "")).strip().lower()

# def _is_subject_variant(phrase:str, subject:str)->bool:
#     ps, ss = _norm(phrase), _norm(subject)
#     if not ps or not ss: return False
#     if ps == ss: return True
#     if ps.startswith(ss+" (") or ps.startswith(ss+" -") or ps.startswith(ss+":"): return True
#     if _variant_rx.sub("", ps) == _variant_rx.sub("", ss): return True
#     if ps.startswith(ss) and any(ch in ps[len(ss):len(ss)+3] for ch in "():-—–[]{}"): return True
#     return False

# def _maybe_is_ne_heuristic(phrase:str)->bool:
#     if not isinstance(phrase,str): return False
#     p = phrase.strip()
#     if not p: return False
#     if _is_date_like(p) or _is_literal_like(p): return False
#     if " " not in p and p.islower(): return False
#     if _titlecase_ratio(p) >= 0.6: return True
#     if " " in p and not p.islower(): return True
#     return False

# # _PARTITIVE_PAT = re.compile(r"\b(of|for|about|regarding|during|between)\b", re.I)
# _PARTITIVE_PAT = re.compile(r"^(of|for|about|regarding|during|between)\b", re.I)

# def _is_partitive_like(s: str) -> bool:
#     if not isinstance(s, str): return False
#     t = s.strip()
#     if not t: return False
#     if _PARTITIVE_PAT.search(" " + t + " "):
#         return True
#     return False

# def _filter_ner_candidates(objs: List[str], subject: Optional[str]=None)->List[str]:
#     uniq:Set[str] = set()
#     for o in objs:
#         if not isinstance(o,str): continue
#         o2 = o.strip()
#         if not o2: continue
#         if len(o2.split())>6: continue
#         if subject and _is_subject_variant(o2, subject): continue
#         if _is_date_like(o2) or _is_literal_like(o2): continue
#         uniq.add(o2)
#     return sorted(uniq)

# # ---------------- response schema (best-effort) ----------------
# COMBINED_SCHEMA_BASE = {
#     "type": "object",
#     "additionalProperties": False,
#     "properties": {
#         "facts": {
#             "type": "array",
#             "items": {
#                 "type": "object",
#                 "additionalProperties": False,
#                 "properties": {
#                     "subject": {"type":"string"},
#                     "predicate":{"type":"string"},
#                     "object":{"type":"string"},
#                     "confidence":{"type":"number","minimum":0.0,"maximum":1.0},
#                     "source":{"type":"string"}
#                 },
#                 "required": ["subject","predicate","object","confidence"]
#             }
#         },
#         "article": {
#             "type": "object",
#             "additionalProperties": False,
#             "properties": {
#                 "title": {"type":"string"},
#                 "summary": {"type":"string"},
#                 "sections": {
#                     "type": "array",
#                     "items": {
#                         "type": "object",
#                         "additionalProperties": False,
#                         "properties": {
#                             "heading":{"type":"string"},
#                             "content_md":{"type":"string"},
#                             "confidence":{"type":"number","minimum":0.0,"maximum":1.0}
#                         },
#                         "required": ["heading","content_md"]
#                     }
#                 },
#                 "references": {
#                     "type":"array",
#                     "items":{
#                         "type":"object",
#                         "additionalProperties": False,
#                         "properties": {
#                             "title_or_url":{"type":"string"},
#                             "type":{"type":"string","enum":["url","book","paper","report","database","other"]},
#                             "confidence":{"type":"number","minimum":0.0,"maximum":1.0}
#                         },
#                         "required":["title_or_url","type","confidence"]
#                     }
#                 },
#                 "categories": {"type":"array","items":{"type":"string"}},
#                 "overall_confidence":{"type":"number","minimum":0.0,"maximum":1.0}
#             },
#             "required": ["title","summary","sections","references","categories"]
#         }
#     },
#     "required": ["facts","article"]
# }

# # ---------------- Self-RAG helpers (optional) ----------------
# SELF_RAG_SCHEMA = {
#     "type":"object",
#     "additionalProperties": False,
#     "properties":{
#         "summary":{"type":"string"},
#         "aliases":{"type":"array","items":{"type":"string"}},
#         "salient_facts":{
#             "type":"array",
#             "items":{
#                 "type":"object",
#                 "additionalProperties": False,
#                 "properties":{
#                     "predicate":{"type":"string"},
#                     "object":{"type":"string"},
#                     "confidence":{"type":"number"}
#                 },
#                 "required":["predicate","object"]
#             }
#         }
#     },
#     "required":["summary","salient_facts"]
# }

# def _build_selfrag_messages(subject: str, root_subject: str)->List[dict]:
#     sys = (
#         "You are a concise, factual grounding assistant. Given a subject entity, return STRICT JSON: "
#         '{"summary":"...", "aliases":["..."], "salient_facts":[{"predicate":"...", "object":"...", "confidence":0.0}]}.\n'
#         "Rules: short, verifiable, no speculation; keep 5–12 salient_facts; confidence in [0,1]."
#     )
#     usr = f"Subject: {subject}\nDomain focus (context): {root_subject}\nReturn only JSON; no markdown or prose."
#     return [{"role":"system","content":sys},{"role":"user","content":usr}]

# def _inject_selfrag_into_messages(elicitation_msgs: List[dict], subject: str, sr: dict):
#     summary = (sr.get("summary") or "").strip()
#     aliases = ", ".join(sr.get("aliases") or [])
#     facts = sr.get("salient_facts") or []
#     lines = []
#     for f in facts[:16]:
#         p = (f.get("predicate") or "").strip()
#         o = (f.get("object") or "").strip()
#         c = f.get("confidence")
#         if p and o:
#             if isinstance(c,(int,float)):
#                 lines.append(f"- {subject} — {p} — {o} (c={c:.2f})")
#             else:
#                 lines.append(f"- {subject} — {p} — {o}")
#     ctx = (
#         "CONTEXT (self-RAG grounding; use when uncertain; do not quote directly):\n"
#         f"Summary: {summary}\n"
#         f"Aliases: {aliases}\n"
#         "Salient facts:\n" + ("\n".join(lines) if lines else "(none)")
#     )
#     elicitation_msgs.insert(0, {"role":"system","content":ctx})

# # ---------------- model helpers ----------------
# def _supports_reasoning_controls(cfg) -> bool:
#     return bool(getattr(cfg, "use_responses_api", False) and str(getattr(cfg, "model","")).lower().startswith("gpt-5"))

# def _apply_reasoning_text_overrides(cfg, effort: str | None, verbosity: str | None):
#     if not _supports_reasoning_controls(cfg): return
#     if cfg.extra_inputs is None:
#         cfg.extra_inputs = {}
#     cfg.extra_inputs.setdefault("reasoning", {})
#     cfg.extra_inputs.setdefault("text", {})
#     if effort is not None:    cfg.extra_inputs["reasoning"]["effort"] = effort
#     if verbosity is not None: cfg.extra_inputs["text"]["verbosity"] = verbosity

# def _apply_stage_runtime(which: str, args, cfg):
#     if getattr(cfg, "use_responses_api", False):
#         cfg.temperature = None; cfg.top_p = None; cfg.top_k = None
#         if cfg.extra_inputs is None: cfg.extra_inputs = {}
#         cfg.extra_inputs.setdefault("reasoning", {}); cfg.extra_inputs.setdefault("text", {})
#     else:
#         t  = getattr(args, f"{which}_temperature", None)
#         tp = getattr(args, f"{which}_top_p", None)
#         tk = getattr(args, f"{which}_top_k", None)
#         if t  is not None: cfg.temperature = t
#         if tp is not None: cfg.top_p = tp
#         if tk is not None: cfg.top_k = tk
#     mt = getattr(args, f"{which}_max_tokens", None)
#     if mt is not None: cfg.max_tokens = mt
#     if getattr(cfg, "max_tokens", None) is None: cfg.max_tokens = 2048
#     if hasattr(cfg, "request_timeout"): cfg.request_timeout = args.timeout
#     elif hasattr(cfg, "timeout"):       cfg.timeout = args.timeout
#     _apply_reasoning_text_overrides(cfg, args.reasoning_effort, args.text_verbosity)

# # ---------------- footer builder ----------------
# def _str2bool(s: str) -> bool:
#     if isinstance(s, bool): return s
#     return str(s).strip().lower() in {"1","true","t","yes","y","on"}

# def _build_footer(args) -> str:
#     return (
#         "Footer — quality + quantity hints:\n"
#         f"- Confidence thresholds: facts ≥ {args.conf_threshold:.2f}; NER ≥ {args.ner_conf_threshold:.2f}.\n"
#         f"- Target entities (guidance): major ≈ {args.major_entities_hint}, minor ≈ {args.min_entities_hint}.\n"
#         f"- Writing budget (soft): average sentence ~{args.avg_tokens} tokens; full article ~{args.article_average_tokens} tokens.\n"
#         "- Strictly use the current subject as the entity of record. The article.title MUST equal the current subject exactly. "
#         "Do not generalize or drop qualifiers (e.g., keep 'TV Series' if present)."
#     )

# # ---------------- acceptance / normalization helpers ----------------
# def _enforce_article_subject(article: dict, subject: str) -> dict:
#     if not isinstance(article, dict):
#         return {}
#     art = dict(article)
#     # Force canonical title to the exact subject string
#     art["title"] = subject
#     # Optional nudge in summary/sections: ensure the subject is the referent (no genericized alias)
#     def _fix_text(s: Optional[str]) -> str:
#         if not isinstance(s, str): return ""
#         # If model used a different leading title, gently re-anchor by inserting the subject once
#         return re.sub(r"^\s*The Big Bang Theory\b", subject, s) if subject.lower().startswith("the big bang theory") else s
#     if isinstance(art.get("summary"), str):
#         art["summary"] = _fix_text(art["summary"])
#     if isinstance(art.get("sections"), list):
#         fixed_sections = []
#         for sec in art["sections"]:
#             if not isinstance(sec, dict): continue
#             h = sec.get("heading")
#             c = _fix_text(sec.get("content_md"))
#             fixed_sections.append({"heading": h, "content_md": c, **({k:v for k,v in sec.items() if k not in {"heading","content_md"}})})
#         art["sections"] = fixed_sections
#     return art

# def _article_is_viable(article: dict) -> bool:
#     if not isinstance(article, dict): return False
#     need = ("title","summary","sections","references","categories")
#     return all(k in article for k in need) and isinstance(article.get("sections"), list) and len(article["sections"])>0

# # ---------------- main ----------------
# def main():
#     ap = argparse.ArgumentParser(description="LLMPedia crawler (facts+article) with classic NER expansion and footer hints (Self-RAG optional).")
#     ap.add_argument("--seed", required=True)
#     ap.add_argument("--output-dir", default=None)

#     ap.add_argument("--elicitation-strategy", default="calibrate", choices=["baseline","icl","dont_know","calibrate"])
#     ap.add_argument("--ner-strategy", default="calibrate", choices=["baseline","icl","dont_know","calibrate"])
#     ap.add_argument("--domain", default="topic", choices=["general","topic"])

#     ap.add_argument("--max-depth", type=int, default=settings.MAX_DEPTH)
#     ap.add_argument("--max-subjects", type=int, default=0)
#     ap.add_argument("--ner-batch-size", type=int, default=settings.NER_BATCH_SIZE)
#     ap.add_argument("--max-facts-hint", default=str(settings.MAX_FACTS_HINT))

#     ap.add_argument("--conf-threshold", type=float, default=0.70)
#     ap.add_argument("--ner-conf-threshold", type=float, default=0.85)

#     ap.add_argument("--min-sections", type=int, default=5)
#     ap.add_argument("--max-sections", type=int, default=9)

#     ap.add_argument("--elicit-model-key", default=settings.ELICIT_MODEL_KEY)
#     ap.add_argument("--ner-model-key", default=settings.NER_MODEL_KEY)

#     ap.add_argument("--elicit-temperature", type=float, default=0.6)
#     ap.add_argument("--ner-temperature", type=float, default=0.3)
#     ap.add_argument("--elicit-top-p", type=float, default=None)
#     ap.add_argument("--ner-top-p", type=float, default=None)
#     ap.add_argument("--elicit-top-k", type=int, default=None)
#     ap.add_argument("--ner-top-k", type=int, default=None)
#     ap.add_argument("--elicit-max-tokens", type=int, default=4096)
#     ap.add_argument("--ner-max-tokens", type=int, default=1024)

#     ap.add_argument("--batch-size", type=int, default=1)
#     ap.add_argument("--concurrency", type=int, default=4)
#     ap.add_argument("--timeout", type=float, default=90.0)
#     ap.add_argument("--max-retries", type=int, default=3)

#     # Footer + guidance inputs (same names)
#     ap.add_argument("--footer-mode", type=_str2bool, default=True,
#                     help="Boolean (true/false). If true (default), append quality/quantity footer hints to prompts.")
#     ap.add_argument("--major-entities-hint", type=int, default=50)
#     ap.add_argument("--min-entities-hint", type=int, default=8)
#     ap.add_argument("--avg-tokens", type=int, default=200)
#     ap.add_argument("--article-average-tokens", type=int, default=2000)

#     # ---- Self-RAG knobs (added; no new output files) ----
#     ap.add_argument("--use-selfrag", action="store_true", help="Enable Self-RAG grounding.")
#     ap.add_argument("--selfrag-model-key", default=None, help="Optional different model for Self-RAG; defaults to elicitation model.")
#     ap.add_argument("--selfrag-max-tokens", type=int, default=600)
#     ap.add_argument("--selfrag-temperature", type=float, default=0.1)
#     ap.add_argument("--selfrag-top-p", type=float, default=None)
#     ap.add_argument("--selfrag-top-k", type=int, default=None)
#     ap.add_argument("--emit-selfrag-debug", action="store_true",
#                     help="If set, include Self-RAG prompt and parsed context inside subjects_jsonl entries for inspection.")

#     ap.add_argument("--debug", action="store_true")
#     ap.add_argument("--progress-metrics", dest="progress_metrics", action="store_true", default=True)
#     ap.add_argument("--no-progress-metrics", dest="progress_metrics", action="store_false")

#     ap.add_argument("--resume", action="store_true")
#     ap.add_argument("--reset-working", action="store_true")

#     ap.add_argument("--reasoning-effort", choices=["minimal","low","medium","high"], default=None)
#     ap.add_argument("--text-verbosity", choices=["low","medium","high"], default=None)

#     ap.add_argument("--prompts-root", default="prompts")

#     args = ap.parse_args()

#     out_dir = _ensure_output_dir(args.output_dir)
#     paths = _build_paths(out_dir)
#     _dbg(f"[runner] output_dir: {out_dir}")
#     start_ts = time.perf_counter()

#     qdb = open_queue_db(paths["queue_sqlite"])
#     fdb = open_facts_db(paths["facts_sqlite"])
#     procq_init_cache(qdb)

#     # seed/resume
#     if args.resume:
#         if not queue_has_rows(qdb):
#             for s, kept_hop, outcome in procq_enqueue(paths["queue_sqlite"], [(args.seed, 0)], leading_articles=PROCQ_LEADING):
#                 if outcome == "inserted":
#                     _append_jsonl(paths["queue_jsonl"], {"subject": s, "hop": kept_hop, "event": outcome})
#             _write_queue_snapshot(qdb, paths["queue_json"], args.max_depth)
#         else:
#             if args.reset_working:
#                 n = reset_working_to_pending(qdb)
#                 _dbg(f"[resume] reset {n} working→pending")
#     else:
#         for s, kept_hop, outcome in procq_enqueue(paths["queue_sqlite"], [(args.seed, 0)], leading_articles=PROCQ_LEADING):
#             if outcome == "inserted":
#                 _append_jsonl(paths["queue_jsonl"], {"subject": s, "hop": kept_hop, "event": outcome})
#         _write_queue_snapshot(qdb, paths["queue_json"], args.max_depth)

#     # models
#     el_cfg = settings.MODELS[args.elicit_model_key].model_copy(deep=True)
#     ner_cfg = settings.MODELS[args.ner_model_key].model_copy(deep=True)
#     _apply_stage_runtime("elicit", args, el_cfg)
#     _apply_stage_runtime("ner", args, ner_cfg)
#     el_llm = make_llm_from_config(el_cfg)
#     ner_llm = make_llm_from_config(ner_cfg)

#     # Self-RAG model
#     if args.use_selfrag:
#         if args.selfrag_model_key:
#             sr_cfg = settings.MODELS[args.selfrag_model_key].model_copy(deep=True)  # type: ignore
#         else:
#             sr_cfg = el_cfg.model_copy(deep=True)  # type: ignore
#         if getattr(sr_cfg, "use_responses_api", False):
#             sr_cfg.temperature = None; sr_cfg.top_p = None; sr_cfg.top_k = None
#         else:
#             sr_cfg.temperature = args.selfrag_temperature
#             if args.selfrag_top_p is not None: sr_cfg.top_p = args.selfrag_top_p
#             if args.selfrag_top_k is not None: sr_cfg.top_k = args.selfrag_top_k
#         sr_cfg.max_tokens = args.selfrag_max_tokens
#         if hasattr(sr_cfg,"request_timeout"): sr_cfg.request_timeout = args.timeout
#         elif hasattr(sr_cfg,"timeout"):       sr_cfg.timeout = args.timeout
#         sr_llm = make_llm_from_config(sr_cfg)
#     else:
#         sr_cfg = None
#         sr_llm = None

#     subjects_done = 0
#     last_progress_ts = 0.0

#     footer_text = _build_footer(args) if args.footer_mode else None

#     # ---------- worker ----------
#     def _process_subject(subject: str, hop: int):
#         try:
#             # ===== optional Self-RAG =====
#             selfrag_context = None
#             sr_prompt_dump = None
#             if args.use_selfrag and sr_llm is not None:
#                 sr_msgs = _build_selfrag_messages(subject, args.seed if args.domain=="topic" else subject)
#                 if args.debug: _print_messages(f"SELF-RAG for [{subject}]", sr_msgs)
#                 try:
#                     sr_resp = sr_llm(sr_msgs, json_schema=SELF_RAG_SCHEMA)
#                 except Exception:
#                     sr_resp = sr_llm(sr_msgs)
#                 sr_txt = _unwrap_text(sr_resp)
#                 sr_obj = _extract_json_block(sr_txt) if sr_txt else (sr_resp if isinstance(sr_resp, dict) else {})
#                 if isinstance(sr_obj, dict):
#                     selfrag_context = {
#                         "summary": sr_obj.get("summary") or "",
#                         "aliases": sr_obj.get("aliases") or [],
#                         "salient_facts": sr_obj.get("salient_facts") or []
#                     }
#                 if args.emit_selfrag_debug:
#                     sr_prompt_dump = {"prompt_messages": sr_msgs, "raw_text": sr_txt, "parsed": selfrag_context}

#             # ===== ELICIT =====
#             attempt = 0
#             facts: List[dict] = []
#             article: dict | None = None
#             raw_text = ""

#             while attempt < max(1, args.max_retries):
#                 subs = {
#                     "root_subject": args.seed if args.domain == "topic" else subject,
#                     "subject_name": subject,
#                     "conf_threshold": f"{args.conf_threshold:.2f}",
#                     "min_sections": args.min_sections,
#                     "max_sections": args.max_sections,
#                 }
#                 el_messages = build_messages_from_prompt(
#                     args.prompts_root, args.domain, args.elicitation_strategy, "elicitation", subs, footer=footer_text
#                 )
#                 if selfrag_context:
#                     _inject_selfrag_into_messages(el_messages, subject, selfrag_context)
#                 if args.debug:
#                     _print_messages(f"ELICIT for [{subject}] (try {attempt+1})", el_messages)

#                 try:
#                     resp = el_llm(el_messages, json_schema=COMBINED_SCHEMA_BASE)
#                 except Exception:
#                     resp = el_llm(el_messages)

#                 facts, article_or_null, raw_text = _extract_facts_article(resp)
#                 # allow article to be null per prompt; normalize to dict or None
#                 article = article_or_null if isinstance(article_or_null, dict) else None

#                 if facts:
#                     break
#                 attempt += 1

#             if not facts:
#                 write_triples_sink(get_thread_facts_conn(paths["facts_sqlite"]),
#                     [(subject,"__empty__","__empty__",hop, el_cfg.model,args.elicitation_strategy,None,"empty_or_unparseable_output")]
#                 )
#                 conn = get_thread_queue_conn(paths["queue_sqlite"])
#                 with conn:
#                     conn.execute("UPDATE queue SET status='done' WHERE subject=? AND hop=? AND status='working'", (subject, hop))
#                 return (subject, hop, None)

#             # accept facts (gate by conf_threshold)
#             acc_rows, objs_for_ner, lowconf_rows = [], [], []
#             for t in facts:
#                 s = t.get("subject"); p = t.get("predicate"); o = t.get("object")
#                 c = t.get("confidence")
#                 if not (isinstance(s,str) and isinstance(p,str) and isinstance(o,str)): continue
#                 try: cf = float(c)
#                 except Exception: cf = None
#                 if isinstance(cf,(int,float)) and cf < args.conf_threshold:
#                     lowconf_rows.append({
#                         "subject": s, "predicate": p, "object": o,
#                         "hop": hop, "model": el_cfg.model, "strategy": args.elicitation_strategy,
#                         "confidence": float(cf), "threshold": float(args.conf_threshold)
#                     })
#                 else:
#                     acc_rows.append((s,p,o,hop,el_cfg.model,args.elicitation_strategy, cf))
#                     objs_for_ner.append(o)

#             if acc_rows:
#                 write_triples_accepted(get_thread_facts_conn(paths["facts_sqlite"]), acc_rows)
#                 with _seen_facts_lock:
#                     for s,p,o,_h,m,st,cf in acc_rows:
#                         _append_jsonl(paths["facts_jsonl"], {
#                             "subject": s, "predicate": p, "object": o,
#                             "hop": hop, "model": m, "strategy": st, "confidence": cf
#                         })
#             if lowconf_rows:
#                 for row in lowconf_rows: _append_jsonl(paths["lowconf_jsonl"], row)

#             # enforce article title == subject if we have an article dict
#             if article is not None:
#                 article = _enforce_article_subject(article, subject)

#             # subjects stream (pack Self-RAG debug if requested)
#             subj_record = {
#                 "subject": subject,
#                 "hop": hop,
#                 "model": el_cfg.model,
#                 "strategy": args.elicitation_strategy,
#                 "facts": facts,
#                 "article": article
#             }
#             if args.emit_selfrag_debug and selfrag_context is not None:
#                 subj_record["selfrag"] = {
#                     "model": getattr(sr_cfg, "model", None) if sr_cfg else None,
#                     "context": selfrag_context,
#                     "debug": sr_prompt_dump
#                 }
#             _append_jsonl(paths["subjects_jsonl"], subj_record)

#             # articles stream (unchanged file; article may be None)
#             _append_jsonl(paths["articles_jsonl"], {"subject": subject, "hop": hop, "article": article})

#             # ===== NER over objects =====
#             cand = _filter_ner_candidates(objs_for_ner, subject)
#             next_subjects: List[str] = []
#             i = 0
#             while i < len(cand):
#                 chunk = cand[i: i + args.ner_batch_size]
#                 subs_ner = {
#                     "root_subject": args.seed if args.domain == "topic" else subject,
#                     "subject_name": subject,
#                     "phrases_block": "\n".join(chunk),
#                     "ner_conf_threshold": f"{args.ner_conf_threshold:.2f}",
#                 }
#                 ner_messages = build_messages_from_prompt(
#                     args.prompts_root, args.domain, args.ner_strategy, "ner", subs_ner, footer=footer_text if args.footer_mode else None
#                 )
#                 if args.debug:
#                     _print_messages(f"NER for [{subject}] chunk[{i}:{i+args.ner_batch_size}]", ner_messages)

#                 try:
#                     out = ner_llm(ner_messages)
#                 except Exception:
#                     out = ner_llm(ner_messages)

#                 obj = out if isinstance(out, dict) else _extract_json_block(_unwrap_text(out))
#                 decisions = obj.get("phrases") if isinstance(obj, dict) else []
#                 if not isinstance(decisions, list): decisions = []

#                 if not decisions:
#                     decisions = [{"phrase": ph, "is_ne": _maybe_is_ne_heuristic(ph), "confidence": 0.90, "is_partitive": _is_partitive_like(ph)} for ph in chunk]

#                 for d in decisions:
#                     phrase = d.get("phrase")
#                     is_ne = bool(d.get("is_ne"))
#                     conf = d.get("confidence")
#                     try: conf = float(conf)
#                     except Exception: conf = None

#                     is_variant = _is_subject_variant(phrase, subject)
#                     is_partitive = bool(d.get("is_partitive")) if isinstance(d, dict) and "is_partitive" in d else _is_partitive_like(phrase)

#                     if is_variant or is_partitive:
#                         is_ne = False
#                         conf = 0.0

#                     conf_ok = (isinstance(conf,(int,float)) and conf >= args.ner_conf_threshold)

#                     record = {
#                         "current_entity": subject,
#                         "child_candidate": phrase,
#                         "hop": hop,
#                         "is_ne": bool(is_ne),
#                         "is_variant": bool(is_variant),
#                         "is_partitive": bool(is_partitive),
#                         "confidence": (float(conf) if isinstance(conf,(int,float)) else None),
#                         "ner_conf_threshold": float(args.ner_conf_threshold),
#                         "passed_threshold": bool(conf_ok),
#                         "ner_model": ner_cfg.model,
#                         "ner_strategy": args.ner_strategy,
#                         "domain": args.domain,
#                         "root_subject": args.seed,
#                         "source": "model_or_fallback"
#                     }
#                     _append_jsonl(paths["ner_jsonl"], record)

#                     if (not conf_ok):
#                         low_item = {**record, "reason":"below_threshold"}
#                         _append_jsonl(paths["ner_lowconf_jsonl"], low_item)

#                     if is_ne and conf_ok and not is_variant and not is_partitive and isinstance(phrase,str):
#                         next_subjects.append(phrase)

#                 i += args.ner_batch_size

#             if next_subjects:
#                 results = _sql_retry(lambda: procq_enqueue(
#                     paths["queue_sqlite"],
#                     [(s, hop+1) for s in next_subjects if (args.max_depth==0 or hop+1<=args.max_depth)],
#                     leading_articles=PROCQ_LEADING
#                 ))
#                 for s, kept_hop, outcome in results:
#                     if outcome == "inserted":
#                         _append_jsonl(paths["queue_jsonl"], {"subject": s, "hop": kept_hop, "event": "inserted"})
#                 if args.debug: _print_enqueue_summary(results)
#                 _write_queue_snapshot(qdb, paths["queue_json"], args.max_depth)

#             # mark done
#             conn = get_thread_queue_conn(paths["queue_sqlite"])
#             with conn:
#                 conn.execute("UPDATE queue SET status='done' WHERE subject=? AND hop=? AND status='working'", (subject, hop))
#             return (subject, hop, None)

#         except Exception:
#             with open(paths["errors_log"], "a", encoding="utf-8") as ef:
#                 ef.write(f"[{datetime.datetime.now().isoformat()}] subject={subject}\n{traceback.format_exc()}\n")
#             conn = get_thread_queue_conn(paths["queue_sqlite"])
#             with conn:
#                 conn.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=? AND hop=? AND status='working'", (subject, hop))
#             return (subject, hop, "error")

#     # ---------------- scheduler loop ----------------
#     while True:
#         if args.progress_metrics:
#             now = time.perf_counter()
#             if now - last_progress_ts >= 2.0:
#                 cur = qdb.cursor()
#                 d = cur.execute("SELECT COUNT(1) FROM queue WHERE status='done'").fetchone()[0]
#                 w = cur.execute("SELECT COUNT(1) FROM queue WHERE status='working'").fetchone()[0]
#                 p = cur.execute("SELECT COUNT(1) FROM queue WHERE status='pending'").fetchone()[0]
#                 t = d+w+p
#                 try:
#                     retry_sum = cur.execute("SELECT SUM(retries) FROM queue").fetchone()[0] or 0
#                 except Exception:
#                     retry_sum = 0
#                 _dbg(f"[progress] done={d} working={w} pending={p} total={t} retries={retry_sum}")
#                 last_progress_ts = now

#         if args.max_subjects and subjects_done >= args.max_subjects:
#             _dbg(f"[stop] max-subjects reached ({subjects_done})")
#             break

#         claim_n = args.concurrency
#         remaining_cap = (args.max_subjects - subjects_done) if args.max_subjects else None
#         if remaining_cap is not None:
#             claim_n = max(1, min(claim_n, remaining_cap))

#         cur = qdb.cursor()
#         pending_batch = cur.execute("""
#             UPDATE queue SET status='working'
#             WHERE rowid IN (
#               SELECT rowid FROM queue WHERE status='pending'
#               AND (?=0 OR hop<=?) ORDER BY hop, created_at LIMIT ?
#             )
#             RETURNING subject, hop
#         """, (args.max_depth, args.max_depth, claim_n)).fetchall()

#         if not pending_batch:
#             d = cur.execute("SELECT COUNT(1) FROM queue WHERE status='done'").fetchone()[0]
#             w = cur.execute("SELECT COUNT(1) FROM queue WHERE status='working'").fetchone()[0]
#             p = cur.execute("SELECT COUNT(1) FROM queue WHERE status='pending'").fetchone()[0]
#             t = d+w+p
#             if t == 0: _dbg("[idle] nothing to do.")
#             else: _dbg(f"[idle] queue drained: done={d} working={w} pending={p} total={t}")
#             break

#         _dbg(f"[path=concurrency] subjects={len(pending_batch)} workers={min(args.concurrency, len(pending_batch))}")
#         results = []
#         with ThreadPoolExecutor(max_workers=min(args.concurrency, len(pending_batch))) as pool:
#             futs = [pool.submit(_process_subject, s, h) for (s,h) in pending_batch]
#             for fut in as_completed(futs):
#                 results.append(fut.result())

#         for _s,_h,err in results:
#             if err is None:
#                 subjects_done += 1
#                 if args.max_subjects and subjects_done >= args.max_subjects:
#                     _dbg(f"[stop] max-subjects reached ({subjects_done})")
#                     break

#     # ---------------- final snapshots ----------------
#     conn = sqlite3.connect(paths["queue_sqlite"])
#     cur = conn.cursor()
#     cur.execute("SELECT subject, hop, status, retries, created_at FROM queue ORDER BY hop, subject")
#     rows = cur.fetchall()
#     with open(paths["queue_json"], "w", encoding="utf-8") as f:
#         json.dump(
#             [{"subject": s, "hop": h, "status": st, "retries": r, "created_at": ts} for (s, h, st, r, ts) in rows],
#             f, ensure_ascii=False, indent=2
#         )
#     conn.close()

#     conn = sqlite3.connect(paths["facts_sqlite"])
#     cur = conn.cursor()
#     cur.execute("SELECT subject, predicate, object, hop, model_name, strategy, confidence FROM triples_accepted ORDER BY subject, predicate, object, hop")
#     rows_acc = cur.fetchall()
#     cur.execute("SELECT subject, predicate, object, hop, model_name, strategy, confidence, reason FROM triples_sink ORDER BY subject, predicate, object, hop")
#     rows_sink = cur.fetchall()
#     with open(paths["facts_json"], "w", encoding="utf-8") as f:
#         json.dump(
#             {
#                 "accepted": [
#                     {"subject": s, "predicate": p, "object": o, "hop": h, "model": m, "strategy": st, "confidence": c}
#                     for (s,p,o,h,m,st,c) in rows_acc
#                 ],
#                 "sink": [
#                     {"subject": s, "predicate": p, "object": o, "hop": h, "model": m, "strategy": st, "confidence": c, "reason": r}
#                     for (s,p,o,h,m,st,c,r) in rows_sink
#                 ],
#             },
#             f, ensure_ascii=False, indent=2
#         )
#     conn.close()

#     # subjects aggregate (unchanged)
#     subs = []
#     if os.path.exists(paths["subjects_jsonl"]):
#         with open(paths["subjects_jsonl"], "r", encoding="utf-8") as f:
#             for line in f:
#                 try: subs.append(json.loads(line))
#                 except Exception: pass
#     with open(paths["subjects_json"], "w", encoding="utf-8") as f:
#         json.dump({"subjects": subs}, f, ensure_ascii=False, indent=2)

#     with open(paths["lowconf_json"], "w", encoding="utf-8") as f:
#         json.dump([], f, ensure_ascii=False, indent=2)

#     run_meta = {
#         "timestamp_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
#         "seed": args.seed, "domain": args.domain,
#         "elicitation_strategy": args.elicitation_strategy, "ner_strategy": args.ner_strategy,
#         "max_depth": args.max_depth, "max_subjects": args.max_subjects,
#         "concurrency": {"batch_size": args.batch_size, "concurrency": args.concurrency, "timeout_s": args.timeout},
#         "footer_mode": bool(args.footer_mode),
#         "hints": {
#             "major_entities": int(args.major_entities_hint),
#             "min_entities": int(args.min_entities_hint),
#             "avg_tokens": int(args.avg_tokens),
#             "article_average_tokens": int(args.article_average_tokens),
#         },
#         "models": {
#             "elicitation": {"provider": getattr(el_cfg,"provider",""), "model": el_cfg.model, "use_responses_api": getattr(el_cfg,"use_responses_api", False)},
#             "ner": {"provider": getattr(ner_cfg,"provider",""), "model": ner_cfg.model, "use_responses_api": getattr(ner_cfg,"use_responses_api", False)},
#             "selfrag": ({"provider": getattr(sr_cfg,"provider",""), "model": getattr(sr_cfg,"model",None)} if sr_cfg else None),
#         },
#         "args_raw": vars(args),
#     }
#     with open(paths["run_meta_json"], "w", encoding="utf-8") as f:
#         json.dump(run_meta, f, ensure_ascii=False, indent=2)

#     dur = time.perf_counter() - start_ts
#     print(f"[done] finished in {dur:.1f}s → {out_dir}")
#     for k in ("queue_json","facts_json","facts_jsonl","subjects_json","subjects_jsonl","articles_jsonl",
#               "lowconf_json","lowconf_jsonl","ner_jsonl","ner_lowconf_json","ner_lowconf_jsonl","run_meta_json","errors_log"):
#         if k in paths:
#             print(f"[out] {k:18}: {paths[k]}")

# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         print("\n[interrupt] bye")
# crawler_self_rag.py
from __future__ import annotations

import argparse, datetime, json, os, re, sqlite3, threading, time, traceback, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Set, Optional

from dotenv import load_dotenv
load_dotenv()

# ---------------- repo deps ----------------
from processing_queue import (
    init_cache as procq_init_cache,
    enqueue_subjects_processed as procq_enqueue,
    DEFAULT_LEADING_ARTICLES as PROCQ_LEADING,
    get_thread_queue_conn as procq_get_thread_conn,
)
from settings import settings
from llm.factory import make_llm_from_config

try:
    from llm.json_utils import best_json as ext_best_json
except Exception:
    ext_best_json = None

# ---------------- tiny utils / IO ----------------
_jsonl_lock = threading.Lock()
_seen_facts_lock = threading.Lock()

def _append_jsonl(path: str, obj: dict):
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _jsonl_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)

def _dbg(msg: str):
    print(msg, flush=True)

def _print_messages(tag: str, msgs: List[dict], limit: int | None = 1500):
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

def _atomic_write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

# ---------------- sqlite ----------------
def open_queue_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    try:
        conn.execute("PRAGMA cache_size=-20000;")   # ~20MB
        conn.execute("PRAGMA mmap_size=268435456;") # 256MB
    except Exception:
        pass
    conn.execute("""
        CREATE TABLE IF NOT EXISTS queue(
          subject        TEXT NOT NULL,
          subject_norm   TEXT NOT NULL,
          subject_canon  TEXT NOT NULL,
          hop            INT  NOT NULL DEFAULT 0,
          status         TEXT NOT NULL DEFAULT 'pending',
          retries        INT  NOT NULL DEFAULT 0,
          created_at     DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
    """)
    return conn

def open_facts_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    try:
        conn.execute("PRAGMA cache_size=-20000;")
        conn.execute("PRAGMA mmap_size=268435456;")
    except Exception:
        pass
    conn.execute("""
        CREATE TABLE IF NOT EXISTS triples_accepted(
          subject     TEXT,
          predicate   TEXT,
          object      TEXT,
          hop         INT,
          model_name  TEXT,
          strategy    TEXT,
          confidence  REAL,
          PRIMARY KEY(subject, predicate, object, hop)
        );
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS triples_sink(
          subject     TEXT,
          predicate   TEXT,
          object      TEXT,
          hop         INT,
          model_name  TEXT,
          strategy    TEXT,
          confidence  REAL,
          reason      TEXT
        );
    """)
    return conn

def write_triples_accepted(conn: sqlite3.Connection, rows: List[Tuple[str,str,str,int,str,str,Optional[float]]]):
    if not rows: return
    def _do():
        with conn:
            conn.executemany("""
                INSERT OR IGNORE INTO triples_accepted(subject,predicate,object,hop,model_name,strategy,confidence)
                VALUES(?,?,?,?,?,?,?)
            """, rows)
    _sql_retry(_do)

def write_triples_sink(conn: sqlite3.Connection, rows: List[Tuple[str,str,str,int,str,str,Optional[float],str]]):
    if not rows: return
    def _do():
        with conn:
            conn.executemany("""
                INSERT INTO triples_sink(subject,predicate,object,hop,model_name,strategy,confidence,reason)
                VALUES(?,?,?,?,?,?,?,?)
            """, rows)
    _sql_retry(_do)

def queue_has_rows(conn: sqlite3.Connection) -> bool:
    return bool(conn.execute("SELECT COUNT(1) FROM queue").fetchone()[0])

def reset_working_to_pending(conn: sqlite3.Connection) -> int:
    def _do():
        with conn:
            cur = conn.execute("UPDATE queue SET status='pending' WHERE status='working'")
            return cur.rowcount if hasattr(cur, "rowcount") else 0
    return _sql_retry(_do)

# ---------------- paths / snapshots ----------------
def _ensure_output_dir(base_dir: Optional[str]) -> str:
    out = base_dir or os.path.join("runs", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    return out

def _build_paths(out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    return {
        "queue_sqlite": os.path.join(out_dir, "queue.sqlite"),
        "facts_sqlite": os.path.join(out_dir, "facts.sqlite"),
        "queue_jsonl": os.path.join(out_dir, "queue.jsonl"),
        "facts_jsonl": os.path.join(out_dir, "facts.jsonl"),
        "articles_jsonl": os.path.join(out_dir, "articles.jsonl"),
        "subjects_jsonl": os.path.join(out_dir, "subjects.jsonl"),
        "queue_json": os.path.join(out_dir, "queue.json"),
        "facts_json": os.path.join(out_dir, "facts.json"),
        "subjects_json": os.path.join(out_dir, "subjects.json"),
        "ner_jsonl": os.path.join(out_dir, "ner_decisions.jsonl"),
        "lowconf_json": os.path.join(out_dir, "facts_lowconf.json"),
        "lowconf_jsonl": os.path.join(out_dir, "facts_lowconf.jsonl"),
        "ner_lowconf_json": os.path.join(out_dir, "ner_lowconf.json"),
        "ner_lowconf_jsonl": os.path.join(out_dir, "ner_lowconf.jsonl"),
        "errors_log": os.path.join(out_dir, "errors.log"),
        "run_meta_json": os.path.join(out_dir, "run_meta.json"),
    }

def _write_queue_snapshot(qdb: sqlite3.Connection, snapshot_path: str, max_depth: int):
    cur = qdb.cursor()
    if max_depth == 0:
        cur.execute("SELECT subject, hop, status, retries, created_at FROM queue ORDER BY hop, subject")
    else:
        cur.execute("SELECT subject, hop, status, retries, created_at FROM queue WHERE hop<=? ORDER BY hop, subject", (max_depth,))
    rows = cur.fetchall()
    _atomic_write_json(snapshot_path, [
        {"subject": s, "hop": h, "status": st, "retries": r, "created_at": ts} for (s, h, st, r, ts) in rows
    ])

# ---------------- per-thread sqlite ----------------
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

# ---------------- sqlite retry ----------------
def _sql_retry(fn, *, tries=8, base_sleep=0.05, jitter=0.05):
    last = None
    for _ in range(tries):
        try:
            return fn()
        except sqlite3.OperationalError as e:
            msg = str(e).lower()
            if "database is locked" in msg or "database table is locked" in msg:
                time.sleep(base_sleep + random.random()*jitter)
                base_sleep *= 1.6
                last = e
                continue
            raise
    if last: raise last

# ---------------- JSON salvage ----------------
_CODEFENCE_RX = re.compile(r"^```(?:json|JSON)?\s*|\s*```$", re.MULTILINE)

def _strip_codefences(s: str) -> str:
    if not isinstance(s, str): return ""
    return _CODEFENCE_RX.sub("", s).strip()

def _unwrap_text(resp):
    if isinstance(resp, str): return resp
    if isinstance(resp, dict):
        ch = resp.get("choices")
        if isinstance(ch, list) and ch:
            c0 = ch[0] or {}
            msg = c0.get("message") or {}
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
            if isinstance(c0.get("text"), str):
                return c0["text"]
        for k in ("text","output_text","content","message","response","raw","_raw"):
            v = resp.get(k)
            if isinstance(v, str): return v
            if isinstance(v, dict):
                for kk in ("text","content","message"):
                    if isinstance(v.get(kk), str): return v[kk]
    return ""

def _find_last_balanced_json_object(s: str) -> str | None:
    if not isinstance(s, str): return None
    t = s.strip()
    if not t: return None
    try:
        json.loads(t); return t
    except Exception:
        pass
    t = _strip_codefences(t)
    start = t.rfind("{")
    while start != -1:
        depth = 0; in_str = False; esc = False
        for i, ch in enumerate(t[start:], start):
            if in_str:
                if esc: esc = False
                elif ch == "\\": esc = True
                elif ch == '"': in_str = False
            else:
                if ch == '"': in_str = True
                elif ch == "{": depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        cand = t[start:i+1]
                        try:
                            json.loads(cand); return cand
                        except Exception: break
        start = t.rfind("{", 0, start)
    return None

def _best_json(text: str):
    if not isinstance(text, str): return {}
    if ext_best_json:
        try:
            obj = ext_best_json(text)
            if isinstance(obj, (dict, list)): return obj
        except Exception:
            pass
    t = _strip_codefences(text)
    try:
        return json.loads(t)
    except Exception:
        pass
    chunk = _find_last_balanced_json_object(t)
    if chunk:
        try:
            return json.loads(chunk)
        except Exception:
            pass
    return {}

def _extract_json_block(x) -> dict | list:
    if isinstance(x, (dict, list)): return x
    txt = _unwrap_text(x)
    obj = _best_json(txt)
    return obj if isinstance(obj, (dict, list)) else {}

# ---------------- response extraction ----------------
def _extract_facts_article(resp) -> Tuple[List[dict], dict, str]:
    txt = _unwrap_text(resp) if not isinstance(resp, dict) else ""
    obj = _extract_json_block(resp)
    if isinstance(obj, dict):
        facts = obj.get("facts") if isinstance(obj.get("facts"), list) else []
        article = obj.get("article") if isinstance(obj.get("article"), dict) else {}
        return facts, article, txt
    return [], {}, txt

# ---------------- prompts loading (file-based) + footer ----------------
def _load_prompt_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception as e:
            raise ValueError(f"Invalid JSON prompt at {path}: {e}")

_PLACEHOLDERS = {"root_subject","subject_name","conf_threshold","min_sections","max_sections","phrases_block","ner_conf_threshold"}
_PLACEHOLDER_RX = re.compile(r"\{(" + "|".join(sorted(_PLACEHOLDERS)) + r")\}")

def _format_prompt(template: str, subs: Dict[str, str | int | float]) -> str:
    if not isinstance(template, str): return ""
    def _repl(m: re.Match):
        k = m.group(1); v = subs.get(k)
        return str(v) if v is not None else "{" + k + "}"
    return _PLACEHOLDER_RX.sub(_repl, template)

def _maybe_append_footer(msgs: List[dict], footer: Optional[str], target: str="user") -> List[dict]:
    """Attach footer either to system, to last user, or as a separate user turn; always appended at end."""
    if not footer:
        return msgs
    if target == "system":
        for m in msgs[::-1]:
            if m.get("role") == "system" and isinstance(m.get("content"), str):
                m["content"] = (m["content"].rstrip() + "\n\n" + footer)
                return msgs
        msgs.insert(0, {"role":"system","content":footer})
        return msgs
    elif target == "user":
        for m in msgs[::-1]:
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                m["content"] = (m["content"].rstrip() + "\n\n" + footer)
                return msgs
        msgs.append({"role":"user","content":footer})
        return msgs
    elif target == "separate_user":
        msgs.append({"role":"user","content":footer})
        return msgs
    return msgs

def build_messages_from_prompt(root: str, domain: str, strategy: str, kind: str,
                               substitutions: Dict[str, str | int | float],
                               footer: Optional[str]=None,
                               footer_target: str="user") -> List[dict]:
    """
    File: {root}/{domain}/{strategy}/{kind}.json  with keys {"system": "...", "user": "..."}.
    """
    path = os.path.join(root, domain, strategy, f"{kind}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    data = _load_prompt_json(path)
    sys = _format_prompt(data.get("system",""), substitutions)
    usr = _format_prompt(data.get("user",""), substitutions)
    msgs = [{"role":"system","content":sys},{"role":"user","content":usr}]
    return _maybe_append_footer(msgs, footer, target=footer_target)

# ---------------- NER filters (expanded) ----------------
_date_rx = re.compile(r"^\d{4}([-/]\d{2}){0,2}$|^(January|February|March|April|May|June|July|August|September|October|November|December)\b", re.I)
_url_rx  = re.compile(r"^https?://", re.I)
_DETERMINER_PAT = re.compile(r"^(the|a|an)\s+", re.I)

def _is_date_like(s:str)->bool: return bool(_date_rx.search(s or ""))

def _is_literal_like(s:str)->bool:
    s = s or ""
    if _url_rx.search(s): return True
    if s.isdigit(): return True
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

_PARTITIVE_PAT = re.compile(
    r"""(?:^|\s)(
        of|for|about|regarding|during|between|
        part\s+of|member\s+of|type\s+of|kind\s+of|subset\s+of|
        group\s+of|section\s+of|portion\s+of|category\s+of|
        episode\s+of|chapter\s+of|version\s+of
    )\b""",
    re.I | re.VERBOSE
)

def _is_partitive_like(s: str) -> bool:
    if not isinstance(s, str): return False
    t = s.strip()
    if not t: return False
    return bool(_PARTITIVE_PAT.search(" " + t + " "))

def _maybe_is_ne_heuristic(phrase:str)->bool:
    if not isinstance(phrase,str): return False
    p = phrase.strip()
    if not p: return False
    if _is_date_like(p) or _is_literal_like(p): return False
    if _DETERMINER_PAT.match(p) and not any(ch.isupper() for ch in p.split()[1:2]):
        return False
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

# ---------------- response schema (best-effort) ----------------
COMBINED_SCHEMA_BASE = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "subject": {"type":"string"},
                    "predicate":{"type":"string"},
                    "object":{"type":"string"},
                    "confidence":{"type":"number","minimum":0.0,"maximum":1.0},
                    "source":{"type":"string"}
                },
                "required": ["subject","predicate","object","confidence"]
            }
        },
        "article": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "title": {"type":"string"},
                "summary": {"type":"string"},
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "heading":{"type":"string"},
                            "content_md":{"type":"string"},
                            "confidence":{"type":"number","minimum":0.0,"maximum":1.0}
                        },
                        "required": ["heading","content_md"]
                    }
                },
                "references": {
                    "type":"array",
                    "items":{
                        "type":"object",
                        "additionalProperties": False,
                        "properties": {
                            "title_or_url":{"type":"string"},
                            "type":{"type":"string","enum":["url","book","paper","report","database","other"]},
                            "confidence":{"type":"number","minimum":0.0,"maximum":1.0}
                        },
                        "required":["title_or_url","type","confidence"]
                    }
                },
                "categories": {"type":"array","items":{"type":"string"}},
                "overall_confidence":{"type":"number","minimum":0.0,"maximum":1.0}
            },
            "required": ["title","summary","sections","references","categories"]
        }
    },
    "required": ["facts","article"]
}

# ---------------- Self-RAG helpers ----------------
SELF_RAG_SCHEMA = {
    "type":"object",
    "additionalProperties": False,
    "properties":{
        "summary":{"type":"string"},
        "aliases":{"type":"array","items":{"type":"string"}},
        "salient_facts":{
            "type":"array",
            "items":{
                "type":"object",
                "additionalProperties": False,
                "properties":{
                    "predicate":{"type":"string"},
                    "object":{"type":"string"},
                    "confidence":{"type":"number"}
                },
                "required":["predicate","object"]
            }
        }
    },
    "required":["summary","salient_facts"]
}

def _build_selfrag_messages(subject: str, root_subject: str)->List[dict]:
    sys = (
        "You are a concise, factual grounding assistant. Given a subject entity, return STRICT JSON: "
        '{"summary":"...", "aliases":["..."], "salient_facts":[{"predicate":"...", "object":"...", "confidence":0.0}]}.\n'
        "Rules: short, verifiable, no speculation; keep 5–12 salient_facts; confidence in [0,1]."
    )
    usr = f"Subject: {subject}\nDomain focus (context): {root_subject}\nReturn only JSON; no markdown or prose."
    return [{"role":"system","content":sys},{"role":"user","content":usr}]

def _format_selfrag_context(subject: str, root_subject: str, sr: dict) -> str:
    """Produce an instruction+context block to append to target message."""
    summary = (sr.get("summary") or "").strip()
    aliases = ", ".join(sr.get("aliases") or [])
    facts = sr.get("salient_facts") or []
    lines = []
    for f in facts[:16]:
        p = (f.get("predicate") or "").strip()
        o = (f.get("object") or "").strip()
        c = f.get("confidence")
        if p and o:
            if isinstance(c,(int,float)):
                lines.append(f"- {subject} — {p} — {o} (c={c:.2f})")
            else:
                lines.append(f"- {subject} — {p} — {o}")
    ctx = (
        f"\n\nSELF-RAG CONTEXT — append-only:\n"
        f"You will extract facts for the subject {{subject_name}} = {subject}. "
        f"Here is rag_context to help you. IMPORTANT: every fact you output MUST be strongly related to "
        f"{{root_subject}} = {root_subject}. If a detail is weakly related, omit it.\n"
        f"Summary: {summary or '(none)'}\n"
        f"Aliases: {aliases or '(none)'}\n"
        "Salient facts:\n" + ("\n".join(lines) if lines else "(none)")
    )
    return ctx

def _inject_selfrag(elicitation_msgs: List[dict], subject: str, root_subject: str,
                    sr: dict, target: str="system") -> None:
    """Append the Self-RAG block to the end of the chosen message role."""
    block = _format_selfrag_context(subject, root_subject, sr)
    if target == "system":
        # append to the last system message, else create one at the front
        for m in elicitation_msgs[::-1]:
            if m.get("role") == "system" and isinstance(m.get("content"), str):
                m["content"] = (m["content"].rstrip() + block)
                return
        elicitation_msgs.insert(0, {"role":"system","content":block.lstrip()})
    elif target == "user":
        # append to the last user message, else add a user message at end
        for m in elicitation_msgs[::-1]:
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                m["content"] = (m["content"].rstrip() + block)
                return
        elicitation_msgs.append({"role":"user","content":block.lstrip()})
    else:  # separate_user
        elicitation_msgs.append({"role":"user","content":block.lstrip()})

# ---------------- model helpers ----------------
def _supports_reasoning_controls(cfg) -> bool:
    return bool(getattr(cfg, "use_responses_api", False) and str(getattr(cfg, "model","")).lower().startswith("gpt-5"))

def _apply_reasoning_text_overrides(cfg, effort: str | None, verbosity: str | None):
    if not _supports_reasoning_controls(cfg): return
    if cfg.extra_inputs is None:
        cfg.extra_inputs = {}
    cfg.extra_inputs.setdefault("reasoning", {})
    cfg.extra_inputs.setdefault("text", {})
    if effort is not None:    cfg.extra_inputs["reasoning"]["effort"] = effort
    if verbosity is not None: cfg.extra_inputs["text"]["verbosity"] = verbosity

def _apply_stage_runtime(which: str, args, cfg):
    if getattr(cfg, "use_responses_api", False):
        cfg.temperature = None; cfg.top_p = None; cfg.top_k = None
        if cfg.extra_inputs is None: cfg.extra_inputs = {}
        cfg.extra_inputs.setdefault("reasoning", {}); cfg.extra_inputs.setdefault("text", {})
    else:
        t  = getattr(args, f"{which}_temperature", None)
        tp = getattr(args, f"{which}_top_p", None)
        tk = getattr(args, f"{which}_top_k", None)
        if t  is not None: cfg.temperature = t
        if tp is not None: cfg.top_p = tp
        if tk is not None: cfg.top_k = tk

    mt = getattr(args, f"{which}_max_tokens", None)
    if mt is not None: cfg.max_tokens = mt
    if getattr(cfg, "max_tokens", None) is None: cfg.max_tokens = 2048

    stage_to = getattr(args, f"{which}_request_timeout", None)
    req_to = stage_to if stage_to is not None else args.timeout
    if hasattr(cfg, "request_timeout"): cfg.request_timeout = req_to
    elif hasattr(cfg, "timeout"):       cfg.timeout = req_to

    _apply_reasoning_text_overrides(cfg, args.reasoning_effort, args.text_verbosity)

# ---------------- LLM call retry helpers ----------------
try:
    from openai import APIConnectionError, APITimeoutError, RateLimitError, APIStatusError
except Exception:  # SDK shape differences
    class _E(Exception): pass
    APIConnectionError = APITimeoutError = RateLimitError = APIStatusError = _E

def _llm_retry(fn, *, tries=6, base_sleep=0.5, jitter=0.3):
    for i in range(tries):
        try:
            return fn()
        except (APIConnectionError, APITimeoutError):
            if i == tries - 1: raise
            time.sleep(base_sleep * (2 ** i) + random.random() * jitter)
        except APIStatusError as e:
            code = getattr(e, "status_code", None)
            if code in (429, 500, 502, 503, 504) and i < tries - 1:
                time.sleep(base_sleep * (2 ** i) + random.random() * jitter)
                continue
            raise
        except RateLimitError:
            if i == tries - 1: raise
            time.sleep(base_sleep * (2 ** i) + random.random() * jitter)

# ---------------- footer builder ----------------
def _str2bool(s: str) -> bool:
    if isinstance(s, bool): return s
    return str(s).strip().lower() in {"1","true","t","yes","y","on"}

def _build_footer(args) -> str:
    return (
        "Footer — quality + quantity hints:\n"
        f"- Confidence thresholds: facts ≥ {args.conf_threshold:.2f}; NER ≥ {args.ner_conf_threshold:.2f}.\n"
        f"- Target entities (guidance): major ≈ {args.major_entities_hint}, minor ≈ {args.min_entities_hint}.\n"
        f"- Writing budget (soft): average sentence ~{args.avg_tokens} tokens; full article ~{args.article_average_tokens} tokens.\n"
        "- Strictly use the current subject as the entity of record. The article.title MUST equal the current subject exactly. "
        "Do not generalize or drop qualifiers."
    )

# ---------------- acceptance / normalization helpers ----------------
def _enforce_article_subject(article: dict, subject: str) -> dict:
    if not isinstance(article, dict):
        return {}
    art = dict(article)
    art["title"] = subject
    return art

def _article_is_viable(article: dict) -> bool:
    if not isinstance(article, dict): return False
    need = ("title","summary","sections","references","categories")
    return all(k in article for k in need) and isinstance(article.get("sections"), list) and len(article["sections"])>0

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser(description="LLMPedia crawler (facts+article) with Self-RAG context injection (hardened).")
    ap.add_argument("--seed", required=True)
    ap.add_argument("--output-dir", default=None)

    ap.add_argument("--elicitation-strategy", default="calibrate", choices=["baseline","icl","dont_know","calibrate"])
    ap.add_argument("--ner-strategy", default="calibrate", choices=["baseline","icl","dont_know","calibrate"])
    ap.add_argument("--domain", default="topic", choices=["general","topic"])

    ap.add_argument("--max-depth", type=int, default=settings.MAX_DEPTH)
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--ner-batch-size", type=int, default=settings.NER_BATCH_SIZE)
    ap.add_argument("--max-facts-hint", default=str(settings.MAX_FACTS_HINT))

    ap.add_argument("--conf-threshold", type=float, default=0.90)
    ap.add_argument("--ner-conf-threshold", type=float, default=0.90)

    ap.add_argument("--min-sections", type=int, default=5)
    ap.add_argument("--max-sections", type=int, default=9)

    ap.add_argument("--elicit-model-key", default=settings.ELICIT_MODEL_KEY)
    ap.add_argument("--ner-model-key", default=settings.NER_MODEL_KEY)

    ap.add_argument("--elicit-temperature", type=float, default=0.6)
    ap.add_argument("--ner-temperature", type=float, default=0.3)
    ap.add_argument("--elicit-top-p", type=float, default=None)
    ap.add_argument("--ner-top-p", type=float, default=None)
    ap.add_argument("--elicit-top-k", type=int, default=None)
    ap.add_argument("--ner-top-k", type=int, default=None)
    ap.add_argument("--elicit-max-tokens", type=int, default=4096)
    ap.add_argument("--ner-max-tokens", type=int, default=1024)

    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=4)

    # Footer controls
    ap.add_argument("--footer-mode", type=_str2bool, default=True)
    ap.add_argument("--footer-target", choices=["system","user","separate_user"], default="user")

    ap.add_argument("--major-entities-hint", type=int, default=50)
    ap.add_argument("--min-entities-hint", type=int, default=8)
    ap.add_argument("--avg-tokens", type=int, default=200)
    ap.add_argument("--article-average-tokens", type=int, default=2000)

    # Self-RAG controls
    ap.add_argument("--use-selfrag", action="store_true", help="Enable Self-RAG grounding.")
    ap.add_argument("--selfrag-model-key", default=None, help="Optional different model key for Self-RAG.")
    ap.add_argument("--selfrag-max-tokens", type=int, default=600)
    ap.add_argument("--selfrag-temperature", type=float, default=0.1)
    ap.add_argument("--selfrag-top-p", type=float, default=None)
    ap.add_argument("--selfrag-top-k", type=int, default=None)
    ap.add_argument("--selfrag-target", choices=["system","user","separate_user"], default="system",
                    help="Where to append the Self-RAG block (appended at end).")
    ap.add_argument("--emit-selfrag-debug", action="store_true")

    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--progress-metrics", dest="progress_metrics", action="store_true", default=True)
    ap.add_argument("--no-progress-metrics", dest="progress_metrics", action="store_false")

    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reset-working", action="store_true")

    ap.add_argument("--reasoning-effort", choices=["minimal","low","medium","high"], default=None)
    ap.add_argument("--text-verbosity", choices=["low","medium","high"], default=None)

    ap.add_argument("--prompts-root", default="prompts")

    args = ap.parse_args()

    _dbg(f"[cfg] thresholds: facts ≥ {args.conf_threshold:.2f}, NER ≥ {args.ner_conf_threshold:.2f}")

    out_dir = _ensure_output_dir(args.output_dir)
    paths = _build_paths(out_dir)
    _dbg(f"[runner] output_dir: {out_dir}")
    _dbg(f"[env] OPENAI_BASE_URL={os.getenv('OPENAI_BASE_URL')!r} HTTP_PROXY={os.getenv('HTTP_PROXY')!r} HTTPS_PROXY={os.getenv('HTTPS_PROXY')!r}")
    start_ts = time.perf_counter()

    qdb = open_queue_db(paths["queue_sqlite"])
    fdb = open_facts_db(paths["facts_sqlite"])
    procq_init_cache(qdb)

    # seed/resume
    if args.resume:
        if not queue_has_rows(qdb):
            _dbg("[resume] queue is empty → enqueueing seed fresh")
            for s, kept_hop, outcome in procq_enqueue(paths["queue_sqlite"], [(args.seed, 0)], leading_articles=PROCQ_LEADING):
                if outcome == "inserted":
                    _append_jsonl(paths["queue_jsonl"], {"subject": s, "hop": kept_hop, "event": outcome})
            _write_queue_snapshot(qdb, paths["queue_json"], args.max_depth)
        else:
            if args.reset_working:
                n = reset_working_to_pending(qdb)
                _dbg(f"[resume] reset {n} working→pending")
            else:
                _dbg("[resume] queue present, continuing without reset")
    else:
        _dbg("[start] fresh run → enqueueing seed")
        for s, kept_hop, outcome in procq_enqueue(paths["queue_sqlite"], [(args.seed, 0)], leading_articles=PROCQ_LEADING):
            if outcome == "inserted":
                _append_jsonl(paths["queue_jsonl"], {"subject": s, "hop": kept_hop, "event": outcome})
        _write_queue_snapshot(qdb, paths["queue_json"], args.max_depth)

    # models
    el_cfg = settings.MODELS[args.elicit_model_key].model_copy(deep=True)
    ner_cfg = settings.MODELS[args.ner_model_key].model_copy(deep=True)
    _apply_stage_runtime("elicit", args, el_cfg)
    _apply_stage_runtime("ner", args, ner_cfg)
    el_llm = make_llm_from_config(el_cfg)
    ner_llm = make_llm_from_config(ner_cfg)

    # Self-RAG model
    if args.use_selfrag:
        if args.selfrag_model_key:
            sr_cfg = settings.MODELS[args.selfrag_model_key].model_copy(deep=True)  # type: ignore
        else:
            sr_cfg = el_cfg.model_copy(deep=True)  # type: ignore
        if getattr(sr_cfg, "use_responses_api", False):
            sr_cfg.temperature = None; sr_cfg.top_p = None; sr_cfg.top_k = None
        else:
            sr_cfg.temperature = args.selfrag_temperature
            if args.selfrag_top_p is not None: sr_cfg.top_p = args.selfrag_top_p
            if args.selfrag_top_k is not None: sr_cfg.top_k = args.selfrag_top_k
        sr_cfg.max_tokens = args.selfrag_max_tokens
        if hasattr(sr_cfg,"request_timeout"): sr_cfg.request_timeout = args.timeout
        elif hasattr(sr_cfg,"timeout"):       sr_cfg.timeout = args.timeout
        sr_llm = make_llm_from_config(sr_cfg)
    else:
        sr_cfg = None
        sr_llm = None

    subjects_done = 0
    last_progress_ts = 0.0

    footer_text = _build_footer(args) if args.footer_mode else None

    # ---------- worker ----------
    def _process_subject(subject: str, hop: int):
        try:
            # ===== optional Self-RAG =====
            selfrag_context = None
            sr_prompt_dump = None
            if args.use_selfrag and sr_llm is not None:
                sr_msgs = _build_selfrag_messages(subject, args.seed if args.domain=="topic" else subject)
                if args.debug: _print_messages(f"SELF-RAG for [{subject}]", sr_msgs)
                try:
                    sr_resp = _llm_retry(lambda: sr_llm(sr_msgs, json_schema=SELF_RAG_SCHEMA))
                except Exception:
                    sr_resp = _llm_retry(lambda: sr_llm(sr_msgs))
                sr_txt = _unwrap_text(sr_resp)
                sr_obj = _extract_json_block(sr_txt) if sr_txt else (sr_resp if isinstance(sr_resp, dict) else {})
                if isinstance(sr_obj, dict):
                    selfrag_context = {
                        "summary": sr_obj.get("summary") or "",
                        "aliases": sr_obj.get("aliases") or [],
                        "salient_facts": sr_obj.get("salient_facts") or []
                    }
                if args.emit_selfrag_debug:
                    sr_prompt_dump = {"prompt_messages": sr_msgs, "raw_text": sr_txt, "parsed": selfrag_context}

            # ===== ELICIT =====
            attempt = 0
            facts: List[dict] = []
            article: dict | None = None
            raw_text = ""

            while attempt < max(1, args.max_retries):
                subs = {
                    "root_subject": args.seed if args.domain == "topic" else subject,
                    "subject_name": subject,
                    "conf_threshold": f"{args.conf_threshold:.2f}",
                    "min_sections": args.min_sections,
                    "max_sections": args.max_sections,
                }
                el_messages = build_messages_from_prompt(
                    args.prompts_root, args.domain, args.elicitation_strategy, "elicitation", subs,
                    footer=footer_text, footer_target=args.footer_target
                )

                # Inject Self-RAG block at configured target (appended at end)
                if selfrag_context:
                    _inject_selfrag(el_messages, subject, subs["root_subject"], selfrag_context, target=args.selfrag_target)

                if args.debug:
                    _print_messages(f"ELICIT for [{subject}] (try {attempt+1})", el_messages)

                try:
                    resp = _llm_retry(lambda: el_llm(el_messages, json_schema=COMBINED_SCHEMA_BASE))
                except Exception:
                    resp = _llm_retry(lambda: el_llm(el_messages))

                facts, article_or_null, raw_text = _extract_facts_article(resp)
                article = article_or_null if isinstance(article_or_null, dict) else None

                if facts:
                    break
                attempt += 1

            if not facts:
                write_triples_sink(get_thread_facts_conn(paths["facts_sqlite"]),
                    [(subject,"__empty__","__empty__",hop, el_cfg.model,args.elicitation_strategy,None,"empty_or_unparseable_output")]
                )
                conn = get_thread_queue_conn(paths["queue_sqlite"])
                def _done():
                    with conn:
                        conn.execute("UPDATE queue SET status='done' WHERE subject=? AND hop=? AND status='working'", (subject, hop))
                _sql_retry(_done)
                return (subject, hop, None)

            # accept facts (gate by conf_threshold)
            acc_rows, objs_for_ner, lowconf_rows = [], [], []
            for t in facts:
                s = t.get("subject"); p = t.get("predicate"); o = t.get("object")
                c = t.get("confidence")
                if not (isinstance(s,str) and isinstance(p,str) and isinstance(o,str)): continue
                try: cf = float(c)
                except Exception: cf = None
                if isinstance(cf,(int,float)) and cf < args.conf_threshold:
                    _append_jsonl(paths["lowconf_jsonl"], {
                        "subject": s, "predicate": p, "object": o,
                        "hop": hop, "model": el_cfg.model, "strategy": args.elicitation_strategy,
                        "confidence": float(cf), "threshold": float(args.conf_threshold)
                    })
                else:
                    acc_rows.append((s,p,o,hop,el_cfg.model,args.elicitation_strategy, cf))
                    objs_for_ner.append(o)

            if acc_rows:
                write_triples_accepted(get_thread_facts_conn(paths["facts_sqlite"]), acc_rows)
                with _seen_facts_lock:
                    for s,p,o,_h,m,st,cf in acc_rows:
                        _append_jsonl(paths["facts_jsonl"], {
                            "subject": s, "predicate": p, "object": o,
                            "hop": hop, "model": m, "strategy": st, "confidence": cf
                        })

            # enforce article title == subject if we have an article dict
            if article is not None:
                article = _enforce_article_subject(article, subject)

            # subjects stream (+ optional Self-RAG debug)
            subj_record = {
                "subject": subject,
                "hop": hop,
                "model": el_cfg.model,
                "strategy": args.elicitation_strategy,
                "facts": facts,
                "article": article
            }
            if args.emit_selfrag_debug and selfrag_context is not None:
                subj_record["selfrag"] = {
                    "model": getattr(sr_cfg, "model", None) if sr_cfg else None,
                    "context": selfrag_context,
                    "target": args.selfrag_target
                }
            _append_jsonl(paths["subjects_jsonl"], subj_record)

            # articles stream
            _append_jsonl(paths["articles_jsonl"], {"subject": subject, "hop": hop, "article": article})

            # ===== NER over objects =====
            cand = _filter_ner_candidates(objs_for_ner, subject)
            next_subjects: List[str] = []
            i = 0
            while i < len(cand):
                chunk = cand[i: i + args.ner_batch_size]
                subs_ner = {
                    "root_subject": args.seed if args.domain == "topic" else subject,
                    "subject_name": subject,
                    "phrases_block": "\n".join(chunk),
                    "ner_conf_threshold": f"{args.ner_conf_threshold:.2f}",
                }
                ner_messages = build_messages_from_prompt(
                    args.prompts_root, args.domain, args.ner_strategy, "ner", subs_ner,
                    footer=None if args.footer_mode else None, footer_target=args.footer_target
                )
                if args.debug:
                    _print_messages(f"NER for [{subject}] chunk[{i}:{i+args.ner_batch_size}]", ner_messages)

                try:
                    out = _llm_retry(lambda: ner_llm(ner_messages))
                except Exception:
                    out = _llm_retry(lambda: ner_llm(ner_messages))

                obj = out if isinstance(out, dict) else _extract_json_block(_unwrap_text(out))
                decisions = obj.get("phrases") if isinstance(obj, dict) else []
                if not isinstance(decisions, list): decisions = []

                if not decisions:
                    decisions = [{"phrase": ph, "is_ne": _maybe_is_ne_heuristic(ph), "confidence": 0.90, "is_partitive": _is_partitive_like(ph)} for ph in chunk]

                for d in decisions:
                    phrase = d.get("phrase")
                    is_ne = bool(d.get("is_ne"))
                    conf = d.get("confidence")
                    try: conf = float(conf)
                    except Exception: conf = None

                    is_variant = _is_subject_variant(phrase, subject)
                    is_partitive = bool(d.get("is_partitive")) if isinstance(d, dict) and "is_partitive" in d else _is_partitive_like(phrase)

                    # Normalize: drop variant/partitive
                    if is_variant or is_partitive:
                        is_ne = False
                        conf = 0.0

                    conf_ok = (isinstance(conf,(int,float)) and conf >= args.ner_conf_threshold)

                    record = {
                        "current_entity": subject,
                        "child_candidate": phrase,
                        "hop": hop,
                        "is_ne": bool(is_ne),
                        "is_variant": bool(is_variant),
                        "is_partitive": bool(is_partitive),
                        "confidence": (float(conf) if isinstance(conf,(int,float)) else None),
                        "ner_conf_threshold": float(args.ner_conf_threshold),
                        "passed_threshold": bool(conf_ok),
                        "ner_model": ner_cfg.model,
                        "ner_strategy": args.ner_strategy,
                        "domain": args.domain,
                        "root_subject": args.seed,
                        "source": "model_or_fallback"
                    }
                    _append_jsonl(paths["ner_jsonl"], record)

                    if not conf_ok:
                        _append_jsonl(paths["ner_lowconf_jsonl"], {**record, "reason":"below_threshold"})

                    if is_ne and conf_ok and not is_variant and not is_partitive and isinstance(phrase,str):
                        next_subjects.append(phrase)

                i += args.ner_batch_size

            if next_subjects:
                results = _sql_retry(lambda: procq_enqueue(
                    paths["queue_sqlite"],
                    [(s, hop+1) for s in next_subjects if (args.max_depth==0 or hop+1<=args.max_depth)],
                    leading_articles=PROCQ_LEADING
                ))
                for s, kept_hop, outcome in results:
                    if outcome == "inserted":
                        _append_jsonl(paths["queue_jsonl"], {"subject": s, "hop": kept_hop, "event": "inserted"})
                if args.debug: _print_enqueue_summary(results)
                _write_queue_snapshot(qdb, paths["queue_json"], args.max_depth)

            # mark done
            conn = get_thread_queue_conn(paths["queue_sqlite"])
            def _done2():
                with conn:
                    conn.execute("UPDATE queue SET status='done' WHERE subject=? AND hop=? AND status='working'", (subject, hop))
            _sql_retry(_done2)
            return (subject, hop, None)

        except Exception:
            with open(paths["errors_log"], "a", encoding="utf-8") as ef:
                ef.write(f"[{datetime.datetime.now().isoformat()}] subject={subject}\n{traceback.format_exc()}\n")
            conn = get_thread_queue_conn(paths["queue_sqlite"])
            def _pend():
                with conn:
                    conn.execute("UPDATE queue SET status='pending', retries=retries+1 WHERE subject=? AND hop=? AND status='working'", (subject, hop))
            _sql_retry(_pend)
            return (subject, hop, "error")

    # ---------------- scheduler loop ----------------
    while True:
        if args.progress_metrics:
            now = time.perf_counter()
            if now - last_progress_ts >= 2.0:
                cur = qdb.cursor()
                d = cur.execute("SELECT COUNT(1) FROM queue WHERE status='done'").fetchone()[0]
                w = cur.execute("SELECT COUNT(1) FROM queue WHERE status='working'").fetchone()[0]
                p = cur.execute("SELECT COUNT(1) FROM queue WHERE status='pending'").fetchone()[0]
                t = d+w+p
                try:
                    retry_sum = cur.execute("SELECT SUM(retries) FROM queue").fetchone()[0] or 0
                except Exception:
                    retry_sum = 0
                elapsed = now - start_ts
                thr = (d/elapsed) if elapsed>0 else 0.0
                _dbg(f"[progress] done={d} working={w} pending={p} total={t} retries={retry_sum} thr={thr:.2f} subj/s")
                last_progress_ts = now

        if args.max_subjects and subjects_done >= args.max_subjects:
            _dbg(f"[stop] max-subjects reached ({subjects_done})")
            break

        claim_n = args.concurrency
        remaining_cap = (args.max_subjects - subjects_done) if args.max_subjects else None
        if remaining_cap is not None:
            claim_n = max(1, min(claim_n, remaining_cap))

        def _claim():
            return qdb.execute("""
                UPDATE queue SET status='working'
                WHERE rowid IN (
                  SELECT rowid FROM queue WHERE status='pending'
                  AND (?=0 OR hop<=?) ORDER BY hop, created_at LIMIT ?
                )
                RETURNING subject, hop
            """, (args.max_depth, args.max_depth, claim_n)).fetchall()

        pending_batch = _sql_retry(_claim)

        if not pending_batch:
            cur = qdb.cursor()
            d = cur.execute("SELECT COUNT(1) FROM queue WHERE status='done'").fetchone()[0]
            w = cur.execute("SELECT COUNT(1) FROM queue WHERE status='working'").fetchone()[0]
            p = cur.execute("SELECT COUNT(1) FROM queue WHERE status='pending'").fetchone()[0]
            t = d+w+p
            if t == 0: _dbg("[idle] nothing to do.")
            else: _dbg(f"[idle] queue drained: done={d} working={w} pending={p} total={t}")
            break

        _dbg(f"[path=concurrency] subjects={len(pending_batch)} workers={min(args.concurrency, len(pending_batch))}")
        results = []
        with ThreadPoolExecutor(max_workers=min(args.concurrency, len(pending_batch))) as pool:
            futs = [pool.submit(_process_subject, s, h) for (s,h) in pending_batch]
            for fut in as_completed(futs):
                results.append(fut.result())

        for _s,_h,err in results:
            if err is None:
                subjects_done += 1
                if args.max_subjects and subjects_done >= args.max_subjects:
                    _dbg(f"[stop] max-subjects reached ({subjects_done})")
                    break

    # ---------------- final snapshots ----------------
    conn = sqlite3.connect(paths["queue_sqlite"])
    cur = conn.cursor()
    cur.execute("SELECT subject, hop, status, retries, created_at FROM queue ORDER BY hop, subject")
    rows = cur.fetchall()
    _atomic_write_json(paths["queue_json"], [
        {"subject": s, "hop": h, "status": st, "retries": r, "created_at": ts} for (s, h, st, r, ts) in rows
    ])
    conn.close()

    conn = sqlite3.connect(paths["facts_sqlite"])
    cur = conn.cursor()
    cur.execute("SELECT subject, predicate, object, hop, model_name, strategy, confidence FROM triples_accepted ORDER BY subject, predicate, object, hop")
    rows_acc = cur.fetchall()
    cur.execute("SELECT subject, predicate, object, hop, model_name, strategy, confidence, reason FROM triples_sink ORDER BY subject, predicate, object, hop")
    rows_sink = cur.fetchall()
    _atomic_write_json(paths["facts_json"],
        {
            "accepted": [
                {"subject": s, "predicate": p, "object": o, "hop": h, "model": m, "strategy": st, "confidence": c}
                for (s,p,o,h,m,st,c) in rows_acc
            ],
            "sink": [
                {"subject": s, "predicate": p, "object": o, "hop": h, "model": m, "strategy": st, "confidence": c, "reason": r}
                for (s,p,o,h,m,st,c,r) in rows_sink
            ],
        }
    )
    conn.close()

    # subjects aggregate — stream (low RAM)
    if os.path.exists(paths["subjects_jsonl"]):
        os.makedirs(os.path.dirname(paths["subjects_json"]), exist_ok=True)
        tmp = paths["subjects_json"] + ".tmp"
        with open(paths["subjects_jsonl"], "r", encoding="utf-8") as src, \
             open(tmp, "w", encoding="utf-8") as out:
            out.write('{"subjects":[')
            first = True
            for line in src:
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)  # validate
                except Exception:
                    continue
                out.write(("" if first else ",") + line)
                first = False
            out.write("]}")
        os.replace(tmp, paths["subjects_json"])
    else:
        _atomic_write_json(paths["subjects_json"], {"subjects": []})

    # summarize low-confidence JSONL → JSON (facts)
    low_facts = []
    if os.path.exists(paths["lowconf_jsonl"]):
        with open(paths["lowconf_jsonl"], "r", encoding="utf-8") as f:
            for line in f:
                try: low_facts.append(json.loads(line))
                except Exception: pass
    _atomic_write_json(paths["lowconf_json"], low_facts)

    # summarize low-confidence JSONL → JSON (NER)
    low_ner = []
    if os.path.exists(paths["ner_lowconf_jsonl"]):
        with open(paths["ner_lowconf_jsonl"], "r", encoding="utf-8") as f:
            for line in f:
                try: low_ner.append(json.loads(line))
                except Exception: pass
    _atomic_write_json(paths["ner_lowconf_json"], low_ner)

    # run meta
    run_meta = {
        "timestamp_utc": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "seed": args.seed, "domain": args.domain,
        "elicitation_strategy": args.elicitation_strategy, "ner_strategy": args.ner_strategy,
        "max_depth": args.max_depth, "max_subjects": args.max_subjects,
        "concurrency": {"concurrency": args.concurrency, "timeout_s": args.timeout},
        "footer_mode": bool(args.footer_mode),
        "footer_target": args.footer_target,
        "selfrag": {"enabled": bool(args.use_selfrag), "target": args.selfrag_target},
        "hints": {
            "major_entities": int(args.major_entities_hint),
            "min_entities": int(args.min_entities_hint),
            "avg_tokens": int(args.avg_tokens),
            "article_average_tokens": int(args.article_average_tokens),
        },
        "models": {
            "elicitation": {
                "provider": getattr(el_cfg,"provider",""),
                "model": el_cfg.model,
                "use_responses_api": getattr(el_cfg,"use_responses_api", False),
                "temperature": getattr(el_cfg, "temperature", None),
                "top_p": getattr(el_cfg, "top_p", None),
                "top_k": getattr(el_cfg, "top_k", None),
                "max_tokens": getattr(el_cfg, "max_tokens", None),
            },
            "ner": {
                "provider": getattr(ner_cfg,"provider",""),
                "model": ner_cfg.model,
                "use_responses_api": getattr(ner_cfg,"use_responses_api", False),
                "temperature": getattr(ner_cfg, "temperature", None),
                "top_p": getattr(ner_cfg, "top_p", None),
                "top_k": getattr(ner_cfg, "top_k", None),
                "max_tokens": getattr(ner_cfg, "max_tokens", None),
            },
            "selfrag": ({"provider": getattr(sr_cfg,"provider",""), "model": getattr(sr_cfg,"model",None)} if sr_cfg else None),
        },
        "args_raw": vars(args),
    }
    _atomic_write_json(paths["run_meta_json"], run_meta)

    dur = time.perf_counter() - start_ts
    acc_ct = len(rows_acc); rej_ct = len(rows_sink)
    print(f"[done] finished in {dur:.1f}s → {out_dir}")
    for k in ("queue_json","facts_json","facts_jsonl","subjects_json","subjects_jsonl","articles_jsonl",
              "lowconf_json","lowconf_jsonl","ner_jsonl","ner_lowconf_json","ner_lowconf_jsonl","run_meta_json","errors_log"):
        if k in paths:
            print(f"[out] {k:18}: {paths[k]}")
    print(f"[summary] triples accepted={acc_ct} rejected={rej_ct} subjects_done={subjects_done}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[interrupt] bye")

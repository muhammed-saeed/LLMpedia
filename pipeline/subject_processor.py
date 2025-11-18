# pipeline/subject_processor.py
from __future__ import annotations
from typing import List, Tuple, Dict, Any, Optional
import json, threading

from settings import (settings, ELICIT_SCHEMA_CAL, NER_SCHEMA_CAL)
from prompter_parser import get_prompt_messages
from db_models import write_triples_accepted
from web.fetchers import fetch_sources_for_subject

from llm.factory import make_llm_from_config

# --- thread-safe jsonl append (like your runner) ---
_jsonl_lock = threading.Lock()
def _append_jsonl(path: Optional[str], obj: dict):
    if not path:
        return
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with _jsonl_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)

# --- tiny helpers / heuristics ---
import re
_URL_RX  = re.compile(r"^https?://", re.I)
_DATE_RX = re.compile(r"^\d{4}([-/]\d{2}){0,2}$|^(January|February|March|April|May|June|July|August|September|October|November|December)\b", re.I)

def _is_literal_like(s: str) -> bool:
    if not isinstance(s, str): return True
    s2 = s.strip()
    if not s2: return True
    if _URL_RX.search(s2): return True
    if _DATE_RX.search(s2): return True
    if s2.isdigit(): return True
    if s2.lower() in {"human","engineer","inventor","person","male","female","yes","no","true","false"}:
        return True
    return False

def _looks_like_entity(obj: str) -> bool:
    if not isinstance(obj, str): return False
    o = obj.strip()
    if not o or _is_literal_like(o): return False
    words = [w for w in o.split() if w]
    if not words: return False
    has_cap_ratio = sum(1 for w in words if w[:1].isupper()) / len(words)
    return bool(has_cap_ratio >= 0.5 or (len(words) >= 2 and not o.islower()))

def _build_context_block(sources: List[Dict[str, Any]]) -> str:
    out = []
    for i, s in enumerate(sources, 1):
        out.append(f"[{i}] {s.get('title')}\nURL: {s.get('url')}\n{(s.get('text') or '').strip()}")
    return "\n\n".join(out)

# --- core steps ---

def elicit_triples_from_context(subject: str, context: str, *, calibrated=True, debug=False) -> Dict[str, Any]:
    cfg = settings.MODELS[settings.ELICIT_MODEL_KEY].model_copy(deep=True)
    llm = make_llm_from_config(cfg)

    msgs = [
        {
            "role":"system",
            "content": (
                "You are a precise information extractor.\n"
                "Extract factual triples ONLY from the provided context.\n"
                "Never invent facts not entailed by the text.\n"
                "For every triple, set confidence based on textual support.\n"
                "Always include one instanceOf triple."
            )
        },
        {
            "role":"user",
            "content": (
                f"Subject: {subject}\n\n"
                "Context (numbered sources):\n"
                f"{context}\n\n"
                "Return strict JSON per schema."
            )
        }
    ]
    schema = ELICIT_SCHEMA_CAL if calibrated else settings.ELICIT_SCHEMA_BASE
    out = llm(msgs, json_schema=schema)
    facts = out.get("facts") if isinstance(out, dict) else []
    if not isinstance(facts, list):
        facts = []
    cleaned = []
    for t in facts:
        s = str(t.get("subject", "")).strip()
        p = str(t.get("predicate", "")).strip()
        o = t.get("object", "")
        if not isinstance(o, str): o = str(o)
        if not (s and p and o): continue
        try:
            conf = float(t.get("confidence"))
        except Exception:
            conf = None
        cleaned.append({"subject": s, "predicate": p, "object": o, "confidence": conf})
    return {"facts": cleaned}

def ner_objects(subject: str, objects: List[str], debug=False) -> List[str]:
    if not objects:
        return []
    cfg = settings.MODELS[settings.NER_MODEL_KEY].model_copy(deep=True)
    llm = make_llm_from_config(cfg)

    cand = [o[:120] for o in objects if isinstance(o, str) and _looks_like_entity(o)]
    if not cand:
        return []

    block = "\n".join(cand[: settings.NER_BATCH_SIZE])
    msgs = [
        {"role":"system","content":"Decide which phrases are named entities (NE). Be conservative and use only the phrase form (no world knowledge)."},
        {"role":"user","content": f"Subject: {subject}\n\nPhrases:\n{block}\n\nReturn strict JSON per schema."}
    ]
    out = llm(msgs, json_schema=NER_SCHEMA_CAL)
    decisions = (out or {}).get("phrases") or []
    follow: List[str] = []
    for d in decisions:
        phrase = d.get("phrase")
        is_ne = bool(d.get("is_ne"))
        try:
            conf = float(d.get("confidence"))
        except Exception:
            conf = 0.0
        if is_ne and conf >= 0.90 and isinstance(phrase, str):
            follow.append(phrase.strip())
    return sorted(set(follow))

def process_subject(
    subject: str,
    hop: int,
    fdb_conn,
    *,
    jsonl_paths: Dict[str, str],
    debug: bool = False,
    conf_threshold: float = 0.70,
) -> Tuple[List[Dict[str,Any]], List[str], List[Dict[str,Any]]]:
    """
    1) fetch sources  → sources_jsonl
    2) elicit triples → facts_jsonl (+ lowconf_jsonl when confidence<thr)
    3) persist to sqlite
    4) run NER        → ner_decisions.jsonl (+ ner_lowconf_jsonl)
    Returns: (accepted_facts, next_subjects, sources)
    """
    # --- fetch ---
    if debug: print(f"[fetch] {subject} (hop {hop})")
    sources = fetch_sources_for_subject(subject)
    for s in sources:
        _append_jsonl(jsonl_paths.get("sources_jsonl"), {
            "subject": subject, "hop": hop, "source": s.get("source"),
            "title": s.get("title"), "url": s.get("url"),
            "text_len": len(s.get("text") or 0)
        })

    if not sources:
        if debug: print(f"[fetch] no sources for {subject}")
        return ([], [], [])

    # --- elicit ---
    context = _build_context_block(sources)
    if debug: print(f"[elicit] {subject} context_chars={len(context)}")
    triples = elicit_triples_from_context(subject, context, calibrated=True, debug=debug)
    facts = triples.get("facts", []) if isinstance(triples, dict) else []

    # route by confidence
    acc_rows = []
    accepted: List[Dict[str,Any]] = []
    lowconf = []
    for t in facts:
        s, p, o = t["subject"], t["predicate"], t["object"]
        c = t.get("confidence")
        if c is not None and c < conf_threshold:
            lowconf.append({**t, "hop": hop, "threshold": conf_threshold})
        else:
            accepted.append(t)
            acc_rows.append((
                s, p, o, hop,
                settings.MODELS[settings.ELICIT_MODEL_KEY].model,
                "calibrate", c
            ))

    # write jsonl streams
    for t in accepted:
        _append_jsonl(jsonl_paths.get("facts_jsonl"), {
            "subject": t["subject"], "predicate": t["predicate"], "object": t["object"],
            "confidence": t.get("confidence"), "hop": hop,
            "model": settings.MODELS[settings.ELICIT_MODEL_KEY].model, "strategy":"calibrate"
        })
    for t in lowconf:
        _append_jsonl(jsonl_paths.get("lowconf_jsonl"), {
            "subject": t["subject"], "predicate": t["predicate"], "object": t["object"],
            "confidence": t.get("confidence"), "threshold": t["threshold"], "hop": hop
        })

    # persist sqlite
    if acc_rows:
        write_triples_accepted(fdb_conn, acc_rows)

    # --- NER over objects to expand ---
    objects = [t.get("object") for t in accepted if isinstance(t, dict)]
    if debug: print(f"[ner] {subject} objects={len(objects)}")
    follow = ner_objects(subject, objects, debug=debug)

    # persist ner decisions jsonl (we only have positives at high conf; also log negatives as heuristic fallback)
    for o in objects:
        is_follow = o in follow
        _append_jsonl(jsonl_paths.get("ner_jsonl"), {
            "current_entity": subject, "hop": hop, "phrase": o,
            "is_ne": bool(is_follow), "confidence": 0.95 if is_follow else 0.5,
            "ner_model": settings.MODELS[settings.NER_MODEL_KEY].model,
            "ner_strategy": "calibrate"
        })
        # OPTIONAL: log low-conf rejections — here we treat non-follow as low
        if not is_follow:
            _append_jsonl(jsonl_paths.get("ner_lowconf_jsonl"), {
                "current_entity": subject, "hop": hop, "phrase": o,
                "is_ne": False, "confidence": 0.5, "reason": "below_threshold"
            })

    return (accepted, follow, sources)

# pipeline/article_writer.py
from __future__ import annotations
from typing import List, Dict, Any
import json, os

from settings import settings
from llm.factory import make_llm_from_config

ARTICLE_SCHEMA = {
  "type": "object",
  "additionalProperties": False,
  "properties": {
    "title": {"type":"string"},
    "sections": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
          "heading": {"type":"string"},
          "content": {"type":"string"}
        },
        "required": ["heading","content"]
      }
    },
    "references": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
          "id": {"type":"integer"},
          "title": {"type":"string"},
          "url": {"type":"string"}
        },
        "required": ["id","title","url"]
      }
    }
  },
  "required": ["title","sections","references"]
}

def _format_facts_for_prompt(subject: str, facts: List[Dict[str,Any]]) -> str:
    lines = []
    for t in facts:
        s = t.get("subject"); p = t.get("predicate"); o = t.get("object")
        c = t.get("confidence")
        if not (isinstance(s,str) and isinstance(p,str) and isinstance(o,str)):
            continue
        if c is not None:
            lines.append(f'- ("{s}", {p}, "{o}") [conf={c:.2f}]')
        else:
            lines.append(f'- ("{s}", {p}, "{o}")')
    return "\n".join(lines)

def _format_sources_for_prompt(sources: List[Dict[str,Any]]) -> str:
    lines = []
    for i, s in enumerate(sources, 1):
        title = s.get("title") or s.get("url")
        url = s.get("url")
        lines.append(f"[{i}] {title} — {url}")
    return "\n".join(lines)

def write_article(subject: str, facts: List[Dict[str,Any]], sources: List[Dict[str,Any]]) -> Dict[str,Any]:
    cfg = settings.MODELS[settings.ELICIT_MODEL_KEY].model_copy(deep=True)
    llm = make_llm_from_config(cfg)

    facts_block = _format_facts_for_prompt(subject, facts)
    refs_block = _format_sources_for_prompt(sources)

    msgs = [
        {
            "role":"system",
            "content": (
                "You are an encyclopedia editor. Write a concise, neutral article using ONLY the provided facts and sources. "
                "Do not add new claims. Cite using bracketed numbers that correspond to the provided reference list."
            )
        },
        {
            "role":"user",
            "content": (
                f"Subject: {subject}\n\n"
                "Facts (with model confidences):\n"
                f"{facts_block}\n\n"
                "Reference list (use these as [1], [2], ...):\n"
                f"{refs_block}\n\n"
                "Return strict JSON per schema with title, sections[], and references[]."
            )
        }
    ]
    out = llm(msgs, json_schema=ARTICLE_SCHEMA)
    if not isinstance(out, dict):
        return {"title": subject, "sections":[{"heading":"Summary","content":"No content."}], "references":[]}
    return out

def save_article_json(out_dir: str, subject: str, article: Dict[str,Any]):
    os.makedirs(out_dir, exist_ok=True)
    # filesystem-safe
    safe = "".join(ch for ch in subject if ch.isalnum() or ch in (" ","-","_")).strip().replace(" ","_")
    path = os.path.join(out_dir, f"{safe}.article.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(article, f, ensure_ascii=False, indent=2)
    return path

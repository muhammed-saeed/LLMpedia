#!/usr/bin/env python3
# evaluate_kb.py
from __future__ import annotations
import argparse, csv, json, os, random, time, re
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

# -----------------------------
# I/O helpers
# -----------------------------
def load_triples(path: str, limit: Optional[int]=None) -> List[Dict[str,str]]:
    p = Path(path)
    rows: List[Dict[str,str]] = []
    if p.suffix.lower() == ".jsonl":
        with open(p, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit and i >= limit: break
                if not line.strip(): continue
                obj = json.loads(line)
                rows.append({
                    "subject": str(obj.get("subject","")).strip(),
                    "predicate": str(obj.get("predicate","")).strip(),
                    "object": str(obj.get("object","")).strip(),
                    "class": str(obj.get("class","")).strip() if "class" in obj else ""
                })
    else:
        with open(p, "r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for i, row in enumerate(r):
                if limit and i >= limit: break
                rows.append({
                    "subject": (row.get("subject") or "").strip(),
                    "predicate": (row.get("predicate") or "").strip(),
                    "object": (row.get("object") or "").strip(),
                    "class": (row.get("class") or "").strip()
                })
    # basic cleanup
    rows = [t for t in rows if t["subject"] and t["predicate"] and t["object"]]
    return rows

def write_jsonl(path: str, rows: Iterable[Dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# -----------------------------
# Sampling
# -----------------------------
def sample_entities(triples: List[Dict[str,str]], n: int) -> List[str]:
    subjects = list({t["subject"] for t in triples})
    random.shuffle(subjects)
    return subjects[:min(n, len(subjects))]

def sample_triples(triples: List[Dict[str,str]], n: int) -> List[Dict[str,str]]:
    n = min(n, len(triples))
    return random.sample(triples, n) if n < len(triples) else triples

# -----------------------------
# Web search adapter (implement one)
# -----------------------------
class SearchResult(Dict[str,str]): pass

def search_snippets(query: str, k: int = 5) -> List[SearchResult]:
    """
    Implement ONE of the following and leave the others commented.

    Option A: Bing Web Search API (recommended)
        - Set env BING_API_KEY
        - pip install requests
        - Endpoint: https://api.bing.microsoft.com/v7.0/search?q=<query>
        - Return top k snippets

    Option B: SerpAPI (Google wrapper)
        - Set env SERPAPI_KEY
        - Endpoint: https://serpapi.com/search.json?q=<query>&engine=google

    Option C: Local/offline fallback
        - Return [] to mark as unverifiable (dry runs)
    """
    BING_KEY = os.getenv("BING_API_KEY")
    SERP_KEY = os.getenv("SERPAPI_KEY")

    if BING_KEY:
        import requests
        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": BING_KEY}
        params = {"q": query, "mkt": "en-US", "count": k}
        r = requests.get(url, headers=headers, params=params, timeout=30)
        r.raise_for_status()
        web = r.json().get("webPages", {}).get("value", []) if isinstance(r.json(), dict) else []
        out = []
        for w in web[:k]:
            out.append({"title": w.get("name",""), "snippet": w.get("snippet",""), "url": w.get("url","")})
        return out

    if SERP_KEY:
        import requests
        url = "https://serpapi.com/search.json"
        params = {"q": query, "engine": "google", "api_key": SERP_KEY, "num": k}
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        results = r.json().get("organic_results", [])
        out = []
        for w in results[:k]:
            out.append({"title": w.get("title",""), "snippet": w.get("snippet",""), "url": w.get("link","")})
        return out

    # Dry/offline: return nothing -> counts as unverifiable unless judged otherwise
    return []

# -----------------------------
# LLM judge adapter (implement one)
# -----------------------------
def llm_judge(prompt: str, system: Optional[str]=None) -> str:
    """
    Return ONE token string label from allowed set, given the prompt context.

    Implement one of:
    - OpenAI Chat Completions via OPENAI_API_KEY (gpt-4o, gpt-4o-mini, etc.)
    - Ollama (local) calling e.g., llama3.1
    - Any HTTP LLM you have

    For simplicity here we implement OpenAI if OPENAI_API_KEY is set; else a dummy.
    """
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_KEY:
        import requests
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {OPENAI_KEY}", "Content-Type": "application/json"}
        model = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
        messages = []
        if system:
            messages.append({"role":"system","content":system})
        messages.append({"role":"user","content":prompt})
        data = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 4  # we want a single-word label
        }
        r = requests.post(url, headers=headers, json=data, timeout=60)
        r.raise_for_status()
        out = r.json()["choices"][0]["message"]["content"].strip()
        return out
    # Fallback: deterministic 'plausible' so the pipeline runs
    return "plausible"

# -----------------------------
# Prompts (NLI-style judging)
# -----------------------------
ENTITY_PROMPT = """You are an expert verifier.
Given an entity label and {k} web snippets, decide one label:
- "verifiable" (snippets clearly support the entity exists as labeled),
- "plausible" (likely exists but evidence is indirect/weak),
- "unverifiable" (no support found in snippets).

Respond with exactly one word: verifiable | plausible | unverifiable.

Entity: {entity}

Snippets:
{snips}
"""

TRIPLE_PROMPT = """You are an expert verifier.
Given a triple (subject, predicate, object) and {k} web snippets (retrieved with subject and object terms),
decide one label:
- "entailed" (snippets clearly support the triple),
- "plausible" (consistent but not explicitly stated),
- "implausible" (unlikely given snippets),
- "false" (contradicted by snippets).

Respond with exactly one word: entailed | plausible | implausible | false.

Triple:
subject = {subj}
predicate = {pred}
object = {obj}

Snippets:
{snips}
"""

def format_snippets(snips: List[SearchResult]) -> str:
    out = []
    for i, s in enumerate(snips, 1):
        out.append(f"[{i}] {s.get('title','')}\n{s.get('snippet','')}\n{ s.get('url','') }")
    return "\n\n".join(out) if out else "(no snippets)"

# -----------------------------
# Evaluations
# -----------------------------
def eval_entities(entities: List[str], k_snips: int, sleep: float) -> Dict[str,int]:
    counts = {"verifiable":0, "plausible":0, "unverifiable":0}
    per = []
    for e in entities:
        snips = search_snippets(e, k=k_snips)
        prompt = ENTITY_PROMPT.format(entity=e, k=len(snips), snips=format_snippets(snips))
        label = llm_judge(prompt).strip().lower()
        label = {"verifiable":"verifiable","plausible":"plausible","unverifiable":"unverifiable"}.get(label,"unverifiable")
        counts[label] += 1
        per.append({"entity": e, "label": label})
        if sleep: time.sleep(sleep)
    return {"counts": counts, "details": per}

def eval_triples(tris: List[Dict[str,str]], k_snips: int, sleep: float) -> Dict[str,int]:
    counts = {"entailed":0, "plausible":0, "implausible":0, "false":0}
    per = []
    for t in tris:
        # Following the paper, query with subject + object (keeps it cheap & general)
        q = f"{t['subject']} {t['object']}"
        snips = search_snippets(q, k=k_snips)
        prompt = TRIPLE_PROMPT.format(
            subj=t["subject"], pred=t["predicate"], obj=t["object"],
            k=len(snips), snips=format_snippets(snips)
        )
        label = llm_judge(prompt).strip().lower()
        label = {"entailed":"entailed","plausible":"plausible","implausible":"implausible","false":"false"}.get(label,"plausible")
        counts[label] += 1
        per.append({**t, "label": label})
        if sleep: time.sleep(sleep)
    return {"counts": counts, "details": per}

# -----------------------------
# Simple structural checks (optional but useful)
# -----------------------------
def check_symmetry(triples: List[Dict[str,str]],
                   symm_predicates: List[str] = ["spouse"]) -> Dict[str, float]:
    # % of symmetric edges that are mirrored
    idx = {}
    for t in triples:
        idx.setdefault((t["predicate"].lower(), t["subject"].lower(), t["object"].lower()), True)
    out = {}
    for p in symm_predicates:
        p_low = p.lower()
        pairs = [(t["subject"].lower(), t["object"].lower())
                 for t in triples if t["predicate"].lower() == p_low]
        if not pairs: 
            out[p] = 0.0
            continue
        mirrored = 0
        for s,o in pairs:
            if (p_low, o, s) in idx:
                mirrored += 1
        out[p] = mirrored / len(pairs)
    return out

def check_inverse(triples: List[Dict[str,str]],
                  inv_map: Dict[str,str] = {"parent_company": "subsidiary", "subsidiary":"parent_company"}) -> Dict[str, float]:
    idx = {}
    for t in triples:
        idx.setdefault((t["predicate"].lower(), t["subject"].lower(), t["object"].lower()), True)
    out = {}
    for p, q in inv_map.items():
        p_low, q_low = p.lower(), q.lower()
        pairs = [(t["subject"].lower(), t["object"].lower())
                 for t in triples if t["predicate"].lower() == p_low]
        if not pairs:
            out[p] = 0.0
            continue
        mirrored = 0
        for s,o in pairs:
            if (q_low, o, s) in idx:
                mirrored += 1
        out[p] = mirrored / len(pairs)
    return out

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser("Evaluate a KB (entity+triple verifiability) with web snippets + LLM judge.")
    ap.add_argument("--kb", required=True, help="Path to triples file (.jsonl or .csv) with subject,predicate,object[,class].")
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    ap.add_argument("--sample-entities", type=int, default=1000)
    ap.add_argument("--sample-triples", type=int, default=1000)
    ap.add_argument("--snippets", type=int, default=5, help="#web snippets per query")
    ap.add_argument("--sleep", type=float, default=0.2, help="Politeness delay between API calls (seconds).")
    ap.add_argument("--out-dir", default="runs/Eval", help="Directory to write JSONL outputs and a summary.json.")
    ap.add_argument("--skip-entities", action="store_true")
    ap.add_argument("--skip-triples", action="store_true")
    ap.add_argument("--no-structure", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    triples = load_triples(args.kb)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    summary = {
        "kb": args.kb,
        "n_triples_loaded": len(triples),
        "judge_model": os.getenv("JUDGE_MODEL", "gpt-4o-mini (or fallback)"),
        "search": "bing" if os.getenv("BING_API_KEY") else ("serpapi" if os.getenv("SERPAPI_KEY") else "none")
    }

    if not args.skip_entities:
        entities = sample_entities(triples, args.sample_entities)
        e_res = eval_entities(entities, k_snips=args.snippets, sleep=args.sleep)
        write_jsonl(os.path.join(args.out_dir, "entities_labeled.jsonl"), e_res["details"])
        ce = e_res["counts"]
        total_e = sum(ce.values()) or 1
        summary["entities"] = {
            **ce,
            "verifiable_pct": round(100*ce["verifiable"]/total_e,1),
            "plausible_pct": round(100*ce["plausible"]/total_e,1),
            "unverifiable_pct": round(100*ce["unverifiable"]/total_e,1),
            "n": total_e
        }

    if not args.skip_triples:
        sample = sample_triples(triples, args.sample_triples)
        t_res = eval_triples(sample, k_snips=args.snippets, sleep=args.sleep)
        write_jsonl(os.path.join(args.out_dir, "triples_labeled.jsonl"), t_res["details"])
        ct = t_res["counts"]
        total_t = sum(ct.values()) or 1
        summary["triples"] = {
            **ct,
            "entailed_pct": round(100*ct["entailed"]/total_t,1),
            "plausible_pct": round(100*ct["plausible"]/total_t,1),
            "implausible_pct": round(100*ct["implausible"]/total_t,1),
            "false_pct": round(100*ct["false"]/total_t,1),
            "n": total_t
        }

    if not args.no_structure:
        summary["structure"] = {
            "symmetry_spouse": check_symmetry(triples).get("spouse", 0.0),
            "inverse_parent_company": check_inverse(triples).get("parent_company", 0.0)
        }

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # console summary
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()

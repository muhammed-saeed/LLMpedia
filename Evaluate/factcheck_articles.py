#!/usr/bin/env python
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime
import json
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

# Load environment variables (for OPENAI_API_KEY etc.)
load_dotenv()

# ---------------- tiny utils & locks ----------------

_jsonl_lock = threading.Lock()


def _str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _dbg(msg: str, debug: bool):
    if debug:
        print(msg, flush=True)


def _append_jsonl(path: str, obj: dict):
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _jsonl_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


def _unwrap_text(resp) -> str:
    """
    Same logic style as in llmpedia main script – robustly get text out of various response shapes.
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

# reuse project config / LLM factory
from settings import settings
from llm.factory import make_llm_from_config


# ---------------- Wikipedia helpers ----------------

DEFAULT_WIKI_USER_AGENT = os.getenv(
    "WIKI_USER_AGENT",
    "LLMPediaFactCheck/0.1 (https://example.com/contact; you@example.com)",
)

WIKI_API_URL = "https://en.wikipedia.org/w/api.php"


def _strip_html(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # crude but good enough for snippets
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&quot;", '"').replace("&amp;", "&")
    return text


def wiki_search_snippets(
    query: str,
    max_snippets: int = 3,
    debug: bool = False,
) -> List[Dict[str, str]]:
    """
    Use MediaWiki search API to get a few snippets for the subject.
    Returns list of {title, snippet_html, snippet_text}.
    """
    headers = {"User-Agent": DEFAULT_WIKI_USER_AGENT}
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "utf8": 1,
        "srlimit": max_snippets,
    }
    try:
        resp = requests.get(WIKI_API_URL, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("query", {}).get("search", []) or []
        out: List[Dict[str, str]] = []
        for it in items[:max_snippets]:
            title = it.get("title", "")
            snippet_html = it.get("snippet", "")
            snippet_text = _strip_html(snippet_html)
            out.append(
                {
                    "title": title,
                    "snippet_html": snippet_html,
                    "snippet_text": snippet_text,
                }
            )
        _dbg(f"[wiki] subject='{query}' → {len(out)} snippet(s)", debug)
        return out
    except Exception as e:
        _dbg(f"[wiki] error while searching '{query}': {e}", debug)
        return []


def wiki_check_link_exists(title: str, debug: bool = False) -> Dict[str, Any]:
    """
    Check whether a link title seems to correspond to any Wikipedia page.
    We do a search with srlimit=1 and see if something comes back.
    """
    headers = {"User-Agent": DEFAULT_WIKI_USER_AGENT}
    params = {
        "action": "query",
        "list": "search",
        "srsearch": title,
        "format": "json",
        "utf8": 1,
        "srlimit": 1,
    }
    try:
        resp = requests.get(WIKI_API_URL, headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("query", {}).get("search", []) or []
        if not items:
            note = "no search results"
            _dbg(f"[wiki-link] '{title}': exists=False ({note})", debug)
            return {
                "title": title,
                "exists_on_wikipedia": False,
                "top_title": None,
                "note": note,
            }
        top = items[0]
        top_title = top.get("title")
        note = "search result found"
        _dbg(f"[wiki-link] '{title}': exists=True, top_title='{top_title}'", debug)
        return {
            "title": title,
            "exists_on_wikipedia": True,
            "top_title": top_title,
            "note": note,
        }
    except requests.HTTPError as e:
        note = f"error during search: HTTPError: {e}"
        _dbg(f"[wiki-link] '{title}': {note}", debug)
        return {
            "title": title,
            "exists_on_wikipedia": None,
            "top_title": None,
            "note": note,
        }
    except Exception as e:
        note = f"error during search: {type(e).__name__}: {e}"
        _dbg(f"[wiki-link] '{title}': {note}", debug)
        return {
            "title": title,
            "exists_on_wikipedia": None,
            "top_title": None,
            "note": note,
        }


# ---------------- fact-checking prompts ----------------


def build_factcheck_messages(
    subject: str,
    wikitext: str,
    evidence_snippets: List[Dict[str, str]],
    max_claims: int,
    mode: str,
) -> List[Dict[str, str]]:
    """
    Build chat messages for article fact-checking.
    Output is expected to be JSON: {"claims":[{...}, ...]}
    """
    sys_lines = [
        "You are a strict fact-checking assistant.",
        "You receive:",
        "1) An article draft about a single subject.",
        "2) Optionally, evidence snippets from web search (mostly Wikipedia).",
        f"Your job is to extract at most {max_claims} distinct, atomic factual claims",
        "from the article and, for each claim, decide whether it is:",
        '- "true"      (well-supported by the evidence and general knowledge)',
        '- "false"     (contradicted by the evidence or known facts)',
        '- "uncertain" (evidence is incomplete, ambiguous, or not enough).',
        "",
        "Very important rules:",
        "- Only check factual claims, not opinions or vague statements.",
        "- If evidence clearly contradicts the claim, mark it as false.",
        "- If the claim goes beyond what evidence/knowledge supports, mark it as uncertain.",
        "- If you are not sure, prefer 'uncertain' over guessing.",
        "",
        "Output MUST be a single valid JSON object with this structure:",
        '{"claims":[{"claim":"...","verdict":"true|false|uncertain","confidence":0.0-1.0,"explanation":"..."}]}',
        "",
        "Where:",
        "- claim: the atomic factual statement from the article.",
        "- verdict: exactly one of 'true', 'false', 'uncertain'.",
        "- confidence: a number between 0.0 and 1.0.",
        "- explanation: 1–3 sentences explaining why.",
    ]
    if mode == "self":
        sys_lines.append(
            "You are NOT given external evidence; rely on your own knowledge but still be conservative."
        )
    else:
        sys_lines.append(
            "Use the evidence snippets as the primary source of truth; your own knowledge is secondary."
        )

    sys_msg = "\n".join(sys_lines)

    ev_lines: List[str] = []
    if evidence_snippets:
        ev_lines.append("Evidence snippets (from Wikipedia search):")
        for i, sn in enumerate(evidence_snippets, 1):
            title = sn.get("title", "")
            txt = sn.get("snippet_text", "")
            ev_lines.append(f"{i}. [{title}] {txt}")
    else:
        ev_lines.append("(No external evidence snippets available – be extra conservative.)")

    user_msg = (
        f"Subject: {subject}\n\n"
        "Article draft (LLMPedia wikitext):\n"
        "--------------------\n"
        f"{wikitext}\n"
        "--------------------\n\n"
        + "\n".join(ev_lines)
        + "\n\nReturn ONLY the JSON object, nothing else."
    )

    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]


def build_link_relatedness_messages(
    subject: str, links: List[str]
) -> List[Dict[str, str]]:
    """
    Ask the model whether each link is strongly related to the subject.
    Output JSON: {"links":[{"title":"...","related":true/false,"reason":"..."}]}
    """
    sys_msg = "\n".join(
        [
            "You are an assistant that judges whether links (titles) are strongly related",
            "to a given subject in an encyclopedic article.",
            "",
            "For each link title, answer whether it is:",
            "- related: the link is clearly relevant and helpful for the subject.",
            "- not related: only weakly connected or irrelevant for the subject.",
            "",
            "Output ONLY JSON with the structure:",
            '{"links":[{"title":"...","related":true/false,"reason":"..."}]}',
        ]
    )

    bullet_lines = [f"- {t}" for t in links]
    user_msg = (
        f"Subject: {subject}\n\n"
        "Link titles extracted from the article:\n"
        + "\n".join(bullet_lines)
        + "\n\nReturn ONLY the JSON object."
    )

    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]


def run_llm_json(llm, messages: List[Dict[str, str]], timeout: float) -> Optional[Dict[str, Any]]:
    """
    Call the LLM and try to parse a top-level JSON object from its text output.
    """
    try:
        try:
            resp = llm(messages, timeout=timeout)
        except TypeError:
            resp = llm(messages)
        txt = _unwrap_text(resp).strip()
        if not txt:
            return None

        # Try direct JSON
        try:
            obj = json.loads(txt)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

        # Fallback: try to extract a JSON object substring
        m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
        if m:
            s = m.group(0)
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                return None
        return None
    except Exception:
        return None


# ---------------- IO helpers ----------------


def _load_jsonl(path: str, max_items: int = 0) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
                if max_items > 0 and len(out) >= max_items:
                    break
    return out


# ---------------- per-article processing ----------------


def process_article(
    art: Dict[str, Any],
    fact_llm,
    args,
    out_path: str,
) -> Tuple[str, bool]:
    """
    Process a single article:
      - fetch evidence (if mode=web)
      - LLM fact-check
      - link existence
      - link relatedness (optional)
      - write JSONL record
    Returns (subject, ok_bool)
    """
    subject = art.get("subject", "")
    hop = art.get("hop", None)
    wikitext = art.get("wikitext", "") or ""
    links_from_markup = art.get("links_from_markup") or []

    debug = args.debug

    _dbg(f"[factcheck] subject='{subject}' (hop={hop})", debug)

    # 1) evidence retrieval
    if args.fact_mode == "web":
        evidence_snippets = wiki_search_snippets(subject, max_snippets=3, debug=debug)
    else:
        evidence_snippets = []

    # 2) LLM fact-checking of article
    fact_msgs = build_factcheck_messages(
        subject=subject,
        wikitext=wikitext,
        evidence_snippets=evidence_snippets,
        max_claims=args.max_claims,
        mode=args.fact_mode,
    )
    article_fact_obj = run_llm_json(fact_llm, fact_msgs, timeout=args.timeout) or {"claims": []}

    # normalize claims
    claims = article_fact_obj.get("claims") or []
    if not isinstance(claims, list):
        claims = []
    # enforce basic shape
    norm_claims: List[Dict[str, Any]] = []
    for c in claims:
        if not isinstance(c, dict):
            continue
        claim_txt = c.get("claim")
        verdict = c.get("verdict")
        if not isinstance(claim_txt, str) or not claim_txt.strip():
            continue
        if verdict not in ("true", "false", "uncertain"):
            continue
        conf = c.get("confidence")
        if isinstance(conf, (int, float)):
            try:
                conf = float(conf)
            except Exception:
                conf = None
        else:
            conf = None
        expl = c.get("explanation")
        if not isinstance(expl, str):
            expl = ""
        norm_claims.append(
            {
                "claim": claim_txt.strip(),
                "verdict": verdict,
                "confidence": conf,
                "explanation": expl.strip(),
            }
        )
    article_factcheck = {"claims": norm_claims}

    # 3) link checks (existence)
    link_checks: List[Dict[str, Any]] = []
    if args.check_links and links_from_markup:
        for title in links_from_markup:
            lc = wiki_check_link_exists(title, debug=debug)
            link_checks.append(lc)

    # 4) link relatedness (LLM)
    link_relatedness: List[Dict[str, Any]] = []
    if args.check_links and args.check_link_relatedness and links_from_markup:
        rel_msgs = build_link_relatedness_messages(subject, links_from_markup)
        rel_obj = run_llm_json(fact_llm, rel_msgs, timeout=args.timeout) or {}
        items = rel_obj.get("links") or []
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    continue
                title = item.get("title")
                if not isinstance(title, str) or not title.strip():
                    continue
                related = bool(item.get("related"))
                reason = item.get("reason")
                if not isinstance(reason, str):
                    reason = ""
                link_relatedness.append(
                    {
                        "title": title.strip(),
                        "related": related,
                        "reason": reason.strip(),
                    }
                )

    record = {
        "subject": subject,
        "hop": hop,
        "model": getattr(args, "_fact_model_name", None),
        "ts": datetime.datetime.utcnow().isoformat() + "Z",
        "mode": args.fact_mode,
        "evidence_snippets": evidence_snippets,
        "article_factcheck": article_factcheck,
        "links_from_markup": links_from_markup,
        "link_checks": link_checks,
        "link_relatedness": link_relatedness,
    }
    _append_jsonl(out_path, record)
    return subject, True


# ---------------- summary table ----------------


SUMMARY_COLS = [
    "subject",
    "hop",
    "n_claims",
    "n_true",
    "n_false",
    "n_uncertain",
    "n_links",
    "n_links_exist_true",
    "n_links_exist_false",
    "n_links_exist_unknown",
]


def summarize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    subject = rec.get("subject", "")
    hop = rec.get("hop", None)

    article_fc = rec.get("article_factcheck") or {}
    claims = article_fc.get("claims") or []
    if not isinstance(claims, list):
        claims = []
    n_claims = len(claims)
    n_true = sum(1 for c in claims if c.get("verdict") == "true")
    n_false = sum(1 for c in claims if c.get("verdict") == "false")
    n_uncertain = sum(1 for c in claims if c.get("verdict") == "uncertain")

    links = rec.get("links_from_markup") or []
    if not isinstance(links, list):
        links = []
    n_links = len(links)

    link_checks = rec.get("link_checks") or []
    if not isinstance(link_checks, list):
        link_checks = []
    n_links_exist_true = sum(1 for lc in link_checks if lc.get("exists_on_wikipedia") is True)
    n_links_exist_false = sum(1 for lc in link_checks if lc.get("exists_on_wikipedia") is False)
    n_links_exist_unknown = sum(1 for lc in link_checks if lc.get("exists_on_wikipedia") is None)

    return {
        "subject": subject,
        "hop": hop,
        "n_claims": n_claims,
        "n_true": n_true,
        "n_false": n_false,
        "n_uncertain": n_uncertain,
        "n_links": n_links,
        "n_links_exist_true": n_links_exist_true,
        "n_links_exist_false": n_links_exist_false,
        "n_links_exist_unknown": n_links_exist_unknown,
    }


def print_table(rows: List[Dict[str, Any]]):
    cols = SUMMARY_COLS

    col_widths = {c: len(c) for c in cols}
    for r in rows:
        for c in cols:
            v = r.get(c, "")
            s = str(v)
            if len(s) > col_widths[c]:
                col_widths[c] = len(s)

    def fmt_row(r: Dict[str, Any]) -> str:
        parts = []
        for c in cols:
            v = r.get(c, "")
            parts.append(str(v).ljust(col_widths[c]))
        return " | ".join(parts)

    header = fmt_row({c: c for c in cols})
    sep = "-+-".join("-" * col_widths[c] for c in cols)
    print("\n=== FACTCHECK SUMMARY TABLE ===")
    print(header)
    print(sep)
    for r in rows:
        print(fmt_row(r))
    print("================================\n")


def write_csv(rows: List[Dict[str, Any]], csv_path: str):
    """
    Save the summary rows to a CSV file.
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        writer.writeheader()
        for r in rows:
            writer.writerow({c: r.get(c, "") for c in SUMMARY_COLS})


# ---------------- main ----------------


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Fact-check LLMPedia articles.jsonl and output "
            "factchecks_articles.jsonl + summary table (+ CSV)."
        )
    )
    ap.add_argument(
        "--run-dir",
        required=True,
        help="LLMPedia run directory containing articles.jsonl",
    )
    ap.add_argument(
        "--articles-file",
        default="articles.jsonl",
        help="Input JSONL file with articles (default: articles.jsonl)",
    )
    ap.add_argument(
        "--output-file",
        default="factchecks_articles.jsonl",
        help="Output JSONL file for fact-check results (default: factchecks_articles.jsonl)",
    )
    ap.add_argument(
        "--summary-csv-file",
        default="factchecks_summary.csv",
        help="Summary CSV filename (will be created inside run-dir).",
    )
    ap.add_argument(
        "--fact-model-key",
        default=settings.ELICIT_MODEL_KEY,
        help="settings.MODELS key for fact-checking model (e.g., gpt4o-mini).",
    )
    ap.add_argument(
        "--fact-mode",
        choices=["self", "web"],
        default="web",
        help="'self' = no external evidence; 'web' = use Wikipedia search snippets as evidence.",
    )
    ap.add_argument(
        "--max-articles",
        type=int,
        default=0,
        help="Max number of articles to fact-check (0 = all).",
    )
    ap.add_argument(
        "--max-claims",
        type=int,
        default=10,
        help="Max claims per article to extract and judge.",
    )
    ap.add_argument(
        "--check-links",
        type=_str2bool,
        default=True,
        help="If true, check whether each wikilink exists on Wikipedia.",
    )
    ap.add_argument(
        "--check-link-relatedness",
        type=_str2bool,
        default=False,
        help="If true, also ask the model whether links are strongly related to the subject.",
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of worker threads for fact-checking.",
    )
    ap.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="Per-request LLM timeout (seconds).",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Verbose debug output.",
    )
    args = ap.parse_args()

    run_dir = args.run_dir
    in_path = os.path.join(run_dir, args.articles_file)
    out_path = os.path.join(run_dir, args.output_file)
    summary_csv_path = os.path.join(run_dir, args.summary_csv_file)

    print(f"[factcheck-articles] run_dir={run_dir}")
    print(f"[factcheck-articles] articles_jsonl={in_path}")
    print(f"[factcheck-articles] output={out_path}")
    print(f"[factcheck-articles] summary_csv={summary_csv_path}")
    print(f"[factcheck-articles] mode={args.fact_mode}, model-key={args.fact_model_key}")
    print(f"[factcheck-articles] Using User-Agent: {DEFAULT_WIKI_USER_AGENT}")

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input file not found: {in_path}")

    # load articles
    articles = _load_jsonl(in_path, max_items=args.max_articles)
    print(f"[factcheck-articles] loaded {len(articles)} articles")

    # prepare model
    fact_cfg = settings.MODELS[args.fact_model_key].model_copy(deep=True)
    args._fact_model_name = getattr(fact_cfg, "model", None)
    fact_llm = make_llm_from_config(fact_cfg)

    # clear output file if exists (we start fresh)
    if os.path.exists(out_path):
        os.remove(out_path)

    start = time.perf_counter()

    if not articles:
        print("[factcheck-articles] no articles to process.")
    else:
        print(
            f"[factcheck-articles] starting ThreadPoolExecutor with {args.concurrency} workers"
        )
        ok_count = 0
        total = len(articles)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, args.concurrency)
        ) as pool:
            futs = []
            for art in articles:
                fut = pool.submit(process_article, art, fact_llm, args, out_path)
                futs.append(fut)

            for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
                try:
                    subject, ok = fut.result()
                    if ok:
                        ok_count += 1
                    _dbg(
                        f"[factcheck-articles] done {i}/{total}: {subject} (ok={ok})",
                        args.debug,
                    )
                except Exception as e:
                    _dbg(f"[factcheck-articles] worker error: {e}", args.debug)

        dur = time.perf_counter() - start
        print(
            f"[factcheck-articles] finished fact-checking in {dur:.1f}s "
            f"({ok_count}/{total} articles ok)"
        )

    # -------- summary table over the produced factchecks_articles.jsonl --------
    if os.path.exists(out_path):
        print(f"[factcheck-articles] generating summary table from {out_path}")
        fact_records = _load_jsonl(out_path, max_items=0)
        rows = [summarize_record(r) for r in fact_records]

        # print nicely to stdout
        print_table(rows)

        # also write CSV
        write_csv(rows, summary_csv_path)
        print(f"[factcheck-articles] summary CSV written to: {summary_csv_path}")
    else:
        print(
            "[factcheck-articles] WARNING: output file not found, cannot generate summary table."
        )


if __name__ == "__main__":
    main()

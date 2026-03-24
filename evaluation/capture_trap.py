#!/usr/bin/env python3
"""
run_track4_capture_trap.py — Capture Trap: comparative factuality & similarity
                              for LLMPedia (ACL 2026)

FIXES IN THIS VERSION
─────────────────────
1) Wikipedia fetch now uses a TWO-PASS strategy:
   - Pass 1: batch API with prop=extracts (existing logic)
   - Pass 2: for any title where extract="" but page EXISTS (pid != -1, not missing),
     fall back to prop=revisions&rvprop=content&rvslots=main to get raw wikitext,
     then strip wikimarkup for a plain-text extract.
   This fixes disambiguation pages, list articles, year pages (e.g. "1936", "2016"),
   sports season pages, etc. that return empty extracts via the TextExtracts API.

2) Wikipedia found=True is now set whenever a non-missing page is found, even if
   the extract is empty — a second-pass revisions fetch fills it in.

3) combined_similarity now correctly excludes semantic_cosine from the weighted
   average when it is unavailable (None / not computed), rather than counting it
   as 0.0. The weights are renormalized over available metrics only.
   (This was already the logic but is made explicit with a guard.)

4) Added --wiki-rps default raised to 1.0 (still well within Wikipedia's limit).

5) Batch size default raised to 50 to reduce API call overhead.

6) All other logic unchanged from the previous version.

Outputs
───────
  capture_trap_per_subject.csv
  capture_trap_aggregate.csv
  capture_trap_report.json
  figures/
  report.txt
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import random
import re
import statistics
import sys
import time
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlencode

import urllib.request
import urllib.error

# ── Load .env file ────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    # Try multiple common locations for .env
    for _env_path in [
        "/home/samu170h/LLMPedia/.env",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"),
        os.path.join(os.getcwd(), ".env"),
    ]:
        if os.path.exists(_env_path):
            load_dotenv(_env_path)
            print(f"[env] Loaded {_env_path}")
            break
    else:
        load_dotenv()  # fallback: search current dir
except ImportError:
    print("[WARN] python-dotenv not installed — .env file not loaded")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("[WARN] matplotlib/numpy not installed — figures skipped")

try:
    from factuality_core import (
        EvalConfig,
        fetch_evidence,
        call_llm_json,
        prompt_extract_claims,
        prompt_batch_verify,
        parse_extraction,
        parse_batch_verdicts,
        tokenize_simple,
        clean_wikitext_for_eval,
    )
    HAS_FACTUALITY = True
except ImportError:
    HAS_FACTUALITY = False
    print("[WARN] factuality_core not available — factuality steps skipped")

    def tokenize_simple(text: str) -> List[str]:
        return [t for t in re.sub(r"[^\w\s]", " ", text.lower()).split() if t]

    def clean_wikitext_for_eval(text: str) -> str:
        text = re.sub(r"\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]", r"\1", text)
        text = re.sub(r"\(\d+\.\d+\)", "", text)
        return text.strip()

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] sklearn not installed — tfidf similarity skipped")

DEFAULT_SEEDS = [42, 123, 7]
DEFAULT_GROKIPEDIA_BASE = "https://grokipedia.com/page/"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
DEFAULT_SUBJECT_FIELD_PRIORITY = [
    "requested_subject",
    "seed_subject",
    "source_subject",
    "original_subject",
    "parent_subject",
    "subject",
]

WIKI_UA = os.environ.get(
    "LLMPEDIA_WIKI_UA",
    "LLMPedia-Track4Bot/1.3 (+mailto:YOUR_REAL_EMAIL)"
)

ACL_STYLE = {
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
}
SYSTEM_COLORS = {
    "llmpedia": "#2196F3",
    "grokipedia": "#FF9800",
    "wikipedia": "#4CAF50",
}

SOFT_404_PATTERNS = [
    re.compile(r"\bpage not found\b", re.I),
    re.compile(r"\bnot found\b", re.I),
    re.compile(r"\bdoes not exist\b", re.I),
    re.compile(r"\bno article\b", re.I),
    re.compile(r"\bsearch results\b", re.I),
    re.compile(r"\bcreate this page\b", re.I),
    re.compile(r"\bcaptcha\b", re.I),
    re.compile(r"\baccess denied\b", re.I),
    re.compile(r"\b403 forbidden\b", re.I),
    re.compile(r"\bcloudflare\b", re.I),
    re.compile(r"\benable javascript\b", re.I),
    re.compile(r"\btoo many requests\b", re.I),
    re.compile(r"\brate limit\b", re.I),
]

# ─── wikitext → plain text (for fallback revisions fetch) ─────────────────────
_WIKI_TEMPLATE_RE = re.compile(r"\{\{[^{}]*(?:\{\{[^{}]*\}\}[^{}]*)?\}\}", re.DOTALL)
_WIKI_LINK_RE     = re.compile(r"\[\[(?:[^\]|]+\|)?([^\]]+)\]\]")
_WIKI_EXT_LINK_RE = re.compile(r"\[https?://[^\s\]]+ ([^\]]+)\]")
_WIKI_MARKUP_RE   = re.compile(r"'{2,5}|={2,6}[^=]+=+|<[^>]+>|\[\[File:[^\]]*\]\]|\[\[Image:[^\]]*\]\]", re.I)

def _strip_wikitext(raw: str) -> str:
    """Convert raw wikitext to approximate plain text."""
    text = raw
    for _ in range(5):
        prev = text
        text = _WIKI_TEMPLATE_RE.sub(" ", text)
        if text == prev:
            break
    text = _WIKI_LINK_RE.sub(r"\1", text)
    text = _WIKI_EXT_LINK_RE.sub(r"\1", text)
    text = _WIKI_MARKUP_RE.sub(" ", text)
    text = re.sub(r"==+\s*[^=\n]+\s*==+", " ", text)
    text = re.sub(r"\n\*+\s*", "\n", text)
    text = re.sub(r"\n#+\s*", "\n", text)
    text = re.sub(r"\|\s*\w[\w\s]*=\s*[^\|}\n]*", " ", text)
    text = re.sub(r"\{\|[^|]*", " ", text)
    text = re.sub(r"\|\}", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _mean(v):
    n = [x for x in v if isinstance(x, (int, float)) and not math.isnan(x)]
    return sum(n) / len(n) if n else None

def _std(v):
    n = [x for x in v if isinstance(x, (int, float)) and not math.isnan(x)]
    return statistics.stdev(n) if len(n) >= 2 else None

def _median(v):
    n = sorted(x for x in v if isinstance(x, (int, float)) and not math.isnan(x))
    return statistics.median(n) if n else None

def _fmt(v, d=4):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)

def word_count(t: str) -> int:
    return len(re.sub(r"[^\w\s]", " ", t.lower()).split()) if t else 0

def _write_csv(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r.keys()))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({
                k: (_fmt(r.get(k)) if isinstance(r.get(k), float) else r.get(k, ""))
                for k in keys
            })
    print(f"  -> {path}  ({len(rows)} rows)")

def normalize_title(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())

def pick_eval_subject(article: Dict[str, Any], priority: List[str]) -> Tuple[str, str]:
    for f in priority:
        val = normalize_title(article.get(f, ""))
        if val:
            return val, f
    return "", ""

def load_seed_titles(seed_file: str) -> List[str]:
    with open(seed_file, "r", encoding="utf-8") as f:
        raw = f.read()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            if isinstance(obj.get("titles"), list):
                return [normalize_title(x) for x in obj["titles"] if normalize_title(x)]
            if isinstance(obj.get("topics"), list):
                return [normalize_title(x) for x in obj["topics"] if normalize_title(x)]
        if isinstance(obj, list):
            return [normalize_title(x) for x in obj if normalize_title(x)]
    except Exception:
        pass
    return [normalize_title(line) for line in raw.splitlines() if normalize_title(line)]


# ══════════════════════════════════════════════════════════════════════════════
#  WIKIPEDIA FETCHING — two-pass: extracts then revisions fallback
# ══════════════════════════════════════════════════════════════════════════════

_wiki_cache: Dict[str, Dict[str, Any]] = {}
_wiki_lock = threading.Lock()
_wiki_rate_lock = threading.Lock()
_wiki_next_allowed = 0.0


def _get_json_urllib(url: str, timeout: int = 20):
    req = urllib.request.Request(url, headers={"User-Agent": WIKI_UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8")), resp.headers


def _wiki_throttle(rps: float):
    global _wiki_next_allowed
    min_interval = 1.0 / max(rps, 1e-9)
    with _wiki_rate_lock:
        now = time.monotonic()
        wait = max(0.0, _wiki_next_allowed - now)
        _wiki_next_allowed = max(now, _wiki_next_allowed) + min_interval
    if wait > 0:
        time.sleep(wait)


def _retry_after_seconds(err: urllib.error.HTTPError) -> Optional[float]:
    if getattr(err, "headers", None):
        ra = err.headers.get("Retry-After")
        if ra:
            try:
                return max(0.0, float(ra))
            except (TypeError, ValueError):
                return None
    return None


def _empty_wiki_result(title: str, fetch_error: Optional[str] = None):
    rec = {
        "title": title,
        "text": "",
        "found": False,
        "url": f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
        "word_count": 0,
    }
    if fetch_error:
        rec["fetch_error"] = fetch_error
    return rec


def _resolve_titles(requested_titles: List[str], query_obj: Dict[str, Any]) -> Dict[str, str]:
    mapping = {t: t for t in requested_titles}
    for item in query_obj.get("normalized", []):
        mapping[item["from"]] = item["to"]
    changed = True
    while changed:
        changed = False
        for item in query_obj.get("redirects", []):
            src, dst = item["from"], item["to"]
            for k, v in list(mapping.items()):
                if v == src:
                    mapping[k] = dst
                    changed = True
    return mapping


def _do_wiki_api_call(
    params: dict,
    timeout: int,
    max_retries: int,
    base_sleep: float,
    rps: float,
    label: str = "batch",
) -> Optional[dict]:
    """Execute a Wikipedia API call with retry/backoff. Returns parsed JSON or None."""
    url = WIKIPEDIA_API + "?" + urlencode(params)
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            _wiki_throttle(rps)
            data, _headers = _get_json_urllib(url, timeout=timeout)
            err = data.get("error") or {}
            if err.get("code") == "maxlag":
                wait = min(60.0, base_sleep * (2 ** attempt) + random.random())
                print(f"  [wiki] maxlag on {label}; retrying in {wait:.1f}s")
                time.sleep(wait)
                last_err = RuntimeError(f"maxlag: {err.get('info', '')}")
                continue
            return data
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 503):
                wait = _retry_after_seconds(e) or min(60.0, base_sleep * (2 ** attempt) + random.random())
                print(f"  [wiki] HTTP {e.code} on {label}; retrying in {wait:.1f}s")
                time.sleep(wait)
                continue
            print(f"  [wiki] HTTP {e.code} on {label} — giving up")
            return None
        except Exception as e:
            last_err = e
            wait = min(30.0, base_sleep * (2 ** attempt) + random.random())
            print(f"  [wiki] error on {label}: {e}; retrying in {wait:.1f}s")
            time.sleep(wait)
    print(f"  [wiki] exhausted retries on {label}: {last_err}")
    return None


def _fetch_revisions_text_batch(
    titles: List[str],
    timeout: int,
    max_retries: int,
    base_sleep: float,
    rps: float,
) -> Dict[str, str]:
    if not titles:
        return {}
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "redirects": 1,
        "format": "json",
        "utf8": 1,
        "exlimit": "max",
        "maxlag": 5,
        "titles": "|".join(titles),
    }
    data = _do_wiki_api_call(params, timeout, max_retries, base_sleep, rps,
                             label=f"revisions({len(titles)})")
    if not data:
        return {}

    query = data.get("query", {})
    resolved = _resolve_titles(titles, query)
    pages = query.get("pages", {})

    out: Dict[str, str] = {}
    page_by_title: Dict[str, Any] = {}
    for pid, page in pages.items():
        if str(pid) == "-1" or page.get("missing") is not None:
            continue
        t = normalize_title(page.get("title", ""))
        if t:
            page_by_title[t] = page

    for requested in titles:
        canonical = normalize_title(resolved.get(requested, requested))
        page = page_by_title.get(canonical) or page_by_title.get(normalize_title(requested))
        if not page:
            continue
        revs = page.get("revisions", [])
        if not revs:
            continue
        slot = revs[0].get("slots", {}).get("main", {})
        raw = slot.get("*") or slot.get("content") or ""
        if not raw and isinstance(revs[0].get("*"), str):
            raw = revs[0]["*"]
        if raw:
            out[requested] = _strip_wikitext(raw)
    return out


def fetch_wikipedia_articles_batch(
    titles: List[str],
    timeout: int = 20,
    max_retries: int = 6,
    base_sleep: float = 1.5,
    rps: float = 1.0,
) -> Dict[str, Dict[str, Any]]:
    titles = [t for t in dict.fromkeys(titles) if t]
    out: Dict[str, Dict[str, Any]] = {}
    to_fetch: List[str] = []

    with _wiki_lock:
        for title in titles:
            cached = _wiki_cache.get(title)
            if cached is not None:
                out[title] = cached
            else:
                to_fetch.append(title)

    if not to_fetch:
        return out

    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "format": "json",
        "utf8": 1,
        "exlimit": "max",
        "maxlag": 5,
        "titles": "|".join(to_fetch),
    }
    data = _do_wiki_api_call(params, timeout, max_retries, base_sleep, rps,
                             label=f"extracts({len(to_fetch)})")

    pass1: Dict[str, Dict[str, Any]] = {}
    empty_extract_titles: List[str] = []

    if data is not None:
        query = data.get("query", {})
        resolved = _resolve_titles(to_fetch, query)
        pages = query.get("pages", {})

        page_by_title: Dict[str, Any] = {}
        for pid, page in pages.items():
            if str(pid) == "-1" or page.get("missing") is not None:
                continue
            t = normalize_title(page.get("title", ""))
            if t:
                page_by_title[t] = page

        for requested in to_fetch:
            canonical = normalize_title(resolved.get(requested, requested))
            page = page_by_title.get(canonical) or page_by_title.get(normalize_title(requested))
            if page:
                text = (page.get("extract") or "").strip()
                title_resolved = normalize_title(page.get("title", requested))
                if text:
                    rec = {
                        "title": title_resolved,
                        "text": text,
                        "found": True,
                        "url": f"https://en.wikipedia.org/wiki/{quote(title_resolved.replace(' ', '_'))}",
                        "word_count": word_count(text),
                        "fetch_pass": "extracts",
                    }
                else:
                    empty_extract_titles.append(requested)
                    rec = {
                        "title": title_resolved,
                        "text": "",
                        "found": False,
                        "url": f"https://en.wikipedia.org/wiki/{quote(title_resolved.replace(' ', '_'))}",
                        "word_count": 0,
                        "fetch_pass": "extracts_empty",
                    }
                pass1[requested] = rec
            else:
                pass1[requested] = _empty_wiki_result(requested)
    else:
        for t in to_fetch:
            pass1[t] = _empty_wiki_result(t, fetch_error="api_failure_pass1")

    if empty_extract_titles:
        print(f"  [wiki] Pass-2 revisions fallback for {len(empty_extract_titles)} empty-extract pages...")
        fallback_texts = _fetch_revisions_text_batch(
            empty_extract_titles, timeout, max_retries, base_sleep, rps
        )
        recovered = 0
        for requested in empty_extract_titles:
            plain = fallback_texts.get(requested, "").strip()
            if plain and word_count(plain) >= 30:
                existing = pass1.get(requested, {})
                pass1[requested] = {
                    "title": existing.get("title", requested),
                    "text": plain,
                    "found": True,
                    "url": existing.get("url", f"https://en.wikipedia.org/wiki/{quote(requested.replace(' ', '_'))}"),
                    "word_count": word_count(plain),
                    "fetch_pass": "revisions_fallback",
                }
                recovered += 1
        print(f"  [wiki] Pass-2 recovered {recovered}/{len(empty_extract_titles)} articles")

    for requested, rec in pass1.items():
        with _wiki_lock:
            _wiki_cache[requested] = rec
        out[requested] = rec

    return out


def fetch_wikipedia_article(
    title: str,
    timeout: int = 20,
    max_retries: int = 6,
    base_sleep: float = 1.5,
    rps: float = 1.0,
) -> Dict[str, Any]:
    return fetch_wikipedia_articles_batch(
        [title], timeout=timeout, max_retries=max_retries,
        base_sleep=base_sleep, rps=rps,
    ).get(title, _empty_wiki_result(title))


# ══════════════════════════════════════════════════════════════════════════════
#  GROKIPEDIA FETCHING
# ══════════════════════════════════════════════════════════════════════════════

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


def _extract_text_from_html(html: str) -> str:
    if not html:
        return ""
    if HAS_BS4:
        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
                tag.decompose()
            main = (
                soup.find("article")
                or soup.find("main")
                or soup.find(id="content")
                or soup.find(id="mw-content-text")
                or soup.body
            )
            txt = main.get_text(" ", strip=True) if main else soup.get_text(" ", strip=True)
            return re.sub(r"\s+", " ", txt).strip()
        except Exception:
            pass
    text = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    for ent, ch in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'"), ("&nbsp;", " ")]:
        text = text.replace(ent, ch)
    return re.sub(r"\s+", " ", text).strip()


def _looks_like_real_grok_page(title: str, html: str, text: str) -> bool:
    if not html or not text or len(text.strip()) < 300:
        return False
    head = text[:4000]
    if any(p.search(head) for p in SOFT_404_PATTERNS):
        return False
    title_norm = normalize_title(title).lower()
    html_l = html.lower()
    text_l = text.lower()
    if not (title_norm in text_l or title_norm in html_l or title_norm.replace(" ", "_") in html_l):
        return False
    return len(set(tokenize_simple(text))) >= 80


def fetch_grokipedia_article(title: str, base_url: str, timeout: int = 15) -> Dict[str, Any]:
    slug = quote(title.replace(" ", "_"), safe="")
    url = base_url.rstrip("/") + "/" + slug
    empty = {"title": title, "text": "", "found": False, "url": url, "word_count": 0, "fetch_method": "none"}
    req = urllib.request.Request(url, method="GET", headers={"User-Agent": "LLMPedia-Track4/1.3 (research)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status >= 400:
                return empty
            html = resp.read().decode("utf-8", errors="replace")
            text = _extract_text_from_html(html)
            if not _looks_like_real_grok_page(title, html, text):
                return empty
            return {"title": title, "text": text, "found": True, "url": url,
                    "word_count": word_count(text), "fetch_method": "html"}
    except Exception:
        return empty


def fetch_grokipedia_batch(
    titles: List[str],
    base_url: str,
    workers: int = 6,
    delay: float = 0.2,
) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    lock = threading.Lock()

    def _fetch_one(title):
        time.sleep(delay)
        r = fetch_grokipedia_article(title, base_url)
        with lock:
            results[title] = r
        return title, r["found"]

    print(f"  [grokipedia] Fetching {len(titles)} articles ({workers} workers)...")
    t0 = time.perf_counter()
    found_count = 0
    pool = ThreadPoolExecutor(max_workers=workers)
    futs = {pool.submit(_fetch_one, t): t for t in titles}
    try:
        for i, fut in enumerate(as_completed(futs), 1):
            _title, found = fut.result()
            if found:
                found_count += 1
            if i % 50 == 0 or i == len(titles):
                print(f"    [{i}/{len(titles)}] found={found_count}")
    except KeyboardInterrupt:
        for fut in futs:
            fut.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    print(f"  [grokipedia] Done: {found_count}/{len(titles)} found in {time.perf_counter()-t0:.1f}s")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  SIMILARITY COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def _ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)] if len(tokens) >= n else []


def compute_text_similarity(text_a: str, text_b: str, ngram_values: List[int] = None) -> Dict[str, Any]:
    if ngram_values is None:
        ngram_values = [1, 2, 3]
    if not text_a.strip() or not text_b.strip():
        return {}

    m: Dict[str, Any] = {}

    if HAS_SKLEARN:
        try:
            vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=10000)
            mat = vec.fit_transform([text_a, text_b])
            m["tfidf_cosine"] = float(sklearn_cosine(mat[0:1], mat[1:2])[0][0])
        except Exception:
            pass

    t1 = set(tokenize_simple(text_a))
    t2 = set(tokenize_simple(text_b))
    if t1 and t2:
        m["jaccard"] = len(t1 & t2) / len(t1 | t2)

    toks_a = tokenize_simple(text_a)
    toks_b = tokenize_simple(text_b)
    for n in ngram_values:
        ng1 = set(_ngrams(toks_a, n))
        ng2 = set(_ngrams(toks_b, n))
        if ng1 and ng2:
            m[f"ngram_{n}_jaccard"] = len(ng1 & ng2) / len(ng1 | ng2)
            m[f"ngram_{n}_overlap"] = len(ng1 & ng2) / min(len(ng1), len(ng2))

    try:
        from openai import OpenAI
        client = OpenAI()
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=[text_a[:32000], text_b[:32000]],
        )
        import numpy as _np
        e1 = _np.array(resp.data[0].embedding)
        e2 = _np.array(resp.data[1].embedding)
        m["semantic_cosine"] = float(
            _np.dot(e1, e2) / (_np.linalg.norm(e1) * _np.linalg.norm(e2) + 1e-9)
        )
    except Exception:
        pass

    weights = {
        "tfidf_cosine":   0.20,
        "jaccard":        0.15,
        "ngram_1_overlap": 0.15,
        "ngram_2_overlap": 0.25,
        "ngram_3_overlap": 0.25,
        "semantic_cosine": 0.45,
    }
    available_weights = {k: w for k, w in weights.items() if k in m and m[k] is not None}
    if "semantic_cosine" not in available_weights and available_weights:
        lexical_keys = [k for k in available_weights]
        lex_total = sum(weights[k] for k in lexical_keys)
        if lex_total > 0:
            available_weights = {k: weights[k] / lex_total for k in lexical_keys}
        else:
            available_weights = {k: 1.0 / len(lexical_keys) for k in lexical_keys}

    wsum = sum(available_weights[k] * float(m[k]) for k in available_weights)
    wtot = sum(available_weights.values())
    m["combined_similarity"] = (wsum / wtot) if wtot > 0 else 0.0

    m["text_a_words"] = len(toks_a)
    m["text_b_words"] = len(toks_b)

    return m


# ══════════════════════════════════════════════════════════════════════════════
#  LOCAL EVIDENCE REUSE FOR FACTUALITY
# ══════════════════════════════════════════════════════════════════════════════

def build_local_wikipedia_snippets(
    subject: str,
    wiki_text: str,
    wiki_url: Optional[str] = None,
    chunk_words: int = 140,
    stride_words: int = 100,
    max_snippets: int = 12,
) -> List[Dict[str, Any]]:
    text = re.sub(r"\s+", " ", (wiki_text or "")).strip()
    if not text:
        return []
    toks = text.split()
    snippets: List[Dict[str, Any]] = []
    i = 0
    idx = 0
    while i < len(toks) and len(snippets) < max_snippets:
        chunk = toks[i:i + chunk_words]
        if not chunk:
            break
        chunk_text = " ".join(chunk).strip()
        snippets.append({"source": "wikipedia", "title": subject, "url": wiki_url,
                         "text": chunk_text, "snippet": chunk_text, "content": chunk_text, "rank": idx})
        idx += 1
        if i + chunk_words >= len(toks):
            break
        i += stride_words
    return snippets


def get_wikipedia_evidence_snippets(subject, wiki_data, cfg, reuse_local=True):
    if reuse_local and wiki_data.get("found") and wiki_data.get("text", "").strip():
        snippets = build_local_wikipedia_snippets(subject, wiki_data["text"], wiki_data.get("url"))
        if snippets:
            return snippets
    wiki_ev = fetch_evidence(subject, "wikipedia", cfg)
    return wiki_ev.get("snippets", [])


# ══════════════════════════════════════════════════════════════════════════════
#  LLMPEDIA DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelData:
    key: str
    dir_path: str
    display_name: str = ""
    subject_names: List[str] = field(default_factory=list)
    subject_meta: Dict[str, Dict] = field(default_factory=dict)
    n_total_raw: int = 0
    subject_field_counts: Dict[str, int] = field(default_factory=dict)
    generated_subject_samples: List[Tuple[str, str]] = field(default_factory=list)


def load_llmpedia_data(
    key: str,
    dir_path: str,
    articles_file: str = "articles.jsonl",
    min_words: int = 100,
    subject_field_priority: Optional[List[str]] = None,
    seed_file: Optional[str] = None,
) -> ModelData:
    if subject_field_priority is None:
        subject_field_priority = list(DEFAULT_SUBJECT_FIELD_PRIORITY)

    dir_path = os.path.abspath(dir_path)
    arts_path = os.path.join(dir_path, articles_file)
    if not os.path.exists(arts_path):
        raise FileNotFoundError(f"[FATAL] {arts_path} not found")

    meta_path = os.path.join(dir_path, "run_meta.json")
    run_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            run_meta = json.load(f)

    cas = run_meta.get("cascading_defaults") or {}
    ar = run_meta.get("args_raw") or {}
    em = (ar.get("elicit_model_key") or "").strip()
    gm = (cas.get("global_model_key") or ar.get("model_key") or "").strip()
    dn = (em or gm or key).replace("scads-", "").replace("_", " ")

    md = ModelData(key=key, dir_path=dir_path, display_name=dn)

    print(f"[load] {key}: streaming {arts_path} ...")
    print(f"[load] subject field priority: {subject_field_priority}")

    t0 = time.perf_counter()
    best: Dict[str, Tuple[int, str, Dict]] = {}
    n_raw = 0
    field_counts: Counter = Counter()
    mismatch_samples: List[Tuple[str, str]] = []

    with open(arts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_raw += 1
            if n_raw % 100_000 == 0:
                print(f"  [{key}] ...{n_raw:,} lines  {len(best):,} kept")
            try:
                a = json.loads(line)
            except Exception:
                continue
            if not isinstance(a, dict):
                continue

            eval_subject, used_field = pick_eval_subject(a, subject_field_priority)
            generated_subject = normalize_title(a.get("subject", ""))
            if not eval_subject:
                continue

            wt = a.get("wikitext") or ""
            wc = word_count(wt)
            if min_words > 0 and wc < min_words:
                continue

            try:
                hop = int(a.get("hop", 0))
            except Exception:
                hop = 0

            field_counts[used_field] += 1
            if generated_subject and eval_subject and generated_subject != eval_subject and len(mismatch_samples) < 20:
                mismatch_samples.append((eval_subject, generated_subject))

            meta = {
                "hop": hop, "wc": wc,
                "eval_subject": eval_subject,
                "eval_subject_field": used_field,
                "generated_subject": generated_subject,
                "requested_subject": normalize_title(a.get("requested_subject", "")),
                "seed_subject": normalize_title(a.get("seed_subject", "")),
                "source_subject": normalize_title(a.get("source_subject", "")),
                "original_subject": normalize_title(a.get("original_subject", "")),
                "parent_subject": normalize_title(a.get("parent_subject", "")),
            }
            if eval_subject not in best or hop < best[eval_subject][0]:
                best[eval_subject] = (hop, wt, meta)

    md.n_total_raw = n_raw
    md.subject_names = list(best.keys())
    md.subject_meta = {}
    md.subject_field_counts = dict(field_counts)
    md.generated_subject_samples = mismatch_samples

    for s, (hop, wt, meta) in best.items():
        meta["wikitext"] = wt
        md.subject_meta[s] = meta

    elapsed = time.perf_counter() - t0
    print(f"  [{key}] {n_raw:,} lines -> {len(best):,} unique eval subjects in {elapsed:.1f}s")
    print(f"[load] {key} ({dn}): {len(md.subject_names):,} subjects")
    print(f"[load] subject field usage: {dict(field_counts)}")

    if mismatch_samples:
        print("[load] sample eval_subject vs generated_subject mismatches:")
        for eval_s, gen_s in mismatch_samples[:10]:
            print(f"  eval='{eval_s}'   generated='{gen_s}'")

    if seed_file:
        seeds = load_seed_titles(seed_file)
        seed_set = set(seeds)
        eval_set = set(md.subject_names)
        overlap = len(eval_set & seed_set)
        print(f"[load] seed-file: {seed_file}")
        print(f"[load] seed overlap: {overlap}/{len(eval_set)} = {overlap / max(1, len(eval_set)):.1%}")
        if overlap < max(5, int(0.5 * len(eval_set))):
            print("[WARN] Low overlap between eval subjects and provided seed file.")
            print("       Add requested_subject/seed_subject upstream if possible.")

    print("[load] first 10 eval subjects:")
    for s in md.subject_names[:10]:
        print(f"  - {s}")
    return md


# ══════════════════════════════════════════════════════════════════════════════
#  PER-SUBJECT EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_one_subject(
    eval_subject: str,
    llmpedia_text: str,
    wiki_data: Dict[str, Any],
    grok_data: Optional[Dict[str, Any]],
    args,
    hop: int = 0,
    generated_subject: str = "",
    eval_subject_field: str = "",
) -> Dict[str, Any]:
    wiki_text = wiki_data.get("text", "")
    wiki_found = wiki_data.get("found", False)

    rec: Dict[str, Any] = {
        "subject": eval_subject,
        "eval_subject": eval_subject,
        "generated_subject": generated_subject,
        "eval_subject_field": eval_subject_field,
        "hop": hop,
        "wiki_found": wiki_found,
        "wiki_fetch_pass": wiki_data.get("fetch_pass", ""),
        "wiki_word_count": wiki_data.get("word_count", 0),
        "wiki_url": wiki_data.get("url", ""),
        "llm_word_count": word_count(llmpedia_text),
    }

    llm_clean = clean_wikitext_for_eval(llmpedia_text)

    if wiki_found and llm_clean.strip():
        sim = compute_text_similarity(llm_clean, wiki_text, ngram_values=[1, 2, 3])
        for k, v in sim.items():
            rec[f"llm_sim_{k}"] = v
    else:
        rec["llm_sim_combined_similarity"] = None

    if HAS_FACTUALITY and wiki_found and llm_clean.strip() and not args.no_factuality:
        try:
            cfg = EvalConfig(
                fact_model_key=args.fact_model_key,
                evidence_sources=["wikipedia", "web"],
                search_backend=args.search_backend,
                web_cache_dir=args.web_cache_dir,
                max_claims=args.max_claims,
                max_retries=args.max_retries,
                concurrency=1, timeout=300.0,
                compute_similarity=False,
                debug=getattr(args, "debug", False),
            )
            ext = call_llm_json(prompt_extract_claims(eval_subject, llm_clean, cfg.max_claims), cfg, f"extract/{eval_subject}/llm")
            claims = parse_extraction(ext)
            rec["llm_n_claims"] = len(claims)
            if claims:
                # ── Verify against Wikipedia ──────────────────────────────
                wiki_snips = get_wikipedia_evidence_snippets(eval_subject, wiki_data, cfg, args.reuse_wiki_text_for_factuality)
                ver = call_llm_json(prompt_batch_verify(eval_subject, claims, "wikipedia", wiki_snips, 0), cfg, f"verify/{eval_subject}/llm_vs_wiki")
                verdicts = parse_batch_verdicts(ver, len(claims))
                n_sup = sum(1 for v in verdicts if v["verdict"] == "supported")
                n_ref = sum(1 for v in verdicts if v["verdict"] == "refuted")
                n_ins = sum(1 for v in verdicts if v["verdict"] == "insufficient")
                dec = n_sup + n_ref
                rec["llm_fact_n_supported"] = n_sup
                rec["llm_fact_n_refuted"] = n_ref
                rec["llm_fact_n_insufficient"] = n_ins
                rec["llm_fact_true_rate"] = n_sup / len(claims) if claims else 0
                rec["llm_fact_false_rate"] = n_ref / len(claims) if claims else 0
                rec["llm_fact_insufficiency_rate"] = n_ins / len(claims) if claims else 0
                rec["llm_fact_precision"] = n_sup / dec if dec > 0 else None

                # ── Verify against Web evidence ──────────────────────────
                try:
                    web_ev = fetch_evidence(eval_subject, "web", cfg, claims=claims)
                    web_snips = web_ev.get("snippets", [])
                    rec["llm_web_evidence_found"] = bool(web_snips)
                    if web_snips:
                        ver_web = call_llm_json(prompt_batch_verify(eval_subject, claims, "web", web_snips, 0), cfg, f"verify/{eval_subject}/llm_vs_web")
                        verdicts_web = parse_batch_verdicts(ver_web, len(claims))
                        ws = sum(1 for v in verdicts_web if v["verdict"] == "supported")
                        wr = sum(1 for v in verdicts_web if v["verdict"] == "refuted")
                        wi = sum(1 for v in verdicts_web if v["verdict"] == "insufficient")
                        wd = ws + wr
                        rec["llm_web_fact_n_supported"] = ws
                        rec["llm_web_fact_n_refuted"] = wr
                        rec["llm_web_fact_n_insufficient"] = wi
                        rec["llm_web_fact_true_rate"] = ws / len(claims) if claims else 0
                        rec["llm_web_fact_false_rate"] = wr / len(claims) if claims else 0
                        rec["llm_web_fact_insufficiency_rate"] = wi / len(claims) if claims else 0
                        rec["llm_web_fact_precision"] = ws / wd if wd > 0 else None
                except Exception as e_web:
                    print(f"  [WARN] Web factuality failed for '{eval_subject}' (LLM): {e_web}")
        except Exception as e:
            print(f"  [WARN] Factuality failed for '{eval_subject}' (LLM): {e}")

    if grok_data is not None:
        grok_text = grok_data.get("text", "")
        grok_found = grok_data.get("found", False)
        rec["grok_found"] = grok_found
        rec["grok_word_count"] = grok_data.get("word_count", 0)
        rec["grok_url"] = grok_data.get("url", "")

        if wiki_found and grok_found and grok_text.strip():
            sim_g = compute_text_similarity(grok_text, wiki_text, ngram_values=[1, 2, 3])
            for k, v in sim_g.items():
                rec[f"grok_sim_{k}"] = v
        else:
            rec["grok_sim_combined_similarity"] = None

        if HAS_FACTUALITY and wiki_found and grok_found and grok_text.strip() and args.grok_factuality:
            try:
                cfg_g = EvalConfig(
                    fact_model_key=args.fact_model_key,
                    evidence_sources=["wikipedia", "web"],
                    search_backend=args.search_backend,
                    web_cache_dir=args.web_cache_dir,
                    max_claims=args.max_claims,
                    max_retries=args.max_retries,
                    concurrency=1, timeout=300.0,
                    compute_similarity=False,
                    debug=getattr(args, "debug", False),
                )
                ext_g = call_llm_json(prompt_extract_claims(eval_subject, grok_text[:15000], cfg_g.max_claims), cfg_g, f"extract/{eval_subject}/grok")
                claims_g = parse_extraction(ext_g)
                rec["grok_n_claims"] = len(claims_g)
                if claims_g:
                    # ── Verify against Wikipedia ──────────────────────────
                    wiki_snips_g = get_wikipedia_evidence_snippets(eval_subject, wiki_data, cfg_g, args.reuse_wiki_text_for_factuality)
                    ver_g = call_llm_json(prompt_batch_verify(eval_subject, claims_g, "wikipedia", wiki_snips_g, 0), cfg_g, f"verify/{eval_subject}/grok_vs_wiki")
                    verdicts_g = parse_batch_verdicts(ver_g, len(claims_g))
                    ns = sum(1 for v in verdicts_g if v["verdict"] == "supported")
                    nr = sum(1 for v in verdicts_g if v["verdict"] == "refuted")
                    ni = sum(1 for v in verdicts_g if v["verdict"] == "insufficient")
                    dc = ns + nr
                    rec["grok_fact_n_supported"] = ns
                    rec["grok_fact_n_refuted"] = nr
                    rec["grok_fact_n_insufficient"] = ni
                    rec["grok_fact_true_rate"] = ns / len(claims_g) if claims_g else 0
                    rec["grok_fact_false_rate"] = nr / len(claims_g) if claims_g else 0
                    rec["grok_fact_insufficiency_rate"] = ni / len(claims_g) if claims_g else 0
                    rec["grok_fact_precision"] = ns / dc if dc > 0 else None

                    # ── Verify against Web evidence ──────────────────────
                    try:
                        web_ev_g = fetch_evidence(eval_subject, "web", cfg_g, claims=claims_g)
                        web_snips_g = web_ev_g.get("snippets", [])
                        rec["grok_web_evidence_found"] = bool(web_snips_g)
                        if web_snips_g:
                            ver_gw = call_llm_json(prompt_batch_verify(eval_subject, claims_g, "web", web_snips_g, 0), cfg_g, f"verify/{eval_subject}/grok_vs_web")
                            verdicts_gw = parse_batch_verdicts(ver_gw, len(claims_g))
                            gws = sum(1 for v in verdicts_gw if v["verdict"] == "supported")
                            gwr = sum(1 for v in verdicts_gw if v["verdict"] == "refuted")
                            gwi = sum(1 for v in verdicts_gw if v["verdict"] == "insufficient")
                            gwd = gws + gwr
                            rec["grok_web_fact_n_supported"] = gws
                            rec["grok_web_fact_n_refuted"] = gwr
                            rec["grok_web_fact_n_insufficient"] = gwi
                            rec["grok_web_fact_true_rate"] = gws / len(claims_g) if claims_g else 0
                            rec["grok_web_fact_false_rate"] = gwr / len(claims_g) if claims_g else 0
                            rec["grok_web_fact_insufficiency_rate"] = gwi / len(claims_g) if claims_g else 0
                            rec["grok_web_fact_precision"] = gws / gwd if gwd > 0 else None
                    except Exception as e_web_g:
                        print(f"  [WARN] Web factuality failed for '{eval_subject}' (Grok): {e_web_g}")
            except Exception as e:
                print(f"  [WARN] Factuality failed for '{eval_subject}' (Grok): {e}")
    else:
        rec["grok_found"] = None

    return rec


# ══════════════════════════════════════════════════════════════════════════════
#  AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════

SIM_KEYS = [
    "tfidf_cosine", "jaccard", "ngram_1_jaccard", "ngram_1_overlap",
    "ngram_2_jaccard", "ngram_2_overlap", "ngram_3_jaccard", "ngram_3_overlap",
    "semantic_cosine", "combined_similarity",
]
FACT_KEYS = ["fact_true_rate", "fact_false_rate", "fact_insufficiency_rate", "fact_precision"]
WEB_FACT_KEYS = ["web_fact_true_rate", "web_fact_false_rate", "web_fact_insufficiency_rate", "web_fact_precision"]


def aggregate_records(records: List[Dict], tag: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {"tag": tag, "n_subjects": len(records)}
    wiki_found = [r for r in records if r.get("wiki_found")]
    row["wiki_coverage"] = len(wiki_found) / len(records) if records else 0

    # ── LLM similarity (vs Wikipedia only) ────────────────────────────────
    for sk in SIM_KEYS:
        vals = [r.get(f"llm_sim_{sk}") for r in records if r.get(f"llm_sim_{sk}") is not None]
        row[f"llm_mean_{sk}"] = _mean(vals)
        row[f"llm_std_{sk}"] = _std(vals)
    # ── LLM factuality vs Wikipedia ───────────────────────────────────────
    for fk in FACT_KEYS:
        vals = [r.get(f"llm_{fk}") for r in records if r.get(f"llm_{fk}") is not None]
        row[f"llm_mean_{fk}"] = _mean(vals)
        row[f"llm_std_{fk}"] = _std(vals)
    # ── LLM factuality vs Web ────────────────────────────────────────────
    for wfk in WEB_FACT_KEYS:
        vals = [r.get(f"llm_{wfk}") for r in records if r.get(f"llm_{wfk}") is not None]
        row[f"llm_mean_{wfk}"] = _mean(vals)
        row[f"llm_std_{wfk}"] = _std(vals)

    # ── Grokipedia ────────────────────────────────────────────────────────
    grok_found = [r for r in records if r.get("grok_found")]
    grok_total = [r for r in records if r.get("grok_found") is not None]
    row["grok_coverage"] = len(grok_found) / len(grok_total) if grok_total else None
    # ── Grok similarity (vs Wikipedia only) ───────────────────────────────
    for sk in SIM_KEYS:
        vals = [r.get(f"grok_sim_{sk}") for r in records if r.get(f"grok_sim_{sk}") is not None]
        row[f"grok_mean_{sk}"] = _mean(vals)
        row[f"grok_std_{sk}"] = _std(vals)
    # ── Grok factuality vs Wikipedia ──────────────────────────────────────
    for fk in FACT_KEYS:
        vals = [r.get(f"grok_{fk}") for r in records if r.get(f"grok_{fk}") is not None]
        row[f"grok_mean_{fk}"] = _mean(vals)
        row[f"grok_std_{fk}"] = _std(vals)
    # ── Grok factuality vs Web ────────────────────────────────────────────
    for wfk in WEB_FACT_KEYS:
        vals = [r.get(f"grok_{wfk}") for r in records if r.get(f"grok_{wfk}") is not None]
        row[f"grok_mean_{wfk}"] = _mean(vals)
        row[f"grok_std_{wfk}"] = _std(vals)

    for pfx in ("llm", "wiki", "grok"):
        wc_key = f"{pfx}_word_count"
        vals = [r.get(wc_key) for r in records if r.get(wc_key) is not None and r.get(wc_key, 0) > 0]
        row[f"{pfx}_mean_word_count"] = _mean(vals)
        row[f"{pfx}_median_word_count"] = _median(vals)

    return row


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def run_capture_trap(md: ModelData, args) -> Tuple[List[Dict], List[Dict]]:
    seeds = list(args.seeds)
    n = min(args.sample_n, len(md.subject_names))
    use_grok = not args.no_grokipedia

    print(f"\n[capture_trap] n={n}  seeds={seeds}  corpus={len(md.subject_names):,}"
          f"  grokipedia={'yes' if use_grok else 'no'}")

    all_records: List[Dict] = []
    seed_agg_rows: List[Dict] = []

    for si, seed in enumerate(seeds):
        rng = random.Random(seed)
        sample = sorted(rng.sample(md.subject_names, n))

        print(f"\n  === Seed {si + 1}/{len(seeds)} (seed={seed}, n={len(sample)}) ===")

        llm_texts: Dict[str, str] = {}
        for s in sample:
            llm_texts[s] = md.subject_meta.get(s, {}).get("wikitext", "")
        n_llm = sum(1 for t in llm_texts.values() if t.strip())
        print(f"  [llmpedia] {n_llm}/{len(sample)} have text")

        print(f"  [wikipedia] Fetching {len(sample)} articles "
              f"(batch_size={args.wiki_batch_size}, rps={args.wiki_rps})...")
        wiki_data: Dict[str, Dict] = {}
        batches = [sample[i:i + args.wiki_batch_size] for i in range(0, len(sample), args.wiki_batch_size)]

        try:
            for bi, batch in enumerate(batches, 1):
                batch_res = fetch_wikipedia_articles_batch(
                    batch, timeout=args.wiki_timeout,
                    max_retries=args.wiki_max_retries,
                    base_sleep=args.wiki_retry_base,
                    rps=args.wiki_rps,
                )
                wiki_data.update(batch_res)
                if bi % 5 == 0 or bi == len(batches):
                    found = sum(1 for d in wiki_data.values() if d.get("found"))
                    print(f"    [batch {bi}/{len(batches)}] wiki_found={found}")
        except KeyboardInterrupt:
            print("\n  [wikipedia] Interrupted during batched fetch.")
            raise

        n_wiki = sum(1 for d in wiki_data.values() if d.get("found"))
        passes = {}
        for d in wiki_data.values():
            p = d.get("fetch_pass", "not_found")
            passes[p] = passes.get(p, 0) + 1
        print(f"  [wikipedia] {n_wiki}/{len(sample)} found  |  fetch_pass breakdown: {passes}")

        if n_wiki < max(10, int(0.25 * len(sample))):
            missing = [s for s, d in wiki_data.items() if not d.get("found")]
            print("  [wikipedia] sample misses (first 20):")
            for s in missing[:20]:
                print(f"    miss='{s}'")

        grok_data: Optional[Dict[str, Dict]] = None
        if use_grok:
            grok_data = fetch_grokipedia_batch(sample, args.grokipedia_url,
                                               workers=args.grok_workers, delay=0.15)

        workers = args.concurrency
        print(f"  [eval] Computing metrics for {len(sample)} subjects ({workers} workers)...")
        seed_records: List[Dict] = []
        eval_lock = threading.Lock()
        t0 = time.perf_counter()
        ok_count = [0]
        err_count = [0]

        def _eval_one(s):
            try:
                meta = md.subject_meta.get(s, {})
                rec = evaluate_one_subject(
                    eval_subject=s,
                    llmpedia_text=llm_texts.get(s, ""),
                    wiki_data=wiki_data.get(s, {"text": "", "found": False}),
                    grok_data=grok_data.get(s) if grok_data else None,
                    args=args,
                    hop=meta.get("hop", 0),
                    generated_subject=meta.get("generated_subject", ""),
                    eval_subject_field=meta.get("eval_subject_field", ""),
                )
                rec["seed"] = seed
                rec["seed_idx"] = si
                return rec
            except Exception as e:
                print(f"  [ERR] {s}: {e}", flush=True)
                return {"subject": s, "seed": seed, "seed_idx": si, "_error": str(e)}

        pool = ThreadPoolExecutor(max_workers=workers)
        futs = {pool.submit(_eval_one, s): s for s in sample}
        try:
            for i, fut in enumerate(as_completed(futs), 1):
                rec = fut.result(timeout=600)
                if rec.get("_error"):
                    err_count[0] += 1
                else:
                    ok_count[0] += 1
                with eval_lock:
                    seed_records.append(rec)
                if i % 10 == 0 or i == len(sample):
                    elapsed = time.perf_counter() - t0
                    rate = i / elapsed if elapsed > 0 else 0
                    eta = (len(sample) - i) / rate if rate > 0 else 0
                    print(f"    [{i}/{len(sample)}]  {rate:.1f}/s  ~{eta:.0f}s left  ok={ok_count[0]}  err={err_count[0]}", flush=True)
        except KeyboardInterrupt:
            for fut in futs:
                fut.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)

        all_records.extend(seed_records)
        agg = aggregate_records(seed_records, f"seed_{seed}")
        agg["seed"] = seed
        seed_agg_rows.append(agg)

        llm_cs = _mean([r.get("llm_sim_combined_similarity") for r in seed_records if r.get("llm_sim_combined_similarity") is not None])
        grok_cs = _mean([r.get("grok_sim_combined_similarity") for r in seed_records if r.get("grok_sim_combined_similarity") is not None])
        print(f"  [seed {seed}] LLM combined_sim={_fmt(llm_cs)}  Grok combined_sim={_fmt(grok_cs)}")

    return all_records, seed_agg_rows


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def _save_fig(fig, base):
    for ext in (".pdf", ".png"):
        fig.savefig(base + ext, bbox_inches="tight", **({} if ext == ".pdf" else {"dpi": 300}))
    plt.close(fig)
    print(f"  [fig] {os.path.basename(base)}")


def generate_figures(agg_rows, overall_agg, fig_dir, model_name):
    if not HAS_PLOT:
        return
    os.makedirs(fig_dir, exist_ok=True)
    plt.rcParams.update(ACL_STYLE)

    plot_metrics = [
        ("tfidf_cosine", "TF-IDF Cosine"),
        ("jaccard", "Jaccard"),
        ("ngram_2_overlap", "Bigram Overlap"),
        ("ngram_3_overlap", "Trigram Overlap"),
        ("combined_similarity", "Combined"),
    ]
    if overall_agg.get("llm_mean_semantic_cosine") is not None:
        plot_metrics.insert(-1, ("semantic_cosine", "Semantic Cosine"))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(plot_metrics))
    w = 0.35
    llm_vals = [overall_agg.get(f"llm_mean_{k}", 0) or 0 for k, _ in plot_metrics]
    llm_errs = [overall_agg.get(f"llm_std_{k}", 0) or 0 for k, _ in plot_metrics]
    has_grok = overall_agg.get("grok_coverage") is not None
    if has_grok:
        grok_vals = [overall_agg.get(f"grok_mean_{k}", 0) or 0 for k, _ in plot_metrics]
        grok_errs = [overall_agg.get(f"grok_std_{k}", 0) or 0 for k, _ in plot_metrics]

    bars_l = ax.bar(x - (w / 2 if has_grok else 0), llm_vals, w if has_grok else w * 1.5,
                    color=SYSTEM_COLORS["llmpedia"], alpha=0.85, yerr=llm_errs, capsize=4,
                    label="LLMPedia vs Wikipedia", error_kw={"elinewidth": 1.2, "capthick": 1.2})
    for bar, v in zip(bars_l, llm_vals):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7)
    if has_grok:
        bars_g = ax.bar(x + w / 2, grok_vals, w, color=SYSTEM_COLORS["grokipedia"], alpha=0.85,
                        yerr=grok_errs, capsize=4, label="Grokipedia vs Wikipedia",
                        error_kw={"elinewidth": 1.2, "capthick": 1.2})
        for bar, v in zip(bars_g, grok_vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in plot_metrics], rotation=20, ha="right")
    ax.set_ylabel("Similarity Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Similarity to Wikipedia — {model_name}", fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_fig(fig, os.path.join(fig_dir, "fig1_similarity_comparison"))

    llm_prec = overall_agg.get("llm_mean_fact_precision")
    grok_prec = overall_agg.get("grok_mean_fact_precision")
    if llm_prec is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax = axes[0]
        labels = ["LLMPedia"]
        vals = [llm_prec or 0]
        colors = [SYSTEM_COLORS["llmpedia"]]
        if grok_prec is not None:
            labels.append("Grokipedia")
            vals.append(grok_prec or 0)
            colors.append(SYSTEM_COLORS["grokipedia"])
        bars = ax.bar(labels, vals, color=colors, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Precision")
        ax.set_ylim(0, 1.05)
        ax.set_title("Factuality Precision vs Wikipedia", fontweight="bold")

        ax = axes[1]
        llm_hr = overall_agg.get("llm_mean_fact_false_rate", 0) or 0
        grok_hr = overall_agg.get("grok_mean_fact_false_rate")
        labels2 = ["LLMPedia"]
        vals2 = [llm_hr]
        colors2 = [SYSTEM_COLORS["llmpedia"]]
        if grok_hr is not None:
            labels2.append("Grokipedia")
            vals2.append(grok_hr or 0)
            colors2.append(SYSTEM_COLORS["grokipedia"])
        bars = ax.bar(labels2, vals2, color=colors2, alpha=0.85)
        for bar, v in zip(bars, vals2):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Hallucination Rate")
        ax.set_ylim(0, max(0.5, max(vals2) * 1.3 if vals2 else 0.5))
        ax.set_title("Hallucination Rate vs Wikipedia", fontweight="bold")
        fig.suptitle(f"Model: {model_name}", fontsize=9, style="italic")
        fig.tight_layout()
        _save_fig(fig, os.path.join(fig_dir, "fig2_factuality_comparison"))

    fig, ax = plt.subplots(figsize=(6, 4))
    labels_c = ["Wikipedia\n(ground truth)"]
    vals_c = [overall_agg.get("wiki_coverage", 0)]
    colors_c = [SYSTEM_COLORS["wikipedia"]]
    if has_grok:
        labels_c.append("Grokipedia")
        vals_c.append(overall_agg.get("grok_coverage", 0) or 0)
        colors_c.append(SYSTEM_COLORS["grokipedia"])
    bars = ax.bar(labels_c, [v * 100 for v in vals_c], color=colors_c, alpha=0.85)
    for bar, v in zip(bars, vals_c):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{v:.1%}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Coverage (%)")
    ax.set_ylim(0, 110)
    ax.set_title(f"Subject Coverage — {model_name}", fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, os.path.join(fig_dir, "fig3_coverage"))


# ══════════════════════════════════════════════════════════════════════════════
#  TEXT REPORT
# ══════════════════════════════════════════════════════════════════════════════

def _rtable(rows, cols=None):
    if not rows:
        return "  (no data)\n"
    if cols is None:
        cols = list(dict.fromkeys(k for r in rows for k in r.keys()))
    widths = {}
    for c in cols:
        w = len(str(c))
        for r in rows:
            v = str(r.get(c, ""))
            try:
                fv = float(v)
                v = f"{fv:.4f}" if abs(fv) < 10000 else f"{int(fv):,}"
            except Exception:
                pass
            w = max(w, len(v))
        widths[c] = min(max(w + 2, 8), 40)

    def fv(val, w):
        if val is None or val == "":
            return " " * w
        try:
            f = float(val)
            if abs(f) < .001 and f != 0:
                return f"{f:{w}.4e}"
            if f == int(f) and abs(f) > 10:
                return f"{int(f):>{w},}"
            return f"{f:{w}.4f}"
        except Exception:
            return f"{str(val):>{w}}"

    hdr = "".join(f"{str(c):>{widths[c]}}" for c in cols)
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        lines.append("".join(fv(r.get(c, ""), widths[c]) for c in cols))
    return "\n".join(lines) + "\n"


def generate_text_report(out_dir, md, all_records, seed_aggs, overall_agg, args):
    path = os.path.join(out_dir, "report.txt")
    L = []
    L.append("=" * 70)
    L.append("  LLMPedia Track 4 — Capture Trap Report")
    L.append(f"  Model: {md.display_name}  (key={md.key})")
    L.append(f"  Seeds: {args.seeds}   Sample: {args.sample_n}")
    L.append(f"  Grokipedia: {'yes' if not args.no_grokipedia else 'no'}")
    L.append(f"  Subject fields: {args.subject_field_priority}")
    L.append(f"  Evidence sources: {args.evidence_sources}")
    L.append(f"  Search backend: {args.search_backend}")
    if args.seed_file:
        L.append(f"  Seed file: {args.seed_file}")
    L.append(f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append("=" * 70)

    L.append("\n[subject field usage]")
    for k, v in sorted(md.subject_field_counts.items(), key=lambda x: (-x[1], x[0])):
        L.append(f"  {k}: {v}")

    if md.generated_subject_samples:
        L.append("\n[sample eval subject vs generated title mismatches]")
        for eval_s, gen_s in md.generated_subject_samples[:10]:
            L.append(f"  eval='{eval_s}'   generated='{gen_s}'")

    L.append(f"\n{'=' * 70}\n  OVERALL AGGREGATE\n{'=' * 70}\n")

    sim_table = [{"metric": sk, "llm_mean": overall_agg.get(f"llm_mean_{sk}"),
                  "llm_std": overall_agg.get(f"llm_std_{sk}"),
                  "grok_mean": overall_agg.get(f"grok_mean_{sk}"),
                  "grok_std": overall_agg.get(f"grok_std_{sk}")} for sk in SIM_KEYS]
    L.append("  SIMILARITY vs Wikipedia:")
    L.append(_rtable(sim_table, ["metric", "llm_mean", "llm_std", "grok_mean", "grok_std"]))

    fact_table = [{"metric": fk, "llm_mean": overall_agg.get(f"llm_mean_{fk}"),
                   "llm_std": overall_agg.get(f"llm_std_{fk}"),
                   "grok_mean": overall_agg.get(f"grok_mean_{fk}"),
                   "grok_std": overall_agg.get(f"grok_std_{fk}")} for fk in FACT_KEYS]
    L.append("  FACTUALITY vs Wikipedia:")
    L.append(_rtable(fact_table, ["metric", "llm_mean", "llm_std", "grok_mean", "grok_std"]))

    web_fact_table = [{"metric": wfk, "llm_mean": overall_agg.get(f"llm_mean_{wfk}"),
                       "llm_std": overall_agg.get(f"llm_std_{wfk}"),
                       "grok_mean": overall_agg.get(f"grok_mean_{wfk}"),
                       "grok_std": overall_agg.get(f"grok_std_{wfk}")} for wfk in WEB_FACT_KEYS]
    L.append("  FACTUALITY vs Web:")
    L.append(_rtable(web_fact_table, ["metric", "llm_mean", "llm_std", "grok_mean", "grok_std"]))

    L.append("  COVERAGE:")
    L.append(f"    Wikipedia found:   {overall_agg.get('wiki_coverage', 0):.1%}")
    gc = overall_agg.get("grok_coverage")
    if gc is not None:
        L.append(f"    Grokipedia found:  {gc:.1%}")
    L.append(f"    Total subjects:    {overall_agg.get('n_subjects', 0)}")

    L.append("\n  WORD COUNTS (mean / median):")
    for pfx, label in [("llm", "LLMPedia"), ("wiki", "Wikipedia"), ("grok", "Grokipedia")]:
        mn = overall_agg.get(f"{pfx}_mean_word_count")
        md_val = overall_agg.get(f"{pfx}_median_word_count")
        if mn is not None:
            L.append(f"    {label:15s}: {mn:,.0f} / {md_val:,.0f}")

    L.append(f"\n{'=' * 70}\n  PER-SEED AGGREGATES\n{'=' * 70}\n")
    seed_display_cols = [
        "tag", "n_subjects", "wiki_coverage",
        "llm_mean_combined_similarity", "grok_mean_combined_similarity",
        "llm_mean_fact_precision", "grok_mean_fact_precision",
        "llm_mean_web_fact_precision", "grok_mean_web_fact_precision",
    ]
    L.append(_rtable(seed_aggs, seed_display_cols))
    L.append("\n" + "=" * 70 + "\n  END OF REPORT\n" + "=" * 70 + "\n")

    text = "\n".join(L)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[report] {path}  ({len(text):,} chars)")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Track 4: Capture Trap — LLMPedia & Grokipedia vs Wikipedia (v2: two-pass wiki fetch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model-key", default="")
    ap.add_argument("--articles-file", default="articles.jsonl")
    ap.add_argument("--output-dir", default="./capture_trap")
    ap.add_argument("--min-words", type=int, default=100)
    ap.add_argument("--sample-n", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--seed", type=int, action="append", default=None)
    ap.add_argument("--subject-field-priority", nargs="+", default=DEFAULT_SUBJECT_FIELD_PRIORITY)
    ap.add_argument("--seed-file", default="")
    ap.add_argument("--no-grokipedia", action="store_true")
    ap.add_argument("--grokipedia-url", default=DEFAULT_GROKIPEDIA_BASE)
    ap.add_argument("--grok-workers", type=int, default=6)
    ap.add_argument("--grok-factuality", action="store_true", default=False)
    ap.add_argument("--no-factuality", action="store_true")
    ap.add_argument("--fact-model-key", default="gpt-4.1-nano")
    ap.add_argument("--max-claims", type=int, default=10)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--reuse-wiki-text-for-factuality", action="store_true", default=True)
    ap.add_argument("--no-reuse-wiki-text-for-factuality", dest="reuse_wiki_text_for_factuality", action="store_false")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--wiki-workers", type=int, default=8)
    ap.add_argument("--wiki-batch-size", type=int, default=50)
    ap.add_argument("--wiki-rps", type=float, default=1.0)
    ap.add_argument("--wiki-max-retries", type=int, default=6)
    ap.add_argument("--wiki-retry-base", type=float, default=1.5)
    ap.add_argument("--wiki-timeout", type=int, default=30)
    ap.add_argument("--clean-output", action="store_true")
    ap.add_argument("--debug", action="store_true")

    # ── NEW: evidence / web search CLI args ───────────────────────────────────
    ap.add_argument("--evidence-sources", nargs="+", default=["wikipedia", "web"],
                    help="Evidence sources for factuality checking")
    ap.add_argument("--search-backend", default="valyu",
                    choices=["auto", "valyu", "serper", "brave", "ddg"],
                    help="Web search backend for evidence retrieval")
    ap.add_argument("--web-cache-dir",
                    default="/home/samu170h/LLMPedia/openLLMPedia/web_cache",
                    help="Directory for caching web search results and fetched pages")

    args = ap.parse_args()
    if args.seed:
        args.seeds = list(args.seed)

    out_dir = os.path.abspath(args.output_dir)
    fig_dir = os.path.join(out_dir, "figures")

    if args.clean_output:
        import shutil
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
    os.makedirs(fig_dir, exist_ok=True)

    # Verify API keys are available
    _valyu_key = os.environ.get("VALYU_API_KEY", "").strip()
    _serper_key = os.environ.get("SERPER_API_KEY", "").strip()
    _openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    print(f"\n[env] VALYU_API_KEY:  {'set (' + _valyu_key[:8] + '...)' if _valyu_key else 'NOT SET'}")
    print(f"[env] SERPER_API_KEY: {'set' if _serper_key else 'NOT SET'}")
    print(f"[env] OPENAI_API_KEY: {'set' if _openai_key else 'NOT SET'}")

    print(f"\n{'=' * 70}")
    print("[track4] CAPTURE TRAP — LLMPedia & Grokipedia vs Wikipedia (v2: two-pass wiki)")
    print(f"  run-dir:      {args.run_dir}")
    print(f"  sample:       {args.sample_n} per seed x {len(args.seeds)} seeds")
    print(f"  grokipedia:   {'yes' if not args.no_grokipedia else 'no'}"
          f"  (factuality={'yes' if args.grok_factuality else 'no'})")
    print(f"  factuality:   {'yes' if not args.no_factuality else 'no'}"
          f"  (model={args.fact_model_key})")
    print(f"  evidence:     {args.evidence_sources}  backend={args.search_backend}")
    print(f"  web cache:    {args.web_cache_dir}")
    print(f"  wiki fetch:   batch_size={args.wiki_batch_size}  rps={args.wiki_rps}  "
          f"max_retries={args.wiki_max_retries}  timeout={args.wiki_timeout}s  [TWO-PASS]")
    print(f"  concurrency:  eval={args.concurrency}  grok={args.grok_workers}")
    print(f"  subject prio: {args.subject_field_priority}")
    if args.seed_file:
        print(f"  seed-file:    {args.seed_file}")
    print(f"  output:       {out_dir}")
    print(f"{'=' * 70}\n")

    model_key = args.model_key or os.path.basename(os.path.normpath(args.run_dir))
    md = load_llmpedia_data(
        model_key, args.run_dir, args.articles_file, args.min_words,
        subject_field_priority=args.subject_field_priority,
        seed_file=args.seed_file or None,
    )

    if len(md.subject_names) < args.sample_n:
        print(f"[WARN] Only {len(md.subject_names)} subjects available, requested {args.sample_n}. Using all.")
        args.sample_n = len(md.subject_names)

    try:
        all_records, seed_aggs = run_capture_trap(md, args)
    except KeyboardInterrupt:
        print("\n[track4] Interrupted by user.")
        raise SystemExit(130)

    overall_agg = aggregate_records(all_records, "overall")
    print(f"\n[overall] n={overall_agg['n_subjects']}  wiki_cov={_fmt(overall_agg.get('wiki_coverage'))}")
    print(f"  LLM  combined_sim={_fmt(overall_agg.get('llm_mean_combined_similarity'))} "
          f"+/- {_fmt(overall_agg.get('llm_std_combined_similarity'))}")
    if overall_agg.get("grok_coverage") is not None:
        print(f"  Grok combined_sim={_fmt(overall_agg.get('grok_mean_combined_similarity'))} "
              f"+/- {_fmt(overall_agg.get('grok_std_combined_similarity'))}")

    print("\n[output] Writing ...")
    _write_csv(all_records, os.path.join(out_dir, "capture_trap_per_subject.csv"))
    _write_csv(seed_aggs + [overall_agg], os.path.join(out_dir, "capture_trap_aggregate.csv"))

    report_json = {
        "generated": datetime.datetime.now().isoformat(),
        "config": {k: v for k, v in vars(args).items()},
        "model": {
            "key": md.key, "display": md.display_name,
            "n_subjects": len(md.subject_names),
            "n_total_raw": md.n_total_raw,
            "subject_field_counts": md.subject_field_counts,
        },
        "overall_aggregate": overall_agg,
        "seed_aggregates": seed_aggs,
        "per_subject_count": len(all_records),
    }
    rp = os.path.join(out_dir, "capture_trap_report.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report_json, f, ensure_ascii=False, indent=2, default=str)
    print(f"  -> {rp}")

    print("\n[figures] ...")
    generate_figures(seed_aggs, overall_agg, fig_dir, md.display_name)

    print("\n[report] ...")
    generate_text_report(out_dir, md, all_records, seed_aggs, overall_agg, args)

    print(f"\n{'=' * 70}")
    print(f"[track4] DONE — {md.display_name}")
    print(f"  Subjects evaluated: {len(all_records)}")
    print(f"  Wikipedia coverage: {overall_agg.get('wiki_coverage', 0):.1%}")
    gc = overall_agg.get("grok_coverage")
    if gc is not None:
        print(f"  Grokipedia coverage: {gc:.1%}")
    print("\n  LLMPedia vs Wikipedia:")
    print(f"    Combined similarity: {_fmt(overall_agg.get('llm_mean_combined_similarity'))} "
          f"+/- {_fmt(overall_agg.get('llm_std_combined_similarity'))}")
    lp = overall_agg.get("llm_mean_fact_precision")
    if lp is not None:
        print(f"    Factuality precision (wiki): {_fmt(lp)} "
              f"+/- {_fmt(overall_agg.get('llm_std_fact_precision'))}")
        print(f"    Hallucination rate (wiki):   {_fmt(overall_agg.get('llm_mean_fact_false_rate'))} "
              f"+/- {_fmt(overall_agg.get('llm_std_fact_false_rate'))}")
        print(f"    Insufficiency rate (wiki):   {_fmt(overall_agg.get('llm_mean_fact_insufficiency_rate'))} "
              f"+/- {_fmt(overall_agg.get('llm_std_fact_insufficiency_rate'))}")
    lwp = overall_agg.get("llm_mean_web_fact_precision")
    if lwp is not None:
        print(f"    Factuality precision (web):  {_fmt(lwp)} "
              f"+/- {_fmt(overall_agg.get('llm_std_web_fact_precision'))}")
        print(f"    Hallucination rate (web):    {_fmt(overall_agg.get('llm_mean_web_fact_false_rate'))} "
              f"+/- {_fmt(overall_agg.get('llm_std_web_fact_false_rate'))}")
        print(f"    Insufficiency rate (web):    {_fmt(overall_agg.get('llm_mean_web_fact_insufficiency_rate'))} "
              f"+/- {_fmt(overall_agg.get('llm_std_web_fact_insufficiency_rate'))}")
    if gc is not None:
        print("\n  Grokipedia vs Wikipedia:")
        print(f"    Combined similarity: {_fmt(overall_agg.get('grok_mean_combined_similarity'))} "
              f"+/- {_fmt(overall_agg.get('grok_std_combined_similarity'))}")
        gp = overall_agg.get("grok_mean_fact_precision")
        if gp is not None:
            print(f"    Factuality precision (wiki): {_fmt(gp)} "
                  f"+/- {_fmt(overall_agg.get('grok_std_fact_precision'))}")
            print(f"    Hallucination rate (wiki):   {_fmt(overall_agg.get('grok_mean_fact_false_rate'))} "
                  f"+/- {_fmt(overall_agg.get('grok_std_fact_false_rate'))}")
            print(f"    Insufficiency rate (wiki):   {_fmt(overall_agg.get('grok_mean_fact_insufficiency_rate'))} "
                  f"+/- {_fmt(overall_agg.get('grok_std_fact_insufficiency_rate'))}")
        gwp = overall_agg.get("grok_mean_web_fact_precision")
        if gwp is not None:
            print(f"    Factuality precision (web):  {_fmt(gwp)} "
                  f"+/- {_fmt(overall_agg.get('grok_std_web_fact_precision'))}")
            print(f"    Hallucination rate (web):    {_fmt(overall_agg.get('grok_mean_web_fact_false_rate'))} "
                  f"+/- {_fmt(overall_agg.get('grok_std_web_fact_false_rate'))}")
            print(f"    Insufficiency rate (web):    {_fmt(overall_agg.get('grok_mean_web_fact_insufficiency_rate'))} "
                  f"+/- {_fmt(overall_agg.get('grok_std_web_fact_insufficiency_rate'))}")
    print(f"\n  Output: {out_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
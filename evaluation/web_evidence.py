"""
web_evidence.py – High-quality web evidence collection for factuality checking.

Designed for HPC (TU Dresden etc.) – all search backends are FREE.

Speed optimizations (v0.5):
  - PARALLEL search queries (3 queries fire simultaneously, was sequential)
  - In-memory cache layer on top of disk cache (avoid repeated disk reads)
  - Connection pooling with auto-retry (429/5xx)
  - Buffered audit writes (100 records per flush, not 1)
  - Parallel page fetching (already existed, now with pooled connections)

Key features:
  - Smart query construction (adds context to avoid garbage results)
  - Authoritative domain preference (.edu, .gov, .org, Britannica, etc.)
  - Large junk-domain blocklist (eBay, Facebook, Pinterest, zhihu, etc.)
  - Multiple search strategies per subject for coverage
  - PARALLEL page fetching (configurable max_fetch_pages, default 3)
  - Granular quality ranking (0-100) for every source
  - Disk cache (~7 day TTL) for HPC reruns
  - Structured per-subject audit trail (.jsonl)

Search backends (priority order, all FREE):
  1. DuckDuckGo  (pip install ddgs)  – no API key
  2. SearXNG     (SEARXNG_API_BASE)  – self-hosted
  3. Google SERP scraping            – fragile fallback

Environment variables:
  SEARXNG_API_BASE             – SearXNG instance URL (optional)
  WEB_EVIDENCE_CACHE_DIR       – Cache root (default: ~/.cache/llmpedia_web_evidence)
  WEB_EVIDENCE_USER_AGENT      – User-Agent string
  WEB_EVIDENCE_TIMEOUT         – HTTP timeout seconds (default: 15)
  WEB_EVIDENCE_MAX_SITES       – Max unique domains per query (default: 5)
  WEB_EVIDENCE_SEARCH_BACKEND  – Force: ddg|searxng|google (default: auto)
  WEB_EVIDENCE_MAX_FETCH_PAGES – Max pages to actually download (default: 3)
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import time
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import quote_plus, urlparse

import requests

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_UA = os.getenv(
    "WEB_EVIDENCE_USER_AGENT",
    "LLMPediaWebEvidence/0.5 (+research; HPC)",
)
CACHE_DIR = os.getenv(
    "WEB_EVIDENCE_CACHE_DIR",
    os.path.expanduser("~/.cache/llmpedia_web_evidence"),
)
MAX_SITES_DEFAULT = int(os.getenv("WEB_EVIDENCE_MAX_SITES", "5"))
TIMEOUT_DEFAULT = float(os.getenv("WEB_EVIDENCE_TIMEOUT", "15"))
SEARCH_BACKEND = os.getenv("WEB_EVIDENCE_SEARCH_BACKEND", "auto").strip().lower()
MAX_FETCH_PAGES_DEFAULT = int(os.getenv("WEB_EVIDENCE_MAX_FETCH_PAGES", "3"))


# ─────────────────────────────────────────────────────────────────────────────
# Connection-pooled session with auto-retry
# ─────────────────────────────────────────────────────────────────────────────

def _make_robust_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": DEFAULT_UA})
    try:
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        retry = Retry(total=3, backoff_factor=0.3,
                      status_forcelist=[429, 500, 502, 503, 504],
                      allowed_methods=["GET", "POST"])
        adapter = HTTPAdapter(max_retries=retry, pool_connections=30, pool_maxsize=30)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
    except ImportError:
        pass
    return s


_session = _make_robust_session()


# ─────────────────────────────────────────────────────────────────────────────
# Domain quality control
# ─────────────────────────────────────────────────────────────────────────────

BLOCKED_DOMAINS: Set[str] = {
    # Search engines
    "google.com", "bing.com", "duckduckgo.com", "yahoo.com", "yandex.com",
    "baidu.com", "sogou.com",
    # Shopping / e-commerce
    "ebay.com", "amazon.com", "amazon.co.uk", "amazon.de", "etsy.com",
    "aliexpress.com", "alibaba.com", "walmart.com", "target.com",
    "shopify.com", "mercadolibre.com",
    # Social media
    "facebook.com", "instagram.com", "twitter.com", "x.com", "tiktok.com",
    "linkedin.com", "pinterest.com", "tumblr.com", "snapchat.com",
    "threads.net", "mastodon.social",
    # Video / streaming
    "youtube.com", "vimeo.com", "dailymotion.com", "twitch.tv",
    # Chinese / non-English general sites
    "zhihu.com", "zhidao.baidu.com", "weibo.com", "bilibili.com",
    "douban.com", "tieba.baidu.com", "csdn.net", "jianshu.com",
    "163.com", "sohu.com", "sina.com.cn", "qq.com",
    # Forums / Q&A
    "reddit.com", "quora.com", "stackoverflow.com", "stackexchange.com",
    "answers.com", "ask.com",
    # Image / media hosting
    "flickr.com", "imgur.com", "deviantart.com", "picryl.com",
    "shutterstock.com", "gettyimages.com", "unsplash.com",
    # Model trains / hobby
    "jbftoysandtrains.com", "broadway-limited.com", "midwestmodelrr.com",
    "ictrainsandhobbies.com",
    # Classifieds / jobs
    "craigslist.org", "indeed.com", "glassdoor.com", "monster.com",
    # Other low-quality
    "blogspot.com", "wordpress.com", "wixsite.com",
    "weebly.com", "squarespace.com",
    "fandom.com",
    # Spam / content farms
    "answers.yahoo.com", "wikihow.com", "ehow.com",
    "hubpages.com", "helium.com", "suite101.com",
    "buzzfeed.com", "boredpanda.com",
}

# Tiered authoritative domain scoring (0-100)
TIER1_DOMAINS: Dict[str, int] = {
    "britannica.com": 100, "en.wikipedia.org": 98,
    "loc.gov": 97, "archives.gov": 97, "si.edu": 96,
    "who.int": 96, "un.org": 95, "worldbank.org": 95,
    "cia.gov": 95, "europa.eu": 95,
}
TIER2_DOMAINS: Dict[str, int] = {
    "reuters.com": 94, "apnews.com": 94,
    "bbc.com": 93, "bbc.co.uk": 93,
    "pbs.org": 92, "npr.org": 92,
    "nytimes.com": 91, "washingtonpost.com": 90,
    "theguardian.com": 90, "economist.com": 90,
    "ft.com": 89, "wsj.com": 89,
    "nature.com": 93, "science.org": 93, "sciencedirect.com": 91,
    "pubmed.ncbi.nlm.nih.gov": 93, "ncbi.nlm.nih.gov": 92,
    "scholar.google.com": 88, "jstor.org": 90,
}
TIER3_DOMAINS: Dict[str, int] = {
    "nationalgeographic.com": 84, "smithsonianmag.com": 84,
    "history.com": 82, "biography.com": 80,
    "merriam-webster.com": 80, "oxforddnb.com": 82,
    "plato.stanford.edu": 85, "iep.utm.edu": 82,
    "mathworld.wolfram.com": 82, "newworldencyclopedia.org": 78,
    "snopes.com": 78, "factcheck.org": 78, "politifact.com": 77,
    "theconversation.com": 77, "scientificamerican.com": 80,
    "spectrum.ieee.org": 79, "wired.com": 76, "arstechnica.com": 76,
    "time.com": 78, "theatlantic.com": 78, "newyorker.com": 78,
    "foreignaffairs.com": 79, "brookings.edu": 80,
    "cfr.org": 79, "rand.org": 79,
    "imdb.com": 75, "allmusic.com": 75, "discogs.com": 75,
    "baseball-reference.com": 78, "basketball-reference.com": 78,
    "pro-football-reference.com": 78,
}
TIER4_DOMAINS: Dict[str, int] = {
    "cnn.com": 72, "abcnews.go.com": 72,
    "nbcnews.com": 72, "cbsnews.com": 72, "foxnews.com": 68,
    "usatoday.com": 70, "latimes.com": 72, "chicagotribune.com": 70,
    "bostonglobe.com": 70, "independent.co.uk": 70, "telegraph.co.uk": 70,
    "dw.com": 72, "aljazeera.com": 72, "france24.com": 72,
    "medium.com": 55, "forbes.com": 65, "businessinsider.com": 63,
    "investopedia.com": 68, "healthline.com": 65,
    "mayoclinic.org": 78, "webmd.com": 62, "verywellhealth.com": 60,
}

_DOMAIN_SCORES: Dict[str, int] = {}
_DOMAIN_SCORES.update(TIER4_DOMAINS)
_DOMAIN_SCORES.update(TIER3_DOMAINS)
_DOMAIN_SCORES.update(TIER2_DOMAINS)
_DOMAIN_SCORES.update(TIER1_DOMAINS)

HIGH_QUALITY_TLDS = {".edu", ".gov", ".org", ".int", ".mil"}
MEDIUM_QUALITY_TLDS = {".ac.uk", ".edu.au", ".ac.jp"}

QUALITY_DOMAIN_KEYWORDS = {
    "encyclopedia": 70, "university": 70, "institute": 68,
    "museum": 68, "library": 68, "academy": 66,
    "journal": 65, "research": 64, "science": 64,
    "review": 55, "archive": 60, "national": 58,
    "history": 58, "education": 58,
}


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _root_domain(domain: str) -> str:
    parts = domain.split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return domain


def _is_blocked(url: str) -> bool:
    d = _domain(url)
    rd = _root_domain(d)
    return d in BLOCKED_DOMAINS or rd in BLOCKED_DOMAINS


def _domain_quality_score(url: str) -> float:
    d = _domain(url)
    rd = _root_domain(d)
    score = _DOMAIN_SCORES.get(d) or _DOMAIN_SCORES.get(rd)
    if score is not None:
        if url.startswith("https"):
            score = min(100, score + 1)
        return float(score)
    for tld in HIGH_QUALITY_TLDS:
        if d.endswith(tld):
            return 70.0
    for tld in MEDIUM_QUALITY_TLDS:
        if d.endswith(tld):
            return 65.0
    best_kw = 0
    for kw, kw_score in QUALITY_DOMAIN_KEYWORDS.items():
        if kw in d:
            best_kw = max(best_kw, kw_score)
    if best_kw > 0:
        return float(best_kw)
    base = 30.0
    if url.startswith("https"):
        base += 2.0
    return base


def _quality_label(score: float) -> str:
    if score >= 95:
        return "gold"
    if score >= 85:
        return "excellent"
    if score >= 75:
        return "good"
    if score >= 60:
        return "acceptable"
    if score >= 40:
        return "fair"
    return "low"


# ─────────────────────────────────────────────────────────────────────────────
# Lazy imports
# ─────────────────────────────────────────────────────────────────────────────

_bs4_ok: Optional[bool] = None
_ddg_ok: Optional[bool] = None


def _has_bs4() -> bool:
    global _bs4_ok
    if _bs4_ok is None:
        try:
            from bs4 import BeautifulSoup  # noqa: F401
            _bs4_ok = True
        except ImportError:
            _bs4_ok = False
    return _bs4_ok


def _has_ddg() -> bool:
    global _ddg_ok
    if _ddg_ok is None:
        try:
            try:
                from ddgs import DDGS  # noqa: F401
                _ddg_ok = True
            except ImportError:
                from duckduckgo_search import DDGS  # noqa: F401
                _ddg_ok = True
        except ImportError:
            _ddg_ok = False
    return _ddg_ok


def _get_ddgs_class():
    try:
        from ddgs import DDGS
        return DDGS
    except ImportError:
        from duckduckgo_search import DDGS
        return DDGS


# ─────────────────────────────────────────────────────────────────────────────
# Two-layer cache: in-memory + disk (7-day TTL)
#
# SPEED FIX: The old code hit disk for every cache lookup.
# Now: check memory first (instant), fall through to disk only on miss.
# Memory cache is bounded to prevent OOM on large runs.
# ─────────────────────────────────────────────────────────────────────────────

_mem_cache: Dict[str, Any] = {}
_mem_cache_lock = threading.Lock()
_MEM_CACHE_MAX = 5000  # max entries before LRU eviction
_disk_cache_lock = threading.Lock()


def _ensure_cache_dir() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_key_str(prefix: str, key: str) -> str:
    return f"{prefix}::{key}"


def _cache_path(prefix: str, key: str) -> str:
    _ensure_cache_dir()
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{prefix}_{h}.json")


def _load_cache(prefix: str, key: str, max_age_hours: float = 168.0) -> Optional[Any]:
    # Layer 1: in-memory (instant)
    ck = _cache_key_str(prefix, key)
    with _mem_cache_lock:
        if ck in _mem_cache:
            return _mem_cache[ck]

    # Layer 2: disk
    path = _cache_path(prefix, key)
    if not os.path.exists(path):
        return None
    try:
        age_h = (time.time() - os.path.getmtime(path)) / 3600
        if age_h > max_age_hours:
            return None
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        # Promote to memory cache
        with _mem_cache_lock:
            if len(_mem_cache) >= _MEM_CACHE_MAX:
                # Simple eviction: drop ~10% oldest
                keys = list(_mem_cache.keys())[:_MEM_CACHE_MAX // 10]
                for k in keys:
                    _mem_cache.pop(k, None)
            _mem_cache[ck] = obj
        return obj
    except Exception:
        return None


def _save_cache(prefix: str, key: str, obj: Any) -> None:
    # Save to memory
    ck = _cache_key_str(prefix, key)
    with _mem_cache_lock:
        if len(_mem_cache) >= _MEM_CACHE_MAX:
            keys = list(_mem_cache.keys())[:_MEM_CACHE_MAX // 10]
            for k in keys:
                _mem_cache.pop(k, None)
        _mem_cache[ck] = obj

    # Save to disk
    with _disk_cache_lock:
        path = _cache_path(prefix, key)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False)
        except Exception:
            pass


def clear_search_cache(subject: Optional[str] = None) -> int:
    _ensure_cache_dir()
    count = 0
    # Clear disk
    for fname in os.listdir(CACHE_DIR):
        if not fname.startswith("search_"):
            continue
        path = os.path.join(CACHE_DIR, fname)
        try:
            os.remove(path)
            count += 1
        except Exception:
            pass
    # Clear memory
    with _mem_cache_lock:
        to_del = [k for k in _mem_cache if k.startswith("search::")]
        for k in to_del:
            del _mem_cache[k]
        count += len(to_del)
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Buffered audit trail (100 records per flush, not 1)
# ─────────────────────────────────────────────────────────────────────────────

_audit_dir: Optional[str] = None
_audit_buffers: Dict[str, List[str]] = defaultdict(list)
_audit_lock = threading.Lock()
_AUDIT_FLUSH_EVERY = 100


def set_audit_dir(path: str) -> None:
    global _audit_dir
    _audit_dir = path
    os.makedirs(path, exist_ok=True)


def _audit_log(subject: str, step: str, data: Dict[str, Any]) -> None:
    if _audit_dir is None:
        return
    safe = re.sub(r"[^\w\-.]", "_", subject)[:120]
    subj_dir = os.path.join(_audit_dir, safe)
    os.makedirs(subj_dir, exist_ok=True)
    path = os.path.join(subj_dir, f"{step}.jsonl")
    record = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "subject": subject, "step": step, **data}
    line = json.dumps(record, ensure_ascii=False) + "\n"

    with _audit_lock:
        _audit_buffers[path].append(line)
        if len(_audit_buffers[path]) >= _AUDIT_FLUSH_EVERY:
            _flush_audit_path(path)


def _flush_audit_path(path: str):
    """Must be called with _audit_lock held."""
    lines = _audit_buffers.pop(path, [])
    if lines:
        with open(path, "a", encoding="utf-8") as f:
            f.writelines(lines)


def flush_audit():
    """Flush all buffered audit records to disk."""
    with _audit_lock:
        for path in list(_audit_buffers.keys()):
            _flush_audit_path(path)


# ─────────────────────────────────────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _extract_page_text(html: str) -> str:
    if not html:
        return ""
    if not _has_bs4():
        return _strip(re.sub(r"<[^>]+>", " ", html))
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        try:
            tag.decompose()
        except Exception:
            pass
    main = soup.find("main") or soup.find("article") or soup.body
    txt = main.get_text(" ", strip=True) if main else soup.get_text(" ", strip=True)
    return _strip(txt)


# ─────────────────────────────────────────────────────────────────────────────
# Smart query construction
# ─────────────────────────────────────────────────────────────────────────────

def build_search_queries(subject: str, max_queries: int = 3) -> List[str]:
    clean = subject.strip()
    paren_match = re.match(r"^(.+?)\s*\((.+?)\)\s*$", clean)
    if paren_match:
        short_name = paren_match.group(1).strip()
        long_name = paren_match.group(2).strip()
        full_name = f"{short_name} {long_name}"
    else:
        short_name = clean
        long_name = None
        full_name = clean

    queries = []
    queries.append(f"{full_name} encyclopedia overview facts")
    queries.append(f"{full_name} wikipedia")
    if long_name:
        queries.append(f'"{long_name}" history overview')
    else:
        queries.append(f'"{clean}" biography history overview')
    return queries[:max_queries]


# ─────────────────────────────────────────────────────────────────────────────
# Search backends (all FREE)
# ─────────────────────────────────────────────────────────────────────────────

def _ddg_search(query: str, max_results: int = 20) -> List[Dict[str, str]]:
    if not _has_ddg():
        return []
    cache_key = f"ddg_v2::{query}::{max_results}"
    cached = _load_cache("search", cache_key)
    if isinstance(cached, list) and cached:
        return cached
    try:
        DDGS = _get_ddgs_class()
        out: List[Dict[str, str]] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results, safesearch="off"):
                if not isinstance(r, dict):
                    continue
                url = str(r.get("href") or r.get("link") or r.get("url", "")).strip()
                if not url:
                    continue
                out.append({
                    "title": str(r.get("title", "")).strip(),
                    "url": url,
                    "snippet": str(r.get("body") or r.get("snippet", "")).strip(),
                })
        _save_cache("search", cache_key, out)
        return out
    except Exception:
        return []


def _searxng_search(query: str, max_results: int = 20) -> List[Dict[str, str]]:
    api_base = os.getenv("SEARXNG_API_BASE", "").strip()
    if not api_base:
        return []
    cache_key = f"searxng_v2::{api_base}::{query}::{max_results}"
    cached = _load_cache("search", cache_key)
    if isinstance(cached, list) and cached:
        return cached
    try:
        url = f"{api_base.rstrip('/')}/search"
        params = {"q": query, "format": "json", "categories": "general",
                  "language": "en", "pageno": 1, "time_range": "year"}
        r = _session.get(url, params=params, timeout=TIMEOUT_DEFAULT)
        r.raise_for_status()
        results = r.json().get("results") or []
        out: List[Dict[str, str]] = []
        for it in results[:max_results]:
            if not isinstance(it, dict):
                continue
            out.append({
                "title": str(it.get("title", "")).strip(),
                "url": str(it.get("url", "")).strip(),
                "snippet": str(it.get("content") or it.get("snippet", "")).strip(),
            })
        _save_cache("search", cache_key, out)
        return out
    except Exception:
        return []


def _google_scrape_search(query: str, max_results: int = 10) -> List[Dict[str, str]]:
    cache_key = f"gscrape_v2::{query}::{max_results}"
    cached = _load_cache("search", cache_key)
    if isinstance(cached, list) and cached:
        return cached
    try:
        url = f"https://www.google.com/search?q={quote_plus(query)}&num={max_results}&hl=en"
        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
        r = requests.get(url, headers=headers, timeout=TIMEOUT_DEFAULT)
        if r.status_code != 200 or not _has_bs4():
            return []
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.text, "html.parser")
        out: List[Dict[str, str]] = []
        for g in soup.select("div.g, div[data-sokoban-container]"):
            a = g.find("a", href=True)
            if not a or not a["href"].startswith("http"):
                continue
            sn_el = g.find("div", class_="VwiC3b") or g.find("span", class_="st")
            out.append({
                "title": a.get_text(strip=True),
                "url": a["href"],
                "snippet": sn_el.get_text(strip=True) if sn_el else "",
            })
            if len(out) >= max_results:
                break
        _save_cache("search", cache_key, out)
        return out
    except Exception:
        return []


def _raw_search(query: str, max_results: int = 20) -> List[Dict[str, str]]:
    if SEARCH_BACKEND == "ddg":
        order = ["ddg", "searxng", "google"]
    elif SEARCH_BACKEND == "searxng":
        order = ["searxng", "ddg", "google"]
    elif SEARCH_BACKEND == "google":
        order = ["google", "ddg", "searxng"]
    else:
        order = ["ddg", "searxng", "google"]

    fns = {"ddg": _ddg_search, "searxng": _searxng_search, "google": _google_scrape_search}
    for name in order:
        results = fns[name](query, max_results)
        if results:
            return results
    return []


def web_search(
    subject: str,
    max_results_per_query: int = 15,
    max_queries: int = 3,
    exclude_domains: Optional[Set[str]] = None,
) -> List[Dict[str, str]]:
    """
    High-quality web search — PARALLEL queries (was sequential).

    SPEED FIX: All 3 search queries fire simultaneously.
    Before: 3 × 1-2s = 3-6s sequential
    After:  max(1-2s) = 1-2s parallel
    """
    queries = build_search_queries(subject, max_queries=max_queries)

    # ── PARALLEL search queries (was: sequential for loop) ────────────────────
    all_raw: List[List[Dict[str, str]]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries)) as pool:
        futures = [pool.submit(_raw_search, q, max_results_per_query) for q in queries]
        for fut in concurrent.futures.as_completed(futures):
            try:
                result = fut.result()
                if result:
                    all_raw.append(result)
            except Exception:
                pass

    # Merge & deduplicate
    all_results: List[Dict[str, str]] = []
    seen_urls: Set[str] = set()
    for raw_list in all_raw:
        for r in raw_list:
            url = r.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                all_results.append(r)

    # Filter blocked + excluded domains
    def _should_exclude(url: str) -> bool:
        if _is_blocked(url):
            return True
        if exclude_domains:
            d = _domain(url)
            rd = _root_domain(d)
            if d in exclude_domains or rd in exclude_domains:
                return True
        return False

    filtered = [r for r in all_results if not _should_exclude(r.get("url", ""))]
    filtered.sort(key=lambda r: _domain_quality_score(r.get("url", "")), reverse=True)

    _audit_log(subject, "search", {
        "queries": queries,
        "raw_count": len(all_results),
        "blocked_count": len(all_results) - len(filtered),
        "filtered_count": len(filtered),
        "exclude_domains": sorted(exclude_domains) if exclude_domains else [],
        "top_urls": [r.get("url", "") for r in filtered[:10]],
        "top_domains": [_domain(r.get("url", "")) for r in filtered[:10]],
        "top_quality": [_domain_quality_score(r.get("url", "")) for r in filtered[:10]],
    })

    return filtered


# ─────────────────────────────────────────────────────────────────────────────
# Page fetching (parallelised with pooled connections)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_url_text(url: str, timeout: float = TIMEOUT_DEFAULT, max_chars: int = 20000) -> str:
    cache_key = f"page_v2::{url}"
    cached = _load_cache("page", cache_key)
    if isinstance(cached, dict) and isinstance(cached.get("text"), str):
        return cached["text"]
    try:
        r = _session.get(url, timeout=timeout, allow_redirects=True)
        ct = (r.headers.get("content-type") or "").lower()
        if not any(t in ct for t in ["text/html", "application/xhtml", "text/plain"]):
            _save_cache("page", cache_key, {"text": "", "status": r.status_code})
            return ""
        text = _extract_page_text(r.text or "")[:max_chars]
        _save_cache("page", cache_key, {"text": text, "status": r.status_code, "url": url})
        return text
    except Exception:
        _save_cache("page", cache_key, {"text": "", "error": "fetch_failed"})
        return ""


def _fetch_pages_parallel(
    urls: List[str],
    timeout: float = TIMEOUT_DEFAULT,
    max_chars: int = 20000,
    max_workers: int = 8,
) -> Dict[str, str]:
    if not urls:
        return {}

    results: Dict[str, str] = {}
    to_fetch: List[str] = []
    for url in urls:
        cache_key = f"page_v2::{url}"
        cached = _load_cache("page", cache_key)
        if isinstance(cached, dict) and isinstance(cached.get("text"), str):
            results[url] = cached["text"]
        else:
            to_fetch.append(url)

    if not to_fetch:
        return results

    workers = min(len(to_fetch), max_workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(fetch_url_text, url, timeout, max_chars): url
            for url in to_fetch
        }
        for future in concurrent.futures.as_completed(future_map):
            url = future_map[future]
            try:
                results[url] = future.result()
            except Exception:
                results[url] = ""

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def web_evidence_snippets(
    query: str,
    max_snippets: int = 5,
    fetch_pages: bool = True,
    max_fetch_pages: int = 0,
    snippet_max_chars: int = 1500,
    exclude_domains: Optional[Set[str]] = None,
) -> List[Dict[str, str]]:
    """
    Search web for a subject, fetch top-k unique-domain pages, return evidence.
    """
    max_snippets = int(max_snippets or MAX_SITES_DEFAULT)
    if max_fetch_pages <= 0:
        max_fetch_pages = MAX_FETCH_PAGES_DEFAULT

    results = web_search(query, max_results_per_query=15, max_queries=3,
                         exclude_domains=exclude_domains)
    if not results:
        return []

    picked: List[Dict[str, str]] = []
    seen_domains: Set[str] = set()
    for r in results:
        url = r.get("url", "")
        d = _root_domain(_domain(url))
        if not url or not d or d in seen_domains:
            continue
        seen_domains.add(d)
        picked.append(r)
        if len(picked) >= max_snippets:
            break

    if not picked:
        return []

    urls_to_fetch: List[str] = []
    if fetch_pages:
        for it in picked[:max_fetch_pages]:
            urls_to_fetch.append(it.get("url", ""))

    page_texts: Dict[str, str] = {}
    if urls_to_fetch:
        page_texts = _fetch_pages_parallel(
            urls_to_fetch,
            timeout=TIMEOUT_DEFAULT,
            max_chars=snippet_max_chars * 4,
            max_workers=min(len(urls_to_fetch), 8),
        )

    out: List[Dict[str, str]] = []
    for rank, it in enumerate(picked, 1):
        url = it.get("url", "")
        quality = _domain_quality_score(url)
        fetched = url in page_texts and bool(page_texts.get(url, ""))

        if fetched:
            page_text = page_texts[url]
            snippet_text = page_text[:snippet_max_chars]
            page_chars = len(page_text)
        else:
            snippet_text = it.get("snippet") or ""
            page_chars = 0

        record = {
            "title": it.get("title", "") or _domain(url),
            "url": url,
            "snippet_text": _strip(snippet_text),
            "domain": _domain(url),
            "page_chars": str(page_chars),
            "quality_score": str(int(quality)),
            "quality_label": _quality_label(quality),
            "source_rank": str(rank),
            "fetched_page": str(fetched).lower(),
        }
        out.append(record)

        _audit_log(query, "evidence_result", {
            "rank": rank, "url": url, "domain": _domain(url),
            "root_domain": _root_domain(_domain(url)),
            "quality_score": quality, "quality_label": _quality_label(quality),
            "fetched_page": fetched, "page_chars": page_chars,
            "snippet_len": len(snippet_text),
        })

    return out


def web_baseline_text(
    query: str,
    max_snippets: int = 10,
    max_fetch_pages: int = 0,
    snippet_max_chars: int = 2000,
    exclude_domains: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    if max_fetch_pages <= 0:
        max_fetch_pages = min(max_snippets, max(MAX_FETCH_PAGES_DEFAULT, 5))

    snips = web_evidence_snippets(
        query, max_snippets=max_snippets, fetch_pages=True,
        max_fetch_pages=max_fetch_pages, snippet_max_chars=snippet_max_chars,
        exclude_domains=exclude_domains,
    )
    texts = [s.get("snippet_text", "") for s in snips if s.get("snippet_text")]
    text = "\n\n".join(texts).strip()

    return {
        "text": text, "snippets": snips,
        "meta": {
            "sources": snips, "n_sources": len(snips),
            "domains": [s.get("domain", "") for s in snips],
            "quality_scores": [s.get("quality_score", "0") for s in snips],
            "quality_labels": [s.get("quality_label", "") for s in snips],
            "n_fetched_pages": sum(1 for s in snips if s.get("fetched_page") == "true"),
        },
        "found": bool(text),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def check_backends() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "ddg_available": _has_ddg(),
        "bs4_available": _has_bs4(),
        "searxng_api_base": os.getenv("SEARXNG_API_BASE", ""),
        "cache_dir": CACHE_DIR,
        "search_backend": SEARCH_BACKEND,
        "timeout": TIMEOUT_DEFAULT,
        "max_fetch_pages": MAX_FETCH_PAGES_DEFAULT,
        "blocked_domains_count": len(BLOCKED_DOMAINS),
        "preferred_domains_count": len(_DOMAIN_SCORES),
    }
    if _has_ddg():
        try:
            r = _ddg_search("Albert Einstein biography", max_results=3)
            info["ddg_test"] = "ok" if r else "no_results"
            if r:
                info["ddg_test_domains"] = [_domain(x.get("url", "")) for x in r[:3]]
        except Exception as e:
            info["ddg_test"] = f"error: {e}"
    if info["searxng_api_base"]:
        try:
            r = _searxng_search("Albert Einstein biography", max_results=3)
            info["searxng_test"] = "ok" if r else "no_results"
        except Exception as e:
            info["searxng_test"] = f"error: {e}"
    return info


def demo_search(subject: str, max_snippets: int = 5) -> None:
    print(f"\n{'=' * 70}")
    print(f"Subject: {subject}")
    print(f"{'=' * 70}")

    queries = build_search_queries(subject)
    print(f"\nGenerated queries:")
    for i, q in enumerate(queries, 1):
        print(f"  {i}. {q}")

    print(f"\nSearching (max {max_snippets} snippets, max {MAX_FETCH_PAGES_DEFAULT} page fetches)...")
    t0 = time.time()
    snips = web_evidence_snippets(subject, max_snippets=max_snippets)
    elapsed = time.time() - t0

    if not snips:
        print("  No results found!")
        return

    print(f"\nFound {len(snips)} results in {elapsed:.1f}s:\n")
    for i, s in enumerate(snips, 1):
        qs = int(s.get("quality_score", 0))
        ql = s.get("quality_label", "?")
        fetched = s.get("fetched_page", "false") == "true"
        fetch_tag = "📄 full page" if fetched else "📝 snippet only"
        print(f"  #{i}  [{qs:3d}/100 {ql:10s}]  {fetch_tag}")
        print(f"      Domain:  {s.get('domain', '')}")
        print(f"      Title:   {s.get('title', '')[:80]}")
        print(f"      URL:     {s.get('url', '')}")
        if fetched:
            print(f"      Chars:   {s.get('page_chars', '0')}")
        txt = s.get("snippet_text", "")
        print(f"      Text:    {txt[:200]}{'...' if len(txt) > 200 else ''}")
        print()

    flush_audit()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        print(json.dumps(check_backends(), indent=2))
    elif len(sys.argv) > 1 and sys.argv[1] == "demo":
        subjects = sys.argv[2:] if len(sys.argv) > 2 else [
            "EMD (Electro-Motive Diesel)",
            "Disabled American Veterans",
            "Albert Einstein",
        ]
        for s in subjects:
            demo_search(s)
    elif len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        demo_search(query)
    else:
        print("Usage:")
        print("  python web_evidence.py check              # check backends")
        print("  python web_evidence.py demo               # test with sample subjects")
        print("  python web_evidence.py <subject name>     # search one subject")
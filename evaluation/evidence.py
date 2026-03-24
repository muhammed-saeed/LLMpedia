#!/usr/bin/env python3
"""
evidence.py — Web evidence retrieval for LLMpedia factuality evaluation.

KEY CHANGES vs original:
  - Serper API backend (cheap: ~$0.001/search vs Brave $0.005)
  - "single" web_mode: 1 search → pick best quality URL → fetch full page → done
  - Two-layer cache:
      Layer 1 (search cache):  subject → search results JSON  (never re-search)
      Layer 2 (page cache):    URL → fetched page text        (never re-fetch)
  - Cache is disk-backed SQLite-style (JSON files, atomic writes)
  - min_quality_score default raised to 60 so you only fetch credible pages

PATCH v2:
  - Expanded DOMAIN_SCORES with ~60 more high-quality reference sources
  - CAPTCHA / access-denied / paywall detection: blocked pages are NOT cached
    as valid content — they get a special "blocked" method tag so they will be
    retried on the next run (once your IP is unblocked or you switch proxies).
  - Added `repair_page_cache()` utility to scan existing cache and purge
    entries that contain blocked/CAPTCHA content.
  - JSTOR, Britannica, and other sites that frequently CAPTCHA are handled
    gracefully: the fetch falls back to the search snippet instead of caching
    garbage text.

PATCH v3:
  - REMOVED topic_context from search query building.
    Evidence is now retrieved based ONLY on the subject name.
    No topic disambiguation is injected into web searches.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    print("[evidence] WARNING: beautifulsoup4 not installed. pip install beautifulsoup4")


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EvidenceConfig:
    # ── mode ─────────────────────────────────────────────────────────────────
    web_mode: str = "single"

    # ── single-mode settings ──────────────────────────────────────────────────
    web_max_snippets: int = 1
    web_max_fetch_pages: int = 1
    web_snippet_max_chars: int = 4000
    web_timeout: float = 12.0

    # ── quality filter ────────────────────────────────────────────────────────
    min_quality_score: float = 60.0

    # ── search backends ───────────────────────────────────────────────────────
    search_backend: str = "auto"
    searxng_api_base: str = ""

    # ── exclusions ───────────────────────────────────────────────────────────
    web_exclude_domains: Set[str] = field(default_factory=lambda: {
        "wikipedia.org", "en.wikipedia.org",
        "wikidata.org", "www.wikidata.org",
    })
    web_extra_exclude: str = ""

    # ── workers ──────────────────────────────────────────────────────────────
    web_fetch_workers: int = 2

    # ── CACHE SETTINGS ────────────────────────────────────────────────────────
    cache_dir: str = ""
    cache_ttl_hours: float = 720.0

    # ── misc ─────────────────────────────────────────────────────────────────
    audit_dir: str = ""
    debug: bool = False


@dataclass
class EvidenceResult:
    source: str
    found: bool
    text: str
    snippets: List[Dict[str, Any]]
    url: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# DOMAIN SCORING  (EXPANDED — v2)
# ──────────────────────────────────────────────────────────────────────────────

DOMAIN_SCORES: Dict[str, int] = {
    # ── Encyclopedias & Reference ─────────────────────────────────────────
    "britannica.com": 100,
    "worldhistory.org": 90,
    "encyclopedia.com": 85,
    "scholarpedia.org": 88,
    "plato.stanford.edu": 95,
    "iep.utm.edu": 88,
    "newworldencyclopedia.org": 75,

    # ── Government & Archives ─────────────────────────────────────────────
    "loc.gov": 97,
    "archives.gov": 97,
    "congress.gov": 95,
    "usa.gov": 93,
    "cia.gov": 90,
    "state.gov": 90,
    "whitehouse.gov": 90,
    "nasa.gov": 95,
    "nih.gov": 94,
    "cdc.gov": 93,
    "fda.gov": 90,
    "epa.gov": 89,
    "noaa.gov": 90,
    "usgs.gov": 90,
    "nps.gov": 85,
    "si.edu": 92,
    "parliament.uk": 88,
    "gov.uk": 87,
    "europarl.europa.eu": 85,
    "un.org": 88,
    "who.int": 90,
    "worldbank.org": 88,
    "imf.org": 87,

    # ── Wire services & Top news ──────────────────────────────────────────
    "reuters.com": 94,
    "apnews.com": 94,
    "bbc.com": 93,
    "bbc.co.uk": 93,
    "nytimes.com": 92,
    "washingtonpost.com": 90,
    "theguardian.com": 91,
    "economist.com": 90,
    "ft.com": 89,
    "wsj.com": 89,
    "theatlantic.com": 87,
    "newyorker.com": 87,
    "npr.org": 88,
    "pbs.org": 88,
    "aljazeera.com": 85,
    "dw.com": 84,
    "france24.com": 83,

    # ── Academic journals & Databases ─────────────────────────────────────
    "nature.com": 95,
    "science.org": 95,
    "sciencedirect.com": 90,
    "springer.com": 89,
    "link.springer.com": 89,
    "wiley.com": 88,
    "onlinelibrary.wiley.com": 88,
    "tandfonline.com": 87,
    "cell.com": 93,
    "thelancet.com": 93,
    "bmj.com": 92,
    "nejm.org": 94,
    "pnas.org": 92,
    "ncbi.nlm.nih.gov": 92,
    "pubmed.ncbi.nlm.nih.gov": 92,
    "jstor.org": 90,
    "arxiv.org": 85,
    "ssrn.com": 82,
    "researchgate.net": 75,
    "scholar.google.com": 80,
    "doaj.org": 80,
    "semanticscholar.org": 82,
    "ieee.org": 89,
    "ieeexplore.ieee.org": 89,
    "acm.org": 88,
    "dl.acm.org": 88,

    # ── Museums, Libraries & Cultural institutions ────────────────────────
    "smithsonianmag.com": 87,
    "nationalgeographic.com": 86,
    "metmuseum.org": 90,
    "moma.org": 87,
    "nga.gov": 88,
    "bl.uk": 90,
    "bnf.fr": 88,
    "dpla.org": 83,
    "europeana.eu": 83,

    # ── History & Biography ───────────────────────────────────────────────
    "history.com": 80,
    "biography.com": 75,
    "oxforddnb.com": 90,
    "anb.org": 88,
    "historytoday.com": 78,
    "historyextra.com": 77,

    # ── Science & Technology ──────────────────────────────────────────────
    "scientificamerican.com": 86,
    "newscientist.com": 83,
    "livescience.com": 75,
    "space.com": 76,
    "phys.org": 78,
    "sciencenews.org": 82,
    "quantamagazine.org": 88,
    "arstechnica.com": 80,
    "spectrum.ieee.org": 84,
    "technologyreview.com": 84,

    # ── Education & Research orgs ─────────────────────────────────────────
    "mit.edu": 93,
    "stanford.edu": 93,
    "harvard.edu": 93,
    "ox.ac.uk": 92,
    "cam.ac.uk": 92,
    "berkeley.edu": 91,
    "caltech.edu": 91,
    "yale.edu": 91,
    "princeton.edu": 91,
    "columbia.edu": 90,
    "uchicago.edu": 90,
    "cornell.edu": 89,
    "cmu.edu": 89,
    "ethz.ch": 89,
    "mpg.de": 89,
    "khanacademy.org": 80,

    # ── Fact-checking & Data ──────────────────────────────────────────────
    "snopes.com": 82,
    "factcheck.org": 84,
    "politifact.com": 80,
    "ourworldindata.org": 88,
    "statista.com": 78,
    "data.gov": 88,
    "census.gov": 90,
    "bls.gov": 89,
    "bea.gov": 88,

    # ── Legal ─────────────────────────────────────────────────────────────
    "law.cornell.edu": 90,
    "supremecourt.gov": 92,
    "courtlistener.com": 80,
    "oyez.org": 82,

    # ── Medical & Health ──────────────────────────────────────────────────
    "mayoclinic.org": 88,
    "clevelandclinic.org": 85,
    "webmd.com": 72,
    "medlineplus.gov": 90,
    "hopkinsmedicine.org": 87,
    "uptodate.com": 88,
}

BLOCKED_DOMAINS: Set[str] = {
    "wikipedia.org", "en.wikipedia.org", "en.m.wikipedia.org",
    "wikidata.org", "www.wikidata.org", "wikimedia.org",
    "google.com", "bing.com", "duckduckgo.com", "yahoo.com",
    "facebook.com", "instagram.com", "twitter.com", "x.com", "tiktok.com",
    "reddit.com", "quora.com", "youtube.com", "pinterest.com",
    "amazon.com", "ebay.com", "perplexity.ai",
}

def _domain(url: str) -> str:
    try: return urlparse(url).netloc.lower()
    except: return ""

def _root_domain(domain: str) -> str:
    parts = domain.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else domain

def _domain_quality_score(url: str) -> float:
    d = _domain(url); rd = _root_domain(d)
    score = DOMAIN_SCORES.get(d) or DOMAIN_SCORES.get(rd)
    if score is not None: return float(score)
    for tld in (".edu", ".gov", ".org"):
        if d.endswith(tld): return 70.0
    return 35.0 if url.startswith("https") else 30.0

def _quality_label(score: float) -> str:
    if score >= 95: return "gold"
    if score >= 85: return "excellent"
    if score >= 75: return "good"
    if score >= 60: return "acceptable"
    if score >= 40: return "fair"
    return "low"

def _should_exclude(url: str, exclude_domains: Set[str]) -> bool:
    d = _domain(url); rd = _root_domain(d)
    if d in BLOCKED_DOMAINS or rd in BLOCKED_DOMAINS: return True
    for excl in (exclude_domains or set()):
        if d == excl or rd == excl or d.endswith(excl): return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# BLOCKED / CAPTCHA / PAYWALL CONTENT DETECTION  (v2)
# ──────────────────────────────────────────────────────────────────────────────

_BLOCKED_PATTERNS = [
    re.compile(r"complete\s+this\s+recaptcha", re.I),
    re.compile(r"detected\s+unusual\s+traffic", re.I),
    re.compile(r"please\s+verify\s+you\s+are\s+(a\s+)?human", re.I),
    re.compile(r"are\s+you\s+a\s+robot", re.I),
    re.compile(r"captcha", re.I),
    re.compile(r"bot\s+detection", re.I),
    re.compile(r"automated\s+access", re.I),
    re.compile(r"browser\s+check", re.I),
    re.compile(r"just\s+a\s+moment.*cloudflare", re.I | re.DOTALL),
    re.compile(r"enable\s+javascript\s+and\s+cookies\s+to\s+continue", re.I),
    re.compile(r"checking\s+(if\s+the\s+site\s+connection\s+is\s+secure|your\s+browser)", re.I),
    re.compile(r"attention\s+required.*cloudflare", re.I | re.DOTALL),
    re.compile(r"access\s+denied", re.I),
    re.compile(r"403\s+forbidden", re.I),
    re.compile(r"you\s+don.?t\s+have\s+permission", re.I),
    re.compile(r"subscribe\s+to\s+(continue\s+reading|read|unlock)", re.I),
    re.compile(r"this\s+(content|article)\s+is\s+(for|available\s+to)\s+(subscribers|members)", re.I),
    re.compile(r"sign\s+in\s+to\s+continue\s+reading", re.I),
    re.compile(r"rate\s+limit(ed)?", re.I),
    re.compile(r"too\s+many\s+requests", re.I),
    re.compile(r"block\s+reference.*go\s+back\s+to", re.I | re.DOTALL),
]

_MIN_USEFUL_CHARS = 200


def _is_blocked_content(text: str) -> bool:
    if not text:
        return False
    check = text[:2000]
    for pattern in _BLOCKED_PATTERNS:
        if pattern.search(check):
            return True
    if len(text.strip()) < _MIN_USEFUL_CHARS:
        sentence_ends = len(re.findall(r'[.!?]\s', text))
        if sentence_ends < 2:
            return True
    return False


def _blocked_reason(text: str) -> str:
    if not text:
        return "empty"
    check = text[:2000].lower()
    if "recaptcha" in check or "captcha" in check:
        return "captcha"
    if "unusual traffic" in check or "detected unusual" in check:
        return "bot-detection"
    if "cloudflare" in check:
        return "cloudflare"
    if "access denied" in check or "403 forbidden" in check:
        return "access-denied"
    if "subscribe" in check and ("continue reading" in check or "unlock" in check):
        return "paywall"
    if "block reference" in check:
        return "ip-blocked"
    if "rate limit" in check or "too many requests" in check:
        return "rate-limited"
    if len(text.strip()) < _MIN_USEFUL_CHARS:
        return "too-short"
    return "blocked-generic"


# ──────────────────────────────────────────────────────────────────────────────
# ROBUST DISK CACHE
# ──────────────────────────────────────────────────────────────────────────────

class DiskCache:
    def __init__(self, cache_dir: str, ttl_hours: float = 720.0):
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_hours * 3600
        self._mem: Dict[str, Any] = {}
        self._lock = threading.Lock()
        if cache_dir:
            for sub in ("search", "pages", "evidence"):
                os.makedirs(os.path.join(cache_dir, sub), exist_ok=True)

    def _path(self, namespace: str, key: str) -> str:
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, namespace, h + ".json")

    def get(self, namespace: str, key: str) -> Optional[Any]:
        mem_key = f"{namespace}::{key}"
        with self._lock:
            if mem_key in self._mem:
                return self._mem[mem_key]
        if not self.cache_dir:
            return None
        path = self._path(namespace, key)
        if not os.path.exists(path):
            return None
        try:
            age = time.time() - os.path.getmtime(path)
            if age > self.ttl_seconds:
                return None
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            with self._lock:
                self._mem[mem_key] = obj
            return obj
        except Exception:
            return None

    def set(self, namespace: str, key: str, value: Any) -> None:
        mem_key = f"{namespace}::{key}"
        with self._lock:
            self._mem[mem_key] = value
        if not self.cache_dir:
            return
        path = self._path(namespace, key)
        tmp = path + f".tmp{os.getpid()}"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(value, f, ensure_ascii=False)
            os.replace(tmp, path)
        except Exception as e:
            print(f"[cache] write error ({namespace}/{key[:20]}): {e}")
            try: os.remove(tmp)
            except: pass

    def delete(self, namespace: str, key: str) -> bool:
        mem_key = f"{namespace}::{key}"
        removed = False
        with self._lock:
            if mem_key in self._mem:
                del self._mem[mem_key]
                removed = True
        if self.cache_dir:
            path = self._path(namespace, key)
            if os.path.exists(path):
                try:
                    os.remove(path)
                    removed = True
                except Exception:
                    pass
        return removed

    def stats(self) -> Dict[str, int]:
        out = {}
        if not self.cache_dir:
            return out
        for ns in ("search", "pages", "evidence"):
            d = os.path.join(self.cache_dir, ns)
            out[ns] = len(os.listdir(d)) if os.path.isdir(d) else 0
        return out


# ──────────────────────────────────────────────────────────────────────────────
# HTTP SESSION
# ──────────────────────────────────────────────────────────────────────────────

_session: Optional[requests.Session] = None
_session_lock = threading.Lock()

def _get_session() -> requests.Session:
    global _session
    with _session_lock:
        if _session is None:
            s = requests.Session()
            s.headers["User-Agent"] = (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            try:
                from urllib3.util.retry import Retry
                retry = Retry(total=2, backoff_factor=0.5,
                              status_forcelist=[500, 502, 503, 504],
                              allowed_methods=["GET", "POST"])
                ad = HTTPAdapter(max_retries=retry,
                                 pool_connections=10, pool_maxsize=10)
                s.mount("https://", ad)
                s.mount("http://", ad)
            except Exception:
                pass
            _session = s
    return _session


# ──────────────────────────────────────────────────────────────────────────────
# PAGE FETCHING  (v2: blocked-content detection)
# ──────────────────────────────────────────────────────────────────────────────

def _extract_text_bs4(html: str) -> str:
    if not html or not HAS_BS4:
        return _extract_text_regex(html)
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "header",
                          "footer", "nav", "aside", "form"]):
            tag.decompose()
        main = (soup.find("article") or soup.find("main")
                or soup.find(id="content") or soup.body)
        txt = main.get_text(" ", strip=True) if main else soup.get_text(" ", strip=True)
        return re.sub(r"\s+", " ", txt).strip()
    except Exception:
        return _extract_text_regex(html)

def _extract_text_regex(html: str) -> str:
    if not html: return ""
    text = re.sub(r"<(script|style|noscript)[^>]*>.*?</\1>", " ", html,
                  flags=re.DOTALL | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    for ent, ch in [("&amp;","&"),("&lt;","<"),("&gt;",">"),
                    ("&quot;",'"'),("&#39;","'"),("&nbsp;"," ")]:
        text = text.replace(ent, ch)
    return re.sub(r"\s+", " ", text).strip()

def _fetch_page(url: str, cfg: EvidenceConfig, cache: DiskCache,
                max_chars: int) -> Tuple[str, str]:
    cached = cache.get("pages", url)
    if isinstance(cached, dict) and "text" in cached:
        cached_text = cached["text"]
        cached_method = cached.get("method", "")
        if cached_method.startswith("blocked"):
            pass
        elif not _is_blocked_content(cached_text):
            return cached_text, "cache"
        else:
            reason = _blocked_reason(cached_text)
            print(f"[cache] Purging blocked content for {url[:80]} "
                  f"(reason={reason})", flush=True)
            cache.delete("pages", url)

    text = ""; method = "none"
    try:
        r = _get_session().get(url, timeout=cfg.web_timeout, allow_redirects=True)
        ct = (r.headers.get("content-type") or "").lower()
        if not any(t in ct for t in ["text/html", "application/xhtml", "text/plain"]):
            cache.set("pages", url, {"text": "", "method": "non-html"})
            return "", "non-html"
        html = r.text or ""
        text = _extract_text_bs4(html) if HAS_BS4 else _extract_text_regex(html)
        method = "requests+bs4" if text else "requests-empty"
    except requests.Timeout:
        method = "timeout"
    except requests.ConnectionError:
        method = "conn-error"
    except Exception as e:
        method = f"error:{type(e).__name__}"

    text = (text or "")[:max_chars]

    if text and _is_blocked_content(text):
        reason = _blocked_reason(text)
        method = f"blocked-{reason}"
        if cfg.debug:
            print(f"[evidence] Blocked content from {url[:80]}: {reason} "
                  f"(NOT caching, will retry next run)", flush=True)
        return "", method

    cache.set("pages", url, {"text": text, "method": method})
    return text, method


# ──────────────────────────────────────────────────────────────────────────────
# SEARCH BACKENDS
# ──────────────────────────────────────────────────────────────────────────────

def _serper_search(query: str, cache: DiskCache,
                   max_results: int = 10) -> List[Dict[str, str]]:
    api_key = os.getenv("SERPER_API_KEY", "").strip()
    if not api_key:
        print("[evidence] SERPER_API_KEY not set. Set it in .env or environment.",
              flush=True)
        return []

    cache_key = f"serper::{query}"
    cached = cache.get("search", cache_key)
    if cached is not None:
        return cached

    _SERPER_EXCLUDE = [
        "wikipedia.org", "reddit.com", "quora.com", "youtube.com",
        "pinterest.com", "facebook.com", "twitter.com", "x.com",
        "instagram.com", "tiktok.com", "amazon.com", "ebay.com",
    ]
    exclude_str = " ".join(f"-site:{d}" for d in _SERPER_EXCLUDE)
    effective_query = f"{query} {exclude_str}"

    try:
        r = _get_session().post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": api_key,
                "Content-Type": "application/json",
            },
            json={"q": effective_query, "gl": "us", "hl": "en", "num": max_results},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[evidence] Serper error: {e}", flush=True)
        return []

    out = []
    for item in (data.get("organic") or [])[:max_results]:
        url = (item.get("link") or "").strip()
        if url:
            out.append({
                "title":   (item.get("title") or "").strip(),
                "url":     url,
                "snippet": (item.get("snippet") or "").strip(),
                "backend": "serper",
            })
    kg = data.get("knowledgeGraph") or {}
    kg_url = (kg.get("website") or "").strip()
    if kg_url and not any(r["url"] == kg_url for r in out):
        out.insert(0, {
            "title":   (kg.get("title") or "").strip(),
            "url":     kg_url,
            "snippet": (kg.get("description") or "").strip(),
            "backend": "serper-kg",
        })

    cache.set("search", cache_key, out)
    return out


def _brave_search(query: str, cache: DiskCache,
                  max_results: int = 10) -> List[Dict[str, str]]:
    api_key = os.getenv("BRAVE_API_KEY", "").strip()
    if not api_key:
        return []

    cache_key = f"brave::{query}"
    cached = cache.get("search", cache_key)
    if cached is not None:
        return cached

    for attempt in range(3):
        try:
            r = _get_session().get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip",
                    "X-Subscription-Token": api_key,
                },
                params={"q": query, "count": min(max_results, 20),
                        "text_decorations": "false", "search_lang": "en"},
                timeout=20,
            )
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            results = []
            for item in ((r.json().get("web") or {}).get("results") or [])[:max_results]:
                url = (item.get("url") or "").strip()
                if url:
                    desc = re.sub(r"<[^>]+>", "", item.get("description") or "")
                    results.append({"title": (item.get("title") or "").strip(),
                                    "url": url, "snippet": desc.strip(),
                                    "backend": "brave"})
            cache.set("search", cache_key, results)
            return results
        except Exception as e:
            if attempt == 2:
                print(f"[evidence] Brave error: {e}", flush=True)
            time.sleep(2 * (attempt + 1))
    return []


def _ddg_search(query: str, cache: DiskCache,
                max_results: int = 10) -> List[Dict[str, str]]:
    cache_key = f"ddg::{query}"
    cached = cache.get("search", cache_key)
    if cached is not None:
        return cached

    try:
        from ddgs import DDGS
    except ImportError:
        print("[evidence] No DDG package. pip install duckduckgo-search")
        return []

    for attempt in range(3):
        try:
            out = []
            with DDGS(timeout=60) as ddgs:
                for r in ddgs.text(query, max_results=max_results, safesearch="off"):
                    url = str(r.get("href") or r.get("link") or r.get("url","")).strip()
                    if url:
                        out.append({"title": str(r.get("title","")).strip(),
                                    "url": url,
                                    "snippet": str(r.get("body") or r.get("snippet","")).strip(),
                                    "backend": "ddg"})
            cache.set("search", cache_key, out)
            return out
        except Exception as e:
            err = str(e)
            if "429" in err or "Too Many" in err:
                time.sleep(8 * (attempt + 1))
            elif attempt < 2:
                time.sleep(3 * (attempt + 1))
            else:
                return []
    return []


def _valyu_search(query: str, cache: DiskCache,
                  cfg: "EvidenceConfig",
                  max_results: int = 10) -> List[Dict[str, str]]:
    api_key = os.getenv("VALYU_API_KEY", "").strip()
    if not api_key:
        print("[evidence] VALYU_API_KEY not set. Set it in .env or environment.",
              flush=True)
        return []

    try:
        from valyu import Valyu
    except ImportError:
        print("[evidence] valyu SDK not installed. pip install valyu", flush=True)
        return []

    cache_key = f"valyu::{query}"
    cached = cache.get("search", cache_key)
    if cached is not None:
        return cached

    excluded = list(BLOCKED_DOMAINS)
    if cfg and cfg.web_exclude_domains:
        excluded.extend(cfg.web_exclude_domains)
    if cfg and cfg.web_extra_exclude:
        for d in cfg.web_extra_exclude.split(","):
            d = d.strip().lower()
            if d:
                excluded.append(d)
    excluded = sorted(set(excluded))

    try:
        client = Valyu(api_key=api_key)
        response = client.search(
            query,
            search_type="all",
            max_num_results=max_results,
            relevance_threshold=0.9,
            **({"excluded_sources": excluded} if excluded else {}),
        )
    except Exception as e:
        print(f"[evidence] Valyu error: {e}", flush=True)
        return []

    out = []
    results_list = getattr(response, "results", None) or []
    for item in results_list[:max_results]:
        url = (getattr(item, "url", "") or "").strip()
        if not url:
            if isinstance(item, dict):
                url = (item.get("url") or "").strip()
        if url:
            title = (getattr(item, "title", "") or "").strip()
            content = (getattr(item, "content", "") or "").strip()
            if isinstance(item, dict):
                title = title or (item.get("title") or "").strip()
                content = content or (item.get("content") or
                                     item.get("snippet") or "").strip()
            out.append({
                "title":   title,
                "url":     url,
                "snippet": content[:500],
                "backend": "valyu",
            })

    if cfg and cfg.debug:
        print(f"[evidence] Valyu: {len(out)} results for '{query[:60]}' "
              f"(excluded {len(excluded)} domains)", flush=True)

    cache.set("search", cache_key, out)
    return out


def _web_search(query: str, cfg: EvidenceConfig,
                cache: DiskCache) -> List[Dict[str, str]]:
    if cfg.search_backend == "valyu":
        return _valyu_search(query, cache, cfg)
    if cfg.search_backend == "serper":
        return _serper_search(query, cache)
    if cfg.search_backend == "brave":
        return _brave_search(query, cache)
    if cfg.search_backend == "ddg":
        return _ddg_search(query, cache)
    if cfg.search_backend == "searxng" and cfg.searxng_api_base:
        return _searxng_search(query, cfg.searxng_api_base, cache)
    # auto: valyu > serper > brave > ddg
    if os.getenv("VALYU_API_KEY", "").strip():
        return _valyu_search(query, cache, cfg)
    if os.getenv("SERPER_API_KEY", "").strip():
        return _serper_search(query, cache)
    if os.getenv("BRAVE_API_KEY", "").strip():
        return _brave_search(query, cache)
    return _ddg_search(query, cache)


def _searxng_search(query: str, api_base: str,
                    cache: DiskCache) -> List[Dict[str, str]]:
    cache_key = f"searxng::{query}"
    cached = cache.get("search", cache_key)
    if cached is not None:
        return cached
    try:
        r = _get_session().get(
            f"{api_base.rstrip('/')}/search",
            params={"q": query, "format": "json", "categories": "general",
                    "language": "en", "pageno": 1},
            timeout=10,
        )
        r.raise_for_status()
        out = [{"title": str(it.get("title","")).strip(),
                "url": str(it.get("url","")).strip(),
                "snippet": str(it.get("content") or it.get("snippet","")).strip(),
                "backend": "searxng"}
               for it in (r.json().get("results") or [])
               if isinstance(it, dict) and it.get("url")]
        cache.set("search", cache_key, out)
        return out
    except Exception as e:
        print(f"[evidence] SearXNG error: {e}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# QUERY BUILDER
#
# CHANGE (v3): Removed topic_context parameter entirely.
# Evidence is retrieved based ONLY on the subject name.
# ──────────────────────────────────────────────────────────────────────────────

def build_search_query(subject: str) -> str:
    """
    Single focused query for one subject.
    No topic context — just search for the subject directly.
    """
    clean = subject.strip()

    # "Subject (disambiguation)" → "Subject disambiguation"
    paren = re.match(r"^(.+?)\s*\((.+?)\)\s*$", clean)
    if paren:
        short = paren.group(1).strip()
        disambig = paren.group(2).strip()
        base = f"{short} {disambig}"
    else:
        base = clean

    return f'"{base}" history facts encyclopedia'


# ──────────────────────────────────────────────────────────────────────────────
# EVIDENCE RETRIEVER
#
# CHANGE (v3): Removed topic_context from .get() and _get_web().
# ──────────────────────────────────────────────────────────────────────────────

class EvidenceRetriever:
    def __init__(self, cfg: EvidenceConfig):
        self.cfg     = cfg
        self._cache  = DiskCache(cfg.cache_dir, cfg.cache_ttl_hours)
        self._lock   = threading.Lock()
        self._result_cache: Dict[Tuple[str, str], EvidenceResult] = {}
        self._exclude = set(cfg.web_exclude_domains)
        if cfg.web_extra_exclude:
            for d in cfg.web_extra_exclude.split(","):
                d = d.strip().lower()
                if d: self._exclude.add(d)

        if cfg.debug and cfg.cache_dir:
            stats = self._cache.stats()
            print(f"[evidence] Cache loaded: search={stats.get('search',0)} "
                  f"pages={stats.get('pages',0)} evidence={stats.get('evidence',0)}")

    def clear_memory_cache(self):
        with self._lock:
            self._result_cache.clear()

    def cache_stats(self) -> Dict[str, int]:
        return self._cache.stats()

    def get(self, source: str, subject: str, *,
            claims: Optional[List[str]] = None) -> EvidenceResult:
        """Retrieve evidence for a subject. No topic_context — subject only."""

        key = (subject, source)
        with self._lock:
            if key in self._result_cache:
                return self._result_cache[key]

        ev_cached = self._cache.get("evidence", f"{source}::{subject}")
        if ev_cached is not None:
            cached_text = ev_cached.get("text", "")
            if cached_text and _is_blocked_content(cached_text):
                reason = _blocked_reason(cached_text)
                print(f"[evidence] Purging blocked evidence cache for "
                      f"'{subject}' (reason={reason})", flush=True)
                self._cache.delete("evidence", f"{source}::{subject}")
            else:
                result = EvidenceResult(
                    source=ev_cached.get("source", source),
                    found=ev_cached.get("found", False),
                    text=ev_cached.get("text", ""),
                    snippets=ev_cached.get("snippets", []),
                    url=ev_cached.get("url"),
                    meta=ev_cached.get("meta", {}),
                )
                with self._lock:
                    self._result_cache[key] = result
                return result

        if source == "web":
            result = self._get_web(subject)
        else:
            result = EvidenceResult(source=source, found=False, text="",
                                    snippets=[], meta={"note": "unsupported"})

        self._cache.set("evidence", f"{source}::{subject}", {
            "source": result.source, "found": result.found,
            "text": result.text, "snippets": result.snippets,
            "url": result.url, "meta": result.meta,
        })
        with self._lock:
            self._result_cache[key] = result
        return result

    def _get_web(self, subject: str) -> EvidenceResult:
        """Retrieve web evidence for subject. No topic_context used."""
        t0   = time.perf_counter()
        mode = (self.cfg.web_mode or "single").lower().strip()

        # ── 1. Search (subject only, no topic) ──────────────────────────────
        query   = build_search_query(subject)
        raw     = _web_search(query, self.cfg, self._cache)

        # ── 2. Filter ────────────────────────────────────────────────────────
        filtered = [r for r in raw
                    if r.get("url")
                    and not _should_exclude(r["url"], self._exclude)]

        if self.cfg.min_quality_score > 0:
            filtered = [r for r in filtered
                        if _domain_quality_score(r["url"]) >= self.cfg.min_quality_score]

        filtered.sort(key=lambda r: _domain_quality_score(r["url"]), reverse=True)

        # ── 3. Pick best source(s) ───────────────────────────────────────────
        n_pick = 1 if mode == "single" else max(1, self.cfg.web_max_snippets)
        picked: List[Dict[str, str]] = []
        seen_domains: Set[str] = set()
        for r in filtered:
            rd = _root_domain(_domain(r["url"]))
            if rd and rd not in seen_domains:
                seen_domains.add(rd)
                picked.append(r)
                if len(picked) >= n_pick:
                    break

        # ── 4. Fetch page ────────────────────────────────────────────────────
        fetch_n  = 0 if mode == "snippets" else (1 if mode == "single"
                                                  else self.cfg.web_max_fetch_pages)
        fetched  = {}
        fetch_url = None
        if fetch_n > 0 and picked:
            for candidate in picked[:min(3, len(picked))]:
                fetch_url = candidate["url"]
                txt, mth  = _fetch_page(fetch_url, self.cfg, self._cache,
                                         max_chars=self.cfg.web_snippet_max_chars)
                if txt and not mth.startswith("blocked"):
                    fetched[fetch_url] = {"text": txt, "method": mth}
                    if self.cfg.debug:
                        print(f"[evidence] Fetched '{subject}' → {fetch_url} "
                              f"({len(txt)} chars, {mth}) "
                              f"in {time.perf_counter()-t0:.1f}s", flush=True)
                    break
                else:
                    if self.cfg.debug:
                        print(f"[evidence] Skipping blocked URL for '{subject}': "
                              f"{fetch_url} ({mth})", flush=True)
                    fetched[fetch_url] = {"text": "", "method": mth}
                    continue

        # ── 5. Build snippet records ─────────────────────────────────────────
        snippets: List[Dict[str, Any]] = []
        for rank, r in enumerate(picked, 1):
            url     = r["url"]
            quality = _domain_quality_score(url)
            ft      = (fetched.get(url, {}) or {}).get("text", "")
            fm      = (fetched.get(url, {}) or {}).get("method", "none")
            use_txt = (ft if ft else r.get("snippet", ""))
            use_txt = re.sub(r"\s+", " ", use_txt).strip()[:self.cfg.web_snippet_max_chars]
            snippets.append({
                "title":         r.get("title", "") or _domain(url),
                "url":           url,
                "snippet_text":  use_txt,
                "domain":        _domain(url),
                "quality_score": int(quality),
                "quality_label": _quality_label(quality),
                "source_rank":   rank,
                "fetched_page":  bool(ft),
                "fetch_method":  fm,
                "backend":       r.get("backend", ""),
                "source_type":   "web",
            })

        text  = "\n\n".join(s["snippet_text"] for s in snippets
                             if s.get("snippet_text")).strip()
        found = bool(text)
        elapsed = time.perf_counter() - t0

        meta = {
            "web_mode":       mode,
            "query":          query,
            "raw_count":      len(raw),
            "filtered_count": len(filtered),
            "picked_count":   len(picked),
            "fetch_url":      fetch_url,
            "fetched_chars":  len(text),
            "elapsed_s":      round(elapsed, 2),
            "search_backend": self.cfg.search_backend,
            "from_cache":     False,
        }

        return EvidenceResult(source="web", found=found, text=text,
                               snippets=snippets,
                               url=fetch_url or (picked[0]["url"] if picked else None),
                               meta=meta)


# ──────────────────────────────────────────────────────────────────────────────
# CACHE REPAIR UTILITY
# ──────────────────────────────────────────────────────────────────────────────

def repair_page_cache(cache_dir: str, dry_run: bool = False) -> Dict[str, int]:
    pages_dir = os.path.join(cache_dir, "pages")
    if not os.path.isdir(pages_dir):
        print(f"[repair] No pages directory at {pages_dir}")
        return {"total": 0, "blocked": 0, "purged": 0, "errors": 0}

    files = [f for f in os.listdir(pages_dir) if f.endswith(".json")]
    total = len(files)
    blocked = 0
    purged = 0
    errors = 0

    print(f"[repair] Scanning {total} page cache entries in {pages_dir} ...")

    for i, fname in enumerate(files):
        if (i + 1) % 500 == 0:
            print(f"  ...{i+1}/{total} checked, {blocked} blocked found")

        fpath = os.path.join(pages_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            errors += 1
            continue

        if not isinstance(obj, dict):
            continue

        text = obj.get("text", "")
        method = obj.get("method", "")

        if method.startswith("blocked"):
            blocked += 1
            if not dry_run:
                try:
                    os.remove(fpath)
                    purged += 1
                except Exception:
                    errors += 1
            continue

        if text and _is_blocked_content(text):
            reason = _blocked_reason(text)
            blocked += 1
            preview = text[:120].replace("\n", " ")
            print(f"  [blocked] {fname[:16]}... reason={reason}  "
                  f"preview: {preview!r}")
            if not dry_run:
                try:
                    os.remove(fpath)
                    purged += 1
                except Exception:
                    errors += 1

    result = {"total": total, "blocked": blocked, "purged": purged, "errors": errors}
    action = "would purge" if dry_run else "purged"
    print(f"\n[repair] Done: {total} scanned, {blocked} blocked, "
          f"{action} {purged if not dry_run else blocked}, {errors} errors")
    return result


def repair_evidence_cache(cache_dir: str, dry_run: bool = False) -> Dict[str, int]:
    ev_dir = os.path.join(cache_dir, "evidence")
    if not os.path.isdir(ev_dir):
        print(f"[repair] No evidence directory at {ev_dir}")
        return {"total": 0, "blocked": 0, "purged": 0, "errors": 0}

    files = [f for f in os.listdir(ev_dir) if f.endswith(".json")]
    total = len(files)
    blocked = 0
    purged = 0
    errors = 0

    print(f"[repair] Scanning {total} evidence cache entries in {ev_dir} ...")

    for fname in files:
        fpath = os.path.join(ev_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            errors += 1
            continue

        if not isinstance(obj, dict):
            continue

        text = obj.get("text", "")
        if text and _is_blocked_content(text):
            reason = _blocked_reason(text)
            blocked += 1
            if not dry_run:
                try:
                    os.remove(fpath)
                    purged += 1
                except Exception:
                    errors += 1

    result = {"total": total, "blocked": blocked, "purged": purged, "errors": errors}
    action = "would purge" if dry_run else "purged"
    print(f"[repair-evidence] {total} scanned, {blocked} blocked, "
          f"{action} {purged if not dry_run else blocked}, {errors} errors")
    return result


# ──────────────────────────────────────────────────────────────────────────────
# UTILITY
# ──────────────────────────────────────────────────────────────────────────────

def check_backends() -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    info["valyu_key"]  = bool(os.getenv("VALYU_API_KEY","").strip())
    try: from valyu import Valyu; info["valyu_sdk"] = True  # noqa
    except ImportError: info["valyu_sdk"] = False
    info["valyu"] = info["valyu_key"] and info["valyu_sdk"]
    info["serper"] = bool(os.getenv("SERPER_API_KEY","").strip())
    info["brave"]  = bool(os.getenv("BRAVE_API_KEY","").strip())
    try: from ddgs import DDGS; info["ddg"] = True  # noqa
    except ImportError: info["ddg"] = False
    info["bs4"] = HAS_BS4
    info["searxng"] = bool(os.getenv("SEARXNG_API_BASE",""))
    if info["valyu"]:     info["auto_pick"] = "valyu"
    elif info["serper"]:  info["auto_pick"] = "serper"
    elif info["brave"]:   info["auto_pick"] = "brave"
    elif info["ddg"]:     info["auto_pick"] = "ddg"
    else:                 info["auto_pick"] = "none"
    return info


def print_cache_stats(cache_dir: str):
    if not cache_dir or not os.path.isdir(cache_dir):
        print("[cache] No cache directory.")
        return
    for ns in ("search", "pages", "evidence"):
        d = os.path.join(cache_dir, ns)
        n = len(os.listdir(d)) if os.path.isdir(d) else 0
        print(f"[cache] {ns:10s}: {n:>6,} cached entries")


# ──────────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3 and sys.argv[1] == "--repair-cache":
        cache_path = sys.argv[2]
        dry = "--dry-run" in sys.argv
        if dry:
            print("[repair] DRY RUN — no files will be deleted\n")
        repair_page_cache(cache_path, dry_run=dry)
        repair_evidence_cache(cache_path, dry_run=dry)
        print_cache_stats(cache_path)
        sys.exit(0)

    print("=== Backend check ===")
    print(check_backends())

    cfg = EvidenceConfig(
        web_mode="single",
        search_backend="auto",
        min_quality_score=60.0,
        cache_dir="./test_cache",
        cache_ttl_hours=720.0,
        debug=True,
    )

    ret = EvidenceRetriever(cfg)
    subject = sys.argv[1] if len(sys.argv) > 1 else "Apple Inc"

    print(f"\n=== First call: '{subject}' (should hit network) ===")
    t0 = time.time()
    ev = ret.get("web", subject)
    print(f"  found={ev.found}  url={ev.url}")
    print(f"  text preview: {ev.text[:300]!r}")
    print(f"  meta: {ev.meta}")
    print(f"  elapsed: {time.time()-t0:.2f}s")

    print(f"\n=== Second call: '{subject}' (should be instant from cache) ===")
    t0 = time.time()
    ev2 = ret.get("web", subject)
    print(f"  found={ev2.found}  text_len={len(ev2.text)}")
    print(f"  elapsed: {time.time()-t0:.4f}s  ← should be near 0")

    print("\n=== Cache stats ===")
    print_cache_stats("./test_cache")




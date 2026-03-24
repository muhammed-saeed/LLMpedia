"""
factuality_multisource.py – Factuality & similarity evaluation pipeline. this is oldest appproach but was really solid and i have used that one for the ablation

Supports:
  - Candidates: ours, grokipedia, wikipedia  (all can be checked against web evidence)
  - Baselines/evidence: wikipedia, web (free DuckDuckGo/SearXNG), webrag
  - Modes: online (concurrent), batch (OpenAI Batch API)
  - Full similarity: TF-IDF, Jaccard, n-gram, semantic embeddings, BERTScore, stylistic
  - Structured audit trail per subject (per-candidate & per-baseline JSON files)
  - HPC-friendly: disk caching, resumable, no browser needed

Speed optimizations:
  - Buffered I/O: audit writes batched (100 records), not per-record
  - Wikipedia result caching: same subject fetched once, not 2-4x
  - Connection pooling with auto-retry (429/5xx)
  - Parallel availability pre-fetch (20 concurrent)
  - Wikitext cleaning removes 40%+ junk tokens before similarity

Evidence logic:
  - wikipedia: fetches the FULL specific subject page only (redirects OK, no search)
  - web:       fetches MAX_FETCH_PAGES full pages and sends ALL to LLM
  - webrag:    uses webrag endpoint with max_snippets

Install (HPC):
  pip install --user duckduckgo-search beautifulsoup4 requests python-dotenv
  pip install --user scikit-learn sentence-transformers   # for similarity
  pip install --user openai                               # for batch mode / OpenAI embeddings
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS: eval_text_utils (wikitext cleaning)
# ═══════════════════════════════════════════════════════════════════════════════

from eval_text_utils import clean_wikitext_for_eval, detect_calibrate_mode

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS: web_evidence (same directory)
# ═══════════════════════════════════════════════════════════════════════════════

from web_evidence import (
    web_evidence_snippets,
    web_baseline_text,
    set_audit_dir as _set_web_audit_dir,
    check_backends as check_web_backends,
    clear_search_cache,
)

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

_jsonl_lock = threading.Lock()
_cache_lock = threading.Lock()


def _str2bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _dbg(msg: str, debug: bool = True):
    if debug:
        print(msg, flush=True)


def _safe_subject_dirname(subject: str) -> str:
    return re.sub(r"[^\w\-.]", "_", subject)[:120]


# ── Buffered JSONL writer ─────────────────────────────────────────────────────

class BufferedJSONLWriter:
    """
    Thread-safe buffered JSONL writer. Accumulates records in memory and
    flushes to disk every `flush_every` records or on explicit flush().
    ~100x fewer file open/close syscalls vs per-record writes.
    """
    def __init__(self, flush_every: int = 100):
        self._buffers: Dict[str, List[str]] = defaultdict(list)
        self._lock = threading.Lock()
        self._flush_every = flush_every

    def append(self, path: str, obj: dict):
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        with self._lock:
            self._buffers[path].append(line)
            if len(self._buffers[path]) >= self._flush_every:
                self._flush_path(path)

    def _flush_path(self, path: str):
        """Must be called with self._lock held."""
        lines = self._buffers.pop(path, [])
        if not lines:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.writelines(lines)

    def flush_all(self):
        with self._lock:
            for path in list(self._buffers.keys()):
                self._flush_path(path)

    def __del__(self):
        try:
            self.flush_all()
        except Exception:
            pass


# Global buffered writer instance
_buf_writer = BufferedJSONLWriter(flush_every=100)


def _append_jsonl(path: str, obj: dict):
    _buf_writer.append(path, obj)


def _flush_all_writers():
    _buf_writer.flush_all()


# ── JSON writer (per-subject audit files) ─────────────────────────────────────

_json_write_lock = threading.Lock()
_json_write_buffer: Dict[str, Any] = {}


def _write_json(path: str, obj: Any):
    """Write a single JSON file (pretty-printed) in a thread-safe way."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with _json_write_lock:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)


def _write_jsonl(path: str, objects: List[dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in objects:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _load_jsonl(path: str, max_items: int = 0) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return out
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
                if 0 < max_items <= len(out):
                    break
    return out


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
    return ""


def _extract_first_json_object(txt: str) -> Optional[Dict[str, Any]]:
    s = txt.strip()
    if not s:
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(s[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


def _ts() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


# ═══════════════════════════════════════════════════════════════════════════════
# REPO IMPORTS (optional – for LLM factory)
# ═══════════════════════════════════════════════════════════════════════════════

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

HAS_SETTINGS = False
settings = None
try:
    from settings import settings as _settings_mod
    from llm.factory import make_llm_from_config
    settings = _settings_mod
    HAS_SETTINGS = True
except ImportError:
    pass

# ═══════════════════════════════════════════════════════════════════════════════
# OPENAI CLIENT (lazy)
# ═══════════════════════════════════════════════════════════════════════════════

_openai_client = None


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI
            _openai_client = OpenAI()
        except ImportError:
            raise ImportError("openai package required: pip install openai")
    return _openai_client


# ═══════════════════════════════════════════════════════════════════════════════
# LAZY ML LIBRARIES
# ═══════════════════════════════════════════════════════════════════════════════

_sentence_transformer_model = None
_spacy_nlp = None


def get_sentence_transformer(model_name: str = "all-MiniLM-L6-v2"):
    global _sentence_transformer_model
    if _sentence_transformer_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _sentence_transformer_model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError("sentence-transformers required: pip install sentence-transformers")
    return _sentence_transformer_model


def get_spacy_nlp():
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy
            try:
                _spacy_nlp = spacy.load("en_core_web_sm")
            except OSError:
                import subprocess
                subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
                _spacy_nlp = spacy.load("en_core_web_sm")
        except ImportError:
            raise ImportError("spacy required: pip install spacy")
    return _spacy_nlp


# ═══════════════════════════════════════════════════════════════════════════════
# WIKIPEDIA API — with connection pooling + retry + result caching
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_WIKI_USER_AGENT = os.getenv(
    "WIKI_USER_AGENT",
    "LLMPediaFactCheck/0.4 (https://example.com/contact; you@example.com)",
)
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"


def _make_robust_session() -> requests.Session:
    """Session with connection pooling and automatic retries for 429/5xx."""
    s = requests.Session()
    s.headers.update({"User-Agent": DEFAULT_WIKI_USER_AGENT})
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


_wiki_session = _make_robust_session()


def _strip_html(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    return text.replace("&quot;", '"').replace("&amp;", "&")


def wiki_search_snippets(query: str, max_snippets: int = 3, debug: bool = False) -> List[Dict[str, str]]:
    params = {"action": "query", "list": "search", "srsearch": query,
              "format": "json", "utf8": 1, "srlimit": max_snippets}
    try:
        resp = _wiki_session.get(WIKI_API_URL, params=params, timeout=15)
        resp.raise_for_status()
        items = resp.json().get("query", {}).get("search", []) or []
        out = []
        for it in items[:max_snippets]:
            page_url = f"https://en.wikipedia.org/wiki/{it.get('title', '').replace(' ', '_')}"
            out.append({
                "title": it.get("title", ""),
                "snippet_html": it.get("snippet", ""),
                "snippet_text": _strip_html(it.get("snippet", "")),
                "url": page_url,
                "source_type": "wikipedia",
                "pageid": it.get("pageid"),
            })
        _dbg(f"[wiki] '{query}' → {len(out)} snippet(s)", debug)
        return out
    except Exception as e:
        _dbg(f"[wiki] search error '{query}': {e}", debug)
        return []


def wiki_fetch_extract(subject: str, debug: bool = False) -> Dict[str, Any]:
    params = {"action": "query", "prop": "extracts", "explaintext": 1,
              "redirects": 1, "format": "json", "utf8": 1, "titles": subject}
    try:
        resp = _wiki_session.get(WIKI_API_URL, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        query_data = data.get("query") or {}
        pages = query_data.get("pages") or {}
        redirects = query_data.get("redirects") or []
        was_redirected = len(redirects) > 0

        for _, p in pages.items():
            title = p.get("title") or subject
            missing = bool(p.get("missing"))
            extract = p.get("extract") or ""
            pageid = p.get("pageid")
            exact_match = not missing and not was_redirected
            page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            _dbg(f"[wiki-extract] '{subject}' → title='{title}', missing={missing}, "
                 f"redirected={was_redirected}", debug)
            return {"title": title, "extract": extract, "pageid": pageid,
                    "missing": missing, "redirected": was_redirected,
                    "exact_match": exact_match,
                    "url": page_url,
                    "source_type": "wikipedia"}
        return {"title": subject, "extract": "", "pageid": None,
                "missing": True, "redirected": False, "exact_match": False,
                "url": None, "source_type": "wikipedia"}
    except Exception as e:
        _dbg(f"[wiki-extract] error '{subject}': {e}", debug)
        return {"title": subject, "extract": "", "pageid": None,
                "missing": None, "redirected": None, "exact_match": False,
                "url": None, "source_type": "wikipedia"}


# ── Wikipedia result cache (eliminates 2-4x duplicate API calls per subject) ──

_wiki_extract_cache: Dict[str, Dict[str, Any]] = {}
_wiki_extract_cache_lock = threading.Lock()


def wiki_fetch_extract_cached(subject: str, debug: bool = False) -> Dict[str, Any]:
    """Cached wrapper — same subject fetched once, not 2-4x."""
    with _wiki_extract_cache_lock:
        if subject in _wiki_extract_cache:
            return _wiki_extract_cache[subject]
    result = wiki_fetch_extract(subject, debug=debug)
    with _wiki_extract_cache_lock:
        _wiki_extract_cache[subject] = result
    return result


def wiki_check_link_exists(title: str, debug: bool = False) -> Dict[str, Any]:
    params = {"action": "query", "list": "search", "srsearch": title,
              "format": "json", "utf8": 1, "srlimit": 1}
    try:
        resp = _wiki_session.get(WIKI_API_URL, params=params, timeout=15)
        resp.raise_for_status()
        items = resp.json().get("query", {}).get("search", []) or []
        if not items:
            return {"title": title, "exists_on_wikipedia": False, "top_title": None, "url": None}
        top = items[0].get("title", "")
        return {"title": title, "exists_on_wikipedia": True,
                "top_title": top,
                "url": f"https://en.wikipedia.org/wiki/{top.replace(' ', '_')}"}
    except Exception:
        return {"title": title, "exists_on_wikipedia": None, "top_title": None, "url": None}


# ═══════════════════════════════════════════════════════════════════════════════
# GROKIPEDIA
# ═══════════════════════════════════════════════════════════════════════════════

def grokipedia_url_for_subject(subject: str) -> str:
    slug = subject.strip().replace(" ", "_")
    return f"https://grokipedia.com/page/{slug}"


def grokipedia_fetch_text(subject: str, debug: bool = False,
                         max_retries: int = 3, retry_backoff: float = 2.0) -> Dict[str, Any]:
    url = grokipedia_url_for_subject(subject)
    last_status = None

    for attempt in range(max_retries + 1):
        try:
            resp = _wiki_session.get(url, timeout=25)
            last_status = resp.status_code

            if last_status == 404:
                _dbg(f"[grokipedia] '{subject}' → 404 (not found)", debug)
                return {"url": url, "status": 404, "text": "", "found": False,
                        "source_type": "grokipedia"}

            if last_status >= 500:
                if attempt < max_retries:
                    wait = retry_backoff * (2 ** attempt) + random.uniform(0, 1)
                    _dbg(f"[grokipedia] '{subject}' → {last_status}, "
                         f"retry {attempt + 1}/{max_retries} in {wait:.1f}s", debug)
                    time.sleep(wait)
                    continue
                else:
                    return {"url": url, "status": last_status, "text": "", "found": False,
                            "source_type": "grokipedia", "retries_exhausted": True}

            if last_status == 429:
                if attempt < max_retries:
                    wait = retry_backoff * (3 ** attempt) + random.uniform(0, 2)
                    time.sleep(wait)
                    continue
                else:
                    return {"url": url, "status": 429, "text": "", "found": False,
                            "source_type": "grokipedia", "retries_exhausted": True}

            html = resp.text or ""
            text = ""
            try:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, "html.parser")
                for sel in ["main", "article", "div#content", "div.page-content", "div.container"]:
                    node = soup.select_one(sel)
                    if node:
                        text = node.get_text(separator="\n", strip=True)
                        break
                if not text and soup.body:
                    text = soup.body.get_text(separator="\n", strip=True)
            except Exception:
                text = _strip_html(html)
            text = re.sub(r"\n{3,}", "\n\n", text).strip()
            found = last_status == 200 and len(text) >= 100
            return {"url": url, "status": last_status, "text": text, "found": found,
                    "source_type": "grokipedia"}

        except requests.exceptions.Timeout:
            if attempt < max_retries:
                time.sleep(retry_backoff * (2 ** attempt))
                continue
            return {"url": url, "status": None, "text": "", "found": False,
                    "source_type": "grokipedia", "error": "timeout"}

        except Exception as e:
            if attempt < max_retries:
                time.sleep(retry_backoff * (2 ** attempt))
                continue
            return {"url": url, "status": None, "text": "", "found": False,
                    "source_type": "grokipedia", "error": str(e)}

    return {"url": url, "status": last_status, "text": "", "found": False,
            "source_type": "grokipedia"}


# ═══════════════════════════════════════════════════════════════════════════════
# WEBRAG
# ═══════════════════════════════════════════════════════════════════════════════

def webrag_search_snippets(query: str, max_snippets: int,
                           endpoint: Optional[str], debug: bool = False) -> List[Dict[str, str]]:
    endpoint = endpoint or os.getenv("WEBRAG_ENDPOINT")
    if not endpoint:
        return []
    try:
        payload = {"q": query, "k": max_snippets}
        r = _wiki_session.post(endpoint, json=payload, timeout=25)
        r.raise_for_status()
        data = r.json()
        items = data.get("snippets") or data.get("results") or (data if isinstance(data, list) else [])
        out = []
        for it in items[:max_snippets]:
            if not isinstance(it, dict):
                continue
            txt = it.get("text") or it.get("snippet") or it.get("content") or ""
            if isinstance(txt, str) and txt.strip():
                out.append({
                    "title": str(it.get("title") or it.get("source") or "WebRAG").strip(),
                    "snippet_text": txt.strip(),
                    "url": str(it.get("url") or it.get("link") or "").strip(),
                    "source_type": "webrag",
                })
        return out
    except Exception as e:
        _dbg(f"[webrag] error: {e}", debug)
        return []


# ═══════════════════════════════════════════════════════════════════════════════
# TEXT PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def tokenize_simple(text: str) -> List[str]:
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return [t for t in text.split() if t]


def tokenize_sentences(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def get_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def extract_domains_from_snippets(snippets: List[Dict[str, str]]) -> List[str]:
    domains = []
    seen = set()
    for s in snippets:
        url = s.get("url", "")
        if not url:
            continue
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()
            if domain and domain not in seen:
                seen.add(domain)
                domains.append(domain)
        except Exception:
            pass
    return domains


# ═══════════════════════════════════════════════════════════════════════════════
# SIMILARITY METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def tfidf_cosine_similarity(text1: str, text2: str) -> float:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
        vec = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=10000)
        mat = vec.fit_transform([text1, text2])
        return float(sk_cosine(mat[0:1], mat[1:2])[0][0])
    except Exception:
        return 0.0


def jaccard_similarity(text1: str, text2: str) -> float:
    t1, t2 = set(tokenize_simple(text1)), set(tokenize_simple(text2))
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


def ngram_jaccard_similarity(text1: str, text2: str, n: int) -> float:
    ng1 = set(get_ngrams(tokenize_simple(text1), n))
    ng2 = set(get_ngrams(tokenize_simple(text2), n))
    if not ng1 or not ng2:
        return 0.0
    return len(ng1 & ng2) / len(ng1 | ng2)


def ngram_overlap_coefficient(text1: str, text2: str, n: int) -> float:
    ng1 = set(get_ngrams(tokenize_simple(text1), n))
    ng2 = set(get_ngrams(tokenize_simple(text2), n))
    if not ng1 or not ng2:
        return 0.0
    return len(ng1 & ng2) / min(len(ng1), len(ng2))


def cosine_sim(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    return dot / (n1 * n2) if n1 and n2 else 0.0


def get_embeddings_batch_openai(texts: List[str], model: str = "text-embedding-3-small",
                                debug: bool = False) -> List[Optional[List[float]]]:
    if not texts:
        return []
    try:
        client = get_openai_client()
        truncated = [t[:32000] if t else "" for t in texts]
        non_empty = [(i, truncated[i]) for i in range(len(truncated)) if truncated[i].strip()]
        if not non_empty:
            return [None] * len(texts)
        resp = client.embeddings.create(model=model, input=[t for _, t in non_empty])
        results: List[Optional[List[float]]] = [None] * len(texts)
        for j, emb in enumerate(resp.data):
            results[non_empty[j][0]] = emb.embedding
        return results
    except Exception as e:
        _dbg(f"[emb-openai] error: {e}", debug)
        return [None] * len(texts)


def get_embeddings_batch_st(texts: List[str], model_name: str = "all-MiniLM-L6-v2",
                            debug: bool = False) -> List[Optional[List[float]]]:
    if not texts:
        return []
    try:
        model = get_sentence_transformer(model_name)
        truncated = [t[:10000] if t else "" for t in texts]
        non_empty = [(i, truncated[i]) for i in range(len(truncated)) if truncated[i].strip()]
        if not non_empty:
            return [None] * len(texts)
        embs = model.encode([t for _, t in non_empty], convert_to_numpy=True)
        results: List[Optional[List[float]]] = [None] * len(texts)
        for j, emb in enumerate(embs):
            results[non_empty[j][0]] = emb.tolist()
        return results
    except Exception as e:
        _dbg(f"[emb-st] error: {e}", debug)
        return [None] * len(texts)


def compute_semantic_similarity(text1: str, text2: str, provider: str = "sentence-transformer",
                                model: str = "all-MiniLM-L6-v2", debug: bool = False) -> Optional[float]:
    if not text1.strip() or not text2.strip():
        return None
    if provider == "openai":
        embs = get_embeddings_batch_openai([text1, text2], model=model, debug=debug)
    else:
        embs = get_embeddings_batch_st([text1, text2], model_name=model, debug=debug)
    if embs[0] is None or embs[1] is None:
        return None
    return cosine_sim(embs[0], embs[1])


def compute_bertscore(text1: str, text2: str, max_sentences: int = 50,
                      debug: bool = False) -> Optional[float]:
    try:
        from bert_score import score as bert_score_fn
        s1 = " ".join(tokenize_sentences(text1)[:max_sentences])
        s2 = " ".join(tokenize_sentences(text2)[:max_sentences])
        if not s1 or not s2:
            return None
        _, _, F1 = bert_score_fn([s1], [s2], lang="en", verbose=False)
        return float(F1[0])
    except ImportError:
        _dbg("[bertscore] not installed: pip install bert-score", debug)
        return None
    except Exception as e:
        _dbg(f"[bertscore] error: {e}", debug)
        return None


def compute_flesch_kincaid_grade(text: str) -> float:
    sents = tokenize_sentences(text)
    words = tokenize_simple(text)
    if not sents or not words:
        return 0.0
    def syllables(w):
        w = w.lower()
        if len(w) <= 3:
            return 1
        c, prev = 0, False
        for ch in w:
            iv = ch in "aeiouy"
            if iv and not prev:
                c += 1
            prev = iv
        if w.endswith('e') and c > 1:
            c -= 1
        return max(1, c)
    total_syl = sum(syllables(w) for w in words)
    grade = 0.39 * (len(words) / len(sents)) + 11.8 * (total_syl / len(words)) - 15.59
    return max(0, grade)


def compute_lexical_diversity(text: str) -> float:
    tokens = tokenize_simple(text)
    return len(set(tokens)) / len(tokens) if tokens else 0.0


def compute_avg_sentence_length(text: str) -> float:
    sents = tokenize_sentences(text)
    words = tokenize_simple(text)
    return len(words) / len(sents) if sents else 0.0


def compute_pos_distribution(text: str, debug: bool = False) -> Dict[str, float]:
    try:
        nlp = get_spacy_nlp()
        doc = nlp(text[:10000])
        counts = Counter(t.pos_ for t in doc)
        total = sum(counts.values())
        if total == 0:
            return {}
        return {pos: counts.get(pos, 0) / total for pos in ["NOUN", "VERB", "ADJ", "ADV", "PROPN"]}
    except Exception as e:
        _dbg(f"[pos] error: {e}", debug)
        return {}


def compute_stylistic_similarity(m1: Dict[str, float], m2: Dict[str, float]) -> float:
    keys = ["flesch_kincaid_grade", "lexical_diversity", "avg_sentence_length"]
    keys += [k for k in m1 if k.startswith("pos_")]
    ranges = {"flesch_kincaid_grade": 20.0, "lexical_diversity": 1.0,
              "avg_sentence_length": 50.0, "pos_noun": 0.5, "pos_verb": 0.3,
              "pos_adj": 0.2, "pos_adv": 0.1, "pos_propn": 0.2}
    total, count = 0.0, 0
    for k in keys:
        v1, v2 = m1.get(k, 0.0), m2.get(k, 0.0)
        r = ranges.get(k, 1.0)
        total += min(1.0, abs(v1 - v2) / r if r > 0 else 0.0)
        count += 1
    return 1.0 - (total / count) if count else 0.0


def compute_combined_similarity(metrics: Dict[str, float],
                                weights: Optional[Dict[str, float]] = None) -> float:
    if weights is None:
        weights = {"tfidf_cosine": 0.15, "jaccard": 0.10, "ngram_1_overlap": 0.10,
                   "ngram_2_overlap": 0.10, "ngram_3_overlap": 0.10,
                   "semantic_cosine": 0.20, "bertscore_f1": 0.15, "stylistic_similarity": 0.10}
    wsum, wtot = 0.0, 0.0
    for k, w in weights.items():
        v = metrics.get(k)
        if v is not None and isinstance(v, (int, float)):
            wsum += w * v
            wtot += w
    return wsum / wtot if wtot else 0.0


def compute_all_similarity_metrics(
    text1: str, text2: str,
    ngram_values: List[int] = None,
    semantic_provider: str = "sentence-transformer",
    semantic_model: str = "all-MiniLM-L6-v2",
    compute_bertscore_flag: bool = False,
    compute_stylistic: bool = False,
    debug: bool = False,
) -> Dict[str, Any]:
    if ngram_values is None:
        ngram_values = [1, 2, 3]
    m: Dict[str, Any] = {}
    if not text1.strip() or not text2.strip():
        return m
    m["tfidf_cosine"] = tfidf_cosine_similarity(text1, text2)
    m["jaccard"] = jaccard_similarity(text1, text2)
    for n in ngram_values:
        m[f"ngram_{n}_jaccard"] = ngram_jaccard_similarity(text1, text2, n)
        m[f"ngram_{n}_overlap"] = ngram_overlap_coefficient(text1, text2, n)
    sem = compute_semantic_similarity(text1, text2, provider=semantic_provider,
                                      model=semantic_model, debug=debug)
    if sem is not None:
        m["semantic_cosine"] = sem
    if compute_bertscore_flag:
        bf = compute_bertscore(text1, text2, debug=debug)
        if bf is not None:
            m["bertscore_f1"] = bf
    if compute_stylistic:
        s1 = {"flesch_kincaid_grade": compute_flesch_kincaid_grade(text1),
              "lexical_diversity": compute_lexical_diversity(text1),
              "avg_sentence_length": compute_avg_sentence_length(text1)}
        s1.update({f"pos_{k.lower()}": v for k, v in compute_pos_distribution(text1, debug).items()})
        s2 = {"flesch_kincaid_grade": compute_flesch_kincaid_grade(text2),
              "lexical_diversity": compute_lexical_diversity(text2),
              "avg_sentence_length": compute_avg_sentence_length(text2)}
        s2.update({f"pos_{k.lower()}": v for k, v in compute_pos_distribution(text2, debug).items()})
        m["stylistic_similarity"] = compute_stylistic_similarity(s1, s2)
        for k, v in s1.items():
            m[f"candidate_{k}"] = v
        for k, v in s2.items():
            m[f"baseline_{k}"] = v
    m["combined_similarity"] = compute_combined_similarity(m)
    m["candidate_word_count"] = len(tokenize_simple(text1))
    m["baseline_word_count"] = len(tokenize_simple(text2))
    return m


# ═══════════════════════════════════════════════════════════════════════════════
# FACT-CHECK PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════

def build_factcheck_messages(subject: str, candidate_source: str, candidate_text: str,
                             evidence_source: str, evidence_snippets: List[Dict[str, str]],
                             max_claims: int, max_chars: int = 0) -> List[Dict[str, str]]:
    """NOTE: This is the old single-call approach (kept for backwards compat).
    The pipeline now uses the two-step approach (extract + verify) instead."""
    sys_lines = [
        "You are a strict fact-checking assistant.",
        f"Extract at most {max_claims} distinct, atomic factual claims from the candidate article.",
        "For each claim, decide: \"true\" (supported), \"false\" (contradicted), \"uncertain\" (ambiguous).",
        'Include `sources`: list of evidence snippet numbers used.',
        "",
        'Output MUST be valid JSON:',
        '{"claims":[{"claim":"...","verdict":"true|false|uncertain","confidence":0.0-1.0,"sources":[1,2],"explanation":"..."}]}',
        "", f"Candidate source: {candidate_source}", f"Evidence source: {evidence_source}",
    ]
    ev_lines = []
    if evidence_snippets:
        ev_lines.append("Evidence snippets:")
        for i, sn in enumerate(evidence_snippets, 1):
            title = sn.get("title", "")
            text = sn.get("snippet_text", "")
            url = sn.get("url", "")
            quality = sn.get("quality_label", "")
            quality_tag = f" [quality: {quality}]" if quality else ""
            ev_lines.append(f"{i}. [{title}]{quality_tag} {text}" + (f" (source: {url})" if url else ""))
    else:
        ev_lines.append("(No evidence available)")

    cand_text = candidate_text[:max_chars] if max_chars > 0 else candidate_text
    user_msg = (f"Subject: {subject}\n\nCandidate article ({candidate_source}):\n"
                f"--------------------\n{cand_text}\n--------------------\n\n"
                + "\n".join(ev_lines) + "\n\nReturn ONLY the JSON object.")
    return [{"role": "system", "content": "\n".join(sys_lines)},
            {"role": "user", "content": user_msg}]


def build_claim_extraction_messages(subject: str, article_source: str,
                                    article_text: str, max_claims: int,
                                    max_chars: int = 0) -> List[Dict[str, str]]:
    """
    Build prompt to extract factual claims from an article.

    The LLM sees ONLY the article (no evidence) to prevent cherry-picking.

    Args:
        max_chars: Max article characters to include (0 = no limit, send everything).
                   Controlled by --max-article-chars CLI argument.
    """
    sys_msg = (f"Extract at most {max_claims} distinct, atomic factual claims from the article.\n"
               "Focus on verifiable facts: dates, numbers, names, events, relationships.\n"
               "Spread claims across the ENTIRE article (beginning, middle, AND end).\n"
               'Output MUST be valid JSON: {"claims":["claim 1","claim 2",...]}\nReturn ONLY the JSON.')
    art_text = article_text[:max_chars] if max_chars > 0 else article_text
    user_msg = (f"Subject: {subject}\nSource: {article_source}\n\nArticle:\n"
                f"--------------------\n{art_text}\n--------------------\nExtract factual claims.")
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]


def build_claim_verification_messages(subject: str, claim: str, evidence_source: str,
                                      evidence_snippets: List[Dict[str, str]],
                                      max_evidence_chars: int = 0) -> List[Dict[str, str]]:
    """
    Build prompt to verify a single claim against evidence.

    Args:
        max_evidence_chars: Max total evidence characters (0 = no limit).
                            Controlled by --max-evidence-chars CLI argument.
    """
    sys_msg = ('Verify if the claim is "true" (supported), "false" (contradicted), or "uncertain".\n'
               'Output JSON: {"verdict":"true|false|uncertain","confidence":0.0-1.0,"sources":[1,2],"explanation":"brief"}')
    ev_lines = []
    if evidence_snippets:
        for i, sn in enumerate(evidence_snippets, 1):
            ev_lines.append(f"{i}. [{sn.get('title', '')}] {sn.get('snippet_text', '')}")
    ev_text = "\n".join(ev_lines) if ev_lines else "(No evidence)"
    if max_evidence_chars > 0:
        ev_text = ev_text[:max_evidence_chars]
    user_msg = (f"Subject: {subject}\nClaim: {claim}\nEvidence source: {evidence_source}\n\n"
                f"Evidence:\n{ev_text}\n\nVerify this claim.")
    return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}]


def normalize_claims(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    claims = obj.get("claims") or []
    if not isinstance(claims, list):
        return []
    out = []
    for c in claims:
        if not isinstance(c, dict):
            continue
        txt = c.get("claim")
        verdict = c.get("verdict")
        if not isinstance(txt, str) or not txt.strip():
            continue
        if verdict not in ("true", "false", "uncertain"):
            continue
        conf = c.get("confidence")
        conf = float(conf) if isinstance(conf, (int, float)) else None
        sources = c.get("sources")
        if isinstance(sources, list):
            sources = sorted({int(s) for s in sources if isinstance(s, (int, float)) and int(s) > 0})
        else:
            sources = []
        out.append({"claim": txt.strip(), "verdict": verdict, "confidence": conf,
                     "sources": sources, "explanation": str(c.get("explanation", "")).strip()})
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# OPENAI BATCH API
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BatchRequest:
    custom_id: str
    messages: List[Dict[str, str]]
    metadata: Dict[str, Any] = field(default_factory=dict)


def create_batch_input_file(reqs: List[BatchRequest], model: str) -> List[Dict[str, Any]]:
    return [{"custom_id": r.custom_id, "method": "POST", "url": "/v1/chat/completions",
             "body": {"model": model, "messages": r.messages,
                      "max_tokens": 2000, "temperature": 0.0}} for r in reqs]


def submit_batch_job(input_path: str, desc: str = "factuality-check",
                     debug: bool = False) -> Optional[str]:
    try:
        client = get_openai_client()
        with open(input_path, "rb") as f:
            file_resp = client.files.create(file=f, purpose="batch")
        _dbg(f"[batch] uploaded file: {file_resp.id}", debug)
        batch = client.batches.create(input_file_id=file_resp.id,
                                      endpoint="/v1/chat/completions",
                                      completion_window="24h",
                                      metadata={"description": desc})
        _dbg(f"[batch] created job: {batch.id}", debug)
        return batch.id
    except Exception as e:
        _dbg(f"[batch] submit error: {e}", debug)
        return None


def wait_for_batch_completion(batch_id: str, poll_interval: float = 30.0,
                              max_wait: float = 86400.0, debug: bool = False) -> Optional[str]:
    client = get_openai_client()
    start = time.time()
    while time.time() - start < max_wait:
        try:
            b = client.batches.retrieve(batch_id)
            _dbg(f"[batch] status={b.status}, done={b.request_counts.completed}/{b.request_counts.total}", debug)
            if b.status == "completed":
                return b.output_file_id
            if b.status in ("failed", "expired", "cancelled"):
                _dbg(f"[batch] job {b.status}", debug)
                return None
            time.sleep(poll_interval)
        except Exception as e:
            _dbg(f"[batch] poll error: {e}", debug)
            time.sleep(poll_interval)
    return None


def download_batch_results(output_file_id: str, output_path: str, debug: bool = False) -> bool:
    try:
        client = get_openai_client()
        content = client.files.content(output_file_id)
        with open(output_path, "wb") as f:
            f.write(content.read())
        _dbg(f"[batch] downloaded → {output_path}", debug)
        return True
    except Exception as e:
        _dbg(f"[batch] download error: {e}", debug)
        return False


def parse_batch_results(output_path: str) -> Dict[str, Dict[str, Any]]:
    results = {}
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                cid = obj.get("custom_id")
                body = obj.get("response", {}).get("body", {})
                choices = body.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    results[cid] = {"raw": content, "parsed": _extract_first_json_object(content), "error": None}
                else:
                    results[cid] = {"raw": "", "parsed": None, "error": "no choices"}
            except Exception:
                continue
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# CANDIDATE / EVIDENCE / BASELINE RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════════════

_candidate_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
_evidence_cache: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
_baseline_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
_availability_cache: Dict[str, Dict[str, Any]] = {}


def get_candidate_text(candidate: str, subject: str, ours_article: Dict[str, Any],
                       debug: bool = False,
                       is_calibrated: bool = False) -> Dict[str, Any]:
    """
    Fetch candidate text. Returns dict with 'text' (raw), 'text_clean' (for eval),
    and 'meta'. When is_calibrated=True, text_clean has (0.XX) stripped.
    Wiki markup ([[...]]) is always stripped from text_clean.
    """
    key = (candidate, subject)
    with _cache_lock:
        if key in _candidate_cache:
            return _candidate_cache[key]

    out: Dict[str, Any] = {"text": "", "text_clean": "", "meta": {}}

    if candidate == "ours":
        text = (ours_article.get("wikitext") or "").strip()
        # Always clean wiki markup; confidence scores stripped if calibrated or auto-detected
        text_clean = clean_wikitext_for_eval(text)
        out = {"text": text, "text_clean": text_clean,
               "meta": {"hop": ours_article.get("hop"),
                        "source": "articles.jsonl",
                        "found": bool(text),
                        "source_type": "ours",
                        "url": None,
                        "is_calibrated": is_calibrated}}

    elif candidate == "wikipedia":
        info = wiki_fetch_extract_cached(subject, debug=debug)
        found = not info.get("missing", True)
        text = (info.get("extract") or "").strip()
        out = {"text": text, "text_clean": text,
               "meta": {"page_title": info.get("title"),
                        "missing": info.get("missing"),
                        "pageid": info.get("pageid"),
                        "redirected": info.get("redirected"),
                        "exact_match": info.get("exact_match"),
                        "found": found,
                        "source_type": "wikipedia",
                        "url": info.get("url")}}

    elif candidate == "grokipedia":
        info = grokipedia_fetch_text(subject, debug=debug)
        text = (info.get("text") or "").strip()
        out = {"text": text, "text_clean": text,
               "meta": {"url": info.get("url"),
                        "status": info.get("status"),
                        "found": info.get("found", False),
                        "source_type": "grokipedia"}}
    else:
        out = {"text": "", "text_clean": "",
               "meta": {"error": f"unknown candidate: {candidate}",
                         "found": False, "source_type": candidate}}

    with _cache_lock:
        _candidate_cache[key] = out
    return out


def get_evidence_snippets(evidence: str, subject: str, max_snippets: int,
                          webrag_endpoint: Optional[str], debug: bool = False,
                          exclude_domains: Optional[Set[str]] = None,
                          max_fetch_pages: int = 0) -> List[Dict[str, str]]:
    key = (evidence, subject)
    with _cache_lock:
        if key in _evidence_cache:
            return _evidence_cache[key]

    if evidence == "wikipedia":
        info = wiki_fetch_extract_cached(subject, debug=debug)
        text = (info.get("extract") or "").strip()
        if text:
            ev = [{"title": info.get("title", subject),
                   "snippet_text": text,
                   "url": info.get("url", ""),
                   "source_type": "wikipedia"}]
        else:
            ev = []
    elif evidence == "webrag":
        ev = webrag_search_snippets(subject, max_snippets=max_snippets,
                                    endpoint=webrag_endpoint, debug=debug)
    elif evidence == "web":
        ev = web_evidence_snippets(subject, max_snippets=max_snippets,
                                   max_fetch_pages=max_fetch_pages,
                                   exclude_domains=exclude_domains)
    else:
        ev = []

    with _cache_lock:
        _evidence_cache[key] = ev
    return ev


def get_baseline_text(baseline: str, subject: str, webrag_endpoint: Optional[str],
                      debug: bool = False,
                      exclude_domains: Optional[Set[str]] = None,
                      max_fetch_pages: int = 0) -> Dict[str, Any]:
    key = (baseline, subject)
    with _cache_lock:
        if key in _baseline_cache:
            return _baseline_cache[key]

    result: Dict[str, Any]

    if baseline == "wikipedia":
        info = wiki_fetch_extract_cached(subject, debug=debug)
        result = {"text": (info.get("extract") or "").strip(),
                  "meta": info,
                  "found": not info.get("missing", True),
                  "source_type": "wikipedia",
                  "url": info.get("url"),
                  "sources": [{"title": info.get("title", subject),
                                "url": info.get("url"),
                                "source_type": "wikipedia",
                                "pageid": info.get("pageid")}] if not info.get("missing") else []}

    elif baseline == "webrag":
        snips = webrag_search_snippets(subject, max_snippets=10,
                                       endpoint=webrag_endpoint, debug=debug)
        text = "\n\n".join(s.get("snippet_text", "") for s in snips).strip()
        result = {"text": text, "meta": {"snippets_count": len(snips)},
                  "found": bool(text), "source_type": "webrag", "url": None,
                  "sources": [{"title": s.get("title", ""), "url": s.get("url", ""),
                                "source_type": "webrag"} for s in snips]}

    elif baseline == "web":
        wb_result = web_baseline_text(subject, max_snippets=10,
                                      max_fetch_pages=max_fetch_pages,
                                      exclude_domains=exclude_domains)
        snippets = wb_result.get("snippets") or wb_result.get("meta", {}).get("sources") or []
        result = {"text": wb_result.get("text", ""),
                  "snippets": snippets,
                  "meta": wb_result.get("meta", {}),
                  "found": wb_result.get("found", False),
                  "source_type": "web", "url": None,
                  "sources": [{"title": s.get("title", ""), "url": s.get("url", ""),
                                "source_type": "web",
                                "domain": _extract_domain(s.get("url", "")),
                                "quality_score": s.get("quality_score", ""),
                                "quality_label": s.get("quality_label", "")}
                               for s in snippets]}
    else:
        result = {"text": "", "meta": {}, "found": False,
                  "source_type": baseline, "url": None, "sources": []}

    with _cache_lock:
        _baseline_cache[key] = result
    return result


def _extract_domain(url: str) -> str:
    try:
        from urllib.parse import urlparse
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def check_subject_availability(subject: str, debug: bool = False) -> Dict[str, Any]:
    with _cache_lock:
        if subject in _availability_cache:
            return _availability_cache[subject]

    avail = {"subject": subject, "wikipedia_available": False,
             "wikipedia_redirected": False, "wikipedia_redirect_target": None,
             "grokipedia_available": False}
    wiki = wiki_fetch_extract_cached(subject, debug=debug)
    avail["wikipedia_available"] = wiki.get("exact_match", False)
    avail["wikipedia_redirected"] = wiki.get("redirected", False)
    avail["wikipedia_pageid"] = wiki.get("pageid")
    avail["wikipedia_page_title"] = wiki.get("title")
    avail["wikipedia_url"] = wiki.get("url")
    if wiki.get("redirected"):
        avail["wikipedia_redirect_target"] = wiki.get("title")
    grok = grokipedia_fetch_text(subject, debug=debug)
    avail["grokipedia_available"] = grok.get("found", False)
    avail["grokipedia_url"] = grok.get("url")

    with _cache_lock:
        _availability_cache[subject] = avail
    return avail


# ═══════════════════════════════════════════════════════════════════════════════
# TASK MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class EvalTask:
    subject: str
    ours_article_idx: int
    ours_hop: Any
    candidate: str       # "ours" | "grokipedia" | "wikipedia"
    baseline: str        # "wikipedia" | "webrag" | "web"


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIT TRAIL — uses buffered writer for speed
# ═══════════════════════════════════════════════════════════════════════════════

_audit_base_dir: Optional[str] = None
_audit_candidate_jsonl: Optional[str] = None
_audit_baseline_jsonl: Optional[str] = None
_audit_claims_jsonl: Optional[str] = None

_audit_written: Set[str] = set()
_audit_written_lock = threading.Lock()


def init_audit(run_dir: str):
    global _audit_base_dir, _audit_candidate_jsonl, _audit_baseline_jsonl, _audit_claims_jsonl
    global _audit_written

    _audit_base_dir = os.path.join(run_dir, "audit")
    os.makedirs(_audit_base_dir, exist_ok=True)
    os.makedirs(os.path.join(_audit_base_dir, "subjects"), exist_ok=True)

    _audit_candidate_jsonl = os.path.join(run_dir, "audit_candidate_texts.jsonl")
    _audit_baseline_jsonl  = os.path.join(run_dir, "audit_baseline_texts.jsonl")
    _audit_claims_jsonl    = os.path.join(run_dir, "audit_claims_full.jsonl")
    _audit_written         = set()

    _set_web_audit_dir(os.path.join(_audit_base_dir, "web_evidence"))


def _subject_audit_dir(subject: str) -> str:
    safe = _safe_subject_dirname(subject)
    path = os.path.join(_audit_base_dir, "subjects", safe)
    os.makedirs(path, exist_ok=True)
    return path


def audit_save_candidate(subject: str, candidate: str, hop: Any,
                         text: str, meta: Dict[str, Any]):
    if _audit_base_dir is None:
        return
    guard_key = f"cand::{subject}::{candidate}"
    with _audit_written_lock:
        if guard_key in _audit_written:
            return
        _audit_written.add(guard_key)

    record = {
        "ts": _ts(), "subject": subject, "hop": hop, "candidate": candidate,
        "source_type": meta.get("source_type", candidate),
        "url": meta.get("url"),
        "found": meta.get("found", bool(text)),
        "word_count": len(tokenize_simple(text)),
        "char_count": len(text),
        "meta": meta, "text": text,
    }
    _append_jsonl(_audit_candidate_jsonl, record)
    subj_dir = _subject_audit_dir(subject)
    _write_json(os.path.join(subj_dir, f"candidate_{candidate}.json"), record)


def audit_save_baseline(subject: str, baseline: str,
                        text: str, sources: List[Dict[str, Any]],
                        evidence_snippets: List[Dict[str, str]],
                        found: bool, meta: Dict[str, Any]):
    if _audit_base_dir is None:
        return
    guard_key = f"base::{subject}::{baseline}"
    with _audit_written_lock:
        if guard_key in _audit_written:
            return
        _audit_written.add(guard_key)

    unique_urls  = list({s.get("url", "") for s in sources if s.get("url")})
    unique_domains = list({_extract_domain(u) for u in unique_urls if u})

    record = {
        "ts": _ts(), "subject": subject, "baseline": baseline,
        "source_type": baseline, "found": found,
        "word_count": len(tokenize_simple(text)),
        "char_count": len(text),
        "n_evidence_snippets": len(evidence_snippets),
        "n_sources": len(sources),
        "source_urls": unique_urls, "source_domains": unique_domains,
        "sources": sources, "evidence_snippets": evidence_snippets,
        "meta": meta, "text": text,
    }
    _append_jsonl(_audit_baseline_jsonl, record)
    subj_dir = _subject_audit_dir(subject)
    _write_json(os.path.join(subj_dir, f"baseline_{baseline}.json"), record)


def audit_save_claims(subject: str, candidate: str, baseline: str,
                      hop: Any, model_name: str,
                      claims: List[Dict[str, Any]],
                      evidence_snippets: List[Dict[str, str]]):
    if _audit_base_dir is None:
        return

    n_true      = sum(1 for c in claims if c.get("verdict") == "true")
    n_false     = sum(1 for c in claims if c.get("verdict") == "false")
    n_uncertain = sum(1 for c in claims if c.get("verdict") == "uncertain")

    annotated_claims = []
    for c in claims:
        cited = []
        for src_idx in (c.get("sources") or []):
            if 1 <= src_idx <= len(evidence_snippets):
                snip = evidence_snippets[src_idx - 1]
                cited.append({
                    "snippet_idx": src_idx,
                    "title": snip.get("title", ""),
                    "url": snip.get("url", ""),
                    "source_type": snip.get("source_type", baseline),
                    "snippet_text": snip.get("snippet_text", ""),
                    "quality_score": snip.get("quality_score", ""),
                    "quality_label": snip.get("quality_label", ""),
                })
        annotated_claims.append({**c, "cited_sources": cited})

    record = {
        "ts": _ts(), "subject": subject, "hop": hop,
        "candidate": candidate, "baseline": baseline,
        "model": model_name,
        "n_claims": len(claims), "n_true": n_true, "n_false": n_false, "n_uncertain": n_uncertain,
        "accuracy_true_vs_false": n_true / (n_true + n_false) if (n_true + n_false) > 0 else None,
        "evidence_source_urls": list({s.get("url", "") for s in evidence_snippets if s.get("url")}),
        "evidence_source_domains": list({_extract_domain(s.get("url", ""))
                                          for s in evidence_snippets if s.get("url")}),
        "claims": annotated_claims,
    }
    _append_jsonl(_audit_claims_jsonl, record)
    subj_dir = _subject_audit_dir(subject)
    _write_json(os.path.join(subj_dir, f"claims_{candidate}_{baseline}.json"), record)


def audit_save_availability(subject: str, availability: Dict[str, Any]):
    if _audit_base_dir is None:
        return
    guard_key = f"avail::{subject}"
    with _audit_written_lock:
        if guard_key in _audit_written:
            return
        _audit_written.add(guard_key)
    subj_dir = _subject_audit_dir(subject)
    _write_json(os.path.join(subj_dir, "availability.json"),
                {"ts": _ts(), **availability})


# ═══════════════════════════════════════════════════════════════════════════════
# ONLINE MODE — uses cleaned text for similarity + LLM
# ═══════════════════════════════════════════════════════════════════════════════

_thread_local = threading.local()


def _build_exclude_domains(args) -> Optional[Set[str]]:
    domains: Set[str] = set()
    if getattr(args, 'exclude_wikipedia_from_web', False):
        domains.add("wikipedia.org")
        domains.add("en.wikipedia.org")
    extra = getattr(args, 'exclude_domains_from_web', '')
    if extra:
        for d in extra.split(","):
            d = d.strip().lower()
            if d:
                domains.add(d)
    return domains if domains else None


def get_fact_llm(fact_cfg):
    if not hasattr(_thread_local, "llm"):
        _thread_local.llm = make_llm_from_config(fact_cfg)
    return _thread_local.llm


def run_llm_json(llm, messages: List[Dict[str, str]], timeout: float) -> Optional[Dict[str, Any]]:
    try:
        try:
            resp = llm(messages, timeout=timeout)
        except TypeError:
            resp = llm(messages)
        return _extract_first_json_object(_unwrap_text(resp).strip())
    except Exception:
        return None


def process_task_online(task: EvalTask, ours_article: Dict[str, Any],
                        fact_cfg, args, out_path: str) -> Tuple[str, bool]:
    debug = args.debug
    is_calibrated = getattr(args, '_is_calibrated', False)
    model_name = getattr(args, "_fact_model_name", args.fact_model_key)
    max_fetch = getattr(args, "max_fetch_pages", 3)

    # ── 1. Candidate text ────────────────────────────────────────────────────
    cand_info = get_candidate_text(task.candidate, task.subject, ours_article,
                                   debug=debug, is_calibrated=is_calibrated)
    cand_text = (cand_info.get("text") or "").strip()
    cand_text_clean = (cand_info.get("text_clean") or cand_text).strip()
    cand_meta = cand_info.get("meta") or {}

    # Audit gets RAW text (preserves original for inspection)
    audit_save_candidate(subject=task.subject, candidate=task.candidate,
                         hop=task.ours_hop, text=cand_text, meta=cand_meta)

    # ── 2. Baseline text ──────────────────────────────────────────────────────
    excl = _build_exclude_domains(args)
    baseline_info = get_baseline_text(task.baseline, task.subject, args.webrag_endpoint,
                                      debug=debug, exclude_domains=excl,
                                      max_fetch_pages=max_fetch)
    baseline_text    = (baseline_info.get("text") or "").strip()
    baseline_sources = baseline_info.get("sources") or []

    # ── 3. Availability ───────────────────────────────────────────────────────
    availability = check_subject_availability(task.subject, debug=debug)
    audit_save_availability(task.subject, availability)

    # ── 4. Similarity — uses CLEANED text ─────────────────────────────────────
    similarity_metrics = {}
    if args.compute_similarity and cand_text_clean and baseline_text:
        ngram_values = [int(n) for n in args.ngram_n.split(",")] if args.ngram_n else [1, 2, 3]
        similarity_metrics = compute_all_similarity_metrics(
            text1=cand_text_clean, text2=baseline_text, ngram_values=ngram_values,
            semantic_provider=args.semantic_provider, semantic_model=args.semantic_model,
            compute_bertscore_flag=args.compute_bertscore,
            compute_stylistic=args.compute_stylistic, debug=debug)

    # ── 5. Evidence snippets ──────────────────────────────────────────────────
    if task.baseline == "web" and baseline_info.get("snippets"):
        evidence_snips = baseline_info["snippets"]
    else:
        evidence_snips = get_evidence_snippets(
            task.baseline, task.subject, max_snippets=args.max_evidence_snippets,
            webrag_endpoint=args.webrag_endpoint, debug=debug,
            exclude_domains=excl, max_fetch_pages=max_fetch)

    audit_save_baseline(subject=task.subject, baseline=task.baseline,
                        text=baseline_text, sources=baseline_sources,
                        evidence_snippets=evidence_snips,
                        found=baseline_info.get("found", False),
                        meta=baseline_info.get("meta", {}))

    # ── 6. Empty candidate → stub ────────────────────────────────────────────
    if not cand_text or not cand_meta.get("found", True):
        record = _build_record(task, cand_meta, False, baseline_info, evidence_snips,
                               {"claims": []}, availability, similarity_metrics, [], [], args)
        record["note"] = "empty_candidate_text"
        _append_jsonl(out_path, record)
        audit_save_claims(task.subject, task.candidate, task.baseline,
                          task.ours_hop, model_name, [], evidence_snips)
        return task.subject, True

    # ── 7. Fact-check — TWO-STEP: extract claims blind, then verify each ─────
    #
    # Step A: LLM sees ONLY your article → extracts claims WITHOUT seeing evidence
    #         (prevents cherry-picking claims the LLM knows are supported)
    # Step B: For each claim, LLM sees the claim + evidence → verdict
    #         (each claim judged independently)
    #
    claims = []
    if args.run_factcheck:
        _run_llm = None
        if fact_cfg is not None:
            _run_llm = lambda msgs: run_llm_json(get_fact_llm(fact_cfg), msgs, timeout=args.timeout)
        elif os.getenv("OPENAI_API_KEY"):
            def _run_llm_openai(msgs):
                try:
                    client = get_openai_client()
                    resp = client.chat.completions.create(
                        model=args.fact_model_key, messages=msgs,
                        max_tokens=32_000, temperature=0.0)
                    return _extract_first_json_object(resp.choices[0].message.content or "")
                except Exception as e:
                    _dbg(f"[factcheck] OpenAI error: {e}", debug)
                    return None
            _run_llm = _run_llm_openai

        if _run_llm is not None:
            # ── Step A: Extract claims (LLM sees ONLY your article, no evidence) ──
            extract_msgs = build_claim_extraction_messages(
                subject=task.subject, article_source=task.candidate,
                article_text=cand_text_clean, max_claims=args.max_claims,
                max_chars=getattr(args, 'max_article_chars', 0))
            extract_obj = _run_llm(extract_msgs) or {}
            raw_claims = extract_obj.get("claims") or []
            if isinstance(raw_claims, list):
                claim_texts = [str(c).strip() for c in raw_claims if c]
            else:
                claim_texts = []

            # ── Step B: Verify each claim in PARALLEL ──────────────────────────
            #    Each claim verified independently against evidence.
            #    All verification calls fire simultaneously.
            def _verify_one_claim(claim_text):
                if not claim_text:
                    return None
                verify_msgs = build_claim_verification_messages(
                    subject=task.subject, claim=claim_text,
                    evidence_source=task.baseline,
                    evidence_snippets=evidence_snips,
                    max_evidence_chars=getattr(args, 'max_evidence_chars', 0))
                verify_obj = _run_llm(verify_msgs) or {}
                verdict = verify_obj.get("verdict")
                if verdict in ("true", "false", "uncertain"):
                    conf = verify_obj.get("confidence")
                    return {
                        "claim": claim_text,
                        "verdict": verdict,
                        "confidence": float(conf) if isinstance(conf, (int, float)) else None,
                        "sources": verify_obj.get("sources", []),
                        "explanation": str(verify_obj.get("explanation", "")).strip(),
                    }
                return None

            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(len(claim_texts), 10)) as verify_pool:
                results = list(verify_pool.map(_verify_one_claim, claim_texts))
            claims = [r for r in results if r is not None]

    audit_save_claims(subject=task.subject, candidate=task.candidate,
                      baseline=task.baseline, hop=task.ours_hop,
                      model_name=model_name, claims=claims,
                      evidence_snippets=evidence_snips)

    # ── 8. Link checks ────────────────────────────────────────────────────────
    links_from_markup, link_checks = [], []
    if task.candidate == "ours" and args.check_links:
        links_from_markup = ours_article.get("links_from_markup") or []
        if isinstance(links_from_markup, list):
            for title in links_from_markup[:50]:
                link_checks.append(wiki_check_link_exists(str(title), debug=debug))

    # ── 9. Write record ───────────────────────────────────────────────────────
    record = _build_record(task, cand_meta, True, baseline_info, evidence_snips,
                           {"claims": claims}, availability, similarity_metrics,
                           links_from_markup, link_checks, args)
    _append_jsonl(out_path, record)
    return task.subject, True


def _build_record(task, cand_meta, cand_found, baseline_info, evidence_snips,
                  factcheck, availability, similarity, links_markup, link_checks, args):
    base_src_urls = [s.get("url", "") for s in (baseline_info.get("sources") or []) if s.get("url")]
    return {
        "subject": task.subject, "hop": task.ours_hop,
        "ours_article_idx": task.ours_article_idx,
        "candidate": task.candidate, "candidate_meta": cand_meta,
        "candidate_found": cand_found,
        "candidate_url": cand_meta.get("url"),
        "baseline": task.baseline,
        "baseline_found": baseline_info.get("found", False),
        "baseline_url": baseline_info.get("url"),
        "baseline_source_urls": base_src_urls,
        "baseline_source_domains": list({_extract_domain(u) for u in base_src_urls if u}),
        "evidence_snippets": evidence_snips,
        "article_factcheck": factcheck,
        "wikipedia_available": availability.get("wikipedia_available", False),
        "wikipedia_redirected": availability.get("wikipedia_redirected", False),
        "wikipedia_redirect_target": availability.get("wikipedia_redirect_target"),
        "wikipedia_url": availability.get("wikipedia_url"),
        "grokipedia_available": availability.get("grokipedia_available", False),
        "grokipedia_url": availability.get("grokipedia_url"),
        "similarity_metrics": similarity,
        "links_from_markup": links_markup,
        "link_checks": link_checks,
        "model": getattr(args, "_fact_model_name", args.fact_model_key),
        "mode": args.mode,
        "ts": _ts(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH MODE — uses cleaned text for similarity + claim extraction
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BatchTaskState:
    task: EvalTask
    ours_article: Dict[str, Any]
    candidate_text: str
    candidate_text_clean: str
    candidate_meta: Dict[str, Any]
    baseline_text: str
    baseline_found: bool
    baseline_info: Dict[str, Any]
    evidence_snippets: List[Dict[str, str]]
    availability: Dict[str, Any]
    similarity: Dict[str, Any] = field(default_factory=dict)
    claims: List[Dict[str, Any]] = field(default_factory=list)


def run_batch_mode(tasks, subject_to_ours, args, out_path, batch_dir) -> int:
    debug = args.debug
    model = args.batch_model or "gpt-4.1-nano"
    max_fetch = getattr(args, "max_fetch_pages", 3)
    is_calibrated = getattr(args, '_is_calibrated', False)

    _dbg(f"[batch] Phase 1: collecting texts ({len(tasks)} tasks)...", True)
    states: List[BatchTaskState] = []
    excl = _build_exclude_domains(args)
    ngram_values = [int(n) for n in args.ngram_n.split(",")] if args.ngram_n else [1, 2, 3]

    for i, task in enumerate(tasks):
        ours = subject_to_ours.get(task.subject) or {}
        ci = get_candidate_text(task.candidate, task.subject, ours,
                               debug=debug, is_calibrated=is_calibrated)
        ct = (ci.get("text") or "").strip()
        ct_clean = (ci.get("text_clean") or ct).strip()
        cm = ci.get("meta") or {}

        bi = get_baseline_text(task.baseline, task.subject, args.webrag_endpoint,
                               debug=debug, exclude_domains=excl, max_fetch_pages=max_fetch)
        bt = (bi.get("text") or "").strip()

        if task.baseline == "web" and bi.get("snippets"):
            ev = bi["snippets"]
        else:
            ev = get_evidence_snippets(task.baseline, task.subject,
                                       max_snippets=args.max_evidence_snippets,
                                       webrag_endpoint=args.webrag_endpoint,
                                       debug=debug, exclude_domains=excl,
                                       max_fetch_pages=max_fetch)

        av = check_subject_availability(task.subject, debug=debug)

        audit_save_candidate(task.subject, task.candidate, task.ours_hop, ct, cm)
        audit_save_baseline(task.subject, task.baseline, bt,
                            bi.get("sources") or [], ev,
                            bi.get("found", False), bi.get("meta", {}))
        audit_save_availability(task.subject, av)

        sim = {}
        if args.compute_similarity and ct_clean and bt:
            sim = compute_all_similarity_metrics(
                ct_clean, bt, ngram_values=ngram_values,
                semantic_provider=args.semantic_provider,
                semantic_model=args.semantic_model,
                compute_bertscore_flag=args.compute_bertscore,
                compute_stylistic=args.compute_stylistic, debug=debug)

        states.append(BatchTaskState(
            task=task, ours_article=ours,
            candidate_text=ct, candidate_text_clean=ct_clean, candidate_meta=cm,
            baseline_text=bt, baseline_found=bi.get("found", False),
            baseline_info=bi, evidence_snippets=ev, availability=av, similarity=sim))

        if (i + 1) % 50 == 0:
            _dbg(f"[batch] collected {i + 1}/{len(tasks)}", True)

    _flush_all_writers()

    if not args.run_factcheck:
        _dbg("[batch] skipping factcheck, writing results...", True)
        for st in states:
            rec = _build_record(st.task, st.candidate_meta, bool(st.candidate_text),
                                st.baseline_info, st.evidence_snippets, {"claims": []},
                                st.availability, st.similarity, [], [], args)
            _append_jsonl(out_path, rec)
            audit_save_claims(st.task.subject, st.task.candidate, st.task.baseline,
                              st.task.ours_hop, model, [], st.evidence_snippets)
        _flush_all_writers()
        return len(states)

    # Phase 2: extract claims
    _dbg(f"[batch] Phase 2: extracting claims...", True)
    ext_reqs, ext_map = [], {}
    for idx, st in enumerate(states):
        if not st.candidate_text_clean:
            continue
        cid = re.sub(r"[^\w-]", "_", f"ext_{idx}_{st.task.subject[:40]}")[:64]
        msgs = build_claim_extraction_messages(st.task.subject, st.task.candidate,
                                               st.candidate_text_clean, args.max_claims,
                                               max_chars=getattr(args, 'max_article_chars', 0))
        ext_reqs.append(BatchRequest(custom_id=cid, messages=msgs))
        ext_map[cid] = idx

    if ext_reqs:
        ext_in = os.path.join(batch_dir, "extract_input.jsonl")
        ext_out = os.path.join(batch_dir, "extract_output.jsonl")
        _write_jsonl(ext_in, create_batch_input_file(ext_reqs, model))
        bid = submit_batch_job(ext_in, "claim-extraction", debug=debug)
        if bid:
            fid = wait_for_batch_completion(bid, poll_interval=args.batch_poll_interval, debug=debug)
            if fid:
                download_batch_results(fid, ext_out, debug=debug)
                for cid, res in parse_batch_results(ext_out).items():
                    if cid in ext_map:
                        parsed = res.get("parsed") or {}
                        raw_claims = parsed.get("claims") or []
                        if isinstance(raw_claims, list):
                            states[ext_map[cid]].claims = [str(c) for c in raw_claims if c]

    # Phase 3: verify claims
    _dbg(f"[batch] Phase 3: verifying claims...", True)
    ver_reqs, ver_map = [], {}
    for idx, st in enumerate(states):
        for ci, claim_text in enumerate(st.claims):
            cid = f"ver_{idx}_{ci}"
            msgs = build_claim_verification_messages(st.task.subject, claim_text,
                                                      st.task.baseline, st.evidence_snippets,
                                                      max_evidence_chars=getattr(args, 'max_evidence_chars', 0))
            ver_reqs.append(BatchRequest(custom_id=cid, messages=msgs))
            ver_map[cid] = (idx, ci)

    verified_claims: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    if ver_reqs:
        ver_in = os.path.join(batch_dir, "verify_input.jsonl")
        ver_out = os.path.join(batch_dir, "verify_output.jsonl")
        _write_jsonl(ver_in, create_batch_input_file(ver_reqs, model))
        bid = submit_batch_job(ver_in, "claim-verification", debug=debug)
        if bid:
            fid = wait_for_batch_completion(bid, poll_interval=args.batch_poll_interval, debug=debug)
            if fid:
                download_batch_results(fid, ver_out, debug=debug)
                for cid, res in parse_batch_results(ver_out).items():
                    if cid in ver_map:
                        idx, ci = ver_map[cid]
                        parsed = res.get("parsed") or {}
                        verdict = parsed.get("verdict")
                        if verdict in ("true", "false", "uncertain"):
                            verified_claims[idx].append({
                                "claim": states[idx].claims[ci] if ci < len(states[idx].claims) else "",
                                "verdict": verdict,
                                "confidence": parsed.get("confidence"),
                                "sources": parsed.get("sources", []),
                                "explanation": parsed.get("explanation", ""),
                            })

    # Phase 4: write results
    _dbg(f"[batch] Phase 4: writing results...", True)
    for idx, st in enumerate(states):
        final_claims = verified_claims.get(idx, [])
        rec = _build_record(st.task, st.candidate_meta, bool(st.candidate_text),
                            st.baseline_info, st.evidence_snippets,
                            {"claims": final_claims}, st.availability,
                            st.similarity, [], [], args)
        _append_jsonl(out_path, rec)
        audit_save_claims(st.task.subject, st.task.candidate, st.task.baseline,
                          st.task.ours_hop, model, final_claims, st.evidence_snippets)

    _flush_all_writers()
    return len(states)


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY & CSV
# ═══════════════════════════════════════════════════════════════════════════════

SUMMARY_COLS = [
    "candidate", "baseline", "subject", "hop",
    "n_claims", "n_true", "n_false", "n_uncertain",
    "accuracy_true_vs_false", "true_rate", "false_rate", "uncertain_rate",
    "candidate_found", "baseline_found",
    "candidate_url", "baseline_source_urls",
    "wikipedia_available", "wikipedia_redirected",
    "grokipedia_available",
    "tfidf_cosine", "jaccard",
    "ngram_1_jaccard", "ngram_1_overlap", "ngram_2_jaccard", "ngram_2_overlap",
    "ngram_3_jaccard", "ngram_3_overlap",
    "semantic_cosine", "bertscore_f1", "stylistic_similarity",
    "combined_similarity", "candidate_word_count", "baseline_word_count",
]

CLAIM_COLS = [
    "candidate", "baseline", "subject", "hop",
    "claim_idx", "claim", "verdict", "confidence", "explanation",
    "cited_source_urls",
]


def summarize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    claims = (rec.get("article_factcheck") or {}).get("claims") or []
    nc = len(claims)
    nt = sum(1 for c in claims if c.get("verdict") == "true")
    nf = sum(1 for c in claims if c.get("verdict") == "false")
    nu = sum(1 for c in claims if c.get("verdict") == "uncertain")
    sm = rec.get("similarity_metrics") or {}

    return {
        "candidate": rec.get("candidate"), "baseline": rec.get("baseline"),
        "subject": rec.get("subject"), "hop": rec.get("hop"),
        "n_claims": nc, "n_true": nt, "n_false": nf, "n_uncertain": nu,
        "accuracy_true_vs_false": nt / (nt + nf) if (nt + nf) > 0 else "",
        "true_rate": nt / nc if nc > 0 else "",
        "false_rate": nf / nc if nc > 0 else "",
        "uncertain_rate": nu / nc if nc > 0 else "",
        "candidate_found": rec.get("candidate_found"),
        "baseline_found": rec.get("baseline_found"),
        "candidate_url": rec.get("candidate_url"),
        "baseline_source_urls": "|".join(rec.get("baseline_source_urls") or []),
        "wikipedia_available": rec.get("wikipedia_available"),
        "wikipedia_redirected": rec.get("wikipedia_redirected"),
        "grokipedia_available": rec.get("grokipedia_available"),
        **{k: sm.get(k, "") for k in [
            "tfidf_cosine", "jaccard",
            "ngram_1_jaccard", "ngram_1_overlap", "ngram_2_jaccard", "ngram_2_overlap",
            "ngram_3_jaccard", "ngram_3_overlap",
            "semantic_cosine", "bertscore_f1", "stylistic_similarity",
            "combined_similarity", "candidate_word_count", "baseline_word_count",
        ]},
    }


def extract_claim_rows(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    claims = (rec.get("article_factcheck") or {}).get("claims") or []
    rows = []
    for i, c in enumerate(claims):
        if not isinstance(c, dict):
            continue
        if c.get("verdict") not in ("true", "false", "uncertain"):
            continue
        rows.append({
            "candidate": rec.get("candidate"), "baseline": rec.get("baseline"),
            "subject": rec.get("subject"), "hop": rec.get("hop"),
            "claim_idx": i, "claim": c.get("claim", ""),
            "verdict": c.get("verdict"), "confidence": c.get("confidence", ""),
            "explanation": c.get("explanation", ""),
            "cited_source_urls": "|".join(
                s.get("url", "") for s in (c.get("cited_sources") or []) if s.get("url")),
        })
    return rows


def write_csv(rows, csv_path, cols):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def build_aggregate_results(srows, candidates, baselines, recs):
    agg: Dict[str, Any] = {}
    unique_subj = {r.get("subject") for r in recs if r.get("subject")}
    agg["total_subjects"] = len(unique_subj)

    for cand in candidates:
        c_recs = [r for r in srows if r.get("candidate") == cand]
        if not c_recs:
            continue
        nt = sum(r.get("n_true", 0) for r in c_recs)
        nf = sum(r.get("n_false", 0) for r in c_recs)
        nu = sum(r.get("n_uncertain", 0) for r in c_recs)
        nc = sum(r.get("n_claims", 0) for r in c_recs)
        agg[f"{cand}_total_claims"] = nc
        agg[f"{cand}_true"] = nt
        agg[f"{cand}_false"] = nf
        agg[f"{cand}_uncertain"] = nu
        agg[f"{cand}_accuracy"] = nt / (nt + nf) if (nt + nf) > 0 else ""
        agg[f"{cand}_true_rate"] = nt / nc if nc > 0 else ""
        agg[f"{cand}_false_rate"] = nf / nc if nc > 0 else ""
        agg[f"{cand}_uncertain_rate"] = nu / nc if nc > 0 else ""

        for mk in ["tfidf_cosine", "jaccard", "semantic_cosine", "combined_similarity"]:
            vals = [r.get(mk) for r in c_recs if isinstance(r.get(mk), (int, float))]
            if vals:
                agg[f"{cand}_avg_{mk}"] = sum(vals) / len(vals)

    return agg


def build_not_found_articles(recs):
    nf = {"wikipedia_not_found": [], "wikipedia_missing": [],
          "wikipedia_redirected": [], "wikipedia_redirect_details": {},
          "grokipedia_not_found": []}
    cw, cg = set(), set()
    for rec in recs:
        s = rec.get("subject", "")
        if not s:
            continue
        if s not in cw:
            cw.add(s)
            if not rec.get("wikipedia_available"):
                nf["wikipedia_not_found"].append(s)
                if rec.get("wikipedia_redirected"):
                    nf["wikipedia_redirected"].append(s)
                    t = rec.get("wikipedia_redirect_target")
                    if t:
                        nf["wikipedia_redirect_details"][s] = t
                else:
                    nf["wikipedia_missing"].append(s)
        if s not in cg:
            cg.add(s)
            if not rec.get("grokipedia_available"):
                nf["grokipedia_not_found"].append(s)
    nf["wikipedia_not_found_count"] = len(nf["wikipedia_not_found"])
    nf["grokipedia_not_found_count"] = len(nf["grokipedia_not_found"])
    nf["total_checked_subjects"] = len(cw | cg)
    return nf


# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLING
# ═══════════════════════════════════════════════════════════════════════════════

def pick_ours_article(arts, policy):
    if not arts:
        return {}
    if policy == "first":
        return arts[0]
    if policy == "last":
        return arts[-1]
    def hk(a):
        try:
            return int(a.get("hop", 10**9))
        except Exception:
            return 10**9 if policy == "minhop" else -(10**9)
    return (max if policy == "maxhop" else min)(arts, key=hk)


def build_tasks_from_run(run_dir, all_articles, candidates, baselines,
                         sample_frac, sample_min, sample_max, max_subjects,
                         seed, ours_pick_policy, min_words=0, debug=False):
    subj_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for idx, a in enumerate(all_articles):
        s = a.get("subject")
        if not isinstance(s, str) or not s.strip():
            continue
        text = a.get("wikitext") or ""
        if min_words > 0 and len(tokenize_simple(text)) < min_words:
            continue
        aa = dict(a)
        aa["_article_idx"] = idx
        subj_map[s.strip()].append(aa)

    subjects = sorted(subj_map.keys())
    n = len(subjects)
    if n == 0:
        return [], {}

    target = int(sample_frac * n) if sample_frac and sample_frac > 0 else n
    target = max(sample_min, target)
    if sample_max and sample_max > 0:
        target = min(target, sample_max)
    if max_subjects and max_subjects > 0:
        target = min(target, max_subjects)
    target = min(target, n)

    rng = random.Random(seed)
    rng.shuffle(subjects)
    picked = subjects[:target]

    subject_to_ours, tasks = {}, []
    for s in picked:
        ours = pick_ours_article(subj_map[s], ours_pick_policy)
        if not ours:
            continue
        idx = int(ours.get("_article_idx", -1))
        hop = ours.get("hop")
        subject_to_ours[s] = ours
        for cand in candidates:
            for bl in baselines:
                tasks.append(EvalTask(subject=s, ours_article_idx=idx,
                                      ours_hop=hop, candidate=cand, baseline=bl))

    _dbg(f"[sample] subjects={n}, picked={len(picked)}, tasks={len(tasks)}", debug)
    return tasks, subject_to_ours


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RUN DRIVER — detects calibrate mode, parallel pre-fetch, buffered I/O
# ═══════════════════════════════════════════════════════════════════════════════

def _clear_run_caches():
    """Clear per-run caches to prevent stale data between runs."""
    global _wiki_extract_cache, _candidate_cache, _evidence_cache
    global _baseline_cache, _availability_cache
    with _cache_lock:
        _candidate_cache.clear()
        _evidence_cache.clear()
        _baseline_cache.clear()
        _availability_cache.clear()
    with _wiki_extract_cache_lock:
        _wiki_extract_cache.clear()


def run_on_run_dir(args, run_dir: str):
    _clear_run_caches()

    in_path    = os.path.join(run_dir, args.articles_file)
    out_path   = os.path.join(run_dir, args.output_file)
    summary_csv = os.path.join(run_dir, args.summary_csv_file)
    claims_csv  = os.path.join(run_dir, args.claims_csv_file)
    agg_csv     = os.path.join(run_dir, args.aggregate_csv_file)
    nf_json     = os.path.join(run_dir, args.not_found_json_file)
    batch_dir   = os.path.join(run_dir, "factcheck_batches")

    audit_cand_jsonl  = os.path.join(run_dir, "audit_candidate_texts.jsonl")
    audit_base_jsonl  = os.path.join(run_dir, "audit_baseline_texts.jsonl")
    audit_claims_jsonl = os.path.join(run_dir, "audit_claims_full.jsonl")

    max_fetch = getattr(args, "max_fetch_pages", 3)

    print(f"\n{'=' * 60}")
    print(f"[factuality] run_dir        = {run_dir}")
    print(f"[factuality] mode           = {args.mode}")
    print(f"[factuality] candidates     = {args.candidates}")
    print(f"[factuality] baselines      = {args.baselines}")
    print(f"[factuality] factcheck      = {args.run_factcheck}")
    print(f"[factuality] similarity     = {args.compute_similarity}")
    print(f"[factuality] max_fetch_pages= {max_fetch}")
    print(f"[factuality] concurrency    = {args.concurrency}")

    excl_info = _build_exclude_domains(args)
    if excl_info:
        print(f"[factuality] web-excluded   = {sorted(excl_info)}")

    wb = check_web_backends()
    print(f"[factuality] web backends: ddg={wb.get('ddg_available')}, "
          f"searxng={'configured' if wb.get('searxng_api_base') else 'not set'}, "
          f"bs4={wb.get('bs4_available')}")
    print(f"{'=' * 60}")

    all_output_files = (out_path, summary_csv, claims_csv, agg_csv, nf_json,
                        audit_cand_jsonl, audit_base_jsonl, audit_claims_jsonl)

    if args.remove_checkedfact:
        for p in all_output_files:
            if os.path.exists(p):
                os.remove(p)
                print(f"  removed: {p}")
        for d in (batch_dir, os.path.join(run_dir, "audit")):
            if os.path.exists(d):
                shutil.rmtree(d)
                print(f"  removed dir: {d}")
        if args.clear_web_cache:
            n = clear_search_cache()
            print(f"  cleared {n} web search cache entries")
        return

    if not os.path.exists(in_path):
        print(f"[factuality] missing {in_path} — skipping.")
        return

    init_audit(run_dir)

    all_articles = _load_jsonl(in_path)
    print(f"[factuality] loaded {len(all_articles)} articles")

    # ── Detect calibrated confidence scores ───────────────────────────────────
    is_calibrated = detect_calibrate_mode(run_dir)
    args._is_calibrated = is_calibrated
    if is_calibrated:
        print(f"[factuality] ⚡ Calibrated mode — will strip (0.XX) confidence from eval text")
    else:
        print(f"[factuality] 📝 Baseline mode — cleaning wiki markup only")

    candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
    baselines  = [b.strip() for b in args.baselines.split(",") if b.strip()]
    candidates = [c for c in candidates if c in {"ours", "grokipedia", "wikipedia"}]
    baselines  = [b for b in baselines  if b in {"wikipedia", "webrag", "web"}]
    if not candidates or not baselines:
        print("[factuality] no valid candidates/baselines.")
        return

    tasks, subject_to_ours = build_tasks_from_run(
        run_dir=run_dir, all_articles=all_articles, candidates=candidates,
        baselines=baselines, sample_frac=args.sample_frac,
        sample_min=args.sample_min, sample_max=args.sample_max,
        max_subjects=args.max_subjects, seed=args.seed,
        ours_pick_policy=args.ours_pick, min_words=args.min_words, debug=args.debug)

    if not tasks:
        print("[factuality] no tasks.")
        return

    # ── Parallel availability pre-fetch (huge speedup) ────────────────────────
    subjects_list = list(subject_to_ours.keys())
    if len(subjects_list) > 3:
        n_prefetch_workers = min(30, len(subjects_list), args.concurrency)
        print(f"[factuality] pre-fetching Wikipedia/Grokipedia availability "
              f"({len(subjects_list)} subjects, {n_prefetch_workers} workers)...")
        t0_pf = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_prefetch_workers) as pool:
            futs = [pool.submit(check_subject_availability, s, args.debug)
                    for s in subjects_list]
            concurrent.futures.wait(futs)
        dt_pf = time.perf_counter() - t0_pf
        print(f"[factuality] pre-fetch done in {dt_pf:.1f}s")

    if not args.resume:
        for p in all_output_files:
            if os.path.exists(p):
                os.remove(p)

    start = time.perf_counter()

    if args.mode == "batch":
        os.makedirs(batch_dir, exist_ok=True)
        ok_count = run_batch_mode(tasks, subject_to_ours, args, out_path, batch_dir)
    else:
        # ── Online mode ───────────────────────────────────────────────────────
        fact_cfg = None
        if args.run_factcheck and HAS_SETTINGS:
            try:
                fact_cfg = settings.MODELS[args.fact_model_key].model_copy(deep=True)
                args._fact_model_name = getattr(fact_cfg, "model", None)
            except Exception:
                pass

        ok_count = 0
        total = len(tasks)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futs = []
            for t in tasks:
                ours = subject_to_ours.get(t.subject) or {}
                futs.append(pool.submit(process_task_online, t, ours, fact_cfg, args, out_path))
            for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
                try:
                    _, ok = fut.result()
                    if ok:
                        ok_count += 1
                except Exception as e:
                    _dbg(f"[online] error: {e}", args.debug)
                if i % 50 == 0 or i == total:
                    elapsed = time.perf_counter() - start
                    rate = i / elapsed if elapsed > 0 else 0
                    print(f"[factuality] {i}/{total} ({rate:.1f}/s, "
                          f"~{(total - i) / rate:.0f}s left)" if rate > 0 else
                          f"[factuality] {i}/{total}")

    # ── Flush buffered writes ─────────────────────────────────────────────────
    _flush_all_writers()

    dur = time.perf_counter() - start
    print(f"[factuality] done in {dur:.1f}s ({ok_count}/{len(tasks)} ok, "
          f"{ok_count / dur:.1f}/s)")

    # ── Generate summary CSVs ─────────────────────────────────────────────────
    if os.path.exists(out_path):
        recs   = _load_jsonl(out_path)
        srows  = [summarize_record(r) for r in recs]
        write_csv(srows, summary_csv, SUMMARY_COLS)

        crows = []
        for r in recs:
            crows.extend(extract_claim_rows(r))
        write_csv(crows, claims_csv, CLAIM_COLS)

        agg = build_aggregate_results(srows, candidates, baselines, recs)
        os.makedirs(os.path.dirname(agg_csv) or ".", exist_ok=True)
        with open(agg_csv, "w", newline="", encoding="utf-8") as f:
            cols = sorted(agg.keys())
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerow({c: agg.get(c, "") for c in cols})

        nf = build_not_found_articles(recs)
        with open(nf_json, "w", encoding="utf-8") as f:
            json.dump(nf, f, ensure_ascii=False, indent=2)

        print(f"[factuality] wrote: {summary_csv}")
        print(f"[factuality] wrote: {claims_csv}")
        print(f"[factuality] wrote: {agg_csv}")
        print(f"[factuality] wrote: {nf_json}")


# ═══════════════════════════════════════════════════════════════════════════════
# SWEEP
# ═══════════════════════════════════════════════════════════════════════════════

def find_run_dirs(root_dir: str) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root_dir):
        if "articles.jsonl" in filenames:
            out.append(dirpath)
    return sorted(out)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Factuality & similarity evaluation pipeline")

    ap.add_argument("--mode", choices=["online", "batch"], default="online")

    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--root-dir", default=None)

    ap.add_argument("--articles-file",       default="articles.jsonl")
    ap.add_argument("--output-file",         default="factchecks_v3.jsonl")
    ap.add_argument("--summary-csv-file",    default="factchecks_v3_summary.csv")
    ap.add_argument("--claims-csv-file",     default="factchecks_v3_claims.csv")
    ap.add_argument("--aggregate-csv-file",  default="aggregate_results_v3.csv")
    ap.add_argument("--not-found-json-file", default="not_found_articles_v3.json")

    ap.add_argument("--fact-model-key",       default="gpt-4.1-nano")
    ap.add_argument("--batch-model",          default="gpt-4.1-nano")
    ap.add_argument("--batch-poll-interval",  type=float, default=30.0)

    ap.add_argument("--candidates", default="ours,grokipedia",
                    help="Comma-sep: ours,grokipedia,wikipedia")
    ap.add_argument("--baselines", default="wikipedia",
                    help="Comma-sep: wikipedia,webrag,web")
    ap.add_argument("--webrag-endpoint", default=None)
    ap.add_argument("--max-evidence-snippets", type=int, default=5)
    ap.add_argument("--max-fetch-pages", type=int, default=3)
    ap.add_argument("--exclude-wikipedia-from-web", type=_str2bool, default=False)
    ap.add_argument("--exclude-domains-from-web", default="")

    ap.add_argument("--sample-frac",  type=float, default=0.1)
    ap.add_argument("--sample-min",   type=int,   default=20)
    ap.add_argument("--sample-max",   type=int,   default=500)
    ap.add_argument("--max-subjects", type=int,   default=0)
    ap.add_argument("--min-words",    type=int,   default=500)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--ours-pick", choices=["minhop", "maxhop", "first", "last"],
                    default="minhop")

    ap.add_argument("--run-factcheck", type=_str2bool, default=True)
    ap.add_argument("--max-claims",    type=int,       default=10)
    ap.add_argument("--max-article-chars", type=int, default=0,
                    help="Max chars of article text to send to LLM (0 = no limit, send full article). "
                         "Old default was 15000 which silently truncated long articles.")
    ap.add_argument("--max-evidence-chars", type=int, default=0,
                    help="Max chars of evidence text to send to LLM (0 = no limit, send everything). "
                         "For Wikipedia baseline this is the full page. For web, all fetched pages.")
    ap.add_argument("--check-links",   type=_str2bool, default=True)

    ap.add_argument("--compute-similarity", type=_str2bool, default=False)
    ap.add_argument("--compute-bertscore",  type=_str2bool, default=False)
    ap.add_argument("--compute-stylistic",  type=_str2bool, default=False)
    ap.add_argument("--ngram-n", default="1,2,3")
    ap.add_argument("--semantic-provider",
                    choices=["sentence-transformer", "openai"],
                    default="sentence-transformer")
    ap.add_argument("--semantic-model", default="all-MiniLM-L6-v2")

    ap.add_argument("--concurrency", type=int,   default=100,
                    help="Max concurrent tasks (threads). Higher = faster but more memory/connections.")
    ap.add_argument("--timeout",     type=float, default=90.0)
    ap.add_argument("--debug",       action="store_true")

    ap.add_argument("--resume",            action="store_true")
    ap.add_argument("--remove-checkedfact", action="store_true")
    ap.add_argument("--clear-web-cache",   action="store_true")
    ap.add_argument("--max-concurrent-runs", type=int, default=4)

    args = ap.parse_args()

    if args.run_dir:
        run_on_run_dir(args, os.path.abspath(args.run_dir))
        _flush_all_writers()
        return

    if not args.root_dir:
        raise SystemExit("Provide --run-dir or --root-dir")

    root = os.path.abspath(args.root_dir)
    run_dirs = find_run_dirs(root)
    print(f"[SWEEP] Found {len(run_dirs)} run dirs under {root}")

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, args.max_concurrent_runs)) as pool:
        futs = [pool.submit(run_on_run_dir, args, rd) for rd in run_dirs]
        for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            try:
                fut.result()
            except Exception as e:
                print(f"[SWEEP] error: {e}")
            print(f"[SWEEP] {i}/{len(futs)}")

    _flush_all_writers()


if __name__ == "__main__":
    main()


    
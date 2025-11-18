# # web/fetchers.py
# from __future__ import annotations
# import time, re
# import requests
# from typing import List, Dict, Any, Optional
# from urllib.parse import quote

# UA = "LLMPediaBot/0.1 (+contact: you@example.com)"

# class SourceDoc(Dict[str, Any]):
#     """
#     Canonical source object:
#     {
#       "source": "wikipedia"|"semanticscholar",
#       "url": "...",
#       "title": "...",
#       "text": "...",
#     }
#     """

# def _get_json(url: str, params: Optional[dict] = None, timeout: float = 20.0):
#     r = requests.get(url, params=params or {}, timeout=timeout, headers={"User-Agent": UA})
#     r.raise_for_status()
#     return r.json()

# def fetch_wikipedia(subject: str, lang: str = "en") -> List[SourceDoc]:
#     """
#     Uses REST summary + mobile sections (polite & ToS-friendly).
#     """
#     out: List[SourceDoc] = []
#     encoded = quote(subject.replace(" ", "_"), safe="_()")
#     base = f"https://{lang}.wikipedia.org/api/rest_v1/page"

#     # summary
#     try:
#         j = _get_json(f"{base}/summary/{encoded}")
#         title = j.get("title") or subject
#         url = j.get("content_urls", {}).get("desktop", {}).get("page") or j.get("content_urls", {}).get("mobile", {}).get("page")
#         extract = j.get("extract") or ""
#         if url and extract:
#             out.append(SourceDoc(source="wikipedia", url=url, title=title, text=extract))
#     except Exception:
#         pass
#     time.sleep(0.25)

#     # mobile-sections for more body (strip tags)
#     try:
#         j = _get_json(f"{base}/mobile-sections/{encoded}")
#         lead = (j.get("lead", {}) or {}).get("sections", [{}])[0].get("text", "") or ""
#         rest_secs = (j.get("remaining", {}) or {}).get("sections", []) or []
#         more = "\n\n".join([re.sub("<[^>]+>", " ", s.get("text", "") or "") for s in rest_secs])
#         text = re.sub(r"\s+", " ", (lead + "\n\n" + more)).strip()
#         if text:
#             url = f"https://{lang}.wikipedia.org/wiki/{encoded}"
#             out.append(SourceDoc(source="wikipedia", url=url, title=subject, text=text))
#     except Exception:
#         pass

#     return _dedupe(out)

# def fetch_semanticscholar_snippets(subject: str, limit: int = 2) -> List[SourceDoc]:
#     """
#     Semantic Scholar Graph API — abstracts only.
#     """
#     out: List[SourceDoc] = []
#     try:
#         j = _get_json(
#             "https://api.semanticscholar.org/graph/v1/paper/search",
#             params={"query": subject, "fields": "title,abstract,url", "limit": limit},
#             timeout=20.0,
#         )
#         for p in j.get("data", []) or []:
#             title = p.get("title") or ""
#             abstract = p.get("abstract") or ""
#             url = p.get("url") or ""
#             if abstract and url:
#                 out.append(SourceDoc(source="semanticscholar", url=url, title=title, text=abstract))
#     except Exception:
#         pass
#     time.sleep(0.25)
#     return _dedupe(out)

# def _dedupe(docs: List[SourceDoc]) -> List[SourceDoc]:
#     seen = set()
#     out: List[SourceDoc] = []
#     for d in docs:
#         key = (d.get("source"), d.get("url"))
#         if key in seen:
#             continue
#         seen.add(key)
#         # cap text
#         t = (d.get("text") or "")[:6000]
#         d["text"] = t
#         out.append(d)
#     return out

# def fetch_sources_for_subject(subject: str) -> List[SourceDoc]:
#     docs: List[SourceDoc] = []
#     docs += fetch_wikipedia(subject)
#     docs += fetch_semanticscholar_snippets(subject, limit=2)
#     return docs
# web_fetcher.py
from __future__ import annotations
import re, time, json, os, html
from typing import List, Tuple, Dict, Optional
from urllib.parse import urlencode
import requests
from bs4 import BeautifulSoup

USER_AGENT = "LLMPediaCrawler/0.1 (+https://llmpedia.net)"

def _clean_text(t: str) -> str:
    t = html.unescape(t or "")
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _fetch_url(url: str, timeout: float = 10.0) -> Tuple[str, str]:
    """Return (clean_text, raw_html)"""
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        r.raise_for_status()
        raw = r.text or ""
        soup = BeautifulSoup(raw, "html.parser")
        # remove script/style/nav/footer
        for bad in soup(["script","style","noscript","header","footer","nav","form","aside"]):
            bad.decompose()
        txt = _clean_text(soup.get_text(" "))
        return txt, raw
    except Exception:
        return "", ""

def search_bing(query: str, max_results: int = 5, timeout: float = 10.0) -> List[str]:
    """Very simple Bing web search via HTML (no key). Replace with official API if available."""
    url = f"https://www.bing.com/search?{urlencode({'q': query})}"
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        r.raise_for_status()
        soup = BeautifulSoup(r.text or "", "html.parser")
        urls = []
        for li in soup.select("li.b_algo h2 a"):
            href = li.get("href")
            if href and href.startswith("http"):
                urls.append(href)
            if len(urls) >= max_results:
                break
        return urls
    except Exception:
        return []

def gather_context(subject: str, max_pages: int = 5, max_chars: int = 4000, debug_dir: Optional[str] = None) -> Dict:
    """Search + fetch a few pages. Returns { 'subject', 'sources': [...], 'context': '...' }"""
    sources = []
    chunks = []
    urls = search_bing(subject, max_results=max_pages)
    for idx, u in enumerate(urls):
        txt, raw = _fetch_url(u)
        if txt:
            sources.append(u)
            # take a head/tail slice to reduce boilerplate
            head = txt[: max_chars // (2 * max_pages)]
            tail = txt[-max_chars // (4 * max_pages):]
            chunks.append(head)
            if tail:
                chunks.append(tail)
        if debug_dir:
            try:
                os.makedirs(debug_dir, exist_ok=True)
                with open(os.path.join(debug_dir, f"page_{idx:02d}.html"), "w", encoding="utf-8") as f:
                    f.write(raw)
            except Exception:
                pass
        if sum(len(c) for c in chunks) >= max_chars:
            break
    ctx = _clean_text(" ".join(chunks))[:max_chars]
    return {"subject": subject, "sources": sources, "context": ctx}

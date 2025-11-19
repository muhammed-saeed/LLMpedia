#!/usr/bin/env python
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter, Retry

# Wikipedia API endpoint
_WIKI_API_URL = "https://en.wikipedia.org/w/api.php"

# Global session (reused)
_session: Optional[requests.Session] = None


def get_wiki_session() -> requests.Session:
    """
    Return a shared requests.Session configured with:
    - Friendly User-Agent
    - Retry logic for transient errors
    """
    global _session
    if _session is not None:
        return _session

    # You can override this in env if you want
    ua = os.environ.get(
        "LLMPEDIA_WIKI_UA",
        "LLMPediaFactCheck/0.1 (https://example.com/contact; muhammed.saeed@example.com)",
    )

    s = requests.Session()
    s.headers.update({"User-Agent": ua})

    retries = Retry(
        total=5,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    _session = s
    return _session


def wiki_search_snippets(
    query: str,
    limit: int = 3,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Search Wikipedia via the search API and return up to `limit` snippets.

    Returns:
        (snippets, error_message)
        - snippets: list of dicts with {title, snippet}
        - error_message: None on success, or a string describing the error
    """
    session = get_wiki_session()

    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "utf8": 1,
        "srlimit": limit,
    }

    try:
        resp = session.get(_WIKI_API_URL, params=params, timeout=10)
        resp.raise_for_status()
    except requests.HTTPError as e:
        return [], f"HTTPError: {e}"
    except requests.RequestException as e:
        return [], f"RequestException: {e}"

    try:
        data = resp.json()
    except Exception as e:
        return [], f"JSON decode error: {e}"

    search = data.get("query", {}).get("search", [])
    snippets: List[Dict[str, Any]] = []
    for item in search[:limit]:
        title = item.get("title", "")
        snippet_html = item.get("snippet", "")
        snippets.append(
            {
                "title": title,
                "snippet": snippet_html,
            }
        )
    return snippets, None


def wiki_check_link_exists(title: str) -> Dict[str, Any]:
    """
    Use the search API to decide whether a link title appears to exist.

    Returns a dict like:
    {
      "title": <input title>,
      "exists_on_wikipedia": True/False/None,
      "top_title": <best match or None>,
      "note": <string>
    }
    """
    snippets, err = wiki_search_snippets(title, limit=1)
    if err is not None:
        return {
            "title": title,
            "exists_on_wikipedia": None,
            "top_title": None,
            "note": f"error during search: {err}",
        }

    if not snippets:
        return {
            "title": title,
            "exists_on_wikipedia": False,
            "top_title": None,
            "note": "no search results",
        }

    top = snippets[0]
    return {
        "title": title,
        "exists_on_wikipedia": True,
        "top_title": top.get("title"),
        "note": "search result found",
    }

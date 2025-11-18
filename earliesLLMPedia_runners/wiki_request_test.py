#!/usr/bin/env python
"""
wiki_request_test.py

Small standalone script to test Wikipedia HTTP requests:
- search for a subject
- fetch intro extracts
- optionally check if link titles exist

Run, for example:

  python wiki_request_test.py --query "Penny (The Big Bang Theory)"

If this script works without 403 errors, you can reuse the same
session + headers in your factcheck_articles.py.
"""

import argparse
import textwrap
from typing import List, Dict, Any, Optional

import requests

# --- Config: set a polite User-Agent (Wikipedia strongly recommends this) ---
USER_AGENT = (
    "LLMPediaFactCheck/0.1 "
    "(https://example.com/contact; muhammed.saeed@example.com)"
)


def make_wiki_session() -> requests.Session:
    """
    Create a Session with a User-Agent header suitable for Wikipedia.
    """
    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT})
    return sess


# --- Wikipedia helpers ------------------------------------------------------


def wiki_search(
    sess: requests.Session, query: str, limit: int = 3
) -> List[Dict[str, Any]]:
    """
    Use the MediaWiki search API to find pages related to the query.
    Returns a list of search result dicts.
    """
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "utf8": 1,
        "srlimit": limit,
    }
    r = sess.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    return data.get("query", {}).get("search", [])


def wiki_extracts_by_pageids(
    sess: requests.Session, page_ids: List[int]
) -> Dict[int, Dict[str, Any]]:
    """
    Fetch plain-text intro extracts for a list of page IDs.
    Returns mapping {pageid: page_dict}.
    """
    if not page_ids:
        return {}

    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": 1,
        "explaintext": 1,
        "pageids": "|".join(str(pid) for pid in page_ids),
        "format": "json",
        "utf8": 1,
    }
    r = sess.get(url, params=params, timeout=10)
    r.raise_for_status()
    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    # keys in 'pages' are strings of pageid
    out: Dict[int, Dict[str, Any]] = {}
    for pid_str, page in pages.items():
        try:
            pid = int(pid_str)
        except Exception:
            continue
        out[pid] = page
    return out


def retrieve_subject_evidence(
    sess: requests.Session, subject: str, max_snippets: int = 3
) -> List[str]:
    """
    High-level helper: given a subject string, return up to max_snippets
    of intro text snippets from the most relevant Wikipedia pages.
    """
    results = wiki_search(sess, subject, limit=max_snippets)
    if not results:
        return []

    page_ids = [int(r["pageid"]) for r in results[:max_snippets]]
    pages = wiki_extracts_by_pageids(sess, page_ids)

    snippets: List[str] = []
    for pid in page_ids:
        page = pages.get(pid)
        if not isinstance(page, dict):
            continue
        title = page.get("title", "")
        extract = (page.get("extract") or "").strip()
        if not extract:
            continue
        if len(extract) > 600:
            extract = extract[:600] + "…"
        snippets.append(f"[{title}] {extract}")
    return snippets


def check_link_exists_on_wikipedia(
    sess: requests.Session, title: str
) -> Dict[str, Optional[str]]:
    """
    Simple "does this look like a real page?" check using the search API.

    Returns a small dict:
      {
        "exists": True/False/None,
        "top_title": <best matching title or None>,
        "note": <string>
      }
    """
    try:
        results = wiki_search(sess, title, limit=1)
        if not results:
            return {
                "exists": False,
                "top_title": None,
                "note": "no search results",
            }
        top = results[0]
        return {
            "exists": True,
            "top_title": top.get("title"),
            "note": "search result found",
        }
    except requests.HTTPError as e:
        return {
            "exists": None,
            "top_title": None,
            "note": f"HTTP error: {e}",
        }
    except Exception as e:
        return {
            "exists": None,
            "top_title": None,
            "note": f"other error: {type(e).__name__}: {e}",
        }


# --- CLI + main -------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Test Wikipedia API requests (search + extracts + link check)."
    )
    ap.add_argument(
        "--query",
        default="Penny (The Big Bang Theory)",
        help="Subject to search on Wikipedia.",
    )
    ap.add_argument(
        "--max-snippets",
        type=int,
        default=3,
        help="Max intro snippets to fetch for the subject.",
    )
    ap.add_argument(
        "--check-link",
        default=None,
        help="Optional: also check if this link title exists on Wikipedia.",
    )

    args = ap.parse_args()

    print(f"[test] Using User-Agent: {USER_AGENT}")
    print(f"[test] Querying subject: {args.query!r}")
    sess = make_wiki_session()

    # 1) Subject evidence
    try:
        snippets = retrieve_subject_evidence(sess, args.query, max_snippets=args.max_snippets)
        if not snippets:
            print("[test] No snippets found.")
        else:
            print(f"[test] Retrieved {len(snippets)} snippet(s):\n")
            for i, sn in enumerate(snippets, 1):
                print(f"--- snippet {i} ---")
                print(textwrap.fill(sn, width=100))
                print()
    except requests.HTTPError as e:
        # This is where you'd see 403, 429, etc
        print(f"[test] HTTPError while retrieving subject evidence: {e}")
    except Exception as e:
        print(f"[test] Unexpected error while retrieving subject evidence: {type(e).__name__}: {e}")

    # 2) Optional: check a link title
    if args.check_link:
        print(f"\n[test] Checking link existence for title: {args.check_link!r}")
        info = check_link_exists_on_wikipedia(sess, args.check_link)
        print(f"[test] link_exists={info['exists']}, top_title={info['top_title']!r}, note={info['note']}")


if __name__ == "__main__":
    main()

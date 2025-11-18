#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

from wiki_client import wiki_search_snippets, wiki_check_link_exists


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        raise FileNotFoundError(f"file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if isinstance(rec, dict):
                records.append(rec)
    return records


def main():
    ap = argparse.ArgumentParser(
        description="ONLY test Wikipedia requests (subjects + links) from LLMPedia JSONL."
    )
    ap.add_argument(
        "--run-dir",
        required=True,
        help="LLMPedia run directory.",
    )
    ap.add_argument(
        "--file",
        default="articles.jsonl",
        help="JSONL file inside run-dir (e.g., 'articles.jsonl' or 'factchecks_articles.jsonl').",
    )
    ap.add_argument(
        "--max-articles",
        type=int,
        default=10,
        help="Max number of articles to test.",
    )
    ap.add_argument(
        "--max-links",
        type=int,
        default=5,
        help="Max number of links_from_markup per article to test.",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Sleep in seconds between requests (be nice to Wikipedia).",
    )

    args = ap.parse_args()

    path = os.path.join(args.run_dir, args.file)
    print(f"[bulk-test] reading from: {path}")
    records = _load_jsonl(path)
    if not records:
        print("[bulk-test] no records found, exiting.")
        return

    records = records[: args.max_articles]

    subj_ok = subj_err = 0
    link_ok = link_err = 0

    for idx, rec in enumerate(records, start=1):
        subject = rec.get("subject", "?")
        print(f"\n=== ARTICLE {idx} / {len(records)}: {subject!r} ===")

        # ---- subject-level search ----
        snippets, err = wiki_search_snippets(subject, limit=3)
        if err is not None:
            subj_err += 1
            print(f"[subject] ERROR for {subject!r}: {err}")
        else:
            subj_ok += 1
            print(f"[subject] got {len(snippets)} snippet(s):")
            for i, s in enumerate(snippets, 1):
                title = s.get("title", "")
                snippet = (s.get("snippet") or "")[:180]
                print(f"  - {i}: [{title}] {snippet}...")
        time.sleep(args.sleep)

        # ---- link-level checks ----
        links = rec.get("links_from_markup") or rec.get("links") or []
        if not links:
            print("[links] no links_from_markup in this record.")
            continue

        to_check = links[: args.max_links]
        print(f"[links] checking first {len(to_check)} link(s):")
        for title in to_check:
            res = wiki_check_link_exists(title)
            exists = res.get("exists_on_wikipedia")
            top_title = res.get("top_title")
            note = res.get("note")

            if exists is None:
                link_err += 1
                print(f"  - {title!r}: ERROR (exists=None) note={note}")
            else:
                link_ok += 1
                print(
                    f"  - {title!r}: exists={exists}, "
                    f"top_title={top_title!r}, note={note}"
                )
            time.sleep(args.sleep)

    # ---- summary ----
    print("\n=== SUMMARY ===")
    print(f"subjects: ok={subj_ok}, errors={subj_err}")
    print(f"links:    ok={link_ok}, errors={link_err}")


if __name__ == "__main__":
    main()

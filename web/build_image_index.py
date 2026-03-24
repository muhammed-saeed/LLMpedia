#!/usr/bin/env python3
"""
build_image_index.py — Download article images from Wikipedia/Wikidata/Commons.

KEY DESIGN:
  - Global image_store under deploy/ (or run_dir/ if no --global-store)
  - In-memory dictionary cache; flushed to disk every SAVE_EVERY images
  - Check global store before downloading (dedup across runs)
  - Parallel downloads with configurable workers

Usage:
    python3 build_image_index.py /path/to/run_dir
    python3 build_image_index.py /path/to/run_dir --global-store /path/to/deploy/image_store
    python3 build_image_index.py /path/to/run_dir --workers 40 --force
    python3 build_image_index.py /path/to/run_dir --dry-run
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ─── Config ──────────────────────────────────────────────────────────────────

WIKIPEDIA_API   = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API    = "https://www.wikidata.org/w/api.php"
COMMONS_API     = "https://commons.wikimedia.org/w/api.php"
USER_AGENT      = "LLMPedia-ImageIndexer/2.0 (https://gptkb.org; research project)"

REQUEST_DELAY   = 0.05
SAVE_EVERY      = 100   # flush cache to disk every N resolved subjects
TIMEOUT         = 15
MAX_THUMB_WIDTH = 500
DEFAULT_WORKERS = 40
WIKI_BATCH_SIZE = 50
IMAGE_STORE_DIR = "image_store"


# ─── HTTP helpers ────────────────────────────────────────────────────────────

def _api_get(url: str) -> Optional[dict]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None

def _download_file(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            dest.write_bytes(resp.read())
        return True
    except (urllib.error.URLError, OSError):
        return False

def _safe_filename(subject: str, ext: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', subject)
    safe = safe.strip(". ")[:200]
    return f"{safe}.{ext}"

def _strip_html_dedup(text: str) -> str:
    inner_texts = re.findall(r">([^<]+)<", text)
    clean = re.sub(r"<[^>]+>", "", text).strip()
    for inner in inner_texts:
        inner = inner.strip()
        if not inner: continue
        doubled = inner + inner
        if doubled in clean:
            clean = clean.replace(doubled, inner)
    return re.sub(r"\s+", " ", clean).strip()

def _detect_ext(url: str, default: str = "jpg") -> str:
    path = urllib.parse.urlparse(url).path
    if "thumb.php" in path: return default
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else default
    if "/" in ext: ext = ext.split("/")[0]
    if ext not in ("jpg", "jpeg", "png", "svg", "webp", "gif"): return default
    return ext


# ─── Global store index ─────────────────────────────────────────────────────

class ImageStoreIndex:
    """Fast lookup dictionary for the global image store.
    Maps safe_filename -> True for files that exist on disk.
    Avoids repeated os.path.exists() calls."""

    def __init__(self, store_dir: Path):
        self.store_dir = store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._index: Dict[str, bool] = {}
        self._lock = threading.Lock()
        self._rebuild_index()

    def _rebuild_index(self):
        """Scan directory once at startup."""
        t0 = time.time()
        count = 0
        for f in self.store_dir.iterdir():
            if f.is_file():
                self._index[f.name] = True
                count += 1
        elapsed = time.time() - t0
        print(f"  Image store index: {count:,} files scanned in {elapsed:.2f}s")

    def exists(self, filename: str) -> bool:
        with self._lock:
            return filename in self._index

    def mark_added(self, filename: str):
        with self._lock:
            self._index[filename] = True

    def filepath(self, filename: str) -> Path:
        return self.store_dir / filename

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._index)


# ─── Commons metadata ───────────────────────────────────────────────────────

def get_commons_metadata(filename: str) -> Dict[str, str]:
    result = {"license": "", "license_url": "", "artist": "", "description": "", "commons_url": ""}
    params = urllib.parse.urlencode({
        "action": "query", "titles": filename, "prop": "imageinfo",
        "iiprop": "extmetadata|url", "format": "json",
    })
    data = _api_get(f"{COMMONS_API}?{params}")
    if not data: return result
    pages = data.get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        if page_id == "-1": continue
        info_list = page.get("imageinfo", [])
        if not info_list: continue
        info = info_list[0]
        result["commons_url"] = info.get("descriptionurl", "")
        ext = info.get("extmetadata", {})
        result["license"] = ext.get("LicenseShortName", {}).get("value", "")
        result["license_url"] = ext.get("LicenseUrl", {}).get("value", "")
        raw = ext.get("Artist", {}).get("value", "")
        if raw: result["artist"] = _strip_html_dedup(raw)[:200]
        desc = ext.get("ImageDescription", {}).get("value", "")
        if desc: result["description"] = _strip_html_dedup(desc)[:300]
    return result


# ─── Wikipedia batch lookup ──────────────────────────────────────────────────

def wikipedia_batch_lookup(subjects: List[str], thumb_width: int = MAX_THUMB_WIDTH
                           ) -> Dict[str, Optional[Dict[str, str]]]:
    results: Dict[str, Optional[Dict[str, str]]] = {}
    titles_str = "|".join(s.replace(" ", "_") for s in subjects)
    params = urllib.parse.urlencode({
        "action": "query", "titles": titles_str, "prop": "pageimages",
        "format": "json", "pithumbsize": thumb_width, "pilicense": "any", "redirects": "1",
    })
    data = _api_get(f"{WIKIPEDIA_API}?{params}")
    if not data:
        for s in subjects: results[s] = None
        return results

    normalized = {}
    for e in data.get("query", {}).get("normalized", []):
        normalized[e.get("from", "")] = e.get("to", "")
    redirects_map = {}
    for e in data.get("query", {}).get("redirects", []):
        redirects_map[e.get("from", "")] = e.get("to", "")

    title_to_subject = {}
    for s in subjects:
        title = s.replace(" ", "_")
        title_sp = s
        if title in normalized: title_sp = normalized[title]
        elif s in normalized: title_sp = normalized[s]
        if title_sp in redirects_map: title_sp = redirects_map[title_sp]
        title_to_subject[title_sp] = s
        title_to_subject[s] = s

    pages = data.get("query", {}).get("pages", {})
    found = set()
    for page_id, page in pages.items():
        if page_id == "-1": continue
        page_title = page.get("title", "")
        image_file = page.get("pageimage")
        thumb = page.get("thumbnail", {}).get("source")
        if not image_file or not thumb: continue
        original = title_to_subject.get(page_title) or title_to_subject.get(page_title.replace("_", " "))
        if not original: continue
        encoded = urllib.parse.quote(page_title.replace(" ", "_"))
        results[original] = {
            "image_url": thumb, "filename": f"File:{image_file}",
            "page_url": f"https://en.wikipedia.org/wiki/{encoded}",
        }
        found.add(original)

    for s in subjects:
        if s not in found: results.setdefault(s, None)
    return results


# ─── Wikidata fallback ───────────────────────────────────────────────────────

def wikidata_image_lookup(subject: str, thumb_width: int = MAX_THUMB_WIDTH
                          ) -> Optional[Dict[str, str]]:
    params = urllib.parse.urlencode({
        "action": "wbsearchentities", "search": subject,
        "language": "en", "limit": "3", "format": "json",
    })
    data = _api_get(f"{WIKIDATA_API}?{params}")
    if not data: return None
    for result in data.get("search", []):
        entity_id = result.get("id")
        if not entity_id: continue
        params2 = urllib.parse.urlencode({
            "action": "wbgetclaims", "entity": entity_id, "property": "P18", "format": "json",
        })
        data2 = _api_get(f"{WIKIDATA_API}?{params2}")
        if not data2: continue
        claims = data2.get("claims", {}).get("P18", [])
        if not claims: continue
        image_name = claims[0].get("mainsnak", {}).get("datavalue", {}).get("value")
        if not image_name: continue
        enc = urllib.parse.quote(image_name.replace(" ", "_"))
        return {
            "image_url": f"https://commons.wikimedia.org/w/thumb.php?f={enc}&w={thumb_width}",
            "filename": f"File:{image_name}",
            "page_url": f"https://www.wikidata.org/wiki/{entity_id}",
        }
    return None


# ─── Download + metadata ────────────────────────────────────────────────────

def resolve_and_download(
    subject: str, img_info: Dict[str, str],
    store: ImageStoreIndex, source_label: str,
) -> Optional[Dict[str, Any]]:
    image_url = img_info.get("image_url", "")
    filename_commons = img_info.get("filename", "")
    if not image_url: return None

    ext = _detect_ext(image_url)
    local_filename = _safe_filename(subject, ext)

    # Check global store first — skip download if already there
    if not store.exists(local_filename):
        dest = store.filepath(local_filename)
        ok = _download_file(image_url, dest)
        if not ok: return None
        if dest.stat().st_size < 500:
            dest.unlink(missing_ok=True)
            return None
        store.mark_added(local_filename)

    meta = get_commons_metadata(filename_commons) if filename_commons else {}
    return {
        "store_path": f"{IMAGE_STORE_DIR}/{local_filename}",
        "source_url": meta.get("commons_url") or img_info.get("page_url", ""),
        "image_url": image_url,
        "license": meta.get("license", ""),
        "license_url": meta.get("license_url", ""),
        "artist": meta.get("artist", ""),
        "source": source_label,
    }


def _wikidata_worker(subject: str, store: ImageStoreIndex, thumb_width: int
                     ) -> Tuple[str, Optional[Dict[str, Any]]]:
    img_info = wikidata_image_lookup(subject, thumb_width)
    if img_info:
        entry = resolve_and_download(subject, img_info, store, "wikidata")
        if entry: return subject, entry
    return subject, None


# ─── Discovery helpers ───────────────────────────────────────────────────────

def find_run_dirs(root: Path) -> List[Path]:
    if not root.is_dir(): return []
    for name in ("articles.jsonl", "articles_wikitext.jsonl"):
        if (root / name).exists(): return [root]
    found = set()
    for name in ("articles.jsonl", "articles_wikitext.jsonl"):
        for p in root.rglob(name): found.add(p.parent)
    return sorted(found)

def load_subjects_for_run(run_dir: Path) -> set:
    subjects = set()
    for name in ("articles.jsonl", "articles_wikitext.jsonl"):
        path = run_dir / name
        if not path.exists(): continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    s = json.loads(line).get("subject", "").strip()
                    if s: subjects.add(s)
                except Exception: pass
        break
    return subjects

def load_cache(run_dir: Path) -> dict:
    p = run_dir / "gptkb_image_cache.json"
    if p.exists():
        try: return json.loads(p.read_text(encoding="utf-8"))
        except Exception: pass
    return {}

def save_cache(run_dir: Path, cache: dict):
    p = run_dir / "gptkb_image_cache.json"
    p.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


# ─── Batch cache save ───────────────────────────────────────────────────────

def _flush_resolved(resolved: Dict[str, Any], all_caches: Dict[Path, dict],
                    run_dirs: List[Path], run_subject_map: Dict[Path, set], force: bool):
    for rd in run_dirs:
        cache = all_caches[rd]
        dirty = False
        for s, entry in resolved.items():
            if s in run_subject_map[rd]:
                if force or s not in cache:
                    cache[s] = entry
                    dirty = True
        if dirty:
            save_cache(rd, cache)


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Download Wikipedia/Wikidata images for LLMPedia")
    parser.add_argument("root", help="Run dir or parent dir")
    parser.add_argument("--thumb", type=int, default=MAX_THUMB_WIDTH)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--force", action="store_true", help="Re-download all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--global-store", type=str, default=None,
                        help="Path to global image_store (e.g. deploy/image_store). "
                             "All images go here; dedup across runs.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"Error: {root} does not exist"); sys.exit(1)

    run_dirs = find_run_dirs(root)
    if not run_dirs:
        print("No articles found."); sys.exit(1)

    print(f"Found {len(run_dirs)} run dir(s)")
    for rd in run_dirs:
        print(f"  - {rd}")

    # Collect subjects
    print("\nCollecting subjects...")
    all_subjects: set = set()
    run_subject_map: Dict[Path, set] = {}
    for rd in run_dirs:
        subjects = load_subjects_for_run(rd)
        run_subject_map[rd] = subjects
        all_subjects |= subjects
    print(f"  Unique subjects: {len(all_subjects):,}")

    all_caches: Dict[Path, dict] = {}
    already_resolved: set = set()
    for rd in run_dirs:
        cache = load_cache(rd)
        all_caches[rd] = cache
        already_resolved |= set(cache.keys())

    subjects_to_resolve = all_subjects - already_resolved if not args.force else all_subjects
    print(f"  Already resolved: {len(already_resolved):,}")
    print(f"  Need resolution:  {len(subjects_to_resolve):,}")

    if not subjects_to_resolve:
        print("\nAll subjects already resolved. Use --force to re-download.")
        _print_summary(all_caches, run_dirs); return

    if args.dry_run:
        print(f"\n[DRY RUN] Would query for {len(subjects_to_resolve):,} subjects")
        _print_summary(all_caches, run_dirs); return

    # Set up image store (global or per-run)
    if args.global_store:
        store_dir = Path(args.global_store).resolve()
    else:
        store_dir = run_dirs[0] / IMAGE_STORE_DIR

    store = ImageStoreIndex(store_dir)
    print(f"\nImage store: {store_dir}  ({store.count:,} existing)")
    print(f"  Thumb: {args.thumb}px  Workers: {args.workers}")
    print(f"  Cache flush interval: every {SAVE_EVERY} subjects\n")

    subjects_list = sorted(subjects_to_resolve)
    resolved: Dict[str, Any] = {}
    resolve_count = 0

    # Phase 1: Wikipedia batch
    wiki_found = 0
    wiki_miss: List[str] = []
    print(f"Phase 1: Wikipedia batch ({len(subjects_list):,} subjects)...")
    t0 = time.time()

    for batch_start in range(0, len(subjects_list), WIKI_BATCH_SIZE):
        batch = subjects_list[batch_start:batch_start + WIKI_BATCH_SIZE]
        batch_results = wikipedia_batch_lookup(batch, args.thumb)

        download_tasks = []
        for subject in batch:
            info = batch_results.get(subject)
            if info:
                download_tasks.append((subject, info))
            else:
                wiki_miss.append(subject)

        if download_tasks:
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(resolve_and_download, subj, info, store, "wikipedia"): subj
                    for subj, info in download_tasks
                }
                for fut in concurrent.futures.as_completed(futures):
                    subj = futures[fut]
                    try:
                        entry = fut.result()
                        if entry:
                            resolved[subj] = entry
                            wiki_found += 1
                            resolve_count += 1
                            if wiki_found <= 20 or wiki_found % 200 == 0:
                                print(f"  [{wiki_found}] ✓ {subj}  [{entry.get('license','?')}]")
                        else:
                            wiki_miss.append(subj)
                    except Exception:
                        wiki_miss.append(subj)

        # Periodic flush
        if resolve_count >= SAVE_EVERY:
            _flush_resolved(resolved, all_caches, run_dirs, run_subject_map, args.force)
            resolve_count = 0

        time.sleep(REQUEST_DELAY)

    _flush_resolved(resolved, all_caches, run_dirs, run_subject_map, args.force)
    resolve_count = 0
    elapsed1 = time.time() - t0
    print(f"  Wikipedia: {wiki_found:,} found, {len(wiki_miss):,} remaining ({elapsed1:.1f}s)")

    # Phase 2: Wikidata fallback
    wikidata_found = 0
    final_miss = 0

    if wiki_miss:
        print(f"\nPhase 2: Wikidata fallback ({len(wiki_miss):,} subjects)...")
        t1 = time.time()

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_wikidata_worker, subj, store, args.thumb): subj
                for subj in wiki_miss
            }
            for fut in concurrent.futures.as_completed(futures):
                subj = futures[fut]
                try:
                    _, entry = fut.result()
                    if entry:
                        resolved[subj] = entry
                        wikidata_found += 1
                        resolve_count += 1
                        if wikidata_found <= 15 or wikidata_found % 100 == 0:
                            print(f"  [{wikidata_found}] ✓ {subj}  (wikidata)")
                    else:
                        resolved[subj] = None
                        final_miss += 1
                        resolve_count += 1
                except Exception:
                    resolved[subj] = None
                    final_miss += 1
                    resolve_count += 1

                if resolve_count >= SAVE_EVERY:
                    _flush_resolved(resolved, all_caches, run_dirs, run_subject_map, args.force)
                    resolve_count = 0

        elapsed2 = time.time() - t1
        print(f"  Wikidata: {wikidata_found:,} found, {final_miss:,} no image ({elapsed2:.1f}s)")

    # Final flush
    _flush_resolved(resolved, all_caches, run_dirs, run_subject_map, args.force)

    total_found = wiki_found + wikidata_found
    total_time = time.time() - t0
    print(f"\n{'='*70}")
    print(f"RESULTS: {total_found:,} found (wp: {wiki_found:,}, wd: {wikidata_found:,}) "
          f"· {final_miss:,} no image  [{total_time:.0f}s]")
    print(f"Global store: {store.count:,} files in {store_dir}")
    _print_summary(all_caches, run_dirs)


def _print_summary(all_caches: dict, run_dirs: list):
    print("\nSUMMARY:")
    for rd in run_dirs:
        cache = all_caches.get(rd, {})
        found = sum(1 for v in cache.values()
                    if isinstance(v, dict) and (v.get("store_path") or v.get("image_url")))
        found += sum(1 for v in cache.values() if isinstance(v, str) and v)
        missing = sum(1 for v in cache.values() if v is None)
        total = len(cache)
        pct = round(found / total * 100, 1) if total else 0
        print(f"  {rd.name}: {found:,}/{total:,} ({pct}%) · {missing:,} no image")

        sources: Dict[str, int] = {}
        licenses: Dict[str, int] = {}
        for v in cache.values():
            if isinstance(v, dict):
                sources[v.get("source", "?")] = sources.get(v.get("source", "?"), 0) + 1
                lic = v.get("license") or "Unknown"
                licenses[lic] = licenses.get(lic, 0) + 1
        if sources:
            print(f"    Sources:  {', '.join(f'{k}:{v}' for k,v in sorted(sources.items(), key=lambda x:-x[1]))}")
        if licenses:
            top = sorted(licenses.items(), key=lambda x: -x[1])[:5]
            print(f"    Licenses: {', '.join(f'{k}:{v}' for k,v in top)}")


if __name__ == "__main__":
    main()
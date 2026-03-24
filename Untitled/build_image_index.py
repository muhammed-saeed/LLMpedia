#!/usr/bin/env python3
"""
build_image_index.py — Download article images from Wikipedia/Wikidata/Commons.

5-phase image resolution for maximum coverage:
  Phase 1: Wikipedia batch pageimages API (fastest, ~40% hit rate)
  Phase 2: Wikipedia REST summary API (fuzzy title matching, catches redirects)
  Phase 3: Wikidata multi-property lookup (P18, P154 logo, P41 flag, P94 coat of arms)
  Phase 4: Commons search (find images by subject name on Wikimedia Commons)
  Phase 5: Wikipedia search + pageimage (full-text search, last resort)

All images include full attribution: artist, license, license URL, Commons source URL.

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
WIKIPEDIA_REST  = "https://en.wikipedia.org/api/rest_v1"
USER_AGENT      = "LLMpedia-ImageIndexer/3.0 (https://llmpedia.net; research project)"

REQUEST_DELAY   = 0.03
SAVE_EVERY      = 200
TIMEOUT         = 15
MAX_THUMB_WIDTH = 500
DEFAULT_WORKERS = 40
WIKI_BATCH_SIZE = 50
IMAGE_STORE_DIR = "image_store"

# Wikidata image properties (in priority order)
WIKIDATA_IMAGE_PROPS = ["P18", "P154", "P41", "P94", "P14", "P15", "P242"]
# P18=image, P154=logo, P41=flag, P94=coat of arms, P14=symbol, P15=route map, P242=locator map

# Skip subjects that are clearly not imageable
_SKIP_PATTERNS = re.compile(
    r"^(list of|lists of|index of|outline of|history of .+ in \d|"
    r"\d{4} in |timeline of|glossary of|comparison of|bibliography of)",
    re.IGNORECASE
)


# ─── HTTP helpers ────────────────────────────────────────────────────────────

def _api_get(url: str, timeout: int = TIMEOUT) -> Optional[dict]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None

def _rest_get(url: str, timeout: int = TIMEOUT) -> Optional[dict]:
    """GET with JSON response, follows redirects, returns None on error."""
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": USER_AGENT,
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None

def _download_file(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = resp.read()
        if len(data) < 500:
            return False
        dest.write_bytes(data)
        return True
    except (urllib.error.URLError, OSError):
        return False

def _safe_filename(subject: str, ext: str) -> str:
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', subject)
    safe = safe.strip(". ")[:200]
    return f"{safe}.{ext}"

def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()

def _detect_ext(url: str, default: str = "jpg") -> str:
    path = urllib.parse.urlparse(url).path
    if "thumb.php" in path:
        return default
    ext = path.rsplit(".", 1)[-1].lower() if "." in path else default
    if "/" in ext:
        ext = ext.split("/")[0]
    if ext not in ("jpg", "jpeg", "png", "svg", "webp", "gif"):
        return default
    return ext


# ─── Global store index ─────────────────────────────────────────────────────

class ImageStoreIndex:
    """Fast in-memory index of the global image store directory."""

    def __init__(self, store_dir: Path):
        self.store_dir = store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._index: Dict[str, bool] = {}
        self._lock = threading.Lock()
        self._rebuild()

    def _rebuild(self):
        t0 = time.time()
        count = 0
        for f in self.store_dir.iterdir():
            if f.is_file():
                self._index[f.name] = True
                count += 1
        print(f"  Image store index: {count:,} files ({time.time()-t0:.2f}s)")

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


# ─── Commons metadata (attribution) ─────────────────────────────────────────

def get_commons_metadata(filename: str) -> Dict[str, str]:
    """Get license, artist, description from Commons for a File: page."""
    result = {"license": "", "license_url": "", "artist": "",
              "description": "", "commons_url": ""}
    params = urllib.parse.urlencode({
        "action": "query", "titles": filename, "prop": "imageinfo",
        "iiprop": "extmetadata|url", "format": "json",
    })
    data = _api_get(f"{COMMONS_API}?{params}")
    if not data:
        return result
    pages = data.get("query", {}).get("pages", {})
    for page_id, page in pages.items():
        if page_id == "-1":
            continue
        info_list = page.get("imageinfo", [])
        if not info_list:
            continue
        info = info_list[0]
        result["commons_url"] = info.get("descriptionurl", "")
        ext = info.get("extmetadata", {})
        result["license"] = ext.get("LicenseShortName", {}).get("value", "")
        result["license_url"] = ext.get("LicenseUrl", {}).get("value", "")
        raw = ext.get("Artist", {}).get("value", "")
        if raw:
            result["artist"] = _strip_html(raw)[:200]
        desc = ext.get("ImageDescription", {}).get("value", "")
        if desc:
            result["description"] = _strip_html(desc)[:300]
    return result


# ─── Download + metadata ────────────────────────────────────────────────────

def resolve_and_download(
    subject: str, img_info: Dict[str, str],
    store: ImageStoreIndex, source_label: str,
) -> Optional[Dict[str, Any]]:
    """Download image to global store and fetch Commons metadata."""
    image_url = img_info.get("image_url", "")
    filename_commons = img_info.get("filename", "")
    if not image_url:
        return None

    ext = _detect_ext(image_url)
    local_filename = _safe_filename(subject, ext)

    # Check global store — skip download if already there
    if not store.exists(local_filename):
        dest = store.filepath(local_filename)
        ok = _download_file(image_url, dest)
        if not ok:
            return None
        store.mark_added(local_filename)

    # Always fetch attribution metadata
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


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Wikipedia batch pageimages API
# ═══════════════════════════════════════════════════════════════════════════

def wikipedia_batch_lookup(subjects: List[str], thumb_width: int = MAX_THUMB_WIDTH
                           ) -> Dict[str, Optional[Dict[str, str]]]:
    """Batch lookup up to 50 subjects via Wikipedia pageimages API."""
    results: Dict[str, Optional[Dict[str, str]]] = {}
    titles_str = "|".join(s.replace(" ", "_") for s in subjects)
    params = urllib.parse.urlencode({
        "action": "query", "titles": titles_str, "prop": "pageimages",
        "format": "json", "pithumbsize": thumb_width, "pilicense": "any",
        "redirects": "1",
    })
    data = _api_get(f"{WIKIPEDIA_API}?{params}")
    if not data:
        for s in subjects:
            results[s] = None
        return results

    # Build normalization maps
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
        if title in normalized:
            title_sp = normalized[title]
        elif s in normalized:
            title_sp = normalized[s]
        if title_sp in redirects_map:
            title_sp = redirects_map[title_sp]
        title_to_subject[title_sp] = s
        title_to_subject[s] = s

    pages = data.get("query", {}).get("pages", {})
    found = set()
    for page_id, page in pages.items():
        if page_id == "-1":
            continue
        page_title = page.get("title", "")
        image_file = page.get("pageimage")
        thumb = page.get("thumbnail", {}).get("source")
        if not image_file or not thumb:
            continue
        original = (title_to_subject.get(page_title)
                    or title_to_subject.get(page_title.replace("_", " ")))
        if not original:
            continue
        encoded = urllib.parse.quote(page_title.replace(" ", "_"))
        results[original] = {
            "image_url": thumb,
            "filename": f"File:{image_file}",
            "page_url": f"https://en.wikipedia.org/wiki/{encoded}",
        }
        found.add(original)

    for s in subjects:
        if s not in found:
            results.setdefault(s, None)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Wikipedia REST summary API (fuzzy matching, catches redirects)
# ═══════════════════════════════════════════════════════════════════════════

def wikipedia_rest_lookup(subject: str, thumb_width: int = MAX_THUMB_WIDTH
                          ) -> Optional[Dict[str, str]]:
    """Use Wikipedia REST API for better redirect/fuzzy matching."""
    encoded = urllib.parse.quote(subject.replace(" ", "_"))
    data = _rest_get(f"{WIKIPEDIA_REST}/page/summary/{encoded}")
    if not data:
        return None
    thumb = data.get("thumbnail", {})
    thumb_url = thumb.get("source", "")
    if not thumb_url:
        # Try originalimage
        orig = data.get("originalimage", {})
        thumb_url = orig.get("source", "")
    if not thumb_url:
        return None
    # Extract filename from URL
    # Typical: https://upload.wikimedia.org/wikipedia/commons/thumb/a/ab/File.jpg/320px-File.jpg
    page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
    # Try to extract File: name
    filename = ""
    parts = thumb_url.split("/")
    for i, p in enumerate(parts):
        if p in ("commons", "en") and i + 3 < len(parts):
            # .../commons/thumb/a/ab/Filename.ext/...
            candidate = parts[i + 3] if parts[i + 1] == "thumb" else ""
            if candidate and "." in candidate:
                filename = f"File:{urllib.parse.unquote(candidate)}"
                break
    return {
        "image_url": thumb_url,
        "filename": filename,
        "page_url": page_url or f"https://en.wikipedia.org/wiki/{encoded}",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Wikidata multi-property lookup
# ═══════════════════════════════════════════════════════════════════════════

def wikidata_image_lookup(subject: str, thumb_width: int = MAX_THUMB_WIDTH
                          ) -> Optional[Dict[str, str]]:
    """Search Wikidata for image across multiple properties."""
    params = urllib.parse.urlencode({
        "action": "wbsearchentities", "search": subject,
        "language": "en", "limit": "5", "format": "json",
    })
    data = _api_get(f"{WIKIDATA_API}?{params}")
    if not data:
        return None

    for result in data.get("search", []):
        entity_id = result.get("id")
        if not entity_id:
            continue

        # Fetch multiple image properties at once
        props = "|".join(WIKIDATA_IMAGE_PROPS)
        params2 = urllib.parse.urlencode({
            "action": "wbgetclaims", "entity": entity_id,
            "property": props, "format": "json",
        })
        data2 = _api_get(f"{WIKIDATA_API}?{params2}")
        if not data2:
            continue

        # Try each property in priority order
        claims = data2.get("claims", {})
        for prop in WIKIDATA_IMAGE_PROPS:
            prop_claims = claims.get(prop, [])
            if not prop_claims:
                continue
            image_name = (prop_claims[0].get("mainsnak", {})
                          .get("datavalue", {}).get("value"))
            if not image_name:
                continue
            enc = urllib.parse.quote(image_name.replace(" ", "_"))
            prop_label = {"P18": "image", "P154": "logo", "P41": "flag",
                          "P94": "coat_of_arms", "P14": "symbol"}.get(prop, prop)
            return {
                "image_url": f"https://commons.wikimedia.org/w/thumb.php?f={enc}&w={thumb_width}",
                "filename": f"File:{image_name}",
                "page_url": f"https://www.wikidata.org/wiki/{entity_id}",
                "wikidata_prop": prop_label,
            }
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: Wikimedia Commons search
# ═══════════════════════════════════════════════════════════════════════════

def commons_search_lookup(subject: str, thumb_width: int = MAX_THUMB_WIDTH
                          ) -> Optional[Dict[str, str]]:
    """Search Wikimedia Commons for an image matching the subject."""
    params = urllib.parse.urlencode({
        "action": "query", "list": "search", "srnamespace": "6",  # File namespace
        "srsearch": subject, "srlimit": "5", "format": "json",
    })
    data = _api_get(f"{COMMONS_API}?{params}")
    if not data:
        return None

    results = data.get("query", {}).get("search", [])
    for r in results:
        title = r.get("title", "")
        if not title.startswith("File:"):
            continue
        # Skip SVG logos, icons, maps if subject is likely a person/place
        lower = title.lower()
        if any(skip in lower for skip in ("icon", "pictogram", "symbol", "button")):
            continue

        # Get thumbnail URL
        params2 = urllib.parse.urlencode({
            "action": "query", "titles": title, "prop": "imageinfo",
            "iiprop": "url|size", "iiurlwidth": thumb_width, "format": "json",
        })
        data2 = _api_get(f"{COMMONS_API}?{params2}")
        if not data2:
            continue
        pages = data2.get("query", {}).get("pages", {})
        for pid, page in pages.items():
            if pid == "-1":
                continue
            info = (page.get("imageinfo") or [{}])[0]
            thumb_url = info.get("thumburl") or info.get("url", "")
            if not thumb_url:
                continue
            # Check minimum size
            width = info.get("width", 0)
            if width < 100:
                continue
            return {
                "image_url": thumb_url,
                "filename": title,
                "page_url": info.get("descriptionurl", ""),
            }
    return None


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5: Wikipedia search + pageimage
# ═══════════════════════════════════════════════════════════════════════════

def wikipedia_search_lookup(subject: str, thumb_width: int = MAX_THUMB_WIDTH
                            ) -> Optional[Dict[str, str]]:
    """Full-text search Wikipedia, then get pageimage of top result."""
    params = urllib.parse.urlencode({
        "action": "query", "list": "search", "srsearch": subject,
        "srlimit": "3", "format": "json",
    })
    data = _api_get(f"{WIKIPEDIA_API}?{params}")
    if not data:
        return None

    results = data.get("query", {}).get("search", [])
    for r in results:
        title = r.get("title", "")
        if not title:
            continue
        # Now get pageimage for this title
        params2 = urllib.parse.urlencode({
            "action": "query", "titles": title, "prop": "pageimages",
            "format": "json", "pithumbsize": thumb_width, "pilicense": "any",
        })
        data2 = _api_get(f"{WIKIPEDIA_API}?{params2}")
        if not data2:
            continue
        pages = data2.get("query", {}).get("pages", {})
        for pid, page in pages.items():
            if pid == "-1":
                continue
            image_file = page.get("pageimage")
            thumb = page.get("thumbnail", {}).get("source")
            if not image_file or not thumb:
                continue
            encoded = urllib.parse.quote(title.replace(" ", "_"))
            return {
                "image_url": thumb,
                "filename": f"File:{image_file}",
                "page_url": f"https://en.wikipedia.org/wiki/{encoded}",
            }
    return None


# ─── Subject normalization ───────────────────────────────────────────────

def _normalize_subject(subject: str) -> List[str]:
    """Generate alternate forms for a subject to try."""
    forms = [subject]
    # Strip parenthetical disambiguators: "Springfield (Illinois)" → "Springfield"
    stripped = re.sub(r"\s*\([^)]+\)\s*$", "", subject).strip()
    if stripped and stripped != subject:
        forms.append(stripped)
    # Strip trailing ", Location": "Dresden, Germany" → "Dresden"
    if ", " in subject:
        forms.append(subject.split(",")[0].strip())
    return forms


# ─── Worker functions ────────────────────────────────────────────────────

def _phase2_worker(subject: str, store: ImageStoreIndex, thumb: int
                    ) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Phase 2: Wikipedia REST API with alternate forms."""
    for form in _normalize_subject(subject):
        info = wikipedia_rest_lookup(form, thumb)
        if info:
            entry = resolve_and_download(subject, info, store, "wikipedia_rest")
            if entry:
                return subject, entry
    return subject, None


def _phase3_worker(subject: str, store: ImageStoreIndex, thumb: int
                    ) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Phase 3: Wikidata multi-property."""
    for form in _normalize_subject(subject):
        info = wikidata_image_lookup(form, thumb)
        if info:
            source = f"wikidata_{info.get('wikidata_prop', 'P18')}"
            entry = resolve_and_download(subject, info, store, source)
            if entry:
                return subject, entry
    return subject, None


def _phase4_worker(subject: str, store: ImageStoreIndex, thumb: int
                    ) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Phase 4: Commons search."""
    info = commons_search_lookup(subject, thumb)
    if info:
        entry = resolve_and_download(subject, info, store, "commons_search")
        if entry:
            return subject, entry
    return subject, None


def _phase5_worker(subject: str, store: ImageStoreIndex, thumb: int
                    ) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Phase 5: Wikipedia full-text search."""
    info = wikipedia_search_lookup(subject, thumb)
    if info:
        entry = resolve_and_download(subject, info, store, "wikipedia_search")
        if entry:
            return subject, entry
    return subject, None


# ─── Discovery helpers ───────────────────────────────────────────────────

def find_run_dirs(root: Path) -> List[Path]:
    if not root.is_dir():
        return []
    for name in ("articles.jsonl", "articles_wikitext.jsonl"):
        if (root / name).exists():
            return [root]
    found = set()
    for name in ("articles.jsonl", "articles_wikitext.jsonl"):
        for p in root.rglob(name):
            found.add(p.parent)
    return sorted(found)


def load_subjects_for_run(run_dir: Path) -> set:
    subjects = set()
    for name in ("articles.jsonl", "articles_wikitext.jsonl"):
        path = run_dir / name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    s = json.loads(line).get("subject", "").strip()
                    if s:
                        subjects.add(s)
                except Exception:
                    pass
        break
    return subjects


def load_cache(run_dir: Path) -> dict:
    p = run_dir / "gptkb_image_cache.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_cache(run_dir: Path, cache: dict):
    p = run_dir / "gptkb_image_cache.json"
    p.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


# ─── Batch cache flush ──────────────────────────────────────────────────

def _flush_resolved(resolved: Dict[str, Any], all_caches: Dict[Path, dict],
                    run_dirs: List[Path], run_subject_map: Dict[Path, set],
                    force: bool):
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


def _run_phase(phase_name, subjects_list, worker_fn, store, thumb, workers,
               resolved, resolve_count_ref, all_caches, run_dirs,
               run_subject_map, force, log_first=20, log_every=200):
    """Run a parallel phase and collect results."""
    if not subjects_list:
        return [], 0

    print(f"\n  {phase_name} ({len(subjects_list):,} subjects)...")
    t0 = time.time()
    found_count = 0
    still_missing = []
    batch_count = [0]  # mutable for closure

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(worker_fn, subj, store, thumb): subj
            for subj in subjects_list
        }
        for fut in concurrent.futures.as_completed(futures):
            subj = futures[fut]
            try:
                _, entry = fut.result()
                if entry:
                    resolved[subj] = entry
                    found_count += 1
                    batch_count[0] += 1
                    if found_count <= log_first or found_count % log_every == 0:
                        src = entry.get("source", "?")
                        lic = entry.get("license", "?")
                        print(f"    [{found_count}] ✓ {subj}  [{src}, {lic}]")
                else:
                    still_missing.append(subj)
                    batch_count[0] += 1
            except Exception:
                still_missing.append(subj)
                batch_count[0] += 1

            # Periodic flush
            if batch_count[0] >= SAVE_EVERY:
                _flush_resolved(resolved, all_caches, run_dirs,
                                run_subject_map, force)
                batch_count[0] = 0

    _flush_resolved(resolved, all_caches, run_dirs, run_subject_map, force)
    elapsed = time.time() - t0
    print(f"    → {found_count:,} found, {len(still_missing):,} remaining ({elapsed:.1f}s)")
    return still_missing, found_count


# ─── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download Wikipedia/Wikidata/Commons images for LLMpedia")
    parser.add_argument("root", help="Run dir or parent dir")
    parser.add_argument("--thumb", type=int, default=MAX_THUMB_WIDTH)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--force", action="store_true", help="Re-download all")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--phases", type=str, default="1,2,3,4,5",
                        help="Comma-separated phases to run (default: 1,2,3,4,5)")
    parser.add_argument("--global-store", type=str, default=None,
                        help="Path to global image_store (e.g. deploy/image_store)")
    args = parser.parse_args()

    phases = set(int(x) for x in args.phases.split(","))

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

    # Filter out un-imageable subjects
    skipped = set()
    for s in all_subjects:
        if _SKIP_PATTERNS.match(s):
            skipped.add(s)
    if skipped:
        all_subjects -= skipped
        print(f"  Skipped (un-imageable): {len(skipped):,}")

    all_caches: Dict[Path, dict] = {}
    already_resolved: set = set()
    for rd in run_dirs:
        cache = load_cache(rd)
        all_caches[rd] = cache
        already_resolved |= set(cache.keys())

    subjects_to_resolve = all_subjects - already_resolved if not args.force else all_subjects
    # Also re-try previously-failed subjects (None entries in cache)
    if not args.force:
        failed_subjects = set()
        for rd in run_dirs:
            cache = all_caches[rd]
            for s, v in cache.items():
                if v is None and s in all_subjects:
                    failed_subjects.add(s)
        if failed_subjects:
            subjects_to_resolve |= failed_subjects
            print(f"  Retrying previously failed: {len(failed_subjects):,}")

    print(f"  Already resolved: {len(already_resolved):,}")
    print(f"  Need resolution:  {len(subjects_to_resolve):,}")

    if not subjects_to_resolve:
        print("\nAll subjects already resolved. Use --force to re-download.")
        _print_summary(all_caches, run_dirs)
        return

    if args.dry_run:
        print(f"\n[DRY RUN] Would query for {len(subjects_to_resolve):,} subjects")
        _print_summary(all_caches, run_dirs)
        return

    # Set up image store
    if args.global_store:
        store_dir = Path(args.global_store).resolve()
    else:
        store_dir = run_dirs[0] / IMAGE_STORE_DIR
    store = ImageStoreIndex(store_dir)

    print(f"\nImage store: {store_dir}  ({store.count:,} existing)")
    print(f"  Thumb: {args.thumb}px  Workers: {args.workers}")
    print(f"  Phases: {sorted(phases)}  Save interval: {SAVE_EVERY}")

    subjects_list = sorted(subjects_to_resolve)
    resolved: Dict[str, Any] = {}
    total_found = 0
    t0_all = time.time()
    missing = subjects_list

    # ── Phase 1: Wikipedia batch pageimages ──
    if 1 in phases and missing:
        print(f"\n{'─'*60}")
        print(f"  PHASE 1: Wikipedia batch pageimages")
        print(f"{'─'*60}")
        t0 = time.time()
        wiki_found = 0
        wiki_miss = []

        for batch_start in range(0, len(missing), WIKI_BATCH_SIZE):
            batch = missing[batch_start:batch_start + WIKI_BATCH_SIZE]
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
                                if wiki_found <= 20 or wiki_found % 500 == 0:
                                    print(f"    [{wiki_found}] ✓ {subj}  [{entry.get('license','?')}]")
                            else:
                                wiki_miss.append(subj)
                        except Exception:
                            wiki_miss.append(subj)

            # Periodic flush
            if (batch_start // WIKI_BATCH_SIZE) % 4 == 0 and batch_start > 0:
                _flush_resolved(resolved, all_caches, run_dirs, run_subject_map, args.force)

            time.sleep(REQUEST_DELAY)

        _flush_resolved(resolved, all_caches, run_dirs, run_subject_map, args.force)
        elapsed = time.time() - t0
        print(f"    → Wikipedia batch: {wiki_found:,} found, {len(wiki_miss):,} remaining ({elapsed:.1f}s)")
        total_found += wiki_found
        missing = wiki_miss

    # ── Phase 2: Wikipedia REST summary API ──
    if 2 in phases and missing:
        print(f"\n{'─'*60}")
        print(f"  PHASE 2: Wikipedia REST summary (fuzzy matching)")
        print(f"{'─'*60}")
        missing, found = _run_phase(
            "REST API", missing, _phase2_worker, store, args.thumb, args.workers,
            resolved, [0], all_caches, run_dirs, run_subject_map, args.force)
        total_found += found

    # ── Phase 3: Wikidata multi-property ──
    if 3 in phases and missing:
        print(f"\n{'─'*60}")
        print(f"  PHASE 3: Wikidata (P18, P154 logo, P41 flag, P94 coat of arms)")
        print(f"{'─'*60}")
        missing, found = _run_phase(
            "Wikidata", missing, _phase3_worker, store, args.thumb, args.workers,
            resolved, [0], all_caches, run_dirs, run_subject_map, args.force)
        total_found += found

    # ── Phase 4: Commons search ──
    if 4 in phases and missing:
        print(f"\n{'─'*60}")
        print(f"  PHASE 4: Wikimedia Commons search")
        print(f"{'─'*60}")
        missing, found = _run_phase(
            "Commons", missing, _phase4_worker, store, args.thumb,
            min(args.workers, 20),  # fewer workers to be polite
            resolved, [0], all_caches, run_dirs, run_subject_map, args.force,
            log_first=10, log_every=500)
        total_found += found

    # ── Phase 5: Wikipedia search + pageimage ──
    if 5 in phases and missing:
        print(f"\n{'─'*60}")
        print(f"  PHASE 5: Wikipedia full-text search")
        print(f"{'─'*60}")
        missing, found = _run_phase(
            "WP search", missing, _phase5_worker, store, args.thumb,
            min(args.workers, 20),
            resolved, [0], all_caches, run_dirs, run_subject_map, args.force,
            log_first=10, log_every=500)
        total_found += found

    # Mark remaining as None (no image found)
    for s in missing:
        if s not in resolved:
            resolved[s] = None

    _flush_resolved(resolved, all_caches, run_dirs, run_subject_map, args.force)

    total_time = time.time() - t0_all
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Found:   {total_found:,}")
    print(f"  Missing: {len(missing):,}")
    print(f"  Time:    {total_time:.0f}s")
    print(f"  Store:   {store.count:,} files in {store_dir}")
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
                src = v.get("source", "?")
                sources[src] = sources.get(src, 0) + 1
                lic = v.get("license") or "Unknown"
                licenses[lic] = licenses.get(lic, 0) + 1
        if sources:
            top = sorted(sources.items(), key=lambda x: -x[1])[:8]
            print(f"    Sources:  {', '.join(f'{k}:{v}' for k,v in top)}")
        if licenses:
            top = sorted(licenses.items(), key=lambda x: -x[1])[:5]
            print(f"    Licenses: {', '.join(f'{k}:{v}' for k,v in top)}")


if __name__ == "__main__":
    main()
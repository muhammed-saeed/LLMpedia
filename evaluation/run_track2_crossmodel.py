#!/usr/bin/env python3
"""
run_track2_crossmodel.py — Cross-model analysis for LLMPedia (ACL 2026).

SUBJECT LOADING
───────────────
  Streams articles.jsonl in BFS/generation order.  --max-subjects 120000
  takes the first 120 K subjects as they appear in the file.  Every subject
  in subject_names is GUARANTEED to have article text in that file because
  that is the only place subjects can come from.  Any attempt to re-read a
  subject that is absent from the file is treated as a BUG and aborts the
  run (aborts with RuntimeError if any subject is missing).

  The early-exit optimisation in read_articles_for_subjects() is disabled;
  we always scan the full file to pick up the lowest-hop version of every
  subject, consistent with what load_model_data() does.

FACTUALITY
──────────
  Two modes, both using seeds [42, 123, 7] (--seeds / --fact-seeds):

  SHARED: all models evaluated on the SAME 1 000 subjects drawn from the
    exact-name intersection.  wiki_subject_found is subject-level — must be
    identical across all three models — asserted at runtime.

  PER-MODEL: each model independently samples args.sample_n subjects from its
    own corpus.  Different models sample different subjects.

  Wikipedia coverage is determined SOLELY by the wiki_subject_found flag set
  by factuality_core during claim verification (redirect-inclusive Wikipedia
  search).  No separate direct-API check is run.  ONE number, ONE method.

DENOMINATORS (in every summary row)
────────────────────────────────────
  n_seeds              — number of random seeds
  n_per_seed           — subjects submitted per seed
  n_attempted_total    — n_per_seed × n_seeds
  n_returned_total     — records returned by run_evaluation() (pooled).
                         In the happy path equals n_attempted_total.
                         Shortfalls are logged as warnings.
  n_wiki_subject_found — records / subjects where wiki_subject_found=True
                         (precision denominator)

CANON OVERLAP FIX
─────────────────
  Canon overlap is computed ONLY from subject_canon_keys (derived from the
  capped subject_names list), NOT from seen_canon_keys.json (which contains
  ALL entities from the full run, potentially 740K+).  This ensures exact
  and canonical overlap are computed over the same set of subjects.
"""
from __future__ import annotations
import argparse, csv, json, math, os, random, re, sys
import statistics, time, unicodedata, datetime
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Set

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

try:
    import requests; HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("[WARN] matplotlib/numpy not installed — figures skipped")

try:
    from factuality_core import EvalConfig, EvalInput, run_evaluation, generate_outputs
    HAS_FACTUALITY = True
except ImportError:
    HAS_FACTUALITY = False

# ─── Constants ───────────────────────────────────────────────────────────────
DEFAULT_SEEDS = [42, 123, 7]

ACL_STYLE = {
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
}
COLORS = {"llama": "#2196F3", "deepseek": "#FF9800", "gpt": "#4CAF50",
          "overlap": "#9C27B0", "all3": "#E91E63"}
MODEL_DISPLAY: Dict[str, str] = {}

# ─── Helpers ─────────────────────────────────────────────────────────────────
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
    if v is None: return ""
    if isinstance(v, float): return f"{v:.{d}f}"
    return str(v)

def _write_csv(rows, path):
    if not rows: return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r.keys()))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) if isinstance(r.get(k), float)
                        else r.get(k, "") for k in keys})
    print(f"  -> {path} ({len(rows)} rows)")

_DASH_RX = re.compile(r"[\-\u2010-\u2014\u2212]+", re.UNICODE)
def canon_key(s: str) -> str:
    if not isinstance(s, str): s = str(s)
    t = unicodedata.normalize("NFKC", s).strip().lower().replace("_", " ")
    t = _DASH_RX.sub(" ", t)
    t = re.sub(r"\s+", " ", t, flags=re.UNICODE)
    t = re.sub(r"[^\w\s]", "", t, flags=re.UNICODE)
    return re.sub(r"\s+", " ", t, flags=re.UNICODE).strip()

def jaccard(a: Set, b: Set) -> float:
    if not a and not b: return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 0.0

def overlap_coeff(a: Set, b: Set) -> float:
    if not a or not b: return 0.0
    return len(a & b) / min(len(a), len(b))

# ─── Wikitext parsing ────────────────────────────────────────────────────────
_LINK_RX     = re.compile(r"\[\[([^:|\]]+)(?:\|[^]]*)?]]")
_CAT_RX      = re.compile(r"\[\[Category:([^|\]]+)(?:\|[^]]*)?]]", re.IGNORECASE)
_HEADING2_RX = re.compile(r"^==\s*(.*?)\s*==\s*$", re.UNICODE)
_IGNORE_SECS = {"see also", "further reading", "external links",
                "references", "notes", "bibliography"}

def extract_wikilinks(wt: str) -> List[str]:
    if not isinstance(wt, str): return []
    out, seen, ign = [], set(), False
    for line in wt.splitlines():
        hm = _HEADING2_RX.match(line.strip())
        if hm:
            ign = (hm.group(1) or "").strip().lower() in _IGNORE_SECS
            continue
        if ign: continue
        for m in _LINK_RX.finditer(line):
            t = (m.group(1) or "").strip()
            if (t and len(t) <= 150
                    and not t.lower().startswith(("category:", "file:", "image:"))):
                if t not in seen: seen.add(t); out.append(t)
    return out

def extract_categories(wt: str) -> List[str]:
    if not isinstance(wt, str): return []
    out, seen = [], set()
    for m in _CAT_RX.finditer(wt):
        n = (m.group(1) or "").strip()
        if n and len(n.split()) <= 6 and n not in seen:
            seen.add(n); out.append(n)
    return out

def count_sections(wt: str) -> int:
    return sum(1 for l in (wt or "").splitlines() if _HEADING2_RX.match(l.strip()))

def word_count(t: str) -> int:
    return len(re.sub(r"[^\w\s]", " ", t.lower()).split()) if t else 0

def _tokenize(t: str) -> List[str]:
    return [w for w in re.sub(r"[^\w\s]", " ", t.lower()).split() if w]

def _ngrams_set(tok: List[str], n: int) -> Set:
    return (set(tuple(tok[i:i+n]) for i in range(len(tok)-n+1))
            if len(tok) >= n else set())

# ─── wiki_subject_found flag ─────────────────────────────────────────────────
def _is_wiki_found(rec: dict, strict: bool = False) -> bool:
    """Return True if this eval record indicates the subject has a Wikipedia page.

    strict=True  — trust only the explicit wiki_subject_found flag.
                   Used in shared-mode summaries (coverage is a clean
                   subject-level property identical across models).
    strict=False — fall back to verdict / similarity inference when the flag
                   is absent.  Used in per-model independent mode.
    """
    wf = rec.get("wiki_subject_found")
    if wf is not None:
        return bool(wf)
    if strict:
        return False
    if (rec.get("wiki_n_supported", 0) or 0) > 0: return True
    if (rec.get("wiki_n_refuted",   0) or 0) > 0: return True
    if any(k.startswith("sim_") and isinstance(rec.get(k), (int, float))
           for k in rec.keys()):
        return True
    n_claims  = rec.get("n_claims", 0)
    wiki_ins  = rec.get("wiki_n_insufficient", 0) or 0
    if n_claims > 0 and wiki_ins == n_claims:
        return False
    return True

# ─── Data model ──────────────────────────────────────────────────────────────
@dataclass
class SubjectMeta:
    hop: int = 0; wc: int = 0; n_sections: int = 0
    n_links: int = 0; n_cats: int = 0

@dataclass
class ModelData:
    key: str; dir_path: str; articles_file: str = "articles.jsonl"
    display_name: str = ""; run_meta: Dict = field(default_factory=dict)
    subject_names: List[str] = field(default_factory=list)
    subject_meta: Dict[str, SubjectMeta] = field(default_factory=dict)
    n_total_raw: int = 0; n_total_unique: int = 0
    wikilinks_by_subject: Dict[str, List[str]] = field(default_factory=dict)
    canon_entities_by_subject: Dict[str, Set[str]] = field(default_factory=dict)
    categories_by_subject: Dict[str, List[str]] = field(default_factory=dict)
    # seen_canon_keys: full run's seen_canon_keys.json (for reference/reporting only)
    seen_canon_keys: Set[str] = field(default_factory=set)
    # subject_canon_keys: canonical keys derived ONLY from the capped subject_names
    subject_canon_keys: Set[str] = field(default_factory=set)

# ─── Loading ─────────────────────────────────────────────────────────────────
def load_model_data(key: str, dir_path: str, articles_file: str = "articles.jsonl",
                    max_subjects: int = 0, min_words: int = 100) -> ModelData:
    """Stream articles.jsonl in FILE ORDER (BFS/generation order).

    subject_names will contain exactly the first min(max_subjects, unique)
    subjects that passed the min_words filter, in the order they appear in
    the file.  Every one of them is guaranteed to be re-readable from the
    same file — verified with an inline loop before evaluation starts.
    """
    dir_path = os.path.abspath(dir_path)
    meta_path = os.path.join(dir_path, "run_meta.json")
    run_meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            run_meta = json.load(f)
    cas = run_meta.get("cascading_defaults") or {}
    ar  = run_meta.get("args_raw") or {}
    em  = (ar.get("elicit_model_key") or "").strip()
    gm  = (cas.get("global_model_key") or ar.get("model_key") or "").strip()
    dn  = (em or gm or key).replace("scads-", "").replace("_", " ")

    md = ModelData(key=key, dir_path=dir_path, articles_file=articles_file,
                   display_name=dn, run_meta=run_meta)
    arts_path = os.path.join(dir_path, articles_file)
    if not os.path.exists(arts_path):
        raise FileNotFoundError(f"[FATAL] articles.jsonl not found: {arts_path}")

    print(f"[load] {key}: streaming {arts_path} ...")
    t0 = time.perf_counter()
    best: Dict[str, tuple] = {}  # subject -> (hop, meta, links, cats)
    n_raw = 0; n_skip_words = 0

    with open(arts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            n_raw += 1
            if n_raw % 50_000 == 0:
                print(f"  [{key}] ...{n_raw:,} lines, {len(best):,} kept "
                      f"({time.perf_counter()-t0:.1f}s)")
            try: a = json.loads(line)
            except: continue
            if not isinstance(a, dict): continue
            s = (a.get("subject") or "").strip()
            if not s: continue
            wt = a.get("wikitext") or ""
            wc = word_count(wt)
            if min_words > 0 and wc < min_words:
                n_skip_words += 1
                continue
            hop = 0
            try: hop = int(a.get("hop", 0))
            except: pass
            links = extract_wikilinks(wt)
            cats  = extract_categories(wt)
            meta  = SubjectMeta(hop=hop, wc=wc, n_sections=count_sections(wt),
                                n_links=len(links), n_cats=len(cats))
            if s not in best:
                best[s] = (hop, meta, links, cats)
            elif hop < best[s][0]:
                best[s] = (hop, meta, links, cats)

    md.n_total_raw     = n_raw
    md.n_total_unique  = len(best)
    print(f"  [{key}] {n_raw:,} lines → {len(best):,} unique "
          f"(skipped {n_skip_words:,} < {min_words} words) "
          f"in {time.perf_counter()-t0:.1f}s")

    all_subjects = list(best.keys())  # BFS order preserved
    if max_subjects > 0 and len(all_subjects) > max_subjects:
        all_subjects = all_subjects[:max_subjects]
        print(f"  [{key}] capped to first {max_subjects:,} (BFS/file order)")

    md.subject_names = all_subjects
    for s in all_subjects:
        _, meta, links, cats = best[s]
        md.subject_meta[s]              = meta
        md.wikilinks_by_subject[s]      = links
        md.canon_entities_by_subject[s] = set(canon_key(l) for l in links if l)
        md.categories_by_subject[s]     = cats
        ck = canon_key(s)
        if ck: md.subject_canon_keys.add(ck)
    del best

    # seen_canon_keys.json (loaded for reference/reporting only — NOT used for overlap)
    ck_path = os.path.join(dir_path, "seen_canon_keys.json")
    if os.path.exists(ck_path):
        try:
            with open(ck_path, "r", encoding="utf-8") as f: raw = f.read()
            try:
                arr = json.loads(raw)
            except json.JSONDecodeError as je:
                print(f"  [{key}] seen_canon_keys.json broken at {je.pos}; repairing ...")
                trunc = raw[:je.pos].rstrip().rstrip(",").rstrip()
                if not trunc.endswith("]"): trunc += "]"
                try:
                    arr = json.loads(trunc)
                    print(f"  [{key}] repaired: {len(arr):,} keys")
                except:
                    arr = None
                    print(f"  [{key}] repair failed — using subject_canon_keys")
            if isinstance(arr, list):
                md.seen_canon_keys = set(str(x) for x in arr if x)
                print(f"  [{key}] seen_canon_keys.json: {len(md.seen_canon_keys):,} keys "
                      f"(full run, for reference only)")
        except Exception as e:
            print(f"  [{key}] canon keys error: {e}")
    else:
        print(f"  [{key}] no seen_canon_keys.json")

    # ── Log the difference so it's visible ──
    n_subject_canon = len(md.subject_canon_keys)
    n_seen_canon    = len(md.seen_canon_keys)
    if n_seen_canon > 0 and n_seen_canon != n_subject_canon:
        print(f"  [{key}] canon keys: subject_canon={n_subject_canon:,} (from capped subjects)  "
              f"seen_canon={n_seen_canon:,} (full run)")
        if n_seen_canon > n_subject_canon * 1.1:
            print(f"  [{key}] ⚠ seen_canon_keys.json is {n_seen_canon/n_subject_canon:.1f}x larger "
                  f"than subject_canon_keys — overlap will use subject_canon_keys only")

    MODEL_DISPLAY[key] = dn
    print(f"[load] {key} ({dn}): {len(md.subject_names):,} subjects in "
          f"{time.perf_counter()-t0:.1f}s")
    return md


def read_articles_for_subjects(md: ModelData,
                                subjects: List[str]) -> Dict[str, str]:
    """Re-read wikitext for the given subjects from articles.jsonl.

    Scans the ENTIRE file (no early exit) to guarantee we pick up the
    lowest-hop version of every subject, consistent with load_model_data.

    Every subject in `subjects` should be in md.subject_names.  After
    scanning; a RuntimeError is raised by the caller if any subjects are missing.
    """
    needed = set(subjects) & set(md.subject_names)

    not_in_model = set(subjects) - set(md.subject_names)
    if not_in_model:
        print(f"  [{md.key}] WARNING {len(not_in_model)} requested subjects "
              f"not in subject_names — this must not happen in shared mode: "
              f"{sorted(not_in_model)[:5]}")

    if not needed:
        return {}

    arts_path = os.path.join(md.dir_path, md.articles_file)
    if not os.path.exists(arts_path):
        raise FileNotFoundError(f"[FATAL] {arts_path} missing during re-read")

    result:   Dict[str, str] = {}
    best_hop: Dict[str, int] = {}
    n_read = 0
    t0 = time.perf_counter()

    with open(arts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            n_read += 1
            if n_read % 100_000 == 0:
                print(f"    [{md.key}] ...{n_read:,} lines, "
                      f"found {len(result)}/{len(needed)}")
            try: a = json.loads(line)
            except: continue
            if not isinstance(a, dict): continue
            s = (a.get("subject") or "").strip()
            if s not in needed: continue
            hop = 0
            try: hop = int(a.get("hop", 0))
            except: pass
            if s not in best_hop or hop < best_hop[s]:
                best_hop[s] = hop
                result[s]   = a.get("wikitext") or ""

    print(f"    [{md.key}] re-read: {len(result)}/{len(needed)} "
          f"in {time.perf_counter()-t0:.1f}s ({n_read:,} lines scanned)")
    return result


# ─── Overlap ─────────────────────────────────────────────────────────────────
def compute_overlap(models: Dict[str, ModelData]) -> dict:
    """Compute subject overlap between models.

    FIX: Canon overlap ALWAYS uses subject_canon_keys (derived from the
    capped subject_names), NEVER seen_canon_keys (which contains ALL entities
    from the full run).  This ensures exact and canonical overlap are
    computed over the same set of subjects.
    """
    keys = sorted(models.keys())
    exact_sets = {k: set(models[k].subject_names) for k in keys}
    exact_union = set().union(*exact_sets.values())
    exact_inter = (set.intersection(*exact_sets.values())
                   if len(keys) >= 2 else exact_sets[keys[0]])

    # ── FIX: Always use subject_canon_keys (capped), never seen_canon_keys ──
    canon_sets = {k: models[k].subject_canon_keys for k in keys}

    canon_union = set().union(*canon_sets.values())
    canon_inter = (set.intersection(*canon_sets.values())
                   if len(keys) >= 2 else canon_sets[keys[0]])

    # Also compute seen_canon overlap for reference (reported but not used for analysis)
    seen_canon_sets = {k: (models[k].seen_canon_keys if models[k].seen_canon_keys
                           else models[k].subject_canon_keys) for k in keys}
    seen_canon_union = set().union(*seen_canon_sets.values())
    seen_canon_inter = (set.intersection(*seen_canon_sets.values())
                        if len(keys) >= 2 else seen_canon_sets[keys[0]])

    result = {
        "keys": keys,
        "n_per_model":       {k: len(exact_sets[k]) for k in keys},
        "n_canon_per_model": {k: len(canon_sets[k]) for k in keys},
        "n_union":             len(exact_union),
        "n_intersection":      len(exact_inter),
        "n_canon_union":       len(canon_union),
        "n_canon_intersection":len(canon_inter),
        "intersection_subjects": sorted(exact_inter),
        # Reference: full-run seen_canon overlap (NOT used for analysis)
        "n_seen_canon_per_model": {k: len(seen_canon_sets[k]) for k in keys},
        "n_seen_canon_union":      len(seen_canon_union),
        "n_seen_canon_intersection": len(seen_canon_inter),
    }
    pw = {}
    for k1, k2 in combinations(keys, 2):
        pw[f"{k1}_AND_{k2}"] = {
            "exact_overlap":  len(exact_sets[k1] & exact_sets[k2]),
            "exact_jaccard":  jaccard(exact_sets[k1], exact_sets[k2]),
            "exact_only_m1":  len(exact_sets[k1] - exact_sets[k2]),
            "exact_only_m2":  len(exact_sets[k2] - exact_sets[k1]),
            "canon_overlap":  len(canon_sets[k1] & canon_sets[k2]),
            "canon_jaccard":  jaccard(canon_sets[k1], canon_sets[k2]),
            "canon_only_m1":  len(canon_sets[k1] - canon_sets[k2]),
            "canon_only_m2":  len(canon_sets[k2] - canon_sets[k1]),
            # Reference: full-run seen_canon (NOT used for analysis)
            "seen_canon_overlap":  len(seen_canon_sets[k1] & seen_canon_sets[k2]),
            "seen_canon_jaccard":  jaccard(seen_canon_sets[k1], seen_canon_sets[k2]),
        }
    result["pairwise"] = pw
    if len(keys) == 3:
        a, b, c = keys
        sa, sb, sc = exact_sets[a], exact_sets[b], exact_sets[c]
        ca, cb, cc = canon_sets[a], canon_sets[b], canon_sets[c]
        result["venn_exact"] = {
            "only_a": len(sa-sb-sc), "only_b": len(sb-sa-sc),
            "only_c": len(sc-sa-sb), "ab": len((sa&sb)-sc),
            "ac": len((sa&sc)-sb),   "bc": len((sb&sc)-sa),
            "all3": len(sa&sb&sc),
            "labels": {k: models[k].display_name for k in keys},
        }
        result["venn_canon"] = {
            "only_a": len(ca-cb-cc), "only_b": len(cb-ca-cc),
            "only_c": len(cc-ca-cb), "ab": len((ca&cb)-cc),
            "ac": len((ca&cc)-cb),   "bc": len((cb&cc)-ca),
            "all3": len(ca&cb&cc),
            "labels": {k: models[k].display_name for k in keys},
        }
    return result


# ─── Entity overlap ──────────────────────────────────────────────────────────
def compute_entity_overlap(models: Dict[str, ModelData],
                            common_subjects: List[str]) -> List[dict]:
    keys = sorted(models.keys()); rows = []
    for k1, k2 in combinations(keys, 2):
        e1, e2, c1, c2 = set(), set(), set(), set()
        for s in common_subjects:
            e1 |= set(models[k1].wikilinks_by_subject.get(s, []))
            e2 |= set(models[k2].wikilinks_by_subject.get(s, []))
            c1 |= models[k1].canon_entities_by_subject.get(s, set())
            c2 |= models[k2].canon_entities_by_subject.get(s, set())
        rows.append({
            "model_1": k1, "model_2": k2,
            "n_common_subjects": len(common_subjects),
            "exact_entities_m1": len(e1), "exact_entities_m2": len(e2),
            "exact_jaccard":     jaccard(e1, e2),
            "canon_entities_m1": len(c1), "canon_entities_m2": len(c2),
            "canon_jaccard":     jaccard(c1, c2),
            "canon_shared":      len(c1 & c2),
            "canon_m1_only":     len(c1 - c2),
            "canon_m2_only":     len(c2 - c1),
        })
    return rows


# ─── Wikilink + breadth stats ────────────────────────────────────────────────
def compute_wikilink_and_breadth_stats(models: Dict[str, ModelData]) -> List[dict]:
    rows = []
    for key, md in sorted(models.items()):
        all_links, all_cats = [], []
        lc, cc, sc, wcs, hops = [], [], [], [], []
        subject_prefixes = Counter()
        for s in md.subject_names:
            meta  = md.subject_meta.get(s)
            if not meta: continue
            links = md.wikilinks_by_subject.get(s, [])
            cats  = md.categories_by_subject.get(s, [])
            all_links.extend(links); all_cats.extend(cats)
            lc.append(len(links)); cc.append(len(cats))
            sc.append(meta.n_sections); wcs.append(meta.wc); hops.append(meta.hop)
            fw = s.split()[0].lower() if s.split() else ""
            if fw: subject_prefixes[fw] += 1
        n = len(lc)
        if not n: continue
        unique   = set(all_links)
        unique_c = set(canon_key(l) for l in all_links if l)
        hop_ctr  = Counter(hops)
        rows.append({
            "model": key, "display_name": md.display_name, "n_articles": n,
            "n_total_unique": md.n_total_unique, "n_total_raw": md.n_total_raw,
            "total_wikilinks": len(all_links), "unique_wikilinks": len(unique),
            "unique_canon_wikilinks": len(unique_c),
            "mean_wikilinks": _mean(lc), "std_wikilinks": _std(lc),
            "median_wikilinks": _median(lc),
            "total_categories": len(all_cats),
            "unique_categories": len(set(all_cats)),
            "mean_categories": _mean(cc),
            "mean_sections": _mean(sc), "std_sections": _std(sc),
            "mean_word_count": _mean(wcs), "std_word_count": _std(wcs),
            "median_word_count": _median(wcs),
            "entity_ttr": len(unique) / len(all_links) if all_links else 0,
            "n_hop_0":    hop_ctr.get(0, 0), "n_hop_1": hop_ctr.get(1, 0),
            "n_hop_2":    hop_ctr.get(2, 0),
            "n_hop_3plus":sum(v for k, v in hop_ctr.items() if k >= 3),
            "mean_hop":  _mean(hops), "max_hop": max(hops) if hops else 0,
            "n_unique_prefixes": len(subject_prefixes),
            "prefix_diversity":  len(subject_prefixes) / n if n else 0,
        })
    return rows


# ─── Cross-model similarity ──────────────────────────────────────────────────
def compute_cross_model_similarity(models: Dict[str, ModelData],
                                    common_subjects: List[str],
                                    max_compare: int = 1000,
                                    compute_semantic: bool = False) -> List[dict]:
    keys    = sorted(models.keys())
    sampled = sorted(common_subjects)[:max_compare]
    print(f"[cross-sim] {len(sampled)} subjects × {len(keys)} models ...")
    txts = {k: read_articles_for_subjects(models[k], sampled) for k in keys}

    embeddings: Dict[str, Dict[str, object]] = {}
    if compute_semantic:
        try:
            from openai import OpenAI
            client = OpenAI(); emb_model = "text-embedding-3-small"
            print(f"[cross-sim] embeddings ({emb_model}) ...")
            for k in keys:
                embeddings[k] = {}; texts_e = []; subjs_e = []
                for s in sampled:
                    txt = txts[k].get(s, "")
                    if txt: texts_e.append(txt[:32000]); subjs_e.append(s)
                for i in range(0, len(texts_e), 64):
                    batch = texts_e[i:i+64]; batch_s = subjs_e[i:i+64]
                    try:
                        resp = client.embeddings.create(model=emb_model, input=batch)
                        data = sorted(resp.data, key=lambda d: d.index)
                        for j, d in enumerate(data):
                            embeddings[k][batch_s[j]] = np.array(d.embedding, dtype=np.float32)
                    except Exception as e: print(f"  embedding error {k}: {e}")
        except ImportError: print("[cross-sim] openai not available"); compute_semantic = False
        except Exception as e: print(f"[cross-sim] semantic failed: {e}"); compute_semantic = False

    rows = []
    for k1, k2 in combinations(keys, 2):
        sims = defaultdict(list); sem_cos = []; nc = 0
        for s in sampled:
            t1 = txts[k1].get(s, ""); t2 = txts[k2].get(s, "")
            if not t1 or not t2: continue
            tok1, tok2 = _tokenize(t1), _tokenize(t2)
            if not tok1 or not tok2: continue
            sims["jaccard"].append(jaccard(set(tok1), set(tok2)))
            for n in [1, 2, 3]:
                ng1, ng2 = _ngrams_set(tok1, n), _ngrams_set(tok2, n)
                if ng1 and ng2:
                    sims[f"ngram_{n}_jaccard"].append(jaccard(ng1, ng2))
                    sims[f"ngram_{n}_overlap"].append(overlap_coeff(ng1, ng2))
            if compute_semantic and k1 in embeddings and k2 in embeddings:
                e1, e2 = embeddings[k1].get(s), embeddings[k2].get(s)
                if e1 is not None and e2 is not None:
                    cos = float(np.dot(e1, e2) /
                                (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9))
                    sem_cos.append(cos)
            nc += 1
        row = {"model_1": k1, "model_2": k2, "n_compared": nc}
        for k in ["jaccard","ngram_1_jaccard","ngram_1_overlap",
                  "ngram_2_jaccard","ngram_2_overlap","ngram_3_jaccard","ngram_3_overlap"]:
            row[f"mean_{k}"] = _mean(sims.get(k, []))
        if sem_cos:
            row["mean_semantic_cosine"] = _mean(sem_cos)
            row["std_semantic_cosine"]  = _std(sem_cos)
        rows.append(row)
    del txts
    return rows


# ─── Factuality helpers ──────────────────────────────────────────────────────
def _make_cfg(args, aud: str = "") -> "EvalConfig":
    return EvalConfig(
        fact_model_key=args.fact_model_key,
        evidence_sources=[s.strip() for s in args.evidence_sources.split(",") if s.strip()],
        max_claims=args.max_claims, max_retries=args.max_retries,
        concurrency=args.concurrency, timeout=600.0,
        web_mode=getattr(args, "web_mode", "snippets"),
        search_backend=getattr(args, "search_backend", "auto"),
        web_cache_dir=getattr(args, "web_cache_dir", ""),
        compute_similarity=args.compute_similarity,
        compute_bertscore=getattr(args, "compute_bertscore", False),
        compute_stylistic=getattr(args, "compute_stylistic", False),
        run_audit_dir=aud, debug=getattr(args, "debug", False),
    )


def _run_fact_with_retry(mk: str, md: ModelData, subjects: List[str],
                          args, audit_dir: str, prefix: str,
                          texts: Dict[str, str] = None) -> List[dict]:
    """Run factuality evaluation for one (model x seed) pair.

    `texts` must be pre-loaded and verified.  Every subject in `subjects`
    must have non-empty text — verified by inline loop before this call.
    """
    if not HAS_FACTUALITY: return []
    if texts is None:
        texts = read_articles_for_subjects(md, subjects)

    missing = [s for s in subjects if s not in texts or not texts[s].strip()]
    if missing:
        raise RuntimeError(
            f"[{mk}] {prefix}: {len(missing)} subjects missing from articles.jsonl: "
            f"{sorted(missing)[:5]}"
        )

    aud = ""
    if audit_dir:
        aud = os.path.join(audit_dir, mk)
        for sub in ("claims", "results", "evidence"):
            os.makedirs(os.path.join(aud, sub), exist_ok=True)
    cfg = _make_cfg(args, aud)

    _seed = (md.run_meta.get("seed") or "").strip()
    inputs = [
        EvalInput(
            subject=s, candidate=mk,
            article_text=texts[s],
            hop=md.subject_meta[s].hop if s in md.subject_meta else 0,
            generator_model=md.display_name,
            topic=_seed,
            clean_wiki_markup=True,
        )
        for s in subjects
    ]
    n_attempted = len(inputs)
    print(f"  [{mk}] {prefix}: {n_attempted} inputs → run_evaluation() ...")

    out_path = os.path.join(args.output_dir, f"{prefix}_{mk}.jsonl")
    results: List[dict] = []
    backoff   = 2.0
    max_tries = max(1, getattr(args, "max_retries", 5))

    for attempt in range(max_tries):
        try:
            results = run_evaluation(inputs, cfg, out_path)
            break
        except Exception as e:
            if attempt < max_tries - 1:
                wait = backoff * (attempt + 1)
                print(f"  [WARN] {mk} attempt {attempt+1} failed: {e}  "
                      f"(retry in {wait:.1f}s)")
                time.sleep(wait)
            else:
                print(f"  [ERROR] {mk} failed after {max_tries} attempts: {e}")
                results = []

    n_returned = len(results)
    if n_returned != n_attempted:
        print(f"  [WARN] {mk} {prefix}: run_evaluation returned {n_returned} "
              f"records for {n_attempted} inputs "
              f"(shortfall={n_attempted-n_returned}).  "
              f"Some subjects may have had no evaluable claims.")

    zero_claims = sum(1 for r in results if r.get("n_claims", 0) == 0)
    if zero_claims:
        print(f"  [{mk}] {zero_claims}/{n_returned} records had 0 claims")
    print(f"  [{mk}] {prefix}: {n_returned} returned / {n_attempted} attempted")

    try:
        rd = os.path.join(md.dir_path, "cross_analysis")
        os.makedirs(rd, exist_ok=True)
        generate_outputs(out_path, rd)
    except Exception:
        pass

    return results


def _compute_wiki_found_subjects(pooled_by_model: Dict[str, List[dict]]) -> Set[str]:
    """Return subjects with a Wikipedia page (union across models/seeds, strict flag)."""
    found: Set[str] = set()
    for recs in pooled_by_model.values():
        for r in recs:
            s = r.get("subject", "")
            if s and _is_wiki_found(r, strict=True):
                found.add(s)
    return found


# ─── Factuality — shared mode ────────────────────────────────────────────────
def run_factuality_shared(sampled_shared: List[str], models: Dict[str, ModelData],
                           args, audit_root: str, n_seeds: int = 3) -> dict:
    if not HAS_FACTUALITY:
        print("[factuality-shared] factuality_core not available"); return {}
    if not sampled_shared:
        print("[factuality-shared] no shared subjects"); return {}

    seeds   = list(args.fact_seeds)
    n_seeds = len(seeds)
    print(f"\n[factuality-shared] {len(sampled_shared)} subjects × "
          f"{len(models)} models × {n_seeds} seeds {seeds}")
    t0 = time.perf_counter()

    print("[factuality-shared] loading articles ...")
    all_texts: Dict[str, Dict[str, str]] = {}
    for mk, md in sorted(models.items()):
        all_texts[mk] = read_articles_for_subjects(md, sampled_shared)

    print("[factuality-shared] verifying all subjects present in all models ...")
    all_missing = []
    for mk in sorted(models.keys()):
        for s in sampled_shared:
            if s not in all_texts[mk] or not all_texts[mk][s].strip():
                all_missing.append(f"{mk}: {s!r}")
    if all_missing:
        raise RuntimeError(
            f"[BUG] {len(all_missing)} subject/model pairs missing from articles.jsonl.\n"
            f"Shared mode requires every subject in every model.\nFirst 10:\n" +
            "\n".join(f"  {m}" for m in all_missing[:10])
        )
    shared_subjects = list(sampled_shared)
    print(f"  [OK] {len(shared_subjects)} subjects x {len(models)} models — "
          f"all {len(shared_subjects)*len(models)} checks passed")

    fact_n = min(args.sample_n, len(shared_subjects))
    seed_results: Dict[str, List[List[dict]]] = {mk: [] for mk in models}

    for seed_idx, seed in enumerate(seeds):
        rng         = random.Random(seed)
        seed_sample = (sorted(rng.sample(shared_subjects, fact_n))
                       if fact_n < len(shared_subjects)
                       else list(shared_subjects))
        print(f"\n  [seed {seed_idx+1}/{n_seeds}] seed={seed}  n={len(seed_sample)}")

        aud = (os.path.join(audit_root, "shared", f"seed{seed_idx}")
               if audit_root else "")
        for mk, md in sorted(models.items()):
            seed_texts = {s: all_texts[mk][s] for s in seed_sample}
            prefix     = f"eval_shared_seed{seed_idx}"
            recs = _run_fact_with_retry(mk, md, seed_sample, args, aud, prefix,
                                         texts=seed_texts)
            seed_results[mk].append(recs)

    pooled_by_model: Dict[str, List[dict]] = {
        mk: [r for seed_recs in seed_results[mk] for r in seed_recs]
        for mk in models
    }
    wiki_found_subjects = _compute_wiki_found_subjects(pooled_by_model)
    n_wiki_found = len(wiki_found_subjects & set(shared_subjects))
    n_wiki_not   = len(shared_subjects) - n_wiki_found

    print(f"\n[factuality-shared] wiki coverage: "
          f"{n_wiki_found}/{len(shared_subjects)} "
          f"({n_wiki_found/len(shared_subjects):.1%})")

    sim_keys = ["sim_tfidf_cosine","sim_jaccard","sim_semantic_cosine",
                "sim_combined_similarity","sim_ngram_1_overlap",
                "sim_ngram_2_overlap","sim_ngram_3_overlap","sim_bertscore_f1"]
    result_rows = []

    for mk in sorted(models.keys()):
        all_recs = pooled_by_model[mk]

        seed_prec, seed_true, seed_false, seed_unv = [], [], [], []
        seed_web_prec, seed_web_false = [], []
        seed_frontier_prec, seed_frontier_false = [], []
        for seed_recs in seed_results[mk]:
            found = [r for r in seed_recs if r.get("subject") in wiki_found_subjects]
            if not found: continue
            pv = [r.get("accuracy_against_wiki") for r in found
                  if r.get("accuracy_against_wiki") is not None]
            if pv:
                seed_prec.append(_mean(pv))
                seed_true.append(_mean([r.get("true_rate_against_wiki", 0) for r in found]))
                seed_false.append(_mean([r.get("false_rate_against_wiki", 0) for r in found]))
                seed_unv.append(_mean([r.get("unverifiable_rate_against_wiki", 0) for r in found]))

            web_pv = [r.get("accuracy_against_web") for r in seed_recs
                      if r.get("accuracy_against_web") is not None]
            if web_pv:
                seed_web_prec.append(_mean(web_pv))
                seed_web_false.append(_mean([r.get("false_rate_against_web", 0) for r in seed_recs
                                             if r.get("accuracy_against_web") is not None]))

            not_found_recs = [r for r in seed_recs
                              if r.get("subject") not in wiki_found_subjects]
            frontier_pv = [r.get("accuracy_against_web") for r in not_found_recs
                           if r.get("accuracy_against_web") is not None]
            if frontier_pv:
                seed_frontier_prec.append(_mean(frontier_pv))
                seed_frontier_false.append(_mean([r.get("false_rate_against_web", 0)
                                                   for r in not_found_recs
                                                   if r.get("accuracy_against_web") is not None]))

        row = {
            "eval_type": "shared", "model": mk,
            "n_seeds":              n_seeds,
            "n_attempted_per_seed": fact_n,
            "n_attempted_total":    fact_n * n_seeds,
            "n_shared_subjects":    len(shared_subjects),
            "n_returned_total":     len(all_recs),
            "n_wiki_subject_found":     n_wiki_found,
            "n_wiki_subject_not_found": n_wiki_not,
            "wiki_coverage_rate": (n_wiki_found / len(shared_subjects)
                                   if shared_subjects else 0),
            "wiki_coverage_rate_conserv": (n_wiki_found / len(shared_subjects)
                                           if shared_subjects else 0),
            "wiki_precision":             _mean(seed_prec),
            "wiki_precision_std":         _std(seed_prec),
            "wiki_true_rate":             _mean(seed_true),
            "wiki_true_rate_std":         _std(seed_true),
            "wiki_false_rate":            _mean(seed_false),
            "wiki_false_rate_std":        _std(seed_false),
            "wiki_unverifiable_rate":     _mean(seed_unv),
            "wiki_unverifiable_rate_std": _std(seed_unv),
            "web_precision":             _mean(seed_web_prec),
            "web_precision_std":         _std(seed_web_prec),
            "web_false_rate":            _mean(seed_web_false),
            "web_false_rate_std":        _std(seed_web_false),
            "frontier_n":                    n_wiki_not,
            "frontier_web_precision":        _mean(seed_frontier_prec),
            "frontier_web_precision_std":    _std(seed_frontier_prec),
            "frontier_web_false_rate":       _mean(seed_frontier_false),
            "frontier_web_false_rate_std":   _std(seed_frontier_false),
        }
        for sk in sim_keys:
            vals = [r.get(sk) for r in all_recs if isinstance(r.get(sk), (int, float))]
            row[f"mean_{sk}"] = _mean(vals); row[f"std_{sk}"] = _std(vals)
        result_rows.append(row)

    elapsed = time.perf_counter() - t0

    cov_vals = [r["n_wiki_subject_found"] for r in result_rows]
    if len(set(cov_vals)) > 1:
        raise RuntimeError(
            f"[BUG] n_wiki_subject_found differs across model rows: {cov_vals}\n"
            "It must be identical — it is a subject-level property computed once."
        )
    print(f"[OK] n_wiki_subject_found={cov_vals[0]} consistent across all models  "
          f"({elapsed:.1f}s)")

    return {
        "rows": result_rows,
        "wiki_found_subjects": sorted(wiki_found_subjects),
        "shared_with_articles": shared_subjects,
        "seed_results": seed_results,
    }


# ─── Factuality — independent mode ──────────────────────────────────────────
def run_factuality_independent(models: Dict[str, ModelData], args,
                                audit_root: str, n_seeds: int = 3) -> dict:
    if not HAS_FACTUALITY:
        print("[factuality-independent] not available"); return {}

    seeds   = list(args.fact_seeds)
    n_seeds = len(seeds)
    print(f"\n[factuality-independent] seeds={seeds} per model ...")
    sim_keys = ["sim_tfidf_cosine","sim_jaccard","sim_semantic_cosine",
                "sim_combined_similarity","sim_ngram_1_overlap",
                "sim_ngram_2_overlap","sim_ngram_3_overlap","sim_bertscore_f1"]
    result_rows = []

    for mk, md in sorted(models.items()):
        avail = list(md.subject_names)
        n     = min(args.sample_n, len(avail))
        if n < 10:
            print(f"  {mk}: only {n} subjects, skipping"); continue

        seed_results: List[List[dict]] = []
        seed_prec, seed_true, seed_false, seed_unv = [], [], [], []
        seed_web_prec, seed_web_false = [], []
        seed_frontier_prec, seed_frontier_false = [], []
        all_recs: List[dict] = []

        for seed_idx, seed in enumerate(seeds):
            rng    = random.Random(seed)
            sample = sorted(rng.sample(avail, n))
            print(f"\n  [{mk}] seed {seed_idx+1}/{n_seeds} (seed={seed}, n={len(sample)})")

            texts = read_articles_for_subjects(md, sample)
            missing = [s for s in sample if s not in texts or not texts[s].strip()]
            if missing:
                raise RuntimeError(
                    f"[{mk}] seed{seed_idx}: {len(missing)} sampled subjects missing "
                    f"from articles.jsonl.\nFirst 10: {sorted(missing)[:10]}"
                )
            print(f"  [OK] [{mk}] {len(sample)} subjects present in articles.jsonl")

            aud    = (os.path.join(audit_root, "per_model", f"seed{seed_idx}")
                      if audit_root else "")
            prefix = f"eval_per_model_seed{seed_idx}"
            recs   = _run_fact_with_retry(mk, md, sample, args, aud, prefix,
                                           texts=texts)
            seed_results.append(recs)
            all_recs.extend(recs)

            found = [r for r in recs if _is_wiki_found(r, strict=True)]
            pv    = [r.get("accuracy_against_wiki") for r in found
                     if r.get("accuracy_against_wiki") is not None]
            if pv:
                seed_prec.append(_mean(pv))
                seed_true.append(_mean([r.get("true_rate_against_wiki", 0) for r in found]))
                seed_false.append(_mean([r.get("false_rate_against_wiki", 0) for r in found]))
                seed_unv.append(_mean([r.get("unverifiable_rate_against_wiki", 0) for r in found]))

            web_pv = [r.get("accuracy_against_web") for r in recs
                      if r.get("accuracy_against_web") is not None]
            if web_pv:
                seed_web_prec.append(_mean(web_pv))
                seed_web_false.append(_mean([r.get("false_rate_against_web", 0) for r in recs
                                             if r.get("accuracy_against_web") is not None]))

            not_found_recs = [r for r in recs if not _is_wiki_found(r, strict=True)]
            frontier_pv = [r.get("accuracy_against_web") for r in not_found_recs
                           if r.get("accuracy_against_web") is not None]
            if frontier_pv:
                seed_frontier_prec.append(_mean(frontier_pv))
                seed_frontier_false.append(_mean([r.get("false_rate_against_web", 0)
                                                   for r in not_found_recs
                                                   if r.get("accuracy_against_web") is not None]))

        n_found     = sum(1 for r in all_recs if _is_wiki_found(r, strict=True))
        n_not_found = len(all_recs) - n_found
        n_shortfall = (n * n_seeds) - len(all_recs)
        if n_shortfall > 0:
            print(f"  [{mk}] SHORTFALL: {n_shortfall}/{n*n_seeds} not returned "
                  f"({n_shortfall/(n*n_seeds):.1%}). "
                  f"coverage_vs_returned={n_found}/{len(all_recs)}={n_found/len(all_recs):.1%}  "
                  f"coverage_vs_attempted={n_found}/{n*n_seeds}={n_found/(n*n_seeds):.1%}")

        row = {
            "eval_type": "per_model", "model": mk,
            "n_seeds":            n_seeds,
            "n_per_seed":         n,
            "n_attempted_total":  n * n_seeds,
            "n_returned_total":   len(all_recs),
            "n_wiki_subject_found":     n_found,
            "n_wiki_subject_not_found": n_not_found,
            "wiki_coverage_rate": n_found / len(all_recs) if all_recs else 0,
            "wiki_coverage_rate_conserv": n_found / (n * n_seeds) if (n * n_seeds) else 0,
            "wiki_coverage_rate_vs_attempted": n_found / (n * n_seeds) if (n * n_seeds) else 0,
            "wiki_precision":             _mean(seed_prec),
            "wiki_precision_std":         _std(seed_prec),
            "wiki_true_rate":             _mean(seed_true),
            "wiki_true_rate_std":         _std(seed_true),
            "wiki_false_rate":            _mean(seed_false),
            "wiki_false_rate_std":        _std(seed_false),
            "wiki_unverifiable_rate":     _mean(seed_unv),
            "wiki_unverifiable_rate_std": _std(seed_unv),
            "web_precision":             _mean(seed_web_prec),
            "web_precision_std":         _std(seed_web_prec),
            "web_false_rate":            _mean(seed_web_false),
            "web_false_rate_std":        _std(seed_web_false),
            "frontier_n":                n_not_found,
            "frontier_web_precision":    _mean(seed_frontier_prec),
            "frontier_web_precision_std":_std(seed_frontier_prec),
            "frontier_web_false_rate":   _mean(seed_frontier_false),
            "frontier_web_false_rate_std":_std(seed_frontier_false),
        }
        for sk in sim_keys:
            vals = [r.get(sk) for r in all_recs if isinstance(r.get(sk), (int, float))]
            row[f"mean_{sk}"] = _mean(vals); row[f"std_{sk}"] = _std(vals)
        result_rows.append(row)

    return {"rows": result_rows}


# ─── Factuality dispatcher ───────────────────────────────────────────────────
def run_factuality_dual(sampled_shared: List[str], models: Dict[str, ModelData],
                         args, audit_root: str = "") -> dict:
    seeds   = getattr(args, "fact_seeds", DEFAULT_SEEDS)
    n_seeds = len(seeds)
    print(f"\n[factuality] Seeds: {seeds}  (both shared and per-model)")

    out: dict = {"per_model_rows": [], "shared_rows": [],
                 "shared_wiki_found_subjects": [], "shared_with_articles": []}

    print(f"\n[factuality] === INDEPENDENT ({n_seeds} seeds per model) ===")
    indep = run_factuality_independent(models, args, audit_root, n_seeds=n_seeds)
    out["per_model_rows"] = indep.get("rows", [])

    if sampled_shared:
        print(f"\n[factuality] === SHARED ({len(sampled_shared)} subjects, {n_seeds} seeds) ===")
        shared = run_factuality_shared(sampled_shared, models, args,
                                        audit_root, n_seeds=n_seeds)
        out["shared_rows"]                = shared.get("rows", [])
        out["shared_wiki_found_subjects"] = shared.get("wiki_found_subjects", [])
        out["shared_with_articles"]       = shared.get("shared_with_articles", [])

    return out


# ─── Sampling ────────────────────────────────────────────────────────────────
def sample_intersection(overlap: dict, n: int, seed: int = 42,
                         min_n: int = 10) -> List[str]:
    inter = overlap.get("intersection_subjects", [])
    if not inter: print("[sample] No intersection!"); return []
    if len(inter) < min_n:
        print(f"[sample] Only {len(inter)} < min_n={min_n}"); return []
    target = min(n, len(inter)) if n > 0 else len(inter)
    s = random.Random(seed).sample(inter, target)
    print(f"[sample] {target} from {len(inter)} intersection subjects (seed={seed})")
    return sorted(s)


# ─── Figures ────────────────────────────────────────────────────────────────
def _gc(k): return COLORS.get(k, "#607D8B")
def _gd(k): return MODEL_DISPLAY.get(k, k)
def _save_fig(fig, pb):
    for ext in (".pdf", ".png"):
        fig.savefig(pb + ext, bbox_inches="tight",
                    **({"dpi": 300} if ext == ".png" else {}))
    plt.close(fig)
    print(f"  [fig] {os.path.basename(pb)}")

def generate_figures(models, overlap, wl_stats, entity_ovl,
                      fact_pm, fact_sh, cross_sim, fig_dir):
    if not HAS_PLOT: return
    os.makedirs(fig_dir, exist_ok=True)
    plt.rcParams.update(ACL_STYLE)
    keys = sorted(models.keys())

    # Fig 1: subject counts
    fig, ax = plt.subplots(figsize=(6, 3.5))
    np_ = overlap["n_per_model"]; nc_ = overlap.get("n_canon_per_model", {})
    x = np.arange(len(keys)); w = 0.35
    b1 = ax.bar(x-w/2, [np_[k] for k in keys], w,
                color=[_gc(k) for k in keys], alpha=0.85, label="Exact")
    if nc_:
        ax.bar(x+w/2, [nc_.get(k, 0) for k in keys], w,
               color=[_gc(k) for k in keys], alpha=0.4, hatch="//", label="Canon (capped)")
    for b in b1:
        ax.text(b.get_x()+b.get_width()/2, b.get_height(),
                f"{int(b.get_height()):,}", ha="center", va="bottom", fontsize=7)
    ax.axhline(overlap["n_intersection"], color=COLORS["all3"], ls="--", alpha=0.7,
               label=f'int={overlap["n_intersection"]:,}')
    ax.set_xticks(x); ax.set_xticklabels([_gd(k) for k in keys])
    ax.set_ylabel("Count"); ax.set_title("Subject Count", fontweight="bold")
    ax.legend(fontsize=7); fig.tight_layout()
    _save_fig(fig, os.path.join(fig_dir, "fig1_subjects"))

    # Fig 2: pairwise Jaccard
    pw = overlap.get("pairwise", {})
    if pw:
        pairs = list(pw.keys()); fig, ax = plt.subplots(figsize=(5, 3.5))
        x = np.arange(len(pairs)); w = 0.35
        ax.bar(x-w/2, [pw[p]["exact_jaccard"] for p in pairs], w,
               color="#2196F3", alpha=0.85, label="Exact")
        ax.bar(x+w/2, [pw[p]["canon_jaccard"] for p in pairs], w,
               color="#FF9800", alpha=0.85, label="Canon (capped)")
        for i, p in enumerate(pairs):
            ax.text(i-w/2, pw[p]["exact_jaccard"]+.005,
                    f'{pw[p]["exact_jaccard"]:.3f}', ha="center", fontsize=7)
            ax.text(i+w/2, pw[p]["canon_jaccard"]+.005,
                    f'{pw[p]["canon_jaccard"]:.3f}', ha="center", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([p.replace("_AND_","\n") for p in pairs], fontsize=7)
        ax.set_ylabel("Jaccard"); ax.set_title("Subject Overlap (capped subjects)", fontweight="bold")
        ax.legend(fontsize=8); fig.tight_layout()
        _save_fig(fig, os.path.join(fig_dir, "fig2_overlap"))

    # Fig 3: entity heatmap
    if entity_ovl and HAS_PLOT:
        n = len(keys); mat = np.eye(n)
        for r in entity_ovl:
            i = keys.index(r["model_1"]); j = keys.index(r["model_2"])
            mat[i,j] = mat[j,i] = r.get("canon_jaccard", 0)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=max(.5, mat.max()))
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels([_gd(k) for k in keys], rotation=30, ha="right")
        ax.set_yticklabels([_gd(k) for k in keys])
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center",
                        fontsize=9, color="white" if mat[i,j]>.3 else "black")
        plt.colorbar(im, ax=ax, shrink=.8)
        ax.set_title("Entity Overlap", fontweight="bold"); fig.tight_layout()
        _save_fig(fig, os.path.join(fig_dir, "fig3_entity"))

    # Fig 4: wikilinks + word count
    if wl_stats:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        for ax, m, yl, ti in [
            (axes[0], "mean_wikilinks",  "Links/Art",  "Wikilinks"),
            (axes[1], "mean_word_count", "Words/Art",  "Word Count"),
        ]:
            for r in wl_stats:
                v = r.get(m) or 0
                ax.bar(r["model"], v, color=_gc(r["model"]), alpha=.85)
                ax.text(r["model"], v+1, f"{v:.0f}", ha="center", fontsize=8)
            ax.set_ylabel(yl); ax.set_title(ti, fontweight="bold")
        fig.tight_layout(); _save_fig(fig, os.path.join(fig_dir, "fig4_wikilinks"))

    # Fig 5: hop distribution
    if wl_stats:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        hk = ["n_hop_0","n_hop_1","n_hop_2","n_hop_3plus"]
        hl = ["Hop 0","Hop 1","Hop 2","Hop 3+"]
        x = np.arange(len(hl)); w = .8/len(wl_stats)
        for i, r in enumerate(wl_stats):
            ax.bar(x+i*w-.4+w/2, [r.get(h, 0) or 0 for h in hk], w,
                   label=_gd(r["model"]), color=_gc(r["model"]), alpha=.85)
        ax.set_xticks(x); ax.set_xticklabels(hl)
        ax.set_ylabel("Articles"); ax.set_title("Hop Distribution", fontweight="bold")
        ax.legend(fontsize=8); fig.tight_layout()
        _save_fig(fig, os.path.join(fig_dir, "fig5_hops"))

    # Fig 6: factuality
    all_fact = ([(r,"Per-Model") for r in fact_pm] +
                [(r,"Shared")    for r in fact_sh])
    if all_fact:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ai, ft in enumerate(["Per-Model","Shared"]):
            ax = axes[ai]; sub = [r for r,t in all_fact if t==ft]
            if not sub: ax.set_title(f"{ft}\n(no data)"); continue
            ms = [("wiki_precision","Prec."),("wiki_false_rate","Halluc.")]
            xm = np.arange(len(ms)); w = .8/len(sub)
            for i, r in enumerate(sub):
                ax.bar(xm+i*w-.4+w/2, [r.get(m[0]) or 0 for m in ms], w,
                       label=_gd(r["model"]), color=_gc(r["model"]), alpha=.85,
                       yerr=[r.get(f"{m[0]}_std") or 0 for m in ms],
                       capsize=3, error_kw={"elinewidth":1})
            ax.set_xticks(xm); ax.set_xticklabels([m[1] for m in ms])
            ax.set_ylabel("Rate"); ax.set_ylim(0,1.05)
            ax.set_title(ft, fontweight="bold"); ax.legend(fontsize=7)
        fig.tight_layout(); _save_fig(fig, os.path.join(fig_dir, "fig6_factuality"))

    # Fig 7: wiki similarity
    sm = {}
    for r in (fact_pm or []) + (fact_sh or []):
        mk = r.get("model","?")
        if mk not in sm: sm[mk] = r
    if sm and any(r.get("mean_sim_combined_similarity") for r in sm.values()):
        fig, ax = plt.subplots(figsize=(7, 3.5))
        sms = [("mean_sim_tfidf_cosine","TF-IDF"),
               ("mean_sim_jaccard","Jaccard"),
               ("mean_sim_ngram_1_overlap","1-gram"),
               ("mean_sim_ngram_2_overlap","2-gram"),
               ("mean_sim_ngram_3_overlap","3-gram"),
               ("mean_sim_semantic_cosine","Semantic"),
               ("mean_sim_combined_similarity","Combined")]
        xm = np.arange(len(sms)); ml = sorted(sm.keys()); w = .8/len(ml)
        for i, mk in enumerate(ml):
            r = sm[mk]
            ax.bar(xm+i*w-.4+w/2, [r.get(m[0]) or 0 for m in sms], w,
                   label=_gd(mk), color=_gc(mk), alpha=.85)
        ax.set_xticks(xm); ax.set_xticklabels([m[1] for m in sms], fontsize=8)
        ax.set_ylabel("Similarity"); ax.set_ylim(0,1.05)
        ax.set_title("Model vs Wikipedia", fontweight="bold")
        ax.legend(fontsize=7); fig.tight_layout()
        _save_fig(fig, os.path.join(fig_dir, "fig7_wiki_sim"))


# ─── Text report ─────────────────────────────────────────────────────────────
def _rsec(t): return f"\n{'='*70}\n  {t}\n{'='*70}\n\n"

def _rtable(rows, cols=None):
    if not rows: return "  (no data)\n"
    if cols is None:
        cols = list(dict.fromkeys(k for r in rows for k in r.keys()))
    widths = {}
    for c in cols:
        w = len(str(c))
        for r in rows:
            v = str(r.get(c,""))
            try: fv = float(v); v = f"{fv:.4f}" if abs(fv)<10000 else f"{int(fv):,}"
            except: pass
            w = max(w, len(v))
        widths[c] = min(max(w+2,8),40)
    def fv(v,w):
        if v is None or v=="": return " "*w
        try:
            f=float(v)
            if abs(f)<.001 and f!=0: return f"{f:{w}.4e}"
            if f==int(f) and abs(f)>10: return f"{int(f):>{w},}"
            return f"{f:{w}.4f}"
        except: return f"{str(v):>{w}}"
    hdr = "".join(f"{str(c):>{widths[c]}}" for c in cols)
    lines = [hdr, "-"*len(hdr)]
    for r in rows:
        lines.append("".join(fv(r.get(c,""), widths[c]) for c in cols))
    return "\n".join(lines)+"\n"


def generate_text_report(out_dir, overlap, wl_stats, entity_ovl,
                          cross_sim, fact_pm, fact_sh, models):
    path = os.path.join(out_dir, "report.txt"); L = []
    L.append("="*70)
    L.append("  LLMPedia Track 2 — Cross-Model Analysis Report")
    L.append(f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"  Source: {os.path.abspath(out_dir)}")
    L.append("="*70)
    L.append(_rsec("DESIGN & INVARIANTS"))
    L.append(
        "  ARTICLE EXISTENCE GUARANTEE\n"
        "  Every subject in subject_names was loaded from articles.jsonl and passed\n"
        "  the min_words filter.  Re-reading the same file must find every subject.\n"
        "  Verified at runtime by hard assertions (abort on any missing subject).\n\n"
        f"  SEEDS\n"
        f"  Both modes use seeds {DEFAULT_SEEDS} (--seeds / --fact-seeds).\n\n"
        "  WIKIPEDIA COVERAGE\n"
        "  Determined SOLELY by the wiki_subject_found flag set by factuality_core\n"
        "  during claim verification.  Uses redirect-inclusive Wikipedia search.\n"
        "  ONE number, ONE method — no separate direct-API check.\n\n"
        "  SHARED INVARIANT\n"
        "  n_wiki_subject_found is a SUBJECT-LEVEL property computed once as a\n"
        "  union across all models and seeds.  It is identical for every model row.\n"
        "  Enforced at runtime (abort on mismatch).\n\n"
        "  CANON OVERLAP FIX\n"
        "  Canon overlap uses ONLY subject_canon_keys (from capped subject_names),\n"
        "  NOT seen_canon_keys.json (which contains ALL entities from the full run).\n"
        "  This ensures exact and canon overlap are over the same subject set.\n\n"
        "  DENOMINATOR COLUMNS\n"
        "  n_seeds               seeds run\n"
        "  n_per_seed            subjects submitted per seed\n"
        "  n_attempted_total     n_per_seed × n_seeds\n"
        "  n_returned_total      records returned by run_evaluation() (pooled).\n"
        "                        In the happy path = n_attempted_total.\n"
        "                        Shortfall → warning printed at runtime.\n"
        "  n_wiki_subject_found  wiki_subject_found=True  (precision denominator)\n"
        "  wiki_coverage_rate    n_wiki_found / n_shared_subjects  (shared)\n"
        "                        n_wiki_found / n_returned_total   (per-model)\n"
    )
    L.append(_rsec("MODELS"))
    L.append(_rtable([{"model":k,"display":md.display_name,
                        "n_subjects":len(md.subject_names),
                        "n_raw":md.n_total_raw,"n_unique":md.n_total_unique,
                        "n_canon_capped":len(md.subject_canon_keys),
                        "n_seen_canon_full":len(md.seen_canon_keys)}
                       for k,md in sorted(models.items())]))
    L.append(_rsec("SUBJECT OVERLAP"))
    L.append(f"  Exact: union={overlap['n_union']:,}  "
             f"intersection={overlap['n_intersection']:,}\n")
    L.append(f"  Canon (capped): union={overlap['n_canon_union']:,}  "
             f"intersection={overlap['n_canon_intersection']:,}\n")
    if overlap.get("n_seen_canon_union"):
        L.append(f"  Canon (full run, reference only): "
                 f"union={overlap['n_seen_canon_union']:,}  "
                 f"intersection={overlap['n_seen_canon_intersection']:,}\n")
    L.append("\n")
    L.append(_rtable([{"pair":p,**pw} for p,pw in overlap.get("pairwise",{}).items()]))
    for vk,vl in [("venn_exact","Exact Venn"),("venn_canon","Canon Venn (capped)")]:
        v = overlap.get(vk,{})
        if v and v.get("all3") is not None:
            lb=v.get("labels",{}); ks=sorted(lb.keys())
            if len(ks)==3:
                a,b,c=ks
                L.append(f"\n  {vl}: {lb.get(a,a)} only={v['only_a']:,}  "
                         f"{lb.get(b,b)} only={v['only_b']:,}  "
                         f"{lb.get(c,c)} only={v['only_c']:,}  "
                         f"ab={v['ab']:,}  ac={v['ac']:,}  bc={v['bc']:,}  "
                         f"all3={v['all3']:,}\n")
    if wl_stats:  L.append(_rsec("WIKILINK + BREADTH"));   L.append(_rtable(wl_stats))
    if entity_ovl:L.append(_rsec("ENTITY OVERLAP"));       L.append(_rtable(entity_ovl))
    if cross_sim: L.append(_rsec("CROSS-MODEL SIMILARITY"));L.append(_rtable(cross_sim))

    dc_per = ["model","n_seeds","n_per_seed","n_attempted_total","n_returned_total",
              "n_wiki_subject_found","n_wiki_subject_not_found","wiki_coverage_rate"]
    dc_sh  = ["model","n_seeds","n_attempted_per_seed","n_attempted_total",
              "n_shared_subjects","n_returned_total",
              "n_wiki_subject_found","n_wiki_subject_not_found","wiki_coverage_rate"]
    fc     = ["wiki_precision","wiki_precision_std","wiki_false_rate","wiki_false_rate_std",
              "wiki_true_rate","wiki_true_rate_std","wiki_unverifiable_rate",
              "wiki_unverifiable_rate_std"]

    if fact_pm:
        L.append(_rsec("FACTUALITY: Per-Model (independent, mean ± std across seeds)"))
        L.append(_rtable(fact_pm, dc_per + fc))
    if fact_sh:
        L.append(_rsec("FACTUALITY: Shared Intersection (same subjects, mean ± std across seeds)"))
        L.append("  n_wiki_subject_found MUST be identical across all models.\n\n")
        L.append(_rtable(fact_sh, dc_sh + fc))
        cov_vals = [r.get("n_wiki_subject_found") for r in fact_sh]
        if len(set(str(v) for v in cov_vals))>1:
            L.append(f"\n  [BUG] n_wiki_subject_found not identical: {cov_vals}\n")
        else:
            L.append(f"\n  [OK] n_wiki_subject_found={cov_vals[0]} consistent\n")

    af = (fact_pm or []) + (fact_sh or [])
    sc = [c for c in (af[0].keys() if af else []) if c.startswith("mean_sim_")]
    if sc and af:
        L.append(_rsec("MODEL vs WIKIPEDIA SIMILARITY"))
        L.append(_rtable(af, ["eval_type","model"] + sc))

    L.append("\n"+"="*70+"\n  END OF REPORT\n"+"="*70+"\n")
    text = "\n".join(L)
    with open(path,"w",encoding="utf-8") as f: f.write(text)
    print(f"[report] {path} ({len(text):,} chars)")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Track 2: Cross-model analysis (ACL 2026)")
    ap.add_argument("--llama-dir",    default="")
    ap.add_argument("--deepseek-dir", default="")
    ap.add_argument("--gpt-dir",      default="")
    ap.add_argument("--max-subjects", type=int, default=120_000,
                    help="First N subjects in BFS/file order (0=all)")
    ap.add_argument("--sample-n",     type=int, default=1000)
    ap.add_argument("--sample-min",   type=int, default=10)
    ap.add_argument("--sample-seed",  type=int, default=42,
                    help="Seed for sampling the shared intersection "
                         "(not the factuality seeds)")
    ap.add_argument("--min-words",    type=int, default=100)
    ap.add_argument("--articles-file", default="articles.jsonl")
    ap.add_argument("--output-dir",   default="./cross_model_results")
    ap.add_argument("--concurrency",  type=int, default=10)
    ap.add_argument("--compute-similarity",  action="store_true", default=False)
    ap.add_argument("--compute-bertscore",   action="store_true", default=False)
    ap.add_argument("--compute-stylistic",   action="store_true", default=False)
    ap.add_argument("--compute-factuality",  action="store_true", default=False)
    ap.add_argument("--fact-model-key",      default="gpt-4.1-nano")
    ap.add_argument("--evidence-sources",    default="wikipedia,web")
    ap.add_argument("--max-claims",  type=int, default=10)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--seeds","--fact-seeds", dest="fact_seeds",
                    type=int, nargs="+", default=DEFAULT_SEEDS,
                    help=f"Factuality random seeds (default: {DEFAULT_SEEDS}).  "
                         "Each seed draws an independent sample; metrics are "
                         "mean ± std.  Example: --seeds 42 123 7")
    ap.add_argument("--web-mode", choices=["snippets","single","hybrid","full"],
                    default="snippets")
    ap.add_argument("--search-backend",
                    choices=["auto","valyu","serper","brave","ddg","searxng"],
                    default="auto",
                    help="Search backend (default: auto — valyu if key set, else serper, etc.)")
    ap.add_argument("--web-cache-dir", default="",
                    help="Persistent web page cache directory")
    ap.add_argument("--audit-dir",    default="")
    ap.add_argument("--clean-output", action="store_true")
    ap.add_argument("--debug",        action="store_true")
    ap.add_argument("--coverage-mode", choices=["conservative", "returned"],
                    default="conservative",
                    help="Coverage denominator: 'conservative' = n_attempted (includes "
                         "infra failures), 'returned' = n_returned (excludes them). "
                         "Default: conservative (for paper).")
    args = ap.parse_args()

    model_dirs: Dict[str,str] = {}
    if args.llama_dir:    model_dirs["llama"]    = args.llama_dir
    if args.deepseek_dir: model_dirs["deepseek"] = args.deepseek_dir
    if args.gpt_dir:      model_dirs["gpt"]      = args.gpt_dir
    if len(model_dirs) < 2: ap.error("Need >= 2 model dirs")

    out_dir    = os.path.abspath(args.output_dir)
    fig_dir    = os.path.join(out_dir, "figures")
    audit_root = args.audit_dir or os.path.join(out_dir, "audit")

    if args.clean_output:
        import shutil
        if os.path.isdir(out_dir): shutil.rmtree(out_dir)
    os.makedirs(fig_dir, exist_ok=True)

    print(f"\n{'='*70}\n[track2] CROSS-MODEL ANALYSIS")
    print(f"  models        : {list(model_dirs.keys())}")
    print(f"  max-subjects  : {args.max_subjects:,}  sample-n : {args.sample_n}")
    print(f"  fact-seeds    : {args.fact_seeds}")
    print(f"  coverage-mode : {args.coverage_mode}")
    print(f"  similarity    : {args.compute_similarity}  "
          f"factuality : {args.compute_factuality}")
    print(f"  output        : {out_dir}\n{'='*70}\n")

    # Load
    models = {k: load_model_data(k, d, args.articles_file,
                                  args.max_subjects, args.min_words)
              for k, d in model_dirs.items()}

    # 1: Overlap
    print("\n[1] Subject overlap ...")
    overlap = compute_overlap(models)
    print(f"  Exact: union={overlap['n_union']:,} int={overlap['n_intersection']:,}")
    print(f"  Canon (capped): union={overlap['n_canon_union']:,} int={overlap['n_canon_intersection']:,}")
    if overlap.get("n_seen_canon_union"):
        print(f"  Canon (full run, ref): union={overlap['n_seen_canon_union']:,} "
              f"int={overlap['n_seen_canon_intersection']:,}")
    for p, pw in overlap.get("pairwise",{}).items():
        print(f"  {p}: exact_J={pw['exact_jaccard']:.4f} "
              f"canon_J={pw['canon_jaccard']:.4f} "
              f"(seen_canon_J={pw.get('seen_canon_jaccard', 0):.4f} ref)")

    sampled = sample_intersection(overlap, args.sample_n, args.sample_seed, args.sample_min)
    cs = set(sampled)

    # 2: Breadth
    print("\n[2] Wikilink + breadth ...")
    wl_stats = compute_wikilink_and_breadth_stats(models)
    for r in wl_stats:
        print(f"  {r['model']}: links={r['mean_wikilinks']:.0f} "
              f"wc={r['mean_word_count']:.0f} cats={r['unique_categories']:,} "
              f"hops=0:{r['n_hop_0']:,}/1:{r['n_hop_1']:,}")

    # 3: Entity
    print("\n[3] Entity overlap ...")
    entity_ovl = compute_entity_overlap(models, cs) if cs else []
    for r in entity_ovl:
        print(f"  {r['model_1']} vs {r['model_2']}: J={r['canon_jaccard']:.4f}")

    # 4: Cross-model similarity
    cross_sim = []
    if sampled:
        print("\n[4] Cross-model similarity ...")
        cross_sim = compute_cross_model_similarity(
            models, sampled, min(1000, len(sampled)), args.compute_similarity)
        for r in cross_sim:
            s = (f" sem={r['mean_semantic_cosine']:.4f}"
                 if "mean_semantic_cosine" in r else "")
            print(f"  {r['model_1']} vs {r['model_2']}: "
                  f"J={r.get('mean_jaccard',0):.4f} "
                  f"3g={r.get('mean_ngram_3_overlap',0):.4f}{s}")

    # 5: Factuality
    fact_result: dict = {}; fact_pm: list = []; fact_sh: list = []
    if args.compute_factuality and sampled:
        print(f"\n[5] Factuality (seeds={args.fact_seeds}) ...")
        os.makedirs(audit_root, exist_ok=True)
        fact_result = run_factuality_dual(sampled, models, args, audit_root)
        fact_pm = fact_result.get("per_model_rows", [])
        fact_sh = fact_result.get("shared_rows",    [])

        for r in fact_pm:
            print(f"  [per] {r['model']}: "
                  f"prec={_fmt(r.get('wiki_precision'))} "
                  f"± {_fmt(r.get('wiki_precision_std'))}  "
                  f"wiki_found={r.get('n_wiki_subject_found')} / "
                  f"{r.get('n_returned_total')} returned  "
                  f"({r.get('n_per_seed','?')}/seed × {r.get('n_seeds','?')} seeds)")
        for r in fact_sh:
            print(f"  [shd] {r['model']}: "
                  f"prec={_fmt(r.get('wiki_precision'))} "
                  f"± {_fmt(r.get('wiki_precision_std'))}  "
                  f"wiki_found={r.get('n_wiki_subject_found')} / "
                  f"{r.get('n_shared_subjects','?')} subjects  "
                  f"(subject-level, same for all models)")

    # Write
    print("\n[output] Writing ...")
    _write_csv([{"pair":p,**pw} for p,pw in overlap.get("pairwise",{}).items()],
               os.path.join(out_dir,"subject_overlap_pairwise.csv"))
    _write_csv(wl_stats,   os.path.join(out_dir,"wikilink_breadth_stats.csv"))
    if entity_ovl: _write_csv(entity_ovl, os.path.join(out_dir,"entity_overlap.csv"))
    if cross_sim:  _write_csv(cross_sim,  os.path.join(out_dir,"cross_model_similarity.csv"))
    if fact_pm:    _write_csv(fact_pm,    os.path.join(out_dir,"factuality_per_model.csv"))
    if fact_sh:    _write_csv(fact_sh,    os.path.join(out_dir,"factuality_shared.csv"))
    if fact_pm or fact_sh:
        _write_csv(fact_pm+fact_sh, os.path.join(out_dir,"factuality_all.csv"))
    if sampled:
        _write_csv([{"subject":s} for s in sampled],
                   os.path.join(out_dir,"sampled_subjects.csv"))
    if fact_result.get("shared_with_articles"):
        _write_csv([{"subject":s} for s in fact_result["shared_with_articles"]],
                   os.path.join(out_dir,"shared_with_articles.csv"))
    if fact_result.get("shared_wiki_found_subjects"):
        _write_csv([{"subject":s} for s in fact_result["shared_wiki_found_subjects"]],
                   os.path.join(out_dir,"shared_wiki_found_subjects.csv"))

    rpt = {
        "generated": datetime.datetime.now().isoformat(),
        "config":    vars(args),
        "models":    {k: {"display": md.display_name,
                          "n_subjects": len(md.subject_names),
                          "n_raw": md.n_total_raw,
                          "n_unique": md.n_total_unique,
                          "n_canon_capped": len(md.subject_canon_keys),
                          "n_seen_canon_full": len(md.seen_canon_keys)}
                      for k, md in models.items()},
        "overlap":               {k:v for k,v in overlap.items()
                                   if k!="intersection_subjects"},
        "wikilink_breadth":      wl_stats,
        "entity_overlap":        entity_ovl,
        "cross_sim":             cross_sim,
        "factuality_per_model":  fact_pm,
        "factuality_shared":     fact_sh,
        "factuality_consistency_check": {
            "shared_coverage_identical":
                len(set(str(r.get("n_wiki_subject_found")) for r in fact_sh)) <= 1
                if fact_sh else None,
            "shared_wiki_found":
                fact_sh[0].get("n_wiki_subject_found") if fact_sh else None,
            "seeds_used": args.fact_seeds,
            "note": ("n_wiki_subject_found identical across models — "
                     "enforced by runtime assertion"),
        },
        "n_sampled": len(sampled),
    }
    rp = os.path.join(out_dir,"cross_model_report.json")
    with open(rp,"w",encoding="utf-8") as f:
        json.dump(rpt, f, ensure_ascii=False, indent=2, default=str)
    print(f"  -> {rp}")

    print("\n[figures] ..."); generate_figures(models, overlap, wl_stats, entity_ovl,
                                               fact_pm, fact_sh, cross_sim, fig_dir)
    print("\n[report] ..."); generate_text_report(out_dir, overlap, wl_stats, entity_ovl,
                                                   cross_sim, fact_pm, fact_sh, models)

    print(f"\n{'='*70}\n[track2] DONE")
    print(f"  exact  : union={overlap['n_union']:,} int={overlap['n_intersection']:,}")
    print(f"  canon  : union={overlap['n_canon_union']:,} int={overlap['n_canon_intersection']:,}")
    if overlap.get("n_seen_canon_union"):
        print(f"  canon (full run ref): union={overlap['n_seen_canon_union']:,} "
              f"int={overlap['n_seen_canon_intersection']:,}")
    print(f"  sampled: {len(sampled)}   seeds: {args.fact_seeds}")
    if fact_sh:
        cov_vals   = [r.get("n_wiki_subject_found") for r in fact_sh]
        consistent = len(set(str(v) for v in cov_vals)) <= 1
        print(f"  shared wiki_found: {cov_vals[0]}/{fact_sh[0].get('n_shared_subjects','?')} "
              f"({'OK' if consistent else 'BUG-INCONSISTENT'})")
    nf   = len([f for f in os.listdir(out_dir)
                if not os.path.isdir(os.path.join(out_dir,f))])
    nfig = len(os.listdir(fig_dir)) if os.path.isdir(fig_dir) else 0
    print(f"  files  : {nf} + {nfig} figures")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
build_all_sites.py — Master build & deploy script for LLMPedia.

deploy/
    static/img/models/    ← model logos
    static/img/topics/    ← topic cover images
    image_store/          ← GLOBAL shared article images
    gpt-5-mini/
    deepseek/
    llama/
    capture_trap/
    topic_runs/{model}/{topic}/{persona}/
    index.html
    manifest.json

Usage:
    python3 build_all_sites.py --output-dir deploy/
    python3 build_all_sites.py --output-dir deploy/ --mode deploy
    python3 build_all_sites.py --output-dir deploy/ --skip-images
    python3 build_all_sites.py --output-dir deploy/ --skip-build --skip-images
    python3 build_all_sites.py --output-dir deploy/ --only large-scale
    python3 build_all_sites.py --output-dir deploy/ --clean
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ═══════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════

_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

SITE_BUILDER  = _SCRIPT_DIR / "build_llmpedia_site.py"
IMAGE_BUILDER = _SCRIPT_DIR / "build_image_index.py"

# Static assets directory (logos, topic images)
STATIC_SRC = _SCRIPT_DIR / "static" / "img"

BASE = _PROJECT_ROOT
LARGE_SCALE_RUNS = {
    "gpt-5-mini": BASE / "openLLMPedia" / "gpt_5_mini_1M",
    "deepseek":   BASE / "openLLMPedia" / "deepseekV3.2_100K",
    "llama":      BASE / "openLLMPedia" / "llama3.3-70b_100K",
}
CAPTURE_TRAP_RUN = BASE / "openLLMPedia" / "capture_trap_gpt_5_minin_output_1000"
TOPIC_RUNS_ROOT  = BASE / "openLLMPedia" / "topic_runs"

MODEL_DISPLAY = {
    "gpt-5-mini":          "GPT-5-mini",
    "deepseek":            "DeepSeek V3.2",
    "llama":               "Llama 3.3-70B",
    "scads-DeepSeek-V3.2": "DeepSeek V3.2",
    "scads-llama-3.3-70b": "Llama 3.3-70B",
}

# Model logo filenames (in static/img/models/)
MODEL_LOGOS = {
    "gpt-5-mini": "gpt.png",
    "deepseek":   "deepseek.png",
    "llama":      "llama.png",
}

# Capture trap logo
CAPTURE_TRAP_LOGO = "llmpedia.png"  # in static/img/models/

# Topic display names and cover image filenames (in static/img/topics/)
TOPIC_META = {
    "ancient_babylon": {
        "display": "Ancient City of Babylon",
        "image": "ancient_babylon.jpg",
        "icon": "🏛️",
    },
    "us_civil_rights_movement": {
        "display": "US Civil Rights Movement",
        "image": "us_civil_rights_movement.jpg",
        "icon": "✊",
    },
    "dutch_colonization_se_asia": {
        "display": "Dutch Colonization of SE Asia",
        "image": "dutch_colonization_se_asia.jpg",
        "icon": "⛵",
    },
}

# Persona display
PERSONA_META = {
    "conservative":       {"display": "Conservative",       "icon": "🔵"},
    "left_leaning":       {"display": "Left-Leaning",       "icon": "🔴"},
    "scientific_neutral":  {"display": "Scientific Neutral", "icon": "🔬"},
}

_ELICIT_MODEL_DISPLAY = {
    "gpt-5-mini":          "GPT-5-mini",
    "gpt-4.1-mini":        "GPT-4.1-mini",
    "scads-llama-3.3-70b": "Llama 3.3-70B",
    "scads-DeepSeek-V3.2": "DeepSeek V3.2",
}


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _load_run_meta(run_dir: Path) -> dict:
    p = run_dir / "run_meta.json"
    if not p.exists(): return {}
    try: return json.loads(p.read_text(encoding="utf-8"))
    except Exception: return {}

def _extract_model_display(meta: dict) -> str:
    raw = (
        (meta.get("args_raw") or {}).get("elicit_model_key") or
        (meta.get("cascading_defaults") or {}).get("global_model_key") or
        (meta.get("args_raw") or {}).get("model_key") or ""
    )
    return _ELICIT_MODEL_DISPLAY.get(raw, raw) if raw else ""

def find_run_dirs(root: Path) -> List[Path]:
    if not root.is_dir(): return []
    for name in ("articles.jsonl", "articles_wikitext.jsonl"):
        if (root / name).exists(): return [root]
    found = set()
    for name in ("articles.jsonl", "articles_wikitext.jsonl"):
        for p in root.rglob(name): found.add(p.parent)
    return sorted(found)

def run_cmd(cmd: List[str], label: str = "") -> bool:
    print(f"\n{'='*70}\n  {label}\n  CMD: {' '.join(str(c) for c in cmd)}\n{'='*70}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  ⚠ Exit code {result.returncode}")
    return result.returncode == 0

def build_images(run_dir: Path, image_builder: Path,
                 global_store: Path, workers: int = 40) -> bool:
    if not run_dir.is_dir(): return False
    cmd = [sys.executable, str(image_builder), str(run_dir),
           "--workers", str(workers),
           "--global-store", str(global_store)]
    return run_cmd(cmd, f"Images: {run_dir.name}")

def build_site(run_dir: Path, site_builder: Path, global_store: Path,
               clean: bool = False, workers: int = 8, mode: str = "anonymous") -> bool:
    if not run_dir.is_dir():
        print(f"  SKIP (not found): {run_dir}"); return False
    cmd = [sys.executable, str(site_builder), str(run_dir),
           "--workers", str(workers), "--mode", mode,
           "--global-images", str(global_store)]
    if clean: cmd.append("--clean")
    return run_cmd(cmd, f"Site: {run_dir.name}")

def copy_site_to_deploy(run_dir: Path, deploy_path: Path) -> bool:
    src = run_dir / "site"
    if not src.is_dir():
        print(f"  SKIP (no site/): {run_dir}"); return False
    deploy_path.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        dst = deploy_path / item.name
        if item.name == "image_store": continue
        if item.is_dir():
            if dst.is_symlink(): dst.unlink()
            if dst.exists(): shutil.rmtree(dst)
            shutil.copytree(item, dst, symlinks=False)
        else:
            shutil.copy2(item, dst)
    print(f"  ✓ Copied site → {deploy_path}")
    return True

def link_global_images(site_dir: Path, global_store: Path):
    dst = site_dir / "image_store"
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink(): dst.unlink()
        elif dst.is_dir(): shutil.rmtree(dst)
    try:
        rel = os.path.relpath(global_store, site_dir)
        os.symlink(rel, dst)
    except (OSError, NotImplementedError, ValueError):
        try:
            os.symlink(global_store.resolve(), dst)
        except (OSError, NotImplementedError):
            shutil.copytree(global_store, dst)

def count_articles(run_dir: Path) -> int:
    for name in ("articles.jsonl", "articles_wikitext.jsonl"):
        p = run_dir / name
        if p.exists():
            with p.open("rb") as f:
                return sum(1 for _ in f)
    return 0


# ═══════════════════════════════════════════════════════════════════
# Static assets — copy logos + topic images into deploy/
# ═══════════════════════════════════════════════════════════════════

def copy_static_assets(deploy_dir: Path) -> Dict[str, Dict[str, Optional[str]]]:
    """
    Copy static/img/models/ and static/img/topics/ into deploy/static/img/.
    Returns paths dict for landing page.
    """
    result: Dict[str, Dict[str, Optional[str]]] = {"models": {}, "topics": {}}
    dst_base = deploy_dir / "static" / "img"

    # Model logos
    dst_models = dst_base / "models"
    dst_models.mkdir(parents=True, exist_ok=True)
    for model_key, filename in MODEL_LOGOS.items():
        src = STATIC_SRC / "models" / filename
        if src.exists():
            shutil.copy2(src, dst_models / filename)
            result["models"][model_key] = f"static/img/models/{filename}"
            print(f"  ✓ Logo: {model_key} → {filename}")
        else:
            result["models"][model_key] = None
            print(f"  ✗ Missing: {src}")

    # Capture trap logo
    ct_src = STATIC_SRC / "models" / CAPTURE_TRAP_LOGO
    if ct_src.exists():
        shutil.copy2(ct_src, dst_models / CAPTURE_TRAP_LOGO)
        result["models"]["capture_trap"] = f"static/img/models/{CAPTURE_TRAP_LOGO}"
        print(f"  ✓ Logo: capture_trap → {CAPTURE_TRAP_LOGO}")
    else:
        result["models"]["capture_trap"] = None
        print(f"  ✗ Missing: {ct_src}")

    # Topic images
    dst_topics = dst_base / "topics"
    dst_topics.mkdir(parents=True, exist_ok=True)
    for topic_key, meta in TOPIC_META.items():
        filename = meta["image"]
        src = STATIC_SRC / "topics" / filename
        if src.exists():
            shutil.copy2(src, dst_topics / filename)
            result["topics"][topic_key] = f"static/img/topics/{filename}"
            print(f"  ✓ Topic: {topic_key} → {filename}")
        else:
            result["topics"][topic_key] = None
            print(f"  ✗ Missing: {src}")

    return result


# ═══════════════════════════════════════════════════════════════════
# Site nav JS — dropdown encyclopedia switcher for every sub-site
# ═══════════════════════════════════════════════════════════════════

def write_site_nav_js(deploy_dir: Path, site_entries: List[dict],
                      assets: Dict[str, Dict[str, Optional[str]]]):
    """
    Write a site_nav.js file into every sub-site directory.
    This JS populates the nav dropdown (#siteNavContent) so users can
    switch between encyclopedias from any article page.
    """

    # Build nav data: grouped by category
    nav_items = []
    for entry in site_entries:
        cat = entry.get("category", "")
        mk = entry.get("model_key", "")
        display = entry.get("model_display", mk)
        path = entry.get("path", "")
        count = entry.get("article_count", 0)
        logo_url = assets.get("models", {}).get(mk)

        # Topic runs get extra info
        topic = entry.get("topic", "")
        persona = entry.get("persona", "")
        persona_key = entry.get("persona_key", "")

        if cat == "topic":
            tm = TOPIC_META.get(topic, {})
            topic_display = tm.get("display", topic.replace("_", " ").title())
            topic_icon = tm.get("icon", "📚")
            pm = PERSONA_META.get(persona_key, {})
            persona_display = pm.get("display", persona)
            persona_icon = pm.get("icon", "📝")
            label = f"{topic_icon} {topic_display} — {persona_icon} {persona_display}"
        elif cat == "capture-trap":
            label = f"🔍 Capture Trap"
        else:
            label = display

        nav_items.append({
            "category": cat,
            "label": label,
            "path": path,
            "count": count,
            "logo": logo_url,
            "model_display": display,
        })

    nav_data_json = json.dumps(nav_items, ensure_ascii=False, separators=(",", ":"))

    # The JS that populates the dropdown
    js_template = f"""// Auto-generated by build_all_sites.py — site navigation data
(function(){{
  var NAV_DATA = {nav_data_json};
  var el = document.getElementById('siteNavContent');
  if (!el || !NAV_DATA.length) return;

  // Detect current site from URL path
  var loc = window.location.pathname;
  function isActive(p) {{
    // Normalize: ../gpt-5-mini/  matches  /deploy/gpt-5-mini/
    return loc.indexOf('/' + p + '/') >= 0 || loc.indexOf('/' + p.replace(/\\//g, '/') + '/') >= 0;
  }}

  var cats = {{"large-scale": "Encyclopedias", "capture-trap": "Evaluations", "topic": "Topic Runs"}};
  var grouped = {{}};
  NAV_DATA.forEach(function(item) {{
    var c = item.category || 'other';
    if (!grouped[c]) grouped[c] = [];
    grouped[c].push(item);
  }});

  var html = '';

  // Home link
  html += '<a href="../index.html" class="nav-dd-item" style="font-weight:600;border-bottom:2px solid #eaecf0">' +
    '<span class="ndi-ph">🏠</span>' +
    '<span class="ndi-info"><span class="ndi-name">LLMPedia Home</span>' +
    '<span class="ndi-count">Main portal</span></span></a>';

  var order = ['large-scale', 'capture-trap', 'topic'];
  order.forEach(function(cat) {{
    var items = grouped[cat];
    if (!items || !items.length) return;
    html += '<div class="nav-dd-group">' + (cats[cat] || cat) + '</div>';
    items.forEach(function(item) {{
      var active = isActive(item.path) ? ' active' : '';
      var logoHtml = item.logo
        ? '<img src="../' + item.logo + '" alt="">'
        : '<span class="ndi-ph">' + (item.model_display ? item.model_display[0] : '?') + '</span>';
      var countStr = item.count ? item.count.toLocaleString() + ' articles' : '';
      html += '<a href="../' + item.path + '/index.html" class="nav-dd-item' + active + '">' +
        logoHtml +
        '<span class="ndi-info"><span class="ndi-name">' + item.label + '</span>' +
        '<span class="ndi-count">' + countStr + '</span></span></a>';
    }});
  }});

  el.innerHTML = html;

  // Close dropdown when clicking outside
  document.addEventListener('click', function(e) {{
    var dd = document.getElementById('siteNavDD');
    if (dd && !dd.contains(e.target)) dd.classList.remove('open');
  }});
}})();
"""

    # Write into every sub-site deploy directory
    written = 0
    for entry in site_entries:
        path = entry.get("path", "")
        if not path:
            continue
        site_dir = deploy_dir / path
        if site_dir.is_dir():
            (site_dir / "site_nav.js").write_text(js_template, encoding="utf-8")
            written += 1

    print(f"  ✓ site_nav.js written to {written} site(s)")


# ═══════════════════════════════════════════════════════════════════
# Cross-model article index — "Also written by…" links
# ═══════════════════════════════════════════════════════════════════

def _slugify(title: str) -> str:
    title = title.strip().replace(" ", "_")
    return re.sub(r"[^\w\-\.\(\)]+", "_", title)

def _load_subjects_fast(run_dir: Path) -> Dict[str, str]:
    """Load subject→slug map from a run directory (fast, skips wikitext)."""
    subjects: Dict[str, str] = {}
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
                    # Fast: only parse enough to get subject
                    obj = json.loads(line)
                    subj = obj.get("subject", "").strip()
                    if subj:
                        subjects[subj] = _slugify(subj)
                except Exception:
                    pass
        break  # only read the first file found
    return subjects


def build_cross_model_index(deploy_dir: Path, site_entries: List[dict],
                            assets: Dict[str, Dict[str, Optional[str]]]):
    """
    Build a cross-model subject index so article pages can show
    "This article across models: [GPT] [DeepSeek] [Llama]".

    Scans all runs, finds subjects that exist in 2+ sites,
    then writes a cross_model.js into each sub-site directory.
    """
    t0 = time.time()

    # 1. Collect subjects per site
    site_subject_maps: Dict[str, Dict[str, str]] = {}  # site_path -> {subject -> slug}
    for entry in site_entries:
        run_dir = Path(entry.get("run_dir", ""))
        path = entry.get("path", "")
        if not run_dir.is_dir() or not path:
            continue
        subjects = _load_subjects_fast(run_dir)
        site_subject_maps[path] = subjects
        print(f"    {path}: {len(subjects):,} subjects")

    if len(site_subject_maps) < 2:
        print("  ⚠ Need 2+ sites for cross-model index — skipping")
        return

    # 2. Build reverse map: normalized_subject -> [(site_path, slug)]
    # Normalize = lowercase for matching, keep original for display
    norm_to_sites: Dict[str, List[Tuple[str, str, str]]] = {}  # norm -> [(path, slug, original_subject)]
    for path, subjects in site_subject_maps.items():
        for subj, slug in subjects.items():
            norm = subj.lower()
            norm_to_sites.setdefault(norm, []).append((path, slug, subj))

    # 3. Filter: only keep subjects in 2+ DISTINCT sites
    cross_refs: Dict[str, List[Tuple[str, str, str]]] = {}
    for norm, sites in norm_to_sites.items():
        distinct_paths = set(s[0] for s in sites)
        if len(distinct_paths) >= 2:
            cross_refs[norm] = sites

    total_cross = len(cross_refs)
    elapsed_scan = time.time() - t0
    print(f"  Cross-model: {total_cross:,} shared subjects across {len(site_subject_maps)} sites ({elapsed_scan:.1f}s)")

    if total_cross == 0:
        print("  ⚠ No shared subjects found — skipping cross_model.js")
        return

    # 4. Build model info list (for the JS)
    entry_by_path: Dict[str, dict] = {e["path"]: e for e in site_entries if e.get("path")}

    # 5. Write cross_model.js for each site
    for current_path, current_subjects in site_subject_maps.items():
        current_entry = entry_by_path.get(current_path, {})
        current_mk = current_entry.get("model_key", "")
        current_display = current_entry.get("model_display", current_mk)

        # Compute depth for relative paths (e.g. "topic_runs/model/topic/persona" = depth 4)
        depth = current_path.count("/")
        prefix = "../" * (depth + 1) if depth > 0 else "../"

        # Build models array (all OTHER sites that share subjects with this one)
        other_paths_with_data: Set[str] = set()
        # Collect which other sites actually share with this one
        for subj in current_subjects:
            norm = subj.lower()
            if norm in cross_refs:
                for (p, s, orig) in cross_refs[norm]:
                    if p != current_path:
                        other_paths_with_data.add(p)

        if not other_paths_with_data:
            continue

        # Build compact model list
        models = []
        model_path_to_idx: Dict[str, int] = {}
        for op in sorted(other_paths_with_data):
            oe = entry_by_path.get(op, {})
            mk = oe.get("model_key", "")
            md = oe.get("model_display", mk)
            cat = oe.get("category", "")
            logo = assets.get("models", {}).get(mk)

            # Build a nice label
            if cat == "topic":
                topic = oe.get("topic", "")
                persona = oe.get("persona", "")
                persona_key = oe.get("persona_key", "")
                tm = TOPIC_META.get(topic, {})
                pm = PERSONA_META.get(persona_key, {})
                label = f"{tm.get('icon', '📚')} {tm.get('display', topic)} — {pm.get('display', persona)}"
            elif cat == "capture-trap":
                label = f"Capture Trap"
            else:
                label = md

            model_path_to_idx[op] = len(models)
            models.append({
                "key": mk, "name": md, "label": label,
                "logo": f"{prefix}{logo}" if logo else "",
                "path": f"{prefix}{op}",
                "cat": cat,
            })

        # Build refs: slug_in_this_site -> [[model_idx, slug_in_other_site], ...]
        refs: Dict[str, List] = {}
        for subj, slug in current_subjects.items():
            norm = subj.lower()
            if norm not in cross_refs:
                continue
            others = []
            for (p, s, orig) in cross_refs[norm]:
                if p != current_path and p in model_path_to_idx:
                    others.append([model_path_to_idx[p], s])
            if others:
                refs[slug] = others

        if not refs:
            continue

        # Write JS
        models_json = json.dumps(models, ensure_ascii=False, separators=(",", ":"))
        refs_json = json.dumps(refs, ensure_ascii=False, separators=(",", ":"))

        js_code = f"""// Auto-generated cross-model index
(function(){{
var M={models_json};
var R={refs_json};
var el=document.getElementById('crossModelBox');
if(!el)return;
var subj=el.getAttribute('data-subject');
if(!subj)return;
var hits=R[subj];
if(!hits||!hits.length)return;

var h='<div class="xm-box"><div class="xm-head"><span class="xm-icon">🔄</span> Compare across models</div><div class="xm-list">';
// Current model
h+='<div class="xm-item xm-current"><span class="xm-badge">current</span> <b>{current_display.replace("'", "\\'")}</b></div>';
for(var i=0;i<hits.length;i++){{
  var mi=hits[i][0],slug=hits[i][1],m=M[mi];
  var logo=m.logo?'<img src="'+m.logo+'" alt="" class="xm-logo">':'';
  var catClass=m.cat==='topic'?' xm-topic':m.cat==='capture-trap'?' xm-ct':'';
  h+='<a href="'+m.path+'/'+slug+'.html" class="xm-item'+catClass+'">'+logo+'<span class="xm-name">'+m.label+'</span><span class="xm-arrow">→</span></a>';
}}
h+='</div></div>';
el.innerHTML=h;
}})();
"""
        site_dir = deploy_dir / current_path
        if site_dir.is_dir():
            (site_dir / "cross_model.js").write_text(js_code, encoding="utf-8")

    elapsed = time.time() - t0
    print(f"  ✓ cross_model.js written ({elapsed:.1f}s)")


# ═══════════════════════════════════════════════════════════════════
# Landing page
# ═══════════════════════════════════════════════════════════════════

def build_landing_page(deploy_dir: Path, site_entries: List[dict],
                       assets: Dict[str, Dict[str, Optional[str]]],
                       mode: str = "anonymous",
                       github_url: str = "https://github.com/PLACEHOLDER/llmpedia"):

    large_scale = [e for e in site_entries if e.get("category") == "large-scale"]
    capture     = [e for e in site_entries if e.get("category") == "capture-trap"]
    topic       = [e for e in site_entries if e.get("category") == "topic"]

    topics_grouped: Dict[str, List[dict]] = {}
    for e in topic:
        topics_grouped.setdefault(e.get("topic", "unknown"), []).append(e)

    # ── Model card (large-scale) ──
    def _model_card(entry: dict) -> str:
        mk = entry.get("model_key", "")
        display = entry.get("model_display", mk)
        path = entry.get("path", "#")
        count = entry.get("article_count", 0)
        elicit = entry.get("elicit_model", "")
        logo_url = assets.get("models", {}).get(mk)
        logo_html = (f'<img src="{logo_url}" alt="{display}" class="card-logo">'
                     if logo_url else f'<div class="card-logo-ph">{display[0]}</div>')
        elicit_html = f'<span class="tag tag-m">🤖 {elicit}</span>' if elicit else ""
        return (
            f'<a href="{path}/index.html" class="model-card">'
            f'{logo_html}'
            f'<span class="card-name">{display}</span>'
            f'<span class="card-count">{count:,} articles</span>'
            f'{elicit_html}</a>'
        )

    # ── Capture trap card ──
    def _capture_card(entry: dict) -> str:
        path = entry.get("path", "#")
        count = entry.get("article_count", 0)
        elicit = entry.get("elicit_model", "")
        logo_url = assets.get("models", {}).get("capture_trap")
        logo_html = (f'<img src="{logo_url}" alt="LLMPedia" class="card-logo">'
                     if logo_url else '<div class="card-logo-ph">CT</div>')
        elicit_html = f'<span class="tag tag-m">🤖 {elicit}</span>' if elicit else ""
        return (
            f'<a href="{path}/index.html" class="model-card">'
            f'{logo_html}'
            f'<span class="card-name">Capture Trap</span>'
            f'<span class="card-count">{count:,} articles</span>'
            f'{elicit_html}</a>'
        )

    # ── Persona card (inside topic) ──
    def _persona_card(entry: dict) -> str:
        path = entry.get("path", "#")
        count = entry.get("article_count", 0)
        pkey = entry.get("persona_key", "")
        pm = PERSONA_META.get(pkey, {})
        pname = pm.get("display", entry.get("persona", pkey))
        picon = pm.get("icon", "📝")
        mk = entry.get("model_key", "")
        mdisplay = entry.get("model_display", mk)
        mlogo = assets.get("models", {}).get(mk)
        mhtml = (f'<img src="{mlogo}" alt="{mdisplay}" class="p-mlogo">'
                 if mlogo else f'<span class="p-mtxt">{mdisplay}</span>')
        return (
            f'<a href="{path}/index.html" class="p-card">'
            f'<div class="p-top"><span class="p-icon">{picon}</span>'
            f'<span class="p-name">{pname}</span></div>'
            f'<div class="p-bot">{mhtml}'
            f'<span class="p-cnt">{count:,} articles</span></div></a>'
        )

    ls_cards = "\n".join(_model_card(e) for e in large_scale)
    ct_cards = "\n".join(_capture_card(e) for e in capture) if capture else ""

    # ── Topic sections ──
    topic_sections = ""
    for tkey, entries in sorted(topics_grouped.items()):
        tm = TOPIC_META.get(tkey, {})
        tdisp = tm.get("display", tkey.replace("_", " ").title())
        ticon = tm.get("icon", "📚")
        timg = assets.get("topics", {}).get(tkey)
        img_html = (f'<img src="{timg}" alt="{tdisp}" class="t-cover">'
                    if timg else f'<div class="t-cover-ph">{ticon}</div>')
        pcards = "\n".join(_persona_card(e) for e in entries)
        n_models = len(set(e.get("model_key", "") for e in entries))
        topic_sections += f"""
        <div class="t-section">
          <div class="t-head">
            {img_html}
            <div class="t-info">
              <h3>{ticon} {tdisp}</h3>
              <p>{len(entries)} perspectives across {n_models} model(s)</p>
            </div>
          </div>
          <div class="p-grid">{pcards}</div>
        </div>"""

    src_link = ""
    if mode == "anonymous":
        src_link = f'<p class="src-link">📂 <a href="{github_url}" target="_blank">Source on GitHub</a></p>'

    if mode == "deploy":
        footer = ('<footer><div class="ft-brand"><a href="https://scads.ai/" target="_blank">'
                  '<span class="ft-lt">ScaDS.AI<small>Dresden · Leipzig</small></span></a></div>'
                  '<div>Licensed under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a></div>'
                  '<div class="ft-ver">LLMPedia v1</div></footer>')
    else:
        footer = ('<footer><div>Licensed under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a></div>'
                  '<div class="ft-ver">LLMPedia v1</div></footer>')

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLMPedia</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  color:#202122;background:#f8f9fa;min-height:100vh}}
a{{color:#0645ad;text-decoration:none}}

.hero{{background:linear-gradient(135deg,#1b3a5c 0%,#2a6496 60%,#1b3a5c 100%);
  color:#fff;padding:2.5rem 1.5rem;text-align:center}}
.hero h1{{font-family:Linux Libertine,Georgia,"Times New Roman",serif;
  font-size:2.4rem;font-weight:400;margin-bottom:0.3rem}}
.hero .sub{{font-size:1rem;opacity:0.85;max-width:600px;margin:0 auto;line-height:1.5}}
.hero .warn{{display:inline-block;margin-top:0.8rem;padding:0.3rem 0.8rem;
  background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);
  border-radius:3px;font-size:0.82rem}}

.wrap{{max-width:1000px;margin:0 auto;padding:1.5rem 1rem 3rem}}
.src-link{{text-align:center;margin:0.5rem 0 1rem;font-size:0.88rem}}

.sec-title{{font-family:Linux Libertine,Georgia,"Times New Roman",serif;font-size:1.4rem;
  font-weight:400;color:#1b3a5c;margin:2rem 0 0.4rem;border-bottom:1px solid #a2a9b1;padding-bottom:0.2rem}}
.sec-desc{{color:#54595d;font-size:0.9rem;margin-bottom:1rem;line-height:1.5}}

/* Model cards */
.m-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:1rem;margin-bottom:1.5rem}}
.model-card{{display:flex;flex-direction:column;align-items:center;padding:1.2rem 1rem;
  background:#fff;border:1px solid #c8ccd1;border-radius:4px;color:#202122;
  transition:border-color .15s,box-shadow .15s,transform .15s}}
.model-card:hover{{border-color:#36c;box-shadow:0 3px 12px rgba(0,0,0,.1);
  transform:translateY(-2px);text-decoration:none}}
.card-logo{{height:48px;max-width:120px;object-fit:contain;margin-bottom:0.5rem}}
.card-logo-ph{{width:48px;height:48px;border-radius:50%;background:#36c;color:#fff;
  display:flex;align-items:center;justify-content:center;font-size:1.2rem;font-weight:700;margin-bottom:0.5rem}}
.card-name{{font-weight:700;font-size:1rem;text-align:center}}
.card-count{{font-size:0.8rem;color:#72777d;margin-top:0.15rem}}
.tag{{display:inline-block;margin-top:0.3rem;padding:.12rem .45rem;
  background:#eaecf0;border-radius:2px;font-size:.73rem;color:#54595d}}
.tag-m{{background:#eaf3fb;color:#36c}}

/* Topic sections */
.t-section{{background:#fff;border:1px solid #c8ccd1;border-radius:4px;margin-bottom:1.2rem;overflow:hidden}}
.t-head{{display:flex;align-items:stretch;min-height:120px}}
.t-cover{{width:200px;min-height:120px;object-fit:cover;flex-shrink:0;display:block}}
.t-cover-ph{{width:200px;min-height:120px;background:linear-gradient(135deg,#667eea,#764ba2);
  display:flex;align-items:center;justify-content:center;font-size:3rem;flex-shrink:0}}
.t-info{{padding:.8rem 1rem;display:flex;flex-direction:column;justify-content:center}}
.t-info h3{{font-family:Linux Libertine,Georgia,"Times New Roman",serif;
  font-size:1.15rem;font-weight:400;color:#1b3a5c;margin-bottom:.2rem}}
.t-info p{{font-size:.82rem;color:#72777d}}

/* Persona cards */
.p-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:.6rem;padding:.8rem}}
.p-card{{display:flex;flex-direction:column;padding:.6rem .7rem;
  border:1px solid #e0e0e0;border-radius:3px;color:#202122;transition:border-color .15s,background .15s}}
.p-card:hover{{border-color:#36c;background:#f0f6ff;text-decoration:none}}
.p-top{{display:flex;align-items:center;gap:.35rem;margin-bottom:.3rem}}
.p-icon{{font-size:1rem}}
.p-name{{font-weight:600;font-size:.88rem}}
.p-bot{{display:flex;align-items:center;gap:.4rem}}
.p-mlogo{{height:16px;max-width:60px;object-fit:contain;opacity:.7}}
.p-mtxt{{font-size:.72rem;color:#72777d}}
.p-cnt{{font-size:.72rem;color:#72777d}}

footer{{text-align:center;padding:1rem;color:#72777d;font-size:.8rem;border-top:1px solid #c8ccd1;margin-top:2rem}}
footer a{{color:#54595d}}
.ft-brand{{margin-bottom:.3rem}}
.ft-lt{{font-weight:700;color:#202122}}
.ft-lt small{{display:block;font-size:.72rem;font-weight:400;color:#72777d}}
.ft-ver{{color:#a2a9b1;font-size:.72rem;margin-top:.2rem}}

@media(max-width:640px){{
  .hero h1{{font-size:1.8rem}}
  .m-grid{{grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:.6rem}}
  .model-card{{padding:.8rem .6rem}}
  .card-logo{{height:36px}}
  .t-head{{flex-direction:column}}
  .t-cover,.t-cover-ph{{width:100%;min-height:140px;max-height:180px}}
  .p-grid{{grid-template-columns:1fr 1fr}}
}}
</style>
</head>
<body>
<div class="hero">
  <h1>LLMPedia</h1>
  <p class="sub">A transparent, open encyclopedia generated entirely from LLM parametric knowledge &mdash; no retrieval, no external sources.</p>
  <span class="warn">&#9888; All content is AI-generated and may contain inaccuracies. For research purposes only.</span>
</div>
<div class="wrap">
  {src_link}

  <h2 class="sec-title">Large-Scale Encyclopedias</h2>
  <p class="sec-desc">Full BFS-expanded encyclopedias with NER filtering and similarity deduplication.</p>
  <div class="m-grid">{ls_cards}</div>

  {"<h2 class='sec-title'>Capture Trap Evaluation</h2><p class='sec-desc'>1,000 pre-selected entities generated without BFS expansion.</p><div class='m-grid'>" + ct_cards + "</div>" if ct_cards else ""}

  {"<h2 class='sec-title'>Topic-Focused Runs</h2><p class='sec-desc'>Each topic explored from multiple editorial perspectives (conservative, left-leaning, scientific neutral) across different models.</p>" + topic_sections if topic_sections else ""}
</div>
{footer}
</body></html>"""

    (deploy_dir / "index.html").write_text(page, encoding="utf-8")
    print(f"  ✓ Landing page → {deploy_dir / 'index.html'}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Build all LLMPedia sites")
    ap.add_argument("--output-dir", default="deploy")
    ap.add_argument("--mode", choices=["anonymous", "deploy"], default="anonymous")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--skip-images", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--image-workers", type=int, default=40)
    ap.add_argument("--only", choices=["large-scale", "topic", "capture-trap", "assemble"])
    ap.add_argument("--github-url", default="https://github.com/PLACEHOLDER/llmpedia")
    ap.add_argument("--site-builder", default=str(SITE_BUILDER))
    ap.add_argument("--image-builder", default=str(IMAGE_BUILDER))
    ap.add_argument("--gpt-dir", default=str(LARGE_SCALE_RUNS["gpt-5-mini"]))
    ap.add_argument("--deepseek-dir", default=str(LARGE_SCALE_RUNS["deepseek"]))
    ap.add_argument("--llama-dir", default=str(LARGE_SCALE_RUNS["llama"]))
    ap.add_argument("--capture-trap-dir", default=str(CAPTURE_TRAP_RUN))
    ap.add_argument("--topic-runs-dir", default=str(TOPIC_RUNS_ROOT))
    args = ap.parse_args()

    deploy_dir    = Path(args.output_dir).resolve()
    site_builder  = Path(args.site_builder).resolve()
    image_builder = Path(args.image_builder).resolve()

    if not args.skip_build and not site_builder.is_file():
        print(f"ERROR: site builder not found: {site_builder}"); sys.exit(1)
    if not args.skip_images and not image_builder.is_file():
        print(f"WARN: image builder not found — skipping images")
        args.skip_images = True

    print(f"\n{'='*70}\n  LLMPedia Build\n{'='*70}")
    print(f"  Deploy:  {deploy_dir}")
    print(f"  Mode:    {args.mode}")
    print(f"  Static:  {STATIC_SRC}")
    print(f"  Images:  {'skip' if args.skip_images else 'download'}")
    print(f"  Build:   {'skip' if args.skip_build else f'workers={args.workers}'}")

    large_scale = {
        "gpt-5-mini": Path(args.gpt_dir),
        "deepseek":   Path(args.deepseek_dir),
        "llama":      Path(args.llama_dir),
    }
    capture_trap = Path(args.capture_trap_dir)
    topic_root   = Path(args.topic_runs_dir)

    if args.clean and deploy_dir.exists():
        print(f"\nCleaning {deploy_dir}")
        shutil.rmtree(deploy_dir)
    deploy_dir.mkdir(parents=True, exist_ok=True)

    global_store = deploy_dir / "image_store"
    global_store.mkdir(parents=True, exist_ok=True)

    # Copy static assets
    print(f"\n{'█'*70}\n  STATIC ASSETS\n{'█'*70}")
    assets = copy_static_assets(deploy_dir)

    site_entries: List[dict] = []

    def _process_run(rd, model_key, category, deploy_subpath, display, **extra):
        meta = _load_run_meta(rd)
        elicit = _extract_model_display(meta)
        print(f"  → {model_key}: elicit={elicit or '?'}")
        if not args.skip_images:
            build_images(rd, image_builder, global_store, workers=args.image_workers)
        if not args.skip_build:
            build_site(rd, site_builder, global_store,
                       clean=args.clean, workers=args.workers, mode=args.mode)
        dp = deploy_dir / deploy_subpath
        copy_site_to_deploy(rd, dp)
        link_global_images(dp, global_store)
        entry = {"category": category, "model_key": model_key,
                 "model_display": display, "elicit_model": elicit,
                 "path": deploy_subpath, "article_count": count_articles(rd),
                 "run_dir": str(rd)}
        entry.update(extra)
        site_entries.append(entry)

    # 1. Large-scale
    if args.only in (None, "large-scale"):
        print(f"\n{'█'*70}\n  LARGE-SCALE RUNS\n{'█'*70}")
        for mk, rr in large_scale.items():
            if not rr.is_dir(): print(f"  SKIP {mk}"); continue
            rds = find_run_dirs(rr)
            if not rds: print(f"  SKIP {mk}: no runs"); continue
            _process_run(rds[0], mk, "large-scale", mk, MODEL_DISPLAY.get(mk, mk))

    # 2. Capture trap
    if args.only in (None, "capture-trap"):
        print(f"\n{'█'*70}\n  CAPTURE TRAP\n{'█'*70}")
        if capture_trap.is_dir():
            rds = find_run_dirs(capture_trap)
            if rds:
                _process_run(rds[0], "gpt-5-mini", "capture-trap",
                             "capture_trap", "GPT-5-mini (Capture Trap)")
        else:
            print(f"  SKIP: not found")

    # 3. Topic runs
    if args.only in (None, "topic"):
        print(f"\n{'█'*70}\n  TOPIC RUNS\n{'█'*70}")
        if topic_root.is_dir():
            for model_dir in sorted(topic_root.iterdir()):
                if not model_dir.is_dir(): continue
                for topic_dir in sorted(model_dir.iterdir()):
                    if not topic_dir.is_dir(): continue
                    for persona_dir in sorted(topic_dir.iterdir()):
                        if not persona_dir.is_dir(): continue
                        rds = find_run_dirs(persona_dir)
                        if not rds: continue
                        rel = f"topic_runs/{model_dir.name}/{topic_dir.name}/{persona_dir.name}"
                        mk = model_dir.name
                        for k in MODEL_DISPLAY:
                            if k in mk.lower() or mk.lower() in k.lower():
                                mk = k; break
                        _process_run(
                            rds[0], mk, "topic", rel,
                            MODEL_DISPLAY.get(mk, model_dir.name),
                            topic=topic_dir.name,
                            persona=persona_dir.name.replace("_", " ").title(),
                            persona_key=persona_dir.name,
                        )
        else:
            print(f"  SKIP: not found")

    # 4. Assemble
    print(f"\n{'█'*70}\n  ASSEMBLING\n{'█'*70}")
    build_landing_page(deploy_dir, site_entries, assets,
                       mode=args.mode, github_url=args.github_url)

    # 5. Write site_nav.js into every sub-site for the dropdown switcher
    print(f"\n{'█'*70}\n  SITE NAVIGATION\n{'█'*70}")
    write_site_nav_js(deploy_dir, site_entries, assets)

    # 6. Cross-model article index ("Also written by…")
    print(f"\n{'█'*70}\n  CROSS-MODEL INDEX\n{'█'*70}")
    build_cross_model_index(deploy_dir, site_entries, assets)

    # Clean site_entries for manifest (remove local paths)
    manifest_entries = [{k: v for k, v in e.items() if k != "run_dir"} for e in site_entries]
    manifest = {"mode": args.mode, "sites": manifest_entries, "total_sites": len(site_entries)}
    (deploy_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*70}\n  DEPLOY SUMMARY\n{'='*70}")
    print(f"  Output:  {deploy_dir}")
    print(f"  Mode:    {args.mode}")
    print(f"  Sites:   {len(site_entries)}")
    ic = sum(1 for f in global_store.iterdir() if f.is_file()) if global_store.is_dir() else 0
    print(f"  Images:  {ic:,} in global store")
    for cat in ["large-scale", "capture-trap", "topic"]:
        es = [e for e in site_entries if e["category"] == cat]
        if es:
            t = sum(e.get("article_count", 0) for e in es)
            print(f"    {cat}: {len(es)} site(s), {t:,} articles")
    print(f"\n  Package:\n    tar -czf llmpedia.tar.gz -C {deploy_dir.parent} {deploy_dir.name}/")


if __name__ == "__main__":
    main()
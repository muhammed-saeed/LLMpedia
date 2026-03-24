#!/usr/bin/env python3
"""
build_all_sites.py — Master build & deploy script for LLMpedia.

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
# Paths & Config
# ═══════════════════════════════════════════════════════════════════

_SCRIPT_DIR   = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

SITE_BUILDER  = _SCRIPT_DIR / "build_llmpedia_site.py"
IMAGE_BUILDER = _SCRIPT_DIR / "build_image_index.py"
STATIC_SRC    = _SCRIPT_DIR / "static" / "img"

BASE = _PROJECT_ROOT
LARGE_SCALE_RUNS = {
    "gpt-5-mini": BASE / "openLLMPedia" / "gpt_5_mini_1M",
    "deepseek":   BASE / "openLLMPedia" / "deepseekV3.2_100K",
    "llama":      BASE / "openLLMPedia" / "llama3.3-70b_100K",
}
CAPTURE_TRAP_RUN = BASE / "openLLMPedia" / "capture_trap_gpt_5_minin_output_1000"
TOPIC_RUNS_ROOT  = BASE / "openLLMPedia" / "topic_runs"

MODEL_DISPLAY = {
    "gpt-5-mini": "GPT-5-mini", "deepseek": "DeepSeek V3.2",
    "llama": "Llama 3.3-70B", "scads-DeepSeek-V3.2": "DeepSeek V3.2",
    "scads-llama-3.3-70b": "Llama 3.3-70B",
}
MODEL_LOGOS = {"gpt-5-mini": "gpt.png", "deepseek": "deepseek.png", "llama": "llama.png"}
CAPTURE_TRAP_LOGO = "llmpedia.png"
SITE_LOGO = "llmpedia.png"

TOPIC_META = {
    "ancient_babylon":            {"display": "Ancient City of Babylon",       "image": "ancient_babylon.jpg",            "icon": "🏛️"},
    "us_civil_rights_movement":   {"display": "US Civil Rights Movement",      "image": "us_civil_rights_movement.jpg",   "icon": "✊"},
    "dutch_colonization_se_asia": {"display": "Dutch Colonization of SE Asia", "image": "dutch_colonization_se_asia.jpg", "icon": "⛵"},
}
PERSONA_META = {
    "conservative": {"display": "Conservative", "icon": "🔵"},
    "left_leaning":  {"display": "Left-Leaning", "icon": "🔴"},
    "scientific_neutral": {"display": "Scientific Neutral", "icon": "🔬"},
}
_ELICIT_MODEL_DISPLAY = {
    "gpt-5-mini": "GPT-5-mini", "gpt-4.1-mini": "GPT-4.1-mini",
    "scads-llama-3.3-70b": "Llama 3.3-70B", "scads-DeepSeek-V3.2": "DeepSeek V3.2",
}

VERSION = "v0.1"
DEFAULT_GITHUB = "https://anonymous.4open.science/r/LLMpedia-7128/README.md"


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def _root_prefix(deploy_subpath: str) -> str:
    """Relative path from sub-site to deploy root.
    'gpt-5-mini' → '..'   |  'topic_runs/m/t/p' → '../../../..'"""
    return "/".join([".."] * (deploy_subpath.count("/") + 1))

def _load_run_meta(rd: Path) -> dict:
    p = rd / "run_meta.json"
    if not p.exists(): return {}
    try: return json.loads(p.read_text(encoding="utf-8"))
    except Exception: return {}

def _extract_model_display(meta: dict) -> str:
    raw = ((meta.get("args_raw") or {}).get("elicit_model_key") or
           (meta.get("cascading_defaults") or {}).get("global_model_key") or
           (meta.get("args_raw") or {}).get("model_key") or "")
    return _ELICIT_MODEL_DISPLAY.get(raw, raw) if raw else ""

def find_run_dirs(root: Path) -> List[Path]:
    if not root.is_dir(): return []
    for n in ("articles.jsonl", "articles_wikitext.jsonl"):
        if (root / n).exists(): return [root]
    found = set()
    for n in ("articles.jsonl", "articles_wikitext.jsonl"):
        for p in root.rglob(n): found.add(p.parent)
    return sorted(found)

def run_cmd(cmd, label=""):
    print(f"\n{'='*70}\n  {label}\n  CMD: {' '.join(str(c) for c in cmd)}\n{'='*70}")
    r = subprocess.run(cmd, capture_output=False)
    if r.returncode != 0: print(f"  ⚠ Exit code {r.returncode}")
    return r.returncode == 0

def build_images(rd, image_builder, global_store, workers=40):
    if not rd.is_dir(): return False
    return run_cmd([sys.executable, str(image_builder), str(rd),
                    "--workers", str(workers), "--global-store", str(global_store)],
                   f"Images: {rd.name}")

def build_site(rd, site_builder, global_store, clean=False, workers=8,
               mode="anonymous", root_prefix=".."):
    if not rd.is_dir():
        print(f"  SKIP (not found): {rd}"); return False
    cmd = [sys.executable, str(site_builder), str(rd),
           "--workers", str(workers), "--mode", mode,
           "--global-images", str(global_store),
           "--root-prefix", root_prefix]
    if clean: cmd.append("--clean")
    return run_cmd(cmd, f"Site: {rd.name}")


# ═══════════════════════════════════════════════════════════════════
# IMAGE SYNC — copy from run-local stores → global deploy store
# ═══════════════════════════════════════════════════════════════════

def sync_local_images_to_global(rd: Path, global_store: Path) -> int:
    """Copy images from run's local image_store/ into the global deploy store.
    This is CRITICAL because:
      - Images may have been downloaded in previous runs (before --global-store)
      - The image builder sees 'Already resolved' and downloads nothing
      - So the global store stays empty unless we sync from local stores.
    Follows symlinks so we don't miss anything.
    """
    count = 0
    search_dirs = [
        rd / "image_store",
        rd / "site" / "image_store",
    ]
    # Also check parent dir in case rd is a sub-dir
    if rd.parent != rd:
        search_dirs.append(rd.parent / "image_store")

    for candidate in search_dirs:
        # Resolve symlinks to get the REAL directory
        try:
            resolved = candidate.resolve()
        except (OSError, ValueError):
            continue
        if not resolved.is_dir():
            continue
        # Don't sync from global store back to itself
        try:
            if resolved.samefile(global_store):
                continue
        except (OSError, ValueError):
            pass
        for f in resolved.iterdir():
            if f.is_file() and f.stat().st_size > 500:
                dst = global_store / f.name
                if not dst.exists():
                    shutil.copy2(f, dst)
                    count += 1
    if count:
        print(f"  ✓ Synced {count:,} images → global store")
    elif not any((global_store / f).exists() for f in ["dummy"] if False):
        # Check if global store has anything
        gc = sum(1 for _ in global_store.iterdir()) if global_store.is_dir() else 0
        if gc == 0:
            print(f"  ⚠ No images found to sync (checked: {', '.join(str(c) for c in search_dirs)})")
    return count


def copy_site_to_deploy(rd: Path, deploy_path: Path) -> bool:
    src = rd / "site"
    if not src.is_dir():
        print(f"  SKIP (no site/): {rd}"); return False
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
        try: os.symlink(global_store.resolve(), dst)
        except (OSError, NotImplementedError):
            shutil.copytree(global_store, dst)

def count_articles(rd: Path) -> int:
    for n in ("articles.jsonl", "articles_wikitext.jsonl"):
        p = rd / n
        if p.exists():
            with p.open("rb") as f: return sum(1 for _ in f)
    return 0


# ═══════════════════════════════════════════════════════════════════
# Static assets
# ═══════════════════════════════════════════════════════════════════

def copy_static_assets(deploy_dir):
    result = {"models": {}, "topics": {}}
    dst_models = deploy_dir / "static" / "img" / "models"
    dst_models.mkdir(parents=True, exist_ok=True)

    for mk, fn in MODEL_LOGOS.items():
        src = STATIC_SRC / "models" / fn
        if src.exists():
            shutil.copy2(src, dst_models / fn)
            result["models"][mk] = f"static/img/models/{fn}"
            print(f"  ✓ Logo: {mk} → {fn}")
        else:
            result["models"][mk] = None; print(f"  ✗ Missing: {src}")

    for key, fn in [("capture_trap", CAPTURE_TRAP_LOGO), ("site_logo", SITE_LOGO)]:
        src = STATIC_SRC / "models" / fn
        if src.exists():
            shutil.copy2(src, dst_models / fn)
            result["models"][key] = f"static/img/models/{fn}"
            print(f"  ✓ Logo: {key} → {fn}")

    dst_topics = deploy_dir / "static" / "img" / "topics"
    dst_topics.mkdir(parents=True, exist_ok=True)
    for tk, meta in TOPIC_META.items():
        fn = meta["image"]
        src = STATIC_SRC / "topics" / fn
        if src.exists():
            shutil.copy2(src, dst_topics / fn)
            result["topics"][tk] = f"static/img/topics/{fn}"
            print(f"  ✓ Topic: {tk} → {fn}")
        else:
            result["topics"][tk] = None; print(f"  ✗ Missing: {src}")
    return result


# ═══════════════════════════════════════════════════════════════════
# Site nav JS — depth-aware dropdown
# ═══════════════════════════════════════════════════════════════════

def write_site_nav_js(deploy_dir, site_entries, assets):
    nav_items = []
    for e in site_entries:
        cat, mk = e.get("category",""), e.get("model_key","")
        display, path = e.get("model_display", mk), e.get("path","")
        count = e.get("article_count", 0)
        logo = assets.get("models", {}).get(mk)
        topic, persona_key = e.get("topic",""), e.get("persona_key","")
        if cat == "topic":
            tm, pm = TOPIC_META.get(topic, {}), PERSONA_META.get(persona_key, {})
            label = f"{tm.get('icon','📚')} {tm.get('display', topic.replace('_',' ').title())} — {pm.get('icon','📝')} {pm.get('display', e.get('persona', persona_key))}"
        elif cat == "capture-trap":
            label = "🔍 Capture Trap"
        else:
            label = display
        nav_items.append({"category": cat, "label": label, "path": path,
                          "count": count, "logo": logo, "model_display": display})

    data_json = json.dumps(nav_items, ensure_ascii=False, separators=(",",":"))
    written = 0
    for e in site_entries:
        path = e.get("path", "")
        if not path: continue
        sd = deploy_dir / path
        if not sd.is_dir(): continue
        P = _root_prefix(path)
        js = f"""(function(){{
var P='{P}';
var D={data_json};
var el=document.getElementById('siteNavContent');
if(!el||!D.length)return;
var loc=window.location.pathname;
function isA(p){{return loc.indexOf('/'+p+'/')>=0;}}
var cats={{"large-scale":"Encyclopedias","capture-trap":"Evaluations","topic":"Topic Runs"}};
var g={{}};D.forEach(function(i){{var c=i.category||'x';if(!g[c])g[c]=[];g[c].push(i);}});
var h='<a href="'+P+'/index.html" class="nav-dd-item" style="font-weight:600;border-bottom:2px solid #eaecf0">'
+'<span class="ndi-ph">🏠</span><span class="ndi-info"><span class="ndi-name">LLMpedia Home</span>'
+'<span class="ndi-count">Main portal</span></span></a>';
['large-scale','capture-trap','topic'].forEach(function(c){{
var items=g[c];if(!items||!items.length)return;
h+='<div class="nav-dd-group">'+(cats[c]||c)+'</div>';
items.forEach(function(i){{
var a=isA(i.path)?' active':'';
var l=i.logo?'<img src="'+P+'/'+i.logo+'" alt="">':'<span class="ndi-ph">'+(i.model_display?i.model_display[0]:'?')+'</span>';
var n=i.count?i.count.toLocaleString()+' articles':'';
h+='<a href="'+P+'/'+i.path+'/index.html" class="nav-dd-item'+a+'">'+l
+'<span class="ndi-info"><span class="ndi-name">'+i.label+'</span>'
+'<span class="ndi-count">'+n+'</span></span></a>';
}});}});
el.innerHTML=h;
document.addEventListener('click',function(e){{var d=document.getElementById('siteNavDD');if(d&&!d.contains(e.target))d.classList.remove('open');}});
}})();
"""
        (sd / "site_nav.js").write_text(js, encoding="utf-8")
        written += 1
    print(f"  ✓ site_nav.js written to {written} site(s)")


# ═══════════════════════════════════════════════════════════════════
# Cross-model article index
# ═══════════════════════════════════════════════════════════════════

def _slugify(t):
    t = t.strip().replace(" ", "_")
    return re.sub(r"[^\w\-\.\(\)]+", "_", t)

def _load_subjects_fast(rd):
    subjects = {}
    for n in ("articles.jsonl", "articles_wikitext.jsonl"):
        p = rd / n
        if not p.exists(): continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    s = json.loads(line).get("subject","").strip()
                    if s: subjects[s] = _slugify(s)
                except: pass
        break
    return subjects

def build_cross_model_index(deploy_dir, site_entries, assets):
    t0 = time.time()
    maps = {}
    for e in site_entries:
        rd, path = Path(e.get("run_dir","")), e.get("path","")
        if not rd.is_dir() or not path: continue
        subjects = _load_subjects_fast(rd)
        maps[path] = subjects
        print(f"    {path}: {len(subjects):,} subjects")
    if len(maps) < 2:
        print("  ⚠ Need 2+ sites — skipping"); return

    norm_map = {}
    for path, subjects in maps.items():
        for s, slug in subjects.items():
            norm_map.setdefault(s.lower(), []).append((path, slug, s))
    cross = {n: sites for n, sites in norm_map.items() if len(set(s[0] for s in sites)) >= 2}
    print(f"  Cross-model: {len(cross):,} shared subjects ({time.time()-t0:.1f}s)")
    if not cross: return

    by_path = {e["path"]: e for e in site_entries if e.get("path")}
    for cur_path, cur_subjects in maps.items():
        ce = by_path.get(cur_path, {})
        cd = ce.get("model_display", "")
        P = _root_prefix(cur_path)

        others_set = set()
        for s in cur_subjects:
            if s.lower() in cross:
                for (p, _, _) in cross[s.lower()]:
                    if p != cur_path: others_set.add(p)
        if not others_set: continue

        models = []; idx_map = {}
        for op in sorted(others_set):
            oe = by_path.get(op, {})
            mk, md, cat = oe.get("model_key",""), oe.get("model_display",""), oe.get("category","")
            logo = assets.get("models",{}).get(mk)
            if cat == "topic":
                tm, pm = TOPIC_META.get(oe.get("topic",""),{}), PERSONA_META.get(oe.get("persona_key",""),{})
                lbl = f"{tm.get('icon','📚')} {tm.get('display','')} — {pm.get('display','')}"
            elif cat == "capture-trap": lbl = "Capture Trap"
            else: lbl = md
            idx_map[op] = len(models)
            models.append({"k":mk,"n":md,"l":lbl,"g":f"{P}/{logo}" if logo else "","p":f"{P}/{op}","c":cat})

        refs = {}
        for s, slug in cur_subjects.items():
            if s.lower() not in cross: continue
            ot = [[idx_map[p], sl] for (p, sl, _) in cross[s.lower()] if p != cur_path and p in idx_map]
            if ot: refs[slug] = ot
        if not refs: continue

        js = f"""(function(){{
var M={json.dumps(models,ensure_ascii=False,separators=(",",":"))};
var R={json.dumps(refs,ensure_ascii=False,separators=(",",":"))};
var el=document.getElementById('crossModelBox');
if(!el)return;var s=el.getAttribute('data-subject');if(!s)return;
var h=R[s];if(!h||!h.length)return;
var o='<div class="xm-box"><div class="xm-head"><span class="xm-icon">🔄</span> Compare across models</div><div class="xm-list">';
o+='<div class="xm-item xm-current"><span class="xm-badge">current</span> <b>{cd.replace(chr(39),chr(92)+chr(39))}</b></div>';
for(var i=0;i<h.length;i++){{var mi=h[i][0],sl=h[i][1],m=M[mi];
var lg=m.g?'<img src="'+m.g+'" alt="" class="xm-logo">':'';
var cc=m.c==='topic'?' xm-topic':m.c==='capture-trap'?' xm-ct':'';
o+='<a href="'+m.p+'/'+sl+'.html" class="xm-item'+cc+'">'+lg+'<span class="xm-name">'+m.l+'</span><span class="xm-arrow">→</span></a>';}}
o+='</div></div>';el.innerHTML=o;
}})();
"""
        sd = deploy_dir / cur_path
        if sd.is_dir(): (sd / "cross_model.js").write_text(js, encoding="utf-8")
    print(f"  ✓ cross_model.js written ({time.time()-t0:.1f}s)")


# ═══════════════════════════════════════════════════════════════════
# Landing page
# ═══════════════════════════════════════════════════════════════════

def build_landing_page(deploy_dir, site_entries, assets, mode="anonymous", github_url=DEFAULT_GITHUB):
    large_scale = [e for e in site_entries if e.get("category") == "large-scale"]
    capture     = [e for e in site_entries if e.get("category") == "capture-trap"]
    topic       = [e for e in site_entries if e.get("category") == "topic"]
    topics_grouped = {}
    for e in topic: topics_grouped.setdefault(e.get("topic","?"), []).append(e)

    def _mc(e):
        mk, d, p, c = e.get("model_key",""), e.get("model_display",""), e.get("path","#"), e.get("article_count",0)
        el = e.get("elicit_model","")
        logo = assets.get("models",{}).get(mk)
        lh = f'<img src="{logo}" alt="{d}" class="card-logo">' if logo else f'<div class="card-logo-ph">{d[0]}</div>'
        eh = f'<span class="tag tag-m">🤖 {el}</span>' if el else ""
        return f'<a href="{p}/index.html" class="model-card">{lh}<span class="card-name">{d}</span><span class="card-count">{c:,} articles</span>{eh}</a>'

    def _cc(e):
        p, c, el = e.get("path","#"), e.get("article_count",0), e.get("elicit_model","")
        logo = assets.get("models",{}).get("capture_trap")
        lh = f'<img src="{logo}" alt="" class="card-logo">' if logo else '<div class="card-logo-ph">CT</div>'
        eh = f'<span class="tag tag-m">🤖 {el}</span>' if el else ""
        return f'<a href="{p}/index.html" class="model-card">{lh}<span class="card-name">Capture Trap</span><span class="card-count">{c:,} articles</span>{eh}</a>'

    def _pc(e):
        p, c = e.get("path","#"), e.get("article_count",0)
        pk = e.get("persona_key","")
        pm = PERSONA_META.get(pk, {})
        pn, pi = pm.get("display", e.get("persona",pk)), pm.get("icon","📝")
        mk, md = e.get("model_key",""), e.get("model_display","")
        ml = assets.get("models",{}).get(mk)
        mh = f'<img src="{ml}" alt="{md}" class="p-mlogo">' if ml else f'<span class="p-mtxt">{md}</span>'
        return (f'<a href="{p}/index.html" class="p-card"><div class="p-top"><span class="p-icon">{pi}</span>'
                f'<span class="p-name">{pn}</span></div><div class="p-bot">{mh}<span class="p-cnt">{c:,} articles</span></div></a>')

    ls = "\n".join(_mc(e) for e in large_scale)
    ct = "\n".join(_cc(e) for e in capture) if capture else ""
    ts = ""
    for tk, ents in sorted(topics_grouped.items()):
        tm = TOPIC_META.get(tk, {})
        td, ti = tm.get("display", tk.replace("_"," ").title()), tm.get("icon","📚")
        timg = assets.get("topics",{}).get(tk)
        ih = f'<img src="{timg}" alt="{td}" class="t-cover">' if timg else f'<div class="t-cover-ph">{ti}</div>'
        pc = "\n".join(_pc(e) for e in ents)
        nm = len(set(e.get("model_key","") for e in ents))
        ts += f'<div class="t-section"><div class="t-head">{ih}<div class="t-info"><h3>{ti} {td}</h3><p>{len(ents)} perspectives across {nm} model(s)</p></div></div><div class="p-grid">{pc}</div></div>'

    src = f'<p class="src-link">📂 <a href="{github_url}" target="_blank">Source Code</a></p>' if mode=="anonymous" else ""
    slogo = assets.get("models",{}).get("site_logo")
    logo_img = f'<img src="{slogo}" alt="LLMpedia" class="hero-logo">' if slogo else ""
    footer = f'<footer><div>Licensed under <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a></div><div class="ft-ver">LLMpedia {VERSION}</div></footer>'

    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>LLMpedia</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;color:#202122;background:#f8f9fa;min-height:100vh}}
a{{color:#0645ad;text-decoration:none}}
.hero{{background:linear-gradient(135deg,#1b3a5c 0%,#2a6496 60%,#1b3a5c 100%);color:#fff;padding:2.5rem 1.5rem;text-align:center}}
.hero-logo{{height:64px;margin-bottom:.5rem}}
.hero h1{{font-family:Linux Libertine,Georgia,"Times New Roman",serif;font-size:2.4rem;font-weight:400;margin-bottom:0.3rem}}
.hero .sub{{font-size:1rem;opacity:0.85;max-width:600px;margin:0 auto;line-height:1.5}}
.hero .warn{{display:inline-block;margin-top:0.8rem;padding:0.3rem 0.8rem;background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.2);border-radius:3px;font-size:0.82rem}}
.wrap{{max-width:1000px;margin:0 auto;padding:1.5rem 1rem 3rem}}
.src-link{{text-align:center;margin:0.5rem 0 1rem;font-size:0.88rem}}
.sec-title{{font-family:Linux Libertine,Georgia,"Times New Roman",serif;font-size:1.4rem;font-weight:400;color:#1b3a5c;margin:2rem 0 0.4rem;border-bottom:1px solid #a2a9b1;padding-bottom:0.2rem}}
.sec-desc{{color:#54595d;font-size:0.9rem;margin-bottom:1rem;line-height:1.5}}
.m-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:1rem;margin-bottom:1.5rem}}
.model-card{{display:flex;flex-direction:column;align-items:center;padding:1.2rem 1rem;background:#fff;border:1px solid #c8ccd1;border-radius:4px;color:#202122;transition:border-color .15s,box-shadow .15s,transform .15s}}
.model-card:hover{{border-color:#36c;box-shadow:0 3px 12px rgba(0,0,0,.1);transform:translateY(-2px);text-decoration:none}}
.card-logo{{height:48px;max-width:120px;object-fit:contain;margin-bottom:0.5rem}}
.card-logo-ph{{width:48px;height:48px;border-radius:50%;background:#36c;color:#fff;display:flex;align-items:center;justify-content:center;font-size:1.2rem;font-weight:700;margin-bottom:0.5rem}}
.card-name{{font-weight:700;font-size:1rem;text-align:center}}.card-count{{font-size:0.8rem;color:#72777d;margin-top:0.15rem}}
.tag{{display:inline-block;margin-top:0.3rem;padding:.12rem .45rem;background:#eaecf0;border-radius:2px;font-size:.73rem;color:#54595d}}.tag-m{{background:#eaf3fb;color:#36c}}
.t-section{{background:#fff;border:1px solid #c8ccd1;border-radius:4px;margin-bottom:1.2rem;overflow:hidden}}
.t-head{{display:flex;align-items:stretch;min-height:120px}}
.t-cover{{width:200px;min-height:120px;object-fit:cover;flex-shrink:0;display:block}}
.t-cover-ph{{width:200px;min-height:120px;background:linear-gradient(135deg,#667eea,#764ba2);display:flex;align-items:center;justify-content:center;font-size:3rem;flex-shrink:0}}
.t-info{{padding:.8rem 1rem;display:flex;flex-direction:column;justify-content:center}}
.t-info h3{{font-family:Linux Libertine,Georgia,"Times New Roman",serif;font-size:1.15rem;font-weight:400;color:#1b3a5c;margin-bottom:.2rem}}.t-info p{{font-size:.82rem;color:#72777d}}
.p-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:.6rem;padding:.8rem}}
.p-card{{display:flex;flex-direction:column;padding:.6rem .7rem;border:1px solid #e0e0e0;border-radius:3px;color:#202122;transition:border-color .15s,background .15s}}
.p-card:hover{{border-color:#36c;background:#f0f6ff;text-decoration:none}}
.p-top{{display:flex;align-items:center;gap:.35rem;margin-bottom:.3rem}}.p-icon{{font-size:1rem}}.p-name{{font-weight:600;font-size:.88rem}}
.p-bot{{display:flex;align-items:center;gap:.4rem}}.p-mlogo{{height:16px;max-width:60px;object-fit:contain;opacity:.7}}.p-mtxt{{font-size:.72rem;color:#72777d}}.p-cnt{{font-size:.72rem;color:#72777d}}
footer{{text-align:center;padding:1rem;color:#72777d;font-size:.8rem;border-top:1px solid #c8ccd1;margin-top:2rem}}footer a{{color:#54595d}}
.ft-ver{{color:#a2a9b1;font-size:.72rem;margin-top:.2rem}}
@media(max-width:640px){{.hero h1{{font-size:1.8rem}}.hero-logo{{height:48px}}.m-grid{{grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:.6rem}}.t-head{{flex-direction:column}}.t-cover,.t-cover-ph{{width:100%;min-height:140px;max-height:180px}}.p-grid{{grid-template-columns:1fr 1fr}}}}
</style></head><body>
<div class="hero">{logo_img}<h1>LLMpedia</h1>
<p class="sub">A transparent, open encyclopedia generated entirely from LLM parametric knowledge &mdash; no retrieval, no external sources.</p>
<span class="warn">&#9888; All content is AI-generated and may contain inaccuracies. For research purposes only.</span></div>
<div class="wrap">{src}
<h2 class="sec-title">Large-Scale Encyclopedias</h2>
<p class="sec-desc">Full BFS-expanded encyclopedias with NER filtering and similarity deduplication.</p>
<div class="m-grid">{ls}</div>
{"<h2 class='sec-title'>Capture Trap Evaluation</h2><p class='sec-desc'>1,000 pre-selected entities generated without BFS expansion.</p><div class='m-grid'>"+ct+"</div>" if ct else ""}
{"<h2 class='sec-title'>Topic-Focused Runs</h2><p class='sec-desc'>Each topic explored from multiple editorial perspectives across different models.</p>"+ts if ts else ""}
</div>{footer}</body></html>"""

    (deploy_dir / "index.html").write_text(page, encoding="utf-8")
    print(f"  ✓ Landing page → {deploy_dir / 'index.html'}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Build all LLMpedia sites")
    ap.add_argument("--output-dir", default="deploy")
    ap.add_argument("--mode", choices=["anonymous","deploy"], default="anonymous")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--skip-images", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--image-workers", type=int, default=40)
    ap.add_argument("--only", choices=["large-scale","topic","capture-trap","assemble"])
    ap.add_argument("--github-url", default=DEFAULT_GITHUB)
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
        print(f"WARN: image builder not found — skipping images"); args.skip_images = True

    print(f"\n{'='*70}\n  LLMpedia Build ({VERSION})\n{'='*70}")
    print(f"  Deploy: {deploy_dir}\n  Mode:   {args.mode}")

    large_scale = {"gpt-5-mini": Path(args.gpt_dir), "deepseek": Path(args.deepseek_dir), "llama": Path(args.llama_dir)}
    capture_trap, topic_root = Path(args.capture_trap_dir), Path(args.topic_runs_dir)

    if args.clean and deploy_dir.exists():
        print(f"\nCleaning {deploy_dir}"); shutil.rmtree(deploy_dir)
    deploy_dir.mkdir(parents=True, exist_ok=True)
    global_store = deploy_dir / "image_store"
    global_store.mkdir(parents=True, exist_ok=True)

    print(f"\n{'█'*70}\n  STATIC ASSETS\n{'█'*70}")
    assets = copy_static_assets(deploy_dir)

    site_entries = []

    def _process_run(rd, model_key, category, deploy_subpath, display, **extra):
        meta = _load_run_meta(rd)
        elicit = _extract_model_display(meta)
        print(f"  → {model_key}: elicit={elicit or '?'}")
        if not args.skip_images:
            build_images(rd, image_builder, global_store, workers=args.image_workers)
        # ── ALWAYS sync local images → global store ──
        sync_local_images_to_global(rd, global_store)
        rp = _root_prefix(deploy_subpath)
        if not args.skip_build:
            build_site(rd, site_builder, global_store, clean=args.clean,
                       workers=args.workers, mode=args.mode, root_prefix=rp)
        dp = deploy_dir / deploy_subpath
        copy_site_to_deploy(rd, dp)
        link_global_images(dp, global_store)
        entry = {"category": category, "model_key": model_key, "model_display": display,
                 "elicit_model": elicit, "path": deploy_subpath,
                 "article_count": count_articles(rd), "run_dir": str(rd)}
        entry.update(extra)
        site_entries.append(entry)

    if args.only in (None, "large-scale"):
        print(f"\n{'█'*70}\n  LARGE-SCALE RUNS\n{'█'*70}")
        for mk, rr in large_scale.items():
            if not rr.is_dir(): print(f"  SKIP {mk}"); continue
            rds = find_run_dirs(rr)
            if not rds: print(f"  SKIP {mk}: no runs"); continue
            _process_run(rds[0], mk, "large-scale", mk, MODEL_DISPLAY.get(mk, mk))

    if args.only in (None, "capture-trap"):
        print(f"\n{'█'*70}\n  CAPTURE TRAP\n{'█'*70}")
        if capture_trap.is_dir():
            rds = find_run_dirs(capture_trap)
            if rds: _process_run(rds[0], "gpt-5-mini", "capture-trap", "capture_trap", "GPT-5-mini (Capture Trap)")
        else: print("  SKIP: not found")

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
                            if k in mk.lower() or mk.lower() in k.lower(): mk = k; break
                        _process_run(rds[0], mk, "topic", rel, MODEL_DISPLAY.get(mk, model_dir.name),
                                     topic=topic_dir.name, persona=persona_dir.name.replace("_"," ").title(),
                                     persona_key=persona_dir.name)
        else: print("  SKIP: not found")

    print(f"\n{'█'*70}\n  ASSEMBLING\n{'█'*70}")
    build_landing_page(deploy_dir, site_entries, assets, mode=args.mode, github_url=args.github_url)

    print(f"\n{'█'*70}\n  SITE NAVIGATION\n{'█'*70}")
    write_site_nav_js(deploy_dir, site_entries, assets)

    print(f"\n{'█'*70}\n  CROSS-MODEL INDEX\n{'█'*70}")
    build_cross_model_index(deploy_dir, site_entries, assets)

    ic = sum(1 for f in global_store.iterdir() if f.is_file()) if global_store.is_dir() else 0
    if ic == 0:
        print(f"\n  ⚠ WARNING: Global image_store is EMPTY! Check run dirs for local image_store/ folders.")
    else:
        print(f"\n  ✓ Global image_store: {ic:,} files")

    me = [{k: v for k, v in e.items() if k != "run_dir"} for e in site_entries]
    (deploy_dir / "manifest.json").write_text(
        json.dumps({"mode": args.mode, "version": VERSION, "sites": me, "total_sites": len(me)},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*70}\n  DEPLOY SUMMARY\n{'='*70}")
    print(f"  Output: {deploy_dir}\n  Mode:   {args.mode}\n  Sites:  {len(site_entries)}\n  Images: {ic:,}")
    for cat in ["large-scale","capture-trap","topic"]:
        es = [e for e in site_entries if e["category"]==cat]
        if es: print(f"    {cat}: {len(es)} site(s), {sum(e.get('article_count',0) for e in es):,} articles")
    print(f"\n  Package (follows symlinks):\n    tar -czhf llmpedia.tar.gz -C {deploy_dir.parent} {deploy_dir.name}/")


if __name__ == "__main__":
    main()
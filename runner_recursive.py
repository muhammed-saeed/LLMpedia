# runner_recursive.py
from __future__ import annotations
import argparse, os, json, sqlite3, time, threading
from typing import Dict, Tuple, Set, List

from settings import settings
from db_models import open_queue_db, open_facts_db, count_queue
from pipeline.subject_processor import process_subject
from pipeline.article_writer import write_article, save_article_json

# ---------- jsonl utils ----------
_jsonl_lock = threading.Lock()
def _append_jsonl(path: str, obj: dict):
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    with _jsonl_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)

def _ensure_output_dir(base_dir: str | None) -> str:
    out = base_dir or os.path.join("runs_llmpedia", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "articles"), exist_ok=True)
    return out

def _build_paths(out_dir: str) -> Dict[str, str]:
    tmp = os.path.join(out_dir, "tmp")
    os.makedirs(tmp, exist_ok=True)
    return {
        "queue_sqlite": os.path.join(out_dir, "queue.sqlite"),
        "facts_sqlite": os.path.join(out_dir, "facts.sqlite"),
        "queue_jsonl": os.path.join(out_dir, "queue.jsonl"),
        "facts_jsonl": os.path.join(out_dir, "facts.jsonl"),
        "lowconf_jsonl": os.path.join(out_dir, "facts_lowconf.jsonl"),
        "ner_jsonl": os.path.join(out_dir, "ner_decisions.jsonl"),
        "ner_lowconf_jsonl": os.path.join(out_dir, "ner_lowconf.jsonl"),
        "sources_jsonl": os.path.join(out_dir, "sources.jsonl"),
        "articles_jsonl": os.path.join(out_dir, "articles.index.jsonl"),
        "queue_json": os.path.join(out_dir, "queue.json"),
        "facts_json": os.path.join(out_dir, "facts.json"),
        "run_meta_json": os.path.join(out_dir, "run_meta.json"),
        "errors_log": os.path.join(out_dir, "errors.log"),
        "articles_dir": os.path.join(out_dir, "articles"),
        "tmp_dir": tmp,
    }

# ---------- queue helpers (minimal) ----------
def _enqueue_seed(qdb: sqlite3.Connection, seed: str):
    cur = qdb.cursor()
    cur.execute(
        "INSERT OR IGNORE INTO queue(subject, subject_norm, subject_canon, hop, status) VALUES(?,?,?,?,?)",
        (seed, seed.lower(), "", 0, "pending")
    )
    qdb.commit()

def _claim_next(qdb: sqlite3.Connection, max_depth: int) -> Tuple[str, int] | None:
    cur = qdb.cursor()
    if max_depth == 0:
        cur.execute("SELECT subject, hop FROM queue WHERE status='pending' ORDER BY hop, created_at LIMIT 1")
    else:
        cur.execute("SELECT subject, hop FROM queue WHERE status='pending' AND hop<=? ORDER BY hop, created_at LIMIT 1", (max_depth,))
    row = cur.fetchone()
    if not row:
        return None
    subject, hop = row
    cur.execute("UPDATE queue SET status='working' WHERE subject=? AND hop=? AND status='pending'", (subject, hop))
    qdb.commit()
    cur.execute("SELECT status FROM queue WHERE subject=? AND hop=?", (subject, hop))
    st = cur.fetchone()[0]
    if st != "working":
        return None
    return (subject, hop)

def _finish_subject(qdb: sqlite3.Connection, subject: str, hop: int, ok: bool):
    cur = qdb.cursor()
    cur.execute("UPDATE queue SET status=? WHERE subject=? AND hop=?", ("done" if ok else "pending", subject, hop))
    qdb.commit()

def _enqueue_many(qdb: sqlite3.Connection, subjects_hop: List[Tuple[str,int]]):
    cur = qdb.cursor()
    for s, h in subjects_hop:
        cur.execute(
            "INSERT OR IGNORE INTO queue(subject, subject_norm, subject_canon, hop, status) VALUES(?,?,?,?,?)",
            (s, s.lower(), "", h, "pending")
        )
    qdb.commit()

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("LLMPedia recursive runner (debug/jsonl)")
    ap.add_argument("--seed", required=True)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--max-depth", type=int, default=settings.MAX_DEPTH)
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--conf-threshold", type=float, default=0.70)
    ap.add_argument("--write-articles", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    out_dir = _ensure_output_dir(args.output_dir)
    paths = _build_paths(out_dir)
    if args.debug: print(f"[runner] output_dir={out_dir}")

    qdb = open_queue_db(paths["queue_sqlite"])
    fdb = open_facts_db(paths["facts_sqlite"])

    _enqueue_seed(qdb, args.seed)
    _append_jsonl(paths["queue_jsonl"], {"event":"seed_enqueued", "subject": args.seed, "hop": 0})

    seen: Set[Tuple[str,int]] = set()
    processed = 0
    last_log = time.perf_counter()

    # pass jsonl path bundle to the processor
    jsonl_bundle = {
        "facts_jsonl": paths["facts_jsonl"],
        "lowconf_jsonl": paths["lowconf_jsonl"],
        "ner_jsonl": paths["ner_jsonl"],
        "ner_lowconf_jsonl": paths["ner_lowconf_jsonl"],
        "sources_jsonl": paths["sources_jsonl"],
    }

    # initial counts
    d,w,p,t = count_queue(qdb)
    if args.debug: print(f"[progress] init done={d} working={w} pending={p} total={t}")

    while True:
        claim = _claim_next(qdb, args.max_depth)
        if not claim:
            d,w,p,t = count_queue(qdb)
            print(f"[idle] queue drained: done={d} working={w} pending={p} total={t}")
            break

        subject, hop = claim
        _append_jsonl(paths["queue_jsonl"], {"event":"claimed", "subject": subject, "hop": hop})

        if (subject, hop) in seen:
            _finish_subject(qdb, subject, hop, ok=True)
            _append_jsonl(paths["queue_jsonl"], {"event":"skipped_seen", "subject": subject, "hop": hop})
            continue
        seen.add((subject, hop))

        try:
            if args.debug:
                print(f"\n=== PROCESS [{subject}] (hop {hop}) ===")
            facts, next_subjects, sources = process_subject(
                subject, hop, fdb,
                jsonl_paths=jsonl_bundle,
                debug=args.debug,
                conf_threshold=args.conf_threshold,
            )

            # write article (and index jsonl)
            if args.write_articles:
                article = write_article(subject, facts, sources)
                path = save_article_json(paths["articles_dir"], subject, article)
                _append_jsonl(paths["articles_jsonl"], {
                    "subject": subject, "hop": hop, "path": path,
                    "title": article.get("title"), "sections": len(article.get("sections", [])),
                    "references": len(article.get("references", []))
                })
                if args.debug:
                    print(f"[article] wrote {path}")

            # expansion
            if next_subjects and (args.max_depth == 0 or hop + 1 <= args.max_depth):
                payload = [(s, hop + 1) for s in next_subjects]
                _enqueue_many(qdb, payload)
                _append_jsonl(paths["queue_jsonl"], {"event":"enqueued_children", "subject": subject, "hop": hop, "children": next_subjects})
                if args.debug:
                    print(f"[expand] +{len(payload)} subjects: {', '.join(next_subjects[:8])}{'…' if len(next_subjects)>8 else ''}")

            _finish_subject(qdb, subject, hop, ok=True)
            _append_jsonl(paths["queue_jsonl"], {"event":"done", "subject": subject, "hop": hop})
            processed += 1

            # cap
            if args.max_subjects and processed >= args.max_subjects:
                print(f"[stop] processed cap reached: {processed}")
                break

            # progress log
            now = time.perf_counter()
            if now - last_log >= 2.0:
                d,w,p,t = count_queue(qdb)
                print(f"[progress] done={d} working={w} pending={p} total={t}")
                last_log = now

        except KeyboardInterrupt:
            print("\n[interrupt] exiting…")
            _append_jsonl(paths["queue_jsonl"], {"event":"interrupt"})
            break
        except Exception as e:
            # log error line and requeue once
            with open(paths["errors_log"], "a", encoding="utf-8") as ef:
                ef.write(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] subject={subject} hop={hop} error={repr(e)}\n")
            _finish_subject(qdb, subject, hop, ok=False)
            _append_jsonl(paths["queue_jsonl"], {"event":"error", "subject": subject, "hop": hop, "error": str(e)})

    # ----- end-of-run snapshots -----
    # queue snapshot
    conn = sqlite3.connect(paths["queue_sqlite"])
    cur = conn.cursor()
    cur.execute("SELECT subject, hop, status, retries, created_at FROM queue ORDER BY hop, subject")
    rows = cur.fetchall()
    with open(paths["queue_json"], "w", encoding="utf-8") as f:
        json.dump(
            [{"subject": s, "hop": h, "status": st, "retries": r, "created_at": ts} for (s, h, st, r, ts) in rows],
            f, ensure_ascii=False, indent=2
        )
    conn.close()

    # facts snapshot
    conn = sqlite3.connect(paths["facts_sqlite"])
    cur = conn.cursor()
    cur.execute("SELECT subject, predicate, object, hop, model_name, strategy, confidence FROM triples_accepted ORDER BY subject, predicate, object, hop")
    rows_acc = cur.fetchall()
    with open(paths["facts_json"], "w", encoding="utf-8") as f:
        json.dump(
            {
                "accepted": [
                    {"subject": s, "predicate": p, "object": o, "hop": h, "model": m, "strategy": st, "confidence": c}
                    for (s,p,o,h,m,st,c) in rows_acc
                ]
            },
            f, ensure_ascii=False, indent=2
        )
    conn.close()

    # meta
    run_meta = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "max_depth": args.max_depth,
        "max_subjects": args.max_subjects,
        "conf_threshold": args.conf_threshold,
        "write_articles": bool(args.write_articles),
        "models": {
            "elicitation": settings.MODELS[settings.ELICIT_MODEL_KEY].model,
            "ner": settings.MODELS[settings.NER_MODEL_KEY].model,
        },
        "paths": paths,
    }
    with open(paths["run_meta_json"], "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    print(f"[done] processed={processed} → {out_dir}")
    for k in ("queue_jsonl","facts_jsonl","lowconf_jsonl","ner_jsonl","ner_lowconf_jsonl","sources_jsonl","articles_jsonl","queue_json","facts_json","run_meta_json","errors_log"):
        print(f"[out] {k:16}: {paths[k]}")
    if args.write_articles:
        print(f"[out] articles_dir   : {paths['articles_dir']}")

if __name__ == "__main__":
    main()

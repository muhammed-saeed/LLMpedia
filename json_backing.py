
# json_backing.py (BUFFERED VERSION)
# Drop-in replacement with buffered writes for much faster parallel processing

from __future__ import annotations
import os, json, threading, datetime, time, random, math
from collections import deque
from typing import Dict, Tuple, List, Optional, Any


def _nowz() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"


def _now_ts() -> float:
    return time.time()


def _atomic_write_json(path: str, obj: Any):
    tmp = path + ".tmp"
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _append_jsonl(path: str, rec: dict):
    """Original unbuffered version - kept for compatibility."""
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def _append_jsonl_batch(path: str, recs: List[dict]):
    """Write multiple records at once."""
    if not recs:
        return
    dir_ = os.path.dirname(path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class JsonQueue:
    """
    JSON-backed queue with parent tracking + retry scheduling.
    
    BUFFERED VERSION: State changes are accumulated in memory and flushed:
      - When buffer reaches buffer_size items
      - When flush_interval seconds have passed since last flush
      - When flush() is called explicitly
    
    This dramatically reduces disk I/O for parallel workloads.

    Each row has:
      subject, hop, status, retries, created_at,
      parent_subject, parent_hop,
      retry_at   (unix timestamp seconds; None means "eligible now")

    Retry behavior:
      - When an item errors and retries remain:
          status -> pending
          retry_at -> now + delay
          item goes to retry_order (priority queue)
      - claim_pending_batch() prefers retry_order items whose retry_at is due,
        then falls back to new pending_order items.
    """

    def __init__(
        self,
        state_path: str,
        events_path: str,
        max_retries: Optional[int] = None,
        retry_sleep: float = 5.0,
        retry_backoff: float = 2.0,
        retry_max_sleep: float = 300.0,
        retry_jitter: float = 0.1,
        # NEW: Buffering parameters
        buffer_size: int = 100,
        flush_interval: float = 5.0,
    ):
        self.state_path = state_path
        self.events_path = events_path
        self.max_retries = max_retries

        # retry tuning
        self.retry_sleep = float(retry_sleep)
        self.retry_backoff = float(retry_backoff)
        self.retry_max_sleep = float(retry_max_sleep)
        self.retry_jitter = float(retry_jitter)

        # NEW: Buffering state
        self.buffer_size = max(1, int(buffer_size))
        self.flush_interval = max(0.1, float(flush_interval))
        self._pending_events: List[dict] = []  # buffered JSONL events
        self._state_dirty = False  # whether state needs to be written
        self._last_flush_ts = time.time()
        self._ops_since_flush = 0  # count of operations since last flush

        self._lock = threading.RLock()
        self.by_key: Dict[str, dict] = {}
        self.pending_order: deque[str] = deque()  # new items (retries == 0)
        self.retry_order: deque[str] = deque()    # retry items (retries > 0)
        self._load()

    @staticmethod
    def _key(subject: str, hop: int) -> str:
        return f"{subject}\u241E{hop}"

    @staticmethod
    def _unkey(k: str) -> Tuple[str, int]:
        s, h = k.rsplit("\u241E", 1)
        return s, int(h)

    def _load(self):
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                arr = json.load(f) or []

            for rec in arr:
                sub = rec.get("subject", "")
                hop = int(rec.get("hop", 0))
                k = self._key(sub, hop)

                self.by_key[k] = {
                    "subject": sub,
                    "hop": hop,
                    "status": rec.get("status", "pending"),
                    "retries": int(rec.get("retries", 0)),
                    "created_at": rec.get("created_at") or _nowz(),
                    "parent_subject": rec.get("parent_subject"),
                    "parent_hop": rec.get("parent_hop"),
                    "retry_at": rec.get("retry_at"),  # float seconds or None
                }

            pend = [k for k, r in self.by_key.items() if r.get("status") == "pending"]

            # split into retries vs new
            retry = [k for k in pend if int(self.by_key[k].get("retries", 0)) > 0]
            new = [k for k in pend if int(self.by_key[k].get("retries", 0)) == 0]

            # best-effort ordering
            retry.sort(
                key=lambda kk: (
                    float(self.by_key[kk].get("retry_at") or 0.0),
                    self.by_key[kk].get("created_at", ""),
                )
            )
            new.sort(key=lambda kk: self.by_key[kk].get("created_at", ""))

            self.retry_order = deque(retry)
            self.pending_order = deque(new)

        except Exception:
            # start empty if state file is broken
            self.by_key = {}
            self.pending_order = deque()
            self.retry_order = deque()

    def _save_state_now(self):
        """Write state JSON immediately (internal use)."""
        _atomic_write_json(self.state_path, list(self.by_key.values()))
        self._state_dirty = False

    def _flush_events_now(self):
        """Write buffered events to JSONL immediately (internal use)."""
        if self._pending_events:
            _append_jsonl_batch(self.events_path, self._pending_events)
            self._pending_events = []

    def _maybe_flush(self):
        """
        Check if we should flush based on buffer size or time interval.
        Called after each operation (but actual I/O only happens when needed).
        """
        self._ops_since_flush += 1
        now = time.time()
        
        should_flush = (
            self._ops_since_flush >= self.buffer_size or
            (now - self._last_flush_ts) >= self.flush_interval
        )
        
        if should_flush:
            self._do_flush_locked()

    def _do_flush_locked(self):
        """Perform the actual flush (must hold lock)."""
        # Write buffered events
        if self._pending_events:
            _append_jsonl_batch(self.events_path, self._pending_events)
            self._pending_events = []
        
        # Write state if dirty
        if self._state_dirty:
            _atomic_write_json(self.state_path, list(self.by_key.values()))
            self._state_dirty = False
        
        self._last_flush_ts = time.time()
        self._ops_since_flush = 0

    def flush(self):
        """
        Force flush all buffered writes to disk.
        Call this at the end of processing or periodically for safety.
        """
        with self._lock:
            self._do_flush_locked()

    def _save(self):
        """
        BUFFERED: Mark state as dirty instead of writing immediately.
        Actual write happens in _maybe_flush() or flush().
        """
        self._state_dirty = True
        self._maybe_flush()

    def _append_event(self, event: dict):
        """
        BUFFERED: Add event to buffer instead of writing immediately.
        """
        self._pending_events.append(event)
        # Note: _maybe_flush is called by the operation methods after _save()

    def has_rows(self) -> bool:
        return bool(self.by_key)

    def get_record(self, subject: str, hop: int) -> Optional[dict]:
        """Read-only view of the queue record (includes parent_* and retry_at)."""
        with self._lock:
            r = self.by_key.get(self._key(subject, hop))
            return dict(r) if isinstance(r, dict) else None

    def reset_working_to_pending(self) -> int:
        with self._lock:
            n = 0
            for k, r in self.by_key.items():
                if r.get("status") == "working":
                    r["status"] = "pending"
                    n += 1
                    # put back into correct queue
                    if int(r.get("retries", 0)) > 0:
                        self.retry_order.append(k)
                    else:
                        self.pending_order.append(k)

            if n:
                self._state_dirty = True
                self._append_event({"ts": _nowz(), "event": "reset_working_to_pending", "count": n})
                self._maybe_flush()
            return n

    def enqueue(
        self,
        subject: str,
        hop: int,
        parent_subject: Optional[str] = None,
        parent_hop: Optional[int] = None,
    ) -> Tuple[str, int, str]:
        with self._lock:
            k = self._key(subject, hop)

            # ✅ If it already exists, MERGE missing parent pointers
            if k in self.by_key:
                r = self.by_key[k]
                changed = False

                if parent_subject is not None and (r.get("parent_subject") is None):
                    r["parent_subject"] = parent_subject
                    changed = True

                if parent_hop is not None and (r.get("parent_hop") is None):
                    r["parent_hop"] = int(parent_hop)
                    changed = True

                if changed:
                    self._state_dirty = True
                    self._append_event({
                        "ts": _nowz(),
                        "event": "enqueue_merge_parent",
                        "subject": subject,
                        "hop": hop,
                        "parent_subject": r.get("parent_subject"),
                        "parent_hop": r.get("parent_hop"),
                    })
                    self._maybe_flush()

                return subject, hop, "exists"

            # ✅ New insert keeps parent pointers
            rec = {
                "subject": subject,
                "hop": hop,
                "status": "pending",
                "retries": 0,
                "created_at": _nowz(),
                "parent_subject": parent_subject,
                "parent_hop": parent_hop,
                "retry_at": None,
            }
            self.by_key[k] = rec
            self.pending_order.append(k)
            self._state_dirty = True

            self._append_event({
                "ts": _nowz(),
                "event": "enqueue",
                "subject": subject,
                "hop": hop,
                "parent_subject": parent_subject,
                "parent_hop": parent_hop,
            })
            self._maybe_flush()
            return subject, hop, "inserted"

    def _compute_retry_delay(self, retry_num: int) -> float:
        """
        retry_num: 1 for first retry attempt after the initial failure, 2 for second, ...
        delay = retry_sleep * retry_backoff^(retry_num-1), capped to retry_max_sleep, with jitter.
        jitter is a fraction, e.g. 0.1 => +/-10%
        """
        base = max(0.0, self.retry_sleep)
        backoff = max(1.0, self.retry_backoff)
        cap = max(0.0, self.retry_max_sleep)
        jitter = max(0.0, self.retry_jitter)

        if base <= 0.0:
            return 0.0

        # exponential
        raw = base * (backoff ** max(0, retry_num - 1))

        # cap before jitter so it doesn't grow unbounded
        raw = min(raw, cap) if cap > 0 else raw

        if jitter > 0 and raw > 0:
            # +/- jitter fraction
            factor = 1.0 + random.uniform(-jitter, jitter)
            raw = max(0.0, raw * factor)

        # final cap
        if cap > 0:
            raw = min(raw, cap)

        return float(raw)

    def claim_pending_batch(self, max_depth: int, n: int) -> List[Tuple[str, int]]:
        """
        Prefer due retries first (retry_order), then new work (pending_order).
        Items with retry_at in the future are skipped (remain queued).
        """
        now = _now_ts()
        out: List[Tuple[str, int]] = []

        def _try_claim_from(dq: deque[str]):
            nonlocal out, now
            if not dq or len(out) >= n:
                return

            # scan each element at most once to avoid infinite loops
            limit = len(dq)
            for _ in range(limit):
                if len(out) >= n:
                    return
                k = dq.popleft()
                r = self.by_key.get(k)
                if r is None:
                    continue
                if r.get("status") != "pending":
                    continue

                if max_depth != 0 and int(r.get("hop", 0)) > max_depth:
                    dq.append(k)
                    continue

                ra = r.get("retry_at")
                if ra is not None and float(ra) > now:
                    dq.append(k)
                    continue

                # claim it
                r["status"] = "working"
                out.append((r.get("subject", ""), int(r.get("hop", 0))))

        with self._lock:
            _try_claim_from(self.retry_order)
            if len(out) < n:
                _try_claim_from(self.pending_order)

            if out:
                self._state_dirty = True
                self._append_event({"ts": _nowz(), "event": "claim", "items": out})
                self._maybe_flush()

        return out

    def mark_done(self, subject: str, hop: int):
        with self._lock:
            k = self._key(subject, hop)
            r = self.by_key.get(k)
            if not r:
                return
            r["status"] = "done"
            r["retry_at"] = None
            self._state_dirty = True
            self._append_event({"ts": _nowz(), "event": "done", "subject": subject, "hop": hop})
            self._maybe_flush()

    def mark_error(self, subject: str, hop: int, max_retries: int, reason: Optional[str] = None):
        """
        On error:
          retries += 1
          if retries >= max_retries => status=failed
          else status=pending and retry scheduled; queued in retry_order
        """
        with self._lock:
            k = self._key(subject, hop)
            r = self.by_key.get(k)
            if not r:
                return

            r["retries"] = int(r.get("retries", 0)) + 1

            if r["retries"] >= int(max_retries):
                r["status"] = "failed"
                r["retry_at"] = None
                delay = None
            else:
                r["status"] = "pending"
                delay_s = self._compute_retry_delay(r["retries"])
                r["retry_at"] = _now_ts() + float(delay_s)
                # requeue into retry priority queue
                self.retry_order.append(k)
                delay = float(delay_s)

            self._state_dirty = True
            self._append_event({
                "ts": _nowz(),
                "event": "error_retry",
                "subject": subject,
                "hop": hop,
                "retries": r["retries"],
                "max_retries": int(max_retries),
                "retry_at": r.get("retry_at"),
                "delay_s": delay,
                "reason": reason,
            })
            self._maybe_flush()

    def next_due_in(self, max_depth: int) -> Optional[float]:
        """
        Returns:
        - 0.0 if at least one retry item is due now
        - positive seconds until the next retry is due
        - None if there are no pending retry items
        """
        now = _now_ts()
        best: Optional[float] = None

        # Snapshot under lock so concurrent writers can't break iteration
        with self._lock:
            vals = list(self.by_key.values())

        for r in vals:
            if r.get("status") != "pending":
                continue
            if int(r.get("retries", 0)) <= 0:
                continue
            if max_depth != 0 and int(r.get("hop", 0)) > max_depth:
                continue

            ra = r.get("retry_at")
            # None means eligible now
            if ra is None:
                return 0.0

            dt = float(ra) - now
            if dt <= 0:
                return 0.0
            if best is None or dt < best:
                best = dt

        return best

    def metrics(self, max_depth: int) -> Tuple[int, int, int, int]:
        # Snapshot under lock so concurrent writers can't break iteration
        with self._lock:
            vals = list(self.by_key.values())

        d = w = p = f = 0
        for r in vals:
            if max_depth != 0 and int(r.get("hop", 0)) > max_depth:
                continue
            st = r.get("status")
            if st == "done":
                d += 1
            elif st == "working":
                w += 1
            elif st == "pending":
                p += 1
            elif st == "failed":
                f += 1
        return d, w, p, f


def write_article_record_jsonl(
    articles_jsonl: str,
    subject: str,
    hop: int,
    model: str,
    wikitext: str,
    overall_confidence: Optional[float],
    parent_subject: Optional[str] = None,
    parent_hop: Optional[int] = None,
):
    if not isinstance(wikitext, str) or not wikitext.strip():
        return
    _append_jsonl(
        articles_jsonl,
        {
            "subject": subject,
            "wikitext": wikitext,
            "hop": hop,
            "model": model,
            "overall_confidence": overall_confidence,
            "parent_subject": parent_subject,
            "parent_hop": parent_hop,
            "created_at": _nowz(),
        },
    )


def stream_build_json_from_jsonl(jsonl_path: str, json_path: str):
    """
    Build a valid JSON array by streaming a .jsonl file. No in-memory list.
    """
    dir_ = os.path.dirname(json_path)
    if dir_:
        os.makedirs(dir_, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as out:
        out.write("[\n")
        first = True
        if os.path.exists(jsonl_path):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if not first:
                        out.write(",\n")
                    out.write(line)
                    first = False
        out.write("\n]\n")

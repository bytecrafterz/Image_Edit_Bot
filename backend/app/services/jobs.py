"""Background jobs.

Generating six images takes minutes, and the client is holding a phone.  So the
HTTP request starts a job and returns; the phone polls.  The important property
is that a job can never leave a run stuck on "running" forever - a crash, a
killed process or a hung provider all end with the run marked failed and a
reason the user can read.
"""
from __future__ import annotations

import logging
import threading
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

from .. import db
from ..config import SETTINGS

log = logging.getLogger("photorobot.jobs")

STALE_AFTER_S = 30 * 60

_executor = ThreadPoolExecutor(
    max_workers=max(1, SETTINGS.limits.max_concurrent_jobs),
    thread_name_prefix="photorobot-job",
)
_lock = threading.Lock()
_jobs: dict[str, Future] = {}
_cancelled: set[str] = set()


def submit(run_id: str, fn: Callable[[], Any]) -> dict:
    """Queue a run.  Refuses to start the same run twice."""
    with _lock:
        existing = _jobs.get(run_id)
        if existing is not None and not existing.done():
            return {"ok": False, "reason": "Ese trabajo ya se esta ejecutando.",
                    "run_id": run_id}
        _cancelled.discard(run_id)
        future = _executor.submit(_wrap, run_id, fn)
        _jobs[run_id] = future
    return {"ok": True, "run_id": run_id, "status": "running"}


def _wrap(run_id: str, fn: Callable[[], Any]) -> Any:
    db.execute("UPDATE runs SET status='running', started_at=? WHERE id=?",
               (db.now(), run_id))
    try:
        return fn()
    except Exception as exc:                              # noqa: BLE001
        log.error("Run %s failed:\n%s", run_id, traceback.format_exc())
        db.execute(
            "UPDATE runs SET status='failed', error=?, finished_at=? WHERE id=?",
            (str(exc)[:500], db.now(), run_id),
        )
        return {"ok": False, "error": str(exc)}
    finally:
        with _lock:
            _jobs.pop(run_id, None)


def cancel(run_id: str) -> bool:
    """Ask a run to stop.  The orchestrator checks between every image."""
    with _lock:
        _cancelled.add(run_id)
        future = _jobs.get(run_id)
    if future is not None and future.cancel():
        db.execute("UPDATE runs SET status='cancelled', finished_at=? WHERE id=?",
                   (db.now(), run_id))
        return True
    db.execute("UPDATE runs SET stage='Deteniendo...' WHERE id=? AND status='running'",
               (run_id,))
    return True


def is_cancelled(run_id: str) -> bool:
    with _lock:
        return run_id in _cancelled


def clear_cancel(run_id: str) -> None:
    with _lock:
        _cancelled.discard(run_id)


def status(run_id: str) -> dict:
    """Current state, with a watchdog for runs whose worker vanished."""
    row = db.row_to_dict(db.q1("SELECT * FROM runs WHERE id=?", (run_id,)))
    if not row:
        return {"ok": False, "reason": "No existe ese trabajo."}

    with _lock:
        future = _jobs.get(run_id)
    running = future is not None and not future.done()

    if (row.get("status") == "running" and not running
            and float(row.get("started_at") or 0) < db.now() - STALE_AFTER_S):
        db.execute(
            "UPDATE runs SET status='failed', error=?, finished_at=? WHERE id=?",
            ("El trabajo se interrumpio inesperadamente.", db.now(), run_id),
        )
        row["status"] = "failed"
        row["error"] = "El trabajo se interrumpio inesperadamente."

    return {"ok": True, "run_id": run_id, "status": row.get("status"),
            "progress": float(row.get("progress") or 0.0),
            "stage": row.get("stage") or "", "error": row.get("error"),
            "alive": running}


def active_count() -> int:
    with _lock:
        return sum(1 for f in _jobs.values() if not f.done())

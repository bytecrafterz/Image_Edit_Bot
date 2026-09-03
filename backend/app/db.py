"""SQLite access layer.

A deliberately small hand written layer instead of an ORM: the schema is fixed,
the queries are simple, and this keeps the dependency surface (and the Windows
install story) minimal.  Connections are thread local because FastAPI runs sync
endpoints in a worker thread pool.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Iterable, Sequence

from .config import DB_PATH, ensure_dirs

_local = threading.local()
_init_lock = threading.Lock()
_initialised = False

SCHEMA_VERSION = 4


# --------------------------------------------------------------- connections

def _connect() -> sqlite3.Connection:
    ensure_dirs()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def get_conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = _connect()
        _local.conn = conn
    if not _initialised:
        init_db()
    return conn


@contextmanager
def tx():
    """Transaction scope.  Commits on success, rolls back on any exception."""
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


# ------------------------------------------------------------------ helpers

def new_id(prefix: str = "") -> str:
    raw = uuid.uuid4().hex[:24]
    return f"{prefix}_{raw}" if prefix else raw


def now() -> float:
    return time.time()


def q(sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
    return get_conn().execute(sql, params).fetchall()


def q1(sql: str, params: Sequence[Any] = ()) -> sqlite3.Row | None:
    return get_conn().execute(sql, params).fetchone()


def execute(sql: str, params: Sequence[Any] = ()) -> sqlite3.Cursor:
    conn = get_conn()
    cur = conn.execute(sql, params)
    conn.commit()
    return cur


def executemany(sql: str, seq: Iterable[Sequence[Any]]) -> None:
    conn = get_conn()
    conn.executemany(sql, seq)
    conn.commit()


def row_to_dict(row: sqlite3.Row | None) -> dict | None:
    """Row -> dict, decoding every ``*_json`` column into a real object."""
    if row is None:
        return None
    d = dict(row)
    for k, v in list(d.items()):
        if k.endswith("_json"):
            d.pop(k)
            d[k[:-5]] = loads(v, None)
    return d


def rows_to_dicts(rows: Iterable[sqlite3.Row]) -> list[dict]:
    return [row_to_dict(r) for r in rows]


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def loads(value: Any, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


# ------------------------------------------------------------------- schema

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS users (
    id             TEXT PRIMARY KEY,
    email          TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash  TEXT NOT NULL,
    display_name   TEXT NOT NULL DEFAULT '',
    role           TEXT NOT NULL DEFAULT 'user',
    status         TEXT NOT NULL DEFAULT 'pending',
    plan           TEXT NOT NULL DEFAULT 'free',
    locale         TEXT NOT NULL DEFAULT 'es',
    daily_limit_usd    REAL NOT NULL DEFAULT 2.0,
    monthly_limit_usd  REAL NOT NULL DEFAULT 25.0,
    free_quota_daily   INTEGER NOT NULL DEFAULT 40,
    created_at     REAL NOT NULL,
    approved_at    REAL,
    last_login_at  REAL,
    notes          TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_users_status ON users(status);

CREATE TABLE IF NOT EXISTS sessions (
    token      TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    user_agent TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

CREATE TABLE IF NOT EXISTS profiles (
    id            TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    person_name   TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'draft',
    n_sources     INTEGER NOT NULL DEFAULT 0,
    coverage_json TEXT NOT NULL DEFAULT '{}',
    face_json     TEXT NOT NULL DEFAULT '{}',
    body_json     TEXT NOT NULL DEFAULT '{}',
    skin_json     TEXT NOT NULL DEFAULT '{}',
    hair_json     TEXT NOT NULL DEFAULT '{}',
    marks_json    TEXT NOT NULL DEFAULT '[]',
    consent_json  TEXT NOT NULL DEFAULT '{}',
    thresholds_json TEXT NOT NULL DEFAULT '{}',
    is_default    INTEGER NOT NULL DEFAULT 0,
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    deleted_at    REAL
);
CREATE INDEX IF NOT EXISTS idx_profiles_user ON profiles(user_id);

CREATE TABLE IF NOT EXISTS originals (
    id           TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    profile_id   TEXT REFERENCES profiles(id) ON DELETE SET NULL,
    filename     TEXT NOT NULL,
    path         TEXT NOT NULL,
    thumb_path   TEXT,
    width        INTEGER NOT NULL DEFAULT 0,
    height       INTEGER NOT NULL DEFAULT 0,
    bytes        INTEGER NOT NULL DEFAULT 0,
    sha256       TEXT NOT NULL DEFAULT '',
    shot_type    TEXT NOT NULL DEFAULT 'unknown',
    quality_json TEXT NOT NULL DEFAULT '{}',
    analysis_json TEXT NOT NULL DEFAULT '{}',
    tags         TEXT NOT NULL DEFAULT '',
    sort_order   INTEGER NOT NULL DEFAULT 0,
    created_at   REAL NOT NULL,
    deleted_at   REAL
);
CREATE INDEX IF NOT EXISTS idx_originals_user ON originals(user_id, deleted_at);
CREATE INDEX IF NOT EXISTS idx_originals_profile ON originals(profile_id);

CREATE TABLE IF NOT EXISTS runs (
    id           TEXT PRIMARY KEY,
    user_id      TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    original_id  TEXT REFERENCES originals(id) ON DELETE SET NULL,
    profile_id   TEXT REFERENCES profiles(id) ON DELETE SET NULL,
    parent_run_id TEXT,
    mode         TEXT NOT NULL DEFAULT 'preview',
    status       TEXT NOT NULL DEFAULT 'queued',
    options_json TEXT NOT NULL DEFAULT '{}',
    plan_json    TEXT NOT NULL DEFAULT '{}',
    n_requested  INTEGER NOT NULL DEFAULT 0,
    n_accepted   INTEGER NOT NULL DEFAULT 0,
    n_rejected   INTEGER NOT NULL DEFAULT 0,
    n_repaired   INTEGER NOT NULL DEFAULT 0,
    attempts_used INTEGER NOT NULL DEFAULT 0,
    est_cost_usd REAL NOT NULL DEFAULT 0,
    cost_usd     REAL NOT NULL DEFAULT 0,
    progress     REAL NOT NULL DEFAULT 0,
    stage        TEXT NOT NULL DEFAULT '',
    error        TEXT,
    created_at   REAL NOT NULL,
    started_at   REAL,
    finished_at  REAL
);
CREATE INDEX IF NOT EXISTS idx_runs_user ON runs(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS attempts (
    id            TEXT PRIMARY KEY,
    run_id        TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    user_id       TEXT NOT NULL,
    variant_index INTEGER NOT NULL DEFAULT 0,
    attempt_no    INTEGER NOT NULL DEFAULT 1,
    provider      TEXT NOT NULL DEFAULT '',
    model         TEXT NOT NULL DEFAULT '',
    operation     TEXT NOT NULL DEFAULT 'generate',
    prompt        TEXT NOT NULL DEFAULT '',
    negative_prompt TEXT NOT NULL DEFAULT '',
    params_json   TEXT NOT NULL DEFAULT '{}',
    verdict_json  TEXT NOT NULL DEFAULT '{}',
    defects_json  TEXT NOT NULL DEFAULT '[]',
    status        TEXT NOT NULL DEFAULT 'pending',
    reject_reason TEXT NOT NULL DEFAULT '',
    cost_usd      REAL NOT NULL DEFAULT 0,
    latency_ms    INTEGER NOT NULL DEFAULT 0,
    image_id      TEXT,
    created_at    REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_attempts_run ON attempts(run_id);

CREATE TABLE IF NOT EXISTS images (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    run_id      TEXT REFERENCES runs(id) ON DELETE SET NULL,
    attempt_id  TEXT,
    original_id TEXT,
    profile_id  TEXT,
    kind        TEXT NOT NULL DEFAULT 'preview',
    path        TEXT NOT NULL,
    thumb_path  TEXT,
    width       INTEGER NOT NULL DEFAULT 0,
    height      INTEGER NOT NULL DEFAULT 0,
    bytes       INTEGER NOT NULL DEFAULT 0,
    sha256      TEXT NOT NULL DEFAULT '',
    provider    TEXT NOT NULL DEFAULT '',
    model       TEXT NOT NULL DEFAULT '',
    cost_usd    REAL NOT NULL DEFAULT 0,
    score       REAL NOT NULL DEFAULT 0,
    verdict_json TEXT NOT NULL DEFAULT '{}',
    meta_json   TEXT NOT NULL DEFAULT '{}',
    is_favorite INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    deleted_at  REAL
);
CREATE INDEX IF NOT EXISTS idx_images_user ON images(user_id, deleted_at, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_images_run ON images(run_id);
CREATE INDEX IF NOT EXISTS idx_images_fav ON images(user_id, is_favorite);

CREATE TABLE IF NOT EXISTS feedback (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    image_id   TEXT,
    run_id     TEXT,
    verdict    TEXT NOT NULL,
    reason     TEXT NOT NULL DEFAULT '',
    context_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback(user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS learning (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    scope      TEXT NOT NULL,
    weights_json TEXT NOT NULL DEFAULT '{}',
    stats_json TEXT NOT NULL DEFAULT '{}',
    updated_at REAL NOT NULL,
    UNIQUE(user_id, scope)
);

CREATE TABLE IF NOT EXISTS ledger (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    provider    TEXT NOT NULL,
    kind        TEXT NOT NULL,
    amount_usd  REAL NOT NULL,
    balance_after REAL NOT NULL DEFAULT 0,
    ref         TEXT NOT NULL DEFAULT '',
    note        TEXT NOT NULL DEFAULT '',
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ledger_user ON ledger(user_id, provider, created_at DESC);

CREATE TABLE IF NOT EXISTS alerts (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    kind       TEXT NOT NULL,
    level      TEXT NOT NULL DEFAULT 'info',
    title      TEXT NOT NULL DEFAULT '',
    message    TEXT NOT NULL DEFAULT '',
    payload_json TEXT NOT NULL DEFAULT '{}',
    read_at    REAL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alerts_user ON alerts(user_id, read_at, created_at DESC);

CREATE TABLE IF NOT EXISTS options (
    id            TEXT PRIMARY KEY,
    user_id       TEXT,
    group_key     TEXT NOT NULL,
    value_key     TEXT NOT NULL,
    label_es      TEXT NOT NULL,
    label_en      TEXT NOT NULL DEFAULT '',
    prompt_fragment   TEXT NOT NULL DEFAULT '',
    negative_fragment TEXT NOT NULL DEFAULT '',
    params_json   TEXT NOT NULL DEFAULT '{}',
    shot_types    TEXT NOT NULL DEFAULT 'closeup,half,full',
    enabled       INTEGER NOT NULL DEFAULT 1,
    sort_order    INTEGER NOT NULL DEFAULT 0,
    builtin       INTEGER NOT NULL DEFAULT 0,
    created_at    REAL NOT NULL,
    UNIQUE(user_id, group_key, value_key)
);
CREATE INDEX IF NOT EXISTS idx_options_group ON options(group_key, enabled);

CREATE TABLE IF NOT EXISTS styles (
    id          TEXT PRIMARY KEY,
    user_id     TEXT,
    key         TEXT NOT NULL,
    name_es     TEXT NOT NULL,
    name_en     TEXT NOT NULL DEFAULT '',
    description TEXT NOT NULL DEFAULT '',
    shot_types  TEXT NOT NULL DEFAULT 'closeup,half,full',
    prompt_template   TEXT NOT NULL DEFAULT '',
    negative_template TEXT NOT NULL DEFAULT '',
    defaults_json TEXT NOT NULL DEFAULT '{}',
    params_json TEXT NOT NULL DEFAULT '{}',
    enabled     INTEGER NOT NULL DEFAULT 1,
    sort_order  INTEGER NOT NULL DEFAULT 0,
    builtin     INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    UNIQUE(user_id, key)
);

CREATE TABLE IF NOT EXISTS user_settings (
    user_id  TEXT NOT NULL,
    key      TEXT NOT NULL,
    value_json TEXT NOT NULL DEFAULT 'null',
    updated_at REAL NOT NULL,
    PRIMARY KEY (user_id, key)
);

CREATE TABLE IF NOT EXISTS audit (
    id         TEXT PRIMARY KEY,
    user_id    TEXT,
    actor      TEXT NOT NULL DEFAULT '',
    kind       TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_created ON audit(created_at DESC);
"""


def init_db() -> None:
    global _initialised
    with _init_lock:
        if _initialised:
            return
        conn = getattr(_local, "conn", None)
        if conn is None:
            conn = _connect()
            _local.conn = conn
        conn.executescript(SCHEMA)
        conn.execute(
            "INSERT INTO meta(key,value) VALUES('schema_version',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(SCHEMA_VERSION),),
        )
        conn.commit()
        _initialised = True


def audit(kind: str, user_id: str | None = None, actor: str = "", **payload) -> None:
    """Fire and forget audit trail.  Never allowed to break a request."""
    try:
        execute(
            "INSERT INTO audit(id,user_id,actor,kind,payload_json,created_at) "
            "VALUES(?,?,?,?,?,?)",
            (new_id("aud"), user_id, actor, kind, dumps(payload), now()),
        )
    except sqlite3.Error:
        pass

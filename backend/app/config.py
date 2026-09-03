"""Central configuration and filesystem layout.

Every path used at runtime is resolved here so the whole application can be
relocated by moving a single directory.  Nothing in this module may import
other application modules - it is the bottom of the dependency graph.
"""
from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ---------------------------------------------------------------- filesystem

BACKEND_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BACKEND_DIR.parent
FRONTEND_DIR = ROOT_DIR / "frontend"
DATA_DIR = Path(os.environ.get("PHOTOROBOT_DATA", ROOT_DIR / "data"))

UPLOAD_DIR = DATA_DIR / "uploads"      # original photos, per user
OUTPUT_DIR = DATA_DIR / "outputs"      # final high quality renders
PREVIEW_DIR = DATA_DIR / "previews"    # cheap preview renders
PROFILE_DIR = DATA_DIR / "profiles"    # identity profiles (json + thumbs)
CACHE_DIR = DATA_DIR / "cache"         # analysis caches keyed by file hash
LOG_DIR = DATA_DIR / "logs"
SCENE_DIR = DATA_DIR / "scenes"        # backgrounds for the free local provider
DB_PATH = DATA_DIR / "photorobot.sqlite3"
SECRET_PATH = DATA_DIR / "secret.key"
KEYSTORE_PATH = DATA_DIR / "keystore.json"

ALL_DIRS = [DATA_DIR, UPLOAD_DIR, OUTPUT_DIR, PREVIEW_DIR, PROFILE_DIR,
            CACHE_DIR, LOG_DIR, SCENE_DIR]


def ensure_dirs() -> None:
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------- secrets

def get_secret_key() -> str:
    """Stable per-installation secret, generated on first run."""
    ensure_dirs()
    if SECRET_PATH.exists():
        text = SECRET_PATH.read_text(encoding="utf-8").strip()
        if text:
            return text
    key = secrets.token_hex(32)
    SECRET_PATH.write_text(key, encoding="utf-8")
    try:  # best effort on Windows; ACLs are handled by the parent folder
        os.chmod(SECRET_PATH, 0o600)
    except OSError:
        pass
    return key


# ------------------------------------------------------------------ settings

@dataclass
class Limits:
    """Hard safety rails.  Deliberately conservative: this system spends money."""
    max_upload_mb: int = 30
    max_originals_per_user: int = 400
    max_previews_per_run: int = 12
    max_repair_rounds: int = 2
    max_retries_per_variant: int = 2
    max_concurrent_jobs: int = 2
    # How many variants of ONE run may be waiting on a provider at the same
    # time.  Almost all of a run's wall clock is the provider's queue, so N
    # images cost N latencies when they are sent one after another; a few at a
    # time collapses that to roughly one.  Deliberately small: the money gate,
    # the provider's own rate limits and the single-threaded MediaPipe checks
    # all prefer a handful of calls in flight over a flood.  1 restores the
    # strictly sequential behaviour.
    max_parallel_generations: int = 3
    default_daily_usd: float = 2.0
    default_monthly_usd: float = 25.0
    # Balance alerting - the client asked explicitly for warnings before zero.
    low_balance_usd: float = 2.0
    critical_balance_usd: float = 0.5


@dataclass
class Settings:
    app_name: str = "Photo Robot"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8080
    # "local" costs nothing and needs no keys; "auto" upgrades when keys exist.
    default_image_provider: str = "auto"
    default_vision_provider: str = "auto"
    session_days: int = 30
    limits: Limits = field(default_factory=Limits)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


SETTINGS = Settings()

# --------------------------------------------------------------- API keyring
# Keys are stored per installation in data/keystore.json (chmod 600 where the
# OS supports it) and can also come from the environment.  They are NEVER sent
# to the browser - the API only ever reports whether a key is present.

_ENV_NAMES = {
    "anthropic": "ANTHROPIC_API_KEY",
    "fal": "FAL_KEY",
    "openai": "OPENAI_API_KEY",
    "replicate": "REPLICATE_API_TOKEN",
    "stability": "STABILITY_API_KEY",
}


def _read_keystore() -> dict:
    if not KEYSTORE_PATH.exists():
        return {}
    try:
        return json.loads(KEYSTORE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_keystore(data: dict) -> None:
    ensure_dirs()
    KEYSTORE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        os.chmod(KEYSTORE_PATH, 0o600)
    except OSError:
        pass


def get_api_key(name: str) -> str | None:
    """Environment wins over the keystore so deployments can override."""
    env = _ENV_NAMES.get(name)
    if env:
        val = os.environ.get(env)
        if val:
            return val.strip()
    val = _read_keystore().get(name)
    return val.strip() if isinstance(val, str) and val.strip() else None


def set_api_key(name: str, value: str | None) -> None:
    store = _read_keystore()
    if value:
        store[name] = value.strip()
    else:
        store.pop(name, None)
    _write_keystore(store)


def key_status() -> dict:
    """Safe-to-serialise view: presence and a masked hint, never the key."""
    out = {}
    for name in _ENV_NAMES:
        key = get_api_key(name)
        out[name] = {
            "present": bool(key),
            "hint": (key[:6] + "..." + key[-4:]) if key and len(key) > 12 else None,
            "from_env": bool(os.environ.get(_ENV_NAMES[name])),
        }
    return out

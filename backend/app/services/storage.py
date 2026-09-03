"""File storage: where bytes live and who is allowed to read them.

Photographs of a person are the most sensitive thing this system holds, so
nothing is ever served by path.  Files are addressed by database id, and an
<img> tag - which cannot send an Authorization header - gets a short HMAC token
instead, signed with the installation secret and bound to one image id.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import re
import shutil
import unicodedata
import uuid
from pathlib import Path

from .. import db
from ..config import (OUTPUT_DIR, PREVIEW_DIR, PROFILE_DIR, UPLOAD_DIR,
                      get_secret_key)

_KIND_DIRS = {
    "upload": UPLOAD_DIR,
    "original": UPLOAD_DIR,
    "preview": PREVIEW_DIR,
    "final": OUTPUT_DIR,
    "repair": OUTPUT_DIR,
    "profile": PROFILE_DIR,
}

_SAFE_RE = re.compile(r"[^\w.\- ]+", re.UNICODE)


def user_dir(user_id: str, kind: str) -> Path:
    base = _KIND_DIRS.get(kind, OUTPUT_DIR)
    path = base / _safe_segment(user_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_segment(value: str) -> str:
    cleaned = _SAFE_RE.sub("_", str(value or "").strip())
    return cleaned[:80] or "anon"


def safe_filename(name: str) -> str:
    """Keep the name recognisable to the user, keep the filesystem safe."""
    name = unicodedata.normalize("NFKC", str(name or "")).strip()
    name = name.replace("\\", "/").split("/")[-1]
    stem, dot, ext = name.rpartition(".")
    if not dot:
        stem, ext = name, "jpg"
    stem = _SAFE_RE.sub("_", stem)[:60] or "foto"
    ext = _SAFE_RE.sub("", ext).lower()[:8] or "jpg"
    return f"{stem}.{ext}"


def unique_path(directory: Path, filename: str) -> Path:
    """Always a fresh name: two uploads called IMG_0001.jpg must not collide."""
    directory.mkdir(parents=True, exist_ok=True)
    safe = safe_filename(filename)
    return directory / f"{uuid.uuid4().hex[:12]}_{safe}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def find_duplicate(user_id: str, sha: str) -> dict | None:
    """Re-uploading the same photograph returns the existing record."""
    row = db.q1(
        "SELECT * FROM originals WHERE user_id=? AND sha256=? AND deleted_at IS NULL",
        (user_id, sha),
    )
    return db.row_to_dict(row)


def store_upload(user_id: str, filename: str, data: bytes) -> dict:
    """Write an uploaded photograph to disk.  Returns path, hash and size."""
    sha = sha256_bytes(data)
    path = unique_path(user_dir(user_id, "upload"), filename)
    path.write_bytes(data)
    return {"path": str(path), "sha256": sha, "bytes": len(data),
            "filename": safe_filename(filename)}


def store_output(user_id: str, src: Path | str, kind: str, run_id: str) -> dict:
    """Move a generated file into its permanent home."""
    src = Path(src)
    target_dir = user_dir(user_id, kind) / _safe_segment(run_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    dst = target_dir / src.name
    if src.resolve() != dst.resolve():
        shutil.move(str(src), str(dst))
    data = dst.read_bytes()
    return {"path": str(dst), "sha256": sha256_bytes(data), "bytes": len(data)}


def delete_file(path: str | None) -> bool:
    if not path:
        return False
    try:
        p = Path(path)
        if p.is_file():
            p.unlink()
            return True
    except OSError:
        pass
    return False


def delete_tree(path: str | Path) -> None:
    shutil.rmtree(Path(path), ignore_errors=True)


# ------------------------------------------------------------------- tokens

def image_token(image_id: str, variant: str = "full") -> str:
    """Unguessable, unexpiring handle for one image, usable in an <img> src.

    Not a session: it grants read access to exactly one file and nothing else,
    which is what a gallery of eighty thumbnails on a phone actually needs.
    """
    payload = f"{image_id}:{variant}"
    sig = hmac.new(get_secret_key().encode(), payload.encode(), hashlib.sha256)
    digest = base64.urlsafe_b64encode(sig.digest()[:18]).decode().rstrip("=")
    ident = base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")
    return f"{ident}.{digest}"


def resolve_token(token: str) -> tuple[str, str] | None:
    """(image_id, variant) when the signature checks out, else None."""
    try:
        ident, _, digest = str(token).partition(".")
        if not ident or not digest:
            return None
        pad = "=" * (-len(ident) % 4)
        payload = base64.urlsafe_b64decode(ident + pad).decode()
        image_id, _, variant = payload.partition(":")
        if not image_id:
            return None
        expected = image_token(image_id, variant or "full")
        if not hmac.compare_digest(expected, str(token)):
            return None
        return image_id, (variant or "full")
    except (ValueError, UnicodeDecodeError):
        return None


def public_url(image_id: str, variant: str = "full") -> str:
    return f"/api/files/{image_token(image_id, variant)}"


def image_urls(image_id: str) -> dict:
    return {"full": public_url(image_id, "full"),
            "thumb": public_url(image_id, "thumb")}

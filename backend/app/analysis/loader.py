"""Pixel input/output for every analysis module.

The client photographs with an iPhone: 4032x3024 JPEG carrying an EXIF
orientation flag.  Decoding such a file without honouring that flag silently
rotates the frame, and every measurement taken afterwards - shot type, body
proportions, face geometry - is then taken on a rotated image and is wrong.
So the whole application reads pixels here and never calls cv2.imread
directly.  This module also owns the small JSON cache that makes a repeated
analysis of the same file free, and the jsonable() conversion that keeps numpy
values out of the database and the API.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageOps

from ..config import CACHE_DIR

__all__ = ["load_image", "load_rgb", "save_image", "make_thumb", "file_sha256",
           "image_info", "cached", "jsonable"]

# Extensions we attempt to decode.  HEIC is listed because the phone produces
# it; whether it decodes depends on the Pillow build, and a failure is a clear
# ValueError rather than a crash halfway through an analysis.
READABLE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif",
                ".bmp", ".tif", ".tiff")
ENCODABLE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")

_HASH_CHUNK = 1 << 20
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")

# EXIF tag numbers (PIL exposes the raw IFD, not the pretty names).
_TAG_ORIENTATION = 274
_TAG_EXIF_IFD = 0x8769
_TAG_DATETIME_ORIGINAL = 36867
_TAG_DATETIME_DIGITIZED = 36868
_TAG_DATETIME = 306
_SWAPPED_ORIENTATIONS = (5, 6, 7, 8)


# ------------------------------------------------------------------ decoding

def _flatten_to_rgb(im: Image.Image) -> Image.Image:
    """Drop alpha onto white so transparent PNGs do not read as black skin."""
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        canvas = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(canvas, im)
    return im.convert("RGB")


def _pil_to_bgr(path: Path) -> np.ndarray | None:
    """Preferred path: honours EXIF orientation before anything else runs."""
    try:
        with Image.open(path) as im:
            im.load()
            fixed = ImageOps.exif_transpose(im) or im
            arr = np.array(_flatten_to_rgb(fixed), dtype=np.uint8, copy=True)
    except Exception:  # PIL raises OSError/ValueError/SyntaxError/bomb errors
        return None
    if arr.ndim != 3 or arr.shape[2] != 3 or arr.size == 0:
        return None
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _cv_to_bgr(path: Path) -> np.ndarray | None:
    """Last resort for files PIL refuses.  np.fromfile keeps Unicode paths
    working, which cv2.imread does not on Windows.  No EXIF here."""
    try:
        raw = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if raw.size == 0:
        return None
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if isinstance(img, np.ndarray) and img.size:
        return img
    return None


def _fit(img: np.ndarray, max_side: int) -> np.ndarray:
    """Shrink so the longest side is max_side.  Never enlarges."""
    if not max_side or max_side <= 0:
        return img
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    scale = float(max_side) / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def load_image(path: str | Path, max_side: int = 0) -> np.ndarray:
    """BGR uint8 with EXIF orientation applied.  max_side=0 means no resize."""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"No existe el archivo de imagen '{p.name}'")
    img = _pil_to_bgr(p)
    if img is None:
        img = _cv_to_bgr(p)
    if img is None:
        raise ValueError(
            f"No se pudo abrir la imagen '{p.name}'. Formatos admitidos: "
            + ", ".join(READABLE_EXT)
        )
    return _fit(img, max_side)


def load_rgb(path: str | Path, max_side: int = 0) -> np.ndarray:
    """RGB uint8 view for the libraries that expect it (MediaPipe, PIL)."""
    return cv2.cvtColor(load_image(path, max_side), cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------ encoding

def _as_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    arr = np.asarray(img)
    if arr.dtype.kind == "f":
        peak = float(np.nanmax(arr)) if arr.size else 0.0
        if math.isfinite(peak) and peak <= 1.0:
            arr = arr * 255.0
    return np.clip(np.nan_to_num(arr, nan=0.0), 0, 255).astype(np.uint8)


def _write_bytes(path: Path, payload: bytes) -> None:
    """Write through a temporary file so a reader never sees half an image.
    The temp name carries pid and thread id: two workers may cache the same
    analysis key at the same moment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_bytes(payload)
        os.replace(tmp, path)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def save_image(img: np.ndarray, path: str | Path, quality: int = 95) -> str:
    """Encode by extension: PNG ignores quality, JPEG/WEBP honour it.

    Returns the path actually written - an unknown extension is rewritten to
    .jpg so the bytes always match the name.
    """
    if not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError("save_image recibio una imagen vacia")
    p = Path(path)
    ext = p.suffix.lower()
    if ext not in ENCODABLE_EXT:
        p = p.with_suffix(".jpg")
        ext = ".jpg"

    data = _as_uint8(img)
    if data.ndim == 3 and data.shape[2] == 4 and ext not in (".png", ".webp"):
        data = cv2.cvtColor(data, cv2.COLOR_BGRA2BGR)

    q = int(max(1, min(100, quality)))
    if ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 6]
    elif ext == ".webp":
        params = [cv2.IMWRITE_WEBP_QUALITY, q]
    elif ext in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, q, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
    else:
        params = []

    ok, buf = cv2.imencode(ext, data, params)
    if not ok:
        raise ValueError(f"No se pudo codificar la imagen como '{ext}'")
    _write_bytes(p, buf.tobytes())
    return str(p)


def make_thumb(src: str | Path, dst: str | Path, size: int = 512) -> str:
    """JPEG thumbnail, longest side = size, aspect preserved, quality 82."""
    target = int(size) if size and size > 0 else 512
    img = load_image(src, max_side=target)
    p = Path(dst)
    if p.suffix.lower() not in (".jpg", ".jpeg"):
        p = p.with_suffix(".jpg")
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82,
                                         cv2.IMWRITE_JPEG_OPTIMIZE, 1])
    if not ok:
        raise ValueError(f"No se pudo generar la miniatura de '{Path(src).name}'")
    _write_bytes(p, buf.tobytes())
    return str(p)


# -------------------------------------------------------------- file metadata

def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _exif_taken_at(exif: Any) -> float | None:
    """DateTimeOriginal as a unix timestamp.  EXIF carries no timezone, so it
    is read as local time of the machine doing the analysis."""
    raw = None
    try:
        sub = exif.get_ifd(_TAG_EXIF_IFD)
    except Exception:
        sub = {}
    for tag in (_TAG_DATETIME_ORIGINAL, _TAG_DATETIME_DIGITIZED):
        value = sub.get(tag) if isinstance(sub, dict) else None
        if isinstance(value, str) and value.strip():
            raw = value
            break
    if raw is None:
        value = exif.get(_TAG_DATETIME)
        if isinstance(value, str) and value.strip():
            raw = value
    if not raw:
        return None
    text = raw.strip().split(".")[0].replace("/", ":").replace("T", " ")
    if text.startswith("0000"):
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except (ValueError, OverflowError, OSError):
            continue
    return None


def image_info(path: str | Path) -> dict:
    """Cheap header read.  width/height are the *displayed* dimensions, i.e.
    after EXIF orientation, so they match what load_image() returns."""
    p = Path(path)
    info: dict[str, Any] = {
        "width": 0, "height": 0, "bytes": 0, "format": "",
        "sha256": "", "exif_orientation": 1, "taken_at": None,
        "path": str(p),
    }
    try:
        info["bytes"] = int(p.stat().st_size)
    except OSError:
        pass
    try:
        info["sha256"] = file_sha256(p)
    except OSError:
        info["reason"] = "archivo ilegible"
        return info

    try:
        with Image.open(p) as im:
            info["format"] = (im.format or p.suffix.lstrip(".")).upper()
            width, height = im.size
            try:
                exif = im.getexif()
                orientation = int(exif.get(_TAG_ORIENTATION, 1) or 1)
                if orientation < 1 or orientation > 8:
                    orientation = 1
                info["exif_orientation"] = orientation
                info["taken_at"] = _exif_taken_at(exif)
            except Exception:
                orientation = 1
            if orientation in _SWAPPED_ORIENTATIONS:
                width, height = height, width
            info["width"] = int(width)
            info["height"] = int(height)
            return info
    except Exception:
        pass

    img = _cv_to_bgr(p)
    if img is None:
        info["reason"] = "formato no reconocido"
        return info
    info["height"], info["width"] = int(img.shape[0]), int(img.shape[1])
    info["format"] = p.suffix.lstrip(".").upper()
    return info


# ----------------------------------------------------------------- JSON cache

def jsonable(obj: Any) -> Any:
    """Convert numpy scalars/arrays and other exotica into JSON primitives.

    Non finite floats become None: json.dumps would otherwise emit NaN, which
    is not valid JSON and breaks the browser.
    """
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return jsonable(obj.item())
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(obj)).decode("ascii")
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.timestamp()
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in obj]
    return str(obj)


def _safe_name(raw: str) -> str:
    text = _SAFE_NAME.sub("_", str(raw)).strip("._")
    if not text:
        text = "x"
    if len(text) > 96:
        digest = hashlib.sha256(str(raw).encode("utf-8")).hexdigest()[:12]
        text = f"{text[:80]}_{digest}"
    return text


def _cache_path(namespace: str, key: str) -> Path:
    return CACHE_DIR / _safe_name(namespace) / f"{_safe_name(key)}.json"


def cached(namespace: str, key: str, fn: Callable[[], dict]) -> dict:
    """Memoise an analysis on disk under data/cache/<namespace>/<key>.json.

    A cache file that is unreadable or not valid JSON is treated as absent and
    recomputed - a corrupt cache must never be able to stop an analysis.  The
    value returned is always the jsonable() form, so a caller cannot tell a hit
    from a miss by the types it gets back.
    """
    path = _cache_path(namespace, key)
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except (OSError, ValueError, UnicodeDecodeError):
        pass

    payload = jsonable(fn())
    if isinstance(payload, dict):
        try:
            blob = json.dumps(payload, ensure_ascii=False, allow_nan=False)
            _write_bytes(path, blob.encode("utf-8"))
        except (OSError, TypeError, ValueError):
            pass  # an uncacheable result is still a valid result
    return payload

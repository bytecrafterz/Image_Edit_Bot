"""Photographic quality of a source picture, measured before anything is spent.

Two jobs.  The first is triage: a blurred, dark or low resolution original
produces a bad render no matter how good the prompt is, and it is far cheaper
to tell the user "repite la foto con mas luz" than to burn provider credit on
it.  The second is the beauty filter check.  The client's photos sometimes
arrive already smoothed by a phone filter; if that goes unnoticed the identity
profile is built from an altered face, and every later verification measures
the wrong person.  A smoothing filter leaves a very specific fingerprint - the
skin inside the face has far less high frequency energy than the skin outside
it - and that is what is looked for here.
"""
from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from .loader import load_image

__all__ = ["assess_quality"]

# Calibration constants.  Each maps a raw physical measurement onto 0..1 so the
# weighted score stays comparable between a 12 MP phone photo and a crop.
_SHARP_HALF = 150.0        # Laplacian variance that scores 0.5
_SHARP_WORK_SIDE = 1024    # sharpness is measured at a fixed scale
_CONTRAST_FULL = 60.0      # std of L that scores 1.0
_NOISE_FULL = 12.0         # residual sigma (grey levels) that scores 1.0 noise
_BLUR_SCORE = 0.40         # below this the photo is called blurred
_MIN_SIDE = 720            # resolution_ok threshold
_CLIP_DARK = 4             # pixel values counted as crushed black
_CLIP_BRIGHT = 251         # pixel values counted as blown white
_BEAUTY_RATIO = 0.45       # facial texture below this fraction = suspected
_BEAUTY_MIN_PX = 1500      # skin pixels needed before the ratio means anything
_BEAUTY_MIN_TILES = 8

_WEIGHTS = {"sharpness": 0.34, "exposure": 0.26, "noise": 0.16,
            "contrast": 0.14, "resolution": 0.10}


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if not math.isfinite(value):
        return lo
    return float(max(lo, min(hi, value)))


def _as_bgr(img: Any, path: str | None) -> np.ndarray | None:
    """Accept an array, or fall back to the path when the caller has none."""
    if isinstance(img, np.ndarray) and img.size:
        arr = img
    elif path:
        try:
            arr = load_image(path)
        except (ValueError, OSError):
            return None
    else:
        return None
    if arr.dtype != np.uint8:
        scaled = np.nan_to_num(arr.astype(np.float32), nan=0.0)
        peak = float(scaled.max()) if scaled.size else 0.0
        if arr.dtype.kind == "f" and peak <= 1.0:
            scaled = scaled * 255.0          # float images normalised to 0..1
        arr = np.clip(scaled, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr
    return None


def _skin_mask(img_bgr: np.ndarray) -> np.ndarray:
    """YCrCb skin gate.  Deliberately loose: it only has to separate skin from
    hair, clothes and background well enough to compare texture."""
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y = ycrcb[:, :, 0]
    cr = ycrcb[:, :, 1]
    cb = ycrcb[:, :, 2]
    mask = ((y > 55) & (y < 250) & (cr > 133) & (cr < 180) &
            (cb > 77) & (cb < 130)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def _region_energy(gray_f: np.ndarray, resid_abs: np.ndarray,
                   selected: np.ndarray) -> float | None:
    """Median high frequency amplitude of a region, normalised by its own
    brightness so a bright face is not credited with more texture than a dim
    arm.  Median, not mean, so a few hard edges cannot carry the region."""
    count = int(np.count_nonzero(selected))
    if count < _BEAUTY_MIN_PX:
        return None
    luma = float(gray_f[selected].mean())
    energy = float(np.median(resid_abs[selected]))
    return energy / max(12.0, luma) * 100.0


def _face_box_px(face: Any, width: int, height: int) -> tuple[int, int, int, int] | None:
    if not isinstance(face, dict) or not face.get("ok", True):
        return None
    box = face.get("bbox")
    values: list[float] = []
    if isinstance(box, (list, tuple)) and len(box) == 4:
        try:
            values = [float(v) for v in box]
        except (TypeError, ValueError):
            values = []
    if not values or values[2] <= 1 or values[3] <= 1:
        norm = face.get("bbox_norm")
        if isinstance(norm, (list, tuple)) and len(norm) == 4:
            try:
                nx, ny, nw, nh = (float(v) for v in norm)
                values = [nx * width, ny * height, nw * width, nh * height]
            except (TypeError, ValueError):
                values = []
    if len(values) != 4:
        return None
    x, y, w, h = values
    if not all(math.isfinite(v) for v in values) or w <= 2 or h <= 2:
        return None
    x0 = int(max(0, min(width - 1, round(x))))
    y0 = int(max(0, min(height - 1, round(y))))
    x1 = int(max(x0 + 1, min(width, round(x + w))))
    y1 = int(max(y0 + 1, min(height, round(y + h))))
    return x0, y0, x1, y1


def _beauty_from_face(gray_f, resid_abs, skin_bool, box) -> tuple[bool, float | None]:
    x0, y0, x1, y1 = box
    h, w = skin_bool.shape
    inside = np.zeros_like(skin_bool)
    inside[y0:y1, x0:x1] = True

    # Neck and hairline sit right against the box; push the exclusion out so the
    # comparison region is genuinely non facial skin.
    pad_x = int(round((x1 - x0) * 0.20))
    pad_y = int(round((y1 - y0) * 0.30))
    excluded = np.zeros_like(skin_bool)
    excluded[max(0, y0 - pad_y):min(h, y1 + pad_y),
             max(0, x0 - pad_x):min(w, x1 + pad_x)] = True

    face_energy = _region_energy(gray_f, resid_abs, inside & skin_bool)
    other_energy = _region_energy(gray_f, resid_abs, skin_bool & ~excluded)
    if face_energy is None or other_energy is None or other_energy <= 1e-6:
        return False, None
    ratio = face_energy / other_energy
    return bool(ratio < _BEAUTY_RATIO), float(ratio)


def _beauty_from_tiles(gray_f, resid_abs, skin_bool) -> tuple[bool, float | None]:
    """No face box available: compare the smoothest fifth of the skin in the
    frame against the skin texture of the frame as a whole."""
    h, w = skin_bool.shape
    cell = int(max(24, min(h, w) // 24))
    if cell < 24 or h < cell or w < cell:
        return False, None
    min_skin = int(cell * cell * 0.55)
    energies: list[float] = []
    for y0 in range(0, h - cell + 1, cell):
        for x0 in range(0, w - cell + 1, cell):
            tile = skin_bool[y0:y0 + cell, x0:x0 + cell]
            if int(np.count_nonzero(tile)) < min_skin:
                continue
            luma = float(gray_f[y0:y0 + cell, x0:x0 + cell][tile].mean())
            energy = float(np.median(resid_abs[y0:y0 + cell, x0:x0 + cell][tile]))
            energies.append(energy / max(12.0, luma) * 100.0)
    if len(energies) < _BEAUTY_MIN_TILES:
        return False, None
    values = np.sort(np.asarray(energies, dtype=np.float64))
    k = max(3, int(round(len(values) * 0.20)))
    smooth = float(values[:k].mean())
    overall = float(np.median(values))
    if overall <= 1e-6:
        return False, None
    ratio = smooth / overall
    return bool(ratio < _BEAUTY_RATIO), float(ratio)


def assess_quality(img_bgr: Any, path: str | None = None,
                   face: dict | None = None) -> dict:
    """Measure a photograph.  ``face`` is optional; when a face bbox is known
    the beauty filter test is far more reliable, so callers that already ran
    analysis/face.py should pass it."""
    result: dict[str, Any] = {
        "ok": False, "score": 0.0, "sharpness": 0.0, "exposure": 0.0,
        "contrast": 0.0, "noise": 0.0, "resolution_ok": False,
        "beauty_filter_suspected": False, "issues": [], "advice": [],
        "reason": "",
    }

    img = _as_bgr(img_bgr, path)
    if img is None:
        result["reason"] = "imagen no legible"
        result["issues"] = ["No se pudo leer la imagen"]
        result["advice"] = ["Vuelve a subir la foto original"]
        return result

    height, width = img.shape[:2]
    if height < 16 or width < 16:
        result["reason"] = "imagen demasiado pequena"
        result["issues"] = ["La imagen es demasiado pequena para analizarla"]
        result["advice"] = ["Envia la foto original sin recortar"]
        return result

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float32)

    # --- sharpness: Laplacian variance at a fixed working scale ------------
    longest = max(height, width)
    if longest > _SHARP_WORK_SIDE:
        scale = _SHARP_WORK_SIDE / float(longest)
        work = cv2.resize(gray, (max(1, int(round(width * scale))),
                                 max(1, int(round(height * scale)))),
                          interpolation=cv2.INTER_AREA)
    else:
        work = gray
    lap_var = float(cv2.Laplacian(work, cv2.CV_64F, ksize=3).var())
    sharpness = _clamp(lap_var / (lap_var + _SHARP_HALF))

    # --- exposure: mean luminance, penalised by clipping -------------------
    total = float(gray.size)
    dark_frac = float(np.count_nonzero(gray < _CLIP_DARK) / total)
    bright_frac = float(np.count_nonzero(gray > _CLIP_BRIGHT) / total)
    mean_lum = float(gray_f.mean()) / 255.0
    base = 1.0 - min(1.0, abs(mean_lum - 0.47) / 0.40)
    clipped = dark_frac + bright_frac
    exposure = _clamp(base * (1.0 - min(1.0, 3.0 * max(0.0, clipped - 0.02))))

    # --- contrast: spread of the L channel ---------------------------------
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_std = float(lab[:, :, 0].std())
    contrast = _clamp(l_std / _CONTRAST_FULL)

    # --- noise: MAD of a high pass residual --------------------------------
    blurred = cv2.GaussianBlur(gray_f, (0, 0), 1.2)
    resid = gray_f - blurred
    resid_abs = np.abs(resid)
    mad = float(np.median(np.abs(resid - float(np.median(resid)))))
    noise_sigma = mad * 1.4826
    noise = _clamp(noise_sigma / _NOISE_FULL)

    resolution_ok = bool(min(height, width) >= _MIN_SIDE)

    # --- beauty filter -----------------------------------------------------
    beauty = False
    beauty_ratio: float | None = None
    beauty_mode = "skipped"
    # A globally blurred or tiny picture has no texture anywhere; the ratio
    # would be noise talking to noise.
    if resolution_ok and lap_var >= 60.0:
        skin_bool = _skin_mask(img).astype(bool)
        box = _face_box_px(face, width, height)
        if box is not None:
            beauty, beauty_ratio = _beauty_from_face(gray_f, resid_abs, skin_bool, box)
            beauty_mode = "face"
        # A tight portrait has almost no skin outside the face box, so the
        # face comparison finds nothing to compare against; fall through to
        # the patch comparison rather than reporting nothing.
        if beauty_ratio is None:
            beauty, beauty_ratio = _beauty_from_tiles(gray_f, resid_abs, skin_bool)
            beauty_mode = "tiles" if box is None else "face_then_tiles"
        if beauty_ratio is None:
            beauty_mode += "_insufficient_skin"

    # --- blended score -----------------------------------------------------
    score = (_WEIGHTS["sharpness"] * sharpness +
             _WEIGHTS["exposure"] * exposure +
             _WEIGHTS["contrast"] * contrast +
             _WEIGHTS["noise"] * (1.0 - noise) +
             _WEIGHTS["resolution"] * (1.0 if resolution_ok else
                                       _clamp(min(height, width) / float(_MIN_SIDE))))
    if beauty:
        score -= 0.08
    score = _clamp(score)

    # --- what the user is told ---------------------------------------------
    issues: list[str] = []
    advice: list[str] = []

    def add(issue: str, tip: str) -> None:
        issues.append(issue)
        if tip not in advice:
            advice.append(tip)

    if sharpness < _BLUR_SCORE:
        add("La foto esta poco nitida",
            "Repite la foto sin movimiento, apoyando el telefono")
    if mean_lum < 0.28:
        add("La foto esta oscura", "Repite la foto con mas luz natural")
    elif mean_lum > 0.72:
        add("La foto esta sobreexpuesta", "Evita la luz directa y baja la exposicion")
    if bright_frac > 0.06:
        add("Hay zonas quemadas sin detalle",
            "Evita la luz directa y baja la exposicion")
    if dark_frac > 0.12:
        add("Hay zonas negras sin detalle", "Repite la foto con mas luz natural")
    if contrast < 0.42:
        add("La foto tiene poco contraste", "Busca una luz mas direccional")
    if noise > 0.55:
        add("La foto tiene mucho ruido",
            "Sube la luz para que el telefono no fuerce el ISO")
    if not resolution_ok:
        add("La resolucion es baja para trabajar con detalle",
            "Envia la foto original, sin recortes ni reenvios por chat")
    if beauty:
        add("La piel de la cara parece suavizada por un filtro",
            "Usa una foto sin filtros de belleza para medir bien la identidad")

    result.update({
        "ok": True,
        "score": round(score, 4),
        "sharpness": round(sharpness, 4),
        "exposure": round(exposure, 4),
        "contrast": round(contrast, 4),
        "noise": round(noise, 4),
        "resolution_ok": resolution_ok,
        "beauty_filter_suspected": bool(beauty),
        "issues": issues,
        "advice": advice,
        "reason": "",
        # Raw measurements, kept for the report and for tuning the thresholds.
        "sharpness_var": round(lap_var, 2),
        "contrast_std": round(l_std, 2),
        "noise_sigma": round(noise_sigma, 3),
        "mean_luma": round(mean_lum, 4),
        "clipped_dark": round(dark_frac, 5),
        "clipped_bright": round(bright_frac, 5),
        "beauty_ratio": None if beauty_ratio is None else round(beauty_ratio, 4),
        "beauty_mode": beauty_mode,
        "width": int(width),
        "height": int(height),
    })
    return result

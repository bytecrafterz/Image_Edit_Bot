"""The free, key-less image provider - and the honest one.

This is NOT a diffusion model and must never pretend to be.  It is a
deterministic computer vision compositing engine: it TRANSFORMS the real
photograph rather than inventing a body.  Every pixel of the person that
leaves this module came from the person's own photograph; what changes is the
background, the colour of the fabric, the light on it, the grade and the crop -
the operations a retoucher performs, not the operations a generator performs.

That is exactly the product's stated principle, which is why the whole robot
pipeline - analyse, plan, generate, verify, detect defects, repair, learn - can
be demonstrated end to end on the developer's machine, at zero cost, before any
card is ever charged.  It also means the identity checks in identity/verify.py
should pass trivially here: nothing moved the face or narrowed the waist.

Everything is driven by ``req.extra``, which the planner fills with the chosen
option values, and by ``req.seed``, from which every random decision is
derived, so the same request always produces the same file.
"""
from __future__ import annotations

import math
import time
import zlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import scenes
from .base import Capabilities, GenRequest, GenResult, ImageProvider

__all__ = ["LocalFreeProvider", "PROVIDER_NAME"]

PROVIDER_NAME = "local"
MODEL_GENERATE = "cv-composite"
MODEL_INPAINT = "cv-inpaint"
MODEL_UPSCALE = "cv-lanczos"

# A person mask outside this band is treated as a failed detection rather than
# trusted: see the background step for why this matters.
MIN_PERSON_COVERAGE = 0.10
MAX_PERSON_COVERAGE = 0.92

MAX_SIDE = 4096
WORK_MAX_SIDE = 2600            # bound the compositing cost, not the output
_DEFAULT_MAX_SIDE = 1024
_MIN_OUT = 64

NOTES_ES = (
    "Transformacion fotografica local, sin IA generativa: reemplaza el fondo, "
    "recolorea la ropa, reilumina, aplica grados de color y reencuadra sobre la "
    "foto real de la persona. No inventa cuerpo ni cara. Pensado para probar el "
    "robot de punta a punta y para el uso gratuito; coste 0 USD por imagen."
)

# Output size per quality tier (longest side).  The provider never invents
# resolution in generate(): if the source is smaller, the source wins and the
# caller is told.  upscale() is the operation that enlarges.
_QUALITY_MAX_SIDE = {
    "draft": 640, "preview": 768, "standard": 1024, "high": 1536, "max": 2048,
}

_FRAMING_ASPECT = {
    "portrait_full": 3.0 / 4.0,
    "portrait_half": 4.0 / 5.0,
    "portrait_closeup": 4.0 / 5.0,
    "portrait_headshot": 4.0 / 5.0,
    "square": 1.0,
    "story_9x16": 9.0 / 16.0,
}


# ---------------------------------------------------------------- primitives

def _as_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else (hi if value > hi else value)


def _clamp_dim(value) -> int:
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        v = _DEFAULT_MAX_SIDE
    return int(min(max(v, _MIN_OUT), MAX_SIDE))


def _key_of(value) -> str:
    return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")


def _stable_seed(*parts) -> int:
    raw = "|".join(str(p) for p in parts).encode("utf-8", "replace")
    return int(zlib.crc32(raw) & 0x7FFFFFFF)


def _resolve_seed(req: GenRequest) -> int:
    if req.seed is not None:
        try:
            return int(req.seed) & 0x7FFFFFFF
        except (TypeError, ValueError):
            pass
    return _stable_seed(req.source_path or "", req.prompt or "", req.quality or "")


def _binary(mask) -> np.ndarray | None:
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.size == 0:
        return None
    return np.where(mask > 127, np.uint8(255), np.uint8(0)).astype(np.uint8)


def _mask_bbox(mask) -> tuple[int, int, int, int] | None:
    if not isinstance(mask, np.ndarray) or mask.size == 0:
        return None
    ys, xs = np.nonzero(mask > 127)
    if xs.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1 - x0 + 1, y1 - y0 + 1)


def _kernel(size: int) -> np.ndarray:
    s = int(max(1, size)) | 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))


def _subtract(base: np.ndarray, other, grow: int = 2) -> np.ndarray:
    """base minus other, with `other` dilated first so we stay clear of it."""
    if not isinstance(other, np.ndarray) or other.shape != base.shape:
        return base
    o = other
    if grow > 0:
        o = cv2.dilate(o, _kernel(grow * 2 + 1), iterations=1)
    return cv2.bitwise_and(base, cv2.bitwise_not(o))


def _union(masks) -> np.ndarray | None:
    out = None
    for m in masks:
        if not isinstance(m, np.ndarray) or m.ndim != 2:
            continue
        out = m.copy() if out is None else cv2.bitwise_or(out, m)
    return out


def _unsharp(img: np.ndarray, amount: float, sigma: float,
             threshold: float = 0.0) -> np.ndarray:
    """Detail preserving sharpen: the gate keeps flat areas (and their noise)
    out of the sharpening entirely."""
    f = img.astype(np.float32)
    blur = cv2.GaussianBlur(f, (0, 0), float(max(0.3, sigma)))
    detail = f - blur
    if threshold > 0.0:
        gate = np.clip(np.abs(detail) / max(1e-3, float(threshold) * 255.0), 0.0, 1.0)
        detail = detail * gate
    return np.clip(f + float(amount) * detail, 0, 255).astype(np.uint8)


def _vignette(img: np.ndarray, amount: float) -> np.ndarray:
    h, w = img.shape[:2]
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32).reshape(1, w)
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1)
    aspect = float(w) / float(max(1, h))
    d = np.sqrt(((xs - 0.5) * aspect) ** 2 + (ys - 0.5) ** 2) / 0.95
    fall = np.clip(1.0 - d, 0.0, 1.0)
    fall = np.broadcast_to(fall, (h, w)).astype(np.float32)
    scale = 1.0 - float(amount) * (1.0 - fall)
    return np.clip(img.astype(np.float32) * scale[..., None], 0, 255).astype(np.uint8)


def _resize_exact(img: np.ndarray, tw: int, th: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    if (iw, ih) == (int(tw), int(th)):
        return img
    interp = cv2.INTER_AREA if (tw * th) < (iw * ih) else cv2.INTER_LANCZOS4
    return cv2.resize(img, (int(tw), int(th)), interpolation=interp)


# ----------------------------------------------------------- analysis access
# Imported lazily inside the functions: analysis modules pull in mediapipe and
# may themselves reach for providers, and none of that may run at import time.

def _load_source(path, max_side: int = 0) -> np.ndarray:
    from ..analysis import loader
    return loader.load_image(path, max_side=int(max_side))


def _save(img: np.ndarray, out_path, quality: int) -> str:
    from ..analysis import loader
    return loader.save_image(img, out_path, quality=int(quality))


def _skin_of(img: np.ndarray, regions: dict) -> np.ndarray | None:
    """Skin envelope, used only to protect skin from the garment recolour.

    Union of the photometric detector with the anatomical face and hand
    regions; legs and torso are deliberately excluded because they are usually
    covered by the very garment we are about to repaint.
    """
    base = None
    try:
        from ..analysis.skin import skin_mask_ycrcb
        candidate = skin_mask_ycrcb(img)
        if isinstance(candidate, np.ndarray) and candidate.shape == img.shape[:2]:
            base = _binary(candidate)
    except Exception:
        base = None
    parts = [base]
    for name in ("face", "hands"):
        m = regions.get(name)
        if isinstance(m, np.ndarray):
            parts.append(m)
    return _union([m for m in parts if m is not None])


def _detect_masks(img: np.ndarray) -> dict:
    """Pose plus every mask the pipeline can get, degrading to empties."""
    out: dict[str, Any] = {
        "pose": {"ok": False, "landmarks": {}},
        "person": None, "coverage": 0.0, "backend": "none",
        "regions": {}, "skin": None, "reasons": [],
    }
    try:
        from ..analysis import pose as pose_mod
        detected = pose_mod.detect_pose(img)
        if isinstance(detected, dict):
            out["pose"] = detected
            if not detected.get("ok"):
                out["reasons"].append(str(detected.get("reason") or "sin pose"))
    except Exception as exc:
        out["reasons"].append("pose no disponible: %s" % exc)

    seg_mod = None
    try:
        from ..analysis import segment as seg_mod  # noqa: F811
    except Exception as exc:
        out["reasons"].append("segmentacion no disponible: %s" % exc)

    if seg_mod is not None:
        try:
            person = seg_mod.person_mask(img)
            if isinstance(person, dict) and person.get("ok"):
                mask = _binary(person.get("mask"))
                if mask is not None and mask.shape == img.shape[:2]:
                    out["person"] = mask
                    out["coverage"] = _as_float(person.get("coverage"), 0.0)
                    out["backend"] = str(person.get("backend") or "")
            elif isinstance(person, dict):
                out["reasons"].append(str(person.get("reason") or "sin mascara"))
        except Exception as exc:
            out["reasons"].append("mascara de persona fallida: %s" % exc)
        try:
            regions = seg_mod.region_masks(img, out["pose"], out["person"])
            if isinstance(regions, dict):
                for name, m in regions.items():
                    binary = _binary(m)
                    if binary is not None and binary.shape == img.shape[:2]:
                        out["regions"][str(name)] = binary
        except Exception as exc:
            out["reasons"].append("mascaras por region fallidas: %s" % exc)

    if out["person"] is not None and out["coverage"] <= 0.0:
        out["coverage"] = float((out["person"] > 127).mean())
    out["skin"] = _skin_of(img, out["regions"])
    return out


# ----------------------------------------------------- 2/3 background + bokeh

def _alpha_matte(mask: np.ndarray, feather_px: float,
                 guide: np.ndarray | None = None) -> np.ndarray:
    """Soft alpha across the silhouette, following the picture where it can.

    A hard 0/255 mask cuts stair steps into the composite; a blurred mask shifts
    the edge inwards.  The signed distance transform puts a linear ramp exactly
    on the boundary and nowhere else.

    That is still not enough for hair.  The segmentation mask is a smooth blob
    with no idea where individual strands are, so compositing it against a new
    background leaves a chunky cut-out silhouette that anybody notices
    immediately.  A guided filter, with the photograph itself as the guide,
    pulls the alpha back onto the real edges in the image - it is what makes the
    difference between a portrait and a scissors-and-glue collage.
    """
    binary = (mask > 127).astype(np.uint8)
    if int(binary.max()) == 0:
        return np.zeros(mask.shape[:2], np.float32)
    inside = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    outside = cv2.distanceTransform(1 - binary, cv2.DIST_L2, 3)
    signed = inside - outside
    f = float(max(1.0, feather_px))
    alpha = np.clip(0.5 + signed / f, 0.0, 1.0).astype(np.float32)

    if guide is None:
        return alpha
    try:
        radius = int(max(4, round(min(guide.shape[:2]) / 110.0)))
        refined = cv2.ximgproc.guidedFilter(
            guide=guide.astype(np.uint8), src=alpha, radius=radius, eps=1e-4)
        refined = np.clip(refined, 0.0, 1.0).astype(np.float32)
        # Only the boundary band is allowed to move: deep interior and far
        # exterior stay exactly as the segmentation decided.
        band = ((alpha > 0.02) & (alpha < 0.98)).astype(np.float32)
        band = cv2.GaussianBlur(band, (0, 0), max(1.0, f))
        return np.clip(alpha * (1.0 - band) + refined * band, 0.0, 1.0).astype(np.float32)
    except (cv2.error, AttributeError):
        return alpha


def render_silhouette(img: np.ndarray, masks: dict, extra: dict,
                      meta: dict) -> np.ndarray:
    """Replace the person with their own outline.

    A contre-jour silhouette is a real editorial style, and here it does one
    more job: it is the only way this engine can produce a full length frame of
    a subject whose reference photographs are all undressed.  Nothing of the
    original body survives - every pixel inside the outline is painted - so the
    result carries her true proportions and stance and no skin whatsoever.

    It is a stylisation and says so in the metadata: this is not a photograph of
    the person wearing clothes, and nothing here should pretend otherwise.
    """
    style = _key_of(extra.get("treatment"))
    if style != "silhouette":
        return img

    person = masks.get("person")
    if person is None:
        meta["notes"].append(
            "No se pudo aislar la silueta; no se aplica el tratamiento.")
        meta["skipped"].append("silhouette")
        return img

    h, w = img.shape[:2]
    coverage = float((person > 127).mean())
    if coverage < MIN_PERSON_COVERAGE or coverage > MAX_PERSON_COVERAGE:
        meta["notes"].append(
            "La silueta detectada no es utilizable (%.0f%% del encuadre)."
            % (coverage * 100.0))
        meta["skipped"].append("silhouette")
        return img

    # Clean the outline: close pinholes, drop specks, keep the largest shape.
    binary = (person > 127).astype(np.uint8)
    k = max(3, int(round(min(h, w) * 0.012)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        binary = (labels == biggest).astype(np.uint8)

    alpha = _alpha_matte(binary * 255, max(2.0, min(h, w) / 260.0), guide=img)
    alpha = alpha[..., None]

    fill = _as_bgr(extra.get("silhouette_color"), (26, 22, 20))
    body = np.empty_like(img, dtype=np.float32)
    body[:] = np.asarray(fill, dtype=np.float32)

    # A flat cut-out reads as a sticker.  A gentle vertical gradient and a rim
    # of light along the edge give the shape volume, which is what makes a
    # silhouette look photographed rather than pasted.
    ramp = np.linspace(1.18, 0.82, h, dtype=np.float32)[:, None, None]
    body *= ramp
    edge = cv2.dilate(binary, kernel) - cv2.erode(binary, kernel)
    rim = cv2.GaussianBlur(edge.astype(np.float32), (0, 0), max(2.0, k / 1.6))
    if float(rim.max()) > 1e-6:
        rim /= float(rim.max())
    glow = _as_bgr(extra.get("rim_color"), (208, 214, 226))
    body += rim[..., None] * np.asarray(glow, dtype=np.float32) * 0.55

    out = img.astype(np.float32) * (1.0 - alpha) + np.clip(body, 0, 255) * alpha
    meta["steps"].append("silhouette")
    meta["silhouette"] = {"coverage": round(coverage, 4),
                          "fill": [int(v) for v in fill]}
    meta["notes"].append(
        "Silueta a contraluz: la figura se pinta por completo, no se ve piel.")
    return np.clip(out, 0, 255).astype(np.uint8)


def _as_bgr(value, default: tuple) -> tuple:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return tuple(int(max(0, min(255, int(v)))) for v in value[:3])
        except (TypeError, ValueError):
            return default
    return default


def _decontaminate_edge(img: np.ndarray, mask: np.ndarray,
                        band_px: float) -> np.ndarray:
    """Pull the boundary pixels toward the person's local median colour.

    Boundary pixels are mixtures of person and old background.  Composited
    unchanged they leave the classic cut-out halo, so they are replaced by the
    median of the person-side neighbourhood before the matte is applied.
    """
    band = float(max(1.0, band_px))
    core = cv2.erode(mask, _kernel(int(band) * 2 + 3), iterations=1)
    core_f = (core > 127).astype(np.float32)
    if float(core_f.sum()) < 32.0:
        return img
    radius = int(max(3, round(band * 6))) | 1
    imgf = img.astype(np.float32)
    num = cv2.blur(imgf * core_f[..., None], (radius, radius))
    den = cv2.blur(core_f, (radius, radius))[..., None] + 1e-5
    field = num / den
    filled = np.where(core_f[..., None] > 0.5, imgf, field)
    local = cv2.medianBlur(np.clip(filled, 0, 255).astype(np.uint8), 5).astype(np.float32)

    inside = cv2.distanceTransform((mask > 127).astype(np.uint8), cv2.DIST_L2, 3)
    weight = np.clip(1.0 - inside / band, 0.0, 1.0) * (mask > 127).astype(np.float32)
    weight = (weight * 0.85)[..., None]
    out = imgf * (1.0 - weight) + local * weight
    return np.clip(out, 0, 255).astype(np.uint8)


def _estimate_grain(img: np.ndarray, mask) -> float:
    """Robust (MAD) estimate of the photograph's own high frequency noise."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    residual = gray - cv2.GaussianBlur(gray, (0, 0), 1.2)
    if isinstance(mask, np.ndarray) and mask.shape == gray.shape:
        values = residual[mask > 127]
    else:
        values = residual.reshape(-1)
    if values.size < 256:
        values = residual.reshape(-1)
    mad = float(np.median(np.abs(values - float(np.median(values)))))
    return float(_clamp(1.4826 * mad, 0.4, 5.0))


def _render_background(h: int, w: int, scene_key: str, blur: float,
                       grain_sigma: float, seed: int, rng) -> np.ndarray:
    """The scene, defocused and grained so it shares the photo's statistics."""
    bg = scenes.render_scene(scene_key, w, h, seed).astype(np.float32)
    short = float(min(h, w))
    if blur > 0.001:
        sigma = float(_clamp(blur * short * 0.020, 0.6, 48.0))
        bg = cv2.GaussianBlur(bg, (0, 0), sigma)
    if grain_sigma > 0.05:
        noise = rng.standard_normal((h, w), dtype=np.float32) * float(grain_sigma)
        noise = cv2.GaussianBlur(noise, (0, 0), 0.7) * 2.4
        bg = bg + noise[..., None]
    return np.clip(bg, 0, 255)


def replace_background(img: np.ndarray, masks: dict, extra: dict, rng,
                       seed: int, meta: dict) -> np.ndarray:
    """Composite the person over a procedural scene (steps 2 and 3)."""
    scene_key = _key_of(extra.get("scene"))
    if not scene_key:
        return img
    person = masks.get("person")
    if person is None:
        meta["notes"].append(
            "No se pudo aislar a la persona; se conserva el fondo original.")
        meta["skipped"].append("background")
        return img
    coverage = float((person > 127).mean())
    # A portrait subject fills a real share of the frame.  A mask covering a few
    # percent is a mis-detection, and trusting it means treating the other 95% of
    # the photograph as background and blurring it - which produced images that
    # were unrecognisable smears yet still scored well.  The honest response is
    # to keep the original background and say so, not to composite confidently
    # onto a wrong silhouette.
    if coverage < MIN_PERSON_COVERAGE or coverage > MAX_PERSON_COVERAGE:
        meta["notes"].append(
            "La silueta detectada no es utilizable (%.0f%% del encuadre); se "
            "conserva el fondo original." % (coverage * 100.0))
        meta["skipped"].append("background")
        return img

    h, w = img.shape[:2]
    feather = float(_clamp(round(min(h, w) / 400.0), 2.0, 4.0))
    cleaned = _decontaminate_edge(img, person, feather * 1.6)
    alpha = _alpha_matte(person, feather, guide=img)[..., None]
    blur = float(_clamp(_as_float(extra.get("blur_background"), 0.35), 0.0, 1.0))
    grain = _estimate_grain(img, person)
    bg = _render_background(h, w, scene_key, blur, grain, seed, rng)

    out = cleaned.astype(np.float32) * alpha + bg * (1.0 - alpha)
    meta["steps"].append("background")
    meta["scene"] = scenes._canonical(scene_key)
    meta["blur_background"] = round(blur, 3)
    meta["feather_px"] = feather
    meta["grain_sigma"] = round(grain, 2)
    return np.clip(out, 0, 255).astype(np.uint8)


# ------------------------------------------------------ 4 garment and hair

# Named colours are stored as hex and converted through the same path as a
# user supplied hex, so there is one colour parser and no second table.
_COLOR_KEYS = {
    "negro": "#1c1c1e", "black": "#1c1c1e",
    "blanco": "#f2f2f0", "white": "#f2f2f0",
    "gris": "#8b8d90", "gray": "#8b8d90", "grey": "#8b8d90",
    "gris_claro": "#c4c6c9", "light_gray": "#c4c6c9",
    "gris_oscuro": "#4a4c50", "dark_gray": "#4a4c50",
    "rojo": "#b3242c", "red": "#b3242c",
    "vino": "#6d1f2e", "borgona": "#6d1f2e", "burgundy": "#6d1f2e",
    "rosa": "#e2879f", "pink": "#e2879f",
    "rosa_palo": "#e8bfc2", "blush": "#e8bfc2",
    "naranja": "#d4702a", "orange": "#d4702a",
    "coral": "#e0705c",
    "amarillo": "#d9b331", "yellow": "#d9b331",
    "mostaza": "#b8892b", "mustard": "#b8892b",
    "verde": "#2f7d4f", "green": "#2f7d4f",
    "verde_oliva": "#6b7042", "olive": "#6b7042",
    "esmeralda": "#146b5c", "emerald": "#146b5c",
    "menta": "#9fd4b8", "mint": "#9fd4b8",
    "turquesa": "#1f9d9b", "teal": "#1f9d9b",
    "celeste": "#7fb4d8", "sky_blue": "#7fb4d8",
    "azul": "#2b4f9e", "blue": "#2b4f9e",
    "azul_marino": "#1e2a4a", "navy": "#1e2a4a",
    "morado": "#5b3a8e", "purple": "#5b3a8e",
    "lila": "#a892cf", "lilac": "#a892cf", "lavanda": "#a892cf",
    "beige": "#d8c6a8", "crema": "#efe4cf", "cream": "#efe4cf",
    "camel": "#b08857",
    "marron": "#6b4a34", "brown": "#6b4a34", "chocolate": "#4a3225",
    "dorado": "#c9a227", "gold": "#c9a227",
    "plateado": "#b9bdc2", "silver": "#b9bdc2",
}

# Fabric response: (contrast around the region mean, specular gain, sat bias).
_FABRIC = {
    "satin": (0.10, 0.55, 0.04), "seda": (0.10, 0.55, 0.04),
    "silk": (0.10, 0.55, 0.04), "raso": (0.10, 0.55, 0.04),
    "cuero": (0.14, 0.70, -0.02), "leather": (0.14, 0.70, -0.02),
    "terciopelo": (0.06, 0.22, 0.06), "velvet": (0.06, 0.22, 0.06),
    "encaje": (0.04, 0.10, 0.00), "lace": (0.04, 0.10, 0.00),
    "gasa": (-0.06, 0.05, -0.05), "chiffon": (-0.06, 0.05, -0.05),
    "algodon": (-0.02, -0.18, -0.02), "cotton": (-0.02, -0.18, -0.02),
    "lino": (-0.03, -0.14, -0.04), "linen": (-0.03, -0.14, -0.04),
    "punto": (-0.05, -0.22, 0.00), "knit": (-0.05, -0.22, 0.00),
    "lana": (-0.05, -0.22, 0.00), "wool": (-0.05, -0.22, 0.00),
    "denim": (0.02, -0.12, -0.06), "mezclilla": (0.02, -0.12, -0.06),
    "mate": (-0.04, -0.30, -0.03), "matte": (-0.04, -0.30, -0.03),
}

# Hair tones this provider can honestly reach: (hue 0..179, sat scale, val scale).
_HAIR_TONE = {
    "rubio": (22.0, 0.95, 1.45), "blonde": (22.0, 0.95, 1.45),
    "platino": (18.0, 0.25, 1.60), "platinum": (18.0, 0.25, 1.60),
    "miel": (20.0, 1.25, 1.20), "honey": (20.0, 1.25, 1.20),
    "caramelo": (18.0, 1.20, 1.15), "caramel": (18.0, 1.20, 1.15),
    "castano": (14.0, 1.05, 0.92), "brunette": (14.0, 1.05, 0.92),
    "chocolate": (12.0, 1.10, 0.80),
    "moreno": (10.0, 1.00, 0.72),
    "negro": (0.0, 0.45, 0.55), "black": (0.0, 0.45, 0.55),
    "pelirrojo": (7.0, 1.55, 1.02), "rojo": (7.0, 1.55, 1.02),
    "red": (7.0, 1.55, 1.02), "cobrizo": (8.0, 1.50, 1.10),
    "copper": (8.0, 1.50, 1.10),
    "canoso": (0.0, 0.18, 1.25), "gray_hair": (0.0, 0.18, 1.25),
    "gris": (0.0, 0.18, 1.25),
}


def _parse_color(value) -> tuple[float, float, float] | None:
    """Accept a catalog key, a hex string, or an [h,s,v] triple.

    Triples are read as OpenCV units (H 0..179, S/V 0..255); a triple whose
    values are all <= 1 is read as normalised, and an H above 179 is read as
    degrees.  Returns OpenCV HSV or None when nothing usable was given.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            h, s, v = (float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(x) for x in (h, s, v)):
            return None
        if h <= 1.0 and s <= 1.0 and v <= 1.0:
            h, s, v = h * 179.0, s * 255.0, v * 255.0
        elif h > 179.0:
            h = h / 2.0
        if s <= 1.0 and v <= 1.0:
            s, v = s * 255.0, v * 255.0
        return (_clamp(h, 0.0, 179.0), _clamp(s, 0.0, 255.0), _clamp(v, 0.0, 255.0))

    text = str(value).strip()
    if not text:
        return None
    key = _key_of(text)
    hexed = _COLOR_KEYS.get(key)
    if hexed is None:
        candidate = text.lstrip("#")
        if len(candidate) in (3, 6) and all(c in "0123456789abcdefABCDEF" for c in candidate):
            hexed = "#" + candidate
    if hexed is None:
        return None
    bgr = scenes._hex_to_bgr(hexed)
    pixel = np.array([[[bgr[0], bgr[1], bgr[2]]]], np.uint8)
    hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]
    return (float(hsv[0]), float(hsv[1]), float(hsv[2]))


def _feathered(mask: np.ndarray, short_side: float, erode_px: int = 1) -> np.ndarray:
    m = mask
    if erode_px > 0:
        m = cv2.erode(m, _kernel(erode_px * 2 + 1), iterations=1)
    soft = cv2.GaussianBlur(m.astype(np.float32) / 255.0, (0, 0),
                            max(1.0, short_side * 0.002))
    return np.clip(soft, 0.0, 1.0)


def _weighted_mean(channel: np.ndarray, weights: np.ndarray, default: float) -> float:
    total = float(weights.sum())
    if total < 8.0:
        return float(default)
    return float((channel * weights).sum() / total)


def _retint(img: np.ndarray, mask_f: np.ndarray, target_h, s_scale: float,
            v_scale: float, hue_weight: float = 1.0) -> np.ndarray:
    """Hue replacement with multiplicative S/V, so shading and folds survive.

    Hue is interpolated the short way round the colour wheel: a linear blend
    from 175 to 5 would sweep through green.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    m = mask_f
    if target_h is not None:
        delta = np.mod(float(target_h) - hsv[..., 0] + 90.0, 180.0) - 90.0
        hsv[..., 0] = hsv[..., 0] + delta * (m * float(hue_weight))
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + (float(s_scale) - 1.0) * m), 0.0, 255.0)
    hsv[..., 2] = np.clip(hsv[..., 2] * (1.0 + (float(v_scale) - 1.0) * m), 0.0, 255.0)
    hsv[..., 0] = np.clip(np.mod(hsv[..., 0], 180.0), 0.0, 179.0)
    return cv2.cvtColor(np.clip(hsv + 0.5, 0.0, 255.0).astype(np.uint8),
                        cv2.COLOR_HSV2BGR)


def _fabric_response(img: np.ndarray, mask_f: np.ndarray,
                     cfg: tuple[float, float, float]) -> np.ndarray:
    """Make satin read as satin and wool as wool: contrast plus specular."""
    contrast, specular, sat_bias = cfg
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    value = hsv[..., 2]
    mean_v = _weighted_mean(value, mask_f, 128.0)
    shaped = mean_v + (value - mean_v) * (1.0 + float(contrast))
    if abs(specular) > 1e-3:
        highlight = np.clip((value - mean_v) / max(1.0, 255.0 - mean_v), 0.0, 1.0)
        shaped = shaped + float(specular) * (highlight ** 3) * 46.0
    hsv[..., 2] = np.clip(value + (shaped - value) * mask_f, 0.0, 255.0)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + float(sat_bias) * mask_f), 0.0, 255.0)
    return cv2.cvtColor(np.clip(hsv + 0.5, 0.0, 255.0).astype(np.uint8),
                        cv2.COLOR_HSV2BGR)


def _garment_mask(img: np.ndarray, masks: dict) -> np.ndarray | None:
    """Clothing only: never skin, never hair, never the face."""
    regions = masks.get("regions") or {}
    garment = _union([regions.get("upper_body"), regions.get("lower_body")])
    if garment is None:
        try:
            from ..analysis import segment as seg_mod
            candidate = seg_mod.garment_mask(img, masks.get("pose") or {},
                                             masks.get("person"))
            garment = _binary(candidate)
        except Exception:
            garment = None
    if garment is None and masks.get("person") is not None:
        # Last resort: the body below the shoulder line, minus skin.
        garment = masks["person"].copy()
        landmarks = ((masks.get("pose") or {}).get("landmarks") or {})
        ys = [float(landmarks[n]["y"]) for n in ("left_shoulder", "right_shoulder")
              if isinstance(landmarks.get(n), dict)]
        if ys:
            cut = int(_clamp(min(ys) * img.shape[0], 0.0, float(img.shape[0] - 1)))
            garment[:cut, :] = 0
    if garment is None:
        return None

    garment = _subtract(garment, masks.get("skin"), grow=2)
    garment = _subtract(garment, (masks.get("regions") or {}).get("hair"), grow=1)
    garment = _subtract(garment, (masks.get("regions") or {}).get("face"), grow=2)
    garment = cv2.morphologyEx(garment, cv2.MORPH_OPEN, _kernel(5))

    total = float(img.shape[0] * img.shape[1])
    count, labels, stats, _ = cv2.connectedComponentsWithStats(garment, 8)
    cleaned = np.zeros_like(garment)
    for i in range(1, count):
        if float(stats[i, cv2.CC_STAT_AREA]) >= 0.0015 * total:
            cleaned[labels == i] = 255
    return cleaned if int(cleaned.max()) > 0 else None


def recolor_garment(img: np.ndarray, masks: dict, extra: dict,
                    meta: dict) -> np.ndarray:
    """Step 4: retarget the fabric colour while keeping its luminance."""
    color = extra.get("garment_color")
    style = _key_of(extra.get("garment_style"))
    if color is None and not style:
        return img
    target = _parse_color(color)
    if target is None and color is not None:
        meta["notes"].append("No se reconocio el color de prenda solicitado.")
    fabric = None
    if style:
        for fragment, cfg in _FABRIC.items():
            if fragment in style:
                fabric = cfg
                break
        if fabric is None:
            meta["unsupported"].append("garment_style:%s" % style)
    if target is None and fabric is None:
        return img

    garment = _garment_mask(img, masks)
    if garment is None:
        meta["notes"].append(
            "No se pudo aislar la ropa; no se cambio el color de la prenda.")
        meta["skipped"].append("garment")
        return img

    short = float(min(img.shape[:2]))
    mask_f = _feathered(garment, short, erode_px=1)
    out = img
    if target is not None:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        cur_s = _weighted_mean(hsv[..., 1], mask_f, 80.0)
        cur_v = _weighted_mean(hsv[..., 2], mask_f, 128.0)
        s_scale = _clamp(target[1] / max(8.0, cur_s), 0.05, 3.0)
        raw_v = target[2] / max(8.0, cur_v)
        v_scale = _clamp(raw_v, 0.72, 1.38)
        if abs(raw_v - v_scale) > 0.02:
            meta["notes"].append(
                "El tono pedido para la prenda es muy distinto al original; "
                "se aplico el maximo que conserva la textura de la tela.")
        hue_weight = 1.0 if target[1] > 28.0 else 0.35
        out = _retint(out, mask_f, target[0], s_scale, v_scale, hue_weight)
        meta["steps"].append("garment_color")
        meta["garment_hsv"] = [round(v, 1) for v in target]
    if fabric is not None:
        out = _fabric_response(out, mask_f, fabric)
        meta["steps"].append("garment_style")
    meta["garment_px"] = int((garment > 127).sum())
    return out


def adjust_hair(img: np.ndarray, masks: dict, extra: dict,
                meta: dict) -> np.ndarray:
    """Hair colour is a real photographic change; hair *shape* is not.

    Anything structural (length, updo, waves) is reported as unsupported
    instead of being faked.
    """
    key = _key_of(extra.get("hair"))
    if not key:
        return img
    tone = None
    for fragment, cfg in _HAIR_TONE.items():
        if fragment in key:
            tone = cfg
            break
    if tone is None:
        meta["unsupported"].append("hair:%s" % key)
        meta["notes"].append(
            "El peinado no se puede cambiar sin IA generativa; se conserva el original.")
        return img
    hair = (masks.get("regions") or {}).get("hair")
    if hair is None or int(hair.max()) == 0:
        meta["skipped"].append("hair")
        meta["notes"].append("No se pudo aislar el cabello; no se cambio el tono.")
        return img
    mask_f = _feathered(hair, float(min(img.shape[:2])), erode_px=1) * 0.85
    out = _retint(img, mask_f, tone[0], tone[1], tone[2], hue_weight=0.8)
    meta["steps"].append("hair_tone")
    return out


# ------------------------------------------------------------- 5 relighting

# amp is the multiplicative swing on Lab L; tint/warm are the a/b shifts that
# come with it.  Everything is deliberately small: a relight that moves skin
# far enough to fail the delta-E check in identity/verify.py is a bug, not a
# style.
_LIGHTING = {
    "window_left": {"amp": 0.13, "tint": 0.4, "warm": 1.6, "kind": "linear",
                    "angle": 180.0},
    "window_right": {"amp": 0.13, "tint": 0.4, "warm": 1.6, "kind": "linear",
                     "angle": 0.0},
    "golden_hour": {"amp": 0.15, "tint": 1.2, "warm": 2.6, "kind": "diagonal",
                    "angle": 205.0},
    "softbox_front": {"amp": 0.10, "tint": 0.0, "warm": 0.4, "kind": "radial"},
    "rim_backlight": {"amp": 0.16, "tint": -0.3, "warm": -0.8, "kind": "rim"},
    "overcast": {"amp": 0.07, "tint": -0.2, "warm": -1.2, "kind": "flat"},
}
_LIGHT_ALIASES = {
    "ventana_izquierda": "window_left", "ventana_derecha": "window_right",
    "hora_dorada": "golden_hour", "frontal": "softbox_front",
    "softbox": "softbox_front", "contraluz": "rim_backlight",
    "nublado": "overcast", "difusa": "overcast",
}


def _light_field(h: int, w: int, cfg: dict, masks: dict) -> np.ndarray:
    """Signed illumination map in -1..1; +1 is the lit side."""
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32).reshape(1, w)
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1)
    kind = cfg.get("kind", "flat")

    if kind in ("linear", "diagonal"):
        angle = math.radians(float(cfg.get("angle", 0.0)))
        t = xs * math.cos(angle) + ys * math.sin(angle)
        t = np.broadcast_to(np.asarray(t, np.float32), (h, w)).astype(np.float32)
        lo, hi = float(t.min()), float(t.max())
        t = (t - lo) / max(1e-6, hi - lo)
        field = (t - 0.42) * 2.1
        if kind == "diagonal":
            field = field + (0.5 - ys) * 0.5
    elif kind == "radial":
        cx, cy = 0.5, 0.42
        bbox = _mask_bbox(masks.get("person"))
        if bbox:
            cx = (bbox[0] + bbox[2] * 0.5) / float(max(1, w))
            cy = (bbox[1] + bbox[3] * 0.35) / float(max(1, h))
        aspect = float(w) / float(max(1, h))
        d = np.sqrt(((xs - cx) * aspect) ** 2 + (ys - cy) ** 2) / 0.85
        field = np.clip(1.0 - d, 0.0, 1.0) * 2.0 - 0.85
    elif kind == "rim":
        person = masks.get("person")
        if person is not None:
            grow = _kernel(max(3, int(min(h, w) * 0.012)) | 1)
            band = cv2.subtract(cv2.dilate(person, grow), cv2.erode(person, grow))
            soft = cv2.GaussianBlur(band.astype(np.float32) / 255.0, (0, 0),
                                    max(2.0, min(h, w) * 0.008))
            field = np.clip(soft * 2.4, 0.0, 1.6) - 0.30
        else:
            t = np.broadcast_to(ys, (h, w)).astype(np.float32)
            field = (0.45 - t) * 1.6
    else:
        field = np.broadcast_to((0.5 - ys) * 0.7, (h, w)).astype(np.float32)

    field = np.asarray(np.broadcast_to(field, (h, w)), np.float32)
    return np.clip(field, -1.0, 1.0)


def relight(img: np.ndarray, masks: dict, extra: dict, meta: dict) -> np.ndarray:
    """Step 5: directional light applied to Lab L, with a matching a/b shift."""
    key = _key_of(extra.get("lighting"))
    key = _LIGHT_ALIASES.get(key, key)
    cfg = _LIGHTING.get(key)
    if cfg is None:
        if key:
            meta["unsupported"].append("lighting:%s" % key)
        return img

    h, w = img.shape[:2]
    field = _light_field(h, w, cfg, masks)
    skin = masks.get("skin")
    if isinstance(skin, np.ndarray) and skin.shape == (h, w):
        damp = np.where(skin > 127, np.float32(0.5), np.float32(1.0))
        field = field * damp
    field = cv2.GaussianBlur(field, (0, 0), max(1.5, min(h, w) * 0.006))

    lab = cv2.cvtColor(img.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    gain = np.clip(1.0 + float(cfg["amp"]) * field, 0.86, 1.16)
    lab[..., 0] = np.clip(lab[..., 0] * gain, 0.0, 100.0)
    lab[..., 1] = np.clip(lab[..., 1] + float(cfg["tint"]) * field, -127.0, 127.0)
    lab[..., 2] = np.clip(lab[..., 2] + float(cfg["warm"]) * field, -127.0, 127.0)
    out = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR) * 255.0
    meta["steps"].append("relight")
    meta["lighting"] = key
    return np.clip(out, 0, 255).astype(np.uint8)


# ------------------------------------------------------------ 6 colour grade

# lift/gamma/gain are per channel in BGR order; contrast is an S curve applied
# before them; sat scales HSV S afterwards.  All of it collapses into one 256
# entry LUT per channel, so the cost is independent of resolution.
_GRADES = {
    "neutral_studio": {
        "lift": (0.000, 0.000, 0.000), "gamma": (1.00, 1.00, 1.00),
        "gain": (1.00, 1.00, 1.00), "contrast": 0.08, "sat": 1.02, "mono": False},
    "warm_film": {
        "lift": (0.018, 0.020, 0.035), "gamma": (1.02, 1.00, 0.98),
        "gain": (0.94, 0.99, 1.04), "contrast": 0.12, "sat": 0.96, "mono": False},
    "cool_editorial": {
        "lift": (0.030, 0.016, 0.006), "gamma": (0.98, 1.00, 1.02),
        "gain": (1.03, 1.00, 0.97), "contrast": 0.20, "sat": 0.92, "mono": False},
    "cinematic_teal_orange": {
        "lift": (0.045, 0.030, 0.000), "gamma": (0.97, 1.00, 1.03),
        "gain": (0.90, 0.98, 1.08), "contrast": 0.18, "sat": 1.06, "mono": False},
    "soft_pastel": {
        "lift": (0.055, 0.052, 0.050), "gamma": (1.06, 1.05, 1.04),
        "gain": (0.96, 0.98, 1.00), "contrast": -0.12, "sat": 0.86, "mono": False},
    "editorial_bw": {
        "lift": (0.012, 0.012, 0.012), "gamma": (1.00, 1.00, 1.00),
        "gain": (1.00, 1.00, 1.00), "contrast": 0.26, "sat": 0.0, "mono": True,
        "weights": (0.17, 0.45, 0.38)},
}
_GRADE_ALIASES = {
    "bw": "editorial_bw", "blanco_y_negro": "editorial_bw",
    "byn": "editorial_bw", "monocromo": "editorial_bw",
    "calido": "warm_film", "film": "warm_film",
    "frio": "cool_editorial", "editorial": "cool_editorial",
    "cine": "cinematic_teal_orange", "cinematic": "cinematic_teal_orange",
    "pastel": "soft_pastel", "suave": "soft_pastel",
    "neutro": "neutral_studio", "estudio": "neutral_studio",
}
_LUT_CACHE: dict[str, np.ndarray] = {}


def _s_curve(x: np.ndarray, k: float) -> np.ndarray:
    if abs(k) < 1e-4:
        return x
    smooth = x * x * (3.0 - 2.0 * x)
    return x + float(k) * (smooth - x)


def _grade_lut(key: str, cfg: dict) -> np.ndarray:
    cached = _LUT_CACHE.get(key)
    if cached is not None:
        return cached
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    shaped = _s_curve(x, float(cfg["contrast"]))
    lut = np.zeros((1, 256, 3), np.uint8)
    for c in range(3):
        y = shaped * float(cfg["gain"][c]) + float(cfg["lift"][c]) * (1.0 - shaped)
        y = np.clip(y, 0.0, 1.0) ** (1.0 / max(0.2, float(cfg["gamma"][c])))
        lut[0, :, c] = np.clip(y * 255.0 + 0.5, 0, 255).astype(np.uint8)
    _LUT_CACHE[key] = lut
    return lut


def color_grade(img: np.ndarray, extra: dict, meta: dict) -> np.ndarray:
    """Step 6: a real grade - tone curve plus per channel lift/gamma/gain."""
    key = _key_of(extra.get("grade"))
    key = _GRADE_ALIASES.get(key, key)
    cfg = _GRADES.get(key)
    if cfg is None:
        if key:
            meta["unsupported"].append("grade:%s" % key)
        return img

    base = img
    if cfg.get("mono"):
        weights = cfg.get("weights", (0.114, 0.587, 0.299))
        gray = (img[..., 0].astype(np.float32) * weights[0] +
                img[..., 1].astype(np.float32) * weights[1] +
                img[..., 2].astype(np.float32) * weights[2])
        gray8 = np.clip(gray + 0.5, 0, 255).astype(np.uint8)
        base = cv2.cvtColor(gray8, cv2.COLOR_GRAY2BGR)

    out = cv2.LUT(base, _grade_lut(key, cfg))
    sat = float(cfg.get("sat", 1.0))
    if not cfg.get("mono") and abs(sat - 1.0) > 1e-3:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0.0, 255.0)
        out = cv2.cvtColor(np.clip(hsv + 0.5, 0.0, 255.0).astype(np.uint8),
                           cv2.COLOR_HSV2BGR)
    meta["steps"].append("grade")
    meta["grade"] = key
    return out


# ----------------------------------------------------------------- 7 framing

def _subject_geometry(masks: dict, w: int, h: int) -> dict:
    """Where the head, torso and feet are, in pixels.

    Every value is filled in: when a landmark is missing the mask bbox answers,
    and when there is no mask the frame answers.  A crop must never depend on
    MediaPipe having been installed.
    """
    pose = masks.get("pose") or {}
    landmarks = pose.get("landmarks") if isinstance(pose.get("landmarks"), dict) else {}
    bbox = _mask_bbox(masks.get("person"))

    def point(name: str, vmin: float = 0.3):
        data = landmarks.get(name)
        if not isinstance(data, dict):
            return None
        try:
            if float(data.get("v", 0.0)) < vmin:
                return None
            return (float(data["x"]) * w, float(data["y"]) * h)
        except (TypeError, ValueError, KeyError):
            return None

    eyes = [p for p in (point("left_eye"), point("right_eye")) if p]
    ears = [p for p in (point("left_ear"), point("right_ear")) if p]
    nose = point("nose")
    shoulders = [p for p in (point("left_shoulder"), point("right_shoulder")) if p]
    hips = [p for p in (point("left_hip"), point("right_hip")) if p]
    ankles = [p for p in (point("left_ankle", 0.25), point("right_ankle", 0.25)) if p]

    eye_y = (sum(p[1] for p in eyes) / len(eyes)) if eyes else (nose[1] if nose else None)
    eye_x = (sum(p[0] for p in eyes) / len(eyes)) if eyes else (nose[0] if nose else None)
    shoulder_y = (sum(p[1] for p in shoulders) / len(shoulders)) if shoulders else None
    shoulder_x = (sum(p[0] for p in shoulders) / len(shoulders)) if shoulders else None
    hip_y = (sum(p[1] for p in hips) / len(hips)) if hips else None
    ankle_y = max(p[1] for p in ankles) if ankles else None

    head_h = None
    if len(ears) == 2:
        span = math.hypot(ears[0][0] - ears[1][0], ears[0][1] - ears[1][1])
        if span > 4.0:
            head_h = span * 1.42
    if head_h is None and eye_y is not None and shoulder_y is not None and shoulder_y > eye_y:
        head_h = (shoulder_y - eye_y) / 0.85       # eye line sits ~0.85 heads up
    if head_h is None and bbox:
        head_h = bbox[3] * 0.17
    if head_h is None:
        head_h = h * 0.16
    head_h = float(_clamp(head_h, h * 0.03, h * 0.90))

    if eye_y is not None:
        head_top = eye_y - 0.52 * head_h
    elif bbox:
        head_top = float(bbox[1])
    else:
        head_top = h * 0.05

    if len(shoulders) == 2:
        shoulder_span = abs(shoulders[0][0] - shoulders[1][0]) * 1.55
        body_x = float(shoulder_x)
    elif bbox:
        shoulder_span = float(bbox[2])
        body_x = bbox[0] + bbox[2] * 0.5
    else:
        shoulder_span = w * 0.70
        body_x = w * 0.5
    shoulder_span = float(max(shoulder_span, head_h * 1.4))

    return {
        "head_top": float(head_top),
        "chin": float(head_top + head_h),
        "head_h": head_h,
        "shoulder_y": shoulder_y,
        "hip_y": hip_y,
        "ankle_y": ankle_y,
        "shoulder_span": shoulder_span,
        "body_x": float(body_x),
        "face_x": float(eye_x if eye_x is not None else body_x),
        "bbox": bbox,
    }


def _crop_rect(img: np.ndarray, masks: dict, framing: str,
               aspect: float) -> list[float]:
    """A crop window that respects head room and keeps the subject whole."""
    h, w = img.shape[:2]
    geo = _subject_geometry(masks, w, h)
    head_top, chin, head_h = geo["head_top"], geo["chin"], geo["head_h"]
    hip_y, ankle_y, bbox = geo["hip_y"], geo["ankle_y"], geo["bbox"]
    shoulder_y = geo["shoulder_y"]
    torso = (hip_y - shoulder_y) if (hip_y is not None and shoulder_y is not None
                                     and hip_y > shoulder_y) else head_h * 2.2

    if framing == "portrait_headshot":
        # Head and shoulders only.  The bottom edge sits just below the collar
        # bone, so the frame contains the face, the hair and the top of the
        # shoulders and nothing else - the framing a profile picture wants, and
        # the one that is safe to produce whatever the source photograph shows.
        y0 = head_top - 0.35 * head_h
        y1 = chin + 0.72 * head_h
        need = head_h * 1.25
        cx = 0.88 * geo["face_x"] + 0.12 * geo["body_x"]
    elif framing == "portrait_closeup":
        y0 = head_top - 0.30 * head_h
        y1 = chin + 2.05 * head_h
        need = head_h * 1.50
        cx = 0.75 * geo["face_x"] + 0.25 * geo["body_x"]
    elif framing == "portrait_half":
        y0 = head_top - 0.38 * head_h
        y1 = (hip_y + 0.60 * torso) if hip_y is not None else (chin + 5.2 * head_h)
        need = max(geo["shoulder_span"] * 1.28, head_h * 2.2)
        cx = 0.45 * geo["face_x"] + 0.55 * geo["body_x"]
    elif framing == "portrait_full":
        y0 = head_top - 0.48 * head_h
        if ankle_y is not None:
            y1 = ankle_y + 0.42 * head_h
        elif bbox:
            y1 = bbox[1] + bbox[3] + 0.15 * head_h
        else:
            y1 = float(h)
        need = (float(bbox[2]) if bbox else geo["shoulder_span"] * 1.8) * 1.12
        cx = (bbox[0] + bbox[2] * 0.5) if bbox else geo["body_x"]
    else:
        y0 = head_top - 0.42 * head_h
        if ankle_y is not None:
            y1 = ankle_y + 0.35 * head_h
        elif bbox:
            y1 = bbox[1] + bbox[3] + 0.10 * head_h
        else:
            y1 = float(h)
        need = (float(bbox[2]) if bbox else geo["shoulder_span"] * 1.6) * 1.10
        cx = (bbox[0] + bbox[2] * 0.5) if bbox else geo["body_x"]

    rh = float(max(48.0, y1 - y0))
    rw = rh * float(aspect)
    if rw < need:
        rw = float(need)
        rh = rw / float(aspect)          # grow downward: head room is fixed
    return [float(cx - rw * 0.5), float(y0), rw, rh]


def _extend_canvas(img: np.ndarray, top: int, bottom: int, left: int,
                   right: int) -> np.ndarray:
    """Replicate and soften the border so a crop can breathe past the frame.

    Replication (not reflection) on purpose: a mirrored strip can duplicate an
    arm or an ear, and the anomaly scanner would be right to flag that.
    """
    out = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE)
    short = float(min(out.shape[:2]))
    keep = np.zeros(out.shape[:2], np.float32)
    keep[top:top + img.shape[0], left:left + img.shape[1]] = 1.0
    keep = cv2.GaussianBlur(keep, (0, 0), max(1.5, short * 0.006))[..., None]
    soft = cv2.GaussianBlur(out, (0, 0), max(2.0, short * 0.014)).astype(np.float32)
    blended = out.astype(np.float32) * keep + soft * (1.0 - keep)
    return np.clip(blended, 0, 255).astype(np.uint8)


def _apply_rect(img: np.ndarray, rect, aspect: float, meta: dict) -> np.ndarray:
    h, w = img.shape[:2]
    cap_x = int(round(w * 0.15))
    cap_y = int(round(h * 0.15))
    x0, y0, rw, rh = (float(v) for v in rect)
    cx = x0 + rw * 0.5

    shrink = min(1.0, (w + 2.0 * cap_x) / max(1.0, rw), (h + 2.0 * cap_y) / max(1.0, rh))
    if shrink < 1.0:
        rw *= shrink
        rh *= shrink
        x0 = cx - rw * 0.5               # vertical anchor stays: keep the head
        meta["notes"].append(
            "El encuadre pedido no cabe en la foto original; se ajusto al maximo posible.")

    x0 = min(max(x0, -cap_x), w + cap_x - rw)
    y0 = min(max(y0, -cap_y), h + cap_y - rh)
    ix0, iy0 = int(math.floor(x0)), int(math.floor(y0))
    irw, irh = max(16, int(round(rw))), max(16, int(round(rh)))

    pad_l, pad_t = max(0, -ix0), max(0, -iy0)
    pad_r, pad_b = max(0, ix0 + irw - w), max(0, iy0 + irh - h)
    if pad_l or pad_t or pad_r or pad_b:
        img = _extend_canvas(img, pad_t, pad_b, pad_l, pad_r)
        ix0 += pad_l
        iy0 += pad_t
        meta["steps"].append("canvas_extend")

    ph, pw = img.shape[:2]
    irw = min(irw, pw)
    irh = min(irh, ph)
    irh = min(max(16, int(round(irw / float(aspect)))), ph)
    irw = min(max(16, int(round(irh * float(aspect)))), pw)
    ix0 = max(0, min(ix0, pw - irw))
    iy0 = max(0, min(iy0, ph - irh))
    return img[iy0:iy0 + irh, ix0:ix0 + irw]


def _target_size(req: GenRequest, aspect, w: int, h: int) -> tuple[int, int]:
    max_side = int(_QUALITY_MAX_SIDE.get(_key_of(req.quality), _DEFAULT_MAX_SIDE))
    max_side = int(min(max_side, MAX_SIDE))
    rw = int(req.width or 0)
    rh = int(req.height or 0)
    ratio = float(aspect) if aspect else (w / float(max(1, h)))
    if rw > 0 and rh > 0:
        return _clamp_dim(rw), _clamp_dim(rh)
    if rw > 0:
        return _clamp_dim(rw), _clamp_dim(rw / ratio)
    if rh > 0:
        return _clamp_dim(rh * ratio), _clamp_dim(rh)
    longest = min(max_side, max(w, h))
    if ratio >= 1.0:
        return _clamp_dim(longest), _clamp_dim(longest / ratio)
    return _clamp_dim(longest * ratio), _clamp_dim(longest)


def frame_image(img: np.ndarray, masks: dict, extra: dict, req: GenRequest,
                meta: dict) -> np.ndarray:
    """Step 7: crop to the requested framing, then resize to the tier size."""
    h, w = img.shape[:2]
    framing = _key_of(extra.get("framing"))
    aspect = _FRAMING_ASPECT.get(framing)
    if framing and aspect is None:
        meta["unsupported"].append("framing:%s" % framing)
    tw, th = _target_size(req, aspect, w, h)
    if aspect is not None:
        img = _apply_rect(img, _crop_rect(img, masks, framing, aspect), aspect, meta)
        meta["steps"].append("framing")
        meta["framing"] = framing
    out = _resize_exact(img, tw, th)
    meta["size"] = [int(tw), int(th)]
    if max(tw, th) >= max(w, h) and not (req.width or req.height):
        meta["notes"].append(
            "La salida se limito al tamano de la foto original para no inventar detalle.")
    return out


# --------------------------------------------------------------- 8 finishing

def finish(img: np.ndarray, extra: dict, meta: dict) -> np.ndarray:
    vignette = _as_float(extra.get("vignette"), 0.0)
    if vignette > 0.01:
        img = _vignette(img, _clamp(vignette, 0.0, 0.75))
        meta["steps"].append("vignette")
        meta["vignette"] = round(float(vignette), 3)
    sigma = max(0.8, float(min(img.shape[:2])) * 0.0012)
    out = _unsharp(img, amount=0.32, sigma=sigma, threshold=0.02)
    meta["steps"].append("unsharp")
    return out


# ------------------------------------------------------------------- repair

_MAX_REGIONS = 24
_SEAMLESS_MAX_PX = 2_500_000


def _telea_fill(img: np.ndarray, region: np.ndarray, area: float) -> np.ndarray:
    radius = int(_clamp(round(math.sqrt(max(1.0, area)) / 2.0), 3.0, 15.0))
    grown = cv2.dilate(region, _kernel(3), iterations=1)
    return cv2.inpaint(img, grown, radius, cv2.INPAINT_TELEA)


def _patch_fill(img: np.ndarray, region: np.ndarray,
                box: tuple[int, int, int, int]):
    """Structure from Telea, texture from the best mirrored neighbourhood.

    A large hole filled by diffusion alone comes out plastic.  Mirroring the
    surrounding content gives it the same grain and weave as its neighbours;
    the mirror is chosen by matching the known pixels, so it is deterministic
    and it is real content from this photograph, not invented detail.
    """
    h, w = img.shape[:2]
    x, y, bw, bh = box
    margin = int(max(8, round(0.6 * max(bw, bh))))
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1, y1 = min(w, x + bw + margin), min(h, y + bh + margin)
    roi = img[y0:y1, x0:x1]
    cut = region[y0:y1, x0:x1]
    if roi.size == 0 or int(cut.max()) == 0:
        return None, None

    area = float((cut > 127).sum())
    base = _telea_fill(roi, cut, area)
    hole = cut > 127
    known = ~hole
    best = None
    best_err = float("inf")
    best_flip = 1
    for flip in (1, 0, -1):
        candidate = cv2.flip(roi, flip)
        donor_known = cv2.flip(known.astype(np.uint8), flip) > 0
        valid = known & donor_known
        if int(valid.sum()) < 64:
            continue
        diff = roi[valid].astype(np.float32) - candidate[valid].astype(np.float32)
        err = float(np.mean(diff * diff))
        if err < best_err:
            best_err, best, best_flip = err, candidate, flip
    if best is None:
        return base, (x0, y0, x1, y1)

    sigma = max(1.2, 0.02 * max(bw, bh))
    texture = best.astype(np.float32) - cv2.GaussianBlur(best, (0, 0), sigma).astype(np.float32)
    donor_hole = cv2.flip(hole.astype(np.uint8), best_flip) > 0
    texture[donor_hole] = 0.0
    filled = np.clip(base.astype(np.float32) + texture * 0.85, 0, 255).astype(np.uint8)
    return filled, (x0, y0, x1, y1)


def _seamless_merge(dst: np.ndarray, patch: np.ndarray, box, region: np.ndarray) -> None:
    """Poisson blend the repaired patch back in, so the seam disappears."""
    x0, y0, x1, y1 = box
    target = dst[y0:y1, x0:x1]
    cut = region[y0:y1, x0:x1]
    pad = 6
    merged = None
    if target.size and target.size // 3 <= _SEAMLESS_MAX_PX:
        src_p = cv2.copyMakeBorder(patch, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        dst_p = cv2.copyMakeBorder(target, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        mask_p = cv2.copyMakeBorder(cut, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        centre = (mask_p.shape[1] // 2, mask_p.shape[0] // 2)
        try:
            cloned = cv2.seamlessClone(src_p, dst_p, mask_p, centre, cv2.NORMAL_CLONE)
            merged = cloned[pad:pad + target.shape[0], pad:pad + target.shape[1]]
        except Exception:
            merged = None       # Poisson solve refused: fall back to feathering
    if merged is None:
        alpha = cv2.GaussianBlur(cut.astype(np.float32) / 255.0, (0, 0), 2.0)[..., None]
        merged = np.clip(target.astype(np.float32) * (1.0 - alpha) +
                         patch.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    dst[y0:y1, x0:x1] = merged


def _load_repair_mask(path, shape) -> np.ndarray | None:
    try:
        raw = _load_source(path, 0)
    except Exception:
        return None
    if raw.ndim == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    if raw.shape[:2] != shape:
        raw = cv2.resize(raw, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return _binary(raw)


def _work_side(req: GenRequest) -> int:
    """Work above the output size so the crop still has resolution to give."""
    tier = int(_QUALITY_MAX_SIDE.get(_key_of(req.quality), _DEFAULT_MAX_SIDE))
    want = max(int(req.width or 0), int(req.height or 0), tier)
    return int(min(WORK_MAX_SIDE, max(768, want * 2)))


def _new_meta() -> dict:
    return {"steps": [], "skipped": [], "unsupported": [], "notes": []}


class LocalFreeProvider(ImageProvider):
    """Zero cost, zero key, no generative model - photographic transformation."""

    name = PROVIDER_NAME

    # ------------------------------------------------------------ metadata

    def capabilities(self) -> Capabilities:
        return Capabilities(
            name=PROVIDER_NAME,
            kind="image",
            img2img=True,
            inpaint=True,
            upscale=True,
            text2img=False,          # it cannot invent a person, and never will
            identity_reference=False,
            # It composites the real photograph and invents nothing: the router
            # uses this to keep clothing/pose/expression/hair requests away.
            generative=False,
            max_side=MAX_SIDE,
            needs_key=False,
            key_name="",
            cost_per_image_usd=0.0,
            notes=NOTES_ES,
        )

    def available(self) -> bool:
        return True

    def estimate_cost(self, req: GenRequest) -> float:
        return 0.0

    def _fail(self, message: str, model: str, started: float,
              meta: dict | None = None) -> GenResult:
        return GenResult(
            ok=False, image_path=None, provider=self.name, model=model,
            cost_usd=0.0, latency_ms=int((time.perf_counter() - started) * 1000),
            error=str(message), meta=meta or {})

    # ------------------------------------------------------------ generate

    def generate(self, req: GenRequest, out_path: str | Path) -> GenResult:
        # A caller that set the operation instead of picking the method gets
        # what it asked for rather than a full retransform of a repair crop.
        if req.operation == "inpaint" and str(req.mask_path or "").strip():
            return self.inpaint(req, out_path)
        if req.operation == "upscale":
            return self.upscale(req, out_path)

        started = time.perf_counter()
        meta = _new_meta()
        source = str(req.source_path or "").strip()
        if not source:
            return self._fail("El proveedor local necesita una foto de origen",
                              MODEL_GENERATE, started, meta)

        seed = _resolve_seed(req)
        rng = np.random.default_rng(seed)
        extra = dict(req.extra or {})

        try:
            img = _load_source(source, _work_side(req))
        except Exception as exc:
            return self._fail("No se pudo abrir la foto de origen: %s" % exc,
                              MODEL_GENERATE, started, meta)

        try:
            masks = _detect_masks(img)
            meta["mask_backend"] = masks.get("backend") or "none"
            meta["coverage"] = round(float(masks.get("coverage") or 0.0), 4)
            if masks.get("reasons"):
                meta["analysis_notes"] = list(masks["reasons"])[:4]

            img = replace_background(img, masks, extra, rng, seed, meta)
            img = render_silhouette(img, masks, extra, meta)
            img = recolor_garment(img, masks, extra, meta)
            img = adjust_hair(img, masks, extra, meta)
            img = relight(img, masks, extra, meta)
            img = color_grade(img, extra, meta)
            img = frame_image(img, masks, extra, req, meta)
            img = finish(img, extra, meta)
        except Exception as exc:
            return self._fail("Fallo la transformacion local: %s" % exc,
                              MODEL_GENERATE, started, meta)

        if _key_of(extra.get("expression")):
            meta["unsupported"].append("expression:%s" % _key_of(extra.get("expression")))
            meta["notes"].append(
                "La expresion facial no se puede cambiar sin IA generativa; "
                "se conserva la expresion real de la foto.")

        try:
            written = _save(img, out_path, 92)
        except Exception as exc:
            return self._fail("No se pudo guardar la imagen: %s" % exc,
                              MODEL_GENERATE, started, meta)

        meta["engine"] = "opencv"
        meta["generative"] = False
        return GenResult(
            ok=True, image_path=written, provider=self.name, model=MODEL_GENERATE,
            cost_usd=0.0, latency_ms=int((time.perf_counter() - started) * 1000),
            seed=seed, meta=meta)

    # ------------------------------------------------------------- inpaint

    def inpaint(self, req: GenRequest, out_path: str | Path) -> GenResult:
        """Repaint only where the mask is white - a repair, not a regeneration.

        Small defects go to Telea diffusion; larger ones get a patch based fill
        sampled from mirrored surrounding content and Poisson blended back in.
        The final feather blend guarantees that every pixel outside the mask is
        the input pixel, unchanged.
        """
        started = time.perf_counter()
        meta = _new_meta()
        source = str(req.source_path or "").strip()
        if not source:
            return self._fail("El proveedor local necesita una foto de origen",
                              MODEL_INPAINT, started, meta)
        if not str(req.mask_path or "").strip():
            return self._fail("El proveedor local necesita una mascara para reparar",
                              MODEL_INPAINT, started, meta)

        try:
            original = _load_source(source, 0)
        except Exception as exc:
            return self._fail("No se pudo abrir la foto a reparar: %s" % exc,
                              MODEL_INPAINT, started, meta)

        mask = _load_repair_mask(req.mask_path, original.shape[:2])
        if mask is None:
            return self._fail("No se pudo leer la mascara de reparacion",
                              MODEL_INPAINT, started, meta)
        area_total = float((mask > 127).sum())
        if area_total < 1.0:
            return self._fail("La mascara de reparacion esta vacia",
                              MODEL_INPAINT, started, meta)

        h, w = original.shape[:2]
        try:
            result = original.copy()
            count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            order = sorted(range(1, count),
                           key=lambda i: int(stats[i, cv2.CC_STAT_AREA]),
                           reverse=True)[:_MAX_REGIONS]
            small_limit = max(256.0, 0.0025 * float(h * w))
            methods = {"telea": 0, "patch": 0}
            repaired_px = 0.0

            for index in order:
                area = float(stats[index, cv2.CC_STAT_AREA])
                region = (labels == index).astype(np.uint8) * 255
                if area <= small_limit:
                    result = _telea_fill(result, region, area)
                    methods["telea"] += 1
                else:
                    box = (int(stats[index, cv2.CC_STAT_LEFT]),
                           int(stats[index, cv2.CC_STAT_TOP]),
                           int(stats[index, cv2.CC_STAT_WIDTH]),
                           int(stats[index, cv2.CC_STAT_HEIGHT]))
                    patch, roi = _patch_fill(result, region, box)
                    if patch is None or roi is None:
                        result = _telea_fill(result, region, area)
                        methods["telea"] += 1
                    else:
                        _seamless_merge(result, patch, roi, region)
                        methods["patch"] += 1
                repaired_px += area

            # Feather the repair back over the original: alpha is exactly zero
            # outside the mask, so untouched pixels are the input pixels.
            feather = max(1.0, float(min(h, w)) * 0.0025)
            alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), feather)
            alpha[alpha < 0.004] = 0.0
            alpha = np.clip(alpha, 0.0, 1.0)[..., None]
            blended = (original.astype(np.float32) * (1.0 - alpha) +
                       result.astype(np.float32) * alpha)
            out = np.clip(blended, 0, 255).astype(np.uint8)
            untouched = alpha[..., 0] <= 0.0
            out[untouched] = original[untouched]
        except Exception as exc:
            return self._fail("Fallo la reparacion local: %s" % exc,
                              MODEL_INPAINT, started, meta)

        try:
            written = _save(out, out_path, 95)
        except Exception as exc:
            return self._fail("No se pudo guardar la reparacion: %s" % exc,
                              MODEL_INPAINT, started, meta)

        meta["steps"].append("inpaint")
        meta["regions"] = len(order)
        meta["methods"] = methods
        meta["mask_px"] = int(area_total)
        meta["area_px"] = int(repaired_px)
        meta["area_ratio"] = round(repaired_px / float(max(1, h * w)), 5)
        meta["engine"] = "opencv"
        meta["generative"] = False
        return GenResult(
            ok=True, image_path=written, provider=self.name, model=MODEL_INPAINT,
            cost_usd=0.0, latency_ms=int((time.perf_counter() - started) * 1000),
            seed=_resolve_seed(req), meta=meta)

    # ------------------------------------------------------------- upscale

    def upscale(self, req: GenRequest, out_path: str | Path) -> GenResult:
        started = time.perf_counter()
        meta = _new_meta()
        source = str(req.source_path or "").strip()
        if not source:
            return self._fail("El proveedor local necesita una foto de origen",
                              MODEL_UPSCALE, started, meta)
        try:
            img = _load_source(source, 0)
        except Exception as exc:
            return self._fail("No se pudo abrir la foto a ampliar: %s" % exc,
                              MODEL_UPSCALE, started, meta)

        h, w = img.shape[:2]
        extra = dict(req.extra or {})
        if int(req.width or 0) > 0 and int(req.height or 0) > 0:
            tw, th = _clamp_dim(req.width), _clamp_dim(req.height)
        else:
            factor = _clamp(_as_float(extra.get("scale"), 2.0), 1.0, 4.0)
            tw, th = _clamp_dim(w * factor), _clamp_dim(h * factor)
        longest = max(tw, th)
        if longest > MAX_SIDE:
            ratio = MAX_SIDE / float(longest)
            tw, th = _clamp_dim(tw * ratio), _clamp_dim(th * ratio)

        try:
            up = cv2.resize(img, (tw, th), interpolation=cv2.INTER_LANCZOS4)
            denoised = cv2.bilateralFilter(up, 5, 18, 7)
            sigma = max(0.9, float(min(th, tw)) * 0.0007)
            out = _unsharp(denoised, amount=0.55, sigma=sigma, threshold=0.015)
            written = _save(out, out_path, 95)
        except Exception as exc:
            return self._fail("Fallo la ampliacion local: %s" % exc,
                              MODEL_UPSCALE, started, meta)

        meta["steps"] = ["lanczos", "bilateral", "unsharp"]
        meta["from"] = [int(w), int(h)]
        meta["to"] = [int(tw), int(th)]
        meta["factor"] = round(float(max(tw, th)) / float(max(1, max(w, h))), 3)
        meta["engine"] = "opencv"
        meta["generative"] = False
        return GenResult(
            ok=True, image_path=written, provider=self.name, model=MODEL_UPSCALE,
            cost_usd=0.0, latency_ms=int((time.perf_counter() - started) * 1000),
            seed=_resolve_seed(req), meta=meta)

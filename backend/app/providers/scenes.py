"""Procedural backgrounds for the free local provider.

The local provider composites a real person over a new background, so it needs
backgrounds.  Shipping a folder of stock photographs would mean licensing,
disk space and a download step on the client machine; instead every backdrop
here is synthesised from numpy noise, gradients and blurred blobs, at whatever
size the caller asks for and deterministically from a seed - the same key plus
the same seed always yields the same pixels, which is what makes a run
reproducible.  A studio that prefers its own backdrop drops
``data/scenes/<key>.jpg`` next to the others and that file wins.
"""
from __future__ import annotations

import math
import zlib

import cv2
import numpy as np

from ..config import SCENE_DIR

__all__ = ["SCENES", "render_scene", "scene_keys"]

_MIN_DIM = 8
_MAX_DIM = 8192
_USER_EXT = (".jpg", ".jpeg", ".png", ".webp")

# label_es/label_en feed the catalog, kind groups them in the UI, and palette
# is both the swatch shown to the user and the actual source of the colours the
# builder paints with - so the two can never drift apart.
SCENES: dict[str, dict] = {
    "studio_gray": {
        "label_es": "Estudio gris", "label_en": "Gray studio sweep",
        "kind": "studio", "palette": ["#8f9296", "#c9ccd1", "#4e5155"]},
    "studio_warm_gray": {
        "label_es": "Estudio gris calido", "label_en": "Warm gray studio sweep",
        "kind": "studio", "palette": ["#9a9088", "#d6cabd", "#57504a"]},
    "studio_blue": {
        "label_es": "Estudio azul", "label_en": "Blue studio sweep",
        "kind": "studio", "palette": ["#3f5c7a", "#7fa6c8", "#1d2c3d"]},
    "studio_terracotta": {
        "label_es": "Estudio terracota", "label_en": "Terracotta studio sweep",
        "kind": "studio", "palette": ["#a9614a", "#d69a7f", "#5d3226"]},
    "white_cyclorama": {
        "label_es": "Ciclorama blanco", "label_en": "White cyclorama",
        "kind": "studio", "palette": ["#f2f2f0", "#ffffff", "#d5d5d3"]},
    "gradient_peach": {
        "label_es": "Degradado durazno", "label_en": "Peach gradient backdrop",
        "kind": "gradient", "palette": ["#f7c6a5", "#ef9f86", "#b96a63"]},
    "gradient_indigo": {
        "label_es": "Degradado indigo", "label_en": "Indigo gradient backdrop",
        "kind": "gradient", "palette": ["#2b3a67", "#4d5fa8", "#8f6fb0"]},
    "bokeh_warm": {
        "label_es": "Luces calidas desenfocadas", "label_en": "Warm bokeh lights",
        "kind": "bokeh", "palette": ["#2a1c12", "#ffb45c", "#ff7b3a", "#ffe0a8"]},
    "city_night": {
        "label_es": "Ciudad de noche", "label_en": "City night bokeh",
        "kind": "bokeh", "palette": ["#0d1526", "#ffc766", "#5fd4ff", "#ff6b8a"]},
    "beach_haze": {
        "label_es": "Playa con neblina", "label_en": "Beach haze",
        "kind": "outdoor", "palette": ["#a8cfe0", "#e8dcc4", "#d8bf98"]},
    "forest_bokeh": {
        "label_es": "Bosque desenfocado", "label_en": "Forest bokeh",
        "kind": "outdoor", "palette": ["#1d2a17", "#5f7f3a", "#c8d98a"]},
    "concrete_wall": {
        "label_es": "Muro de concreto", "label_en": "Concrete wall",
        "kind": "wall", "palette": ["#9b9a96", "#b8b7b2", "#6d6c69"]},
    "brick_wall": {
        "label_es": "Muro de ladrillo", "label_en": "Brick wall",
        "kind": "wall", "palette": ["#8c4a34", "#a9614a", "#cfc4b4"]},
    "marble_interior": {
        "label_es": "Interior de marmol", "label_en": "Marble interior",
        "kind": "interior", "palette": ["#ece9e2", "#b9b4a9", "#d8cfc0"]},
    "golden_hour_sky": {
        "label_es": "Cielo de hora dorada", "label_en": "Golden hour sky",
        "kind": "sky", "palette": ["#2f4f7a", "#f0a95c", "#ffd9a0"]},
    "dark_moody": {
        "label_es": "Fondo oscuro dramatico", "label_en": "Dark moody backdrop",
        "kind": "studio", "palette": ["#14161a", "#4a5058", "#0a0b0d"]},
    "textured_canvas": {
        "label_es": "Lienzo texturizado", "label_en": "Textured canvas backdrop",
        "kind": "studio", "palette": ["#8a8377", "#b5aea0", "#4f4a42"]},
    "window_light_interior": {
        "label_es": "Interior con luz de ventana", "label_en": "Window light interior",
        "kind": "interior", "palette": ["#b9b3aa", "#eef2f6", "#6c665e"]},
}

# Tolerated spellings coming from the catalog or from a hand written brief.
_ALIASES = {
    "studio": "studio_gray", "estudio": "studio_gray", "estudio_gris": "studio_gray",
    "gris": "studio_gray", "gray": "studio_gray", "grey": "studio_gray",
    "estudio_calido": "studio_warm_gray", "warm_gray": "studio_warm_gray",
    "azul": "studio_blue", "estudio_azul": "studio_blue",
    "terracota": "studio_terracotta", "terracotta": "studio_terracotta",
    "blanco": "white_cyclorama", "white": "white_cyclorama",
    "ciclorama": "white_cyclorama", "fondo_blanco": "white_cyclorama",
    "degradado": "gradient_peach", "gradient": "gradient_peach",
    "durazno": "gradient_peach", "indigo": "gradient_indigo",
    "bokeh": "bokeh_warm", "luces": "bokeh_warm", "luces_calidas": "bokeh_warm",
    "ciudad": "city_night", "ciudad_noche": "city_night", "night": "city_night",
    "playa": "beach_haze", "beach": "beach_haze",
    "bosque": "forest_bokeh", "forest": "forest_bokeh",
    "concreto": "concrete_wall", "cemento": "concrete_wall", "concrete": "concrete_wall",
    "ladrillo": "brick_wall", "brick": "brick_wall",
    "marmol": "marble_interior", "marble": "marble_interior",
    "hora_dorada": "golden_hour_sky", "atardecer": "golden_hour_sky",
    "golden_hour": "golden_hour_sky", "cielo": "golden_hour_sky",
    "oscuro": "dark_moody", "dramatico": "dark_moody", "moody": "dark_moody",
    "lienzo": "textured_canvas", "canvas": "textured_canvas",
    "ventana": "window_light_interior", "interior": "window_light_interior",
    "window": "window_light_interior",
}

_FALLBACK = "studio_gray"


# ------------------------------------------------------------------- helpers

def _dim(value) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        v = 0
    return int(min(max(v, _MIN_DIM), _MAX_DIM))


def _rng(key: str, seed) -> np.random.Generator:
    """Key and seed both enter the state, so two scenes never share noise."""
    digest = zlib.crc32(str(key).encode("utf-8")) & 0xFFFFFFFF
    try:
        s = int(seed) & 0xFFFFFFFF
    except (TypeError, ValueError):
        s = 0
    return np.random.default_rng((digest << 32) | s)


def _hex_to_bgr(text) -> np.ndarray:
    raw = str(text).strip().lstrip("#")
    if len(raw) == 3:
        raw = "".join(c * 2 for c in raw)
    if len(raw) != 6:
        return np.array([128.0, 128.0, 128.0], np.float32)
    try:
        r = int(raw[0:2], 16)
        g = int(raw[2:4], 16)
        b = int(raw[4:6], 16)
    except ValueError:
        return np.array([128.0, 128.0, 128.0], np.float32)
    return np.array([float(b), float(g), float(r)], np.float32)


def _palette(key: str) -> list[np.ndarray]:
    entry = SCENES.get(key) or {}
    values = entry.get("palette") or ["#808080", "#b0b0b0", "#505050"]
    pal = [_hex_to_bgr(v) for v in values]
    while len(pal) < 4:
        pal.append(pal[-1])
    return pal


def _grid(h: int, w: int):
    """Broadcastable normalised coordinates: xs is (1,w), ys is (h,1)."""
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32).reshape(1, w)
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1)
    return xs, ys


def _ramp(h: int, w: int, angle_deg: float) -> np.ndarray:
    xs, ys = _grid(h, w)
    a = math.radians(float(angle_deg))
    t = np.asarray(xs * math.cos(a) + ys * math.sin(a), dtype=np.float32)
    t = np.broadcast_to(t, (h, w)).astype(np.float32)
    lo = float(t.min())
    hi = float(t.max())
    return (t - lo) / max(1e-6, hi - lo)


def _radial(h: int, w: int, cx: float, cy: float, radius: float,
            power: float = 1.0) -> np.ndarray:
    xs, ys = _grid(h, w)
    aspect = float(w) / float(max(1, h))
    dx = (xs - float(cx)) * aspect
    dy = ys - float(cy)
    d = np.sqrt(dx * dx + dy * dy) / max(1e-6, float(radius))
    out = np.clip(1.0 - d, 0.0, 1.0) ** float(power)
    return np.broadcast_to(out, (h, w)).astype(np.float32)


def _noise(rng, h: int, w: int, cells_y: int, cells_x: int = 0) -> np.ndarray:
    """One octave: a coarse random grid stretched up with cubic interpolation."""
    cy = int(max(2, cells_y))
    cx = int(cells_x) if cells_x and int(cells_x) >= 2 else int(
        max(2, round(cy * w / float(max(1, h)))))
    cx = int(min(cx, 4096))
    small = rng.random((cy, cx), dtype=np.float32)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _fbm(rng, h: int, w: int, octaves: int = 4, base: int = 3) -> np.ndarray:
    acc = np.zeros((h, w), np.float32)
    amp = 1.0
    total = 0.0
    cells = int(max(2, base))
    for _ in range(int(max(1, octaves))):
        acc += _noise(rng, h, w, cells) * amp
        total += amp
        amp *= 0.5
        cells = int(min(cells * 2, 512))
    return acc / max(1e-6, total)


def _mix(c0, c1, t) -> np.ndarray:
    a = np.asarray(c0, np.float32)
    b = np.asarray(c1, np.float32)
    tt = np.asarray(t, np.float32)[..., None]
    return a[None, None, :] + (b - a)[None, None, :] * tt


def _soft_rect(h: int, w: int, x0: float, x1: float, y0: float, y1: float,
               soft: float, shear: float = 0.0) -> np.ndarray:
    xs, ys = _grid(h, w)
    s = max(1e-3, float(soft))
    xx = xs + (ys - 0.5) * float(shear)
    fx = np.clip((xx - x0) / s, 0.0, 1.0) * np.clip((x1 - xx) / s, 0.0, 1.0)
    fy = np.clip((ys - y0) / s, 0.0, 1.0) * np.clip((y1 - ys) / s, 0.0, 1.0)
    t = np.asarray(fx * fy, np.float32)
    t = np.broadcast_to(t, (h, w)).astype(np.float32)
    return t * t * (3.0 - 2.0 * t)


def _bokeh(rng, h: int, w: int, colors, count: int, r_min: float, r_max: float,
           intensity: float) -> np.ndarray:
    """Out of focus highlights: soft discs with a slightly brighter rim."""
    layer = np.zeros((h, w, 3), np.float32)
    n = int(max(0, min(int(count), 160)))
    if n == 0 or not colors:
        return layer
    xs = rng.random(n)
    ys = rng.random(n)
    rs = rng.random(n)
    amps = 0.45 + 0.55 * rng.random(n)
    idx = rng.integers(0, len(colors), n)
    short = float(min(h, w))
    for i in range(n):
        r = int(round((r_min + (r_max - r_min) * float(rs[i]) ** 1.7) * short))
        if r < 2:
            continue
        cx = int(float(xs[i]) * w)
        cy = int(float(ys[i]) * h)
        col = np.asarray(colors[int(idx[i])], np.float32) * float(amps[i]) * float(intensity)
        cv2.circle(layer, (cx, cy), r, tuple(float(v) for v in col), -1, cv2.LINE_8)
        cv2.circle(layer, (cx, cy), max(2, int(r * 0.93)),
                   tuple(float(v) * 0.35 for v in col),
                   max(1, int(r * 0.12)), cv2.LINE_8)
    return cv2.GaussianBlur(layer, (0, 0), max(1.0, short * 0.006))


def _grain(rng, img: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.01:
        return img
    n = rng.standard_normal((img.shape[0], img.shape[1]), dtype=np.float32)
    n = cv2.GaussianBlur(n, (0, 0), 0.7)
    return img + n[..., None] * float(amount) * 2.4


def _vignette(img: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.001:
        return img
    h, w = img.shape[:2]
    r = _radial(h, w, 0.5, 0.5, 0.95, 1.0)
    return img * (1.0 - float(amount) * (1.0 - r))[..., None]


def _finalize(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.clip(img, 0.0, 255.0).astype(np.uint8))


# ------------------------------------------------------------------ builders
# Every builder takes (rng, h, w, palette) and returns float32 BGR in 0..255.

def _sweep(rng, h, w, pal, glow_y=0.40, glow_r=0.95, glow_k=0.55,
           mottle=3.0, vig=0.18, grain=1.1):
    """A seamless studio sweep: paper gradient plus one soft key light pool."""
    top, glow, floor = pal[0], pal[1], pal[2]
    t = _ramp(h, w, 90.0)
    img = _mix(top, floor, np.clip(t * 1.15 - 0.08, 0.0, 1.0) ** 1.1)
    g = _radial(h, w, 0.5, glow_y, glow_r, 1.7) * float(glow_k)
    img = img + (glow[None, None, :] - img) * g[..., None]
    img = img + (_fbm(rng, h, w, 3, 3) - 0.5)[..., None] * (2.0 * float(mottle))
    return _grain(rng, _vignette(img, vig), grain)


def _cyclorama(rng, h, w, pal):
    wall, hot, floor = pal[0], pal[1], pal[2]
    t = _ramp(h, w, 90.0)
    curve = np.clip((t - 0.72) / 0.28, 0.0, 1.0) ** 1.6
    img = _mix(wall, floor, curve * 0.55)
    g = _radial(h, w, 0.5, 0.34, 1.05, 1.4) * 0.5
    img = img + (hot[None, None, :] - img) * g[..., None]
    shadow = _radial(h, w, 0.5, 0.985, 0.42, 1.2)
    img = img * (1.0 - 0.16 * shadow)[..., None]
    img = img + (_fbm(rng, h, w, 2, 2) - 0.5)[..., None] * 2.0
    return _grain(rng, _vignette(img, 0.08), 0.9)


def _gradient(rng, h, w, pal, angle=115.0):
    """Three stop gradient.  The grain is not decoration: without dithering a
    smooth ramp bands badly once it is saved as JPEG."""
    t = _ramp(h, w, angle)
    lo = np.clip(t / 0.5, 0.0, 1.0)
    hi = np.clip((t - 0.5) / 0.5, 0.0, 1.0)
    img = _mix(pal[0], pal[1], lo)
    img = img + (pal[2][None, None, :] - img) * hi[..., None]
    img = img + (_fbm(rng, h, w, 2, 2) - 0.5)[..., None] * 3.0
    return _grain(rng, _vignette(img, 0.10), 1.6)


def _bokeh_warm(rng, h, w, pal):
    base = _mix(pal[0], pal[0] * 1.45, _radial(h, w, 0.55, 0.35, 1.2, 1.3))
    lights = _bokeh(rng, h, w, [pal[1], pal[2], pal[3]], 46, 0.05, 0.20, 0.85)
    return _grain(rng, _vignette(base + lights, 0.34), 1.4)


def _city_night(rng, h, w, pal):
    t = _ramp(h, w, 90.0)
    img = _mix(pal[0] * 1.6, pal[0] * 0.55, t)
    blocks = np.zeros((h, w, 3), np.float32)
    n = 70
    xs = rng.random(n)
    ys = rng.random(n) * 0.8 + 0.05
    ws = 0.006 + 0.010 * rng.random(n)
    hs = 0.010 + 0.018 * rng.random(n)
    idx = rng.integers(1, 4, n)
    amps = 0.35 + 0.50 * rng.random(n)
    for i in range(n):
        x0 = int(float(xs[i]) * w)
        y0 = int(float(ys[i]) * h)
        x1 = x0 + max(2, int(float(ws[i]) * w))
        y1 = y0 + max(2, int(float(hs[i]) * h))
        col = np.asarray(pal[int(idx[i])], np.float32) * float(amps[i])
        cv2.rectangle(blocks, (x0, y0), (x1, y1),
                      tuple(float(v) for v in col), -1, cv2.LINE_8)
    blocks = cv2.GaussianBlur(blocks, (0, 0), max(2.0, min(h, w) * 0.010))
    img = img + blocks * 0.8
    img = img + _bokeh(rng, h, w, [pal[1], pal[2], pal[3]], 38, 0.03, 0.13, 0.95)
    return _grain(rng, _vignette(img, 0.42), 1.8)


def _beach(rng, h, w, pal):
    sky, haze, sand = pal[0], pal[1], pal[2]
    t = _ramp(h, w, 90.0)
    horizon = 0.58
    img = _mix(sky, haze, np.clip(t / horizon, 0.0, 1.0) ** 0.8)
    below = np.clip((t - horizon) / (1.0 - horizon), 0.0, 1.0) ** 0.7
    img = img + (sand[None, None, :] - img) * below[..., None]
    glow = _radial(h, w, 0.68, horizon - 0.03, 0.55, 1.5)
    sun = np.array([210.0, 235.0, 255.0], np.float32)
    img = img + (sun[None, None, :] - img) * (0.45 * glow)[..., None]
    img = img + (_noise(rng, h, w, 4, 14) - 0.5)[..., None] * 8.0
    img = cv2.GaussianBlur(img, (0, 0), max(1.5, min(h, w) * 0.012))
    return _grain(rng, _vignette(img, 0.16), 1.2)


def _forest(rng, h, w, pal):
    dark, leaf, light = pal[0], pal[1], pal[2]
    img = _mix(dark, leaf, _fbm(rng, h, w, 3, 3))
    trunks = np.zeros((h, w), np.float32)
    for _ in range(int(rng.integers(3, 6))):
        x = int(float(rng.random()) * w)
        tw = max(3, int((0.02 + 0.05 * float(rng.random())) * w))
        cv2.rectangle(trunks, (x, 0), (x + tw, h), 1.0, -1, cv2.LINE_8)
    trunks = cv2.GaussianBlur(trunks, (0, 0), max(3.0, min(h, w) * 0.02))
    img = img * (1.0 - 0.45 * trunks)[..., None]
    img = img + _bokeh(rng, h, w, [light, leaf * 1.4], 40, 0.02, 0.10, 0.75)
    return _grain(rng, _vignette(img, 0.30), 1.3)


def _concrete(rng, h, w, pal):
    base, high, low = pal[0], pal[1], pal[2]
    img = _mix(base, high, _fbm(rng, h, w, 5, 4))
    stain = _fbm(rng, h, w, 3, 2) ** 2.2
    img = img + (low[None, None, :] - img) * (0.40 * stain)[..., None]
    speck = (rng.random((h, w), dtype=np.float32) - 0.5) * 10.0
    img = img + cv2.GaussianBlur(speck, (0, 0), 0.6)[..., None]
    img = img * (0.86 + 0.24 * _ramp(h, w, 200.0))[..., None]
    return _grain(rng, _vignette(img, 0.22), 1.0)


def _brick(rng, h, w, pal):
    """Running bond courses with mortar joints and per brick colour jitter."""
    brick, brick2, mortar = pal[0], pal[1], pal[2]
    bh = max(6.0, h / 15.0)
    bw = bh * 2.45
    mt = max(1.0, bh * 0.13)
    ys = np.arange(h, dtype=np.float32).reshape(h, 1)
    xs = np.arange(w, dtype=np.float32).reshape(1, w)
    row = np.floor(ys / bh)
    offset = np.where(np.mod(row, 2.0) > 0.5, bw * 0.5, 0.0)
    xoff = xs + offset
    col = np.floor(xoff / bw)
    in_mortar = (np.mod(ys, bh) < mt) | (np.mod(xoff, bw) < mt)
    in_mortar = np.broadcast_to(in_mortar, (h, w))
    nr, nc = 48, 48
    jitter = rng.normal(0.0, 1.0, (nr, nc)).astype(np.float32)
    ri = np.broadcast_to(np.mod(row, nr).astype(np.int32), (h, w))
    ci = np.broadcast_to(np.mod(col, nc).astype(np.int32), (h, w))
    tone = np.clip(0.5 + 0.22 * jitter[ri, ci], 0.0, 1.0)
    img = _mix(brick, brick2, tone)
    img = np.where(in_mortar[..., None], mortar[None, None, :], img)
    img = img + (_fbm(rng, h, w, 4, 6) - 0.5)[..., None] * 16.0
    img = cv2.GaussianBlur(img, (0, 0), max(0.7, min(h, w) * 0.0016))
    img = img * (0.88 + 0.20 * _ramp(h, w, 160.0))[..., None]
    return _grain(rng, _vignette(img, 0.24), 1.2)


def _marble(rng, h, w, pal):
    base, vein, warm = pal[0], pal[1], pal[2]
    n = _fbm(rng, h, w, 4, 3)
    xs, ys = _grid(h, w)
    t = (xs * 2.6 + ys * 1.1 + n * 2.4) * math.pi * 2.0
    veins = np.clip(1.0 - np.abs(np.sin(t)) * 4.0, 0.0, 1.0) ** 1.4
    fine = np.clip(1.0 - np.abs(np.sin(t * 2.7 + n * 3.0)) * 8.0, 0.0, 1.0) * 0.5
    img = _mix(base, warm, n * 0.45)
    strength = np.clip(veins + fine, 0.0, 1.0) * 0.75
    img = img + (vein[None, None, :] - img) * strength[..., None]
    img = cv2.GaussianBlur(img, (0, 0), max(0.8, min(h, w) * 0.002))
    img = img * (0.90 + 0.18 * _radial(h, w, 0.45, 0.30, 1.3, 1.2))[..., None]
    return _grain(rng, _vignette(img, 0.14), 0.9)


def _golden_sky(rng, h, w, pal):
    deep, gold, pale = pal[0], pal[1], pal[2]
    t = _ramp(h, w, 90.0)
    img = _mix(deep, gold, np.clip(t * 1.25, 0.0, 1.0) ** 1.5)
    sun = _radial(h, w, 0.66, 0.74, 0.85, 2.0)
    img = img + (pale[None, None, :] * 1.15 - img) * (0.75 * sun)[..., None]
    clouds = _noise(rng, h, w, 5, 22)
    band = np.clip(1.0 - np.abs(t - 0.52) / 0.30, 0.0, 1.0)
    cl = np.clip((clouds - 0.48) * 2.4, 0.0, 1.0) * band
    img = img + (pale[None, None, :] - img) * (0.55 * cl)[..., None]
    img = cv2.GaussianBlur(img, (0, 0), max(1.2, min(h, w) * 0.006))
    return _grain(rng, _vignette(img, 0.20), 1.3)


def _dark_moody(rng, h, w, pal):
    base, pool, deep = pal[0], pal[1], pal[2]
    img = _mix(deep, base, _ramp(h, w, 90.0) * 0.5 + 0.25)
    g = _radial(h, w, 0.38, 0.34, 0.95, 2.2)
    img = img + (pool[None, None, :] - img) * (0.70 * g)[..., None]
    img = img + (_fbm(rng, h, w, 3, 3) - 0.5)[..., None] * 5.0
    return _grain(rng, _vignette(img, 0.55), 1.6)


def _canvas(rng, h, w, pal):
    base, light, dark = pal[0], pal[1], pal[2]
    mottle = _fbm(rng, h, w, 4, 3)
    img = _mix(dark, light, np.clip(mottle * 1.2 - 0.05, 0.0, 1.0))
    img = img + (base[None, None, :] - img) * 0.35
    period = max(3.0, min(h, w) / 260.0)
    xs, ys = _grid(h, w)
    weave = (np.sin(xs * w / period * math.pi * 2.0) +
             np.sin(ys * h / period * math.pi * 2.0))
    img = img + np.asarray(weave, np.float32)[..., None] * 2.6
    img = img * (0.86 + 0.26 * _radial(h, w, 0.5, 0.42, 1.15, 1.3))[..., None]
    return _grain(rng, _vignette(img, 0.30), 1.4)


def _window_interior(rng, h, w, pal):
    wall, light, shade = pal[0], pal[1], pal[2]
    img = _mix(wall, shade, _ramp(h, w, 0.0) ** 1.2)
    pane_a = _soft_rect(h, w, -0.05, 0.34, 0.05, 0.62, 0.10, shear=0.10)
    pane_b = _soft_rect(h, w, 0.38, 0.62, 0.12, 0.58, 0.09, shear=0.10)
    pool = np.clip(pane_a + pane_b * 0.65, 0.0, 1.3)
    pool = cv2.GaussianBlur(pool, (0, 0), max(2.0, min(h, w) * 0.018))
    img = img + (light[None, None, :] * 1.05 - img) * (0.55 * pool)[..., None]
    bounce = _radial(h, w, 0.30, 0.95, 0.70, 1.4)
    img = img + (light[None, None, :] - img) * (0.14 * bounce)[..., None]
    img = img + (_fbm(rng, h, w, 3, 3) - 0.5)[..., None] * 4.0
    return _grain(rng, _vignette(img, 0.26), 1.1)


_BUILDERS = {
    "studio_gray": lambda rng, h, w, pal: _sweep(rng, h, w, pal),
    "studio_warm_gray": lambda rng, h, w, pal: _sweep(
        rng, h, w, pal, glow_y=0.36, glow_k=0.60, vig=0.20),
    "studio_blue": lambda rng, h, w, pal: _sweep(
        rng, h, w, pal, glow_y=0.42, glow_k=0.50, vig=0.26, mottle=2.4),
    "studio_terracotta": lambda rng, h, w, pal: _sweep(
        rng, h, w, pal, glow_y=0.38, glow_r=1.05, glow_k=0.58, vig=0.22),
    "white_cyclorama": _cyclorama,
    "gradient_peach": lambda rng, h, w, pal: _gradient(rng, h, w, pal, 115.0),
    "gradient_indigo": lambda rng, h, w, pal: _gradient(rng, h, w, pal, 65.0),
    "bokeh_warm": _bokeh_warm,
    "city_night": _city_night,
    "beach_haze": _beach,
    "forest_bokeh": _forest,
    "concrete_wall": _concrete,
    "brick_wall": _brick,
    "marble_interior": _marble,
    "golden_hour_sky": _golden_sky,
    "dark_moody": _dark_moody,
    "textured_canvas": _canvas,
    "window_light_interior": _window_interior,
}


# -------------------------------------------------------------- user scenes

def _canonical(key) -> str:
    name = str(key or "").strip().lower().replace(" ", "_").replace("-", "_")
    if name in SCENES:
        return name
    return _ALIASES.get(name, _FALLBACK)


def _cover(img: np.ndarray, w: int, h: int) -> np.ndarray:
    """Scale to cover the frame, then centre crop - never distorts."""
    ih, iw = img.shape[:2]
    scale = max(w / float(max(1, iw)), h / float(max(1, ih)))
    nw = max(w, int(math.ceil(iw * scale)))
    nh = max(h, int(math.ceil(ih * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    x = (nw - w) // 2
    y = (nh - h) // 2
    out = resized[y:y + h, x:x + w]
    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(out[:, :, :3])


def _user_scene(key: str, w: int, h: int) -> np.ndarray | None:
    """A file the studio dropped in data/scenes wins over the procedural one."""
    for ext in _USER_EXT:
        path = SCENE_DIR / f"{key}{ext}"
        try:
            if not path.is_file():
                continue
            from ..analysis import loader
            img = loader.load_image(path)
        except Exception:
            continue
        if isinstance(img, np.ndarray) and img.size:
            try:
                return _cover(img, w, h)
            except Exception:
                continue
    return None


# ------------------------------------------------------------------- public

def scene_keys() -> list[str]:
    return list(SCENES.keys())


def render_scene(key: str, width: int, height: int, seed: int = 0) -> np.ndarray:
    """BGR uint8 backdrop of exactly (height, width).

    Never raises: an unknown key falls back to the gray studio sweep and a
    builder that trips over a degenerate size falls back to the same sweep, so
    a background is always available to composite against.
    """
    w = _dim(width)
    h = _dim(height)
    name = _canonical(key)
    user = _user_scene(name, w, h)
    if user is not None:
        return user
    builder = _BUILDERS.get(name) or _BUILDERS[_FALLBACK]
    try:
        img = builder(_rng(name, seed), h, w, _palette(name))
    except Exception:
        img = _sweep(_rng(_FALLBACK, seed), h, w, _palette(_FALLBACK))
    return _finalize(img)

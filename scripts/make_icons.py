"""Generate the PWA icons and the favicon.

Kept as a script rather than checked-in binaries so the branding can be changed
by editing three constants instead of opening an image editor.
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
ICON_DIR = ROOT / "frontend" / "icons"

BG_TOP = (28, 26, 38)
BG_BOTTOM = (58, 44, 74)
ACCENT = (232, 168, 124)
ACCENT_SOFT = (246, 214, 186)


def _rounded_gradient(size: int) -> Image.Image:
    img = Image.new("RGB", (size, size), BG_TOP)
    px = img.load()
    for y in range(size):
        t = y / max(1, size - 1)
        row = tuple(int(BG_TOP[i] + (BG_BOTTOM[i] - BG_TOP[i]) * t) for i in range(3))
        for x in range(size):
            px[x, y] = row
    return img


def _aperture(draw: ImageDraw.ImageDraw, size: int) -> None:
    """A camera aperture: six blades around a lens - reads at 48px."""
    cx = cy = size / 2
    r_out = size * 0.30
    r_in = size * 0.115
    width = max(2, int(size * 0.035))

    draw.ellipse([cx - r_out, cy - r_out, cx + r_out, cy + r_out],
                 outline=ACCENT, width=width)

    for i in range(6):
        a0 = math.radians(i * 60 - 20)
        a1 = math.radians(i * 60 + 40)
        draw.line(
            [(cx + r_in * math.cos(a0), cy + r_in * math.sin(a0)),
             (cx + r_out * math.cos(a1), cy + r_out * math.sin(a1))],
            fill=ACCENT_SOFT, width=max(2, int(width * 0.75)),
        )

    draw.ellipse([cx - r_in, cy - r_in, cx + r_in, cy + r_in], fill=ACCENT)
    # highlight so the lens does not read as a flat dot
    hr = r_in * 0.42
    draw.ellipse([cx - r_in * 0.55 - hr, cy - r_in * 0.55 - hr,
                  cx - r_in * 0.55 + hr, cy - r_in * 0.55 + hr],
                 fill=(255, 246, 238))


def make_icon(size: int) -> Image.Image:
    scale = 4
    big = _rounded_gradient(size * scale)
    draw = ImageDraw.Draw(big)
    _aperture(draw, size * scale)
    return big.resize((size, size), Image.LANCZOS)


def main() -> None:
    ICON_DIR.mkdir(parents=True, exist_ok=True)
    for size in (192, 512, 180):
        img = make_icon(size)
        name = "apple-touch-icon.png" if size == 180 else f"icon-{size}.png"
        img.save(ICON_DIR / name, "PNG")
        print(f"wrote {ICON_DIR / name}")

    fav = make_icon(64)
    fav.save(ICON_DIR / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48), (64, 64)])
    print(f"wrote {ICON_DIR / 'favicon.ico'}")


if __name__ == "__main__":
    main()

"""Calibration harness for the identity gate.

The client's headline complaint was that a previous tool made her slimmer without
being asked, and she never found out until she looked at the picture.  This
system claims to catch that automatically.  This script proves the claim -- or
proves it false -- with numbers instead of confidence.

Method:
  1. Build an identity profile from a set of real photographs.
  2. Hold out photographs the profile did not learn from.
  3. Synthesise the exact failure mode: compress the body horizontally toward
     its own midline by a known percentage, leaving the head untouched, which is
     what "beautifying" image tools actually do.
  4. Run the real verification gate over originals and over the slimmed copies.

Two numbers matter and they pull against each other:
    DETECTION  - slimmed images that the gate rejects.       Higher is better.
    FALSE ALARM- untouched images that the gate rejects.     Lower is better.

A gate that rejects everything scores 100% detection and is worthless.  Both
numbers are reported at every slimming level, always.

Usage:
    python scripts\\calibrate_identity.py [--dir <folder>] [--holdout 6]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import cv2                                                    # noqa: E402
import numpy as np                                            # noqa: E402

from app.analysis import loader, pose, segment                # noqa: E402
from app.identity import profile as profile_mod               # noqa: E402
from app.identity import verify as verify_mod                 # noqa: E402

DEFAULT_DIR = ROOT / "input" / "Nayane"
SLIM_LEVELS = (0.94, 0.88, 0.82)          # 6%, 12%, 18% narrower


# --------------------------------------------------------------- synthesis

def slim_body(img_bgr: np.ndarray, factor: float) -> np.ndarray | None:
    """Compress the body horizontally toward its midline by ``factor``.

    The head is deliberately left alone: real beautifying filters narrow the
    body and keep the face recognisable, which is precisely why a face-only
    identity check fails to notice.
    """
    h, w = img_bgr.shape[:2]
    po = pose.detect_pose(img_bgr)
    lm = po.get("landmarks") or {}
    if not lm:
        return None

    def pt(name):
        p = lm.get(name)
        if not p or p.get("v", 0) < 0.3:
            return None
        return np.array([p["x"] * w, p["y"] * h], dtype=np.float64)

    ls, rs = pt("left_shoulder"), pt("right_shoulder")
    lh, rh = pt("left_hip"), pt("right_hip")
    if ls is None or rs is None:
        return None

    shoulder_y = float((ls[1] + rs[1]) / 2.0)
    centre_x = float((ls[0] + rs[0]) / 2.0)
    if lh is not None and rh is not None:
        centre_x = float((ls[0] + rs[0] + lh[0] + rh[0]) / 4.0)

    # Ramp from untouched at the chin to fully compressed a little below the
    # shoulders, so there is no visible seam across the neck.
    ramp_top = max(0.0, shoulder_y - 0.10 * h)
    ramp_bot = min(float(h), shoulder_y + 0.06 * h)

    ys = np.arange(h, dtype=np.float32)
    weight = np.clip((ys - ramp_top) / max(1.0, ramp_bot - ramp_top), 0.0, 1.0)
    scale = 1.0 - weight * (1.0 - factor)            # per-row compression

    xs = np.arange(w, dtype=np.float32)[None, :]
    map_x = (centre_x + (xs - centre_x) / scale[:, None]).astype(np.float32)
    map_y = np.repeat(ys[:, None], w, axis=1).astype(np.float32)

    return cv2.remap(img_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def _measurable(path: Path) -> bool:
    """True when this photograph yields at least two gate-worthy widths.

    A closeup with no hips in frame produces no torso length and therefore no
    proportions at all.  That is a property of the photograph, not a fault in
    the gate, and mixing the two hides both.
    """
    from app.analysis import body as body_mod

    img = loader.load_image(str(path), max_side=1600)
    po = pose.detect_pose(img)
    pm = segment.person_mask(img)
    measured = body_mod.measure_body(img, po, pm.get("mask"))
    if not measured.get("ok"):
        return False
    # Same definition the profile builder uses, so the harness cannot disagree
    # with the system it is measuring.
    usable = [m for m in profile_mod.usable_metrics(measured)
              if m in verify_mod.GATED_METRICS]
    return len(usable) >= 2


# ------------------------------------------------------------------ report

def _verdict_line(name: str, verdict: dict) -> str:
    failed = [c["name"] for c in verdict.get("checks", []) if not c.get("passed", True)]
    return (f"    {name:22s} {'RECHAZA' if not verdict.get('passed') else 'acepta '}"
            f"  score={verdict.get('score', 0):.2f}"
            f"  fallos={','.join(failed) if failed else '-'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--holdout", type=int, default=6)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--paired", action="store_true",
                    help="compare each result against its own source photograph, "
                         "which is what the orchestrator does in production")
    ap.add_argument("--measurable-only", action="store_true",
                    help="test only on photos whose torso can actually be measured, "
                         "which separates 'the gate does not work' from "
                         "'these photographs cannot be measured'")
    args = ap.parse_args()

    src = Path(args.dir)
    photos = sorted([p for p in src.iterdir()
                     if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")])
    if len(photos) < args.holdout + 4:
        print(f"Se necesitan al menos {args.holdout + 4} fotos en {src}")
        return 2

    if args.measurable_only:
        usable = [p for p in photos if _measurable(p)]
        print(f"Fotos con torso medible: {len(usable)} de {len(photos)}\n")
        if len(usable) < 4:
            print("Muy pocas fotos medibles para calibrar. Hacen falta fotos de")
            print("cuerpo entero o de medio cuerpo con las caderas visibles.")
            return 2
        holdout = usable[1::2][:args.holdout]
        train = [p for p in photos if p not in holdout]
    else:
        # Hold out every other photo so the two sets span the same sessions.
        holdout = photos[1::2][:args.holdout]
        train = [p for p in photos if p not in holdout]

    print("=" * 74)
    print("CALIBRACION DEL CONTROL DE IDENTIDAD")
    print("=" * 74)
    print(f"Entrenamiento: {len(train)} fotos    Prueba (no vistas): {len(holdout)} fotos\n")

    prof = profile_mod.build_profile([str(p) for p in train], "Calibracion")
    body = prof.get("body") or {}
    if not body:
        print("El perfil no pudo medir ninguna proporcion. Sin bandas no hay control.")
        return 3

    print("Bandas aprendidas (metrica: media, banda aceptada, anchura relativa):")
    for name, b in body.items():
        span = (b["hi"] - b["lo"]) / (2 * abs(b["mean"])) if b.get("mean") else 0
        gate = "GATE" if name in verify_mod.GATED_METRICS else "    "
        print(f"   {gate} {name:26s} {b['mean']:.3f}  [{b['lo']:.3f}, {b['hi']:.3f}]"
              f"  +/-{span * 100:4.1f}%  n={b['n']}")
    print()

    tmp = Path(tempfile.mkdtemp(prefix="calib_"))
    try:
        clean_pass: list[bool] = []
        # Net detection only counts photos the gate accepted untouched.  Without
        # that condition a gate that rejects everything scores 100%.
        net: dict[int, list[bool]] = {int(round((1 - f) * 100)): [] for f in SLIM_LEVELS}
        body_fired: dict[int, int] = {k: 0 for k in net}
        fail_counts: dict[str, int] = {}

        for p in holdout:
            img = loader.load_image(str(p), max_side=1600)
            print(f"  {p.name}")

            # Production always knows which photograph a result came from, so
            # the harness must test that same path: result compared with source.
            brief = {"source_path": str(p)} if args.paired else {}
            v = verify_mod.verify_image(str(p), prof, dict(brief))
            ok_clean = bool(v.get("passed"))
            clean_pass.append(ok_clean)
            for c in v.get("checks", []):
                if not c.get("passed", True):
                    fail_counts[c["name"]] = fail_counts.get(c["name"], 0) + 1
            print(_verdict_line("original", v))

            for f in SLIM_LEVELS:
                pct = int(round((1 - f) * 100))
                slim = slim_body(img, f)
                if slim is None:
                    print(f"    {'-' + str(pct) + '% ancho':22s} (sin pose, omitida)")
                    continue
                out = tmp / f"{p.stem}_slim{pct}.jpg"
                cv2.imwrite(str(out), slim, [cv2.IMWRITE_JPEG_QUALITY, 95])
                vs = verify_mod.verify_image(str(out), prof, dict(brief))
                body_failed = any(c["name"] == "body_proportions" and not c.get("passed", True)
                                  for c in vs.get("checks", []))
                if body_failed:
                    body_fired[pct] += 1
                if ok_clean:
                    net[pct].append(not vs.get("passed"))
                print(_verdict_line(f"-{pct}% de ancho", vs))
            print()

        print("=" * 74)
        print("RESULTADO")
        print("=" * 74)
        n_clean = len(clean_pass)
        n_ok = sum(clean_pass)
        fa = 100.0 * (1 - n_ok / n_clean) if n_clean else 0.0
        print(f"  Falsas alarmas (fotos reales rechazadas): {fa:5.1f}%  "
              f"({n_clean - n_ok} de {n_clean})   -> cuanto MENOR, mejor")
        if fail_counts:
            worst = sorted(fail_counts.items(), key=lambda kv: -kv[1])
            print("     causas: " + ", ".join(f"{k} x{v}" for k, v in worst))
        print(f"  Base para la deteccion neta: {n_ok} fotos aceptadas sin tocar\n")

        for f in SLIM_LEVELS:
            pct = int(round((1 - f) * 100))
            r = net[pct]
            if not r:
                print(f"  Adelgazamiento del {pct:2d}%: sin base limpia para medir")
                continue
            det = 100.0 * sum(r) / len(r)
            verdict = "OK" if det >= 80 else ("DEBIL" if det >= 50 else "NO DETECTA")
            print(f"  Deteccion NETA del adelgazamiento del {pct:2d}%: {det:5.1f}%  "
                  f"({sum(r)} de {len(r)})   [{verdict}]")
            print(f"     de las cuales por proporciones corporales: {body_fired[pct]}")
        print()
        print("  El control solo sirve si 'proporciones corporales' es lo que dispara.")
        print("  Si rechaza por otra causa, esta acertando por el motivo equivocado.")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

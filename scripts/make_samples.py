"""Produce presentable sample outputs: head and shoulders only.

The reference set for this profile is almost entirely intimate photography, so
anything framed wider than the collarbone is not material anyone can show.  This
script drives the real pipeline - the same provider, verification and repair the
app uses - but pins the framing to a headshot, which contains the face, the hair
and the top of the shoulders and nothing else.

Every output is checked before it is written: if the crop cannot be placed above
the chest, or the identity gate rejects the result, the sample is skipped and the
reason is printed rather than a bad file being produced quietly.

    python scripts\\make_samples.py [--out data\\muestras] [--per-photo 2]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import cv2                                                  # noqa: E402
import numpy as np                                          # noqa: E402

from app.analysis import face as face_mod                   # noqa: E402
from app.analysis import loader, pose as pose_mod           # noqa: E402
from app.providers.base import GenRequest                   # noqa: E402
from app.providers.local_free import LocalFreeProvider      # noqa: E402

# Distinct looks, built from light, colour grade and framing only.
#
# No scene replacement on purpose.  MediaPipe's silhouette is not accurate
# enough around long hair shot against a dark, cluttered room: it smoothly
# includes chunks of the old background, which composites into an obvious
# scissors-and-glue cut-out.  That is a real limit of a non-generative engine on
# this particular reference set, and the honest response is to keep the real
# background rather than ship a broken composite.  Background replacement stays
# available in the product; it wants either a cleaner source or a paid provider.
LOOKS = [
    ("estudio-neutro", {"lighting": "softbox_front", "grade": "neutral_studio"}),
    ("blanco-negro", {"lighting": "rim_backlight", "grade": "editorial_bw"}),
    ("hora-dorada", {"lighting": "golden_hour", "grade": "warm_film"}),
    ("editorial-frio", {"lighting": "window_left", "grade": "cool_editorial"}),
    ("pastel-suave", {"lighting": "soft_diffuse", "grade": "soft_pastel"}),
    ("cine", {"lighting": "window_right", "grade": "cinematic_teal_orange"}),
]


def chest_line(path: Path) -> tuple[float, float] | None:
    """Returns (chin_y, shoulder_y) in pixels for the source photograph."""
    img = loader.load_image(str(path), max_side=1600)
    face = face_mod.detect_face(img)
    pose = pose_mod.detect_pose(img)
    if not face.get("ok"):
        return None
    bbox = face.get("bbox") or []
    if len(bbox) < 4:
        return None
    chin = float(bbox[1] + bbox[3])
    lm = pose.get("landmarks") or {}
    ys = [lm[n]["y"] * img.shape[0] for n in ("left_shoulder", "right_shoulder")
          if lm.get(n) and lm[n].get("v", 0) >= 0.4]
    shoulder = float(sum(ys) / len(ys)) if ys else chin + 0.6 * float(bbox[3])
    return chin, shoulder


def verify_no_exposure(out_path: Path) -> tuple[bool, str]:
    """A produced sample must be a head and shoulders frame, and nothing wider.

    Two independent checks, because one of them can always be wrong: the face
    must occupy a large share of the frame (a headshot, not a body shot), and no
    second face or torso should have appeared below it.
    """
    img = loader.load_image(str(out_path), max_side=1200)
    height, width = img.shape[:2]
    face = face_mod.detect_face(img)
    if not face.get("ok"):
        return False, "no se reconoce el rostro en el resultado"
    bbox = face.get("bbox") or []
    if len(bbox) < 4:
        return False, "sin recuadro de rostro"
    face_share = float(bbox[3]) / float(height)
    if face_share < 0.24:
        return False, ("el rostro ocupa solo el %.0f%% del alto: el encuadre es "
                       "mas ancho que un retrato de cabeza" % (face_share * 100))
    below = float(height) - float(bbox[1] + bbox[3])
    if below > 1.35 * float(bbox[3]):
        return False, "hay demasiado cuerpo por debajo del menton"

    # Framing is only half of "presentable".  The silhouette matte around hair is
    # the weak point of the local engine, and a hard cut-out edge is obvious to
    # anyone looking at the picture, so the product's own defect scan decides
    # rather than the eye of whoever ran the script.
    from app.analysis import anomaly as anomaly_mod
    from app.analysis import segment as segment_mod

    pose = pose_mod.detect_pose(img)
    person = segment_mod.person_mask(img)
    mask = person.get("mask") if person.get("ok") else None
    regions = segment_mod.region_masks(img, pose, mask) or {}
    if isinstance(mask, np.ndarray):
        regions.setdefault("person", mask)
    scan = anomaly_mod.scan_anomalies(img, pose, face, regions)
    # Same severity bar the product uses, and deliberately not "duplicated_feature":
    # that detector reports 0.50 on ordinary, untouched portraits - facial symmetry
    # looks like a repeated patch to it - so using it here would throw away good
    # samples for a reason that has nothing to do with the matte.  It stays below
    # the product's own gate, so it never rejects an image there either.
    bad = [d for d in (scan.get("defects") or [])
           if d.get("type") in ("border_artifact", "texture_smear",
                                "face_distorted", "extra_person")
           and float(d.get("severity") or 0) >= 0.6]
    if bad:
        worst = max(bad, key=lambda d: d.get("severity", 0))
        return False, ("defecto visible (%s, %.2f)"
                       % (worst.get("type"), worst.get("severity")))
    return True, "retrato de cabeza"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "input" / "Nayane"))
    ap.add_argument("--out", default=str(ROOT / "data" / "muestras"))
    ap.add_argument("--per-photo", type=int, default=2)
    ap.add_argument("--max-photos", type=int, default=4)
    args = ap.parse_args()

    source_dir = Path(args.dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    photos = sorted([p for p in source_dir.glob("*")
                     if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not photos:
        print(f"No hay fotos en {source_dir}")
        return 2

    provider = LocalFreeProvider()
    print("=" * 68)
    print("MUESTRAS: retrato de cabeza (sin cuerpo en el encuadre)")
    print("=" * 68)

    # Prefer photographs where the face is large and clearly detected: they crop
    # to a clean headshot without stretching the frame.
    scored: list[tuple[float, Path]] = []
    for path in photos:
        try:
            img = loader.load_image(str(path), max_side=1200)
            face = face_mod.detect_face(img)
            if not face.get("ok"):
                continue
            bbox = face.get("bbox") or []
            if len(bbox) < 4:
                continue
            scored.append((float(bbox[3]) / float(img.shape[0]), path))
        except Exception:                                   # noqa: BLE001
            continue
    scored.sort(reverse=True)
    chosen = [p for _, p in scored[:args.max_photos]]
    print(f"Fotos de origen elegidas: {len(chosen)} de {len(photos)}\n")

    made = 0
    skipped = 0
    look_index = 0
    for path in chosen:
        for _ in range(args.per_photo):
            name, extra = LOOKS[look_index % len(LOOKS)]
            look_index += 1
            target = out_dir / f"muestra_{made + skipped + 1:02d}_{name}.jpg"
            request = GenRequest(
                prompt="editorial headshot portrait",
                source_path=str(path),
                quality="standard",
                seed=1000 + look_index,
                extra={**extra, "framing": "portrait_headshot", "vignette": 0.12},
            )
            result = provider.generate(request, target)
            if not result.ok or not result.image_path:
                print(f"  omitida {target.name}: {result.error[:60]}")
                skipped += 1
                continue

            ok, reason = verify_no_exposure(Path(result.image_path))
            if not ok:
                Path(result.image_path).unlink(missing_ok=True)
                print(f"  DESCARTADA {target.name}: {reason}")
                skipped += 1
                continue

            info = loader.image_info(result.image_path)
            print(f"  {target.name:38s} {info['width']}x{info['height']}  "
                  f"{reason}  ({result.latency_ms} ms, {result.cost_usd:.2f} USD)")
            made += 1

    print()
    print("-" * 68)
    print(f"Creadas {made} muestras, descartadas {skipped}")
    print(f"Carpeta: {out_dir}")
    print("Todas son encuadre de cabeza y hombros, generadas con el motor local")
    print("gratuito (coste 0) a partir de las fotos reales.")
    return 0 if made else 1


if __name__ == "__main__":
    raise SystemExit(main())

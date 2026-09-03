"""Does this image pass as a real photograph of her?

The gate in identity/verify.py answers "would the robot ship this".  That is
not quite the question the client asks, which is "is this still my skin and my
body".  So this script asks hers, with the gate's own instruments and against
her own photograph - never against a constant measured on somebody else:

    piel      verify._texture_loss, the fine band read at one face width, must
              sit inside +/- SMOOTH_TEXTURE_LOSS_MIN (0.09).  That is the line
              at which the gate starts merely REPORTING a loss, so landing
              inside it means the grain is not just unpunished, it is intact.
    cuerpo    verify._profile_ratio on the head ruler and on the width ruler
              against the source photograph.  |mediana - 1| < 0.03, tighter
              than the 0.04 the gate allows on the head ruler and far above
              the instrument's own scatter (0.4% between two resolutions of
              the same file, 1.1% in the worst case measured).
    rostro    the identity_face check the gate computed, >= 0.95 - well past
              the 0.72 it needs to pass, because an engine that only replaced
              the background has no excuse for having moved her face.
    tono      the skin_tone check must have passed.
    veredicto verify_image itself, printed with the detail string of every
              check, so the reason is the gate's own words and not a summary.

Nothing here touches the network and nothing costs money: it is local computer
vision over files that already exist.

    python scripts/acceptance_final.py --run-id run_xxx
    python scripts/acceptance_final.py --image PATH --original-id org_xxx
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app import db                                        # noqa: E402
from app.analysis import body as body_mod                 # noqa: E402
from app.analysis import face as face_mod                 # noqa: E402
from app.analysis import loader                           # noqa: E402
from app.analysis import pose as pose_mod                 # noqa: E402
from app.analysis import segment as segment_mod           # noqa: E402
from app.identity import verify as verify_mod             # noqa: E402

TEXTURE_TOL = verify_mod.SMOOTH_TEXTURE_LOSS_MIN          # 0.09
PROFILE_TOL = 0.03
FACE_MIN = 0.95


# ------------------------------------------------------------------ measuring

def _measure(path: str) -> dict:
    """Everything the two body rulers need, read at the gate's own size."""
    img = loader.load_image(str(path), max_side=verify_mod.VERIFY_MAX_SIDE)
    pose_d = pose_mod.detect_pose(img) or {}
    face_d = face_mod.detect_face(img) or {}
    seg = segment_mod.person_mask(img) or {}
    mask = seg.get("mask") if seg.get("ok") else None
    body_d = body_mod.measure_body(img, pose_d, mask, face_d) or {}
    return {"img": img, "face": face_d, "body": body_d}


def _ratio(gen_body: dict, src_body: dict, key: str) -> dict | None:
    return verify_mod._profile_ratio(gen_body.get(key), src_body.get(key))


def _check_of(verdict: dict, name: str) -> dict:
    for check in verdict.get("checks") or []:
        if check.get("name") == name:
            return check
    return {}


def judge(image_path: str, source_path: str, profile: dict,
          shot_type: str) -> dict:
    """One image, five verdicts and the numbers behind every one of them."""
    brief = {"source_path": source_path, "shot_type": shot_type,
             "generative": True}
    gen = _measure(image_path)
    src = _measure(source_path)

    # Asked for exactly as the gate asks for it, so a number printed here and a
    # number in the ficha can never disagree.
    tex = verify_mod._texture_loss(gen["img"], gen["face"], brief)
    loss = float(tex.get("loss") or 0.0) if tex.get("ok") else None

    head = _ratio(gen["body"], src["body"], verify_mod.HEAD_PROFILE_KEY)
    width = _ratio(gen["body"], src["body"], verify_mod.WIDTH_PROFILE_KEY)

    # generative=True on purpose.  The engine that made this file only
    # composites, so being judged as if it could have invented a waist is the
    # harder test, not the easier one.
    verdict = verify_mod.verify_image(image_path, profile, brief)
    face_check = _check_of(verdict, "identity_face")
    skin_check = _check_of(verdict, "skin_tone")

    rows = [("piel (textura)", loss, TEXTURE_TOL,
             loss is not None and abs(loss) < TEXTURE_TOL,
             "" if tex.get("ok") else "no medible",
             "banda fina %.3f frente a la suya %.3f"
             % (tex.get("fine") or 0.0, tex.get("ref") or 0.0))]
    for label, got in (("cuerpo (cabeza)", head), ("cuerpo (torso)", width)):
        value = (got["median"] - 1.0) if got else None
        rows.append((label, value, PROFILE_TOL,
                     value is not None and abs(value) < PROFILE_TOL,
                     "" if got else "no medible",
                     "%d filas comparadas, dispersion %.4f"
                     % (got["n"], got["spread"]) if got else ""))
    face_value = float(face_check.get("value") or 0.0)
    rows.append(("rostro (identidad)", face_value, FACE_MIN,
                 face_value >= FACE_MIN, "", face_check.get("detail") or ""))
    rows.append(("tono de piel", float(skin_check.get("value") or 0.0),
                 float(skin_check.get("threshold") or 0.0),
                 bool(skin_check.get("passed")), "",
                 skin_check.get("detail") or ""))

    return {"image": image_path, "rows": rows, "verdict": verdict,
            "texture": tex, "head": head, "width": width,
            "passed": all(r[3] for r in rows) and bool(verdict.get("passed")),
            "score": float(verdict.get("score") or 0.0)}


# -------------------------------------------------------------------- report

def _cell(value, ok: bool, note: str) -> str:
    if value is None:
        return "%-14s" % (note or "n/d")
    return "%+8.4f %-5s" % (value, "OK" if ok else "FALLA")


def report(results: list[dict]) -> None:
    print()
    print("=" * 100)
    print("%-22s %-14s %-14s %-14s %-14s %-14s"
          % ("imagen", "piel perdida", "cabeza m-1", "torso m-1", "rostro",
             "tono piel"))
    print("-" * 100)
    for res in results:
        cells = [_cell(r[1], r[3], r[4]) for r in res["rows"]]
        print("%-22s %-14s %-14s %-14s %-14s %-14s"
              % (Path(res["image"]).name, *cells))
    print("-" * 100)
    print("limites: piel < %.2f | cabeza y torso < %.2f | rostro >= %.2f | "
          "tono segun el umbral del perfil"
          % (TEXTURE_TOL, PROFILE_TOL, FACE_MIN))
    print("=" * 100)

    for res in results:
        print()
        print("%s   score %.4f   verify_image: %s"
              % (Path(res["image"]).name, res["score"],
                 "ACEPTADA" if res["verdict"]["passed"] else "RECHAZADA"))
        for label, value, limit, ok, note, detail in res["rows"]:
            shown = "no medible" if value is None else "%+.4f" % value
            print("   [%-5s] %-20s %-12s limite %-9.4f %s"
                  % ("OK" if ok else "FALLA", label, shown, limit, note))
            if detail:
                print("           %s" % detail[:160])
        for check in res["verdict"]["checks"]:
            print("   check %-18s %-6s valor %-10s umbral %-10s %s"
                  % (check["name"], "ok" if check["passed"] else "FALLA",
                     check["value"], check["threshold"],
                     (check.get("detail") or "")[:160]))
        print("   resumen: %s" % (res["verdict"].get("summary") or "")[:220])
        print("   ACEPTACION: %s" % ("cumple todo" if res["passed"]
                                     else "no cumple"))


# ----------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", action="append", default=[],
                    help="ruta de una imagen a juzgar (repetible)")
    ap.add_argument("--run-id", default="",
                    help="juzga todas las imagenes aceptadas de esa tanda")
    ap.add_argument("--original-id", default="",
                    help="la foto suya contra la que se compara")
    ap.add_argument("--person", default="Nayane")
    ap.add_argument("--copy-best", default="",
                    help="copia a esta ruta la mejor imagen que cumpla todo")
    args = ap.parse_args()

    images = list(args.image)
    original_id = args.original_id
    if args.run_id:
        rows = db.q("SELECT id, path, original_id FROM images WHERE run_id=? "
                    "AND deleted_at IS NULL ORDER BY created_at", (args.run_id,))
        images += [r["path"] for r in rows]
        if rows and not original_id:
            original_id = rows[0]["original_id"]
    if not images:
        print("Nada que juzgar: pasa --image o --run-id.")
        return 2

    original = db.row_to_dict(db.q1("SELECT * FROM originals WHERE id=?",
                                    (original_id,))) if original_id else None
    if not original:
        print("Falta la foto de origen: pasa --original-id.")
        return 2
    profile = db.row_to_dict(db.q1(
        "SELECT * FROM profiles WHERE person_name=? AND deleted_at IS NULL "
        "ORDER BY updated_at DESC LIMIT 1", (args.person,)))
    if not profile:
        print("No hay perfil de %s." % args.person)
        return 2

    print("origen : %s  (%s)" % (original["filename"], original["path"]))
    print("perfil : %s  %s" % (profile["id"], profile.get("person_name")))

    results = [judge(path, original["path"], profile,
                     original.get("shot_type") or "unknown")
               for path in images]
    report(results)

    good = [r for r in results if r["passed"]]
    print()
    print("cumplen todos los criterios: %d de %d" % (len(good), len(results)))
    if args.copy_best and good:
        best = max(good, key=lambda r: r["score"])
        dest = Path(args.copy_best)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(best["image"], dest)
        print("mejor imagen: %s -> %s (score %.4f)"
              % (Path(best["image"]).name, dest, best["score"]))
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())

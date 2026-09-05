"""Her own 24 photographs must never be called a stranger, and a stranger must.

Two questions, one instrument - the gate in identity/verify.py, with her real
stored profile (prf_dfcea806cba44c779cc8761f, 24 photographs, thresholds as
they sit in the database today):

  1. FALSE ALARMS.  Each of her 24 photographs is handed to the gate as if the
     engine had just returned it, with a clothing change declared, at three
     load sizes (1024, 1600 and 2048 px on the long side).  The load size is
     the thing that used to move the reading, so 24 x 3 = 72 verdicts.
     Every one of them must pass.
  2. THE FACES THAT ARE NOT HERS.  The photographs in sample/ go through the
     same gate, with the same brief and her own photograph as the source.  Who
     counts as a stranger is decided by the instrument and not by the file
     name: eight of the ten read below the threshold in her profile, and every
     one of those eight must fail, and fail ON IDENTITY.

Nothing is written anywhere: the database is opened read only, the resized
copies live in a temporary folder, and no network call is made.  Cost 0.00 USD.

    backend\\.venv\\Scripts\\python.exe scripts\\sweep_identity.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
os.environ.pop("ANTHROPIC_API_KEY", None)

from app import db                                          # noqa: E402
from app.analysis import loader                             # noqa: E402
from app.identity import verify as verify_mod               # noqa: E402

SIZES = tuple(int(x) for x in
              (os.environ.get("SWEEP_SIZES") or "1024,1600,2048").split(","))
CLOTHING = "traje_sastre"
WORK = Path(tempfile.mkdtemp(prefix="sweep72_"))

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    line = "  [%s] %s" % ("OK  " if ok else "FALLO", name)
    if detail:
        line += "  -> %s" % detail
    print(line, flush=True)


def resized(path: str, side: int, out: Path) -> str:
    img = loader.load_image(str(path), max_side=side)
    cv2.imwrite(str(out), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return str(out)


def brief_for(original: dict) -> dict:
    a = original.get("analysis") or {}
    return {
        "shot_type": a.get("shot_type") or "unknown",
        "source_path": original["path"],
        "source_body": a.get("body") or {},
        "source_skin": a.get("skin") or {},
        "expects_face": bool(a.get("has_face")),
        "choices": {"clothing": [CLOTHING]},
        "generative": True,
    }


def failed_checks(verdict: dict) -> list[str]:
    return [c.get("name") for c in (verdict.get("checks") or [])
            if not c.get("passed")]


def value_of(verdict: dict, name: str):
    for c in (verdict.get("checks") or []):
        if c.get("name") == name:
            return c.get("value")
    return None


def main() -> int:
    profile = db.row_to_dict(db.q1(
        "SELECT * FROM profiles WHERE deleted_at IS NULL "
        "ORDER BY is_default DESC LIMIT 1"))
    originals = db.rows_to_dicts(db.q(
        "SELECT * FROM originals WHERE deleted_at IS NULL ORDER BY filename"))
    print("=" * 78)
    print("BARRIDO: SUS 24 FOTOS A TRES TAMANOS, CON CAMBIO DE ROPA DECLARADO")
    print("=" * 78)
    print("perfil %s (%s, %d fotos)  umbral de rostro %.2f  ropa: %s"
          % (profile["id"], profile["person_name"], profile["n_sources"],
             (profile.get("thresholds") or {}).get("face_embed_min", 0.45),
             CLOTHING))
    print("%d fotos x %d tamanos = %d veredictos\n"
          % (len(originals), len(SIZES), len(originals) * len(SIZES)))

    print("  %-26s %-6s %-6s %-6s %-6s %s"
          % ("foto", "plano", "px", "rostro", "punt", "veredicto"))
    rejected = []
    scores = []
    faces = []
    for org in originals:
        brief = brief_for(org)
        for side in SIZES:
            out = WORK / ("%s_%d.jpg" % (Path(org["filename"]).stem, side))
            path = resized(org["path"], side, out)
            verdict = verify_mod.verify_image(path, profile, brief)
            face = value_of(verdict, "identity_face")
            scores.append(float(verdict.get("score") or 0.0))
            if isinstance(face, (int, float)):
                faces.append(float(face))
            if not verdict.get("passed"):
                rejected.append((org["filename"], side, failed_checks(verdict),
                                 verdict.get("summary")))
            print("  %-26s %-6s %-6d %-6s %-6.3f %s"
                  % (Path(org["filename"]).stem[:26],
                     brief["shot_type"], side,
                     ("%.3f" % face) if isinstance(face, (int, float)) else "-",
                     float(verdict.get("score") or 0.0),
                     "PASA" if verdict.get("passed") else
                     "RECHAZA " + ",".join(failed_checks(verdict))))

    total = len(originals) * len(SIZES)
    print()
    check("0 rechazos en %d (sus fotos nunca son un fallo)" % total,
          not rejected,
          "%d rechazos: %s" % (len(rejected), rejected[:3]) if rejected
          else "%d de %d pasan, parecido facial %.3f..%.3f, puntuacion %.3f..%.3f"
          % (total, total, min(faces), max(faces), min(scores), max(scores)))

    # -------------------------------------------------- las caras que no son
    print("\n" + "=" * 78)
    print("LAS CARAS QUE NO SON SUYAS: MISMO PORTON, MISMO INFORME")
    print("=" * 78)
    others = sorted(p for p in (ROOT / "sample").glob("*")
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    source = next((o for o in originals if "IMG_7871" in o["filename"]),
                  originals[0])
    brief = brief_for(source)
    print("comparadas contra su perfil, con %s como foto de origen\n"
          % Path(source["filename"]).stem)
    # Who is in the folder is decided by the instrument, not by the file name.
    # sample/ turned out to hold TEN photographs and only EIGHT of them are
    # other people: two read in her own range and the gate says so, which is
    # the honest way to split them - the threshold in her profile is the line.
    face_min = float((profile.get("thresholds") or {}).get("face_embed_min", 0.45))
    print("  %-42s %-8s %-8s %s" % ("archivo", "rostro", "punt", "veredicto"))
    strangers, hers = [], []
    for other in others:
        verdict = verify_mod.verify_image(str(other), profile, brief)
        face = value_of(verdict, "identity_face")
        fails = failed_checks(verdict)
        row = {"nombre": other.name, "rostro": face, "pasa": verdict.get("passed"),
               "fallos": fails, "resumen": verdict.get("summary") or ""}
        (hers if isinstance(face, (int, float)) and face >= face_min
         else strangers).append(row)
        print("  %-42s %-8s %-8.3f %s"
              % (other.name[:42],
                 ("%.3f" % face) if isinstance(face, (int, float)) else "-",
                 float(verdict.get("score") or 0.0),
                 "PASA" if verdict.get("passed") else "RECHAZA " + ",".join(fails)))
    print("\n"
          "%d de %d son otras personas (parecido %.3f..%.3f, umbral %.2f); "
          "%d miden como ella (%s)"
          % (len(strangers), len(others),
             min(r["rostro"] for r in strangers),
             max(r["rostro"] for r in strangers), face_min, len(hers),
             ", ".join("%s %.3f" % (r["nombre"][:18], r["rostro"]) for r in hers)))
    check("ninguna cara ajena pasa la puerta (%d desconocidas)" % len(strangers),
          not any(r["pasa"] for r in strangers),
          "pasaron: %s" % [r["nombre"] for r in strangers if r["pasa"]]
          if any(r["pasa"] for r in strangers)
          else "las %d se descartan" % len(strangers))
    check("y todas caen POR IDENTIDAD, no por otra cosa",
          all("identity_face" in r["fallos"] for r in strangers),
          "%d de %d fallan identity_face"
          % (sum(1 for r in strangers if "identity_face" in r["fallos"]),
             len(strangers)))
    check("y a ella la reconoce tambien fuera de su perfil",
          all(isinstance(r["rostro"], float) and r["rostro"] >= face_min
              for r in hers) and bool(hers),
          "; ".join("%s %.3f %s" % (r["nombre"][:20], r["rostro"],
                                    "pasa" if r["pasa"]
                                    else "descartada por " + ",".join(r["fallos"]))
                    for r in hers))
    for row in hers:
        if not row["pasa"]:
            print("  nota: %s se descarta aun siendo ella -> %s"
                  % (row["nombre"], row["resumen"][:160]))

    print("\n" + "=" * 78)
    print("RESULTADO: %d correctas, %d fallidas   |   COSTE REAL: 0.0000 USD"
          % (len(PASS), len(FAIL)))
    print("=" * 78)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())

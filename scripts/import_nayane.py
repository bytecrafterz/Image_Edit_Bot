"""Load a folder of reference photographs and build the identity profile.

This is how the developer checks the identity engine against real pictures
without touching the web interface.  It prints what was measured, what could not
be measured and why, and - when the reference set cannot support the body check -
the exact instructions to send the client.

    python scripts\\import_nayane.py
    python scripts\\import_nayane.py --dir "C:\\fotos\\Ana" --name Ana
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from app import db, security                              # noqa: E402
from app.config import SETTINGS, ensure_dirs              # noqa: E402
from app.identity import profile as profile_mod           # noqa: E402
from app.identity import verify as verify_mod             # noqa: E402

SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def ensure_admin(email: str, password: str) -> dict:
    row = db.q1("SELECT COUNT(*) AS n FROM users")
    if int(row["n"] or 0) > 0:
        existing = db.row_to_dict(
            db.q1("SELECT * FROM users WHERE role='admin' ORDER BY created_at LIMIT 1"))
        if existing:
            return existing
    user_id = db.new_id("usr")
    now = db.now()
    db.execute(
        "INSERT INTO users(id,email,password_hash,display_name,role,status,"
        "daily_limit_usd,monthly_limit_usd,created_at,approved_at) "
        "VALUES(?,?,?,?,'admin','active',?,?,?,?)",
        (user_id, email.lower(), security.hash_password(password), "Administrador",
         SETTINGS.limits.default_daily_usd, SETTINGS.limits.default_monthly_usd,
         now, now))
    print(f"  Cuenta administradora creada: {email} / {password}")
    print("  Cambia esa contrasena al entrar por primera vez.\n")
    return db.row_to_dict(db.q1("SELECT * FROM users WHERE id=?", (user_id,)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(ROOT / "input" / "Nayane"))
    parser.add_argument("--name", default="Nayane")
    parser.add_argument("--email", default="admin@photorobot.local")
    parser.add_argument("--password", default="photorobot-2026")
    args = parser.parse_args()

    ensure_dirs()
    db.init_db()

    source = Path(args.dir)
    photos = sorted([p for p in source.rglob("*") if p.suffix.lower() in SUFFIXES])
    if not photos:
        print(f"No hay fotos en {source}")
        return 2

    print("=" * 70)
    print(f"IMPORTAR Y MEDIR: {args.name}")
    print("=" * 70)
    print(f"Carpeta: {source}")
    print(f"Fotos:   {len(photos)}\n")

    user = ensure_admin(args.email, args.password)

    from app.routers.originals import _ingest

    profile_row = db.row_to_dict(db.q1(
        "SELECT * FROM profiles WHERE user_id=? AND person_name=? "
        "AND deleted_at IS NULL", (user["id"], args.name)))
    if profile_row:
        profile_id = profile_row["id"]
        print(f"  Reutilizando el perfil existente ({profile_id})")
    else:
        profile_id = db.new_id("prf")
        now = db.now()
        db.execute(
            "INSERT INTO profiles(id,user_id,person_name,status,is_default,"
            "created_at,updated_at) VALUES(?,?,?,'draft',1,?,?)",
            (profile_id, user["id"], args.name, now, now))
        print(f"  Perfil creado ({profile_id})")

    from app.safety import consent as consent_mod
    consent_mod.record_consent(user["id"], profile_id, {"relationship": "self"})

    added = duplicated = failed = 0
    for path in photos:
        try:
            result = _ingest(user, path.name, path.read_bytes(), profile_id)
            if result.get("duplicate"):
                duplicated += 1
            else:
                added += 1
        except Exception as exc:                          # noqa: BLE001
            failed += 1
            print(f"    no se pudo leer {path.name}: {exc}")
    print(f"  Importadas {added}, ya estaban {duplicated}, fallidas {failed}\n")

    print("  Midiendo... (unos segundos por foto)")
    rows = db.q("SELECT path FROM originals WHERE profile_id=? AND deleted_at IS NULL "
                "ORDER BY sort_order", (profile_id,))
    built = profile_mod.build_profile([r["path"] for r in rows], args.name)

    from app.routers.profiles import _store_profile
    _store_profile(profile_id, built)

    # ------------------------------------------------------------- report
    print("\n" + "=" * 70)
    print("RESULTADO DE LA MEDICION")
    print("=" * 70)

    coverage = built.get("coverage") or {}
    print(f"\nFotos aceptadas: {built.get('n_sources')}")
    rejected = built.get("rejected") or []
    if rejected:
        print(f"Fotos descartadas: {len(rejected)}")
        for item in rejected[:8]:
            print(f"   - {Path(item['path']).name}: {item['reason']}")

    print("\nPor tipo de plano:")
    for key, label in (("full", "Cuerpo entero"), ("half", "Medio cuerpo"),
                       ("closeup", "Primer plano"), ("unknown", "Sin identificar")):
        print(f"   {label:18s} {coverage.get(key, 0)}")
    print(f"   {'Utiles para medir':18s} {coverage.get('usable_body_shots', 0)}")

    face = built.get("face") or {}
    print(f"\nFirma facial: construida con {face.get('n', 0)} fotos "
          f"({len(face.get('descriptor') or [])} valores)")

    skin = built.get("skin") or {}
    if skin.get("lab_mean"):
        lab = skin["lab_mean"]
        print(f"Tono de piel: L*={lab[0]:.1f} a*={lab[1]:.1f} b*={lab[2]:.1f}  "
              f"(ITA {skin.get('ita_deg')} grados, {skin.get('n')} fotos)")

    hair = built.get("hair") or {}
    if hair.get("length"):
        print(f"Pelo: {hair['length']}, medido en {hair.get('n', 0)} fotos")

    marks = built.get("marks") or []
    if marks:
        print(f"\nMarcas y tatuajes detectados: {len(marks)}")
        for mark in marks[:6]:
            print(f"   - {mark.get('type')} en {mark.get('region')} "
                  f"(visto en {mark.get('seen_in')} fotos)")

    body = built.get("body") or {}
    if body:
        print("\nProporciones medidas (banda aceptada):")
        for name, band in body.items():
            gate = "PUEDE RECHAZAR" if band.get("gated") else "solo informativa"
            span = 100 * (band["hi"] - band["lo"]) / 2 / abs(band["mean"]) \
                if band.get("mean") else 0
            print(f"   {name:26s} {band['mean']:.3f}  "
                  f"[{band['lo']:.3f} - {band['hi']:.3f}]  +/-{span:.0f}%  "
                  f"n={band['n']}  {gate}")
    else:
        print("\nNo se pudo medir ninguna proporcion corporal.")

    ready = coverage.get("ready_for_body_check")
    print("\n" + "-" * 70)
    if ready:
        print("EL CONTROL DE PROPORCIONES ESTA ACTIVO.")
        print("Cada imagen generada se comparara con estas medidas y se")
        print("descartara si te estrechan los hombros, la cintura o las caderas.")
    else:
        print("EL CONTROL DE PROPORCIONES TODAVIA NO PUEDE FUNCIONAR.")
        print("Se puede cuidar la cara, la piel y las marcas, pero no las")
        print("proporciones del cuerpo, que es justo lo que fallo la otra vez.")
        print("\nPidele estas fotos a la persona:")
        for line in (coverage.get("advice") or profile_mod.REFERENCE_ADVICE):
            print(f"   - {line}")
    print("-" * 70)
    print(f"\nPerfil guardado. Entra en la aplicacion como {args.email}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

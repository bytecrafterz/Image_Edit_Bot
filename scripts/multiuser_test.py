"""Prueba de que el sistema sirve para VARIAS personas, con coste 0 USD.

Tres cosas, en una base de datos temporal y con el motor local gratuito:

1. AISLAMIENTO.  Dos cuentas reales, cada una con sus fotos, su perfil, su
   tirada y sus imagenes.  Despues cada una intenta leer, descargar, marcar,
   borrar y generar con los identificadores de la otra, ruta por ruta.  La
   administradora tambien: en este sistema un administrador NO puede ver las
   fotos de nadie.
2. ALTA.  Una cuenta nueva no puede generar hasta subir el minimo medido de
   fotos reales (identity/onboarding.py explica de donde sale el numero), y se
   le dice por que, que se hace con sus fotos y cuantas le faltan.
3. ANALISIS INICIAL.  En la primera tirada de una persona se miran sus fotos a
   fondo y se le ensena el resultado ANTES de gastar: que se puede comprobar de
   ella y que no.

La segunda persona se construye con las fotos de OTRA mujer (sample/), no con
las de la primera, que es la unica forma de comprobar que nada esta atado a una
persona concreta.

    python scripts\\multiuser_test.py [--keep]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PASSED: list[str] = []
FAILED: list[str] = []

# sample/ contiene diez fotos: dos son de la MISMA persona que input/Nayane
# (0.8267 y 0.7565 contra su firma facial) y las otras ocho son de otras
# mujeres.  Las cinco capturas IMG_70xx son una sola mujer distinta - se
# parecen entre si de 0.36 a 0.70 y ninguna pasa de 0.18 contra la primera
# persona - asi que son el material honesto para montar una SEGUNDA persona.
PERSON_TWO = ("IMG_7027.png", "IMG_7028.png", "IMG_7029.png", "IMG_7030.png",
              "IMG_7031.png")


def check(name: str, ok: bool, detail: str = "") -> bool:
    (PASSED if ok else FAILED).append(name)
    line = "  [%s] %s" % ("OK  " if ok else "FALLO", name)
    if detail:
        line += "  -> %s" % detail
    print(line, flush=True)
    return bool(ok)


def wait_run(client, run_id: str, limit: int = 240) -> dict:
    for _ in range(limit):
        time.sleep(1.5)
        r = client.get("/api/generate/status/%s" % run_id)
        if r.status_code != 200:
            return {}
        data = r.json()
        if data.get("status") in ("done", "failed", "cancelled",
                                  "stopped_no_balance"):
            return data
    return {}


def build_account(client, photos, person: str, quiet: bool = False) -> dict:
    up = client.post("/api/originals",
                     files=[("files", (p.name, p.read_bytes(), "image/jpeg"))
                            for p in photos]).json()
    origs = up.get("originals") or []
    prof = client.post("/api/profiles", json={"person_name": person}).json()
    client.post("/api/profiles/%s/consent" % prof["id"],
                json={"relationship": "self"})
    built = client.post("/api/profiles/%s/build" % prof["id"], json={}).json()
    if built.get("async"):
        wait_run(client, built["run_id"])
    plan = client.post("/api/generate/analyze", json={
        "original_id": origs[0]["id"], "profile_id": prof["id"],
        "options": {}, "n_previews": 1, "quality": "preview"}).json()
    client.post("/api/generate/run", json={"run_id": plan["run_id"]})
    state = wait_run(client, plan["run_id"])
    client.post("/api/settings/recharge", json={"provider": "fal",
                                                "amount_usd": 5})
    client.post("/api/settings/options", json={
        "group_key": "scene", "value_key": "sitio_" + person.lower(),
        "label_es": "Sitio de " + person, "prompt_fragment": "a quiet room"})
    return {"originals": origs, "profile": prof["id"], "run": plan["run_id"],
            "images": state.get("images") or []}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="photorobot_multi_"))
    os.environ["PHOTOROBOT_DATA"] = str(workdir)
    sys.path.insert(0, str(ROOT / "backend"))
    logging.getLogger("httpx").setLevel(logging.WARNING)

    print("=" * 78)
    print("PRUEBA MULTIUSUARIO (sin servicios de pago, coste 0.00 USD)")
    print("=" * 78)
    print("Datos temporales en: %s\n" % workdir)

    from fastapi.testclient import TestClient
    from app.main import create_app
    from app.identity import onboarding as onboarding_mod

    app = create_app()
    A = TestClient(app)
    B = TestClient(app)

    photos_one = sorted((ROOT / "input" / "Nayane").glob("*.jpeg"))
    photos_two = [ROOT / "sample" / n for n in PERSON_TWO]
    if not photos_one or not all(p.is_file() for p in photos_two):
        check("hay fotos de las dos personas", False)
        return finish(workdir, args.keep)

    # ------------------------------------------------------------ 1. altas
    print("1. Dos cuentas")
    ra = A.post("/api/auth/register", json={
        "email": "ana@prueba.local", "password": "clave-1234",
        "display_name": "Ana"})
    check("se crea la primera cuenta (administradora)",
          ra.status_code == 200 and ra.json()["user"]["role"] == "admin")
    A.headers.update({"Authorization": "Bearer " + ra.json()["token"]})
    uid_a = ra.json()["user"]["id"]
    onb = ra.json().get("onboarding") or {}
    check("el alta ya dice cuantas fotos hacen falta",
          onb.get("minimo") == onboarding_mod.MIN_PHOTOS and onb.get("faltan") ==
          onboarding_mod.MIN_PHOTOS, onb.get("mensaje", ""))
    check("y explica por que y que se hace con ellas",
          len(onb.get("porque") or []) >= 3
          and len(onb.get("que_pasa_con_tus_fotos") or []) >= 3)

    rb = B.post("/api/auth/register", json={
        "email": "bea@prueba.local", "password": "clave-5678",
        "display_name": "Bea"})
    check("la segunda cuenta queda pendiente de aprobacion",
          rb.status_code == 200 and rb.json().get("needs_approval") is True)
    uid_b = rb.json()["user"]["id"]
    check("y ya se le avisa de las fotos antes de entrar",
          str(onboarding_mod.MIN_PHOTOS) in rb.json().get("message", ""),
          rb.json().get("message", "")[:90])
    A.post("/api/admin/users/%s/approve" % uid_b)
    lb = B.post("/api/auth/login", json={"email": "bea@prueba.local",
                                         "password": "clave-5678"})
    check("la administradora la aprueba y ya puede entrar", lb.status_code == 200)
    B.headers.update({"Authorization": "Bearer " + lb.json()["token"]})

    # ------------------------------------------------- 2. minimo de fotos
    print("\n2. El minimo de fotos, medido")
    few = photos_two[:onboarding_mod.MIN_PHOTOS - 1]
    up = B.post("/api/originals", files=[
        ("files", (p.name, p.read_bytes(), "image/jpeg")) for p in few]).json()
    check("se suben %d fotos (una menos que el minimo)" % len(few),
          len(up.get("originals") or []) == len(few))
    state = up.get("onboarding") or {}
    check("la respuesta de la subida dice cuantas faltan",
          state.get("faltan") == 1, state.get("mensaje", ""))
    prof_b = B.post("/api/profiles", json={"person_name": "Bea"}).json()
    B.post("/api/profiles/%s/consent" % prof_b["id"],
           json={"relationship": "self"})
    denied = B.post("/api/generate/analyze", json={
        "original_id": up["originals"][0]["id"], "profile_id": prof_b["id"],
        "n_previews": 1, "quality": "preview"})
    check("con menos del minimo NO se puede generar",
          denied.status_code == 400, "HTTP %d" % denied.status_code)
    check("y el motivo esta en su idioma y dice cuantas faltan",
          "falta" in denied.json().get("detail", "").lower(),
          denied.json().get("detail", "")[:110])

    last = photos_two[onboarding_mod.MIN_PHOTOS - 1]
    up2 = B.post("/api/originals", files=[
        ("files", (last.name, last.read_bytes(), "image/jpeg"))]).json()
    check("al llegar al minimo, ya puede generar",
          (up2.get("onboarding") or {}).get("puede_generar") is True,
          (up2.get("onboarding") or {}).get("mensaje", ""))

    # ------------------------------------------- 3. analisis de la primera vez
    print("\n3. Analisis detallado en la primera tirada (persona 2, no la 1)")
    started = time.time()
    fr = B.post("/api/profiles/%s/first-run" % prof_b["id"], json={})
    check("se lanza el analisis inicial", fr.status_code == 200,
          "HTTP %d" % fr.status_code)
    informe = fr.json().get("informe") or {}
    if fr.json().get("async"):
        wait_run(B, fr.json()["run_id"])
        informe = (B.get("/api/profiles/%s/first-run"
                         % prof_b["id"]).json().get("informe") or {})
    check("el informe esta hecho", bool(informe.get("hecho")),
          "%.0f s" % (time.time() - started))
    print("\n--- INFORME QUE VE BEA ANTES DE GASTAR ---")
    print(informe.get("resumen", ""))
    for key in ("rostro", "cuerpo", "manos", "piel"):
        block = informe.get(key) or {}
        print("  %-8s %s" % (key + ":", block.get("veredicto", "")))
    print("  referencias: %s" % (informe.get("referencias") or {}).get("motivo", ""))
    print("  SE PUEDE COMPROBAR:")
    for line in informe.get("se_puede_comprobar") or []:
        print("    + %s" % line)
    print("  NO SE PUEDE COMPROBAR:")
    for line in informe.get("no_se_puede_comprobar") or ["(nada: todo medible)"]:
        print("    - %s" % line)
    print("--- fin del informe ---\n")

    check("el informe habla de SU perfil, no del de nadie mas",
          informe.get("perfil_id") == prof_b["id"]
          and informe.get("persona") == "Bea")
    check("mide su rostro con sus propias fotos",
          (informe.get("rostro") or {}).get("fotos_con_rostro", 0) >= 2,
          "peor foto %s, limite %s"
          % ((informe.get("rostro") or {}).get("peor_foto"),
             (informe.get("rostro") or {}).get("limite")))
    check("dice si sus proporciones se pueden medir o no",
          bool((informe.get("cuerpo") or {}).get("veredicto")))
    check("elige sus fotos de referencia",
          (informe.get("referencias") or {}).get("n", 0) >= 1,
          "%d fotos" % (informe.get("referencias") or {}).get("n", 0))
    check("y ninguna de esas referencias es de la otra persona",
          all("Nayane" not in n
              for n in (informe.get("referencias") or {}).get("fotos") or []))

    # ------------------------------------------------ 4. generar de verdad
    print("\n4. La segunda persona genera con el motor gratuito")
    cat = B.get("/api/catalog/options",
                params={"original_id": up["originals"][0]["id"]}).json()
    choices = {}
    for group in cat.get("groups", []):
        if group["group_key"] in ("scene",) and group["values"]:
            choices[group["group_key"]] = [v["value_key"]
                                           for v in group["values"][:2]]
    plan = B.post("/api/generate/analyze", json={
        "original_id": up["originals"][0]["id"], "profile_id": prof_b["id"],
        "options": choices, "n_previews": 2, "quality": "preview"})
    check("se planifica y se calcula el coste", plan.status_code == 200,
          "HTTP %d" % plan.status_code)
    if plan.status_code != 200:
        print(plan.text[:400])
        return finish(workdir, args.keep)
    planned = plan.json()
    check("el coste con el motor local es cero",
          float(planned["estimate"]["total_usd"]) == 0.0,
          "%s USD" % planned["estimate"]["total_usd"])
    check("la estimacion trae el analisis de sus fotos",
          bool(planned.get("analisis_inicial")),
          "resumen de %d caracteres"
          % len((planned.get("analisis_inicial") or {}).get("resumen", "")))
    B.post("/api/generate/run", json={"run_id": planned["run_id"]})
    final = wait_run(B, planned["run_id"])
    check("la tirada termina", final.get("status") == "done",
          "estado=%s %s" % (final.get("status"), final.get("error") or ""))
    imgs_b = final.get("images") or []
    check("produce al menos una imagen", len(imgs_b) >= 1,
          "%d imagenes" % len(imgs_b))

    print("\n5. La primera persona hace lo mismo, con sus propias fotos")
    DA = build_account(A, photos_one[:6], "Ana")
    check("la primera cuenta tiene sus imagenes", len(DA["images"]) >= 1,
          "%d imagenes" % len(DA["images"]))
    prof_a = A.get("/api/profiles/%s" % DA["profile"]).json()
    fr_a = (prof_a.get("first_run") or {})
    check("cada persona tiene SU propio informe",
          fr_a.get("persona") == "Ana" and informe.get("persona") == "Bea",
          "%s / %s" % (fr_a.get("persona"), informe.get("persona")))
    check("y cada informe mide numeros distintos",
          (fr_a.get("rostro") or {}).get("peor_foto")
          != (informe.get("rostro") or {}).get("peor_foto"),
          "Ana %s vs Bea %s" % ((fr_a.get("rostro") or {}).get("peor_foto"),
                                (informe.get("rostro") or {}).get("peor_foto")))

    # ---------------------------------------------------- 6. aislamiento
    print("\n6. Aislamiento: cada una intenta alcanzar los datos de la otra")
    from app import db

    def victim_ids(uid, data):
        row = db.q1("SELECT id FROM options WHERE user_id=?", (uid,))
        return {"org": data["originals"][0]["id"], "prof": data["profile"],
                "run": data["run"],
                "img": data["images"][0]["id"] if data["images"] else "img_x",
                "opt": row["id"] if row else "opt_x"}

    DB = {"originals": up["originals"] + up2["originals"], "profile": prof_b["id"],
          "run": planned["run_id"], "images": imgs_b}
    IA, IB = victim_ids(uid_a, DA), victim_ids(uid_b, DB)

    def sweep(label, attacker, victim, own):
        # LA SESION DEL ATACANTE TIENE QUE ESTAR VIVA ANTES DE EMPEZAR.  Un
        # barrido cruzado se auto-anula con un silencio total si el atacante ha
        # perdido la sesion: TODA peticion contesta 401, el bucle lo cuenta
        # como "denegada" y el barrido pasa sin haber probado nada.  Medido el
        # 2026-09-04 montando el barrido en el orden contrario: como la lista
        # de la administradora incluia POST /api/admin/users/{id}/suspend, la
        # segunda cuenta quedaba suspendida y sus 34 peticiones contra la
        # primera devolvieron 401 - 34 de 34 "denegadas" sin tocar una sola
        # comprobacion de propiedad.  Esta lista no contiene hoy ninguna ruta
        # que suspenda ni cambie la contrasena de nadie, asi que el fallo no se
        # da; la comprobacion existe para que se oiga el dia que alguien anada
        # una, en vez de convertir la prueba en un aprobado vacio.
        alive = attacker.get("/api/auth/me")
        if alive.status_code != 200:
            check("%s: la sesion del atacante sigue viva antes del barrido" % label,
                  False, "HTTP %d: el barrido no probaria nada" % alive.status_code)
            return 0, 0

        calls = [
            ("originals", "GET", "/api/originals/%s/analysis" % victim["org"], {}),
            ("originals", "PATCH", "/api/originals/%s" % victim["org"],
             {"json": {"tags": "robado"}}),
            ("originals", "DELETE", "/api/originals/%s" % victim["org"], {}),
            ("originals", "PATCH", "/api/originals/%s" % own["org"],
             {"json": {"profile_id": victim["prof"]}}),
            ("files", "GET", "/api/files/original/%s" % victim["org"], {}),
            ("files", "GET", "/api/files/id/%s" % victim["img"], {}),
            ("profiles", "GET", "/api/profiles/%s" % victim["prof"], {}),
            ("profiles", "GET", "/api/profiles/%s/first-run" % victim["prof"], {}),
            ("profiles", "POST", "/api/profiles/%s/first-run" % victim["prof"],
             {"json": {}}),
            ("profiles", "POST", "/api/profiles/%s/build" % victim["prof"],
             {"json": {}}),
            ("profiles", "POST", "/api/profiles/%s/forget-originals" % victim["prof"],
             {"json": {"confirm": True}}),
            ("profiles", "DELETE", "/api/profiles/%s" % victim["prof"], {}),
            ("runs", "GET", "/api/generate/status/%s" % victim["run"], {}),
            ("runs", "GET", "/api/generate/report/%s" % victim["run"], {}),
            ("runs", "POST", "/api/generate/run", {"json": {"run_id": victim["run"]}}),
            ("runs", "POST", "/api/generate/cancel/%s" % victim["run"], {}),
            ("runs", "POST", "/api/generate/analyze",
             {"json": {"original_id": victim["org"], "n_previews": 1,
                       "quality": "preview"}}),
            ("attempts", "POST", "/api/generate/final",
             {"json": {"run_id": own["run"], "image_ids": [victim["img"]]}}),
            ("images", "GET", "/api/album/%s/download" % victim["img"], {}),
            ("images", "POST", "/api/album/%s/final" % victim["img"], {"json": {}}),
            ("images", "POST", "/api/album/%s/feedback" % victim["img"],
             {"json": {"verdict": "like"}}),
            ("images", "DELETE", "/api/album/%s" % victim["img"], {}),
            ("images", "POST", "/api/album/%s/restore" % victim["img"], {"json": {}}),
            ("images", "POST", "/api/favorites/%s" % victim["img"], {"json": {}}),
            ("catalog", "GET", "/api/catalog/options?original_id=%s" % victim["org"], {}),
            ("options", "DELETE", "/api/settings/options/%s" % victim["opt"], {}),
        ]
        blocked = 0
        for table, method, path, kw in calls:
            r = getattr(attacker, method.lower())(path, **kw)
            ok = r.status_code >= 400
            blocked += 1 if ok else 0
            if not ok:
                print("      FUGA %-8s %s %s -> %d %s"
                      % (table, method, path, r.status_code, r.text[:120]))
        check("%s: ninguna ruta con id ajeno responde" % label,
              blocked == len(calls), "%d de %d denegadas" % (blocked, len(calls)))
        return blocked, len(calls)

    sweep("la ADMINISTRADORA contra la usuaria", A, IB, IA)
    sweep("la usuaria contra la ADMINISTRADORA", B, IA, IB)

    # Las listas filtradas responden 200 pero vacias: se comprueba el contenido.
    check("una lista filtrada por la tirada ajena vuelve vacia",
          A.get("/api/album?run_id=" + IB["run"]).json()["total"] == 0)
    check("una lista filtrada por el perfil ajeno vuelve vacia",
          A.get("/api/originals?profile_id=" + IB["prof"]).json()["total"] == 0)
    check("marcar en bloque una imagen ajena no marca nada",
          A.post("/api/favorites/bulk",
                 json={"image_ids": [IB["img"]], "favorite": True}).json()["n"] == 0)
    check("borrar en bloque una imagen ajena no borra nada",
          A.post("/api/album/bulk-delete",
                 json={"image_ids": [IB["img"]]}).json()["deleted"] == 0)

    print("\n7. Nada de la otra cuenta se movio")
    def one(sql, params=()):
        row = db.q1(sql, params)
        return list(row)[0] if row else None

    check("las fotos de cada una siguen enteras",
          one("SELECT COUNT(*) FROM originals WHERE user_id=? AND deleted_at IS NULL",
              (uid_b,)) == len(DB["originals"])
          and one("SELECT COUNT(*) FROM originals WHERE user_id=? AND deleted_at IS NULL",
                  (uid_a,)) == 6)
    check("ninguna foto cuelga del perfil de otra cuenta",
          one("SELECT COUNT(*) FROM originals o JOIN profiles p ON p.id=o.profile_id "
              "WHERE p.user_id<>o.user_id") == 0)
    check("ninguna valoracion apunta a una imagen de otra cuenta",
          one("SELECT COUNT(*) FROM feedback f WHERE EXISTS (SELECT 1 FROM images i "
              "WHERE i.id=f.image_id AND i.user_id<>f.user_id)") == 0)
    check("ningun perfil quedo borrado",
          one("SELECT COUNT(*) FROM profiles WHERE deleted_at IS NOT NULL") == 0)

    print("\n8. Secretos de la instalacion")
    from app import config as cfg
    cfg.set_api_key("fal", "fal-secreto-1234567890abcdef")
    keys_b = B.get("/api/settings").json()["keys"]["fal"]
    keys_a = A.get("/api/settings").json()["keys"]["fal"]
    check("una usuaria normal no ve ni un trozo de la clave de pago",
          keys_b.get("hint") is None and keys_b.get("present") is True,
          json.dumps(keys_b, ensure_ascii=False))
    check("la administradora si ve la pista de su propia clave",
          bool(keys_a.get("hint")), json.dumps(keys_a, ensure_ascii=False))
    cfg.set_api_key("fal", None)
    for path in ("/api/admin/users", "/api/admin/stats", "/api/admin/audit"):
        check("una usuaria normal no entra en %s" % path,
              B.get(path).status_code == 403)
    check("una usuaria normal no puede importar carpetas del servidor",
          B.post("/api/originals/import-folder",
                 json={"path": str(ROOT / "input")}).status_code == 403)

    print("\n9. Coste")
    usage_a = A.get("/api/settings/usage").json()
    usage_b = B.get("/api/settings/usage").json()
    check("no se ha gastado nada en ninguna cuenta",
          float(usage_a["total_usd"]) == 0.0 and float(usage_b["total_usd"]) == 0.0,
          "A %.4f USD, B %.4f USD" % (usage_a["total_usd"], usage_b["total_usd"]))

    return finish(workdir, args.keep)


def finish(workdir: Path, keep: bool) -> int:
    print("\n" + "=" * 78)
    print("RESULTADO: %d correctas, %d fallidas" % (len(PASSED), len(FAILED)))
    print("=" * 78)
    for name in FAILED:
        print("  FALLO: %s" % name)
    if keep:
        print("\nDatos conservados en %s" % workdir)
    else:
        shutil.rmtree(workdir, ignore_errors=True)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())

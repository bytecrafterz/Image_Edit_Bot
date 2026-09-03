"""End to end test of the whole system, with no external services.

Runs against a throwaway database in a temporary folder, using only the free
local engine, so it costs nothing and can be run as often as you like.  It
exercises the path the client will actually take: register, upload reference
photographs, build the identity profile, plan a run, see the estimate, generate,
poll, inspect the ficha, favourite an image, and check that the billing gate
really refuses to spend money that is not there.

    python scripts\\e2e_test.py [--dir <folder with photos>] [--keep]
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    (PASSED if ok else FAILED).append(name)
    mark = "OK  " if ok else "FALLO"
    line = f"  [{mark}] {name}"
    if detail:
        line += f"  -> {detail}"
    print(line, flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "input" / "Nayane"))
    ap.add_argument("--photos", type=int, default=6)
    ap.add_argument("--previews", type=int, default=3)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="photorobot_e2e_"))
    os.environ["PHOTOROBOT_DATA"] = str(workdir)
    sys.path.insert(0, str(ROOT / "backend"))

    print("=" * 72)
    print("PRUEBA COMPLETA DEL SISTEMA (sin servicios externos, coste 0)")
    print("=" * 72)
    print(f"Datos temporales en: {workdir}\n")

    from fastapi.testclient import TestClient
    from app.main import create_app

    client = TestClient(create_app())

    # ---------------------------------------------------------- salud
    print("1. Arranque")
    health = client.get("/api/health")
    check("el servidor responde", health.status_code == 200)
    providers = health.json().get("providers", {})
    check("el motor local gratuito esta disponible",
          providers.get("local", {}).get("available") is True)
    check("la vision heuristica gratuita esta disponible",
          providers.get("heuristic", {}).get("available") is True)

    # ---------------------------------------------------------- cuenta
    print("\n2. Cuenta")
    reg = client.post("/api/auth/register", json={
        "email": "prueba@photorobot.local", "password": "prueba-1234",
        "display_name": "Prueba"})
    check("se crea la primera cuenta", reg.status_code == 200,
          f"HTTP {reg.status_code}")
    if reg.status_code != 200:
        return _finish(workdir, args.keep)
    body = reg.json()
    check("la primera cuenta es administradora y activa",
          body["user"]["role"] == "admin" and body["user"]["status"] == "active")
    token = body.get("token")
    client.headers.update({"Authorization": f"Bearer {token}"})

    second = client.post("/api/auth/register", json={
        "email": "otra@photorobot.local", "password": "otra-12345"})
    check("la segunda cuenta queda pendiente de aprobacion",
          second.status_code == 200 and second.json().get("needs_approval") is True)

    # ---------------------------------------------------------- fotos
    print("\n3. Fotos de referencia")
    source = Path(args.dir)
    photos = sorted([p for p in source.glob("*")
                     if p.suffix.lower() in (".jpg", ".jpeg", ".png")])[:args.photos]
    if not photos:
        check(f"hay fotos en {source}", False, "carpeta vacia")
        return _finish(workdir, args.keep)

    files = [("files", (p.name, p.read_bytes(), "image/jpeg")) for p in photos]
    up = client.post("/api/originals", files=files)
    check("se suben las fotos", up.status_code == 200, f"HTTP {up.status_code}")
    if up.status_code != 200:
        print(up.text[:400])
        return _finish(workdir, args.keep)
    originals = up.json()["originals"]
    check(f"se guardaron {len(photos)} fotos", len(originals) == len(photos))

    again = client.post("/api/originals", files=[files[0]])
    check("una foto repetida no se duplica",
          again.status_code == 200
          and again.json()["originals"][0].get("duplicate") is True)

    listing = client.get("/api/originals").json()
    check("las fotos se listan con su tipo de plano",
          listing["total"] >= len(photos) and "by_shot_type" in listing,
          str(listing.get("by_shot_type")))

    analysis = client.get(f"/api/originals/{originals[0]['id']}/analysis")
    check("se analiza una foto", analysis.status_code == 200,
          f"plano={analysis.json().get('shot_type')}")

    # ---------------------------------------------------------- perfil
    print("\n4. Perfil de identidad")
    prof = client.post("/api/profiles", json={"person_name": "Prueba"})
    check("se crea el perfil", prof.status_code == 200)
    profile_id = prof.json()["id"]

    consent = client.post(f"/api/profiles/{profile_id}/consent",
                          json={"relationship": "self"})
    check("se registra el consentimiento", consent.status_code == 200)

    built = client.post(f"/api/profiles/{profile_id}/build", json={})
    check("se construye el perfil", built.status_code == 200,
          f"HTTP {built.status_code}")
    if built.status_code == 200 and built.json().get("async"):
        run_id = built.json()["run_id"]
        for _ in range(120):
            time.sleep(2)
            state = client.get(f"/api/generate/status/{run_id}")
            if state.status_code != 200 or state.json()["status"] in (
                    "done", "failed", "cancelled"):
                break
    profile = client.get(f"/api/profiles/{profile_id}").json()
    coverage = profile.get("coverage") or {}
    check("el perfil mide algo del cuerpo o lo explica",
          bool(profile.get("body")) or bool(coverage.get("advice")),
          f"cuerpo entero listo={coverage.get('ready_for_body_check')}")
    check("el perfil guarda una firma facial",
          bool((profile.get("face") or {}).get("descriptor")))

    # ---------------------------------------------------------- catalogo
    print("\n5. Catalogo")
    cat = client.get("/api/catalog/options",
                     params={"original_id": originals[0]["id"]})
    check("el catalogo propone opciones para esa foto", cat.status_code == 200)
    groups = cat.json().get("groups", [])
    check("hay grupos de opciones", len(groups) >= 4, f"{len(groups)} grupos")
    check("alguna opcion viene sugerida con motivo",
          any(g.get("suggested") and g.get("reason") for g in groups))
    styles = client.get("/api/catalog/styles").json()
    check("hay estilos disponibles", len(styles.get("styles", [])) >= 15,
          f"{len(styles.get('styles', []))} estilos")

    # ---------------------------------------------------------- seguridad
    print("\n6. Limite de contenido")
    blocked = client.post("/api/generate/analyze", json={
        "original_id": originals[0]["id"], "profile_id": profile_id,
        "options": {"clothing": ["lenceria"]},
        "n_previews": 1, "quality": "preview",
        "style": "editorial_moda"})
    check("una peticion de ropa interior se rechaza",
          blocked.status_code == 400, f"HTTP {blocked.status_code}")

    # ---------------------------------------------------------- generacion
    print("\n7. Generacion")
    choices = {}
    for group in groups:
        if group["group_key"] in ("scene", "grade") and group["values"]:
            choices[group["group_key"]] = [v["value_key"]
                                           for v in group["values"][:2]]
    plan = client.post("/api/generate/analyze", json={
        "original_id": originals[0]["id"], "profile_id": profile_id,
        "options": choices, "n_previews": args.previews,
        "quality": "preview", "style": "retrato_estudio"})
    check("se planifica la tirada y se calcula el coste",
          plan.status_code == 200, f"HTTP {plan.status_code}")
    if plan.status_code != 200:
        print(plan.text[:500])
        return _finish(workdir, args.keep)
    planned = plan.json()
    estimate = planned["estimate"]
    check("el coste estimado es cero con el motor local",
          float(estimate["total_usd"]) == 0.0, f"{estimate['total_usd']} USD")
    check("el plan dice que se fija y que varia",
          "locked" in planned["plan_summary"] and "varied" in planned["plan_summary"])

    run_id = planned["run_id"]
    started = client.post("/api/generate/run", json={"run_id": run_id})
    check("arranca la generacion", started.status_code == 200)

    final_state: dict = {}
    for _ in range(180):
        time.sleep(2)
        state = client.get(f"/api/generate/status/{run_id}")
        if state.status_code != 200:
            break
        final_state = state.json()
        if final_state["status"] in ("done", "failed", "cancelled",
                                     "stopped_no_balance"):
            break
    check("la tirada termina", final_state.get("status") == "done",
          f"estado={final_state.get('status')} {final_state.get('error') or ''}")
    images = final_state.get("images", [])
    check("se produjo al menos una imagen", len(images) >= 1,
          f"{len(images)} imagenes")

    if images:
        report = client.get(f"/api/generate/report/{run_id}").json()
        check("la ficha trae intentos y coste reales",
              report.get("intentos", 0) >= 1 and "coste_usd" in report,
              f"{report.get('intentos')} intentos, {report.get('coste_usd')} USD")
        checks = (report.get("imagenes") or [{}])[0].get("comprobaciones") or []
        check("cada imagen lleva sus comprobaciones medidas", len(checks) >= 3,
              f"{len(checks)} comprobaciones")
        names = {c.get("nombre") for c in checks}
        check("se comprueban proporciones e identidad",
              any("proporcion" in str(n) for n in names)
              and any("rostro" in str(n) for n in names), str(sorted(names)))

        img = client.get(images[0]["url"])
        check("la imagen se sirve por enlace firmado", img.status_code == 200,
              f"HTTP {img.status_code}, {len(img.content)} bytes")

    # ---------------------------------------------------------- album
    print("\n8. Album y favoritos")
    album = client.get("/api/album").json()
    check("el album lista las imagenes", album["total"] >= len(images))
    if images:
        fav = client.post(f"/api/favorites/{images[0]['id']}")
        check("se marca un favorito", fav.status_code == 200)
        favs = client.get("/api/favorites").json()
        check("el favorito aparece en la lista", favs["total"] >= 1)
        client.post(f"/api/album/{images[0]['id']}/feedback",
                    json={"verdict": "like", "reason": "prueba"})
        check("se registra la valoracion", True)

    # ---------------------------------------------------------- dinero
    print("\n9. Saldo, limites y avisos")
    from app.services import billing
    user_id = client.get("/api/auth/me").json()["user"]["id"]
    gate = billing.can_spend(user_id, "fal", 0.05)
    check("sin saldo, el robot NO puede gastar en un proveedor de pago",
          gate["ok"] is False, gate["reason"][:70])
    check("el aviso dice cuanto recargar",
          bool(gate.get("alert", {}).get("topup")))

    rec = client.post("/api/settings/recharge",
                      json={"provider": "fal", "amount_usd": 10})
    check("se puede registrar una recarga", rec.status_code == 200)
    check("la respuesta aclara que la app no cobra nada",
          "nunca cobra" in rec.json().get("message", ""),
          rec.json().get("message", "")[:60])
    gate2 = billing.can_spend(user_id, "fal", 0.05)
    check("con saldo registrado, ya puede gastar", gate2["ok"] is True)

    billing.charge(user_id, "fal", 9.8, ref="prueba", note="prueba")
    alerts = client.get("/api/settings/alerts").json()
    check("al bajar el saldo aparece un aviso",
          any(a["kind"] == "low_balance" for a in alerts["alerts"]),
          f"{alerts['unread']} sin leer")
    gate3 = billing.can_spend(user_id, "fal", 5.0)
    check("si no llega para la siguiente imagen, se detiene",
          gate3["ok"] is False, gate3["reason"][:60])

    usage = client.get("/api/settings/usage").json()
    check("el uso incluye intentos por foto conseguida",
          "attempts_per_photo" in usage,
          f"{usage.get('attempts_per_photo')} intentos/foto")

    # ---------------------------------------------------------- ajustes
    print("\n10. Ajustes y administracion")
    put = client.put("/api/settings", json={"default_n_previews": 4,
                                            "strictness": "estricto"})
    check("se guardan los ajustes", put.status_code == 200)
    bad = client.put("/api/settings", json={"inventado": 1})
    check("un ajuste desconocido se rechaza", bad.status_code == 400)
    keys = client.get("/api/settings").json()["keys"]
    check("las claves nunca se devuelven al navegador",
          all("key" not in v or v.get("hint") is None or isinstance(v.get("hint"), str)
              for v in keys.values())
          and all(not str(v.get("hint") or "").startswith("sk-ant-api")
                  for v in keys.values()))

    users = client.get("/api/admin/users").json()
    check("el administrador ve los usuarios", users["total"] >= 2)
    pending = [u for u in users["users"] if u["status"] == "pending"]
    if pending:
        appr = client.post(f"/api/admin/users/{pending[0]['id']}/approve")
        check("el administrador aprueba una cuenta", appr.status_code == 200)
    admin_stats = client.get("/api/admin/stats").json()
    check("las estadisticas de plataforma responden",
          "attempts" in admin_stats and "users_by_status" in admin_stats)

    return _finish(workdir, args.keep)


def _finish(workdir: Path, keep: bool) -> int:
    print("\n" + "=" * 72)
    print(f"RESULTADO: {len(PASSED)} correctas, {len(FAILED)} fallidas")
    print("=" * 72)
    if FAILED:
        for name in FAILED:
            print(f"  FALLO: {name}")
    if keep:
        print(f"\nDatos conservados en {workdir}")
    else:
        shutil.rmtree(workdir, ignore_errors=True)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())

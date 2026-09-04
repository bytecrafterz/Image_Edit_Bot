"""Rehearsal of the PAID path, without paying: ensayo del camino de pago.

The keyed path is the half of this product that could only ever be tested by
buying it.  The 15 fal images this installation owns cost 0.040-0.080 USD each,
and every plumbing defect found in them - the square 1:1 box that reframed her
body, three different prices for one image, a reservation that was not the bill
- was discovered after the invoice, on images that were already paid for.  This
script walks the whole of that path end to end and spends nothing:

  * ``PHOTOROBOT_FAL_REPLAY`` (see providers/fal, MODO ENSAYO) makes the fal
    provider build the real payload and then answer with an image from a folder
    instead of calling the queue API;
  * ``PHOTOROBOT_DATA`` points at a throwaway folder, so the real database, the
    real album and the real keystore are never opened;
  * the fal key used is a fake one, written through the same settings endpoint
    the client uses;
  * every outbound TCP connection is blocked at the socket, so the run cannot
    reach fal.ai or anybody else even by accident - and the blocked attempts
    are counted and reported at the end.

What it proves, with numbers, is that the estimate the client is shown equals
what her ledger settles, that every reservation is closed, that her skin grain
is measured and given back, that the gate judges each image, that a repairable
defect really opens a repair round, and that the accepted image reaches the
album with its verdict and its ficha.

    python scripts\\rehearse_paid.py [--quality high] [--variants 3] [--keep]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PASSED: list[str] = []
FAILED: list[str] = []

# A fake key, and one that says so in the only place anybody will read it: the
# masked hint on the settings page shows the first six characters.
FAKE_KEY = "ENSAYO-clave-falsa-sin-valor-0000000000"
# Her full body photographs.  These are the ones the paid images on disk were
# made from, so the gate compares a replayed render against the very photograph
# it came from, which is the honest version of this rehearsal.
PREFERRED = ("IMG_7871", "IMG_8825", "IMG_8898")


def check(name: str, ok: bool, detail: str = "") -> bool:
    (PASSED if ok else FAILED).append(name)
    line = "  [%s] %s" % ("OK  " if ok else "FALLO", name)
    if detail:
        line += "  -> %s" % detail
    print(line, flush=True)
    return ok


def money(value) -> str:
    return "%.4f" % float(value or 0.0)


# --------------------------------------------------------------- red cerrada

_blocked: list[str] = []


def block_network() -> None:
    """Refuse every connection that is not the loopback.

    The rehearsal has to be safe to run on the client's own machine, with her
    real key sitting in the keystore, so "it does not call the network" cannot
    be a promise made by the code under test - it is enforced from outside it.
    OSError is raised on purpose: httpx maps it to ConnectError, which the
    settings page already handles as "guardada, no se pudo verificar ahora", so
    what gets rehearsed is the branch an installation without internet takes.
    """
    real_connect = socket.socket.connect
    real_create = socket.create_connection

    def guard(host) -> None:
        if str(host) not in ("127.0.0.1", "::1", "localhost", ""):
            _blocked.append(str(host))
            raise OSError("ENSAYO: conexion a %s bloqueada; este ensayo no "
                          "sale de la maquina." % host)

    def connect(self, address, *args, **kwargs):
        guard(address[0] if isinstance(address, tuple) else address)
        return real_connect(self, address, *args, **kwargs)

    def create_connection(address, *args, **kwargs):
        guard(address[0] if isinstance(address, (tuple, list)) else address)
        return real_create(address, *args, **kwargs)

    socket.socket.connect = connect                      # type: ignore[assignment]
    socket.create_connection = create_connection         # type: ignore[assignment]


# ------------------------------------------------------------------- espias
# The rehearsal records what happened without changing it: every wrapper calls
# the real function and stores its arguments and its answer.  The numbers that
# never reach the database live only here - which endpoint was asked for, with
# which aspect ratio, whether a reservation was settled or given back, and what
# the repair round decided.

_lock = threading.Lock()
CALLS: list[dict] = []
MONEY: list[dict] = []
REPAIRS: list[dict] = []


def install_spies() -> None:
    from app.providers.fal import FalProvider
    from app.services import billing
    from app.generation import repair as repair_mod

    real_generate = FalProvider.generate

    def generate(self, req, out_path):
        started = time.time()
        result = real_generate(self, req, out_path)
        meta = result.meta or {}
        with _lock:
            CALLS.append({
                "operation": req.operation,
                "quality": req.quality,
                "pedido": "%dx%d" % (req.width, req.height),
                "aspect": meta.get("aspect_ratio", "-"),
                "endpoint": result.model,
                "entregado": "%sx%s" % (meta.get("width"), meta.get("height")),
                "aspect_dado": meta.get("delivered_aspect", "-"),
                "coste": float(result.cost_usd or 0.0),
                "archivo": meta.get("replay_file", "-"),
                "ensayo": bool(meta.get("replay")),
                "salida": Path(str(out_path)).name,
                "ms": int((time.time() - started) * 1000),
            })
        return result

    FalProvider.generate = generate                      # type: ignore[assignment]

    for verb in ("reserve", "settle", "release"):
        real = getattr(billing, verb)

        def make(verb=verb, real=real):
            def wrapper(*args, **kwargs):
                out = real(*args, **kwargs)
                row = {"que": verb, "ref": kwargs.get("ref", "")}
                if verb == "reserve":
                    row["importe"] = float(args[2] if len(args) > 2 else 0.0)
                    row["ok"] = bool(out.get("ok"))
                    row["hold"] = out.get("hold_id", "")
                elif verb == "settle":
                    row["importe"] = float(args[3] if len(args) > 3 else 0.0)
                    row["hold"] = args[2] if len(args) > 2 else ""
                else:
                    row["importe"] = float(out or 0.0)
                    row["hold"] = args[0] if args else ""
                with _lock:
                    MONEY.append(row)
                return out
            return wrapper

        setattr(billing, verb, make())

    real_repair = repair_mod.repair

    def repair(image_path, defects, brief, profile, provider, out_path):
        out = real_repair(image_path, defects, brief, profile, provider,
                          out_path)
        with _lock:
            REPAIRS.append({
                "defectos": [d.get("type") for d in (defects or [])],
                "zonas": out.get("regions"), "rondas": out.get("rounds"),
                "ok": bool(out.get("ok")),
                "coste": float(out.get("cost_usd") or 0.0),
                "motivo": out.get("reason") or "",
                "notas": list(out.get("notes") or []),
            })
        return out

    repair_mod.repair = repair                           # type: ignore[assignment]


# ------------------------------------------------------------ imagenes fal

def collect_replay(target: Path, explicit: str | None,
                   photos: list[Path]) -> tuple[list[Path], str]:
    """The images the rehearsal hands back instead of buying new ones.

    Order of preference: a folder the caller names; the paid images this
    installation already owns, read from the REAL database read-only, because
    they are the only files that behave like what fal returns - about one
    megapixel, skin sanded, the same person; and only then her own photographs,
    which still exercise every step but compare the gate against an unaltered
    original.  Which one was used is returned, because it changes what the
    numbers below mean.
    """
    target.mkdir(parents=True, exist_ok=True)
    if explicit:
        found = sorted(p for p in Path(explicit).iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        for i, src in enumerate(found):
            shutil.copyfile(src, target / ("fal_%02d%s" % (i, src.suffix.lower())))
        return sorted(target.iterdir()), "carpeta indicada: %s" % explicit

    db_path = ROOT / "data" / "photorobot.sqlite3"
    rows: list[str] = []
    if db_path.exists():
        con = sqlite3.connect("file:%s?mode=ro" % db_path.as_posix(), uri=True)
        try:
            rows = [r[0] for r in con.execute(
                "SELECT path FROM images WHERE provider='fal' "
                "AND deleted_at IS NULL ORDER BY created_at")]
        except sqlite3.Error:
            rows = []
        finally:
            con.close()
    kept = [Path(p) for p in rows if Path(p).exists()]
    for i, src in enumerate(kept):
        shutil.copyfile(src, target / ("fal_%02d.jpg" % i))
    if kept:
        return sorted(target.iterdir()), "imagenes de fal ya compradas"

    for i, src in enumerate(photos[:6]):
        shutil.copyfile(src, target / ("src_%02d.jpg" % i))
    return sorted(target.iterdir()), "sus propias fotos (no hay imagenes de fal)"


# -------------------------------------------------------------------- ensayo

def main() -> int:
    ap = argparse.ArgumentParser(
        prog="rehearse_paid.py",
        description="Ensayo del camino de pago: recorre entero el trabajo con "
                    "clave (fal.ai) SIN GASTAR NI UN CENTIMO.",
        epilog="Coste: 0 USD, siempre. No se llama a ningun servicio de pago. "
               "El proveedor fal responde con imagenes de una carpeta (modo "
               "ensayo), la clave que se usa es falsa, la base de datos y el "
               "album son temporales y se borran al terminar, y toda conexion "
               "de red que no sea a la propia maquina esta bloqueada y se "
               "cuenta. Sirve para comprobar que, cuando pongas tu clave de "
               "verdad, el camino de pago funciona a la primera.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=str(ROOT / "input" / "Nayane"),
                    help="carpeta con sus fotos de referencia")
    ap.add_argument("--photos", type=int, default=3,
                    help="cuantas fotos se importan (por defecto 3)")
    ap.add_argument("--variants", type=int, default=3,
                    help="cuantas vistas previas se piden (por defecto 3)")
    ap.add_argument("--quality", default="high",
                    choices=("preview", "standard", "high", "max"),
                    help="nivel de calidad a ensayar (por defecto alta)")
    ap.add_argument("--clothing", default="traje_sastre",
                    help="cambio de ropa: es lo que obliga a usar el motor de "
                         "pago, porque el local no sabe cambiar la ropa")
    ap.add_argument("--recharge", type=float, default=5.0,
                    help="saldo ficticio que se registra antes de empezar")
    ap.add_argument("--replay-dir", default=None,
                    help="carpeta de imagenes que devolvera el modo ensayo; "
                         "por defecto, las imagenes de fal ya existentes")
    ap.add_argument("--keep", action="store_true",
                    help="no borrar la carpeta temporal al terminar")
    args = ap.parse_args()

    workdir = Path(tempfile.mkdtemp(prefix="photorobot_ensayo_"))
    replay = workdir / "replay"
    os.environ["PHOTOROBOT_DATA"] = str(workdir / "data")
    # A real key in the environment would win over the temporary keystore, and
    # this script has to be safe to run on the client's own machine: the keys
    # are taken out of this process before anything is loaded.
    for name in ("FAL_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        os.environ.pop(name, None)
    sys.path.insert(0, str(ROOT / "backend"))

    print("=" * 74)
    print("ENSAYO DEL CAMINO DE PAGO - coste real 0.00 USD")
    print("=" * 74)
    print("Datos temporales: %s" % workdir)

    photos = sorted(p for p in Path(args.dir).glob("*")
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    files, origin = collect_replay(replay, args.replay_dir, photos)
    os.environ["PHOTOROBOT_FAL_REPLAY"] = str(replay)
    block_network()
    print("Carpeta de ensayo: %d imagenes (%s)" % (len(files), origin))
    print("Red saliente: bloqueada\n")

    from fastapi.testclient import TestClient
    from app.main import create_app
    install_spies()
    from app import db
    from app.services import billing
    from app.providers import fal as fal_mod

    client = TestClient(create_app())

    # ------------------------------------------------------------ 1. arranque
    print("1. Arranque y modo ensayo")
    health = client.get("/api/health").json()
    check("el servidor responde", bool(health.get("ok")))
    check("el modo ensayo esta activo", fal_mod.replay_dir() is not None,
          "%s=%s" % (fal_mod.REPLAY_ENV, replay))
    check("fal aun NO esta disponible: no hay clave",
          health["providers"]["fal"]["available"] is False)

    # -------------------------------------------------------------- 2. cuenta
    print("\n2. Cuenta y fotos")
    reg = client.post("/api/auth/register", json={
        "email": "ensayo@photorobot.local", "password": "ensayo-1234",
        "display_name": "Ensayo"})
    if not check("se crea la cuenta", reg.status_code == 200,
                 "HTTP %d" % reg.status_code):
        return finish(workdir, args.keep)
    user_id = reg.json()["user"]["id"]
    client.headers.update({"Authorization": "Bearer %s" % reg.json()["token"]})

    wanted = [p for p in photos if any(k in p.name for k in PREFERRED)]
    chosen = (wanted or photos)[:max(1, args.photos)]
    if not chosen:
        check("hay fotos en %s" % args.dir, False, "carpeta vacia")
        return finish(workdir, args.keep)
    up = client.post("/api/originals", files=[
        ("files", (p.name, p.read_bytes(), "image/jpeg")) for p in chosen])
    if not check("se importan sus fotos", up.status_code == 200,
                 ", ".join(p.name for p in chosen)):
        print(up.text[:400])
        return finish(workdir, args.keep)
    originals = up.json()["originals"]
    source = originals[0]
    print("     foto de origen: %s  %sx%s  plano=%s"
          % (source.get("filename"), source.get("width"), source.get("height"),
             source.get("shot_type")))

    # -------------------------------------------------------------- 3. perfil
    print("\n3. Perfil de identidad")
    profile_id = client.post("/api/profiles",
                             json={"person_name": "Ensayo"}).json()["id"]
    client.post("/api/profiles/%s/consent" % profile_id,
                json={"relationship": "self"})
    built = client.post("/api/profiles/%s/build" % profile_id, json={})
    if built.status_code == 200 and built.json().get("async"):
        for _ in range(150):
            time.sleep(2)
            state = client.get("/api/generate/status/%s" % built.json()["run_id"])
            if state.status_code != 200 or state.json()["status"] in (
                    "done", "failed", "cancelled"):
                break
    profile = client.get("/api/profiles/%s" % profile_id).json()
    coverage = profile.get("coverage") or {}
    check("el perfil guarda su firma facial",
          bool((profile.get("face") or {}).get("descriptor")))
    check("el perfil mide su cuerpo", bool(profile.get("body")),
          "cuerpo entero listo=%s" % coverage.get("ready_for_body_check"))

    # ---------------------------------------------------------- 4. clave falsa
    print("\n4. Clave de fal (falsa) por la via normal de Ajustes")
    key_resp = client.post("/api/settings/keys",
                           json={"provider": "fal", "key": FAKE_KEY})
    check("la clave se guarda por POST /api/settings/keys",
          key_resp.status_code == 200,
          key_resp.json().get("message", key_resp.text[:120]))
    keys = client.get("/api/settings").json()["keys"]
    check("la clave nunca vuelve entera al navegador",
          keys["fal"]["present"] is True and FAKE_KEY not in json.dumps(keys),
          "pista=%s" % keys["fal"].get("hint"))
    check("ahora fal esta disponible",
          client.get("/api/health").json()["providers"]["fal"]["available"]
          is True)
    real_store = ROOT / "data" / "keystore.json"
    check("la clave falsa vive solo en la carpeta temporal",
          (workdir / "data" / "keystore.json").exists()
          and (not real_store.exists()
               or FAKE_KEY not in real_store.read_text(encoding="utf-8")))

    # --------------------------------------------------------------- 5. saldo
    print("\n5. Saldo")
    gate = billing.can_spend(user_id, "fal", 0.08)
    check("sin saldo, el robot no puede gastar en fal", gate["ok"] is False,
          gate["reason"][:70])
    client.post("/api/settings/recharge",
                json={"provider": "fal", "amount_usd": args.recharge})
    balance_before = billing.balance(user_id, "fal")
    check("se registra el saldo (la app no cobra nada)",
          abs(balance_before - args.recharge) < 1e-9,
          "%s USD" % money(balance_before))
    settings = client.get("/api/settings").json()
    print("     precio por imagen que anuncia Ajustes: %s"
          % json.dumps(settings.get("prices") or {}))

    # ---------------------------------------------------------------- 6. plan
    print("\n6. Plan y estimacion (calidad %s, con cambio de ropa)"
          % args.quality)
    plan = client.post("/api/generate/analyze", json={
        "original_id": source["id"], "profile_id": profile_id,
        "options": {"clothing": [args.clothing]},
        "n_previews": args.variants, "quality": args.quality,
        "style": "editorial_moda"})
    if not check("se planifica y se cotiza", plan.status_code == 200,
                 "HTTP %d" % plan.status_code):
        print(plan.text[:600])
        return finish(workdir, args.keep)
    planned = plan.json()
    est = planned["estimate"]
    run_id = planned["run_id"]
    print("     proveedor=%s modelo=%s" % (est["provider"], est["model"]))
    print("     %d variante(s) x %s USD  + factor %.2f  = %s USD"
          % (est["n_images"], money(est["per_image_usd"]), est["factor"],
             money(est["total_usd"])))
    print("     motivo: %s" % est["reason"])
    for warn in planned.get("warnings") or []:
        print("     aviso: %s" % warn)
    check("la tirada se enruta al motor de PAGO", est["provider"] == "fal",
          "%s (%s)" % (est["provider"], est["model"]))
    check("la calidad dice cuantos pixeles puede entregar de verdad",
          bool(est.get("aviso_resolucion"))
          or args.quality in ("preview", "standard"),
          (est.get("aviso_resolucion") or "-")[:80])

    # ------------------------------------------------------------ 7. ejecucion
    print("\n7. Ejecucion (cada imagen sale de la carpeta de ensayo)")
    started = time.time()
    client.post("/api/generate/run", json={"run_id": run_id})
    state: dict = {}
    for _ in range(300):
        time.sleep(2)
        resp = client.get("/api/generate/status/%s" % run_id)
        if resp.status_code != 200:
            break
        state = resp.json()
        if state["status"] in ("done", "failed", "cancelled",
                               "stopped_no_balance"):
            break
    seconds = time.time() - started
    check("la tirada termina", state.get("status") == "done",
          "estado=%s %s" % (state.get("status"), state.get("error") or ""))
    print("     %.1f s, %d intentos, %d aceptadas, %d descartadas"
          % (seconds, state.get("attempts", 0), state.get("accepted", 0),
             state.get("rejected", 0)))

    # ----------------------------------------------------- 8. lo que se envio
    print("\n8. Lo que se envio a fal (y lo que habria costado)")
    print("     %-9s %-28s %-10s %-6s %-10s %-6s %-8s %s"
          % ("operacion", "endpoint", "pedido", "aspec", "entregado", "dado",
             "USD", "archivo"))
    for call in CALLS:
        print("     %-9s %-28s %-10s %-6s %-10s %-6s %-8s %s"
              % (call["operation"], call["endpoint"], call["pedido"],
                 call["aspect"], call["entregado"], call["aspect_dado"],
                 money(call["coste"]), call["archivo"]))
    check("ninguna llamada salio a la red",
          bool(CALLS) and all(c["ensayo"] for c in CALLS),
          "%d llamadas, todas en modo ensayo" % len(CALLS))
    check("se pidio SU forma, no un cuadrado",
          bool(CALLS) and all(c["aspect"] in ("3:4", "-") for c in CALLS),
          "aspectos pedidos: %s" % sorted({c["aspect"] for c in CALLS}))
    # Asking for the right shape and never looking at the answer is how the
    # square survived 23 paid images: both halves are recorded now, so a run
    # can be audited from its own rows.  The shape DELIVERED here is the shape
    # of a file copied out of the ensayo folder, not fal's, so it is printed
    # and not judged.
    with_aspect = [c for c in CALLS if c["aspect"] != "-"]
    check("de cada llamada consta la forma pedida y la entregada",
          bool(with_aspect) and all(c["aspect_dado"] != "-" for c in with_aspect),
          "%d de %d llamadas llevan las dos formas"
          % (len(with_aspect), len(CALLS)))

    # ------------------------------------------------------------- 9. dinero
    print("\n9. Dinero: lo cotizado, lo retenido y lo liquidado")
    reserves = [m for m in MONEY if m["que"] == "reserve" and m["ok"]]
    settles = [m for m in MONEY if m["que"] == "settle"]
    releases = [m for m in MONEY if m["que"] == "release" and m["importe"] > 0]
    gen_reserves = [m for m in reserves if ":repair" not in m["ref"]]
    settled = round(sum(m["importe"] for m in settles), 6)
    ledger = db.rows_to_dicts(db.q(
        "SELECT kind, amount_usd, ref, note FROM ledger WHERE user_id=? "
        "ORDER BY created_at", (user_id,)))
    spends = [row for row in ledger if row["kind"] == "spend"]
    balance_after = billing.balance(user_id, "fal")
    print("     %-26s %s" % ("estimado (con factor):", money(est["total_usd"])))
    print("     %-26s %s x %d = %s"
          % ("cotizado por imagen:", money(est["per_image_usd"]),
             est["n_images"], money(est["per_image_usd"] * est["n_images"])))
    print("     %-26s %s en %d reservas"
          % ("retenido:", money(sum(m["importe"] for m in reserves)),
             len(reserves)))
    print("     %-26s %s en %d liquidaciones"
          % ("liquidado:", money(settled), len(settles)))
    print("     %-26s %s en %d devoluciones"
          % ("devuelto:", money(sum(m["importe"] for m in releases)),
             len(releases)))
    print("     %-26s %s -> %s" % ("saldo:", money(balance_before),
                                   money(balance_after)))
    gen_calls = [c for c in CALLS if c["operation"] != "inpaint"]
    rep_calls = [c for c in CALLS if c["operation"] == "inpaint"]
    print("     de donde sale: %d generacion(es) x %s + %d reparacion(es) x %s "
          "= %s USD de llamadas reales"
          % (len(gen_calls), money(gen_calls[0]["coste"] if gen_calls else 0),
             len(rep_calls), money(rep_calls[0]["coste"] if rep_calls else 0),
             money(sum(c["coste"] for c in CALLS))))
    # Per reservation, not in total: a run whose holds add up can still contain
    # one call that spent more than the gate approved for it, and that single
    # call is how a balance goes negative on a provider that bills per request.
    held_by_ref: dict[str, float] = {}
    paid_by_ref: dict[str, float] = {}
    for row in reserves:
        held_by_ref[row["ref"]] = held_by_ref.get(row["ref"], 0.0) + row["importe"]
    for row in settles:
        paid_by_ref[row["ref"]] = paid_by_ref.get(row["ref"], 0.0) + row["importe"]
    overruns = [(ref, held_by_ref.get(ref, 0.0), spent)
                for ref, spent in paid_by_ref.items()
                if spent > held_by_ref.get(ref, 0.0) + 1e-9]
    check("ninguna llamada gasta mas de lo que reservo", not overruns,
          "; ".join("%s cobro %s con %s retenido" % (r, money(s), money(h))
                    for r, h, s in overruns[:3]) or
          "%s retenidos cubren %s gastados" % (money(sum(
              m["importe"] for m in reserves)), money(settled)))
    check("cada generacion se reserva por el precio cotizado",
          bool(gen_reserves) and all(
              abs(m["importe"] - est["per_image_usd"]) < 1e-9
              for m in gen_reserves),
          "reservas de generacion: %s"
          % sorted({money(m["importe"]) for m in gen_reserves}))
    check("lo liquidado es lo que dice el libro mayor",
          abs(settled - abs(sum(r["amount_usd"] for r in spends))) < 1e-6,
          "%s USD en %d apuntes" % (money(settled), len(spends)))
    # And the ledger against the ENGINE, which is the only comparison that can
    # catch money leaving without being written down: the check above compares
    # the ledger with itself, so it stayed green while 0.0500 USD of a 0.6500
    # USD run went unrecorded (a reverted repaint) and again while 0.0500 USD
    # of a 0.2900 USD run went unrecorded (the zones already painted when the
    # account ran dry, 2026-09-03).  fal charges per request, so every call in
    # CALLS is an invoice line and their sum is what her card will say.
    charged = round(sum(c["coste"] for c in CALLS), 6)
    check("cada llamada al proveedor llega al libro mayor",
          abs(charged - settled) < 1e-6,
          "%s USD cobrados por %d llamada(s), %s USD en el libro mayor%s"
          % (money(charged), len(CALLS), money(settled),
             "" if abs(charged - settled) < 1e-6
             else "  <- faltan %s USD" % money(charged - settled)))
    check("no queda ninguna retencion abierta",
          abs(billing.held(user_id)) < 1e-9,
          "%s USD" % money(billing.held(user_id)))
    check("el saldo cuadra: recarga - gastado",
          abs(balance_after - (args.recharge - settled)) < 1e-6,
          "%s = %s - %s" % (money(balance_after), money(args.recharge),
                            money(settled)))
    # The two numbers the client reads before pressing the button.  The
    # estimate is a typical cost - one image in three is repeated and
    # repainted - so a run that is rejected more often than that will exceed
    # it, and demanding otherwise would only push the quote up for everybody.
    # What must never be exceeded is the ceiling the estimate now carries next
    # to it: every attempt the limits allow, repainted in as many zones as
    # repair.MAX_REGIONS permits.  Measured on 2026-09-03, before the estimate
    # priced repairs at all, this run settled 0.6500 USD against a 0.3240 USD
    # quote with no ceiling shown anywhere.
    ceiling = float(est.get("total_max_usd") or 0.0)
    check("el gasto real cabe en el techo anunciado",
          ceiling > 0.0 and settled <= ceiling + 1e-9,
          "%s gastados, %s estimados, techo %s"
          % (money(settled), money(est["total_usd"]), money(ceiling)))
    if settled > est["total_usd"] + 1e-9:
        print("     nota: %s gastados frente a %s estimados (%+.0f%%): mas "
              "rechazos que 1 de cada 3; el aviso de coste ya lo anticipa."
              % (money(settled), money(est["total_usd"]),
                 100.0 * (settled / max(1e-9, est["total_usd"]) - 1.0)))

    # ------------------------------------------------------------- 10. ficha
    print("\n10. Textura, veredicto y reparacion")
    report = client.get("/api/generate/report/%s" % run_id).json()
    attempts = db.rows_to_dicts(db.q(
        "SELECT * FROM attempts WHERE run_id=? ORDER BY variant_index, "
        "attempt_no", (run_id,)))
    print("     %-3s %-3s %-9s %-6s %-7s %-6s %s"
          % ("v", "int", "estado", "punt", "textura", "gan", "motivo/resumen"))
    textures = 0
    for att in attempts:
        tex = (att.get("params") or {}).get("textura") or {}
        verdict = att.get("verdict") or {}
        if tex.get("aplicada"):
            textures += 1
        print("     %-3s %-3s %-9s %-6s %-7s %-6s %s"
              % (att["variant_index"], att["attempt_no"], att["status"],
                 round(float(verdict.get("score") or 0.0), 3),
                 "si" if tex.get("aplicada") else "no",
                 tex.get("ganancia", "-"),
                 (att.get("reject_reason") or verdict.get("summary")
                  or "")[:60]))
    check("la textura de su piel se mide en cada imagen",
          bool(attempts) and all(
              isinstance((a.get("params") or {}).get("textura"), dict)
              for a in attempts if a["status"] != "error"),
          "devuelta en %d de %d intentos" % (textures, len(attempts)))
    for att in attempts[:1]:
        for chk in (att.get("verdict") or {}).get("checks") or []:
            print("       %-20s valor=%-8s limite=%-8s paso=%s"
                  % (chk.get("name"), chk.get("value"), chk.get("threshold"),
                     chk.get("passed")))
    check("la puerta juzga cada imagen con sus numeros",
          bool(attempts) and all(
              len((a.get("verdict") or {}).get("checks") or []) >= 4
              for a in attempts if a["status"] in ("accepted", "rejected")),
          "%d intentos con veredicto" % sum(1 for a in attempts
                                            if a.get("verdict")))
    for rep in REPAIRS:
        print("     reparacion: defectos=%s zonas=%s rondas=%s ok=%s coste=%s"
              % (rep["defectos"], rep["zonas"], rep["rondas"], rep["ok"],
                 money(rep["coste"])))
        for note in (rep["notas"] or [rep["motivo"]])[:3]:
            print("       %s" % note)
    check("un defecto abre una ronda de reparacion", bool(REPAIRS),
          "%d rondas, %d con mejora aceptada"
          % (len(REPAIRS), sum(1 for r in REPAIRS if r["ok"])))
    repair_holds = [m for m in MONEY
                    if m["que"] == "reserve" and ":repair" in m["ref"]]
    if repair_holds:
        print("     la reparacion se reserva a su propio precio %s USD "
              "(inpaint); la generacion, a %s USD"
              % (money(repair_holds[0]["importe"]), money(est["per_image_usd"])))

    # ------------------------------------------------------------- 11. album
    print("\n11. Album y ficha")
    album = client.get("/api/album").json()
    images = album.get("images") or []
    for img in images:
        row = db.q1("SELECT meta_json FROM images WHERE id=?", (img["id"],))
        meta = json.loads(row["meta_json"]) if row else {}
        print("     %s  %sx%s  punt=%s  %s:%s  ensayo=%s  %s"
              % (img["id"][:16], img["width"], img["height"], img["score"],
                 img["provider"], (img["model"] or "").split("/")[-1],
                 meta.get("ensayo"), (img.get("summary") or "")[:44]))
    check("la imagen llega al album con su veredicto",
          album["total"] >= 1 and all(i.get("summary") for i in images),
          "%d imagenes" % album["total"])
    check("cada imagen del ensayo queda marcada como ensayo",
          bool(images) and all(
              json.loads(db.q1("SELECT meta_json FROM images WHERE id=?",
                               (i["id"],))["meta_json"]).get("ensayo") is True
              for i in images),
          "no se pueden confundir con imagenes compradas")
    if images:
        served = client.get(images[0]["url"])
        check("la imagen se sirve por enlace firmado",
              served.status_code == 200, "%d bytes" % len(served.content))
    print("     ficha: %s intentos, %s aceptadas, %s descartadas, %s USD, %s "
          "intentos/foto"
          % (report.get("intentos"), report.get("aceptadas"),
             report.get("descartadas"), money(report.get("coste_usd")),
             report.get("intentos_por_foto")))
    print("     modelos: %s" % ", ".join(report.get("modelos") or []))
    print("     textura restaurada en %s imagen(es), ganancia media %s"
          % (report.get("textura_restaurada"), report.get("textura_ganancia")))
    if report.get("defectos_detectados"):
        print("     defectos: %s"
              % json.dumps(report["defectos_detectados"], ensure_ascii=False))
    for aviso in (report.get("avisos") or [])[:4]:
        print("     aviso: %s" % aviso[:100])
    check("la ficha se genera con datos reales",
          bool(report.get("ok")) and report.get("intentos", 0) >= 1
          and report.get("imagenes") is not None)
    check("lo que vio estimado es lo que se liquido",
          abs(float(report.get("coste_usd") or 0.0) - settled) < 1e-6,
          "ficha %s USD = libro mayor %s USD"
          % (money(report.get("coste_usd")), money(settled)))

    # --------------------------------------------------------------- 12. cero
    print("\n12. Coste real del ensayo")
    print("     conexiones bloqueadas: %d %s"
          % (len(_blocked), sorted(set(_blocked))))
    check("no se gasto dinero real: clave falsa, imagenes de disco, red cerrada",
          bool(CALLS) and all(c["ensayo"] for c in CALLS))
    return finish(workdir, args.keep)


def finish(workdir: Path, keep: bool) -> int:
    print("\n" + "=" * 74)
    print("RESULTADO: %d correctas, %d fallidas   |   COSTE REAL: 0.0000 USD"
          % (len(PASSED), len(FAILED)))
    print("=" * 74)
    for name in FAILED:
        print("  FALLO: %s" % name)
    if keep:
        print("\nDatos conservados en %s" % workdir)
    else:
        shutil.rmtree(workdir, ignore_errors=True)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())

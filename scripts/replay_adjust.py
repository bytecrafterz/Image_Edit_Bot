"""Drive the retry loop through every failure it can meet.  Spends 0.00 USD.

    backend\\.venv\\Scripts\\python.exe scripts\\replay_adjust.py

WHAT IT PROVES.  ``orchestrator._run_variant`` is the function that spends the
client's money, and the thing being changed inside it - "when an image is
rejected, work out WHICH option caused it and change that one" - can only be
trusted if it is watched doing it.  So this harness runs the real function, with
the real risk counters, the real billing ledger, the real attempt rows and the
real prompt builder, against a provider that answers from a script instead of
from fal.  For each scenario it checks four things:

  * the change chosen is the one the RECORD supports, named and quoted;
  * at most ONE adjusted attempt is ever bought;
  * the money settled equals the money the fake provider charged, and the
    ledger moves by exactly that;
  * the learning row is written - including when the adjustment did not work,
    which is what lets a bad idea be dropped instead of repeated.

WHY A FAKE PROVIDER AND NOT THE REPLAY FOLDER.  ``PHOTOROBOT_FAL_REPLAY``
answers with a real image file, which is right for rehearsing the payment path
but cannot produce a chosen verdict: the four cases that matter here are "the
face is not hers", "the provider kept the money and sent a black file", "the
hand is wrong" and "it worked", and three of those are properties of the
VERDICT, not of the bytes.  So the verdict is scripted and everything that
decides what to do about it is real.

NOTHING TOUCHES THE INSTALLATION.  ``PHOTOROBOT_DATA`` is a throwaway folder
under the system temp directory, outbound sockets are blocked before the app is
imported, and any key in the environment is removed first.
"""
from __future__ import annotations

import os
import shutil
import socket
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PASS: list[str] = []
FAIL: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    (PASS if ok else FAIL).append(label)
    print("   %s %s%s" % ("[ok]" if ok else "[NO]", label,
                          ("  -  %s" % detail) if detail else ""))
    return bool(ok)


def block_network() -> None:
    """Refuse every connection that is not the loopback.

    Enforced from outside the code under test, exactly as scripts/rehearse_paid
    does it, and for the same reason: "it does not call fal" must not be a
    promise the thing being tested makes about itself.  Only ``connect`` is
    replaced - swapping the socket class itself breaks the standard library's
    own imports (ssl subclasses it) and would stop the app loading at all.
    """
    real_connect = socket.socket.connect
    real_create = socket.create_connection

    def guard(host) -> None:
        if str(host) not in ("127.0.0.1", "::1", "localhost", ""):
            raise OSError("ENSAYO: conexion a %s bloqueada; este ensayo no "
                          "sale de la maquina." % host)

    def connect(self, address, *args, **kwargs):
        guard(address[0] if isinstance(address, tuple) else address)
        return real_connect(self, address, *args, **kwargs)

    def create_connection(address, *args, **kwargs):
        guard(address[0] if isinstance(address, (tuple, list)) else address)
        return real_create(address, *args, **kwargs)

    socket.socket.connect = connect                       # type: ignore
    socket.create_connection = create_connection          # type: ignore


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="photorobot_ajuste_"))
    os.environ["PHOTOROBOT_DATA"] = str(work / "data")
    for name in ("FAL_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        os.environ.pop(name, None)
    sys.path.insert(0, str(ROOT / "backend"))
    block_network()

    from app import db
    from app.providers.base import Capabilities, GenResult, ProviderError
    from app.services import billing
    from app.generation import adjust as adjust_mod
    from app.generation import orchestrator as orch
    from app.generation import risk as risk_mod

    print("=" * 78)
    print("ENSAYO DEL AJUSTE TRAS UN RECHAZO - coste real 0.00 USD")
    print("=" * 78)
    print("Datos temporales: %s" % work)
    print("Red saliente: bloqueada\n")

    # ------------------------------------------------------------ the account
    photo = work / "foto.jpg"
    photo.write_bytes(_tiny_jpeg())
    now = time.time()
    user_id, profile_id, original_id = "usr_ensayo", "prf_ensayo", "org_ensayo"
    db.execute("INSERT INTO users(id,email,password_hash,display_name,status,"
               "daily_limit_usd,monthly_limit_usd,created_at) "
               "VALUES(?,?,?,?,?,?,?,?)",
               (user_id, "ajuste@local", "x", "Ensayo", "active", 50.0, 50.0, now))
    db.execute("INSERT INTO profiles(id,user_id,person_name,status,created_at,"
               "updated_at) VALUES(?,?,?,?,?,?)",
               (profile_id, user_id, "Ensayo", "ready", now, now))
    db.execute("INSERT INTO originals(id,user_id,profile_id,filename,path,"
               "width,height,bytes,sha256,shot_type,created_at) "
               "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
               (original_id, user_id, profile_id, "foto.jpg", str(photo),
                1024, 1536, photo.stat().st_size, "sha", "full_body", now))
    billing.recharge(user_id, "fal", 5.0, note="ensayo")

    # THE SEED THE RULES ARE READ FROM.  ``shared_pool`` backfills from the
    # installation's own attempts table when there is one and falls back to
    # risk.PRIOR when there is not; this database is empty, so what the robot
    # knows here is exactly the anonymous seed that ships - 74 paid calls, no
    # person in it.  Every "the measurement supports it" claim below is a claim
    # about that table and can be checked against it.
    pool = risk_mod.shared_pool()
    print("Memoria de partida: %d claves (semilla anonima de risk.PRIOR)\n"
          % len([k for k in pool if not k.startswith("_")]))

    profile = db.row_to_dict(db.q1("SELECT * FROM profiles WHERE id=?",
                                   (profile_id,)))
    user = db.row_to_dict(db.q1("SELECT * FROM users WHERE id=?", (user_id,)))
    original = db.row_to_dict(db.q1("SELECT * FROM originals WHERE id=?",
                                    (original_id,)))
    profile["face"] = {"descriptor": [0.0] * 128}
    profile["body"] = {}

    # -------------------------------------------------------------- the stubs
    # Only the things that are not being tested: the file the engine hands back
    # (a real one, written to the target path), the verdict on it, and the two
    # free repair passes.  Everything that DECIDES anything - the retry loop,
    # the diagnosis, the option counters, the billing gate, the attempt rows,
    # the prompt - is the shipped code.
    script: dict = {}

    class Fake:
        name = "fal"

        def capabilities(self):
            return Capabilities(name="fal", generative=True,
                                identity_reference=True)

        def estimate_cost(self, req):
            return 0.04

        def generate(self, req, target):
            script["prompts"].append(req.prompt)
            answer = script["answers"][min(script["n"], len(script["answers"]) - 1)]
            script["n"] += 1
            if answer[0] == "block":
                raise ProviderError("fal no ha entregado la imagen",
                                    code="content_filter", billed=True,
                                    meta={"endpoint": "fal-ai/flux-pro/kontext/multi"},
                                    latency_ms=7800)
            Path(target).write_bytes(photo.read_bytes())
            return GenResult(ok=True, image_path=str(target), provider="fal",
                             model="fal-ai/flux-pro/kontext/multi",
                             cost_usd=0.04, latency_ms=18000)

    fake = Fake()
    orch.router_mod.choose_provider = lambda *a, **k: (fake, "fal-ai/flux-pro/kontext/multi", "ensayo")
    orch.protect_mod.shield_for = lambda *a, **k: {"mask_path": "", "reason": "ensayo"}
    orch._restore_texture = lambda *a, **k: {"aplicada": False, "motivo": "ensayo"}
    orch._correct_free = lambda run_id, img, prof, brf, verdict, defects: (verdict, [], [])
    orch._store_image = lambda *a, **k: "img_ensayo"
    orch.verify_mod.verify_image = lambda *a, **k: _verdict_now(script)

    total_before = billing.balance(user_id, "fal")

    # Each scenario names, IN ADVANCE, the change the seed table supports for
    # it - so the harness is checking a prediction rather than describing
    # whatever happened.  ``clave`` is the learning row that must exist
    # afterwards and ``ok`` is whether that adjustment is recorded as having
    # rescued the image.
    scenarios = [
        {"titulo": "A. NO SE PARECE A TI, y el cambio funciona",
         "pide": {"clothing": "vestido_rojo", "scene": "playa_atardecer"},
         "cobertura": False,
         "respuestas": [("draw", _fail_identity()), ("draw", _ok())],
         "porque": ("el vestido pierde su cara en 6 de 6 imagenes pagadas de la "
                    "semilla y ninguna otra prenda lo hace"),
         "espera": "se ha cambiado la ropa por Vaqueros y camiseta",
         "pagadas": 2, "clave": "adj:identidad:cambiar:clothing:vaqueros_camiseta",
         "ok": 1},
        {"titulo": "B. EL PROVEEDOR NO LA ENTREGA y la cobra",
         "pide": {"clothing": "camisa_blanca"}, "cobertura": True,
         "respuestas": [("block", None), ("draw", _ok())],
         "porque": ("la frase de cobertura viaja en 13 de 13 prompts "
                    "bloqueados y en 0 de los otros 61"),
         "espera": "se ha quitado del texto la frase",
         "pagadas": 2, "clave": "adj:bloqueo:texto_cobertura", "ok": 1,
         "sin_frase": True},
        {"titulo": "C. UNA MANO MAL DIBUJADA, y el cambio no funciona",
         "pide": {"clothing": "gabardina", "pose": "sentada"},
         "cobertura": False,
         "respuestas": [("draw", _fail_hands()), ("draw", _fail_hands())],
         "porque": ("sentada sale mal en 6 de 8 frente al 10% de las demas "
                    "posturas; re-posarla es el cambio mas pequeno de los dos "
                    "que esas mismas 8 imagenes permiten"),
         "espera": "se ha cambiado la postura por Caminando",
         "pagadas": 2, "clave": "adj:anatomia:cambiar:pose:caminando", "ok": 0},
        {"titulo": "D. SALE BIEN A LA PRIMERA",
         "pide": {"clothing": "vaqueros_camiseta", "scene": "ciudad_noche"},
         "cobertura": False, "respuestas": [("draw", _ok())],
         "porque": "no hay nada que arreglar",
         "espera": None, "pagadas": 1, "clave": None},
        {"titulo": "E. NO SE PARECE A TI y no hay nada que el historial respalde",
         "pide": {"framing": "vertical_9_16"}, "cobertura": False,
         "respuestas": [("draw", _fail_identity()), ("draw", _ok())],
         "porque": ("el encuadre no redibuja a nadie y no hay ninguna otra "
                    "opcion en la peticion a la que culpar"),
         "espera": None, "pagadas": 1, "clave": None,
         "parar": "Ningun cambio de opciones"},
        {"titulo": "F. SOLO LA PIEL DEMASIADO LISA",
         "pide": {"clothing": "vaqueros_camiseta"}, "cobertura": False,
         "respuestas": [("draw", _fail_smooth()), ("draw", _ok())],
         "porque": ("la textura ya se devuelve gratis desde su foto y los 8 "
                    "reintentos pagados por este motivo rescataron 2"),
         "espera": None, "pagadas": 1, "clave": None,
         "parar": "ya se corrige gratis"},
        {"titulo": ("G. LA MISMA PETICION QUE C, OTRA VEZ: el cambio que "
                    "tocaria ya se pago"),
         "pide": {"clothing": "gabardina", "pose": "sentada"},
         "cobertura": False,
         "respuestas": [("draw", _fail_hands()), ("draw", _fail_hands())],
         "porque": ("la peticion ajustada - misma foto, gabardina, caminando - "
                    "ya se compro en C y tampoco salio"),
         "espera": None, "pagadas": 1, "clave": None,
         "parar": "ya se pago"},
    ]

    for row in scenarios:
        _one(row, script, user, profile, original, db, billing, orch,
             risk_mod, work)

    # ------------------------------- H. la recomendacion que se aprende a no dar
    print("\nH. UNA RECOMENDACION QUE YA FALLO DOS VECES NO SE VUELVE A DAR")
    print("   " + "-" * 72)
    print("   Sin gastar nada: se le pregunta a adjust.decide con la memoria")
    print("   puesta a mano, que es la unica forma de ver el limite ADJ_MIN_N")
    print("   sin comprar dos imagenes mas.")
    pool_h = risk_mod.pool_for(user["id"], profile["id"])
    antes = adjust_mod.decide("anatomia", "anatomia_manos",
                              {"clothing": "gabardina", "pose": "sentada"},
                              set(), pool_h)
    pool_h["adj:anatomia:cambiar:pose:caminando"] = {
        "n": 2, "ok": 0, "f": {"anatomia": 2}, "m": {"anatomia": 2}, "t": 0.0}
    despues = adjust_mod.decide("anatomia", "anatomia_manos",
                                {"clothing": "gabardina", "pose": "sentada"},
                                set(), pool_h)
    print("   con 1 fallo apuntado : %s" % (antes.get("texto") or antes.get("parar")))
    print("   con 2 fallos apuntados: %s" % (despues.get("texto") or despues.get("parar")))
    check("con un fallo todavia se recomienda el cambio de postura",
          antes.get("ok") and antes.get("por") == "caminando")
    check("con dos fallos ya no se recomienda ese cambio",
          despues.get("por") != "caminando")

    # ------------------------------------- lo que la clienta lee en la ficha
    print("\nLA FICHA, TAL Y COMO LA LEE ELLA")
    print("   " + "-" * 72)
    for row in scenarios:
        ficha = orch.build_report(row["run_id"])
        for made in (ficha.get("ajustes") or []):
            print("   %s %s" % ("[salio bien]" if made["funciono"]
                                else "[aun asi no salio]", made["texto"]))
            print("       Se decidio asi porque %s." % made["medida"])
    hechos = [m for row in scenarios
              for m in (orch.build_report(row["run_id"]).get("ajustes") or [])]
    check("la ficha cuenta los tres cambios que se hicieron",
          len(hechos) == 3, "%d" % len(hechos))
    check("y dice de cada uno si funciono",
          sum(1 for m in hechos if m["funciono"]) == 2)

    # ---------------------------- I. y el proximo presupuesto ya no la ofrece
    print("\nI. LA PROXIMA VEZ, EL PRESUPUESTO YA NO LA OFRECE EN SILENCIO")
    print("   " + "-" * 72)
    print("   La peticion de C (gabardina + sentada) se pago y no salio. Se")
    print("   vuelve a pedir, esta vez dentro de un plan de DOS variantes, que")
    print("   es como llega de verdad desde la pantalla de previsualizacion.")
    plan = {"source_path": original["path"],
            "locked": {},
            "variants": [
                {"index": 0, "choices": {"clothing": "vaqueros_camiseta",
                                         "scene": "ciudad_noche"}},
                {"index": 1, "choices": {"clothing": "gabardina",
                                         "pose": "sentada"}}],
            "envio": {"masked_inpaint": False, "reference_photos": 0,
                      "outfit_coverage_text": False}}
    veredicto = risk_mod.assess(plan, user_id=user["id"],
                                profile_id=profile["id"],
                                endpoint="fal-ai/flux-pro/kontext/multi",
                                quality="preview", style="editorial_moda")
    print("   nivel: %s" % veredicto.get("nivel"))
    for motivo in (veredicto.get("motivos") or [])[:3]:
        print("      >> %s" % motivo["texto"])
    if veredicto.get("ajuste"):
        print("      boton: %s" % veredicto["ajuste"].get("titulo"))
    check("el presupuesto avisa de la combinacion ya pagada y fallida",
          veredicto.get("nivel") in ("aviso", "confirmar"))
    check("y dice que es esa peticion exacta la que ya se pago",
          any(m.get("tipo") == "peticion"
              for m in (veredicto.get("motivos") or [])))

    # ------------------------------------------------------------- the ledger
    print("\n" + "=" * 78)
    print("EL LIBRO DE CUENTAS")
    print("=" * 78)
    rows = db.q("SELECT provider, SUM(amount_usd) AS t, COUNT(*) AS n FROM ledger "
                "WHERE user_id=? AND kind='spend' GROUP BY provider", (user_id,))
    spent = sum(float(r["t"] or 0.0) for r in (rows or []))
    calls = sum(int(r["n"] or 0) for r in (rows or []))
    paid = db.q1("SELECT COUNT(*) AS n, COALESCE(SUM(cost_usd),0) AS t "
                 "FROM attempts WHERE user_id=? AND cost_usd>0", (user_id,))
    print("   llamadas cobradas por el proveedor de ensayo : %d" % int(paid["n"]))
    print("   suma en la tabla attempts                    : %.4f USD" % float(paid["t"]))
    print("   suma en el libro (ledger, gastos)            : %.4f USD" % abs(spent))
    print("   apuntes de gasto en el libro                 : %d" % calls)
    print("   saldo antes / despues                        : %.4f / %.4f USD"
          % (total_before, billing.balance(user_id, "fal")))
    print("   dinero real gastado en fal.ai                : 0.0000 USD"
          " (la red estaba bloqueada y el proveedor es de mentira)")
    check("el libro y las filas de intento cuadran al centimo",
          abs(abs(spent) - float(paid["t"])) < 1e-9)
    check("no queda dinero prometido a ninguna llamada",
          abs(billing.held(user_id)) < 1e-9, "%.4f USD" % billing.held(user_id))

    print("\n" + "=" * 78)
    print("%d comprobaciones correctas, %d fallidas" % (len(PASS), len(FAIL)))
    for bad in FAIL:
        print("   FALLA: %s" % bad)
    print("=" * 78)
    shutil.rmtree(work, ignore_errors=True)
    return 0 if not FAIL else 1


# ------------------------------------------------------------------ scenarios

def _one(row, script, user, profile, original, db, billing, orch, risk_mod,
         work) -> None:
    choices = dict(row["pide"])
    cover = bool(row.get("cobertura"))
    answers = row["respuestas"]
    print("\n%s" % row["titulo"])
    print("   " + "-" * 72)
    print("   pide: %s%s"
          % (", ".join("%s=%s" % kv for kv in sorted(choices.items())),
             "   [+ texto de cobertura]" if cover else ""))
    print("   lo que dice la medida: %s" % row["porque"])
    print("   cambio esperado: %s"
          % (row["espera"] or "ninguno, y ninguna imagen mas"))

    run_id = db.new_id("run")
    now = time.time()
    db.execute("INSERT INTO runs(id,user_id,original_id,profile_id,mode,status,"
               "options_json,plan_json,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
               (run_id, user["id"], original["id"], profile["id"], "preview",
                "running", db.dumps({}), db.dumps({"notes": []}), now))
    script.clear()
    script.update({"answers": answers, "n": 0, "prompts": [], "verdicts": [],
                   "vn": 0})
    script["verdicts"] = [a[1] for a in answers if a[0] == "draw"]

    brief = {"user_id": user["id"], "source_path": original["path"],
             "shot_type": "full_body", "reference_paths": [],
             "preserve": [], "generative": True,
             "envio": {"masked_inpaint": False, "reference_photos": 0,
                       "outfit_coverage_text": cover}}
    variant = {"index": 0, "choices": dict(choices), "params": {}, "seed": 11}
    batch = orch._Batch(run_id, 1, "ensayo", 1)
    out = Path(work) / run_id
    out.mkdir(parents=True, exist_ok=True)

    before = billing.balance(user["id"], "fal")
    row["run_id"] = run_id
    result = orch._run_variant(user, run_id, variant, brief, profile,
                               {"key": "editorial_moda"}, "preview", out,
                               original, batch, "preview")
    after = billing.balance(user["id"], "fal")

    rows = [db.row_to_dict(r) for r in db.q(
        "SELECT * FROM attempts WHERE run_id=? ORDER BY attempt_no", (run_id,))]
    notes = ((db.row_to_dict(db.q1("SELECT plan_json FROM runs WHERE id=?",
                                   (run_id,))) or {}).get("plan")
             or {}).get("notes") or []
    paid = [r for r in rows if float(r["cost_usd"] or 0.0) > 0]
    # ``row_to_dict`` already decodes every *_json column into a real object
    # and drops the suffix, so params_json arrives as ``params``.
    adjusted = [r for r in rows if (r.get("params") or {}).get("ajuste")]

    print("   llamadas pagadas: %d   coste %.4f USD   saldo %.4f -> %.4f"
          % (len(paid), sum(float(r["cost_usd"]) for r in paid), before, after))
    for att in rows:
        print("      intento %d  %-8s  %.2f USD  %s"
              % (att["attempt_no"], att["status"], float(att["cost_usd"]),
                 (att["reject_reason"] or "")[:46]))
    for note in notes:
        if "asi que" in note or "no se paga" in note or "ajust" in note:
            print("      >> %s" % note)

    check("como mucho un intento ajustado (%s)"
          % ("ninguno" if not adjusted else "uno"), len(adjusted) <= 1)
    check("se compran las %d llamadas previstas y ni una mas" % row["pagadas"],
          len(paid) == row["pagadas"], "%d" % len(paid))
    check("lo cobrado y lo apuntado en el libro coinciden",
          abs((before - after) - sum(float(r["cost_usd"]) for r in paid)) < 1e-9,
          "%.4f USD" % (before - after))
    check("el resultado dice lo que costo", abs(float(result.get("cost") or 0.0)
          - sum(float(r["cost_usd"]) for r in paid)) < 1e-9)

    # The change chosen is the one the measurement supports, word for word.
    if row.get("espera"):
        said = " ".join(notes)
        check("el cambio es el que respalda la medida",
              row["espera"] in said, row["espera"])
        check("la segunda llamada lleva la peticion ya cambiada",
              bool(adjusted) and len(paid) == 2)
    else:
        check("no se ajusta nada y no se compra una segunda imagen",
              not adjusted and len(paid) == row["pagadas"])
    if row.get("parar"):
        check("se explica por que no se paga otro intento",
              any(row["parar"] in n for n in notes), row["parar"])
    if row.get("sin_frase"):
        # The clause really left the text that was sent, not merely the flag.
        first, second = script["prompts"][0], script["prompts"][1]
        check("la frase de cobertura estaba en el primer envio",
              "only clothing visible anywhere in the frame" in first.lower())
        check("y NO esta en el segundo",
              "only clothing visible anywhere in the frame" not in second.lower())

    # The learning row: written for her AND for the installation, so the next
    # run and the next person start from it.
    if row.get("clave"):
        mine = risk_mod.person_pool(user["id"], profile["id"])
        shared = risk_mod.shared_pool()
        cell = mine.get(row["clave"]) or {}
        cshared = shared.get(row["clave"]) or {}
        check("se apunta el resultado del ajuste (%s)" % row["clave"],
              int(cell.get("n") or 0) >= 1
              and int(cell.get("ok") or 0) == row["ok"],
              "n=%s ok=%s" % (cell.get("n"), cell.get("ok")))
        check("y tambien en la memoria compartida del robot",
              int(cshared.get("n") or 0) >= 1)


# ------------------------------------------------------------------- verdicts

def _verdict_now(script: dict) -> dict:
    out = script["verdicts"][min(script["vn"], len(script["verdicts"]) - 1)]
    script["vn"] += 1
    return dict(out)


def _check(name, value, threshold, passed, detail=""):
    return {"name": name, "value": value, "threshold": threshold,
            "passed": passed, "detail": detail}


def _base(**over):
    checks = [_check("identity_face", 0.81, 0.45, True),
              _check("body_proportions", 0.03, 0.16, True),
              _check("anatomy", 0.95, 0.60, True),
              _check("skin_tone", 1.2, 12.0, True),
              _check("quality", 0.80, 0.45, True)]
    verdict = {"passed": True, "score": 0.8, "checks": checks, "defects": [],
               "repairable_defects": [], "summary": "ensayo"}
    verdict.update(over)
    return verdict


def _ok():
    return _base()


def _fail_identity():
    v = _base(passed=False, summary="no se parece")
    v["checks"][0] = _check("identity_face", 0.3363, 0.45, False,
                            "el rostro no es el suyo")
    return v


def _fail_hands():
    v = _base(passed=False, summary="una mano mal dibujada")
    v["checks"][2] = _check("anatomy", 0.31, 0.60, False, "mano deformada")
    v["defects"] = [{"type": "hand_malformed", "where": "right_hand",
                     "severity": 0.69}]
    v["repairable_defects"] = list(v["defects"])
    return v


def _fail_smooth():
    v = _base(passed=False, summary="piel demasiado lisa")
    v["checks"][2] = _check("anatomy", 0.40, 0.45, False,
                            "el rostro solo conserva el 71% del grano")
    v["defects"] = [{"type": "oversmoothed_skin", "where": "face",
                     "severity": 0.55}]
    v["repairable_defects"] = []
    return v


def _tiny_jpeg() -> bytes:
    """A real 8x8 JPEG, so every path that opens the file finds an image."""
    import base64
    return base64.b64decode(
        "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRof"
        "Hh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/wAALCAAIAAgBAREA/8QAHwAA"
        "AQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQR"
        "BRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RF"
        "RkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ip"
        "qrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEB"
        "AAA/APn+iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiv//Z")


if __name__ == "__main__":
    raise SystemExit(main())

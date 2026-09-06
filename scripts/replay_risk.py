"""Replay every paid call ever made through the risk assessment.

    python scripts\\replay_risk.py [--db data\\photorobot.sqlite3] [--con-semilla]

SPENDS NOTHING.  It reads the ``attempts`` table of a real database, rebuilds
the request each row was bought with, and asks ``generation/risk.assess`` what
it would have said BEFORE the money moved - then folds the real outcome in and
moves to the next call, exactly as the shipped code does.  The memory is
written to a throwaway database under the system temp folder, so the
installation's own learning row is never touched.

WHY THE REPLAY IS TEMPORAL AND STARTS EMPTY.  Asking the assessment about a
call whose own outcome is already in its counters is not a measurement, it is
a lookup: every failure would be "predicted" and the numbers would be perfect
and worthless.  So by default the pool starts with nothing at all - not even
the seed in risk.PRIOR, which was distilled from these very calls - and each
request is judged only on the calls that were paid for BEFORE it.  That is the
honest question: on the day this request was made, would the robot have known?

``--con-semilla`` runs it the other way, with the shipped seed loaded, which is
what a NEW account really gets.  It is the optimistic bound and it is labelled
as one.

THE DECISION IS TAKEN ONCE PER REQUEST, NOT PER CALL.  A variant that is
retried inside its own run is one decision the client made and up to three
calls the robot bought; the assessment happens before the run, so its answer is
computed at the first attempt and applied to every attempt that request went on
to buy.  The confusion matrix is then reported over both units: per paid call,
because calls are what cost money, and per request, because requests are what
the client is being warned about.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_record(db_path: Path) -> list[dict]:
    """Every paid attempt, with the request it was bought with."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    runs = {r["id"]: r for r in conn.execute("SELECT * FROM runs")}
    out: list[dict] = []
    for row in conn.execute("SELECT * FROM attempts ORDER BY created_at, id"):
        if float(row["cost_usd"] or 0.0) <= 0.0:
            continue
        run = runs.get(row["run_id"])
        if not run:
            continue
        plan = json.loads(run["plan_json"] or "{}")
        opts = json.loads(run["options_json"] or "{}")
        index = row["variant_index"]
        variant = next((v for v in (plan.get("variants") or [])
                        if isinstance(v, dict)
                        and int(v.get("index", -1)) == int(index or 0)), None)
        choices: dict[str, str] = {}
        for group, value in ((variant or {}).get("choices") or {}).items():
            if isinstance(value, str) and value.strip():
                choices[str(group)] = value.strip()
        for group, value in (plan.get("locked") or {}).items():
            if isinstance(value, str) and value.strip():
                choices.setdefault(str(group), value.strip())
        if not choices:
            continue
        prompt = str(row["prompt"] or "").lower()
        out.append({
            "run_id": row["run_id"], "index": int(index or 0),
            "attempt_no": int(row["attempt_no"] or 1),
            "status": str(row["status"] or ""),
            "reason": str(row["reject_reason"] or ""),
            "verdict": json.loads(row["verdict_json"] or "{}"),
            "cost": float(row["cost_usd"] or 0.0),
            "choices": choices,
            "endpoint": str(row["model"] or ""),
            "quality": str(opts.get("quality") or ""),
            "style": str(opts.get("style") or ""),
            "source": str(plan.get("source_path") or run["original_id"] or ""),
            "profile_id": str(run["profile_id"] or ""),
            "user_id": str(row["user_id"] or ""),
            # The one switchable property of a request the record can convict,
            # read off the text that was really sent rather than off a setting
            # that has changed since.  It is the tail of prompt.OUTFIT_REPLACE.
            "cobertura": "only clothing visible anywhere in the frame" in prompt,
        })
    conn.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data" / "photorobot.sqlite3"))
    ap.add_argument("--con-semilla", action="store_true",
                    dest="seeded",
                    help="cargar risk.PRIOR (lo que hereda una cuenta nueva)")
    ap.add_argument("--detalle", action="store_true")
    args = ap.parse_args()

    record = read_record(Path(args.db))
    if not record:
        print("No hay llamadas pagadas en esa base de datos.")
        return 1

    work = Path(tempfile.mkdtemp(prefix="replay_risk_"))
    os.environ["PHOTOROBOT_DATA"] = str(work)
    sys.path.insert(0, str(ROOT / "backend"))
    from app import db                                    # noqa: E402
    from app.generation import risk                       # noqa: E402
    db.init_db()

    if not args.seeded:
        # An empty-but-present row: ``shared_pool`` returns what it finds and
        # only reaches for the seed when it finds nothing, so this is how the
        # replay starts from genuine ignorance without touching the module.
        risk._save(risk.SHARED_USER, risk.SCOPE, {"_grupos": []})

    decisions: dict[tuple, dict] = {}
    results: list[tuple[dict, dict]] = []
    for row in record:
        key = (row["run_id"], row["index"])
        if key not in decisions:
            plan = {
                "source_path": row["source"],
                "profile_id": row["profile_id"],
                "style": row["style"],
                "variants": [{"index": row["index"], "choices": row["choices"]}],
                "locked": {},
                "envio": {"outfit_coverage_text": row["cobertura"],
                          "masked_inpaint": False, "reference_photos": 0},
            }
            decisions[key] = risk.assess(
                plan, user_id=row["user_id"], profile_id=row["profile_id"],
                endpoint=row["endpoint"], quality=row["quality"],
                style=row["style"])
        results.append((row, decisions[key]))
        risk.observe(
            user_id=row["user_id"], profile_id=row["profile_id"],
            choices=row["choices"],
            feats=({"texto_cobertura"} if row["cobertura"] else set()),
            fp=risk.fingerprint(row["source"], row["choices"], row["quality"],
                                row["style"], row["endpoint"]),
            status=row["status"], reason=row["reason"], verdict=row["verdict"])

    report(results, seeded=args.seeded, detail=args.detalle)
    shutil.rmtree(work, ignore_errors=True)
    return 0


def _bucket(status: str) -> str:
    return "aceptada" if status == "accepted" else "fallo"


def report(results, seeded: bool, detail: bool) -> None:
    print()
    print("=" * 72)
    print("REPLAY DEL EVALUADOR DE RIESGO  (%s)"
          % ("con la semilla del robot cargada"
             if seeded else "memoria vacia, aprendiendo sobre la marcha"))
    print("=" * 72)

    per_call = collections.Counter()
    money = collections.Counter()
    for row, verdict in results:
        per_call[(_bucket(row["status"]), verdict["nivel"])] += 1
        money[(_bucket(row["status"]), verdict["nivel"])] += row["cost"]

    seen: set = set()
    per_req = collections.Counter()
    req_state: dict[tuple, str] = {}
    for row, verdict in results:
        key = (row["run_id"], row["index"])
        if row["status"] == "accepted" or key not in req_state:
            req_state[key] = row["status"]
        seen.add((key, verdict["nivel"]))
    for key, level in seen:
        per_req[(_bucket(req_state[key]), level)] += 1

    def table(counter, title, unit):
        levels = ("ninguno", "aviso", "confirmar")
        rows = ("fallo", "aceptada")
        print()
        print(title)
        print("  %-22s %10s %10s %12s %8s" % ("", "sin aviso", "aviso",
                                              "confirmar", "total"))
        for r in rows:
            total = sum(counter[(r, lv)] for lv in levels)
            print("  %-22s %10d %10d %12d %8d"
                  % ("la imagen fallo" if r == "fallo" else "la imagen valio",
                     counter[(r, "ninguno")], counter[(r, "aviso")],
                     counter[(r, "confirmar")], total))
        n_f = sum(counter[("fallo", lv)] for lv in levels)
        n_ok = sum(counter[("aceptada", lv)] for lv in levels)
        caught = n_f - counter[("fallo", "ninguno")]
        wrong = n_ok - counter[("aceptada", "ninguno")]
        print("  avisa de %d de los %d %s que fallaron  (%.0f%%)"
              % (caught, n_f, unit, 100.0 * caught / max(1, n_f)))
        print("  desanima %d de los %d %s que valieron  (%.0f%%)"
              % (wrong, n_ok, unit, 100.0 * wrong / max(1, n_ok)))

    table(per_call, "POR LLAMADA PAGADA (lo que cuesta el dinero)", "cobros")
    table(per_req, "POR PETICION (lo que decide la clienta)", "pedidos")

    spent_bad = sum(v for (b, _lv), v in money.items() if b == "fallo")
    caught_money = sum(v for (b, lv), v in money.items()
                       if b == "fallo" and lv != "ninguno")
    good_money = sum(v for (b, lv), v in money.items()
                     if b == "aceptada" and lv != "ninguno")
    print()
    print("DINERO: de %.4f USD gastados en llamadas que no dieron imagen, el "
          "aviso llega antes de %.4f USD (%.0f%%)."
          % (spent_bad, caught_money, 100.0 * caught_money / max(1e-9, spent_bad)))
    print("        y pone un aviso delante de %.4f USD que si valieron."
          % good_money)

    if detail:
        print()
        print("--- ACEPTADAS QUE HABRIA DESANIMADO ---")
        shown: set = set()
        for row, verdict in results:
            key = (row["run_id"], row["index"])
            if row["status"] != "accepted" or verdict["nivel"] == "ninguno":
                continue
            if key in shown:
                continue
            shown.add(key)
            print(" [%s] %s" % (verdict["nivel"], row["choices"]))
            for motivo in verdict["motivos"][:2]:
                print("       -", motivo["texto"])
        print()
        print("--- FALLOS QUE NO HABRIA VISTO VENIR ---")
        shown = set()
        for row, verdict in results:
            key = (row["run_id"], row["index"])
            if row["status"] == "accepted" or verdict["nivel"] != "ninguno":
                continue
            if key in shown:
                continue
            shown.add(key)
            print(" %-9s %s" % (row["status"], row["choices"]))
        print()
        print("--- QUE DIRIA, PETICION POR PETICION ---")
        shown = set()
        for row, verdict in results:
            key = (row["run_id"], row["index"])
            if key in shown:
                continue
            shown.add(key)
            print(" %-9s [%-9s] %s"
                  % (row["status"], verdict["nivel"],
                     verdict["resumen"][:150] or "(nada que objetar)"))


if __name__ == "__main__":
    raise SystemExit(main())

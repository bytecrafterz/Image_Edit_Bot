"""Generate a demo batch on the LIVE site from her photographs.  Spends money.

    python scripts/demo_batch.py <cookiejar.txt> <out.json> [--shots full,half]
        [--dresses vestido_rojo_cruzado,...] [--scenes restaurante,calle_dia]
        [--engine identity_banana] [--n 1] [--photos IMG_7871,IMG_7880]

Every chosen photograph is edited once per dress, scenes alternating.  The
cookie jar is a Mozilla-format jar of a logged-in session (the account whose
profile holds her photographs).  Results go to <out.json> for demo_bundle.py.
"""
import sys, json, time, argparse, urllib.request, http.cookiejar

B = "https://fotografica.duckdns.org"
DRESSES = ["vestido_rojo_cruzado", "vestido_blanco_escote", "vestido_negro_solapa", "vestido_negro_escote"]
SCENES = ["restaurante", "calle_dia"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jar"); ap.add_argument("out")
    ap.add_argument("--shots", default="full,half")
    ap.add_argument("--dresses", default=",".join(DRESSES))
    ap.add_argument("--scenes", default=",".join(SCENES))
    ap.add_argument("--engine", default="identity_banana")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--photos", default="", help="comma separated filename prefixes; default all")
    ap.add_argument("--base", default=B)
    a = ap.parse_args()
    cj = http.cookiejar.MozillaCookieJar(); cj.load(a.jar, ignore_discard=True, ignore_expires=True)
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

    def call(path, body=None, timeout=300):
        req = urllib.request.Request(a.base + path, data=(json.dumps(body).encode() if body is not None else None),
                                     headers={"Content-Type": "application/json"}, method="POST" if body is not None else "GET")
        with op.open(req, timeout=timeout) as r:
            return json.loads(r.read())

    shots = {s.strip() for s in a.shots.split(",") if s.strip()}
    prefixes = [p.strip() for p in a.photos.split(",") if p.strip()]
    originals = call("/api/originals")["originals"]
    chosen = []
    for o in sorted(originals, key=lambda o: o["filename"]):
        shot = (o.get("analysis") or {}).get("shot_type") or o.get("shot_type") or ""
        if shots and shot not in shots:
            continue
        if prefixes and not any(o["filename"].startswith(p) for p in prefixes):
            continue
        chosen.append(o)
    dresses = [d for d in a.dresses.split(",") if d]
    scenes = [s for s in a.scenes.split(",") if s]
    print("fotos:", len(chosen), "vestidos:", len(dresses), "-> celdas:", len(chosen) * len(dresses), file=sys.stderr)
    runs = {}
    i = 0
    for src in chosen:
        for d in dresses:
            sc = scenes[i % len(scenes)]; i += 1
            body = {"original_id": src["id"], "options": {"clothing": [d], "scene": [sc]},
                    "n_previews": a.n, "quality": "preview", "engine": a.engine}
            try:
                est = call("/api/generate/analyze", body)
                r = call("/api/generate/run", {"run_id": est["run_id"], "confirmar_riesgo": True})
                runs[src["filename"] + "|" + d] = {"run_id": est["run_id"], "dress": d, "scene": sc,
                                                   "est": (est.get("estimate") or {}).get("total_usd")}
                print(f"[{src['filename'][:16]} {d:22} {sc:11}] {est['run_id']} -> {r.get('status')}", file=sys.stderr)
            except Exception as exc:                          # noqa: BLE001
                print(f"[{src['filename'][:16]} {d}] FALLO: {exc}", file=sys.stderr)
            time.sleep(1)
    t0 = time.time(); done = {}
    while len(done) < len(runs) and time.time() - t0 < 3600:
        time.sleep(15)
        for k, v in runs.items():
            if k in done:
                continue
            try:
                st = call(f"/api/generate/status/{v['run_id']}")
            except Exception:                                  # noqa: BLE001
                continue
            if st["status"] in ("done", "failed", "cancelled", "stopped_no_balance"):
                done[k] = st
                print(f"   [{k[:44]}] {st['status']} accepted={st.get('accepted')} spent={st.get('spent_usd')}", file=sys.stderr)
        print(f"   ... {int(time.time() - t0)}s {len(done)}/{len(runs)}", file=sys.stderr)
    json.dump({"runs": runs, "status": done}, open(a.out, "w"))
    print("escrito", a.out, file=sys.stderr)


if __name__ == "__main__":
    main()

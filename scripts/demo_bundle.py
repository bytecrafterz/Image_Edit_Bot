"""Bundle the accepted images of one or more demo batches whose face score
clears a bar, into data/demo/<name>.zip with a Spanish read-me.  Never opens
an image.  Reads the live database directly, so run it as the service user or
with a copy:

    sudo -u photorobot python scripts/demo_bundle.py --min 0.80 --name nayane-mejores \
        --db /opt/photorobot/data/photorobot.sqlite3 out1.json [out2.json ...]
"""
import argparse, datetime, json, os, re, shutil, sqlite3, sys, zipfile

LABEL = {"vestido_rojo_cruzado": "Vestido rojo cruzado con botones dorados",
         "vestido_blanco_escote": "Vestido blanco ajustado con escote profundo",
         "vestido_negro_solapa": "Vestido negro palabra de honor con solapa blanca",
         "vestido_negro_escote": "Vestido negro ajustado con escote profundo",
         "restaurante": "restaurante", "calle_dia": "calle de dia"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--min", type=float, default=0.80)
    ap.add_argument("--name", default="nayane-mejores")
    ap.add_argument("--db", default=os.path.join("data", "photorobot.sqlite3"))
    ap.add_argument("--out", default=os.path.join("data", "demo"))
    a = ap.parse_args()
    run_ids = []
    for f in a.results:
        run_ids += [v["run_id"] for v in json.load(open(f))["runs"].values()]
    con = sqlite3.connect(a.db); con.row_factory = sqlite3.Row
    q = ("SELECT a.image_id, a.verdict_json, i.path, r.options_json, o.filename "
         "FROM attempts a JOIN images i ON i.id=a.image_id JOIN runs r ON r.id=a.run_id "
         "JOIN originals o ON o.id=r.original_id WHERE a.operation='generate' "
         "AND a.status='accepted' AND a.run_id IN (%s)" % ",".join("?" * len(run_ids)))
    picked, scores = [], []
    for r in con.execute(q, run_ids):
        m = re.search(r'"identity_face", "value": ([0-9.]+)', r["verdict_json"] or "")
        if not m:
            continue
        v = float(m.group(1)); scores.append(v)
        ch = json.loads(r["options_json"]).get("choices", {})
        if v >= a.min:
            picked.append({"face": v, "path": r["path"], "src": r["filename"],
                           "dress": (ch.get("clothing") or [""])[0], "scene": (ch.get("scene") or [""])[0]})
    picked.sort(key=lambda p: -p["face"])
    day = datetime.date.today().isoformat()
    folder = os.path.join(a.out, "%s-%s" % (a.name, day))
    shutil.rmtree(folder, ignore_errors=True); os.makedirs(folder)
    lines = ["# %d imagenes con parecido facial >= %.2f (%s)" % (len(picked), a.min, day), "",
             "Se generaron %d imagenes aceptadas; estas %d superan %.2f de parecido facial contra su firma."
             % (len(scores), len(picked), a.min), ""]
    for n, p in enumerate(picked, 1):
        name = "%02d_cara%.2f_%s_%s_de_%s%s" % (n, p["face"], p["dress"] or "sin_vestido", p["scene"],
                                                p["src"].split()[0], os.path.splitext(p["path"])[1])
        shutil.copyfile(p["path"], os.path.join(folder, name))
        lines.append("- %s - %s, %s (desde %s) - cara %.2f" % (
            name, LABEL.get(p["dress"], p["dress"] or "sin cambio de ropa"), LABEL.get(p["scene"], p["scene"]),
            p["src"], p["face"]))
    open(os.path.join(folder, "LEEME.md"), "w").write("\n".join(lines) + "\n")
    zpath = os.path.join(a.out, "%s-%s.zip" % (a.name, day))
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(folder)):
            z.write(os.path.join(folder, f), f)
    print("aceptadas:", len(scores), "| >= %.2f:" % a.min, len(picked), "| zip:", zpath, os.path.getsize(zpath) // 1024, "KB")


if __name__ == "__main__":
    main()

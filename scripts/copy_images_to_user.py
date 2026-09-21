"""Put copies of finished images into another account's album.

    sudo -u photorobot python3 scripts/copy_images_to_user.py \
        --db /opt/photorobot/data/photorobot.sqlite3 --data /opt/photorobot/data \
        --to someone@example.com img_aaa img_bbb ...

Each image file (and its thumbnail) is COPIED under the target account's own
outputs folder and a new album row is written for it, so the copy survives
whatever the source account later does with the original.  The copy costs the
target nothing (cost_usd 0) and says where it came from in its meta.  Run it
again and images already copied are skipped.  --dry-run changes nothing.
"""
import argparse, hashlib, json, os, shutil, sqlite3, sys, time, uuid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--to", required=True, help="email of the target account")
    ap.add_argument("--db", required=True)
    ap.add_argument("--data", required=True, help="the data directory (holds outputs/)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-copy", action="store_true", help="testing only: point at the same files")
    a = ap.parse_args()
    con = sqlite3.connect(a.db, timeout=30); con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    user = con.execute("SELECT id, email FROM users WHERE lower(email)=?", (a.to.strip().lower(),)).fetchone()
    if not user:
        sys.exit("No existe la cuenta %s" % a.to)
    done = 0
    for image_id in a.images:
        src = con.execute("SELECT * FROM images WHERE id=? AND deleted_at IS NULL", (image_id,)).fetchone()
        if not src:
            print("no existe o esta borrada:", image_id); continue
        already = con.execute("SELECT id FROM images WHERE user_id=? AND meta_json LIKE ? AND deleted_at IS NULL",
                              (user["id"], '%"copiada_de": "' + image_id + '"%')).fetchone()
        if already:
            print("ya copiada:", image_id, "->", already["id"]); continue
        new_id = "img_" + uuid.uuid4().hex[:24]
        folder = os.path.join(a.data, "outputs", user["id"], str(src["run_id"]))
        ext = os.path.splitext(src["path"])[1] or ".jpg"
        path, thumb = src["path"], src["thumb_path"]
        if not a.no_copy:
            path = os.path.join(folder, new_id + ext)
            thumb = os.path.join(folder, new_id + "_thumb.jpg") if src["thumb_path"] else None
        meta = json.loads(src["meta_json"] or "{}")
        meta.update({"copiada_de": image_id, "copiada_de_usuario": src["user_id"]})
        print("%s %s (%sx%s) -> %s de %s" % ("COPIARIA" if a.dry_run else "copia", image_id, src["width"], src["height"], new_id, user["email"]))
        if a.dry_run:
            continue
        if not a.no_copy:
            os.makedirs(folder, exist_ok=True)
            shutil.copyfile(src["path"], path)
            if thumb and os.path.isfile(src["thumb_path"]):
                shutil.copyfile(src["thumb_path"], thumb)
            else:
                thumb = None
        con.execute(
            "INSERT INTO images(id,user_id,run_id,attempt_id,original_id,profile_id,kind,path,thumb_path,"
            "width,height,bytes,sha256,provider,model,cost_usd,score,verdict_json,meta_json,is_favorite,created_at) "
            "VALUES(?,?,?,NULL,?,NULL,?,?,?,?,?,?,?,?,?,0,?,?,?,0,?)",
            (new_id, user["id"], src["run_id"], src["original_id"], src["kind"], path, thumb,
             src["width"], src["height"], src["bytes"], src["sha256"], src["provider"], src["model"],
             src["score"], src["verdict_json"], json.dumps(meta, ensure_ascii=False), time.time()))
        con.commit(); done += 1
    print("copiadas:", done)


if __name__ == "__main__":
    main()

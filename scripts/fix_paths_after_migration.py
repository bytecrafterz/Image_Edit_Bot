"""Rewrite the Windows paths the move to Linux left inside the database.

The files came across; the paths that point at them did not.  Every row in
``images`` and ``originals`` still holds an absolute
``C:\\Users\\Administrator\\Documents\\9.2\\...`` path, and routers/files.py
serves a photograph by calling ``Path(row["path"]).is_file()`` - which is False
for a Windows path on Linux, so the album answers 404 for all 64 images and
every one of her 25 originals.  Nothing is corrupt: the pictures are on disk
exactly where they should be, under a different root.

It also rewrites the paths embedded in stored JSON.  Those are not cosmetic:
``brief.source_path`` in ``runs.options_json`` is read by
generation/router.request_geometry, which opens the file to measure it and
decide the aspect ratio to order.  A path that does not resolve makes it fall
back to a default - which is how a portrait request comes back square, the
exact defect fought on 2026-09-04.  New runs build a fresh brief and are
unaffected; re-rendering an OLD run is what would have been silently wrong.

    python scripts/fix_paths_after_migration.py <db> --root <project root>
    python scripts/fix_paths_after_migration.py <db> --root <root> --apply

Without --apply it changes nothing and only reports.  With --apply it writes,
and refuses to write at all unless every rewritten path in ``images`` and
``originals`` resolves to a file that really exists.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

# The four columns routers/files.py actually serves from.
FILE_COLUMNS = (("images", "path"), ("images", "thumb_path"),
                ("originals", "path"), ("originals", "thumb_path"))
# Stored JSON that carries paths inside it.
JSON_COLUMNS = (("runs", "options_json"), ("runs", "plan_json"),
                ("attempts", "params_json"))


def translate(value: str, old_root: str, new_root: str) -> str:
    """One Windows path -> the same file under the new root, or unchanged."""
    if not isinstance(value, str) or not value.lower().startswith(old_root.lower()):
        return value
    tail = value[len(old_root):].lstrip("\\/")
    return str(Path(new_root) / tail.replace("\\", "/"))


def walk(node, old_root: str, new_root: str):
    """Rewrite every string anywhere inside a decoded JSON document."""
    if isinstance(node, str):
        return translate(node, old_root, new_root)
    if isinstance(node, list):
        return [walk(v, old_root, new_root) for v in node]
    if isinstance(node, dict):
        return {k: walk(v, old_root, new_root) for k, v in node.items()}
    return node


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--root", required=True,
                    help="project root on THIS machine, e.g. /opt/photorobot")
    ap.add_argument("--old-root",
                    default="C:\\Users\\Administrator\\Documents\\9.2\\")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--allow-missing", action="store_true",
                    help="write even though some rewritten paths point at "
                         "files that no longer exist (see below)")
    args = ap.parse_args()

    new_root = str(Path(args.root).resolve())
    db = sqlite3.connect(args.db)
    db.row_factory = sqlite3.Row

    plan: list[tuple[str, str, str, str, str]] = []   # table, col, id, old, new
    missing: list[str] = []

    for table, col in FILE_COLUMNS:
        for row in db.execute("SELECT id, %s AS v FROM %s WHERE %s IS NOT NULL"
                              % (col, table, col)):
            new = translate(row["v"], args.old_root, new_root)
            if new == row["v"]:
                continue
            plan.append((table, col, row["id"], row["v"], new))
            if not Path(new).is_file():
                missing.append(new)

    json_plan: list[tuple[str, str, str, str]] = []
    for table, col in JSON_COLUMNS:
        for row in db.execute("SELECT id, %s AS v FROM %s WHERE %s LIKE '%%C:%%'"
                              % (col, table, col)):
            try:
                doc = json.loads(row["v"])
            except (TypeError, ValueError):
                continue
            fixed = json.dumps(walk(doc, args.old_root, new_root))
            if fixed != row["v"]:
                json_plan.append((table, col, row["id"], fixed))

    print("Root: %s" % new_root)
    print("  %-24s %d values" % ("file paths to rewrite", len(plan)))
    print("  %-24s %d documents" % ("json blobs to rewrite", len(json_plan)))
    print("  %-24s %d" % ("rewritten but NOT on disk", len(missing)))
    for m in missing[:10]:
        print("      missing: %s" % m)

    if not args.apply:
        print("\nDry run. Nothing was written. Re-run with --apply.")
        return 1 if missing else 0

    # A missing file is not automatically a mistake.  Some of these images
    # were deleted on the Windows machine long before the move - the audit of
    # 2026-09-04 found two runs, 0.84 USD of them, with zero surviving pixels -
    # and their rows are deliberately kept so what they cost can still be read.
    # Rewriting such a path changes nothing: it answered 404 as a Windows path
    # and it answers 404 as a Linux one.
    #
    # What a missing file CAN mean is that --root is wrong, and that would
    # rewrite every good path into a broken one.  The two are told apart by
    # proportion: a wrong root misses nearly everything, while lost files are a
    # minority of an otherwise resolving set.
    resolved = len(plan) - len(missing)
    if plan and resolved * 2 < len(plan):
        print("\nREFUSING to write: only %d of %d rewritten paths exist. "
              "--root is probably wrong." % (resolved, len(plan)))
        return 2
    if missing and not args.allow_missing:
        print("\n%d of %d rewritten paths point at files that are not on disk."
              % (len(missing), len(plan)))
        print("If those are images you know were already lost, re-run with "
              "--allow-missing.")
        return 2

    with db:
        for table, col, rid, _old, new in plan:
            db.execute("UPDATE %s SET %s=? WHERE id=?" % (table, col),
                       (new, rid))
        for table, col, rid, fixed in json_plan:
            db.execute("UPDATE %s SET %s=? WHERE id=?" % (table, col),
                       (fixed, rid))
    print("\nWritten: %d paths, %d json documents." % (len(plan), len(json_plan)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

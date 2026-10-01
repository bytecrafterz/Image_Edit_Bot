"""Dated copy of the data directory, kept for two weeks, on the same server.

    sudo -u photorobot /opt/photorobot/backend/.venv/bin/python scripts/backup_data.py \
        --data /opt/photorobot/data --dest /var/backups/photorobot --keep 14

Run nightly by deploy/systemd/photorobot-backup.timer.  Each run writes
<dest>/<YYYY-MM-DD_HHMMSS>/, a full browsable copy of data/:

  * the SQLite database is copied with SQLite's own backup API, so the copy is
    consistent while the app keeps writing to it, and is checked with
    PRAGMA quick_check before the run counts as done;
  * every other file that has not changed since the previous copy (same size,
    same modification time) is a hard link to that copy's file, so fourteen
    days of copies cost little more disk than one - her photographs and the
    finished images never change once written;
  * the copy is built under a ".partial-" name and only renamed when complete,
    so a run that dies half way never looks like a good backup.

Restore = stop the service, copy one dated folder back over data/, chown -R
photorobot, start the service.  This guards against a bad delete, a broken
database or a bad deploy; it does NOT survive losing the server itself (the
DigitalOcean droplet vanished with everything on it on 2026-09-30) - that is
what the provider's snapshots are for.

The folder holds her private photographs and the live keys (secret.key,
keystore.json), so it is created mode 0700; --without-keys leaves the two key
files out.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sqlite3
import sys
import time
from pathlib import Path

SNAPSHOT_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{6}$")
KEY_FILES = {"secret.key", "keystore.json"}
SQLITE_SUFFIXES = (".sqlite3", ".sqlite", ".db")
SIDE_FILES = ("-wal", "-shm", "-journal")


def _is_sqlite_side_file(name: str) -> bool:
    return any(name.endswith(s + side) for s in SQLITE_SUFFIXES for side in SIDE_FILES)


def _snapshots(dest: Path) -> list[Path]:
    return sorted(p for p in dest.iterdir() if p.is_dir() and SNAPSHOT_RE.match(p.name))


def _copy_database(src: Path, dst: Path) -> None:
    source = sqlite3.connect(f"file:{src}?mode=ro", uri=True, timeout=60)
    target = sqlite3.connect(dst)
    try:
        source.backup(target)
        # The copy stands alone: no -wal beside it, readable by any sqlite3.
        target.execute("PRAGMA journal_mode=DELETE")
        verdict = target.execute("PRAGMA quick_check").fetchone()[0]
    finally:
        target.close()
        source.close()
    if verdict != "ok":
        raise RuntimeError(f"{src.name}: the copy fails quick_check ({verdict})")
    shutil.copystat(src, dst)


def snapshot(data: Path, dest: Path, keep: int, without_keys: bool = False) -> dict:
    data, dest = data.resolve(), dest.resolve()
    if not data.is_dir():
        raise SystemExit(f"No data directory at {data}")
    if dest == data or data in dest.parents:
        raise SystemExit("--dest must be outside the data directory")
    dest.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(dest, 0o700)
    except OSError:
        pass

    # A run that died earlier leaves a .partial- folder: it is not a backup.
    for stale in dest.glob(".partial-*"):
        shutil.rmtree(stale, ignore_errors=True)

    previous = (_snapshots(dest) or [None])[-1]
    name = time.strftime("%Y-%m-%d_%H%M%S")
    work = dest / f".partial-{name}"
    work.mkdir()
    started = time.time()
    stats = {"linked": 0, "copied": 0, "databases": 0, "bytes_copied": 0, "bytes_total": 0}

    for root, dirs, files in os.walk(data):
        dirs.sort()
        rel_root = Path(root).relative_to(data)
        (work / rel_root).mkdir(parents=True, exist_ok=True)
        for fname in sorted(files):
            if _is_sqlite_side_file(fname):
                continue
            if without_keys and rel_root == Path(".") and fname in KEY_FILES:
                continue
            src = Path(root) / fname
            dst = work / rel_root / fname
            try:
                st = src.stat()
            except FileNotFoundError:
                continue                      # deleted while we walked
            stats["bytes_total"] += st.st_size
            if fname.endswith(SQLITE_SUFFIXES):
                _copy_database(src, dst)
                stats["databases"] += 1
                stats["bytes_copied"] += st.st_size
                continue
            old = previous / rel_root / fname if previous else None
            if old is not None:
                try:
                    ost = old.stat()
                    if ost.st_size == st.st_size and ost.st_mtime_ns == st.st_mtime_ns:
                        os.link(old, dst)
                        stats["linked"] += 1
                        continue
                except OSError:
                    pass                      # missing, or links unsupported: copy
            shutil.copy2(src, dst)
            stats["copied"] += 1
            stats["bytes_copied"] += st.st_size

    if stats["databases"] == 0:
        shutil.rmtree(work, ignore_errors=True)
        raise SystemExit(f"No database found in {data}: refusing to call this a backup")

    final = dest / name
    work.rename(final)
    removed = []
    for old in _snapshots(dest)[:-max(1, keep)]:
        shutil.rmtree(old, ignore_errors=True)
        removed.append(old.name)

    stats.update(snapshot=str(final), seconds=round(time.time() - started, 1),
                 kept=len(_snapshots(dest)), removed=removed)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=os.environ.get("PHOTOROBOT_DATA", ""),
                    help="the data directory (default: $PHOTOROBOT_DATA)")
    ap.add_argument("--dest", required=True, help="folder that holds the dated copies")
    ap.add_argument("--keep", type=int, default=14, help="how many dated copies to keep")
    ap.add_argument("--without-keys", action="store_true",
                    help="leave secret.key and keystore.json out of the copy")
    a = ap.parse_args()
    if not a.data:
        sys.exit("--data is required when PHOTOROBOT_DATA is not set")
    s = snapshot(Path(a.data), Path(a.dest), a.keep, a.without_keys)
    print("backup %s: %d new files (%.1f MB), %d unchanged linked, %d database(s), "
          "data %.1f MB, %ss; %d copies kept%s"
          % (Path(s["snapshot"]).name, s["copied"], s["bytes_copied"] / 1e6, s["linked"],
             s["databases"], s["bytes_total"] / 1e6, s["seconds"], s["kept"],
             ("; removed " + ", ".join(s["removed"])) if s["removed"] else ""))


if __name__ == "__main__":
    main()

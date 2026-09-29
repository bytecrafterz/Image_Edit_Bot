"""What each engine's content reader has already said about her photographs.

GPT Image 2 - the model behind ChatGPT's images - drew her from her clothed
half-length photograph at 0.84-0.85 on her face signature on 2026-09-29, the
closest likeness this product has ever measured, and refused her full-length
photographs and one close-up outright, even with a one-line neutral request.
A request is refused whole when ANY picture in it is refused, so sending seven
of her photographs meant seven chances to lose the call.

So the verdicts are kept.  A photograph that went through is safe to send
again, also as a reference; one that was refused on its own is not sent to that
engine again - the estimate says so and the run goes straight to the engine
that stands in for it (providers/fal._REFUSAL_FALLBACK).  Keyed by the file's
content, so a rename or a move of the data folder does not lose it.
"""
from __future__ import annotations

import hashlib
import os
import threading
from typing import Iterable

from .. import db

PREFIX = "motor_foto:"
# The engines whose reader refuses photographs of her.  The others read what
# they are sent, and remembering for them would only be noise.
REMEMBERED = ("identity_gpt2",)

_digests: dict[tuple, str] = {}
_lock = threading.Lock()


def _digest(path: str) -> str:
    try:
        st = os.stat(path)
    except (OSError, TypeError, ValueError):
        return ""
    key = (str(path), int(st.st_size), int(st.st_mtime))
    with _lock:
        cached = _digests.get(key)
    if cached:
        return cached
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    value = sha.hexdigest()[:20]
    with _lock:
        _digests[key] = value
    return value


def _key(engine: str, path: str) -> str:
    digest = _digest(path)
    return (PREFIX + str(engine) + ":" + digest) if digest else ""


def status(engine: str, path: str) -> bool | None:
    """True accepted, False refused, None never asked (or not remembered)."""
    if engine not in REMEMBERED or not path:
        return None
    key = _key(engine, path)
    if not key:
        return None
    try:
        row = db.q1("SELECT value FROM meta WHERE key=?", (key,))
    except Exception:                                     # noqa: BLE001
        return None
    got = db.loads(row["value"], {}) if row else {}
    if not isinstance(got, dict) or "ok" not in got:
        return None
    return bool(got["ok"])


def record(engine: str, paths: Iterable[str], ok: bool) -> None:
    """Write down the reader's verdict on each of these photographs."""
    if engine not in REMEMBERED:
        return
    for path in paths or []:
        key = _key(engine, str(path or ""))
        if not key:
            continue
        try:
            db.execute("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) "
                       "DO UPDATE SET value=excluded.value",
                       (key, db.dumps({"ok": bool(ok), "at": db.now()})))
        except Exception:                                 # noqa: BLE001
            continue


def accepted(engine: str, paths: Iterable[str]) -> list[str]:
    return [p for p in (paths or []) if status(engine, p) is True]


def effective_engine(engine: str | None, source_path: str) -> tuple[str, str]:
    """The engine to plan with, and the sentence to show when it changed.

    Only a photograph this engine already refused ON ITS OWN moves the choice;
    everything else keeps what she picked (an empty choice stays empty, which
    is the server default).
    """
    from ..providers import fal as fal_mod

    wanted = str(engine or "").strip()
    role = wanted or fal_mod.DEFAULT_IDENTITY_ENGINE
    if status(role, source_path) is False:
        fallback = fal_mod._REFUSAL_FALLBACK.get(role, "")
        if fallback:
            return fallback, (
                "El motor de ChatGPT ya rechazo esta foto otra vez (su filtro de "
                "contenido no la acepta), asi que esta vez la hace Gemini Pro "
                "directamente. Para usar el motor de ChatGPT elige una foto tuya "
                "con ropa de calle, sin escotes ni hombros descubiertos.")
    return wanted, ""


def learn_from(request, meta: dict | None, model: str, ok: bool) -> None:
    """Record what one paid call taught about her photographs.

    ``ok`` with the remembered engine as ``model``: every photograph that went
    out was accepted.  A refusal is only pinned on a photograph when it
    travelled ALONE - with several pictures in the call nobody can say which
    one the reader objected to, and blaming them all would lock her best
    photographs out of the engine that draws her best.
    """
    from ..providers import fal as fal_mod

    meta = meta if isinstance(meta, dict) else {}
    source = str(getattr(request, "source_path", "") or "")
    refs = [str(r) for r in (getattr(request, "reference_paths", None) or [])
            if r and str(r) != source]
    garment = str(((getattr(request, "extra", None) or {}).get("garment_path")) or "")
    for role in REMEMBERED:
        endpoint = str((fal_mod.MODELS.get(role) or {}).get("endpoint") or "")
        if ok and model == endpoint:
            sent = [source] + refs[:int(meta.get("refs_sent") or 0)]
            if meta.get("garment_sent") and garment:
                # The garment picture is remembered too, so that a later
                # refusal of "her photo + this dress" can be pinned on her photo.
                sent.append(garment)
            record(role, [p for p in sent if p], True)
        for refused in meta.get("motor_rechazado") or []:
            if not isinstance(refused, dict) or refused.get("motor") != role:
                continue
            count = int(refused.get("imagenes") or 0)
            alone = count == 1
            with_known_garment = (count == 2 and refused.get("prenda")
                                  and garment and status(role, garment) is True)
            if source and (alone or with_known_garment):
                record(role, [source], False)

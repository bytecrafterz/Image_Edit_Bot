"""Offline checks of the fal engines: payloads, fallback, reference choice.

Costs nothing and sends nothing: the pictures are synthetic, the network calls
are replaced in-process, and no database is opened.  Run it after touching
providers/fal.py, generation/prompt.natural_prompt or identity/gallery's
reference chooser:

    python scripts/engines_offline_test.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASSED if ok else FAILED).append(name)
    print("  [%s] %s%s" % ("OK  " if ok else "FALLO", name,
                           ("  -> " + detail) if detail else ""), flush=True)


def _photo(folder: Path, name: str, size=(1200, 1600)) -> str:
    """A flat picture whose colour depends on its name, so no two are equal."""
    import zlib
    from PIL import Image
    path = folder / name
    seed = zlib.crc32(name.encode("utf-8"))
    colour = (seed & 0xFF, (seed >> 8) & 0xFF, (seed >> 16) & 0xFF)
    Image.new("RGB", size, colour).save(path, "JPEG", quality=80)
    return str(path)


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="photorobot_engines_"))
    os.environ.setdefault("PHOTOROBOT_DATA", str(work))
    from app.providers import fal
    from app.providers.base import GenRequest, ProviderError
    from app.generation import prompt as prompt_mod

    source = _photo(work, "source.jpg")
    refs = [_photo(work, "ref%d.jpg" % i) for i in range(8)] + [source]
    garment = _photo(work, "garment.jpg", (800, 1200))
    natural = prompt_mod.natural_prompt(
        ["red wrap midi dress with gold buttons", "full body framing, head to feet in frame"],
        "", "a city street in front of an elegant restaurant entrance",
        {"outfit", "framing", "scene"}, "the tattoo on the shoulder unchanged and in the same place")
    pictured = prompt_mod.natural_prompt(
        [prompt_mod.NATURAL_PICTURED_GARMENT], "", "a restaurant entrance", {"outfit", "scene"}, "")
    provider = fal.FalProvider()

    def req(engine: str, with_garment: bool = False) -> GenRequest:
        return GenRequest(prompt="LONG KONTEXT CHECKLIST, same bust, waist and hip proportions",
                          negative_prompt="beautified, slimmer body",
                          source_path=source, reference_paths=list(refs),
                          width=768, height=1024, seed=12345,
                          extra={"engine": engine, "prompt_natural": natural,
                                 "garment_path": garment if with_garment else ""})

    print("1. Texto natural (la redaccion que acepto GPT Image 2 el 2026-09-29)")
    check("empieza como una peticion a ChatGPT",
          natural.startswith("Make a realistic photo of this same woman"))
    check("pide su cara, su pelo y su figura exactos",
          "Keep her face, hair and figure exactly" in natural)
    check("sin las palabras que hicieron rechazar la peticion",
          not any(w in natural.lower() for w in ("identity", "de-age", "unmistakably",
                                                  " hips", " waist", "body -")), natural)
    check("lleva los cambios, el lugar y las marcas",
          "red wrap midi dress" in natural and "restaurant entrance" in natural
          and "tattoo" in natural)
    check("no promete el pelo cuando se cambia el peinado",
          "hair" not in prompt_mod.natural_prompt([], "", "", {"hair"}, "").lower())
    check("una prenda con foto se nombra por la foto",
          "outfit shown in the last image" in pictured
          and "Copy only the outfit from the last image" in pictured, pictured)

    print("2. GPT Image 2")
    r = req("identity_gpt2")
    role = provider.pick_model(r)
    payload, meta = provider._payload(role, r)
    size = payload.get("image_size") or {}
    check("se elige GPT Image 2", role == "identity_gpt2", role)
    check("siete fotos suyas (la editada + 6)", len(payload.get("image_urls") or []) == 7,
          str(len(payload.get("image_urls") or [])))
    check("tamano en pixeles, multiplos de 16, forma 3:4",
          size.get("width") == 1536 and size.get("height") == 2048, str(size))
    check("calidad alta y JPEG", payload.get("quality") == "high"
          and payload.get("output_format") == "jpeg")
    check("va el texto natural, no la lista de Kontext",
          payload["prompt"].startswith("Make a realistic photo") and "CHECKLIST" not in payload["prompt"])
    check("la foto editada no viaja dos veces", meta.get("refs_sent") == 6
          and len(set(meta.get("image_digests") or [])) == 7, str(meta.get("image_digests")))
    check("motor por defecto: GPT Image 2",
          provider.pick_model(GenRequest(prompt="x", source_path=source)) == "identity_gpt2")
    check("sin lista negativa ni semilla", "negative_prompt" not in payload and "seed" not in payload)
    check("precio reservado 0.30", abs(provider.estimate_cost(r) - 0.30) < 1e-9,
          str(provider.estimate_cost(r)))
    check("salida anunciada 2048 px", provider.delivered_side(r) == 2048)
    rg = req("identity_gpt2", with_garment=True)
    pg, mg = provider._payload("identity_gpt2", rg)
    check("con la prenda: 1 + 6 + prenda = 8 imagenes",
          len(pg["image_urls"]) == 8 and mg.get("garment_sent") is True, str(len(pg["image_urls"])))
    check("con la prenda, sin el parrafo que hizo rechazarla",
          "different person" not in pg["prompt"] and "last image" in pg["prompt"].lower())
    check("images_sent cuenta lo que viaja", provider.images_sent(r) == 7, str(provider.images_sent(r)))

    print("3. Gemini Pro")
    r = req("identity_banana_pro")
    payload, meta = provider._payload("identity_banana_pro", r)
    check("siete fotos suyas", len(payload["image_urls"]) == 7, str(len(payload["image_urls"])))
    check("2K, forma 3:4, JPEG", payload.get("resolution") == "2K"
          and payload.get("aspect_ratio") == "3:4" and payload.get("output_format") == "jpeg",
          "%s %s" % (payload.get("resolution"), payload.get("aspect_ratio")))
    check("lleva semilla", payload.get("seed") == 12345)
    check("precio 0.15", abs(provider.estimate_cost(r) - 0.15) < 1e-9)

    print("4. Los motores antiguos no cambian")
    r = req("identity_multi")
    payload, _ = provider._payload("identity_multi", r)
    check("Kontext multi: la editada + 3", len(payload["image_urls"]) == 4, str(len(payload["image_urls"])))
    check("Kontext recibe la lista larga", "CHECKLIST" in payload["prompt"])
    r = req("identity_banana", with_garment=True)
    payload, _ = provider._payload("identity_banana", r)
    check("Gemini 2.5 con prenda: 1 + 2 + prenda", len(payload["image_urls"]) == 4,
          str(len(payload["image_urls"])))
    check("fal.engine_references: 6 / 6 / 0",
          (fal.engine_references("identity_gpt2"), fal.engine_references("identity_banana_pro"),
           fal.engine_references("identity_multi")) == (6, 6, 0))

    print("5. Si un motor rechaza la foto, pide al otro en el mismo intento")
    calls: list[str] = []
    fal.get_api_key = lambda name: "offline-test-key"     # never used on the wire

    def fake_submit(client, endpoint, payload):
        calls.append(endpoint)
        if endpoint == fal.MODELS["identity_gpt2"]["endpoint"]:
            err = ProviderError("rechazada", retryable=False, code="http_422")
            err.refused = True
            raise err
        return "req-2", "status-url", "response-url"

    provider._submit = fake_submit
    provider._wait = lambda client, s, r, d: {"images": [{"url": "data:image/jpeg;base64,/9j/", "width": 1792, "height": 2400}]}
    provider._download = lambda client, url, out: Path(out).write_bytes(b"\xff\xd8\xff") or 3
    result = provider.generate(req("identity_gpt2"), work / "out.jpg")
    check("el segundo motor dibuja la imagen",
          calls == [fal.MODELS["identity_gpt2"]["endpoint"], fal.MODELS["identity_banana_pro"]["endpoint"]],
          " -> ".join(calls))
    check("la fila dice quien la rechazo",
          (result.meta.get("motor_rechazado") or [{}])[0].get("motor") == "identity_gpt2")
    check("se cobra el precio del que la hizo", abs(result.cost_usd - 0.15) < 1e-9, str(result.cost_usd))
    calls.clear()

    def refuse_all(client, endpoint, payload):
        calls.append(endpoint)
        err = ProviderError("rechazada", retryable=False, code="http_422")
        err.refused = True
        raise err

    provider._submit = refuse_all
    try:
        provider.generate(req("identity_gpt2"), work / "out2.jpg")
        check("si los dos rechazan, falla sin tercer intento", False)
    except ProviderError:
        check("si los dos rechazan, falla sin tercer intento", len(calls) == 2, str(len(calls)))

    def malformed(client, endpoint, payload):
        calls.append(endpoint)
        raise ProviderError("mal formada", retryable=False, code="http_422")

    calls.clear()
    provider._submit = malformed
    try:
        provider.generate(req("identity_gpt2"), work / "out3.jpg")
    except ProviderError:
        pass
    check("un error que no es un rechazo no cambia de motor", len(calls) == 1, str(len(calls)))

    print("6. Eleccion de siete referencias")
    import random
    from app.identity import gallery
    rnd = random.Random(7)
    base = [rnd.gauss(0, 1) for _ in range(128)]
    cands = []
    for i in range(20):
        emb = [b + rnd.gauss(0, 0.6) for b in base]
        cands.append({"path": "p%d.jpg" % i, "shot_type": "full" if i % 5 == 0 else "closeup",
                      "loo": 0.8, "face": {"embedding": emb, "face_px": 130 + 20 * i}})
    vectors = [c["face"]["embedding"] for c in cands]
    row = gallery._greedy_references(cands, vectors, 7, 3, "full", "p3.jpg")
    paths = [p["path"] for p in (row or {}).get("combo", [])]
    check("siete distintas", len(paths) == 7 and len(set(paths)) == 7, str(paths))
    check("incluye la foto que se edita", "p3.jpg" in paths)
    check("incluye una de cuerpo entero", any(p["shot_type"] == "full" and p["path"] != "p3.jpg"
                                              for p in row["combo"]))

    print("7. Memoria de lo que acepta cada motor")
    from app import db
    from app.generation import engine_memory as mem
    db.init_db()
    photo_ok, photo_bad = _photo(work, "acepta.jpg"), _photo(work, "rechaza.jpg")
    check("una foto nunca enviada: sin veredicto", mem.status("identity_gpt2", photo_ok) is None)
    mem.record("identity_gpt2", [photo_ok], True)
    check("aceptada queda aceptada", mem.status("identity_gpt2", photo_ok) is True
          and mem.accepted("identity_gpt2", [photo_ok, photo_bad]) == [photo_ok])
    solo = GenRequest(prompt="x", source_path=photo_bad, extra={})
    mem.learn_from(solo, {"motor_rechazado": [{"motor": "identity_gpt2", "imagenes": 1}]}, "", False)
    check("rechazada sola: queda rechazada", mem.status("identity_gpt2", photo_bad) is False)
    engine, note = mem.effective_engine("", photo_bad)
    check("una foto rechazada va directa a Gemini Pro y se dice",
          engine == "identity_banana_pro" and "Gemini Pro" in note, engine)
    check("una foto sin veredicto conserva la eleccion", mem.effective_engine("", photo_ok) == ("", ""))
    with_dress = _photo(work, "otra.jpg")
    mem.record("identity_gpt2", [garment], True)
    both = GenRequest(prompt="x", source_path=with_dress, extra={"garment_path": garment})
    mem.learn_from(both, {"motor_rechazado": [{"motor": "identity_gpt2", "imagenes": 2,
                                               "prenda": True}]}, "", False)
    check("rechazada con una prenda ya aceptada: la culpa es de la foto",
          mem.status("identity_gpt2", with_dress) is False)
    many = _photo(work, "varias.jpg")
    mem.learn_from(GenRequest(prompt="x", source_path=many, extra={}),
                   {"motor_rechazado": [{"motor": "identity_gpt2", "imagenes": 4}]}, "", False)
    check("rechazada con varias fotos: no se culpa a ninguna", mem.status("identity_gpt2", many) is None)
    check("los motores que no rechazan no se recuerdan",
          mem.status("identity_banana_pro", photo_bad) is None)

    print("\nRESULTADO: %d correctas, %d fallidas" % (len(PASSED), len(FAILED)))
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())

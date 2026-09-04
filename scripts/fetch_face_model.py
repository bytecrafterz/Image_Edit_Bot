"""Download the two face models the identity check needs.

They live under ``backend/app/models`` and are NOT in the repository: the
recogniser alone is 37 MB, which is bigger than everything else here put
together, so .gitignore excludes ``backend/app/models/*.onnx`` and this script
puts them back on a fresh clone.

Both come from the OpenCV Zoo (github.com/opencv/opencv_zoo), Apache-2.0, and
run entirely on this machine through OpenCV's own ``cv2.FaceRecognizerSF`` and
``cv2.FaceDetectorYN``.  Downloading them is not a paid API call and sends no
image anywhere; nothing here talks to fal.ai or to Anthropic.

Every file is checked against the sha256 of the exact revision that was
measured and calibrated in identity/embedding.py.  A file that does not match
is deleted rather than kept, because a silently different revision would move
the numbers the threshold was fitted to.

    python scripts/fetch_face_model.py [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "backend", "app", "models")
BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"

MODELS = (
    {
        "name": "face_recognition_sface_2021dec.onnx",
        "url": BASE + "/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "bytes": 38696353,
        "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        "what": "SFace, la firma facial de 128 numeros que reconoce a la persona",
    },
    {
        "name": "face_detection_yunet_2023mar.onnx",
        "url": BASE + "/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "bytes": 232589,
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        "what": "YuNet, que situa los cinco puntos con los que se alinea la cara",
    },
)

CHUNK = 1 << 20


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch(model: dict, force: bool) -> bool:
    target = os.path.join(MODEL_DIR, model["name"])
    if os.path.exists(target) and not force:
        if _sha256(target) == model["sha256"]:
            print("ya esta: %s" % model["name"])
            return True
        print("el archivo %s no coincide con su sha256, se descarga de nuevo"
              % model["name"])

    print("descargando %s (%.1f MB) - %s"
          % (model["name"], model["bytes"] / 1e6, model["what"]))
    os.makedirs(MODEL_DIR, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=MODEL_DIR, suffix=".part")
    os.close(handle)
    try:
        with urllib.request.urlopen(model["url"], timeout=180) as response, \
                open(tmp, "wb") as out:
            while True:
                block = response.read(CHUNK)
                if not block:
                    break
                out.write(block)
        size = os.path.getsize(tmp)
        digest = _sha256(tmp)
        if digest != model["sha256"]:
            print("  ERROR: sha256 %s, se esperaba %s (%d bytes)"
                  % (digest, model["sha256"], size))
            return False
        os.replace(tmp, target)
        print("  ok, %d bytes, sha256 correcto" % size)
        return True
    except Exception as exc:
        print("  ERROR: %s" % exc)
        return False
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="vuelve a descargar aunque el archivo ya exista")
    args = parser.parse_args()
    ok = all(_fetch(model, args.force) for model in MODELS)
    if ok:
        print("\nListo. La comprobacion de identidad ya puede reconocer caras.")
    else:
        print("\nFaltan modelos: la comprobacion de identidad dira que no puede "
              "juzgar el rostro en lugar de aprobarlo en silencio.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

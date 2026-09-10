"""Download the models the identity and body checks need.

The two face models live under ``backend/app/models`` and are NOT in the
repository: the recogniser alone is 37 MB, which is bigger than everything else
here put together, so .gitignore excludes ``backend/app/models/*.onnx`` and
this script puts them back on a fresh clone.

The third is the body model.  MediaPipe ships only its "full" pose model inside
the wheel and fetches the "heavy" one - the ruler every body threshold was
calibrated with - from Google storage on first use, writing it INTO
site-packages.  On the Linux service that directory is read-only
(ProtectSystem=strict), so the first pose ever asked for on the deployed box
tried that download during a paid image on 2026-09-10, failed on the write, and
every body measurement after it was silently omitted.  Fetching it here, into
the venv this script runs from, is what makes the deployed measurement the
same as the one the thresholds were fitted to.  Run it with the DEPLOYED venv's
python, as the service user, or copy the file there by hand.

The face models come from the OpenCV Zoo (github.com/opencv/opencv_zoo),
Apache-2.0, and run entirely on this machine through OpenCV's own
``cv2.FaceRecognizerSF`` and ``cv2.FaceDetectorYN``; the pose model comes from
the same storage MediaPipe itself uses.  Downloading them is not a paid API
call and sends no image anywhere; nothing here talks to fal.ai or to Anthropic.

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
    {
        "name": "pose_landmark_heavy.tflite",
        "url": "https://storage.googleapis.com/mediapipe-assets/pose_landmark_heavy.tflite",
        "bytes": 27709200,
        "sha256": "59e42d71bcd44cbdbabc419f0ff76686595fd265419566bd4009ef703ea8e1fe",
        "what": "MediaPipe Pose 'heavy', los 33 puntos del cuerpo con los que se miden tus proporciones",
        # Not under MODEL_DIR: MediaPipe looks for it in ONE place, inside its
        # own package, and there is no setting to point it elsewhere.
        "dir": "mediapipe",
    },
)

CHUNK = 1 << 20


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _model_dir(model: dict) -> str:
    """Where this model has to live - our folder, or MediaPipe's own."""
    if model.get("dir") != "mediapipe":
        return MODEL_DIR
    try:
        import mediapipe
    except ImportError:
        print("  mediapipe no esta instalado en este venv: %s no se descarga"
              % model["name"])
        return ""
    # Same rule MediaPipe uses in python/solutions/download_utils.py: the
    # package root, then modules/pose_landmark.
    root = os.path.dirname(os.path.abspath(mediapipe.__file__))
    return os.path.join(root, "modules", "pose_landmark")


def _fetch(model: dict, force: bool) -> bool:
    model_dir = _model_dir(model)
    if not model_dir:
        return False
    target = os.path.join(model_dir, model["name"])
    if os.path.exists(target) and not force:
        if _sha256(target) == model["sha256"]:
            print("ya esta: %s" % model["name"])
            return True
        print("el archivo %s no coincide con su sha256, se descarga de nuevo"
              % model["name"])

    print("descargando %s (%.1f MB) - %s"
          % (model["name"], model["bytes"] / 1e6, model["what"]))
    os.makedirs(model_dir, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=model_dir, suffix=".part")
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
        print("\nListo. La comprobacion de identidad ya puede reconocer caras y "
              "medir el cuerpo con el modelo completo.")
    else:
        print("\nFaltan modelos: la comprobacion de identidad dira que no puede "
              "juzgar el rostro en lugar de aprobarlo en silencio, y el cuerpo "
              "se medira con el modelo reducido (analysis/pose.py lo avisa).")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

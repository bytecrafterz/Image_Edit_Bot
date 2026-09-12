"""Register a garment picture as a clothing value of an account, exactly as
the "Foto de una prenda" button does, from a file on disk.

    PHOTOROBOT_DATA=/opt/photorobot/data python scripts/register_garment.py <user_id> <image> ["texto"]

Needs the Anthropic key in the keystore (Claude describes the garment).  Run
as the service user so the copy lands where the service can read it.
"""
import shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(2)
    user_id, image = sys.argv[1], Path(sys.argv[2])
    texto = sys.argv[3] if len(sys.argv) > 3 else ""
    from app import db
    from app.config import DATA_DIR
    from app.catalog import options as options_mod
    from app.providers import registry
    vision = registry.get_vision_provider("claude")
    folder = DATA_DIR / "referencias" / user_id
    folder.mkdir(parents=True, exist_ok=True)
    ref_id = db.new_id("ref")
    path = folder / f"{ref_id}.jpg"
    shutil.copyfile(image, path)
    desc = vision.describe_garment(str(path), texto)
    if not desc.get("ok"):
        print("no se pudo leer la prenda:", desc.get("error")); sys.exit(1)
    row = options_mod.add_user_value(
        user_id, desc["grupo"], desc["label_es"], desc["prompt"], desc.get("negative", ""),
        params={"garment_image": str(path)} if desc["grupo"] == "clothing" else {})
    print("valor creado:", row["value_key"], "|", desc["label_es"], "| coste", desc.get("cost_usd"))


if __name__ == "__main__":
    main()

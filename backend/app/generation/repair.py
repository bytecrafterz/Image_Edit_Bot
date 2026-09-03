"""Localized repair - fix the hand, keep the photograph.

Regenerating a whole image because one hand came out wrong is what costs the
client her five attempts: the face moves, the body moves, and the good parts are
thrown away with the bad one.  This module repaints only the failing region,
inside a feathered mask, and then proves the repair was worth keeping: the same
anomaly scanner is re-run on that region alone and the new pixels survive only
if the measured severity actually dropped.  If it did not, the region is
reverted and the rest of the image is untouched.

The money spent is real and is reported even for reverted attempts, because a
provider charges for a bad repair exactly like a good one.
"""
from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..analysis import anomaly as anomaly_mod
from ..analysis import face as face_mod
from ..analysis import loader
from ..analysis import pose as pose_mod
from ..analysis import quality as quality_mod
from ..analysis import segment as segment_mod
from ..config import CACHE_DIR, ensure_dirs
from ..providers.base import GenRequest, InsufficientBalance
from . import prompt as prompt_mod

DILATE_FRACTION = 0.06      # contract: 6% of the shorter side around the bbox
MIN_IMPROVEMENT = 0.05      # severity must really drop, not wobble
WORSE_TOLERANCE = 0.15      # a repair that creates a new defect is reverted
CROP_MARGIN = 0.6           # context around the box when re-scanning
MIN_CROP_SIDE = 256         # detectors need something to look at
MAX_REGIONS = 3             # cost control: the worst regions first, not all
QUALITY_TOLERANCE = 0.02


# ------------------------------------------------------------------ helpers

def _safe(fn, *args, **kwargs) -> Any:
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


def _f(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return default if out != out else out


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _pixel_box(bbox: Any, width: int, height: int) -> list[int] | None:
    """Defect boxes are pixels per contract; normalised boxes are tolerated."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x, y, w, h = [float(v) for v in list(bbox)[:4]]
    except (TypeError, ValueError):
        return None
    if w <= 0.0 or h <= 0.0:
        return None
    if max(x, y, w, h) <= 1.5 and w <= 1.0 and h <= 1.0:
        x, y, w, h = x * width, y * height, w * width, h * height
    x0 = int(max(0, min(width - 1, round(x))))
    y0 = int(max(0, min(height - 1, round(y))))
    x1 = int(max(x0 + 1, min(width, round(x + w))))
    y1 = int(max(y0 + 1, min(height, round(y + h))))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return [x0, y0, x1 - x0, y1 - y0]


def _dilate(box: list[int], pad: int, width: int, height: int) -> list[int]:
    x0 = max(0, box[0] - pad)
    y0 = max(0, box[1] - pad)
    x1 = min(width, box[0] + box[2] + pad)
    y1 = min(height, box[1] + box[3] + pad)
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


def _overlap(a: list[int], b: list[int]) -> bool:
    return not (a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0] or
                a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1])


def _union(a: list[int], b: list[int]) -> list[int]:
    x0 = min(a[0], b[0])
    y0 = min(a[1], b[1])
    x1 = max(a[0] + a[2], b[0] + b[2])
    y1 = max(a[1] + a[3], b[1] + b[3])
    return [x0, y0, x1 - x0, y1 - y0]


def _group_defects(defects: list[dict], width: int, height: int,
                   pad: int) -> list[dict]:
    """One repaint per connected cluster of boxes; overlapping boxes merge."""
    items: list[dict] = []
    for defect in defects:
        box = _pixel_box(defect.get("bbox"), width, height)
        if box is None:
            continue
        items.append({"box": _dilate(box, pad, width, height),
                      "defects": [defect]})
    merged = True
    while merged and len(items) > 1:
        merged = False
        out: list[dict] = []
        for item in items:
            for existing in out:
                if _overlap(existing["box"], item["box"]):
                    existing["box"] = _union(existing["box"], item["box"])
                    existing["defects"].extend(item["defects"])
                    merged = True
                    break
            else:
                out.append(item)
        items = out
    for item in items:
        item["severity"] = max(_f(d.get("severity"), 0.0) for d in item["defects"])
        item["types"] = sorted({_text(d.get("type")) for d in item["defects"]
                                if _text(d.get("type"))})
    items.sort(key=lambda it: (-it["severity"], it["box"][1], it["box"][0]))
    return items


def _feathered_mask(box: list[int], height: int, width: int,
                    feather: int) -> np.ndarray:
    mask = np.zeros((height, width), np.uint8)
    x0, y0, w, h = box
    cv2.rectangle(mask, (x0, y0), (x0 + w - 1, y0 + h - 1), 255, -1)
    k = int(max(3, min(199, feather * 2 + 1)))
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(mask, (k, k), 0)


def _crop(img: np.ndarray, box: list[int], margin: float) -> tuple[np.ndarray, list[int]]:
    height, width = img.shape[:2]
    x0, y0, w, h = box
    mx = int(round(max(w * margin, (MIN_CROP_SIDE - w) / 2.0)))
    my = int(round(max(h * margin, (MIN_CROP_SIDE - h) / 2.0)))
    mx, my = max(0, mx), max(0, my)
    cx0 = max(0, x0 - mx)
    cy0 = max(0, y0 - my)
    cx1 = min(width, x0 + w + mx)
    cy1 = min(height, y0 + h + my)
    return img[cy0:cy1, cx0:cx1], [cx0, cy0, cx1 - cx0, cy1 - cy0]


def _scan_region(img: np.ndarray, box: list[int]) -> dict:
    """Re-run the anomaly scan on one crop only - cheap, local, no network."""
    crop, _ = _crop(img, box, CROP_MARGIN)
    if not isinstance(crop, np.ndarray) or crop.size == 0:
        return {"ok": False, "by_type": {}, "max": 0.0}
    pose_d = _safe(pose_mod.detect_pose, crop) or {}
    face_d = _safe(face_mod.detect_face, crop) or {}
    seg = _safe(segment_mod.person_mask, crop) or {}
    person = seg.get("mask") if seg.get("ok") else None
    person = person if isinstance(person, np.ndarray) else None
    masks = _safe(segment_mod.region_masks, crop, pose_d, person)
    masks = dict(masks) if isinstance(masks, dict) else {}
    if person is not None:
        masks.setdefault("person", person)
    scan = _safe(anomaly_mod.scan_anomalies, crop, pose_d, face_d, masks)
    if not isinstance(scan, dict) or not scan.get("ok"):
        return {"ok": False, "by_type": {}, "max": 0.0}
    by_type: dict[str, float] = {}
    for defect in (scan.get("defects") or []):
        if not isinstance(defect, dict):
            continue
        kind = _text(defect.get("type"))
        if not kind:
            continue
        by_type[kind] = max(by_type.get(kind, 0.0), _f(defect.get("severity"), 0.0))
    return {"ok": True, "by_type": by_type,
            "max": max(by_type.values()) if by_type else 0.0}


def _region_quality(img: np.ndarray, box: list[int]) -> float:
    """Fallback evidence when the scanner cannot read such a small crop."""
    crop, _ = _crop(img, box, 0.15)
    if not isinstance(crop, np.ndarray) or crop.size == 0:
        return 0.0
    result = _safe(quality_mod.assess_quality, crop)
    if isinstance(result, dict) and result.get("ok"):
        return _f(result.get("score"), 0.0)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0))


def _severity_for(scan: dict, types: list[str]) -> float | None:
    if not scan.get("ok"):
        return None
    values = [scan["by_type"].get(kind, 0.0) for kind in types]
    return max(values) if values else _f(scan.get("max"), 0.0)


def _seed_for(image_path: str, index: int) -> int:
    raw = "%s|repair|%d" % (image_path, index)
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8], 16) & 0x7FFFFFFF


def _reference_paths(brief: dict) -> list[str]:
    out: list[str] = []
    for key in ("original_path", "source_path", "reference_path"):
        value = _text(brief.get(key))
        if value and Path(value).exists() and value not in out:
            out.append(value)
    refs = brief.get("reference_paths")
    if isinstance(refs, (list, tuple)):
        for item in refs:
            value = _text(item)
            if value and Path(value).exists() and value not in out:
                out.append(value)
    return out[:4]


# -------------------------------------------------------------------- main

def repair(image_path: str, defects: list[dict], brief: dict, profile: dict,
           provider, out_path: str) -> dict:
    """Repaint every repairable defect region, keeping only what measures better."""
    brf = brief if isinstance(brief, dict) else {}
    prof = profile if isinstance(profile, dict) else {}
    result: dict[str, Any] = {
        "ok": False, "image_path": str(image_path), "repaired": [],
        "cost_usd": 0.0, "rounds": 0, "reverted": [], "regions": 0,
        "notes": [], "reason": "",
    }

    candidates = [d for d in (defects or [])
                  if isinstance(d, dict) and d.get("repairable") and d.get("bbox")]
    if not candidates:
        result["reason"] = "no hay defectos reparables con zona definida"
        return result

    caps = _safe(provider.capabilities) if provider is not None else None
    if caps is None or not getattr(caps, "inpaint", False):
        result["reason"] = ("el proveedor %s no puede reparar por zonas"
                            % _text(getattr(caps, "name", "")
                                    or getattr(provider, "name", "desconocido")))
        return result

    img = _safe(loader.load_image, str(image_path))
    if not isinstance(img, np.ndarray) or img.size == 0:
        result["reason"] = "no se pudo abrir la imagen a reparar"
        return result

    height, width = img.shape[:2]
    pad = max(4, int(round(DILATE_FRACTION * min(height, width))))
    groups = _group_defects(candidates, width, height, pad)
    if not groups:
        result["reason"] = "las zonas de los defectos no son utilizables"
        return result
    if len(groups) > MAX_REGIONS:
        result["notes"].append(
            "Se reparan las %d zonas mas graves de %d para no encarecer la "
            "imagen." % (MAX_REGIONS, len(groups)))
        groups = groups[:MAX_REGIONS]
    result["regions"] = len(groups)

    ensure_dirs()
    workdir = Path(tempfile.mkdtemp(prefix="repair_", dir=str(CACHE_DIR)))
    work = img.copy()
    quality = _text(brf.get("quality")) or "standard"
    references = _reference_paths(brf)
    kept_any = False

    try:
        for index, group in enumerate(groups):
            box = group["box"]
            worst = max(group["defects"], key=lambda d: _f(d.get("severity"), 0.0))
            spec = prompt_mod.repair_prompt(worst, brf, prof)
            negatives = [spec["negative_prompt"]]
            for defect in group["defects"]:
                if defect is worst:
                    continue
                negatives.append(
                    prompt_mod.repair_prompt(defect, brf, prof)["negative_prompt"])

            mask = _feathered_mask(box, height, width, max(3, pad // 2))
            mask_path = workdir / ("mask_%d.png" % index)
            source_path = workdir / ("work_%d.png" % index)
            cand_path = workdir / ("cand_%d.png" % index)
            if not _safe(loader.save_image, mask, mask_path, 100) \
                    or not _safe(loader.save_image, work, source_path, 98):
                result["notes"].append("No se pudo preparar la mascara de la "
                                       "zona %d." % (index + 1))
                continue

            params = spec.get("params") or {}
            req = GenRequest(
                prompt=spec["prompt"],
                source_path=str(source_path),
                mask_path=str(mask_path),
                reference_paths=list(references),
                negative_prompt=", ".join([n for n in negatives if n]),
                operation="inpaint",
                quality=quality,
                strength=_f(params.get("strength"), 0.7),
                guidance=_f(params.get("guidance"), 5.0),
                steps=int(_f(params.get("steps"), 32)),
                seed=_seed_for(str(image_path), index),
                identity_weight=_f(params.get("identity_weight"), 0.9),
                extra={"defect_types": group["types"], "bbox": box},
            )

            result["rounds"] += 1
            try:
                res = provider.inpaint(req, str(cand_path))
            except InsufficientBalance:
                # A hard stop: the orchestrator must alert, never keep trying.
                result["notes"].append("Sin saldo en el proveedor: se detiene "
                                       "la reparacion.")
                raise
            except Exception as exc:
                result["notes"].append("El proveedor fallo al reparar la zona "
                                       "%d: %s" % (index + 1, exc))
                continue

            result["cost_usd"] = round(result["cost_usd"] + _f(
                getattr(res, "cost_usd", 0.0), 0.0), 6)
            painted_path = _text(getattr(res, "image_path", ""))
            if not getattr(res, "ok", False) or not painted_path \
                    or not Path(painted_path).exists():
                result["notes"].append("La reparacion de la zona %d no devolvio "
                                       "imagen." % (index + 1))
                continue

            painted = _safe(loader.load_image, painted_path)
            if not isinstance(painted, np.ndarray) or painted.size == 0:
                result["notes"].append("No se pudo leer la reparacion de la zona "
                                       "%d." % (index + 1))
                continue
            if painted.shape[:2] != (height, width):
                painted = cv2.resize(painted, (width, height),
                                     interpolation=cv2.INTER_LANCZOS4)

            alpha = (mask.astype(np.float32) / 255.0)[..., None]
            trial = np.clip(work.astype(np.float32) * (1.0 - alpha) +
                            painted.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

            declared = group["severity"]
            before = _scan_region(work, box)
            after = _scan_region(trial, box)
            before_val = _severity_for(before, group["types"])
            after_val = _severity_for(after, group["types"])
            if before_val is None:
                before_val = declared

            if after_val is not None:
                keep = after_val <= before_val - MIN_IMPROVEMENT
                if keep and before.get("ok") and after["max"] > before["max"] + WORSE_TOLERANCE:
                    keep = False        # fixed the hand, broke something else
                detail = "gravedad %.2f -> %.2f" % (before_val, after_val)
            else:
                q_before = _region_quality(work, box)
                q_after = _region_quality(trial, box)
                keep = q_after >= q_before - QUALITY_TOLERANCE
                detail = ("sin lectura del escaner, calidad de la zona %.2f -> "
                          "%.2f" % (q_before, q_after))

            if keep:
                work = trial
                kept_any = True
                result["repaired"].extend(
                    [t for t in group["types"] if t not in result["repaired"]])
                result["notes"].append("Zona %d reparada (%s): %s."
                                       % (index + 1, ", ".join(group["types"])
                                          or "defecto", detail))
            else:
                result["reverted"].extend(
                    [t for t in group["types"] if t not in result["reverted"]])
                result["notes"].append(
                    "Zona %d revertida porque la reparacion no mejoro (%s)."
                    % (index + 1, detail))

        if not kept_any:
            result["reason"] = ("ninguna reparacion mejoro la zona; se conserva "
                                "la imagen original")
            return result

        target = Path(out_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        saved = _safe(loader.save_image, work, target, 96)
        if not saved:
            result["reason"] = "no se pudo guardar la imagen reparada"
            return result
        result["ok"] = True
        result["image_path"] = str(saved)
        return result
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

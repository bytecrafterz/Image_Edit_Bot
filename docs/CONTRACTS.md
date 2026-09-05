# Photo Robot — internal contracts

Every module in `backend/app` builds against the interfaces on this page.
If a signature here disagrees with code, **this page wins** — fix the code.

Import style everywhere: relative, e.g. `from ..config import SETTINGS`,
`from .. import db`. Python 3.12, `from __future__ import annotations` at the
top of every module.

Comments in code: Spanish only for user-facing strings, English for code
comments. No accented characters in Python source string literals that are
written to the console (Windows cp1252 console); the API returns UTF-8 JSON so
accents are fine inside JSON responses — but keep source ASCII-safe to avoid
Windows console encoding failures.

---

## 0. Vocabulary

| term | meaning |
|---|---|
| **original** | a real photograph the user uploaded |
| **profile** | the measured identity of one person (face + body + skin + marks) |
| **run** | one press of "Analizar" — produces N previews, or one final render |
| **variant** | one planned image inside a run (a combination of chosen options) |
| **attempt** | one actual call to a provider for a variant (a variant may need 2) |
| **image** | a stored, accepted output file |
| **shot type** | `closeup` \| `half` \| `full` \| `unknown` |

---

## 1. `analysis/` — pure functions over pixels

No database access, no network. Everything takes a path or an ndarray and
returns plain dicts of JSON-safe primitives (floats, ints, strings, lists).
All of these must degrade gracefully: if MediaPipe is missing or finds nothing,
return `{"ok": False, "reason": "..."}` instead of raising.

### `analysis/loader.py`
```python
def load_image(path: str | Path, max_side: int = 0) -> np.ndarray
    # BGR uint8. Applies EXIF orientation. max_side=0 means no resize.
def load_rgb(path, max_side: int = 0) -> np.ndarray
def save_image(img: np.ndarray, path: str | Path, quality: int = 95) -> str
def make_thumb(src: str | Path, dst: str | Path, size: int = 512) -> str
def file_sha256(path: str | Path) -> str
def image_info(path: str | Path) -> dict
    # {"width","height","bytes","format","sha256","exif_orientation","taken_at"}
def cached(namespace: str, key: str, fn: Callable[[], dict]) -> dict
    # JSON cache in data/cache/<namespace>/<key>.json
```

### `analysis/pose.py`
```python
def detect_pose(img_bgr) -> dict
# {"ok": bool,
#  "landmarks": {name: {"x":0..1,"y":0..1,"z":float,"v":0..1}},  # normalised
#  "visible_count": int, "backend": "mediapipe"|"heuristic", "reason": str}
```
Landmark names are the MediaPipe Pose names: `nose, left_eye, right_eye,
left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow,
left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee,
left_ankle, right_ankle, left_heel, right_heel, left_foot_index,
right_foot_index`.

### `analysis/face.py`
```python
def detect_face(img_bgr) -> dict
# {"ok", "bbox":[x,y,w,h] in px, "bbox_norm":[..],
#  "landmarks": {..468 mesh points or 5 key points..},
#  "descriptor": [float,...],   # L2-normalised geometric signature, len 64
#  "yaw","pitch","roll": float degrees, "backend": str}
def face_descriptor(img_bgr, face: dict) -> list[float]
def compare_faces(desc_a, desc_b) -> float   # 0..1 similarity, 1 = identical
```
The descriptor is **geometric + photometric**, not a deep embedding: ratios
between inter-ocular distance, nose width, mouth width, jaw width, face height,
eye/eyebrow spacing, plus normalised colour statistics of eye/lip/hair regions.
It must be invariant to scale and in-plane rotation, and reasonably stable
across +/- 25 degrees of yaw.

**It is NOT the identity gate and must never be used as one.** Measured against
the stored profile it cannot separate people at all: 24 photographs of the
subject scored 0.9832-0.9993 and 8 photographs of 8 other women 0.9577-0.9945 -
overlapping populations, with the old `face_min` of 0.72 a quarter of the scale
below both. Identity is decided by `identity/embedding.py` (below); this
descriptor survives only as a consistency signal.

### `identity/embedding.py`
```python
def available() -> bool                 # weights present and loadable
def unavailable_reason() -> str         # Spanish, for the client, "" when available
def face_embedding(img_bgr, face: dict | None) -> list[float] | None   # len 128
def similarity(a, b) -> float | None    # cosine, 0..1, 1 = same person
def self_consistency(embeddings) -> dict   # {"n", "min", "mean"}
DIMS = 128
```
SFace (`cv2.FaceRecognizerSF`) plus YuNet (`cv2.FaceDetectorYN`), weights from
the OpenCV Zoo (Apache-2.0) under `backend/app/models/*.onnx`, gitignored and
fetched by `scripts/fetch_face_model.py` with sha256 verification. Runs entirely
locally; no image ever leaves the machine. The head is cut out with the mesh and
the orientation YuNet is most confident about over the four right angles is the
one embedded - three of the subject's photographs are stored rotated and were
being embedded upside down, reading 0.15-0.18, below every impostor. Degrades to
`None` instead of raising; callers must report "not computed", never a pass.

### `analysis/body.py`
```python
def measure_body(img_bgr, pose: dict, mask=None, face=None) -> dict
# {"ok", "shot_type", "metrics": {name: float}, "px": {...raw pixel values...},
#  "confidence": 0..1, "reason": str, "unreliable": [str],
#  "width_profile": [[t, w_over_torso], ...],   # 9 heights of the torso
#  "shape_profile": [[t, w_over_width], ...],
#  "head_profile":  [[s, w_over_head], ...]}    # needs face= , else []

def head_profile(mask, face, height: int, width: int, img=None) -> list
```
`head_profile` hangs rows from the chin every 0.25 head lengths (1.0..8.0) and
divides each row's silhouette width by the length of her own head, so the
numbers do not change when the picture is re-framed. It abstains (returns `[]`)
when the face mesh is missing, the head is under 8 px, fewer than 4 rows survive,
or a row touches a side border. Passing `img=` lets it re-measure the head on a
fixed-size crop, averaged with its mirror; without it the whole-frame mesh is
used and carries the framing drift.
`metrics` — **all normalised by torso length**, never by image size, and never
shoulder-over-hip alone (a uniform slim-down leaves that ratio unchanged; this
is the exact failure the client experienced):

| metric | definition |
|---|---|
| `shoulder_w_over_torso` | shoulder width / torso length |
| `hip_w_over_torso` | hip width / torso length |
| `waist_w_over_torso` | silhouette width at mid-torso / torso length |
| `shoulder_over_hip` | shoulder width / hip width (secondary, reported not gated) |
| `head_h_over_torso` | head height / torso length |
| `arm_len_over_torso` | shoulder→wrist / torso length |
| `leg_len_over_torso` | hip→ankle / torso length (full shots only) |
| `neck_w_over_torso` | neck width / torso length |
| `bust_w_over_torso` | silhouette width at chest line / torso length |

Torso length = midpoint(shoulders) → midpoint(hips) in pixels.
When a metric cannot be computed it is simply absent from the dict.

### `analysis/skin.py`
```python
def skin_stats(img_bgr, pose: dict, face: dict) -> dict
# {"ok", "lab_mean":[L,a,b], "lab_std":[..], "rgb_mean":[r,g,b],
#  "ita_deg": float, "samples": int, "regions": {"face":[..],"chest":[..],"arm":[..]}}
def compare_skin(a: dict, b: dict) -> dict
# {"delta_e": float, "delta_L": float, "ita_delta": float, "similarity": 0..1}
```
Delta-E is CIE76 over the mean Lab. `similarity = clamp(1 - deltaE/25, 0, 1)`.

### `analysis/segment.py`
```python
def person_mask(img_bgr) -> dict
# {"ok", "mask": np.uint8 HxW 0/255, "coverage": 0..1, "backend": str}
def region_masks(img_bgr, pose: dict, person: np.ndarray | None) -> dict
# {"face","hair","upper_body","lower_body","arms","hands","legs","background"}
#   each np.uint8 HxW 0/255; missing regions omitted
def garment_mask(img_bgr, pose, person_mask) -> np.ndarray
def bbox_of(mask) -> list[int]   # [x,y,w,h], [] when empty
```

### `analysis/quality.py`
```python
def assess_quality(img_bgr, path: str | None = None) -> dict
# {"ok", "score": 0..1, "sharpness": float, "exposure": 0..1,
#  "contrast": float, "noise": float, "resolution_ok": bool,
#  "beauty_filter_suspected": bool, "issues": [str], "advice": [str]}
```
`beauty_filter_suspected` fires when facial skin texture energy is far below
the energy of non-facial skin (the signature of a smoothing filter).

### `analysis/shot.py`
```python
def classify_shot(img_bgr, pose: dict, face: dict) -> dict
# {"shot_type": "closeup"|"half"|"full"|"unknown", "confidence": 0..1,
#  "framing": {"head_room":..,"subject_fill":..}, "orientation":"portrait"|"landscape"|"square"}
```

### `analysis/anomaly.py`
```python
def scan_anomalies(img_bgr, pose: dict, face: dict, masks: dict) -> dict
# {"ok", "defects": [Defect], "score": 0..1,
#  "unjudged": [{"where": "left_hand"|"right_hand"|"hand",
#                "reason": str,        # Spanish, e.g. "sale del encuadre"
#                "bbox": [x, y, w, h]}]}
```
`Defect`:
```python
{"type": "hand_malformed" | "extra_limb" | "extra_person" | "face_distorted"
       | "eye_asymmetry" | "missing_limb" | "texture_smear" | "duplicated_feature"
       | "border_artifact" | "oversmoothed_skin",
 "where": "left_hand" | "face" | ...,
 "bbox": [x, y, w, h],          # pixels, [] if global
 "severity": 0..1,
 "repairable": bool,            # True -> localized inpaint can fix it
 "detail": str}
```
Hand checks use MediaPipe Hands: finger count, digit length ratios, palm area
vs wrist width, and count of detected hands vs count of detected wrists.

`unjudged` is the other half of the contract and it is not optional: a hand that
is cut by the frame, or smaller than `HAND_MIN_PX` (170), cannot produce a
severity that reaches `ANATOMY_SEVERITY_MAX` (0.60) however deformed it is, so an
empty `defects` list does NOT mean the hands were checked. Measured 2026-09-04
over the 24 originals and every generated frame on disk, 26 of 38 detected hands
survive the truncation test and none of them reaches 170 px (largest 157,
median 68); no geometric or edge-energy measure separates her real hands from
melted generated ones at that size. Callers must report `unjudged` rather than
imply the hands passed.

---

## 2. `identity/`

### `identity/profile.py`
```python
def build_profile(image_paths: list[str], person_name: str) -> dict
def profile_from_analyses(analyses: list[dict], person_name: str) -> dict
```
Returns the **IdentityProfile** stored across the `profiles` columns:
```python
{
 "person_name": str,
 "n_sources": int,
 "coverage": {"closeup": int, "half": int, "full": int, "unknown": int,
              "ready_for_body_check": bool, "missing": [str], "advice": [str]},
 "face":  {"descriptor": [64 floats], "descriptor_std": [...],
           "n": int, "yaw_range": [min,max],
           # the signature identity is actually decided on, see embedding.py:
           "embeddings": [[128 floats], ...], "embedding_mean": [128 floats],
           "embedding_n": int,
           "embedding_self": {"n": int, "min": f, "mean": f}},
 "body":  {metric: {"mean": f, "std": f, "n": int, "lo": f, "hi": f,
                    "spread": f, "dropped": int,
                    "gated": bool,        # may this band reject on its own?
                    "band_capped": bool}},# the +/-12% cap had to narrow it
 "skin":  {"lab_mean": [...], "lab_std": [...], "ita_deg": f, "n": int},
 "hair":  {"lab_mean": [...], "length": "short"|"medium"|"long", "n": int},
 "marks": [{"type":"tattoo","region":str,"bbox_norm":[..],"seen_in":int}],
 # face_embed_min is the identity gate: cosine of the SFace embedding
 # against embedding_mean. Calibrated on measured populations - her own
 # photographs 0.6429-0.8715 leave-one-out, 8 other women 0.0194-0.1948, six
 # generated faces that are not her 0.1829-0.3968 - so 0.45 sits inside an
 # empty band. face_min (0.72) is the retired geometric-descriptor threshold,
 # kept only so old stored profiles still load; nothing reads it to decide.
 "thresholds": {"face_embed_min": 0.45, "face_min": 0.72, "delta_e_max": 8.0,
                "metric_tol_sigma": 2.5, "metric_tol_floor": 0.06},
 "sources": [{"path":str,"sha256":str,"shot_type":str}]
}
```
`lo`/`hi` are the accepted band: `mean ± max(tol_floor*mean, sigma*std)`,
clipped to `± BAND_MAX_REL` (12%) of the mean. When that cap bites, the band is
narrower than the photographs it was learned from, so `band_capped` is True and
`gated` is False: the metric is reported but never rejects.
**Originals can be deleted afterwards** — the profile alone is sufficient.

### `identity/verify.py`
```python
def verify_image(image_path: str, profile: dict, brief: dict | None = None) -> dict
```
Returns the **Verdict**:
```python
{"passed": bool,
 "score": 0..1,                       # weighted mean of check scores
 "checks": [{"name": str, "value": float, "threshold": float,
             "passed": bool, "weight": float, "detail": str,
             # optional, and every one of them means "this check passed but
             # did not verify what its name promises":
             "advisory": bool,        # the engine could not have caused it
             "parcial_es": str,       # body: only a NARROWING could still fail
             "unjudged_es": str}],    # anatomy: "2 manos" nobody could judge
 "defects": [Defect],
 "repairable_defects": [Defect],
 "summary": str,                      # Spanish, shown to the user
 "elapsed_ms": int}
```
Check names and default weights:
`identity_face` .30, `body_proportions` .25, `skin_tone` .15,
`anatomy` .20, `quality` .10.
`body_proportions` prefers the **paired** comparison: when `brief["source_path"]`
is known, the generated image is measured against that same photograph instead of
against the population band. Three rulers are compared, each as a median ratio:
the skeletal metrics, the silhouette `width_profile`/`shape_profile` (9 heights of
the torso), and `head_profile` — the figure measured at up to 29 heights below the
chin in units of her own head length, which is the only ruler that survives a
reframe (`HEAD_TOL` 0.04, `PAIRED_TOL` 0.08 widened by the measured pair noise up
to `PAIRED_TOL_MAX` 0.16; the torso rulers abstain when the torso unit has moved
against the head unit by more than `UNIT_SHIFT_MAX` 0.06, i.e. the framing changed).
When the brief changes the clothing, the torso rulers stand down (a coat moves
them for honest reasons) and `head_profile` is read **one-sided**: a figure that
came back wider is reported and excused, a figure that came back narrower still
rejects, because no garment in the catalogue removes volume.
Only when no source photograph can be measured does it fall back to the population
band, and there the check fails if a *gated* metric falls outside its band. Gated
metrics: `shoulder_w_over_torso`, `hip_w_over_torso`, `waist_w_over_torso`,
`bust_w_over_torso`, `head_h_over_torso` — each of them only while its band is not
`band_capped`.

---

## 3. `generation/`

### `generation/prompt.py`
```python
def build_prompt(brief: dict, profile: dict, style: dict, options: dict) -> dict
# {"prompt": str, "negative_prompt": str, "identity_clause": str,
#  "params": {"strength":..,"guidance":..,"steps":..}, "tokens": [str]}
def repair_prompt(defect: dict, brief: dict, profile: dict) -> dict
```
Every prompt carries a mandatory **identity clause** and every negative prompt
carries the mandatory **no-beautify block**:
`slimmer body, slimmed waist, narrowed shoulders, reshaped face, airbrushed
skin, plastic skin, beauty filter, face slimming, body slimming, changed skin
tone, removed tattoos, altered breast size, different person`.

### `generation/planner.py`
```python
def plan_run(brief: dict, options: dict, n_previews: int, profile: dict,
             style: dict, learning: dict | None = None) -> dict
# {"variants": [Variant], "locked": {group: value}, "varied": [group],
#  "notes": [str]}
```
`Variant`: `{"index": int, "choices": {group: value_key}, "seed": int,
"params": {...}, "why": str}`.
Rule from the client: a group with **one** chosen value is *locked* across all
variants; a group with several is *crossed*; a group with **none** is left free
and the planner varies it deliberately so previews are not near-duplicates.

### `generation/router.py`
```python
def choose_provider(req_kind: str, quality: str, budget_usd: float,
                    prefer: str | None = None,
                    changes=None) -> tuple[ImageProvider, str, str]
# -> (provider, model, reason)
# `changes` = the option groups the user asked to change (the variant's
# `choices` dict is accepted as-is).  A provider whose `Capabilities.generative`
# is False cannot perform `clothing`, `pose`, `expression`, `hair` or
# `transparency`, so it is dropped from the candidates instead of winning on
# price; when only that provider is left it is still returned - the run never
# fails - and `reason` names, in Spanish, the changes that will not be applied.
def unsupported_changes(provider, changes=None) -> list[str]
def estimate_run_cost(plan: dict, quality: str, limits: dict | None = None,
                      user_id: str = "") -> dict
# {"total_usd": f, "per_image_usd": f, "provider": str, "model": str,
#  "breakdown": [...], "free": bool, "total_max_usd": f, "intentos_maximos": int,
#  "aviso_coste": str, "aviso_opciones": str, ...}

# The retry/repair limits this user set in Ajustes, clamped by the configured
# maxima.  The CALLER passes the same dict to the run: reading the setting twice
# is how an estimate that prices two attempts ends up beside a run that buys
# three.  {"max_retries": int, "max_repair_rounds": int}
def user_limits(user_id: str) -> dict

# Paid images per catalogue option and how many were really her, seeded from a
# hand-measured table (SEED_HISTORY, re-scored 2026-09-04 off the files on disk)
# and extended from the database for everything paid after SEED_UNTIL.  Only
# attempts whose stored verdict carries identity_face.threshold == 0.45 count:
# 0.72 is the retired blind descriptor that read 0.99 on a stranger.
def option_history(user_id: str = "") -> dict[tuple[str, str], tuple[int, int]]
```

### `generation/protect.py`
```python
def plan_mask(choices: dict) -> dict
# {"safe": bool, "regions": [str], "blocked": [str], "groups": [str],
#  "reason": str} - from the option GROUPS alone, before any photograph is
# opened.  Safe when nothing the user asked for moves the person inside the
# frame (clothing, colour, scene); not safe for pose, framing, expression,
# hair.

def shield_for(source_path, choices, work_dir=None) -> dict
# {"masked": bool, "mask_path": str, "cover": float, "regions": [str],
#  "blocked": [str], "groups": [str], "reason": str,
#  "estado": "dibujada"|"reutilizada"|"sin dibujar"|"sin zona"|"bloqueado"}
# THE ONE DECIDER.  Runs plan_mask AND draws the real mask once, into the run's
# own folder, under a name made of the photograph and the region set; a lock
# makes it safe for the parallel variants.  Both the ESTIMATE and the RUN call
# this, so the model that is priced is the model that is sent.  It can answer
# "not masked" for pixel reasons too: no face found, no region located, cover
# below MIN_COVER 2% or above MAX_COVER 92%.

def compose(source_path, painted_path, mask_path, out_path, quality=96) -> dict
# {"ok": bool, "fuera_cambiado": int, "cover": float, "comprobante": str,
#  "reason": str}
# Puts her own pixels back everywhere the mask is black and REFUSES to write
# the file unless exactly 0 pixels outside the mask differ from her photograph.
```
White repaints, black stays hers. The face and both hands are grown by
`PROTECT_MARGIN` and forced back to black after the feather, so nothing bleeds
into them. `verify` is told which area was repainted (`brief["repaint_mask"]`)
and a finding whose box is under 10% inside it is reported as `de_tu_foto`: it
can never fail the image, never buy a repaint and never teach the learner.

### `generation/repair.py`
```python
def repair(image_path: str, defects: list[dict], brief: dict, profile: dict,
           provider, out_path: str) -> dict
# {"ok", "image_path", "repaired": [defect types], "cost_usd", "rounds": int}
```
Builds a feathered mask around the defect bbox (dilated 6% of the shorter side)
and repaints **only** that region.

### `generation/orchestrator.py`
The robot. Single entry points, both synchronous, both safe to run in a thread:

`prepare_run` also reads the source photograph with a vision provider through
`_read_photo(user, original, analysis)`, which is the only place in the product
that spends money BEFORE the user presses the button. It is gated with
`billing.can_spend`, charged to the ledger under `"anthropic"`, cached inside
`originals.analysis_json["vision"]` so a photograph is read once, and it puts
the sentence into the estimate warnings. With no balance it falls back to the
free local reader and says so.

```python
def run_previews(user: dict, run_id: str) -> dict
def run_final(user: dict, run_id: str) -> dict
```
They read the `runs` row, do the work, write `attempts`/`images`/`ledger`
rows, update `runs.progress`/`stage`, and return a summary dict.
Pipeline per variant:
`analyze → guard → prompt → route → budget-gate → generate → verify →
 repair (≤2 rounds) → re-verify → accept | retry (≤2) | reject`.
**Budget gate**: before every provider call, `services/billing.can_spend()`.
On refusal the run stops with status `stopped_no_balance`, an alert row is
written, and no further calls are made.

### `generation/learning.py`
```python
def record_feedback(user_id: str, image_id: str, verdict: str, reason: str = "") -> None
def get_weights(user_id: str, scope: str = "global") -> dict
def apply_learning(plan: dict, weights: dict) -> dict
```
Weights are per option value and per parameter, updated with a simple
exponential rule (`w = w*(1-a) + a*outcome`, a = 0.2), plus per-defect-type
counters that push `strength` down when the same defect keeps appearing.

---

## 4. `services/`

### `services/billing.py`
```python
def balance(user_id: str, provider: str) -> float
def all_balances(user_id: str) -> dict          # {provider: {"balance","spent_30d",...}}
def recharge(user_id: str, provider: str, amount_usd: float, note: str = "") -> dict
def can_spend(user_id: str, provider: str, amount_usd: float) -> dict
# {"ok": bool, "reason": str, "balance": f, "remaining_daily": f,
#  "remaining_monthly": f, "alert": dict | None}
def charge(user_id: str, provider: str, amount_usd: float, ref: str, note: str = "") -> float
def usage(user_id: str, days: int = 30) -> dict
def check_and_raise_alerts(user_id: str) -> list[dict]
```
Rules the client asked for, verbatim:
* warn while there is still balance (`low_balance_usd`, `critical_balance_usd`);
* at zero, **stop generating** and alert immediately;
* **never** auto-recharge a card — `recharge()` is only ever called from an
  explicit user action.

### `services/storage.py`
```python
def user_dir(user_id: str, kind: str) -> Path
def store_upload(user_id: str, filename: str, data: bytes) -> dict
def store_output(user_id: str, src: Path, kind: str, run_id: str) -> dict
def delete_file(path: str) -> bool
def public_url(path: str) -> str            # -> /api/files/<token>
```

### `services/jobs.py`
```python
def submit(run_id: str, fn: Callable[[], Any]) -> None
def status(run_id: str) -> dict
def cancel(run_id: str) -> bool
```
A bounded `ThreadPoolExecutor` (max 2) with a run-id keyed registry.

---

## 5. `safety/`

### `safety/guard.py`
```python
def check_request(brief: dict, options: dict, profile: dict, user: dict) -> dict
# {"allowed": bool, "reason": str, "code": str, "blocked_terms": [str]}
def check_upload(image_path: str) -> dict
# {"allowed": bool, "reason": str, "flags": [str]}
```
Blocks: sexual/intimate imagery of real identifiable people, minors in any
suggestive framing, and prompts that name a third party. This is the boundary
the developer already stated to the client; the code enforces it rather than
relying on a promise.

### `safety/consent.py`
```python
def record_consent(user_id: str, profile_id: str, payload: dict) -> dict
def has_valid_consent(profile_id: str) -> bool
def revoke(profile_id: str, reason: str) -> None
```
Consent payload: `{"granted_by","relationship":"self"|"client","statement",
"signed_at","ip","evidence_note","scope":[...]}`.

---

## 6. HTTP API

All under `/api`. JSON in, JSON out. Auth via `Authorization: Bearer <token>`
or the `pr_session` cookie. Errors: `{"detail": "mensaje en espanol"}`.

```
POST   /api/auth/register        {email,password,display_name} -> {user, token?, needs_approval}
POST   /api/auth/login           {email,password} -> {user, token, expires_at}
POST   /api/auth/logout
GET    /api/auth/me              -> {user, balances, alerts_unread}

GET    /api/profiles                       -> [profile]
POST   /api/profiles                       {person_name} -> profile
POST   /api/profiles/{id}/build            {original_ids?} -> profile (measures everything)
POST   /api/profiles/{id}/consent          {...} -> profile
DELETE /api/profiles/{id}
POST   /api/profiles/{id}/forget-originals -> {deleted: n}   # keeps measurements

GET    /api/originals?profile_id=          -> [original]
POST   /api/originals            multipart files[] -> [original]
POST   /api/originals/import-folder        {path} (admin) -> [original]
PATCH  /api/originals/{id}                 {sort_order?, profile_id?, tags?}
DELETE /api/originals/{id}
GET    /api/originals/{id}/analysis        -> analysis report

GET    /api/catalog/options?shot_type=&original_id=  -> {groups:[...], suggested:[...]}
GET    /api/catalog/styles?shot_type=                -> [style]
POST   /api/generate/analyze     {original_id, options, n_previews, quality}
                                  -> {run_id, estimate, plan_summary, warnings}
POST   /api/generate/run         {run_id} -> {run_id, status}
GET    /api/generate/status/{run_id} -> {status, progress, stage, images:[...], report}
POST   /api/generate/final       {run_id, image_ids[], quality} -> {run_id}
POST   /api/generate/cancel/{run_id}
GET    /api/generate/report/{run_id} -> full ficha: attempts, costs, defects, decisions

GET    /api/album?kind=&limit=&offset=   -> {images, total}
DELETE /api/album/{image_id}
GET    /api/album/{image_id}/download
POST   /api/album/{image_id}/feedback    {verdict, reason}
POST   /api/album/{image_id}/final       -> {ok, kind: "final", cost_usd: 0.0, mensaje}
       # Relabels an image that ALREADY passed every check as a final, for
       # 0.00 USD.  409 if the verdict did not pass, 410 if the file is gone.
       # run_final re-renders and bills; this does not, because the pixels are
       # already bought.

GET    /api/favorites                    -> {images}
POST   /api/favorites/{image_id}
DELETE /api/favorites/{image_id}

GET    /api/settings                     -> {settings, keys, limits, catalog}
PUT    /api/settings                     {key: value}
POST   /api/settings/keys                {provider, key}  (stored server side)
DELETE /api/settings/keys/{provider}
GET    /api/settings/usage?days=30       -> usage rollup
POST   /api/settings/recharge            {provider, amount_usd}
GET    /api/settings/alerts              -> [alert]
POST   /api/settings/alerts/{id}/read

GET    /api/admin/users                  -> [user]
POST   /api/admin/users/{id}/approve
POST   /api/admin/users/{id}/suspend
PATCH  /api/admin/users/{id}             {role, limits, plan, free_quota_daily}
DELETE /api/admin/users/{id}
GET    /api/admin/stats                  -> platform totals
GET    /api/admin/audit?limit=           -> audit rows
GET    /api/admin/providers              -> provider availability matrix

GET    /api/files/{image_id}?variant=full|thumb   (auth-checked file serving)
GET    /api/health                       -> {ok, app, version, python,
                                            codigo, providers}
         codigo = {huella, modulos, mas_reciente, arrancado, al_dia}
         al_dia is False when a watched module on disk is NEWER than the
         process that imported it - i.e. the server is serving stale code.
         Added after a --no-reload server quoted a 0.080 USD whole-frame
         endpoint for a run the files on disk priced at 0.050 with a mask.
```

---

## 7. Frontend

`frontend/index.html` + vanilla ES modules. **No build step** — it must run by
opening the served page on an iPhone and on a desktop browser with nothing
installed. Mobile-first, safe-area aware, 44px minimum touch targets.

Pages (hash router): `#/login`, `#/generate`, `#/album`, `#/favorites`,
`#/originals`, `#/settings`, `#/admin`.

Shared JS modules:
* `js/api.js` — `api.get/post/put/del/upload`, token storage, 401 → login.
* `js/store.js` — tiny observable state.
* `js/router.js` — hash router + page registry.
* `js/ui.js` — DOM helpers (`el`, `toast`, `modal`, `confirm`, `spinner`).
* `js/i18n.js` — Spanish default, English fallback strings.

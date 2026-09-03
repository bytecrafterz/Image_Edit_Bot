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

### `analysis/body.py`
```python
def measure_body(img_bgr, pose: dict, mask: np.ndarray | None = None) -> dict
# {"ok", "shot_type", "metrics": {name: float}, "px": {...raw pixel values...},
#  "confidence": 0..1, "reason": str}
```
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
# {"ok", "defects": [Defect], "score": 0..1}
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
           "n": int, "yaw_range": [min,max]},
 "body":  {metric: {"mean": f, "std": f, "n": int, "lo": f, "hi": f}},
 "skin":  {"lab_mean": [...], "lab_std": [...], "ita_deg": f, "n": int},
 "hair":  {"lab_mean": [...], "length": "short"|"medium"|"long", "n": int},
 "marks": [{"type":"tattoo","region":str,"bbox_norm":[..],"seen_in":int}],
 "thresholds": {"face_min": 0.72, "delta_e_max": 8.0,
                "metric_tol_sigma": 2.5, "metric_tol_floor": 0.06},
 "sources": [{"path":str,"sha256":str,"shot_type":str}]
}
```
`lo`/`hi` are the accepted band: `mean ± max(tol_floor*mean, sigma*std)`.
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
             "passed": bool, "weight": float, "detail": str}],
 "defects": [Defect],
 "repairable_defects": [Defect],
 "summary": str,                      # Spanish, shown to the user
 "elapsed_ms": int}
```
Check names and default weights:
`identity_face` .30, `body_proportions` .25, `skin_tone` .15,
`anatomy` .20, `quality` .10.
A run of `body_proportions` compares every metric present in **both** the
generated image and the profile; the check fails if any *gated* metric falls
outside its band. Gated metrics: `shoulder_w_over_torso`, `hip_w_over_torso`,
`waist_w_over_torso`, `bust_w_over_torso`, `head_h_over_torso`.

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
def estimate_run_cost(plan: dict, quality: str) -> dict
# {"total_usd": f, "per_image_usd": f, "provider": str, "model": str,
#  "breakdown": [...], "free": bool}
```

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
GET    /api/health                       -> {ok, version, providers, python}
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

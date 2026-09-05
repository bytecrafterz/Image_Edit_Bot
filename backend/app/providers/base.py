"""Provider abstraction.

The whole point of this layer is that the orchestrator never names a vendor.
It asks the registry for "an image provider that can inpaint at this quality"
and gets one back.  Swapping fal.ai for something cheaper later is a registry
edit, not a rewrite - which is exactly what the client asked for.

Two families:

  ImageProvider   - makes pixels (generate / inpaint / upscale)
  VisionProvider  - reads pixels and returns judgement in words or numbers

Both have a zero-cost local implementation so the entire pipeline runs with no
API keys at all.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

Operation = Literal["generate", "inpaint", "upscale"]
Quality = Literal["draft", "preview", "standard", "high", "max"]


# ------------------------------------------------------------------ requests

@dataclass
class GenRequest:
    """Everything a provider needs to make one image.

    ``source_path`` is the anchor: the transformation always starts from a real
    photograph of the person, never from noise.  A provider that only supports
    text to image must declare ``img2img=False`` in capabilities so the router
    never hands it identity critical work.
    """
    prompt: str
    source_path: str | None = None          # img2img anchor
    mask_path: str | None = None            # white = repaint, black = keep
    reference_paths: list[str] = field(default_factory=list)  # identity refs
    negative_prompt: str = ""
    operation: Operation = "generate"
    quality: Quality = "preview"
    width: int = 0                          # 0 = follow source
    height: int = 0
    strength: float = 0.55                  # img2img denoise; lower = closer to source
    guidance: float = 4.0
    steps: int = 28
    seed: int | None = None
    identity_weight: float = 0.85           # how hard to hold the face/body
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GenResult:
    """What a provider gives back.  ``cost_usd`` must be real, not a guess:
    the client explicitly wants to see what each image cost."""
    ok: bool
    image_path: str | None = None
    provider: str = ""
    model: str = ""
    cost_usd: float = 0.0
    latency_ms: int = 0
    seed: int | None = None
    error: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Capabilities:
    name: str
    kind: str = "image"                     # image | vision
    img2img: bool = True
    inpaint: bool = True
    upscale: bool = False
    text2img: bool = False
    identity_reference: bool = False        # accepts reference images for identity
    # False for engines that only transform the photograph they are handed -
    # background, fabric colour, light, grade, crop.  Those cannot put the
    # person in other clothes, another pose, another expression or other hair,
    # so the router must not hand them that work and then charge the user with
    # six images that were never what she asked for.  True by default: every
    # remote generative provider behaves exactly as before.
    generative: bool = True
    max_side: int = 2048
    # Longest side of the file the provider really HANDS BACK, when that is
    # smaller than what it accepts.  0 means "the same as max_side".  It exists
    # because fal's Kontext endpoints take a 2048 px photograph and answer with
    # about one megapixel whatever the quality tier charged for, and a tier that
    # cannot say so is selling pixels that never arrive.
    out_max_side: int = 0
    needs_key: bool = True
    key_name: str = ""
    cost_per_image_usd: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class ProviderError(RuntimeError):
    """Raised for provider side failures that the orchestrator may retry.

    ``billed`` is the difference between "the call never happened" and "the
    call happened, the vendor charged for it, and what came back is unusable".
    Measured on this installation on 2026-09-04: two fal jobs ran to completion
    on fal-ai/flux-pro/v1/fill - 19.1 s and 3.3 s of GPU, HTTP 200 - and
    returned an all black PNG because fal's own safety checker had flagged the
    result.  The orchestrator handed the reservation back, so the run reported
    0.0000 USD for work a vendor bills at 0.050 USD an image.  A ledger that
    only records the calls that went well is not a ledger.

    ``meta`` is the same dictionary a successful GenResult carries, attached
    to the failure as well.  It exists because of what the block of
    2026-09-05 cost to investigate: the attempt row said only "fal devolvio un
    archivo en negro", and the request_id that identifies the job on fal's side
    had to be read out of an httpx log line, because the provider built it and
    then dropped it on the way out.  An error that a vendor charges for has to
    leave the same trail as a success.
    """

    def __init__(self, message: str, *, retryable: bool = True, code: str = "",
                 billed: bool = False, meta: dict | None = None,
                 latency_ms: int = 0):
        super().__init__(message)
        self.retryable = retryable
        self.code = code
        self.billed = billed
        self.meta = dict(meta or {})
        self.latency_ms = int(latency_ms or 0)


class InsufficientBalance(ProviderError):
    """Raised when the remote account has no money left.

    The orchestrator turns this into a hard stop plus a user alert - it must
    never silently keep trying, and it must never trigger a top up.
    """

    def __init__(self, message: str, provider: str = ""):
        super().__init__(message, retryable=False, code="insufficient_balance")
        self.provider = provider


# ----------------------------------------------------------------- interface

class ImageProvider(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def capabilities(self) -> Capabilities:
        ...

    @abc.abstractmethod
    def available(self) -> bool:
        """True when this provider can actually run right now (keys present)."""

    @abc.abstractmethod
    def estimate_cost(self, req: GenRequest) -> float:
        ...

    @abc.abstractmethod
    def generate(self, req: GenRequest, out_path: str | Path) -> GenResult:
        ...

    def inpaint(self, req: GenRequest, out_path: str | Path) -> GenResult:
        """Default: providers that do not distinguish the two just repaint."""
        req.operation = "inpaint"
        return self.generate(req, out_path)

    def upscale(self, req: GenRequest, out_path: str | Path) -> GenResult:
        raise ProviderError(f"{self.name} cannot upscale", retryable=False)


class VisionProvider(abc.ABC):
    """Reads an image and answers structured questions about it."""

    name: str = "base-vision"

    @abc.abstractmethod
    def available(self) -> bool:
        ...

    @abc.abstractmethod
    def estimate_cost(self, n_images: int = 1) -> float:
        ...

    @abc.abstractmethod
    def describe_photo(self, image_path: str, context: dict | None = None) -> dict:
        """Return a structured reading of the source photo.

        Contract (all keys always present, values may be empty):
            {"shot_type", "subject", "clothing", "hair", "expression",
             "pose", "setting", "lighting", "camera", "colors",
             "preserve": [...], "notes": str, "cost_usd": float}
        """

    @abc.abstractmethod
    def critique_result(self, image_path: str, brief: dict,
                        reference_path: str | None = None) -> dict:
        """Judge a generated image against the brief.

        Contract:
            {"ok": bool, "score": 0..1,
             "defects": [{"type","where","severity","repairable","bbox"}],
             "identity_notes": str, "cost_usd": float}
        """

    def write_prompt(self, brief: dict) -> dict:
        """Optional: author a prompt from a structured brief.

        Returning ``{}`` means "no opinion" and the deterministic builder in
        generation/prompt.py is used unchanged.
        """
        return {}

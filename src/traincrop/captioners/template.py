"""Offline template captioner.

Needs nothing beyond Pillow and NumPy. It fills a format string with
attributes measured from the crop itself -- head direction from the detector's
landmarks, lighting and colour from the pixels -- so every caption is
deterministic and reproducible. Useful as a fallback, for dry runs, and as the
baseline you edit by hand.
"""

from __future__ import annotations

import numpy as np

from .base import BaseCaptioner, CaptionRequest

DEFAULT_TEMPLATE = "{shot}, {pose}, {light}, {tone}"

#: Every token a template may use.
TOKENS = ("shot", "pose", "light", "tone", "label", "tier", "index", "source")


class TemplateCaptioner(BaseCaptioner):
    name = "template"

    def __init__(
        self,
        template: str = DEFAULT_TEMPLATE,
        *,
        padding: float = 0.0,
        **caption_kwargs,
    ) -> None:
        super().__init__(**caption_kwargs)
        self.template = template
        self.padding = padding

    def _describe(self, request: CaptionRequest) -> str:
        tokens = {
            "shot": _shot_type(request.region.label, self.padding),
            "pose": _head_pose(request),
            "light": _lighting(request),
            "tone": _tone(request),
            "label": request.region.label,
            "tier": str(request.tier),
            "index": str(request.index),
            "source": request.source_path,
        }
        try:
            return self.template.format(**tokens)
        except KeyError as exc:
            raise ValueError(
                f"unknown token {exc} in caption template; available: "
                f"{', '.join(sorted(tokens))}"
            ) from None


def _shot_type(label: str, padding: float) -> str:
    """Name the framing from how much context the padding pulled in."""
    if label not in ("face", "head"):
        return {
            "person": "full body shot",
            "upper": "upper body shot",
            "torso": "torso shot",
            "legs": "shot of the legs",
            "hands": "shot of the hands",
        }.get(label, f"{label} shot")
    if padding <= 0.35:
        return "portrait"
    if padding <= 0.75:
        return "head and shoulders portrait"
    return "upper body portrait"


def _head_pose(request: CaptionRequest) -> str:
    """Estimate yaw from how far the nose sits between the eyes.

    On a frontal face the nose is midway between the pupils; as the head turns
    it slides toward the nearer eye. The ratio is scale-invariant, so it works
    the same on a 200 px and a 2000 px face.
    """
    marks = request.region.landmarks
    left, right, nose = marks.get("left_eye"), marks.get("right_eye"), marks.get("nose")
    if not (left and right and nose):
        return "facing the camera"

    eye_span = abs(left[0] - right[0])
    if eye_span < 1e-3:
        return "in profile"

    midpoint = (left[0] + right[0]) / 2.0
    # +ve means the nose has moved toward the image-left eye.
    offset = (nose[0] - midpoint) / eye_span
    magnitude = abs(offset)
    if magnitude < 0.12:
        turn = "facing the camera"
    elif magnitude < 0.30:
        turn = "head turned slightly to the {side}"
    elif magnitude < 0.55:
        turn = "head turned to the {side}"
    else:
        turn = "head in near profile to the {side}"
    return turn.format(side="left" if offset > 0 else "right")


def _luma(request: CaptionRequest) -> np.ndarray:
    arr = np.asarray(request.image.convert("L"), dtype=np.float32) / 255.0
    return arr


def _lighting(request: CaptionRequest) -> str:
    luma = _luma(request)
    mean = float(luma.mean())
    spread = float(luma.std())

    if mean < 0.25:
        level = "dim low-key lighting"
    elif mean < 0.45:
        level = "moody lighting"
    elif mean < 0.68:
        level = "even natural lighting"
    else:
        level = "bright lighting"

    # Compare the outer thirds: a strong imbalance reads as a side key light.
    third = max(1, luma.shape[1] // 3)
    left = float(luma[:, :third].mean())
    right = float(luma[:, -third:].mean())
    imbalance = left - right
    if abs(imbalance) > 0.09:
        level += " from the left" if imbalance > 0 else " from the right"
    elif spread > 0.26:
        level += " with strong contrast"
    return level


def _tone(request: CaptionRequest) -> str:
    arr = np.asarray(request.image.convert("RGB"), dtype=np.float32) / 255.0
    # Saturation as max-minus-min per pixel: cheap, and enough to separate
    # greyscale from muted from vivid.
    saturation = float((arr.max(axis=2) - arr.min(axis=2)).mean())
    if saturation < 0.05:
        return "black and white"
    if saturation < 0.14:
        return "muted desaturated colors"
    if saturation < 0.30:
        return "natural colors"
    return "vivid saturated colors"

"""Run configuration: one dataclass, loadable from TOML and overridable by CLI flags."""

from __future__ import annotations

import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .geometry import BaseSideMode, EdgeMode
from .images import FillMode


@dataclass
class Config:
    # -- where -----------------------------------------------------------
    input: Path = Path(".")
    output: Path = Path("./dataset")
    exclude: tuple[str, ...] = ()
    follow_symlinks: bool = False

    # -- detection -------------------------------------------------------
    detector: str = "yunet"
    detector_opts: dict[str, Any] = field(default_factory=dict)
    min_score: float = 0.8
    multi: str = "all"          # all | largest | skip
    max_per_image: int = 0      # 0 = unlimited

    # -- crop geometry ---------------------------------------------------
    padding: float = 0.20       # fraction of the base side, added to EACH side
    sizes: tuple[int, ...] = (512, 768, 1024)
    min_ratio: float = 0.80
    base_mode: BaseSideMode = "max"
    offset_x: float = 0.0
    offset_y: float | None = None
    """None defers to the detector's ``recommended_offset_y``; an explicit
    value (including 0.0) overrides it."""
    edge: EdgeMode = "shift"
    fill: FillMode = "blur"
    fill_color: tuple[int, int, int] = (0, 0, 0)
    min_sharpness: float = 10.0  # Laplacian-variance floor; 0 disables the check
    upscale: bool = False
    """When a crop is smaller than its chosen tier, enlarge it with AI
    upscaling instead of a plain resize. A crop that already meets or
    exceeds the tier is always just resized down; this never applies then."""

    # -- masks -----------------------------------------------------------
    mask_faces: bool = False
    """Write a ``-masklabel.png`` beside each crop that blacks out every face
    in it, so masked training learns body, outfit and style but not identity.
    Needs a body/object detector; a face detector's crops would be all mask."""
    mask_margin: float = 0.35   # fraction of the face box added to EACH side
    mask_missing: str = "skip"  # skip | keep -- a crop where no face was found
    mask_min_score: float = 0.5
    """Lower than min_score on purpose: a missed face is learned, while a
    false positive only costs a patch of background."""
    mask_background: bool = False
    """Weight the background down in the same ``-masklabel.png``, using a
    person matte, so the model learns the subject rather than the places the
    photos were taken. Segments people, so needs a face or person detector."""
    background_weight: float = 0.1
    """Loss weight of the background, 0-1. Not 0 by default: a little weight
    keeps the model from drifting on the background it is never scored on."""
    mask_clothing: bool = False
    """Weight clothing down in the same ``-masklabel.png``: learn the person, not the outfit."""
    clothing_weight: float = 0.0
    """Loss weight of clothing, 0-1."""
    mask_dir: Path | None = None
    """Write masks here, named like their crop, instead of ``-masklabel.png``
    beside it: the layout ai-toolkit and kohya read."""

    # -- output ----------------------------------------------------------
    prefix: str = ""
    start_index: int = 1
    digits: int = 4
    format: str = "png"         # png | jpg | webp
    quality: int = 95
    per_size_dirs: bool = False

    # -- captions --------------------------------------------------------
    captioner: str = "vlm"
    caption_model: str = "smolvlm"
    caption_prompt: str = ""    # empty = the backend default
    caption_device: str = "auto"
    caption_batch: int = 4
    caption_tokens: int = 96
    caption_template: str = ""  # template backend only
    trigger: str = ""
    caption_prefix: str = ""
    caption_suffix: str = ""
    caption_drop: tuple[str, ...] = ()
    caption_max_chars: int = 0

    # -- run -------------------------------------------------------------
    workers: int = 0            # 0 = auto
    dry_run: bool = False
    resume: bool = False
    overwrite: bool = False
    quiet: bool = False
    verbose: bool = False
    limit: int = 0              # 0 = no limit

    # -- training config ---------------------------------------------------
    training_config: bool = False
    training_config_only: bool = False

    # -- alternate run modes ----------------------------------------------
    caption_only: bool = False
    """Skip detection/cropping; caption an existing --output dataset,
    rebuilding crop regions from its manifest.jsonl."""

    # -- validation ------------------------------------------------------
    def validate(self) -> None:
        problems: list[str] = []

        if not self.sizes:
            problems.append("at least one --sizes value is required")
        if any(s <= 0 for s in self.sizes):
            problems.append("--sizes values must be positive")
        if self.padding < 0:
            problems.append("--padding cannot be negative")
        if not 0 < self.min_ratio <= 1:
            problems.append("--min-ratio must be in (0, 1]")
        if not 0 <= self.min_score <= 1:
            problems.append("--min-score must be in [0, 1]")
        if self.min_sharpness < 0:
            problems.append("--min-sharpness cannot be negative")
        if self.multi not in ("all", "largest", "skip"):
            problems.append("--multi-face must be all, largest or skip")
        if self.format not in ("png", "jpg", "jpeg", "webp"):
            problems.append("--format must be png, jpg or webp")
        if not 1 <= self.quality <= 100:
            problems.append("--quality must be in [1, 100]")
        if self.digits < 1:
            problems.append("--digits must be at least 1")
        if self.edge not in ("shift", "extend", "skip"):
            problems.append("--edge must be shift, extend or skip")
        if self.fill not in ("edge", "blur", "reflect", "color"):
            problems.append("--fill must be edge, blur, reflect or color")
        for name in ("workers", "limit", "max_per_image", "start_index", "caption_max_chars"):
            if getattr(self, name) < 0:
                # A negative --limit or --max-per-image would slice from the
                # end and silently drop the last photos or detections.
                problems.append(f"--{name.replace('_', '-')} cannot be negative")
        if self.caption_batch < 1:
            problems.append("--caption-batch must be at least 1")
        if self.caption_tokens < 1:
            problems.append("--caption-tokens must be at least 1")
        if self.captioner == "template" and self.caption_template:
            problem = _check_template(self.caption_template)
            if problem:
                problems.append(problem)
        if self.mask_margin < 0:
            problems.append("--mask-margin cannot be negative")
        if self.mask_missing not in ("skip", "keep"):
            problems.append("--mask-missing must be skip or keep")
        if not 0 <= self.mask_min_score <= 1:
            problems.append("--mask-min-score must be in [0, 1]")
        if not 0 <= self.background_weight <= 1:
            problems.append("--background-weight must be in [0, 1]")
        if not 0 <= self.clothing_weight <= 1:
            problems.append("--clothing-weight must be in [0, 1]")
        if self.mask_background and not self._detects_people():
            problems.append(
                "--mask-background segments people, so it needs a face "
                "detector or --detector yolox with classes including person"
            )
        if self.mask_clothing and not self._detects_people():
            problems.append(
                "--mask-clothing parses people, so it needs a face "
                "detector or --detector yolox with classes including person"
            )
        if self.mask_faces and self.detector in ("yunet", "haar"):
            problems.append(
                "--mask-faces needs a body or object detector (e.g. --detector "
                "yolox); with a face detector every crop would be all mask"
            )
        if self.mask_dir is not None:
            mask_dir = Path(self.mask_dir).expanduser().resolve()
            if mask_dir.is_relative_to(Path(self.output).expanduser().resolve()):
                problems.append(
                    "--mask-dir must be outside --output, or trainers that read "
                    "every image in the dataset folder would train on the masks"
                )
            if self.training_config or self.training_config_only:
                problems.append(
                    "--training-config writes a OneTrainer config, and OneTrainer "
                    "only reads masks beside the crops; drop --mask-dir or --training-config"
                )
        if self.overwrite and self.resume:
            problems.append("--overwrite and --resume are mutually exclusive")
        if self.caption_only and self.training_config_only:
            problems.append("--caption-only and --training-config-only are mutually exclusive")
        if self.caption_only and self.captioner == "none":
            problems.append("--caption-only needs a --captioner other than 'none'")

        if problems:
            raise ValueError("invalid configuration:\n  - " + "\n  - ".join(problems))

    def _detects_people(self) -> bool:
        if self.detector != "yolox":
            return True  # faces, or a custom detector we cannot see into
        classes = self.detector_opts.get("classes", "person")
        names = classes.split(",") if isinstance(classes, str) else classes
        names = {str(n).strip().lower() for n in names if str(n).strip()}
        return not names or bool(names & {"person", "all"})  # empty means all

    @property
    def writes_masks(self) -> bool:
        return self.mask_faces or self.mask_background or self.mask_clothing

    @property
    def extension(self) -> str:
        return "jpg" if self.format in ("jpg", "jpeg") else self.format

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key, value in out.items():
            if isinstance(value, Path):
                out[key] = str(value)
            elif isinstance(value, tuple):
                out[key] = list(value)
        return out


def _check_template(template: str) -> str:
    """Why a --caption-template would fail, or "" if it is fine.

    Checked up front: otherwise a typo surfaces only in the caption pass,
    after every photo has already been cropped.
    """
    from .captioners.template import TOKENS

    try:
        template.format(**{token: "" for token in TOKENS})
    except KeyError as exc:
        return (f"--caption-template has an unknown token {exc}; "
                f"available: {', '.join(sorted(TOKENS))}")
    except (IndexError, ValueError) as exc:
        return f"--caption-template is not a valid format string: {exc}"
    return ""


#: Fields that are tuples in the dataclass but lists in TOML.
_TUPLE_FIELDS = {"sizes", "exclude", "caption_drop", "fill_color"}


def load_toml(path: Path) -> dict[str, Any]:
    """Read a config file into a plain dict of known keys."""
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    # Accept both a flat file and one nested under [robocrop].
    data = raw.get("robocrop", raw) if isinstance(raw, dict) else {}
    known = {f.name for f in fields(Config)}
    out: dict[str, Any] = {}
    unknown: list[str] = []

    for key, value in data.items():
        name = key.replace("-", "_")
        if name not in known:
            unknown.append(key)
            continue
        if name in ("input", "output", "mask_dir"):
            value = Path(str(value)).expanduser()
        elif name in _TUPLE_FIELDS and isinstance(value, list):
            value = tuple(value)
        out[name] = value

    if unknown:
        raise ValueError(
            f"unknown key(s) in {path}: {', '.join(sorted(unknown))}"
        )
    return out

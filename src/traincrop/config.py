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
        if self.overwrite and self.resume:
            problems.append("--overwrite and --resume are mutually exclusive")
        if self.caption_only and self.training_config_only:
            problems.append("--caption-only and --training-config-only are mutually exclusive")
        if self.caption_only and self.captioner == "none":
            problems.append("--caption-only needs a --captioner other than 'none'")

        if problems:
            raise ValueError("invalid configuration:\n  - " + "\n  - ".join(problems))

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


#: Fields that are tuples in the dataclass but lists in TOML.
_TUPLE_FIELDS = {"sizes", "exclude", "caption_drop", "fill_color"}


def load_toml(path: Path) -> dict[str, Any]:
    """Read a config file into a plain dict of known keys."""
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    # Accept both a flat file and one nested under [traincrop].
    data = raw.get("traincrop", raw) if isinstance(raw, dict) else {}
    known = {f.name for f in fields(Config)}
    out: dict[str, Any] = {}
    unknown: list[str] = []

    for key, value in data.items():
        name = key.replace("-", "_")
        if name not in known:
            unknown.append(key)
            continue
        if name in ("input", "output"):
            value = Path(str(value)).expanduser()
        elif name in _TUPLE_FIELDS and isinstance(value, list):
            value = tuple(value)
        out[name] = value

    if unknown:
        raise ValueError(
            f"unknown key(s) in {path}: {', '.join(sorted(unknown))}"
        )
    return out

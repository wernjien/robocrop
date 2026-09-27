"""Command-line interface."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Sequence

# Must be set before OpenCV is imported anywhere, or its DNN backend chatters
# on stderr about graph engine targets on every detector construction.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")          # MediaPipe / TensorFlow Lite
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # avoids a fork warning

from . import __version__, captioners, detectors  # noqa: E402
from .config import Config, load_toml  # noqa: E402
from .pipeline import Pipeline, TRAINING_CONFIG_NAME  # noqa: E402


class _Formatter(argparse.RawDescriptionHelpFormatter):
    def __init__(self, prog: str) -> None:
        super().__init__(prog, max_help_position=34, width=100)


EPILOG = """
examples:
  # faces (default): recurse ./photos, local VLM captions
  robocrop -i ./photos -o ./dataset

  # one subject: trigger word, tighter framing, JPEG output
  robocrop -i ./photos -o ./dataset --trigger "my subject" --padding 15 --format jpg

  # crop dogs instead of faces
  robocrop -i ./photos -o ./dataset --detector yolox --detector-opt classes=dog

  # crop multiple classes at once
  robocrop -i ./photos -o ./dataset --detector yolox --detector-opt classes=cat,bird

  # crop whole bodies
  robocrop -i ./photos -o ./dataset --detector yolox --detector-opt classes=person

  # learn the body and outfit, not the face: bodies cropped, faces
  # masked out of training (OneTrainer masked training switched on)
  robocrop -i ./photos -o ./dataset --detector yolox --detector-opt classes=person \
      --mask-faces --training-config

  # learn the subject, not the backgrounds the photos were taken against
  robocrop -i ./photos -o ./dataset --mask-background --training-config

  # learn the person, not the outfits they were photographed in
  robocrop -i ./photos -o ./dataset --mask-clothing --training-config

  # see what would happen, without writing anything or loading a model
  robocrop -i ./photos -o ./dataset --dry-run

  # force a single output size and accept smaller objects
  robocrop -i ./photos -o ./dataset --sizes 768 --min-ratio 0.8

  # crop only, no captions (write .txt files later, or never)
  robocrop -i ./photos -o ./dataset --captioner none

  # captions only from measured attributes, no model download
  robocrop -i ./photos -o ./dataset --captioner template

  # caption an already-cropped --output dataset, no re-detection
  robocrop -o ./dataset --caption-only --captioner vlm

  # also emit a OneTrainer config for this dataset
  robocrop -i ./photos -o ./dataset --training-config

  # (re)generate just the OneTrainer config for an existing dataset
  robocrop -o ./dataset --training-config-only

  # loosen or disable the blur check
  robocrop -i ./photos -o ./dataset --min-sharpness 5
  robocrop -i ./photos -o ./dataset --min-sharpness 0

masks:
  --mask-faces, --mask-background and --mask-clothing write NAME-masklabel.png
  beside each crop: a greyscale loss weight per pixel, white where the trainer
  should learn. --mask-faces puts a soft black oval over every face;
  --mask-background drops everything but the person (found with a person
  matting model) to --background-weight; --mask-clothing drops clothing and
  accessories (found with a clothes parser) to --clothing-weight. Any
  combination. The crop itself is left untouched -- covering the face or the
  background would teach the model the cover -- and masked training scores
  each pixel only as much as its mask allows.
  OneTrainer reads these masks with masked training on, which
  --training-config then enables.

sizing rule:
  A crop is produced at the largest configured size whose minimum is met by the
  *detected* box, before padding is added: the detection must reach the
  --min-ratio fraction of that size.

  Detections below the smallest threshold are skipped and listed in
  manifest.json, so nothing is upscaled from an unusable source.

padding:
  --padding is a percentage of the detected box added to *every* side, so the
  crop grows by twice that percentage in total before being resized to the
  chosen output size.

blur check:
  --min-sharpness scores the central portion of each resized crop with a
  Laplacian-variance blur heuristic (higher is sharper) and drops crops that
  fall below it. Every crop's score is recorded in manifest.jsonl either way,
  kept or dropped, so the default can be tuned from real data. 0 disables it.

upscaling:
  A crop smaller than its tier is enlarged either way; --upscale only changes
  the method, from a plain resize to AI upscaling using FSRCNN, a small
  neural network trained to enlarge images (needs opencv-contrib-python; see
  requirements-upscale.txt). A crop that already meets or exceeds its tier is
  always just resized down, --upscale or not.

  The bundled model enlarges 2x per pass, and the most a qualifying crop can
  need is 1/min-ratio times its own size -- so keep --min-ratio above 0.5 to
  stay inside that in one pass. Below it, the remainder past 2x falls back
  to a plain resize, and robocrop warns about it.
"""


def _default(name: str) -> str:
    """Config's actual default for `name`, formatted for --help text.

    Reads it live from the dataclass field instead of hand-typing the value,
    so --help can never drift out of sync with the real default.
    """
    value = Config.__dataclass_fields__[name].default
    if isinstance(value, tuple):
        return ",".join(str(v) for v in value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _default_pct(name: str) -> str:
    """Like `_default`, for a Config fraction shown as a whole percent."""
    value = Config.__dataclass_fields__[name].default * 100
    return str(int(value)) if value.is_integer() else str(value)


def _describe_choices(descriptions: dict[str, str], config_field: str) -> str:
    """Render a choice group's --help text, marking Config's actual default."""
    default = _default(config_field)
    return "; ".join(
        f"{k} ({v}, default)" if k == default else f"{k} ({v})"
        for k, v in descriptions.items()
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="robocrop",
        description=(
            "Find and crop objects (faces, bodies, animals, or any COCO class) "
            "square at 512/768/1024, and write a caption beside each crop, "
            "ready for training an image model."
        ),
        epilog=EPILOG,
        formatter_class=_Formatter,
    )
    # SUPPRESS everywhere: the namespace then holds only what the user actually
    # typed, which is what lets flags override a --config file cleanly.
    def add(group, *names, **kwargs):
        kwargs.setdefault("default", argparse.SUPPRESS)
        group.add_argument(*names, **kwargs)

    io = parser.add_argument_group("input and output")
    add(io, "-i", "--input", type=Path, metavar="DIR",
        help="directory to scan recursively for images")
    add(io, "-o", "--output", type=Path, metavar="DIR",
        help=f"directory to write crops and captions into (default: ./{_default('output')})")
    add(io, "--config", type=Path, metavar="FILE",
        help="TOML file of settings (auto-detected from ./robocrop.toml if it exists); "
             "command-line flags win over it")
    add(io, "--exclude", action="append", metavar="GLOB",
        help="skip paths matching this glob (repeatable)")
    add(io, "--follow-symlinks", action="store_true",
        help="descend into symlinked directories")
    add(io, "--limit", type=int, metavar="N",
        help="stop after N source images (for trying settings out)")

    det = parser.add_argument_group("detection")
    add(det, "-d", "--detector", choices=detectors.available(), metavar="NAME",
        help="detector to use: "
             + _describe_choices(detectors.DESCRIPTIONS, "detector"))
    add(det, "--detector-opt", action="append", metavar="K=V",
        help="extra option for the detector, e.g. classes=person (repeatable)")
    add(det, "--min-score", type=float, metavar="F",
        help=f"discard detections below this confidence, 0-1 (default: {_default('min_score')})")
    add(det, "--multi-face", dest="multi", choices=("all", "largest", "skip"),
        help="images with several detections: crop all of them, only the "
             f"largest, or skip the image (default: {_default('multi')})")
    add(det, "--max-per-image", type=int, metavar="N",
        help="cap crops taken from one image (0 = no cap)")

    crop = parser.add_argument_group("crop geometry")
    add(crop, "-p", "--padding", type=float, metavar="PCT",
        help=f"percent of the detection added to every side (default: {_default_pct('padding')})")
    add(crop, "-s", "--sizes", metavar="LIST",
        help=f"candidate square sizes, largest achievable wins (default: {_default('sizes')})")
    add(crop, "--min-ratio", type=float, metavar="F",
        help="fraction of a size the detection needs to reach to earn it "
             f"(default: {_default('min_ratio')})")
    add(crop, "--base-mode", choices=("max", "mean", "width", "height", "diag"),
        help=f"how the detection box becomes a square side (default: {_default('base_mode')})")
    add(crop, "--offset-y", type=float, metavar="F",
        help="shift the crop vertically by this fraction of the box; negative "
             "moves up. Defaults to the detector's own correction, to keep "
             "hair and forehead in frame; pass 0 to disable")
    add(crop, "--offset-x", type=float, metavar="F",
        help="shift the crop horizontally by this fraction of the box")
    add(crop, "--edge", choices=("shift", "extend", "skip"),
        help="when the padded crop runs off the image: slide it back in, "
             f"extend the canvas, or skip the detection (default: {_default('edge')})")
    add(crop, "--fill", choices=("edge", "blur", "reflect", "color"),
        help=f"how to fill invented area when --edge extend (default: {_default('fill')})")
    add(crop, "--fill-color", metavar="R,G,B",
        help=f"fill colour for --fill color (default: {_default('fill_color')})")
    add(crop, "--min-sharpness", type=float, metavar="F",
        help="drop crops below this Laplacian-variance sharpness score, "
             "measured on the central portion of the resized crop; "
             f"0 disables the check (default: {_default('min_sharpness')})")
    add(crop, "--upscale", action="store_true",
        help="enlarge undersized crops with AI upscaling instead of a "
             "plain resize; needs opencv-contrib-python. Crops that already "
             "meet or exceed their tier are unaffected")

    mask = parser.add_argument_group("masks")
    add(mask, "--mask-faces", action="store_true",
        help="write a NAME-masklabel.png beside each crop that masks every "
             "face out of training, to learn a body, outfit or style but not "
             "the face; needs a body detector such as yolox")
    add(mask, "--mask-margin", type=float, metavar="PCT",
        help="percent of the face box added to every side of its mask, to "
             f"cover hair, ears and jaw (default: {_default_pct('mask_margin')})")
    add(mask, "--mask-missing", choices=("skip", "keep"),
        help="crops where no face was found (back of the head, strong "
             "profile, face out of frame): skip them, or keep them with "
             f"nothing masked (default: {_default('mask_missing')})")
    add(mask, "--mask-min-score", type=float, metavar="F",
        help="face confidence needed to mask it; lower misses fewer faces "
             f"(default: {_default('mask_min_score')})")
    add(mask, "--mask-background", action="store_true",
        help="write a NAME-masklabel.png beside each crop that weights the "
             "background down, so the model learns the person rather than "
             "where the photos were taken; combines with --mask-faces")
    add(mask, "--background-weight", type=float, metavar="F",
        help="loss weight of the background, 0-1; a little keeps the model "
             f"from drifting there (default: {_default('background_weight')})")
    add(mask, "--mask-clothing", action="store_true",
        help="write a NAME-masklabel.png beside each crop that weights "
             "clothing and accessories down, so the model learns the person "
             "rather than their outfits; combines with the other masks")
    add(mask, "--clothing-weight", type=float, metavar="F",
        help="loss weight of clothing, 0-1; raise it a little if the model "
             f"drifts on clothing (default: {_default('clothing_weight')})")

    out = parser.add_argument_group("output files")
    add(out, "--prefix", metavar="STR", help="filename prefix before the number")
    add(out, "--start-index", type=int, metavar="N",
        help=f"first output number (default: {_default('start_index')})")
    add(out, "--digits", type=int, metavar="N",
        help=f"zero-padded width of the number (default: {_default('digits')})")
    add(out, "-f", "--format", choices=("png", "jpg", "webp"),
        help=f"output image format (default: {_default('format')})")
    add(out, "--quality", type=int, metavar="N",
        help=f"quality for jpg/webp, 1-100 (default: {_default('quality')})")
    add(out, "--per-size-dirs", action="store_true",
        help="write into 512/, 768/, 1024/ subdirectories")

    cap = parser.add_argument_group("captions")
    add(cap, "-c", "--captioner", choices=captioners.available(), metavar="NAME",
        help="caption backend: "
             + _describe_choices(captioners.DESCRIPTIONS, "captioner"))
    add(cap, "--caption-model", metavar="NAME",
        help="VLM preset or Hugging Face id (see --list-models)")
    add(cap, "--caption-prompt", metavar="TEXT",
        help="instruction given to the VLM; overrides the built-in prompt")
    add(cap, "--caption-device", choices=("auto", "mps", "cuda", "cpu"),
        help=f"where to run the VLM (default: {_default('caption_device')})")
    add(cap, "--caption-batch", type=int, metavar="N",
        help=f"crops per VLM batch; lower it if memory is tight (default: {_default('caption_batch')})")
    add(cap, "--caption-tokens", type=int, metavar="N",
        help=f"max new tokens per caption (default: {_default('caption_tokens')})")
    add(cap, "--caption-template", metavar="STR",
        help="format string for --captioner template, e.g. "
             "'{shot}, {pose}, {light}, {tone}'")
    add(cap, "-t", "--trigger", metavar="WORD",
        help="trigger word placed first in every caption, e.g. 'my subject'")
    add(cap, "--caption-prefix", metavar="STR", help="text after the trigger")
    add(cap, "--caption-suffix", metavar="STR", help="text at the end")
    add(cap, "--caption-drop", action="append", metavar="REGEX",
        help="remove text matching this regex from captions (repeatable)")
    add(cap, "--caption-max-chars", type=int, metavar="N",
        help="trim captions to roughly this length at a comma (0 = no limit)")
    add(cap, "--caption-only", action="store_true",
        help="caption an existing --output dataset instead of running "
             "detection/crop: rebuilds regions from manifest.jsonl and "
             "(re)writes .txt files; with --resume, only crops missing a "
             "caption file are captioned")

    tc = parser.add_argument_group("training config")
    add(tc, "--training-config", action="store_true",
        help="write training_config.json into the output dir: a copy of "
             "robocrop.onetrainer.example.json with 'resolution' set to the "
             "smallest tier produced and 'epochs' computed from the crop "
             "count (targets ~4000 steps)")
    add(tc, "--training-config-only", action="store_true",
        help="only (re)generate training_config.json for an existing "
             "--output dataset, reading manifest.jsonl; no cropping or "
             "captioning happens")

    run = parser.add_argument_group("run")
    add(run, "-j", "--workers", type=int, metavar="N",
        help="threads for the detect/crop pass (0 = auto)")
    add(run, "-n", "--dry-run", action="store_true",
        help="report what would be produced; writes nothing and loads no "
             "caption model (detection models still run, to report real results)")
    add(run, "--resume", action="store_true",
        help="continue a previous run, skipping sources already in the manifest")
    add(run, "--overwrite", action="store_true",
        help="replace an existing output directory's files")
    add(run, "-q", "--quiet", action="store_true", help="only report errors")
    add(run, "-v", "--verbose", action="store_true", help="log every crop")
    add(run, "--list-models", action="store_true",
        help="show the caption model presets and exit")
    parser.add_argument("--version", action="version", version=f"robocrop {__version__}")
    return parser


def _parse_sizes(raw: str) -> tuple[int, ...]:
    try:
        values = sorted({int(part) for part in raw.replace(" ", "").split(",") if part})
    except ValueError:
        raise SystemExit(f"--sizes must be a comma-separated list of integers, got {raw!r}")
    if not values:
        raise SystemExit("--sizes needs at least one value")
    return tuple(values)


def _parse_color(raw: str) -> tuple[int, int, int]:
    parts = raw.replace(" ", "").split(",")
    if len(parts) != 3:
        raise SystemExit(f"--fill-color needs three values, got {raw!r}")
    try:
        rgb = tuple(max(0, min(255, int(p))) for p in parts)
    except ValueError:
        raise SystemExit(f"--fill-color must be integers, got {raw!r}")
    return rgb  # type: ignore[return-value]


def _parse_detector_opts(pairs: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--detector-opt needs KEY=VALUE, got {pair!r}")
        key, _, value = pair.partition("=")
        out[key.strip()] = _coerce(value.strip())
    return out


def _coerce(value: str) -> Any:
    low = value.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


#: Settings typed as percentages but held in Config as fractions.
_PERCENT_FIELDS = ("padding", "mask_margin")


def build_config(argv: Sequence[str] | None = None) -> Config:
    parser = build_parser()
    args = vars(parser.parse_args(argv))

    if args.pop("list_models", False):
        _print_models()
        raise SystemExit(0)

    settings: dict[str, Any] = {}
    config_path = args.pop("config", None)
    if config_path is None:
        # Auto-detect robocrop.toml in current directory if --config not passed
        default_config = Path("robocrop.toml")
        if default_config.is_file():
            config_path = default_config

    if config_path is not None:
        path = Path(config_path).expanduser()
        if not path.is_file():
            raise SystemExit(f"config file not found: {path}")
        try:
            settings.update(load_toml(path))
        except (ValueError, OSError) as exc:
            raise SystemExit(f"could not read {path}: {exc}")

    # Command-line values land on top of the file.
    if "sizes" in args:
        args["sizes"] = _parse_sizes(args["sizes"])
    if "fill_color" in args:
        args["fill_color"] = _parse_color(args["fill_color"])
    if "exclude" in args:
        args["exclude"] = tuple(args["exclude"])
    if "caption_drop" in args:
        args["caption_drop"] = tuple(args["caption_drop"])
    if "detector_opt" in args:
        args["detector_opts"] = _parse_detector_opts(args.pop("detector_opt"))
    for pct in _PERCENT_FIELDS:
        if pct in args:
            args[pct] = args[pct] / 100.0   # the flag is a percentage
    settings.update(args)

    # A TOML file gives these as percentages too; normalise them the same
    # way, but only when the flag did not already do so.
    if config_path is not None:
        for pct in _PERCENT_FIELDS:
            if pct in settings and pct not in args:
                settings[pct] = float(settings[pct]) / 100.0

    known = {f.name for f in fields(Config)}
    config = Config(**{k: v for k, v in settings.items() if k in known})
    config.input = Path(config.input).expanduser()
    config.output = Path(config.output).expanduser()

    try:
        config.validate()
    except ValueError as exc:
        raise SystemExit(str(exc))
    if not (config.caption_only or config.training_config_only):
        if not config.input.is_dir():
            raise SystemExit(f"input directory not found: {config.input}")
        if config.input.resolve() == config.output.resolve():
            # Crops would land among the photos and be cropped again next run.
            # (An output *inside* the input is fine: the scan skips it.)
            raise SystemExit("--output must be a different directory from --input")
    return config


def _print_models() -> None:
    from .captioners.vlm import PRESETS

    print("caption model presets (--caption-model):\n")
    width = max(len(k) for k in PRESETS)
    for name, (model_id, size, note) in PRESETS.items():
        print(f"  {name:<{width}}  {size:>8}  {note}")
        print(f"  {'':<{width}}            {model_id}")
    print("\nAny Hugging Face image-text-to-text model id also works.")
    print("Models download once, then run fully offline.")


def _reporter(config: Config):
    def emit(level: str, message: str) -> None:
        if config.quiet and level not in ("warn", "error"):
            return
        stream = sys.stderr if level in ("warn", "error") else sys.stdout
        prefix = {"warn": "warning: ", "error": "error: "}.get(level, "")
        print(f"{prefix}{message}", file=stream, flush=True)

    return emit


def main(argv: Sequence[str] | None = None) -> int:
    config = build_config(argv)
    emit = _reporter(config)

    if config.dry_run:
        emit("info", "dry run: nothing will be written")

    pipeline = Pipeline(config, on_event=emit)
    try:
        stats = pipeline.run()
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted; progress so far is in the manifest "
              "(re-run with --resume)", file=sys.stderr)
        return 130
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _summarise(config, stats, emit)
    return 0 if stats.written or not stats.scanned else 1


def _summarise(config: Config, stats, emit) -> None:
    if config.quiet:
        return

    if config.training_config_only:
        _summarise_training_config_only(config, stats, emit)
        return
    if config.caption_only:
        _summarise_caption_only(config, stats, emit)
        return

    tiers = ", ".join(f"{size}: {count}" for size, count in sorted(stats.by_tier.items()))
    lines = [
        "",
        f"scanned          {stats.scanned} image(s) in {stats.seconds:.1f}s",
        f"detections       {stats.detections} across {stats.with_detections} image(s)",
        f"crops written    {stats.written}" + (f"  ({tiers})" if tiers else ""),
    ]
    if config.captioner != "none":
        lines.append(f"captions written {stats.captioned}")
    if stats.skipped_no_detection:
        lines.append(f"skipped          {stats.skipped_no_detection} with no detection")
    if stats.skipped_too_small:
        lines.append(f"skipped          {stats.skipped_too_small} detection(s) below "
                     f"{config.min_ratio:.0%} of {min(config.sizes)}px")
    if stats.skipped_blurry:
        lines.append(f"skipped          {stats.skipped_blurry} blurry crop(s) below "
                     f"{config.min_sharpness:.1f} sharpness")
    if stats.skipped_multi:
        lines.append(f"skipped          {stats.skipped_multi} multi-face image(s)")
    if config.mask_faces:
        lines.append(f"faces masked     in {stats.masked} crop(s)")
    if stats.skipped_no_face:
        lines.append(f"skipped          {stats.skipped_no_face} crop(s) with no face "
                     f"found to mask")
    if config.mask_background:
        lines.append(f"bg masked        in {stats.background_masked} crop(s)")
    if stats.background_box:
        lines.append(f"masked by box    {stats.background_box} crop(s) where no person "
                     f"outline was found")
    if config.mask_clothing:
        lines.append(f"clothing masked  in {stats.clothing_masked} crop(s)")
    if stats.unmasked_kept:
        lines.append(f"kept unmasked    {stats.unmasked_kept} crop(s) with no face found "
                     f"-- check these for a visible face")
    if stats.skipped_existing:
        lines.append(f"skipped          {stats.skipped_existing} already done (resume)")
    if stats.errors:
        lines.append(f"errors           {stats.errors} (see manifest.json)")
    if config.training_config and (config.output / TRAINING_CONFIG_NAME).exists():
        # Checked on disk: a --resume with nothing new still writes it.
        lines.append(f"training config  {config.output / TRAINING_CONFIG_NAME}")
    if not config.dry_run and stats.written:
        lines.append(f"\noutput           {config.output}")
    emit("info", "\n".join(lines))


def _summarise_caption_only(config: Config, stats, emit) -> None:
    lines = [
        "",
        f"crops loaded     {stats.written}",
        f"captions written {stats.captioned}",
    ]
    if stats.skipped_existing:
        lines.append(f"skipped          {stats.skipped_existing} already captioned (resume)")
    if stats.errors:
        lines.append(f"errors           {stats.errors}")
    emit("info", "\n".join(lines))


def _summarise_training_config_only(config: Config, stats, emit) -> None:
    tiers = ", ".join(f"{size}: {count}" for size, count in sorted(stats.by_tier.items()))
    lines = [
        "",
        f"dataset          {stats.written} crop(s)" + (f"  ({tiers})" if tiers else ""),
    ]
    if not config.dry_run and stats.written:
        lines.append(f"training config  {config.output / TRAINING_CONFIG_NAME}")
    emit("info", "\n".join(lines))

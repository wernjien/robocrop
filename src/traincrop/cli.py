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
  # faces (default): recurse ./photos, 20% padding, local VLM captions
  traincrop -i ./photos -o ./dataset

  # a character LoRA: trigger word, tighter framing, JPEG output
  traincrop -i ./photos -o ./dataset --trigger "my subject" --padding 15 --format jpg

  # crop dogs instead of faces
  traincrop -i ./photos -o ./dataset --detector yolox --detector-opt classes=dog

  # crop multiple classes at once
  traincrop -i ./photos -o ./dataset --detector yolox --detector-opt classes=cat,bird

  # crop whole bodies
  traincrop -i ./photos -o ./dataset --detector yolox --detector-opt classes=person

  # see what would happen, without writing anything or loading a model
  traincrop -i ./photos -o ./dataset --dry-run

  # force a single output size and accept smaller objects
  traincrop -i ./photos -o ./dataset --sizes 768 --min-ratio 0.8

  # crop only, no captions (write .txt files later, or never)
  traincrop -i ./photos -o ./dataset --captioner none

  # captions only from measured attributes, no model download
  traincrop -i ./photos -o ./dataset --captioner template

  # caption an already-cropped --output dataset, no re-detection
  traincrop -o ./dataset --caption-only --captioner vlm

  # also emit a OneTrainer config for this dataset
  traincrop -i ./photos -o ./dataset --training-config

  # (re)generate just the OneTrainer config for an existing dataset
  traincrop -o ./dataset --training-config-only

  # loosen or disable the blur check
  traincrop -i ./photos -o ./dataset --min-sharpness 5
  traincrop -i ./photos -o ./dataset --min-sharpness 0

sizing rule:
  A crop is produced at the largest configured size whose minimum is met by the
  *detected* box, before padding is added:

      1024  needs a detection >= 819 px   (80% of 1024)
       768  needs a detection >= 614 px
       512  needs a detection >= 410 px

  Detections below the smallest threshold are skipped and listed in
  manifest.json, so nothing is ever upscaled from an unusable source.

padding:
  --padding is a percentage of the detected box added to *every* side.
  --padding 20 turns a 400 px detection into a 560 px crop (400 + 2x80),
  which is then resized to the chosen size.

blur check:
  --min-sharpness scores the central portion of each resized crop with a
  Laplacian-variance blur heuristic (higher is sharper) and drops crops that
  fall below it. Every crop's score is recorded in manifest.jsonl either way,
  kept or dropped, so the default can be tuned from real data. 0 disables it.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="traincrop",
        description=(
            "Find and crop objects (faces, bodies, animals, or any COCO class) "
            "square at 512/768/1024, and write a caption beside each crop for LoRA training."
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
        help="directory to write crops and captions into (default: ./dataset)")
    add(io, "--config", type=Path, metavar="FILE",
        help="TOML file of settings (auto-detected from ./traincrop.toml if it exists); "
             "command-line flags win over it")
    add(io, "--exclude", action="append", metavar="GLOB",
        help="skip paths matching this glob (repeatable)")
    add(io, "--follow-symlinks", action="store_true",
        help="descend into symlinked directories")
    add(io, "--limit", type=int, metavar="N",
        help="stop after N source images (for trying settings out)")

    det = parser.add_argument_group("detection")
    add(det, "-d", "--detector", choices=detectors.available(), metavar="NAME",
        help="detector to use: " + "; ".join(
            f"{k} ({v})" for k, v in detectors.DESCRIPTIONS.items()))
    add(det, "--detector-opt", action="append", metavar="K=V",
        help="extra option for the detector, e.g. classes=person (repeatable)")
    add(det, "--min-score", type=float, metavar="F",
        help="discard detections below this confidence, 0-1 (default: 0.8)")
    add(det, "--multi-face", dest="multi", choices=("all", "largest", "skip"),
        help="images with several detections: crop all of them (default), "
             "only the largest, or skip the image")
    add(det, "--max-per-image", type=int, metavar="N",
        help="cap crops taken from one image (0 = no cap)")

    crop = parser.add_argument_group("crop geometry")
    add(crop, "-p", "--padding", type=float, metavar="PCT",
        help="percent of the detection added to every side (default: 20)")
    add(crop, "-s", "--sizes", metavar="LIST",
        help="candidate square sizes, largest achievable wins "
             "(default: 512,768,1024)")
    add(crop, "--min-ratio", type=float, metavar="F",
        help="fraction of a size the detection must reach to earn it "
             "(default: 0.8)")
    add(crop, "--base-mode", choices=("max", "mean", "width", "height", "diag"),
        help="how the detection box becomes a square side (default: max)")
    add(crop, "--offset-y", type=float, metavar="F",
        help="shift the crop vertically by this fraction of the box; negative "
             "moves up. Defaults to the detector's own correction (-0.08 for "
             "face detectors, to keep hair and forehead in frame); pass 0 to "
             "disable")
    add(crop, "--offset-x", type=float, metavar="F",
        help="shift the crop horizontally by this fraction of the box")
    add(crop, "--edge", choices=("shift", "extend", "skip"),
        help="when the padded crop runs off the image: slide it back in "
             "(default), extend the canvas, or skip the detection")
    add(crop, "--fill", choices=("edge", "blur", "reflect", "color"),
        help="how to fill invented area when --edge extend (default: blur)")
    add(crop, "--fill-color", metavar="R,G,B",
        help="fill colour for --fill color (default: 0,0,0)")
    add(crop, "--min-sharpness", type=float, metavar="F",
        help="drop crops below this Laplacian-variance sharpness score, "
             "measured on the central portion of the resized crop; "
             "0 disables the check (default: 10)")

    out = parser.add_argument_group("output files")
    add(out, "--prefix", metavar="STR", help="filename prefix before the number")
    add(out, "--start-index", type=int, metavar="N",
        help="first output number (default: 1)")
    add(out, "--digits", type=int, metavar="N",
        help="zero-padded width of the number (default: 4 -> 0001.png)")
    add(out, "-f", "--format", choices=("png", "jpg", "webp"),
        help="output image format (default: png)")
    add(out, "--quality", type=int, metavar="N",
        help="quality for jpg/webp, 1-100 (default: 95)")
    add(out, "--per-size-dirs", action="store_true",
        help="write into 512/, 768/, 1024/ subdirectories")

    cap = parser.add_argument_group("captions")
    add(cap, "-c", "--captioner", choices=captioners.available(), metavar="NAME",
        help="caption backend: " + "; ".join(
            f"{k} ({v})" for k, v in captioners.DESCRIPTIONS.items()))
    add(cap, "--caption-model", metavar="NAME",
        help="VLM preset or Hugging Face id (see --list-models)")
    add(cap, "--caption-prompt", metavar="TEXT",
        help="instruction given to the VLM; overrides the built-in prompt")
    add(cap, "--caption-device", choices=("auto", "mps", "cuda", "cpu"),
        help="where to run the VLM (default: auto)")
    add(cap, "--caption-batch", type=int, metavar="N",
        help="crops per VLM batch; lower it if memory is tight (default: 4)")
    add(cap, "--caption-tokens", type=int, metavar="N",
        help="max new tokens per caption (default: 96)")
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
             "traincrop.onetrainer.example.json with 'resolution' set to the "
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
        help="report what would be produced; writes nothing, loads no model")
    add(run, "--resume", action="store_true",
        help="continue a previous run, skipping sources already in the manifest")
    add(run, "--overwrite", action="store_true",
        help="replace an existing output directory's files")
    add(run, "-q", "--quiet", action="store_true", help="only report errors")
    add(run, "-v", "--verbose", action="store_true", help="log every crop")
    add(run, "--list-models", action="store_true",
        help="show the caption model presets and exit")
    parser.add_argument("--version", action="version", version=f"traincrop {__version__}")
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


def build_config(argv: Sequence[str] | None = None) -> Config:
    parser = build_parser()
    args = vars(parser.parse_args(argv))

    if args.pop("list_models", False):
        _print_models()
        raise SystemExit(0)

    settings: dict[str, Any] = {}
    config_path = args.pop("config", None)
    if config_path is None:
        # Auto-detect traincrop.toml in current directory if --config not passed
        default_config = Path("traincrop.toml")
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
    if "padding" in args:
        args["padding"] = args["padding"] / 100.0   # the flag is a percentage
    settings.update(args)

    # A TOML file may also give padding as a percentage; normalise it the same
    # way, but only when the flag did not already do so.
    if config_path is not None and "padding" in settings and "padding" not in args:
        settings["padding"] = float(settings["padding"]) / 100.0

    known = {f.name for f in fields(Config)}
    config = Config(**{k: v for k, v in settings.items() if k in known})
    config.input = Path(config.input).expanduser()
    config.output = Path(config.output).expanduser()

    try:
        config.validate()
    except ValueError as exc:
        raise SystemExit(str(exc))
    if not (config.caption_only or config.training_config_only) and not config.input.is_dir():
        raise SystemExit(f"input directory not found: {config.input}")
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
    if stats.skipped_existing:
        lines.append(f"skipped          {stats.skipped_existing} already done (resume)")
    if stats.errors:
        lines.append(f"errors           {stats.errors} (see manifest.json)")
    if config.training_config and not config.dry_run and stats.written:
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

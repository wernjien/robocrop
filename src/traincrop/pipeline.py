"""The run itself: scan, detect, crop, then caption.

Two passes, deliberately:

1. **Crop.** Threaded over images. Decoding and detection are the slow part and
   both release the GIL, so this scales with cores.
2. **Caption.** Single-threaded and batched, re-reading the crops from disk.
   A VLM is loaded once and fed batches; re-reading rather than holding every
   crop in memory keeps a 20,000-photo run flat, and makes the pass resumable
   on its own.
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from PIL import Image

from . import captioners, detectors, images
from .captioners.base import CaptionRequest
from .config import Config
from .detectors.base import Region
from .geometry import CropRejected, Rect, plan_crop

MANIFEST_NAME = "manifest.jsonl"
SUMMARY_NAME = "manifest.json"
TRAINING_CONFIG_NAME = "training_config.json"
ONETRAINER_TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "traincrop.onetrainer.example.json"


@dataclass
class CropRecord:
    """One produced crop, as written to the manifest."""

    index: int
    source: str
    image_file: str
    caption_file: str | None
    tier: int
    label: str
    score: float
    base_side: float
    fitted_side: float
    scale: float
    sharpness: float
    padding: float
    offset_y: float
    crop: list[float]
    clamped: bool
    extended: bool
    face_index: int
    faces_in_image: int
    landmarks: dict[str, list[float]] = field(default_factory=dict)
    """Detector landmarks in source pixels, when the backend supplies them.
    Carried through so the caption pass can read head pose, including on a
    resumed run where the detector is never re-run."""
    caption: str = ""


@dataclass
class SkipRecord:
    source: str
    reason: str
    detail: str = ""
    sharpness: float | None = None


@dataclass
class Stats:
    scanned: int = 0
    with_detections: int = 0
    detections: int = 0
    written: int = 0
    captioned: int = 0
    skipped_no_detection: int = 0
    skipped_too_small: int = 0
    skipped_blurry: int = 0
    skipped_multi: int = 0
    skipped_existing: int = 0
    errors: int = 0
    by_tier: dict[int, int] = field(default_factory=dict)
    seconds: float = 0.0


@dataclass
class _ImageResult:
    """What one worker produces for one source image."""

    source: Path
    crops: list[tuple[Region, Any, Image.Image, float]] = field(default_factory=list)
    skips: list[SkipRecord] = field(default_factory=list)
    detections: int = 0
    offset_y: float = 0.0
    error: str = ""


class Pipeline:
    def __init__(self, config: Config, *, on_event: Callable[[str, str], None] | None = None):
        self.cfg = config
        self._emit = on_event or (lambda level, message: None)
        self._local = threading.local()
        self.stats = Stats()
        self.records: list[CropRecord] = []
        self.skips: list[SkipRecord] = []
        self._carried: list[str] = []
        """Verbatim manifest lines from the run being resumed. The caption
        pass rewrites the manifest, and this run only holds its own records,
        so the earlier rows have to be carried across explicitly."""

    # -- detector per thread ---------------------------------------------
    def _detector(self):
        """One detector per worker thread.

        The OpenCV DNN and MediaPipe graph objects are stateful and not safe to
        call concurrently, and they are cheap to build, so each thread gets its
        own rather than sharing one behind a lock.
        """
        existing = getattr(self._local, "detector", None)
        if existing is None:
            existing = detectors.create(
                self.cfg.detector,
                min_score=self.cfg.min_score,
                quiet=self.cfg.quiet,
                **self.cfg.detector_opts,
            )
            self._local.detector = existing
        return existing

    # -- entry point ------------------------------------------------------
    def run(self) -> Stats:
        started = time.monotonic()
        cfg = self.cfg

        paths = list(images.iter_images(
            cfg.input,
            exclude=cfg.exclude,
            follow_symlinks=cfg.follow_symlinks,
        ))
        if cfg.limit:
            paths = paths[: cfg.limit]
        self._emit("info", f"found {len(paths)} image(s) under {cfg.input}")
        if not paths:
            self.stats.seconds = time.monotonic() - started
            return self.stats

        if not cfg.dry_run:
            cfg.output.mkdir(parents=True, exist_ok=True)

        done_sources, next_index = self._resume_state()
        if done_sources:
            self._emit("info", f"resuming: {len(done_sources)} source(s) already done")
            before = len(paths)
            paths = [p for p in paths if str(p) not in done_sources]
            self.stats.skipped_existing = before - len(paths)

        self._crop_pass(paths, next_index)

        if cfg.captioner != "none" and self.records and not cfg.dry_run:
            self._caption_pass()

        if not cfg.dry_run:
            self._write_summary()
            if cfg.training_config:
                self._write_training_config()

        self.stats.seconds = time.monotonic() - started
        return self.stats

    # -- pass 1 -----------------------------------------------------------
    def _crop_pass(self, paths: Sequence[Path], next_index: int) -> None:
        cfg = self.cfg
        workers = cfg.workers or _default_workers()
        manifest = None if cfg.dry_run else (cfg.output / MANIFEST_NAME).open("a", encoding="utf-8")

        try:
            for result in _ordered_map(self._process_image, paths, workers):
                self.stats.scanned += 1
                if result.error:
                    self.stats.errors += 1
                    self.skips.append(SkipRecord(str(result.source), "error", result.error))
                    self._emit("warn", f"{result.source}: {result.error}")
                    continue

                self.stats.detections += result.detections
                if result.detections:
                    self.stats.with_detections += 1
                self.skips.extend(result.skips)
                for skip in result.skips:
                    self._tally_skip(skip.reason)

                total = len(result.crops)
                for face_index, (region, plan, crop, sharpness) in enumerate(result.crops):
                    record = CropRecord(
                        index=next_index,
                        source=str(result.source),
                        image_file=self._image_name(next_index, plan.tier),
                        caption_file=(
                            None if cfg.captioner == "none"
                            else self._caption_name(next_index, plan.tier)
                        ),
                        tier=plan.tier,
                        label=region.label,
                        score=round(region.score, 4),
                        base_side=round(plan.base_side, 1),
                        fitted_side=round(plan.fitted_side, 1),
                        scale=round(plan.scale, 4),
                        sharpness=sharpness,
                        padding=cfg.padding,
                        offset_y=result.offset_y,
                        crop=[round(v, 1) for v in (plan.rect.x, plan.rect.y, plan.rect.w, plan.rect.h)],
                        clamped=plan.clamped,
                        extended=plan.extends,
                        face_index=face_index,
                        faces_in_image=total,
                        landmarks={
                            name: [round(x, 1), round(y, 1)]
                            for name, (x, y) in region.landmarks.items()
                        },
                    )

                    if not cfg.dry_run:
                        self._save(crop, record)
                        manifest.write(json.dumps(asdict(record)) + "\n")
                        manifest.flush()

                    self.records.append(record)
                    self.stats.written += 1
                    self.stats.by_tier[plan.tier] = self.stats.by_tier.get(plan.tier, 0) + 1
                    next_index += 1

                    if cfg.verbose:
                        self._emit(
                            "info",
                            f"  {record.image_file} <- {result.source.name} "
                            f"[{region.label} {region.score:.2f}] "
                            f"base={plan.base_side:.0f}px tier={plan.tier}",
                        )
                    crop.close()

                if not cfg.quiet and not cfg.verbose and self.stats.scanned % 25 == 0:
                    self._emit(
                        "progress",
                        f"  {self.stats.scanned}/{len(paths)} scanned, "
                        f"{self.stats.written} crops",
                    )
        finally:
            if manifest is not None:
                manifest.close()
            self._close_detectors()

    def _process_image(self, path: Path) -> _ImageResult:
        cfg = self.cfg
        result = _ImageResult(source=path)
        try:
            image = images.load_image(path)
        except Exception as exc:  # noqa: BLE001
            result.error = f"could not read image: {exc}"
            return result

        detector = self._detector()
        # An explicit --offset-y wins; otherwise the detector says how its own
        # boxes need recentring.
        offset_y = (
            cfg.offset_y if cfg.offset_y is not None
            else detector.recommended_offset_y
        )
        result.offset_y = offset_y
        try:
            regions = detector.detect(images.to_bgr(image))
            result.detections = len(regions)

            if not regions:
                result.skips.append(SkipRecord(str(path), "no_detection"))
                return result

            if len(regions) > 1 and cfg.multi == "skip":
                result.skips.append(
                    SkipRecord(str(path), "multi", f"{len(regions)} detections")
                )
                return result
            if cfg.multi == "largest":
                regions = regions[:1]  # detectors return largest-first
            if cfg.max_per_image:
                regions = regions[: cfg.max_per_image]

            for region in regions:
                try:
                    plan = plan_crop(
                        region.rect,
                        image.width,
                        image.height,
                        padding=cfg.padding,
                        sizes=cfg.sizes,
                        min_ratio=cfg.min_ratio,
                        base_mode=cfg.base_mode,
                        offset_x=cfg.offset_x,
                        offset_y=offset_y,
                        edge=cfg.edge,
                    )
                except CropRejected as exc:
                    result.skips.append(SkipRecord(str(path), "too_small", exc.reason))
                    continue

                crop = images.extract(
                    image, plan, fill=cfg.fill, fill_color=cfg.fill_color
                )

                inner_fraction = max(0.5, min(1.0, 1.0 / (1.0 + 2.0 * cfg.padding)))
                sharpness = round(images.measure_sharpness(crop, inner_fraction=inner_fraction), 1)

                if cfg.min_sharpness > 0 and sharpness < cfg.min_sharpness:
                    result.skips.append(SkipRecord(
                        str(path), "blurry",
                        f"sharpness {sharpness:.1f} below {cfg.min_sharpness:.1f}",
                        sharpness=sharpness,
                    ))
                    crop.close()
                    continue

                result.crops.append((region, plan, crop, sharpness))
        except Exception as exc:  # noqa: BLE001
            result.error = f"{type(exc).__name__}: {exc}"
        finally:
            image.close()
        return result

    # -- pass 2 -----------------------------------------------------------
    def _caption_pass(self) -> None:
        cfg = self.cfg
        pending = [r for r in self.records if r.caption_file]
        if not pending:
            return

        self._emit("info", f"captioning {len(pending)} crop(s) with '{cfg.captioner}'")
        try:
            captioner = captioners.create(cfg.captioner, **self._caption_kwargs())
        except Exception as exc:  # noqa: BLE001
            self._emit("warn", f"captioner unavailable ({exc}); crops were still written")
            self.stats.errors += 1
            return

        try:
            batch = max(1, cfg.caption_batch)
            for start in range(0, len(pending), batch):
                chunk = pending[start : start + batch]
                requests, live = [], []
                for record in chunk:
                    path = cfg.output / record.image_file
                    try:
                        image = Image.open(path)
                        image.load()
                    except Exception as exc:  # noqa: BLE001
                        self._emit("warn", f"could not reopen {path.name}: {exc}")
                        self.stats.errors += 1
                        continue
                    requests.append(
                        CaptionRequest(
                            image=image,
                            source_path=record.source,
                            region=self._region_for(record),
                            tier=record.tier,
                            index=record.index,
                        )
                    )
                    live.append((record, image))

                if not requests:
                    continue

                texts = captioner.caption_batch(requests)
                for (record, image), text in zip(live, texts):
                    record.caption = text
                    if text:
                        (cfg.output / record.caption_file).write_text(
                            text + "\n", encoding="utf-8"
                        )
                        self.stats.captioned += 1
                    image.close()

                if not cfg.quiet:
                    done = min(start + batch, len(pending))
                    self._emit("progress", f"  captioned {done}/{len(pending)}")
        finally:
            captioner.close()

        self._rewrite_manifest()

    def _caption_kwargs(self) -> dict[str, Any]:
        cfg = self.cfg
        shared: dict[str, Any] = {
            "trigger": cfg.trigger,
            "prefix": cfg.caption_prefix,
            "suffix": cfg.caption_suffix,
            "drop": list(cfg.caption_drop),
            "max_chars": cfg.caption_max_chars,
        }
        if cfg.captioner == "vlm":
            from .captioners.vlm import DEFAULT_PROMPT

            shared.update(
                model=cfg.caption_model,
                device=cfg.caption_device,
                prompt=cfg.caption_prompt or DEFAULT_PROMPT,
                max_new_tokens=cfg.caption_tokens,
                batch_size=cfg.caption_batch,
                quiet=cfg.quiet,
            )
        elif cfg.captioner == "template":
            from .captioners.template import DEFAULT_TEMPLATE

            shared.update(
                template=cfg.caption_template or DEFAULT_TEMPLATE,
                padding=cfg.padding,
            )
        return shared

    @staticmethod
    def _region_for(record: CropRecord) -> Region:
        """Rebuild the Region the captioner needs from a manifest row.

        Landmark coordinates stay in source pixels; every attribute derived
        from them (head yaw) uses ratios between points, so the crop's offset
        and scale do not matter.
        """
        x, y, w, h = record.crop
        return Region(
            rect=Rect(x, y, w, h),
            score=record.score,
            label=record.label,
            landmarks={
                name: (float(pt[0]), float(pt[1]))
                for name, pt in (record.landmarks or {}).items()
                if len(pt) >= 2
            },
        )

    # -- output helpers ---------------------------------------------------
    def _image_name(self, index: int, tier: int) -> str:
        cfg = self.cfg
        stem = f"{cfg.prefix}{index:0{cfg.digits}d}"
        name = f"{stem}.{cfg.extension}"
        return f"{tier}/{name}" if cfg.per_size_dirs else name

    def _caption_name(self, index: int, tier: int) -> str:
        cfg = self.cfg
        stem = f"{cfg.prefix}{index:0{cfg.digits}d}"
        return f"{tier}/{stem}.txt" if cfg.per_size_dirs else f"{stem}.txt"

    def _save(self, crop: Image.Image, record: CropRecord) -> None:
        cfg = self.cfg
        target = cfg.output / record.image_file
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not cfg.overwrite:
            # Numbering is derived from the manifest, so a collision means the
            # output folder holds files this run did not account for.
            raise FileExistsError(
                f"{target} already exists; use --overwrite, --resume, "
                f"or an empty output directory"
            )
        if cfg.extension == "jpg":
            crop.save(target, "JPEG", quality=cfg.quality, subsampling=0, optimize=True)
        elif cfg.extension == "webp":
            crop.save(target, "WEBP", quality=cfg.quality, method=6)
        else:
            crop.save(target, "PNG", compress_level=6)

    # -- manifest ---------------------------------------------------------
    def _resume_state(self) -> tuple[set[str], int]:
        """Sources already processed, and the next free output number."""
        cfg = self.cfg
        manifest = cfg.output / MANIFEST_NAME
        if not manifest.exists():
            return set(), cfg.start_index
        if not cfg.resume:
            if cfg.overwrite:
                # Start the manifest afresh; appending to the old one would
                # leave rows describing files this run has replaced.
                if not cfg.dry_run:
                    manifest.unlink()
                return set(), cfg.start_index
            if cfg.dry_run:
                return set(), cfg.start_index
            raise FileExistsError(
                f"{manifest} exists from a previous run.\n"
                f"Pass --resume to continue it, --overwrite to replace it, "
                f"or choose a different --output."
            )

        done: set[str] = set()
        highest = cfg.start_index - 1
        for line in manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a torn final line from an interrupted run
            if "source" in row:
                done.add(row["source"])
            highest = max(highest, int(row.get("index", highest)))
            self._carried.append(line)
        return done, highest + 1

    def _rewrite_manifest(self) -> None:
        """Re-emit the manifest once captions exist, so rows are complete.

        Rows carried over from a resumed run are written back first, in their
        original order, ahead of this run's records.
        """
        path = self.cfg.output / MANIFEST_NAME
        with path.open("w", encoding="utf-8") as handle:
            for line in self._carried:
                handle.write(line + "\n")
            for record in self.records:
                handle.write(json.dumps(asdict(record)) + "\n")

    def _write_summary(self) -> None:
        cfg = self.cfg
        summary = {
            "version": 1,
            "config": cfg.to_dict(),
            "stats": {
                **{k: v for k, v in asdict(self.stats).items() if k != "by_tier"},
                "by_tier": {str(k): v for k, v in sorted(self.stats.by_tier.items())},
            },
            "skipped": [asdict(s) for s in self.skips],
        }
        (cfg.output / SUMMARY_NAME).write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )

    def _write_training_config(self) -> None:
        """Fill a copy of the bundled OneTrainer template from this run's stats.

        Only ``resolution`` and ``epochs`` are dataset-derived; everything else
        is copied verbatim. ``resolution`` is pinned to the *smallest* tier
        produced so OneTrainer's multi-resolution training never upscales a
        crop beyond what ``min_ratio`` already allowed. ``epochs`` targets a
        fixed total step count (4000) rather than a fixed epoch count, since
        epochs * (images / batch_size) = steps; the clamp guards against
        absurd epoch counts at either end of the dataset-size range.
        """
        if not self.stats.by_tier:
            return
        try:
            template = json.loads(ONETRAINER_TEMPLATE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._emit("warn", f"could not read {ONETRAINER_TEMPLATE_PATH.name}: {exc}")
            return

        smallest_tier = min(self.stats.by_tier)
        template["resolution"] = str(smallest_tier)

        batch_size = template.get("batch_size") or 1
        target_steps = 4000
        epochs = target_steps * batch_size / self.stats.written
        template["epochs"] = int(round(min(300, max(10, epochs))))

        (self.cfg.output / TRAINING_CONFIG_NAME).write_text(
            json.dumps(template, indent=2) + "\n", encoding="utf-8"
        )
        self._emit(
            "info",
            f"wrote {TRAINING_CONFIG_NAME} (resolution={smallest_tier}, "
            f"epochs={template['epochs']})",
        )

    # -- misc -------------------------------------------------------------
    def _tally_skip(self, reason: str) -> None:
        if reason == "no_detection":
            self.stats.skipped_no_detection += 1
        elif reason == "too_small":
            self.stats.skipped_too_small += 1
        elif reason == "blurry":
            self.stats.skipped_blurry += 1
        elif reason == "multi":
            self.stats.skipped_multi += 1

    def _close_detectors(self) -> None:
        detector = getattr(self._local, "detector", None)
        if detector is not None:
            detector.close()
            self._local.detector = None


def _default_workers() -> int:
    """Threads for the crop pass.

    Decode and detection both release the GIL, so this scales with cores;
    capped because past a point the bottleneck is disk and memory, not CPU.
    """
    return max(1, min(8, os.cpu_count() or 4))


def _ordered_map(
    fn: Callable[[Path], _ImageResult],
    items: Sequence[Path],
    workers: int,
) -> Iterator[_ImageResult]:
    """Thread-pool map that yields in input order with bounded look-ahead.

    ``Executor.map`` would queue every path up front and hold every produced
    crop in memory at once. This keeps at most ``2 * workers`` images in
    flight while still yielding strictly in order, which is what makes output
    numbering identical from run to run.
    """
    if workers <= 1:
        for item in items:
            yield fn(item)
        return

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: list = []
        source = iter(items)
        lookahead = workers * 2

        for item in _take(source, lookahead):
            pending.append(pool.submit(fn, item))

        while pending:
            yield pending.pop(0).result()
            for item in _take(source, 1):
                pending.append(pool.submit(fn, item))


def _take(iterator: Iterable, count: int) -> list:
    out = []
    for item in iterator:
        out.append(item)
        if len(out) >= count:
            break
    return out

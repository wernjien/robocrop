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
import tempfile
import threading
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np
from PIL import Image

from . import captioners, clothing, detectors, face_parse, images, masks, segment, superres
from .captioners.base import CaptionRequest
from .config import Config
from .detectors.base import Region
from .geometry import CropRejected, Rect, plan_crop, plan_native, plan_whole

MANIFEST_NAME = "manifest.jsonl"
SUMMARY_NAME = "manifest.json"
TRAINING_CONFIG_NAME = "training_config.json"
ONETRAINER_TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "robocrop.onetrainer.example.json"


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
    mask_file: str | None = None
    """The mask written with a --mask-* flag, relative to --output, or to
    ``mask_dir`` when that is set; else None."""
    mask_dir: str | None = None
    """The --mask-dir folder holding ``mask_file``, when one was given."""
    masked_faces: int = 0
    background_mask: str | None = None
    """How the background was masked: ``outline`` from the person matte,
    ``box`` from the detection box when the matte found no person, or None."""
    masked_clothing: float = 0.0
    """Fraction of the crop masked as clothing."""
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
    skipped_no_face: int = 0
    masked: int = 0
    unmasked_kept: int = 0
    background_masked: int = 0
    background_box: int = 0
    clothing_masked: int = 0
    errors: int = 0
    by_tier: dict[int, int] = field(default_factory=dict)
    seconds: float = 0.0


@dataclass
class _Crop:
    """One crop a worker produced, ready to be numbered and saved."""

    region: Region
    plan: Any
    image: Image.Image
    sharpness: float
    mask: Image.Image | None = None
    masked_faces: int = 0
    background: str | None = None
    masked_clothing: float = 0.0
    boxes: list[Rect] = field(default_factory=list)
    """Every detection in the image, for a --no-crop background mask."""

    def close(self) -> None:
        self.image.close()
        if self.mask is not None:
            self.mask.close()


@dataclass
class _ImageResult:
    """What one worker produces for one source image."""

    source: Path
    crops: list[_Crop] = field(default_factory=list)
    skips: list[SkipRecord] = field(default_factory=list)
    detections: int = 0
    offset_y: float = 0.0
    error: str = ""


class Pipeline:
    def __init__(self, config: Config, *, on_event: Callable[[str, str], None] | None = None):
        self.cfg = config
        self._emit = on_event or (lambda level, message: None)
        self._local = threading.local()
        self._mask_models: dict[str, Any] = {}
        self._mask_models_lock = threading.Lock()
        self.stats = Stats()
        self.records: list[CropRecord] = []
        self.skips: list[SkipRecord] = []
        self._resuming = False
        """True once an existing manifest is being continued with --resume."""
        self._carried: list[str] = []
        """Verbatim manifest lines from the run being resumed. The caption
        pass rewrites the manifest, and this run only holds its own records,
        so the earlier rows have to be carried across explicitly."""
        self._completed_crops: dict[str, set[int]] = {}
        """Saved crop positions, including partially completed source images."""

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

    def _face_detector(self):
        """The per-thread YuNet that finds faces to mask inside each crop."""
        existing = getattr(self._local, "face_detector", None)
        if existing is None:
            existing = detectors.create(
                "yunet", min_score=self.cfg.mask_min_score, quiet=self.cfg.quiet
            )
            self._local.face_detector = existing
        return existing

    def _segmenter(self):
        return self._mask_model("segmenter", segment.create)

    def _clothing_segmenter(self):
        return self._mask_model("clothing", clothing.create)

    def _face_parser(self):
        return self._mask_model("face_parser", face_parse.create)

    def _mask_model(self, name, factory):
        """Large models are shared; each backend serializes its inference."""
        with self._mask_models_lock:
            if name not in self._mask_models:
                self._mask_models[name] = factory(quiet=self.cfg.quiet)
            return self._mask_models[name]

    # -- entry point ------------------------------------------------------
    def run(self) -> Stats:
        started = time.monotonic()
        cfg = self.cfg

        if cfg.training_config_only:
            self._training_config_only_run()
            self.stats.seconds = time.monotonic() - started
            return self.stats
        if cfg.caption_only:
            self._caption_only_run()
            self.stats.seconds = time.monotonic() - started
            return self.stats

        if cfg.upscale and cfg.min_ratio < 1.0 / superres.FACTOR:
            self._emit(
                "warn",
                f"--min-ratio {cfg.min_ratio:g} can call for more than "
                f"{superres.FACTOR}x enlargement, beyond what the bundled "
                "AI upscaling model does in one pass -- the remainder "
                "of an oversized enlargement falls back to a plain resize",
            )

        paths = list(images.iter_images(
            cfg.input,
            exclude=cfg.exclude,
            follow_symlinks=cfg.follow_symlinks,
            # With the output inside the input (the defaults: . and
            # ./dataset), earlier crops would otherwise be re-cropped -- this
            # run's output, and any other robocrop dataset under the input.
            skip_dirs=(cfg.output, cfg.mask_dir) if cfg.mask_dir else (cfg.output,),
            skip_marker=MANIFEST_NAME,
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

        if cfg.captioner != "none" and not cfg.dry_run:
            if self._resuming:
                # A crop-time interruption can leave all earlier images
                # saved but none captioned. Resume both passes, even when
                # there are no new source images to crop.
                self.records = self._read_manifest_records()
                self._carried.clear()
                pending = self._pending_captions(self.records)
                if pending:
                    self._caption_pass(pending)
                elif self.records:
                    self._rewrite_manifest()
            elif self.records:
                self._caption_pass()

        if not cfg.dry_run:
            if cfg.training_config:
                # From the manifest, not this run's records: on --resume the
                # dataset is every run's crops, not just the latest batch.
                self._write_training_config(self._read_manifest_records())

        self.stats.seconds = time.monotonic() - started
        if not cfg.dry_run:
            self._write_summary()
        return self.stats

    # -- pass 1 -----------------------------------------------------------
    def _crop_pass(self, paths: Sequence[Path], next_index: int) -> None:
        cfg = self.cfg
        workers = cfg.workers or _default_workers()
        manifest_path = cfg.output / MANIFEST_NAME
        created = not cfg.dry_run and not manifest_path.exists()
        manifest = None if cfg.dry_run else manifest_path.open("a", encoding="utf-8")
        result = None

        try:
            for result in _ordered_map(self._process_image, paths, workers):
                self.stats.scanned += 1
                if result.error:
                    self.stats.errors += 1
                    self.skips.append(SkipRecord(str(result.source), "error", result.error))
                    self._emit("warn", f"{result.source}: {result.error}")
                    for item in result.crops:
                        item.close()
                    continue

                self.stats.detections += result.detections
                if result.detections:
                    self.stats.with_detections += 1
                self.skips.extend(result.skips)
                for skip in result.skips:
                    self._tally_skip(skip.reason)

                total = len(result.crops)
                for face_index, item in enumerate(result.crops):
                    if face_index in self._completed_crops.get(str(result.source), set()):
                        item.close()
                        continue
                    region, plan, crop = item.region, item.plan, item.image
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
                        sharpness=item.sharpness,
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
                        mask_file=(
                            self._mask_name(next_index, plan.tier)
                            if item.mask is not None else None
                        ),
                        mask_dir=(
                            str(cfg.mask_dir.resolve())
                            if item.mask is not None and cfg.mask_dir else None
                        ),
                        masked_faces=item.masked_faces,
                        background_mask=item.background,
                        masked_clothing=item.masked_clothing,
                    )

                    if not cfg.dry_run:
                        self._save(crop, record)
                        if item.mask is not None:
                            self._save_mask(item.mask, record)
                        manifest.write(json.dumps(asdict(record)) + "\n")
                        manifest.flush()

                    self.records.append(record)
                    self.stats.written += 1
                    if item.masked_faces:
                        self.stats.masked += 1
                    elif cfg.mask_faces:
                        self.stats.unmasked_kept += 1
                    if item.background:
                        self.stats.background_masked += 1
                    if item.background == "box":
                        self.stats.background_box += 1
                    if item.masked_clothing:
                        self.stats.clothing_masked += 1
                    self.stats.by_tier[plan.tier] = self.stats.by_tier.get(plan.tier, 0) + 1
                    next_index += 1

                    if cfg.verbose:
                        self._emit(
                            "info",
                            f"  {record.image_file} <- {result.source.name} "
                            f"[{region.label} {region.score:.2f}] "
                            f"base={plan.base_side:.0f}px tier={plan.tier}",
                        )
                    item.close()

                if not cfg.quiet and not cfg.verbose and self.stats.scanned % 25 == 0:
                    self._emit(
                        "progress",
                        f"  {self.stats.scanned}/{len(paths)} scanned, "
                        f"{self.stats.written} crops",
                    )
        finally:
            if result is not None:
                for item in result.crops:
                    item.close()
            if manifest is not None:
                manifest.close()
                # Left empty, it would make the next run refuse this folder.
                if created and manifest_path.stat().st_size == 0:
                    manifest_path.unlink()
            self._close_detectors()

    def _process_image(self, path: Path) -> _ImageResult:
        cfg = self.cfg
        result = _ImageResult(source=path)

        # Outside the per-image error handling on purpose: a model that will
        # not download or load is the run's problem, not this image's, and
        # must stop the run once rather than fail every image in turn.
        detector = None if cfg.skip_detection else self._detector()
        if cfg.mask_faces:
            self._face_detector()
        if cfg.mask_background:
            self._segmenter()
        if cfg.mask_clothing:
            self._clothing_segmenter()
        if cfg.mask_faces and cfg.face_mask == "outline":
            self._face_parser()

        try:
            image = images.load_image(path)
        except Exception as exc:  # noqa: BLE001
            result.error = f"could not read image: {exc}"
            return result
        # An explicit --offset-y wins; otherwise the detector says how its own
        # boxes need recentring.
        offset_y = (
            cfg.offset_y
            if cfg.offset_y is not None
            else detector.recommended_offset_y
            if detector is not None
            else 0.0
        )
        result.offset_y = offset_y
        try:
            if cfg.skip_detection:
                regions = [
                    Region(
                        Rect(0, 0, image.width, image.height), score=0.0, label="image"
                    )
                ]
            else:
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
            boxes = [r.rect for r in regions]
            whole = cfg.no_crop or cfg.keep_size or cfg.skip_detection
            if whole:
                regions = regions[:1]  # one output per photo

            for region in regions:
                try:
                    if cfg.keep_size:
                        plan = plan_native(
                            image.width, image.height,
                            min_side=cfg.min_side, max_side=cfg.max_side,
                        )
                    elif cfg.no_crop:
                        plan = plan_whole(
                            image.width, image.height, sizes=cfg.sizes, min_ratio=cfg.min_ratio,
                        )
                    elif cfg.skip_detection:
                        plan = plan_native(
                            image.width,
                            image.height,
                            min_side=1,
                            max_side=max(image.size),
                        )
                    else:
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
                    image, plan, fill=cfg.fill, fill_color=cfg.fill_color,
                    upscale=cfg.upscale,
                )

                if whole:
                    sharpness = round(_subject_sharpness(crop, plan, region.rect), 1)
                else:
                    inner_fraction = max(0.5, min(1.0, 1.0 / (1.0 + 2.0 * cfg.padding)))
                    sharpness = round(images.measure_sharpness(crop, inner_fraction=inner_fraction), 1)

                if (
                    not cfg.skip_detection
                    and cfg.min_sharpness > 0
                    and sharpness < cfg.min_sharpness
                ):
                    result.skips.append(SkipRecord(
                        str(path), "blurry",
                        f"sharpness {sharpness:.1f} below {cfg.min_sharpness:.1f}",
                        sharpness=sharpness,
                    ))
                    crop.close()
                    continue

                item = _Crop(region, plan, crop, sharpness, boxes=boxes if whole else [])
                if cfg.writes_masks and not self._mask(item, path, result):
                    crop.close()
                    continue
                result.crops.append(item)
        except Exception as exc:  # noqa: BLE001
            result.error = f"{type(exc).__name__}: {exc}"
        finally:
            image.close()
        return result

    def _mask(self, item: _Crop, path: Path, result: _ImageResult) -> bool:
        """Build ``item``'s loss mask. False means the crop is to be skipped."""
        cfg = self.cfg
        plan, bgr = item.plan, images.to_bgr(item.image)

        faces: list[Region] = []
        if cfg.mask_faces:
            faces = self._face_detector().detect(bgr)
            if not faces and cfg.mask_missing == "skip":
                result.skips.append(SkipRecord(
                    str(path), "no_face",
                    "no face found to mask (--mask-missing keep to use it anyway)",
                    sharpness=item.sharpness,
                ))
                return False

        person = None
        if cfg.mask_background:
            # The detection boxes, in the resized crop's pixels.
            boxes = [
                _project_box(r, plan, item.image.size)
                for r in item.boxes or [item.region.rect]
            ]
            matte = self._segmenter().matte(bgr)
            people = [p for p in (masks.isolate_person(matte, box) for box in boxes) if p is not None]
            item.background = "outline" if people else "box"
            if not people:
                people = [masks.box_person(item.image.size, box) for box in boxes]
            person = people[0]
            for other in people[1:]:
                person = np.maximum(person, other)

        face_matte, ovals = None, [f.rect for f in faces]
        if ovals and cfg.face_mask == "outline":
            face_matte, ovals = self._face_parser().outline(bgr, faces)

        garments = None
        if cfg.mask_clothing:
            garments = self._clothing_segmenter().matte(bgr)
            item.masked_clothing = round(float((garments >= 0.5).mean()), 3)

        # A kept crop with no face still gets a mask, all white if nothing
        # else is masked: with masked training on, a missing mask file does
        # not mean "learn everything" in every trainer.
        item.mask = masks.build_mask(
            item.image.size, ovals, cfg.mask_margin,
            person=person, background=cfg.background_weight,
            clothing=garments, clothing_weight=cfg.clothing_weight,
            face=face_matte,
        )
        item.masked_faces = len(faces)
        return True

    # -- pass 2 -----------------------------------------------------------
    def _caption_pass(self, pending: list[CropRecord] | None = None) -> None:
        """Caption ``pending`` (default: every record of this run), then
        rewrite the manifest with all of ``self.records``."""
        cfg = self.cfg
        if pending is None:
            pending = self.records
        pending = [r for r in pending if r.caption_file]
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
                try:
                    for record in chunk:
                        path = self._output_path(record.image_file)
                        image = None
                        try:
                            image = Image.open(path)
                            image.load()
                        except Exception as exc:  # noqa: BLE001
                            if image is not None:
                                image.close()
                            self._emit("warn", f"could not reopen {path.name}: {exc}")
                            self.stats.errors += 1
                            continue
                        live.append((record, image))
                        requests.append(
                            CaptionRequest(
                                image=image,
                                source_path=record.source,
                                region=self._region_for(record),
                                tier=record.tier,
                                index=record.index,
                            )
                        )

                    if not requests:
                        continue

                    texts = captioner.caption_batch(requests)
                    if len(texts) != len(live) or any(not isinstance(t, str) for t in texts):
                        raise ValueError(
                            f"captioner returned {len(texts)} captions for {len(live)} images; "
                            "expected one string per image"
                        )
                    for (record, image), text in zip(live, texts, strict=True):
                        if text:
                            self._output_path(record.caption_file).write_text(
                                text + "\n", encoding="utf-8"
                            )
                            self.stats.captioned += 1
                            record.caption = text
                        else:
                            self.stats.errors += 1
                            self._emit("warn", f"no caption returned for {record.image_file}")
                finally:
                    for record, image in live:
                        image.close()

                if not cfg.quiet:
                    done = min(start + batch, len(pending))
                    self._emit("progress", f"  captioned {done}/{len(pending)}")
        finally:
            try:
                captioner.close()
            finally:
                # Preserve completed captions if a later batch fails.
                self._rewrite_manifest()

    def _pending_captions(self, records: list[CropRecord]) -> list[CropRecord]:
        """Keep existing nonempty captions and retry missing/empty files."""
        pending = []
        for record in records:
            if not record.caption_file:
                continue
            path = self._output_path(record.caption_file)
            text = path.read_text(encoding="utf-8").strip() if path.is_file() else ""
            if text:
                record.caption = text
            else:
                pending.append(record)
        return pending

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
                # A whole photo frames a face widely; this names it an upper body portrait.
                padding=1.0 if cfg.no_crop or cfg.keep_size else cfg.padding,
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

    # -- alternate run modes -----------------------------------------------
    def _caption_only_run(self) -> None:
        """Caption an existing --output dataset without detecting or cropping.

        Regions and landmarks come back from manifest.jsonl, so measured
        (template) and pose-aware captions work exactly as they would fresh
        off a crop pass.
        """
        cfg = self.cfg
        self.records = self._load_manifest_records()

        # caption_file is trusted from the image path, not the manifest, so
        # this also works on rows written with --captioner none.
        for record in self.records:
            record.caption_file = Path(record.image_file).with_suffix(".txt").as_posix()

        pending = self.records
        if cfg.resume:
            pending = self._pending_captions(self.records)
            self.stats.skipped_existing = len(self.records) - len(pending)

        self.stats.scanned = self.stats.written = len(pending)
        self.stats.by_tier = dict(Counter(r.tier for r in pending))

        if not cfg.dry_run and pending:
            # _caption_pass rewrites manifest.jsonl with every record, not
            # only the ones captioned now; manifest.json is left alone so its
            # crop-time stats (detections, skips, ...) from the original run
            # are not clobbered by this partial one.
            self._caption_pass(pending)
        elif not cfg.dry_run and cfg.resume:
            self._rewrite_manifest()

    def _training_config_only_run(self) -> None:
        """(Re)generate training_config.json for an existing --output dataset,
        without cropping or captioning anything."""
        cfg = self.cfg
        records = self._load_manifest_records()

        self.stats.scanned = self.stats.written = len(records)
        self.stats.by_tier = dict(Counter(r.tier for r in records))

        if not cfg.dry_run:
            self._write_training_config(records)

    def _load_manifest_records(self) -> list[CropRecord]:
        """Every record in the output's manifest; raises when there are none."""
        manifest = self.cfg.output / MANIFEST_NAME
        if not manifest.exists():
            raise FileExistsError(
                f"{manifest} not found; run robocrop without --caption-only / "
                f"--training-config-only first to produce a dataset"
            )
        records = self._read_manifest_records()
        if not records:
            raise FileExistsError(f"{manifest} has no crop records")
        return records

    def _read_manifest_records(self) -> list[CropRecord]:
        """Every record in the output's manifest, or [] if there is none."""
        known = {f.name for f in fields(CropRecord)}
        records = []
        for number, row in enumerate(_read_rows(self.cfg.output / MANIFEST_NAME), 1):
            try:
                records.append(CropRecord(**{k: v for k, v in row.items() if k in known}))
            except TypeError as exc:
                raise ValueError(f"invalid crop record {number} in {MANIFEST_NAME}: {exc}") from exc
        return records

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

    def _mask_name(self, index: int, tier: int) -> str:
        image_file = self._image_name(index, tier)
        if self.cfg.mask_dir:
            return f"{Path(image_file).stem}.png"
        return masks.mask_name(image_file)

    def _output_path(self, relative: str, root: Path | None = None) -> Path:
        """``relative`` inside --output (or ``root``), refusing anything that
        escapes it.

        Paths come from manifest.jsonl, and a manifest can be edited or come
        with a downloaded dataset: a row naming ``../../x`` must not get
        this tool to write, or on --overwrite delete, outside the dataset.
        """
        base = (root or self.cfg.output).resolve()
        path = (base / relative).resolve()
        if not path.is_relative_to(base):
            raise ValueError(f"manifest path {relative!r} points outside {root or self.cfg.output}")
        return path

    def _claim(self, relative: str, root: Path | None = None) -> Path:
        """Where to write a new output file, checking it is free to take."""
        cfg = self.cfg
        target = self._output_path(relative, root)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Numbering is derived from the manifest, so a collision means the
        # output folder holds files this run did not account for. When a
        # manifest is being resumed, that is an orphan: a run interrupted
        # after saving a crop but before writing its manifest row. It is safe
        # to replace -- but not on a bare --resume into a folder that has no
        # manifest, where the file is someone else's.
        if target.exists() and not (cfg.overwrite or self._resuming):
            raise FileExistsError(
                f"{target} already exists; use --overwrite, --resume, "
                f"or an empty output directory"
            )
        return target

    def _save(self, crop: Image.Image, record: CropRecord) -> None:
        cfg = self.cfg
        target = self._claim(record.image_file)
        if cfg.extension == "jpg":
            crop.save(target, "JPEG", quality=cfg.quality, subsampling=0, optimize=True)
        elif cfg.extension == "webp":
            crop.save(target, "WEBP", quality=cfg.quality, method=6)
        else:
            crop.save(target, "PNG", compress_level=6)

    def _save_mask(self, mask: Image.Image, record: CropRecord) -> None:
        root = self.cfg.mask_dir if record.mask_dir else None
        mask.save(self._claim(record.mask_file, root), "PNG", compress_level=6)

    # -- manifest ---------------------------------------------------------
    def _resume_state(self) -> tuple[set[str], int]:
        """Sources already processed, and the next free output number."""
        cfg = self.cfg
        manifest = cfg.output / MANIFEST_NAME
        if not manifest.exists():
            return set(), cfg.start_index
        if not cfg.resume:
            if cfg.overwrite:
                # Start afresh. The old run's files go too: this run may make
                # fewer crops, and a leftover 0042.png -- or a leftover
                # 0001.txt now beside a different 0001.png -- would still be
                # trained on.
                if not cfg.dry_run:
                    self._remove_previous_run(manifest)
                return set(), cfg.start_index
            if cfg.dry_run:
                return set(), cfg.start_index
            raise FileExistsError(
                f"{manifest} exists from a previous run.\n"
                f"Pass --resume to continue it, --overwrite to replace it, "
                f"or choose a different --output."
            )

        self._resuming = True
        done: set[str] = set()
        expected: dict[str, int] = {}
        highest = cfg.start_index - 1
        text = manifest.read_text(encoding="utf-8")
        for line, row in _manifest_lines(manifest):
            if "source" in row:
                source = row["source"]
                saved = self._completed_crops.setdefault(source, set())
                saved.add(int(row.get("face_index", 0)))
                expected[source] = max(expected.get(source, 1), int(row.get("faces_in_image", 1)))
            highest = max(highest, int(row.get("index", highest)))
            self._carried.append(line)
        for source, total in expected.items():
            if set(range(total)).issubset(self._completed_crops[source]):
                done.add(source)

        # Appending after a torn or unterminated last line would glue the
        # next row onto it and lose that row too, so start from clean rows.
        if text and not text.endswith("\n") and not cfg.dry_run:
            self._write_manifest(self._carried)
        return done, highest + 1

    def _remove_previous_run(self, manifest: Path) -> None:
        """Delete the files an earlier run's manifest recorded, then the manifest."""
        removed = 0
        for row in _read_rows(manifest):
            image_file = row.get("image_file")
            if not image_file:
                continue
            # Captions and masks by their derived names as well as the
            # recorded ones: a caption-only run or an older manifest may not
            # have recorded them.
            names = {
                image_file,
                str(Path(image_file).with_suffix(".txt")),
                masks.mask_name(image_file),
                row.get("caption_file") or "",
                "" if row.get("mask_dir") else row.get("mask_file") or "",
            }
            targets = [(name, None) for name in filter(None, names)]
            # Only this run's --mask-dir: a folder named by the manifest alone is not trusted.
            mask_dir = self.cfg.mask_dir
            if mask_dir and row.get("mask_file") and row.get("mask_dir") == str(mask_dir.resolve()):
                targets.append((row["mask_file"], mask_dir))
            for name, root in targets:
                try:
                    path = self._output_path(name, root)
                except ValueError:
                    continue  # never delete outside the output directory
                if path.is_file():
                    path.unlink()
                    removed += 1
        manifest.unlink()
        if removed:
            self._emit("info", f"--overwrite: removed {removed} file(s) from the previous run")
        if (self.cfg.output / TRAINING_CONFIG_NAME).exists() and not self.cfg.training_config:
            # Kept, since it may carry hand edits -- but it describes the old
            # dataset (resolution, epochs, masked training) until regenerated.
            self._emit(
                "warn",
                f"{TRAINING_CONFIG_NAME} still describes the previous run; "
                "add --training-config, or run --training-config-only afterwards",
            )

    def _write_manifest(self, lines: Iterable[str]) -> None:
        """Replace manifest.jsonl atomically.

        It is the only record of what the dataset holds, so it must never be
        left half-written by an interruption: write a sibling, then rename.
        """
        path = self.cfg.output / MANIFEST_NAME
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".manifest-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                for line in lines:
                    handle.write(line + "\n")
            os.replace(tmp_name, path)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise

    def _rewrite_manifest(self) -> None:
        """Re-emit the manifest once captions exist, so rows are complete.

        Rows carried over from a resumed run are written back first, in their
        original order, ahead of this run's records.
        """
        self._write_manifest([
            *self._carried,
            *(json.dumps(asdict(record)) for record in self.records),
        ])

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

    def _write_training_config(self, records: Sequence[CropRecord]) -> None:
        """Fill a copy of the bundled OneTrainer template from ``records``,
        which is every crop in the dataset -- all runs of it, not the last.

        Only ``resolution`` and ``epochs`` are dataset-derived; everything else
        is copied verbatim. ``resolution`` is pinned to the *smallest* tier
        produced so OneTrainer's multi-resolution training never upscales a
        crop beyond what ``min_ratio`` already allowed. ``epochs`` targets a
        fixed total step count (4000) rather than a fixed epoch count, since
        epochs * (images / batch_size) = steps; the clamp guards against
        absurd epoch counts at either end of the dataset-size range.

        The one exception: with masks, masked training is switched on and
        each pixel's weight is left to its mask value alone -- no floor, and
        no whole-image steps. OneTrainer's defaults still train on the
        unmasked image one step in ten, which would teach the face after
        all, and floor every pixel at 0.1, which would override the mask.
        """
        if not records:
            self._emit("warn", "no crops in the dataset, so no training config was written")
            return
        if any(r.mask_dir for r in records):
            self._emit(
                "warn",
                "the masks are in a --mask-dir folder, which OneTrainer doesn't "
                "read, so no training config was written",
            )
            return
        try:
            template = json.loads(ONETRAINER_TEMPLATE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self._emit("warn", f"could not read {ONETRAINER_TEMPLATE_PATH.name}: {exc}")
            return

        # Native sizes vary per photo, so the smallest could be tiny; the template's value stands.
        native = (
            self.cfg.keep_size
            or (self.cfg.skip_detection and not self.cfg.no_crop)
            or any(r.tier not in self.cfg.sizes for r in records)
        )
        if not native:
            template["resolution"] = str(min(r.tier for r in records))
        resolution = template.get("resolution")

        batch_size = template.get("batch_size") or 1
        target_steps = 4000
        epochs = target_steps * batch_size / len(records)
        template["epochs"] = int(round(min(300, max(10, epochs))))

        unmasked = sum(1 for r in records if not r.mask_file)
        masked = unmasked < len(records)
        if masked and unmasked:
            # Typically a --resume that added or dropped a --mask-* flag.
            self._emit(
                "warn",
                f"{unmasked} of {len(records)} crops have no mask file; with "
                "masked training on, re-crop them with the same --mask-* "
                "flags as the rest",
            )
        if masked:
            template["masked_training"] = True
            template["unmasked_probability"] = 0.0
            template["unmasked_weight"] = 0.0

        (self.cfg.output / TRAINING_CONFIG_NAME).write_text(
            json.dumps(template, indent=2) + "\n", encoding="utf-8"
        )
        self._emit(
            "info",
            f"wrote {TRAINING_CONFIG_NAME} (resolution={resolution}"
            + (" from the template" if native else "") + ", "
            f"epochs={template['epochs']}"
            + (", masked training" if masked else "") + ")",
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
        elif reason == "no_face":
            self.stats.skipped_no_face += 1

    def _close_detectors(self) -> None:
        for model in self._mask_models.values():
            model.close()
        self._mask_models.clear()
        for attr in ("detector", "face_detector"):
            detector = getattr(self._local, attr, None)
            if detector is not None:
                detector.close()
                setattr(self._local, attr, None)


def _subject_sharpness(image: Image.Image, plan: Any, box: Rect) -> float:
    """Sharpness of the detected subject only, so a soft background doesn't count against it."""
    projected = _project_box(box, plan, image.size)
    left, top = max(0, int(projected.x)), max(0, int(projected.y))
    right = min(image.width, int(projected.x2))
    bottom = min(image.height, int(projected.y2))
    if right - left < 16 or bottom - top < 16:
        return images.measure_sharpness(image)
    return images.measure_sharpness(image.crop((left, top, right, bottom)))


def _project_box(box: Rect, plan: Any, size: tuple[int, int]) -> Rect:
    """Source pixels to output pixels, using the exact extraction window."""
    left, top, right, bottom = plan.rect.rounded()
    sx, sy = size[0] / (right - left), size[1] / (bottom - top)
    return Rect((box.x - left) * sx, (box.y - top) * sy, box.w * sx, box.h * sy)


def _default_workers() -> int:
    """Threads for the crop pass.

    Decode and detection both release the GIL, so this scales with cores;
    capped because past a point the bottleneck is disk and memory, not CPU.
    """
    return max(1, min(8, os.cpu_count() or 4))


def _read_rows(path: Path) -> Iterator[dict[str, Any]]:
    """The JSON rows of a manifest, skipping a torn final line."""
    for line, row in _manifest_lines(path):
        yield row


def _manifest_lines(path: Path) -> list[tuple[str, dict[str, Any]]]:
    """Validate the entire manifest before callers modify any dataset files."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    lines = [(n, line.strip()) for n, line in enumerate(text.splitlines(), 1) if line.strip()]
    rows = []
    for n, line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if n == lines[-1][0] and not text.endswith("\n"):
                continue  # only an unterminated final row can be a torn write
            raise ValueError(f"invalid manifest {path} at line {n}: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"invalid manifest {path} at line {n}: expected a JSON object")
        rows.append((line, row))
    return rows


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
        pending: deque = deque()
        source = iter(items)
        lookahead = workers * 2

        for item in _take(source, lookahead):
            pending.append(pool.submit(fn, item))

        while pending:
            yield pending.popleft().result()
            for item in _take(source, 1):
                pending.append(pool.submit(fn, item))


def _take(iterator: Iterable, count: int) -> list:
    out = []
    for item in iterator:
        out.append(item)
        if len(out) >= count:
            break
    return out

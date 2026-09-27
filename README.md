# RoboCrop

Walks a folder of photos (including every subfolder), detects and crops objects
of interest square at 512 / 768 / 1024, and writes a caption next to each crop,
ready for training or fine-tuning an image model. Works with faces, bodies,
animals, or any COCO object class.

The output directory holds one image + caption pair per crop, plus two
manifests summarizing the run:

```
dataset/
  0001.png            crop
  0001.txt            its caption
  0002.png
  0002.txt
  ...
  manifest.jsonl      one row per crop
  manifest.json       run settings, totals, and every skipped image with a reason
```

## Run it

```bash
# Crop faces (default)
./robocrop --input ~/Pictures/portraits --output ./dataset

# Crop whole bodies
./robocrop -i ~/Pictures/bodies -o ./dataset --detector yolox --detector-opt classes=person

# Crop whole bodies, with faces masked out of training (learn the body, not the identity)
./robocrop -i ~/Pictures/bodies -o ./dataset --detector yolox --detector-opt classes=person \
    --mask-faces --training-config

# Learn the subject, not the backgrounds the photos were taken against
./robocrop -i ~/Pictures/portraits -o ./dataset --mask-background --training-config

# Both: learn the body and outfit, but neither the face nor the background
./robocrop -i ~/Pictures/bodies -o ./dataset --detector yolox --detector-opt classes=person \
    --mask-faces --mask-background --training-config

# Learn the person, not the outfits they were photographed in
./robocrop -i ~/Pictures/portraits -o ./dataset --mask-clothing --training-config

# Crop dogs
./robocrop -i ~/Pictures/dogs -o ./dataset --detector yolox --detector-opt classes=dog

# Crop cats
./robocrop -i ~/Pictures/cats -o ./dataset --detector yolox --detector-opt classes=cat

# Crop only, add captions later (or never)
./robocrop -i ~/Pictures/portraits -o ./dataset --captioner none
```

Or set up a `robocrop.toml` file once and reuse it:

```bash
./robocrop    # auto-loads ./robocrop.toml
```

The first run takes a couple of minutes to set itself up (see
[Installation & Requirements](#installation--requirements) below); after that
it starts immediately.

Try the settings before committing to them:

```bash
./robocrop -i ~/Pictures/portraits -o ./dataset --dry-run
```

`--dry-run` reports exactly what would be produced, at which size, and what
would be skipped. It writes nothing and loads no caption model.

## Installation & Requirements

### Prerequisites
**Only requirement:** Python 3.11 or newer. That's it.

Works on macOS, Linux, and Windows. Supports any detectable object: faces, bodies, animals, vehicles, or any of 80 COCO classes.

```bash
python3 --version    # should be 3.11+
```

### First Run
The first time you run `./robocrop`, it:
1. Creates a private virtualenv next to the script (`.venv/`)
2. Installs detection and image processing libraries (~200 MB)
3. If `--captioner vlm` (the default), also installs captioning libraries —
   torch, transformers (~900 MB) — and downloads a vision-language model
   (~4.5 GB on first use)
4. Caches detector weights (~50 MB)

**This takes 2–5 minutes on first run.** After that, startup is instant.

Nothing is installed system-wide — everything is local to this directory.

### Disk & Memory Requirements

| What | Size | Location |
|---|---|---|
| `.venv/` (Python packages) | 1.1 GB | This directory |
| `~/.cache/robocrop/` (detector and mask weights) | ~40–175 MB | Your home directory |
| `~/.cache/huggingface/` (caption model) | 4.5–7.5 GB | Your home directory |
| Output dataset | ~2–5 MB per 1000 crops | `--output` directory |

**Total first-run disk needed:** ~5.6 GB (1.1 GB + 4.5 GB). Reused across all runs.

**Memory during run:** 
- Detection only (`--captioner template` or `--captioner none`): ~500 MB
- With default VLM captioning: ~6–8 GB (less with `smolvlm-small`)
- Apple Silicon / NVIDIA GPU is recommended but not required; CPU works fine

### Skip the Vision-Language Model
If you don't want to download 4+ GB for the caption model, use the template captioner instead:

```bash
./robocrop -i ./photos -o ./dataset --captioner template
```

This generates captions from attributes (head direction from landmarks, lighting from pixels) — no model, no download, instant. Or use `--captioner none` for crops only.

To skip the model on a machine that has it, set:
```bash
ROBOCROP_NO_VLM=1 ./robocrop -i ./photos -o ./dataset
```

### Network Requirements
- First run needs internet to download models from Hugging Face (~4.5 GB)
- After that, runs are **fully offline** — no images are sent anywhere
- If you're behind a proxy, the OS proxy settings are used automatically

## How a size is chosen

Each crop is produced at the **largest size the detected object can fill**.
An object qualifies for a size when it reaches the `--min-ratio` fraction of
it, measured **before** padding is added — padding brings in context, not
detail, so it is not allowed to inflate the size.

Objects that fall short of even the smallest configured size are skipped and
listed in `manifest.json` with the reason, so nothing is upscaled beyond 1.25x
from a source that cannot support it. Change the threshold with `--min-ratio`,
or the sizes with `--sizes 768,1024`.

A crop that qualifies for a tier but doesn't quite fill it is enlarged to fit.
By default that's a plain resize; `--upscale` switches it to AI upscaling for
a sharper result on the enlarged portion, using FSRCNN, a small neural
network trained specifically to enlarge images. A crop that already meets or
exceeds its tier is unaffected either way. This needs `opencv-contrib-python`
in place of `opencv-python` — see `requirements-upscale.txt`.

The bundled FSRCNN model enlarges by 2x in one pass, and the most a
qualifying crop can need enlarging is `1 / min-ratio` times its own size —
so keeping `--min-ratio` above 0.5 keeps every enlargement inside what the
model does in one pass. Below that, the portion beyond 2x falls back to a
plain resize; `robocrop` warns when a run is configured this way.

## Padding

`--padding` is a percentage of the detected object added to **every** side:

```
--padding 10   400 px object -> 400 + 2x40 = 480 px crop -> resized to the chosen size
--padding 0    the detection box exactly
--padding 60   much wider context around the object
```

Different detectors apply different centering corrections:
- **Face detectors** (yunet, haar) lift crops to keep hair and forehead in frame
- **Body detectors** (yolox with person) have no automatic shift
- Other objects have no automatic shift

Override it per run with `--offset-y`: a more negative value gives more
headroom, and `--offset-y 0` turns the correction off entirely.

When a padded crop runs off the edge of the photo it slides back inside by
default. `--edge extend` instead grows the canvas and fills the new area
(`--fill blur|edge|reflect|color`), and `--edge skip` refuses such crops
outright.

## Dropping blurry crops

An object can be large enough to pass `--min-ratio` and still be motion-blurred
or out of focus — `--min-sharpness` catches that separately, by scoring the
final resized crop with a standard blur heuristic (variance of the Laplacian:
higher means more genuine edge detail, i.e. sharper).

Only the **central portion** of the crop is scored, not the padded area
around it — a plain or intentionally out-of-focus background would otherwise
drag a genuinely sharp face's score down. The window size is derived from
`--padding` (a tighter crop excludes less background, a looser one excludes
more), so it stays correct whatever padding you're using.

The default threshold was calibrated against real face crops rather than
picked arbitrarily: real, sharp photos scored 15–96 in testing, while mild
artificial blur scored 3–6 — the default sits in the gap between them. It's
still a heuristic, so every crop's score is recorded in `manifest.jsonl`
(`sharpness`) whether it was kept or dropped, letting you sanity-check or
retune it from your own dataset:

```bash
./robocrop -i ./photos -o ./dataset --min-sharpness 5   # looser
./robocrop -i ./photos -o ./dataset --min-sharpness 0   # disabled
```

## Captions

Captions are written by a vision-language model that runs **on your machine**.
The model downloads once (about 4.5 GB by default) and never sends an image
anywhere afterwards.

```bash
./robocrop -i ./photos -o ./dataset --trigger "my subject"
```

```
0001.txt: my subject, facing right, smiling, blonde short hair,
          blue t shirt over denim jacket, soft natural light, white wall
```

The trigger word always comes first — that is the token the trained model
binds the subject to. Everything after it describes what *varies* between shots, which is
what you want the model to learn as changeable.

| `--caption-model` | Size | Notes |
|---|---|---|
| `smolvlm` | ~4.5 GB | default, good balance |
| `smolvlm-small` | ~1.0 GB | fastest, noticeably rougher |
| `qwen` | ~7.5 GB | most detailed, wants plenty of free memory |
| `blip` | ~1.9 GB | quick and generic, not steerable |

Any Hugging Face image-text-to-text model id also works. Run
`./robocrop --list-models` to see the list.

Other options:

- `--captioner template` — no model at all. Captions are built from attributes
  measured off the crop (head direction from the detector's landmarks, lighting
  and colour from the pixels). Instant, offline, deterministic.
- `--captioner none` — crops only, no `.txt` files.
- `--caption-prompt "..."` — tell the model what to describe.
- `--caption-drop "\bblurry\b"` — delete phrases you never want (repeatable).

### Captioning a dataset you already cropped

`--caption-only` skips detection and cropping altogether and writes captions
into an existing `--output` dataset, rebuilding each crop's region (and
landmarks, for pose hints) from its `manifest.jsonl`:

```bash
# cropped earlier with --captioner none; add captions now
./robocrop -o ./dataset --caption-only --captioner vlm

# only caption crops that don't have a .txt file yet
./robocrop -o ./dataset --caption-only --captioner template --resume
```

This is also how you switch caption backends after the fact, or add a
trigger word/suffix you forgot the first time — it rewrites every `.txt`
(and `manifest.jsonl`) unless `--resume` is given. `--input` is not needed;
only `--output` is read.

## Several faces in one photo

Every face becomes its own numbered crop by default. A face too small for the
512 tier is skipped individually, so a group shot contributes only the faces
that are actually usable.

```bash
./robocrop -i ./photos -o ./dataset --multi-face largest   # only the most prominent face per photo
./robocrop -i ./photos -o ./dataset --multi-face skip      # ignore photos with more than one face
```

## Detecting different objects

The detector is pluggable — everything downstream works off a bounding box, so it doesn't care what was detected. Pick a detector and optionally specify which classes to crop:

```bash
# Faces (default): fast, accurate, includes landmarks for pose hints
./robocrop -i ./photos -o ./dataset

# Faces (fallback): no landmarks, frontal faces only
./robocrop -i ./photos -o ./dataset --detector haar

# Whole bodies
./robocrop -i ./photos -o ./dataset --detector yolox --detector-opt classes=person

# Multiple classes at once
./robocrop -i ./photos -o ./dataset --detector yolox --detector-opt classes=dog,cat,bird

# Single class
./robocrop -i ./photos -o ./dataset --detector yolox --detector-opt classes=car
```

| `--detector` | What it finds | Notes |
|---|---|---|
| `yunet` (default) | faces | Fast, accurate, provides landmarks for pose hints |
| `haar` | faces | Fallback only; slower, frontal faces only, no landmarks |
| `yolox` | 80 COCO classes | General object detection; bodies, animals, vehicles, etc. |

### YOLOX detectable classes (80 COCO classes):

All of these work with `--detector yolox --detector-opt classes=<comma-separated list>`:

```
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog,
horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella,
handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite,
baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle,
wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange,
broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant,
bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone,
microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors,
teddy bear, hair drier, toothbrush
```

### Examples by object type:

```bash
# Dogs and cats
./robocrop -i ./photos -o ./dataset --detector yolox --detector-opt classes=dog,cat

# Vehicles
./robocrop -i ./photos -o ./dataset --detector yolox --detector-opt classes=car,truck,bus

# Furniture
./robocrop -i ./photos -o ./dataset --detector yolox --detector-opt classes=chair,couch,table

# Food items
./robocrop -i ./photos -o ./dataset --detector yolox --detector-opt classes=pizza,donut,cake,apple

# Sports equipment
./robocrop -i ./photos -o ./dataset --detector yolox --detector-opt classes=tennis_racket,skateboard,kite
```

### Adding custom detectors

To detect objects not in COCO (80 classes), you can add your own detector. Drop a module in
`src/robocrop/detectors/` exposing a `BaseDetector` subclass whose `detect()` returns
`Region` objects, and add one line to `_BACKENDS` in that package's `__init__.py`.
Nothing else changes.

## Training without the face

When the goal is a body, outfit or style that works with *other* faces, the
subject's face in the training images is a problem: the model learns it along
with everything else, and pulls every generation towards that person.

Covering the face (a blur, a white box) makes it worse — the model learns
that this subject has a blob for a face. `--mask-faces` leaves the face alone
in the image and writes a mask beside each crop instead:

```bash
./robocrop -i ./photos -o ./dataset --detector yolox --detector-opt classes=person \
    --mask-faces --training-config
```

```
dataset/0001.png             the crop, untouched
dataset/0001-masklabel.png   white = learn, soft black oval over each face
```

With masked (loss-weighted) training, the trainer scores its output only on
the white area, so the face is seen as context but never learned. The masks
are plain greyscale PNGs named `NAME-masklabel.png`, the convention OneTrainer
reads directly; trainers with a different layout just need them moved or
renamed (kohya's sd-scripts, for one, reads masks from a separate folder under
the same filenames). `--training-config` switches masking on in the generated
config — see [Generating a OneTrainer config](#generating-a-onetrainer-config).

- **Every face in a crop is masked**, not just the subject's — a bystander's
  face is identity the model should not learn either.
- **`--mask-margin`** (default 35) is the percent of the face box added to
  each side of the oval, so hair, ears and jaw are covered as well.
- **`--mask-missing`** decides what happens when no face is found in a crop
  (back of the head, strong profile, face out of frame): `skip` (default)
  drops it, since a missed face would be learned; `keep` uses it with nothing
  masked, and the summary counts these so you can check them by eye.
- **`--mask-min-score`** (default 0.5) is the face confidence needed to mask
  it — deliberately lower than `--min-score`, because a missed face costs
  more than masking a patch of background by mistake.

It needs a body or object detector; with a face detector every crop would
be all mask, so that combination is refused. Masking cuts identity leakage
sharply but not to zero — skin tone and hair colour are still visible outside
the oval — so keep the trigger word and captions about the body and outfit.

## Masking the background

Photos from the same shoot share a backdrop, and a model trained on them
learns that backdrop along with the subject: generations drift towards the
same studio wall or bedroom. `--mask-background` weights the background
down in the same `-masklabel.png`:

```bash
./robocrop -i ./photos -o ./dataset --mask-background --training-config
```

A person matting model ([MODNet](https://github.com/ZHKKKe/MODNet), 26 MB,
downloaded on first use) outlines the person in each crop, hair included.
The person stays white; everything else drops to `--background-weight`
(default 0.1). Not 0 by default — a little weight keeps the model from
drifting on a background it is never scored on; `--background-weight 0`
ignores it completely.

- **Only the detected person is kept.** The matte marks every foreground
  thing in the crop; blobs that don't reach into the detection box — a
  bystander, a ball on its own — are treated as background.
- **When no outline is found**, the detection box stands in for it, and the
  summary counts these (`background_mask: "box"` in `manifest.jsonl`).
- **It combines with `--mask-faces`** into one mask: body 1, background 0.1,
  face 0. Masked training multiplies each pixel's loss by the mask's grey
  level, so all three weights come through from a single file (check that
  your trainer reads masks as weights rather than thresholding them to
  black and white):

  ```bash
  ./robocrop -i ./photos -o ./dataset --detector yolox --detector-opt classes=person \
      --mask-faces --mask-background --training-config
  ```

It works with face crops as well as body crops — to learn a person without
the room they were photographed in. It segments people, so it is refused
with a YOLOX class list that doesn't include `person`.

## Masking clothing

A character trained on a few shoots learns the outfits along with the
person: prompt something else to wear and the old outfit bleeds through.
`--mask-clothing` weights clothing and accessories down in the same
`-masklabel.png`, so the model learns the face, hair and body but not what
they had on:

```bash
./robocrop -i ./photos -o ./dataset --mask-clothing --training-config
```

A clothes parser ([SegFormer-B2 fine-tuned on ATR](https://huggingface.co/mattmdjaga/segformer_b2_clothes),
110 MB, downloaded on first use) finds hats, sunglasses, tops, skirts,
trousers, dresses, belts, shoes, bags and scarves in each crop, and they drop
to `--clothing-weight` (default 0). Hair, face and skin are left alone.

- **Every outfit in a crop is masked**, not just the subject's, as with faces.
- **`--clothing-weight`** is 0 by default, since any weight teaches the
  outfit a little; raise it towards 0.1 if the model starts drawing clothing
  badly.
- **The mask reaches slightly past each garment's edge**, so the outfit's
  outline is not learned either; a sliver of the skin beside it goes too.
- **`masked_clothing`** in `manifest.jsonl` is the fraction of each crop
  masked as clothing; the summary counts the crops where any was found.
- **It combines with the other masks**, each pixel taking the lowest weight.
  With `--mask-background` that is person 1, clothing 0, background 0.1 —
  a character without their wardrobe or their rooms:

  ```bash
  ./robocrop -i ./photos -o ./dataset --mask-clothing --mask-background --training-config
  ```

  With `--mask-faces` as well, little but bare skin is left to learn.

Like `--mask-background` it parses people, so it is refused with a YOLOX
class list that doesn't include `person`.

## Resuming

Interrupt a long run and pick it up where it stopped:

```bash
./robocrop -i ./photos -o ./dataset --resume
```

Sources already recorded in `manifest.jsonl` are skipped and numbering carries
on without gaps. A run killed mid-crop resumes cleanly: a half-written
manifest row, or a crop saved just before the interruption, is replaced
rather than tripping over.

Pointing a fresh run at a non-empty output directory is refused rather than
silently mixed, unless you pass `--resume` or `--overwrite`. `--overwrite`
removes the crops, captions and masks the previous run's manifest lists
before starting, so a smaller new run can't leave old files behind for the
trainer to pick up; files it never made are left alone.

An output directory inside the input one is fine — the scan never enters it,
or any other folder holding a robocrop `manifest.jsonl`, and never treats
`-masklabel.png` files as photos.

## Generating a OneTrainer config

```bash
./robocrop -i ./photos -o ./dataset --training-config
```

Writes `training_config.json` into the output directory: a copy of
`robocrop.onetrainer.example.json` (bundled at the repo root) with a few
fields filled in from what this run actually produced. Everything else in
that file is yours to hand-edit for your own base model, LoRA rank,
optimizer, and so on — there is no separate override flag, the bundled file
*is* the template.

The bundled template is set up for LoRA training. For a full fine-tune, set
`training_method` to `FINE_TUNE` and lower `learning_rate` well below the
LoRA default of `1e-4` — and expect to want a larger, more varied dataset
than the ~4000-step target below was sized for.

The fields that get overwritten:

- **`resolution`** — set to the smallest output tier that appears anywhere in
  the run (e.g. `"512"` if the dataset has 512 and 768px crops but no 1024).
  OneTrainer trains at every resolution it's given, resizing images up to
  fit each one — so listing a larger tier here would have it upscale crops
  that `--min-ratio` sized precisely to avoid needing that.
- **`epochs`** — computed from this run's crop count, targeting roughly 4000
  total training steps (`epochs = 4000 * batch_size / images`, using
  whatever `batch_size` is already in the template), clamped to between 10
  and 300 so a very small or very large dataset doesn't produce a nonsensical
  epoch count.
- **`masked_training`**, **`unmasked_probability`**, **`unmasked_weight`** —
  only when the dataset has masks (see
  [Training without the face](#training-without-the-face),
  [Masking the background](#masking-the-background) and
  [Masking clothing](#masking-clothing)): masked training on,
  with no whole-image steps and no weight floor, so every pixel is weighted
  exactly as its mask says. OneTrainer's `unmasked_weight` is a floor under
  the mask — the default 0.1 would lift the masked-out faces to 0.1.

If the dataset has zero crops, or the bundled template is missing/invalid,
nothing is written and a warning is printed — the rest of the run is
unaffected.

To (re)generate `training_config.json` for a dataset you already produced,
without touching crops or captions, use `--training-config-only`:

```bash
./robocrop -o ./dataset --training-config-only
```

It reads `manifest.jsonl` for the tier counts and crop total and writes the
config the same way `--training-config` would — useful after tweaking
`robocrop.onetrainer.example.json`, or if you skipped `--training-config`
on the original run. Like `--caption-only`, it only reads `--output`.

## Useful flags

```
-p, --padding PCT       percent added to every side
-s, --sizes LIST        candidate sizes
    --min-ratio F       fraction of a size a face needs to reach
    --min-sharpness F   drop blurry crops below this score, 0 disables
    --upscale           enlarge undersized crops with AI upscaling instead of
                        a plain resize
    --offset-y F        shift the crop up/down, fraction of the box;
                        defaults to the detector's own correction, 0 disables
-f, --format png|jpg|webp
    --per-size-dirs     write into 512/, 768/, 1024/ subfolders
-t, --trigger WORD      trigger word placed first in every caption
-c, --captioner NAME    vlm | template | none
    --caption-only      caption an existing --output dataset, no re-cropping
    --multi-face MODE   all | largest | skip
-d, --detector NAME     yunet | haar | yolox
    --exclude GLOB      skip matching paths (repeatable)
    --limit N           stop after N photos, for trying settings out
-j, --workers N         threads for the detect/crop pass
-n, --dry-run           report only, write nothing
-v, --verbose           log every crop
    --mask-faces        mask every face out of training (learn body/outfit, not identity)
    --mask-background   weight the background down, keep the person
    --mask-clothing     mask clothing and accessories (learn the person, not the outfit)
    --training-config       write a filled-in OneTrainer config alongside the crops
    --training-config-only  write it for an existing --output dataset, nothing else
```

`./robocrop --help` lists all of them.

## Configuration Files

Settings can live in a TOML file instead of on the command line. By default, **`robocrop.toml` in the current directory is auto-loaded** — you don't need to pass `--config` every time:

```bash
# Auto-loads ./robocrop.toml if it exists:
./robocrop

# Or override specific settings:
./robocrop --padding 40 --min-sharpness 0
```

**To set up your config**:
1. Copy the example: `cp robocrop.example.toml robocrop.toml`
2. Edit `robocrop.toml`: set `input` to your photos folder and `output` to where you want crops
3. Adjust other settings as needed (padding, min_sharpness, trigger word, etc.)

Command-line flags always win over the config file, so you can tune a single setting without editing the file:

```bash
./robocrop --min-sharpness 15       # override just this, use rest from robocrop.toml
./robocrop --config other.toml      # use a different config file
```

**Personal config files** (`robocrop.toml`, `*.local.toml`, and generated `training_config.json`) are in `.gitignore` — they're yours to customize and won't be committed to version control.

## System Information

**Python:** Requires 3.11 or newer.  
**Platform:** macOS (Apple Silicon), Linux, Windows.

**Image formats:** Reads JPEG, PNG, WebP, TIFF, BMP, GIF, and **HEIC/HEIF/AVIF** (iPhone photos). EXIF rotation is applied before detection, so sideways-stored photos are detected and cropped upright.

**Models:**
- **Detector weights** (YuNet: 230 KB, YOLOX: 35 MB) — downloaded once, cached in `~/.cache/robocrop/`
- **Upscale model** (FSRCNN: ~40 KB, only with `--upscale`) — downloaded once, cached alongside the detector weights
- **Matting model** (MODNet: 26 MB, only with `--mask-background`) — downloaded once, cached alongside the detector weights
- **Clothes parser** (SegFormer-B2: 110 MB, only with `--mask-clothing`) — downloaded once, cached alongside the detector weights
- **Caption model** (SmolVLM by default: 4.5 GB) — downloaded once, cached in Hugging Face's cache (`~/.cache/huggingface/`), fully offline after first download
- Every model file is checked against a pinned SHA-256 before it is cached; an interrupted or corrupted download is discarded and fetched again on the next run

**GPU support:** NVIDIA CUDA and Apple Silicon (MPS) are auto-detected and used if available. CPU is fine too.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

## License

[MIT](LICENSE)

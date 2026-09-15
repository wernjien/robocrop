# TrainCrop

Walks a folder of photos (including every subfolder), detects and crops objects
of interest square at 512 / 768 / 1024, and writes a caption next to each crop
for LoRA training. Works with faces, bodies, animals, or any COCO object class.

```
dataset/
  0001.png   0001.txt
  0002.png   0002.txt
  ...
  manifest.jsonl    one row per crop
  manifest.json     run settings, totals, and every skipped image with a reason
```

## Run it

```bash
# Crop faces (default)
./traincrop --input ~/Pictures/portraits --output ./dataset

# Crop whole bodies
./traincrop -i ~/Pictures/bodies -o ./dataset --detector yolox --detector-opt classes=person

# Crop dogs
./traincrop -i ~/Pictures/dogs -o ./dataset --detector yolox --detector-opt classes=dog

# Crop cats
./traincrop -i ~/Pictures/cats -o ./dataset --detector yolox --detector-opt classes=cat

# Crop only, add captions later (or never)
./traincrop -i ~/Pictures/portraits -o ./dataset --captioner none
```

Or set up a `traincrop.toml` file once and reuse it:

```bash
./traincrop    # auto-loads ./traincrop.toml
```

The first run builds a private virtualenv next to the script and installs what
it needs; after that it starts immediately. Nothing is installed system-wide.

Try the settings before committing to them:

```bash
./traincrop -i ~/Pictures/portraits -o ./dataset --dry-run
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
The first time you run `./traincrop`, it:
1. Creates a private virtualenv next to the script (`.venv/`)
2. Installs detection and image processing libraries (~200 MB)
3. If `--captioner vlm` (the default), downloads a vision-language model (~4.5 GB on first use)
4. Caches detector weights (~50 MB)

**This takes 2–5 minutes on first run.** After that, startup is instant.

Nothing is installed system-wide — everything is local to this directory.

### Disk & Memory Requirements

| What | Size | Location |
|---|---|---|
| `.venv/` (Python packages) | 1.1 GB | This directory |
| `~/.cache/traincrop/` (models) | 4.5–7.5 GB | Your home directory |
| Output dataset | ~2–5 MB per 1000 crops | `--output` directory |

**Total first-run disk needed:** ~5.6 GB (1.1 GB + 4.5 GB). Reused across all runs.

**Memory during run:** 
- Detection only (`--captioner template` or `--captioner none`): ~500 MB
- With default VLM captioning: ~6–8 GB (less with `smolvlm-small`)
- Apple Silicon / NVIDIA GPU is recommended but not required; CPU works fine

### Skip the Vision-Language Model
If you don't want to download 4+ GB for the caption model, use the template captioner instead:

```bash
./traincrop -i ./photos -o ./dataset --captioner template
```

This generates captions from attributes (head direction from landmarks, lighting from pixels) — no model, no download, instant. Or use `--captioner none` for crops only.

To skip the model on a machine that has it, set:
```bash
TRAINCROP_NO_VLM=1 ./traincrop -i ./photos -o ./dataset
```

### Network Requirements
- First run needs internet to download models from Hugging Face (~4.5 GB)
- After that, runs are **fully offline** — no images are sent anywhere
- If you're behind a proxy, the OS proxy settings are used automatically

## How a size is chosen

Each crop is produced at the **largest size the detected object can fill**.
An object qualifies for a size when it reaches 80% of it, measured **before**
padding is added — padding brings in context, not detail, so it is not allowed
to inflate the size:

| Output | Face must be at least |
|-------:|----------------------:|
| 1024   | 819 px                |
|  768   | 614 px                |
|  512   | 410 px                |

Objects below 410 px are skipped and listed in `manifest.json` with the reason,
so nothing is upscaled beyond 1.25x from a source that cannot support it.
Change the threshold with `--min-ratio`, or the sizes with `--sizes 768,1024`.

## Padding

`--padding` is a percentage of the detected object added to **every** side:

```
--padding 20   400 px object -> 400 + 2x80 = 560 px crop -> resized to the chosen size
--padding 0    the detection box exactly
--padding 60   much wider context around the object
```

Different detectors apply different centering corrections:
- **Face detectors** (yunet, haar) lift crops by 8% to keep hair and forehead in frame
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

The default threshold, **10**, was calibrated against real face crops rather
than picked arbitrarily: real, sharp photos scored 15–96 in testing, while
mild artificial blur scored 3–6 — 10 sits in the gap between them. It's
still a heuristic, so every crop's score is recorded in `manifest.jsonl`
(`sharpness`) whether it was kept or dropped, letting you sanity-check or
retune it from your own dataset:

```bash
./traincrop -i ./photos -o ./dataset --min-sharpness 5   # looser
./traincrop -i ./photos -o ./dataset --min-sharpness 0   # disabled
```

## Captions

Captions are written by a vision-language model that runs **on your machine**.
The model downloads once (about 4.5 GB by default) and never sends an image
anywhere afterwards.

```bash
./traincrop -i ./photos -o ./dataset --trigger "my subject"
```

```
0001.txt: my subject, facing right, smiling, blonde short hair,
          blue t shirt over denim jacket, soft natural light, white wall
```

The trigger word always comes first — that is the token your LoRA binds the
subject to. Everything after it describes what *varies* between shots, which is
what you want the model to learn as changeable.

| `--caption-model` | Size | Notes |
|---|---|---|
| `smolvlm` | ~4.5 GB | default, good balance |
| `smolvlm-small` | ~1.0 GB | fastest, noticeably rougher |
| `qwen` | ~7.5 GB | most detailed, wants plenty of free memory |
| `blip` | ~1.9 GB | quick and generic, not steerable |

Any Hugging Face image-text-to-text model id also works. Run
`./traincrop --list-models` to see the list.

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
./traincrop -o ./dataset --caption-only --captioner vlm

# only caption crops that don't have a .txt file yet
./traincrop -o ./dataset --caption-only --captioner template --resume
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
--multi-face largest   # only the most prominent face per photo
--multi-face skip      # ignore photos with more than one face
```

## Detecting different objects

The detector is pluggable — everything downstream works off a bounding box, so it doesn't care what was detected. Pick a detector and optionally specify which classes to crop:

```bash
# Faces (default): fast, accurate, includes landmarks for pose hints
./traincrop -i ./photos -o ./dataset

# Faces (fallback): no landmarks, frontal faces only
./traincrop -i ./photos -o ./dataset --detector haar

# Whole bodies
./traincrop -i ./photos -o ./dataset --detector yolox --detector-opt classes=person

# Multiple classes at once
./traincrop -i ./photos -o ./dataset --detector yolox --detector-opt classes=dog,cat,bird

# Single class
./traincrop -i ./photos -o ./dataset --detector yolox --detector-opt classes=car
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
./traincrop -i ./photos -o ./dataset --detector yolox --detector-opt classes=dog,cat

# Vehicles
./traincrop -i ./photos -o ./dataset --detector yolox --detector-opt classes=car,truck,bus

# Furniture
./traincrop -i ./photos -o ./dataset --detector yolox --detector-opt classes=chair,couch,table

# Food items
./traincrop -i ./photos -o ./dataset --detector yolox --detector-opt classes=pizza,donut,cake,apple

# Sports equipment
./traincrop -i ./photos -o ./dataset --detector yolox --detector-opt classes=tennis_racket,skateboard,kite
```

### Adding custom detectors

To detect objects not in COCO (80 classes), you can add your own detector. Drop a module in
`src/traincrop/detectors/` exposing a `BaseDetector` subclass whose `detect()` returns
`Region` objects, and add one line to `_BACKENDS` in that package's `__init__.py`.
Nothing else changes.

## Resuming

Interrupt a long run and pick it up where it stopped:

```bash
./traincrop -i ./photos -o ./dataset --resume
```

Sources already recorded in `manifest.jsonl` are skipped and numbering carries
on without gaps. Pointing a fresh run at a non-empty output directory is
refused rather than silently mixed, unless you pass `--resume` or
`--overwrite`.

## Generating a OneTrainer config

```bash
./traincrop -i ./photos -o ./dataset --training-config
```

Writes `training_config.json` into the output directory: a copy of
`traincrop.onetrainer.example.json` (bundled at the repo root) with two
fields filled in from what this run actually produced. Everything else in
that file is yours to hand-edit for your own base model, LoRA rank,
optimizer, and so on — there is no separate override flag, the bundled file
*is* the template.

The two fields that get overwritten:

- **`resolution`** — set to the smallest output tier that appears anywhere in
  the run (e.g. `"512"` if the dataset has 512 and 768px crops but no 1024).
  OneTrainer trains at every resolution it's given, resizing images up to
  fit each one — so listing a larger tier here would upscale the smaller
  crops that `--min-ratio` was protecting from exactly that.
- **`epochs`** — computed from this run's crop count, targeting roughly 4000
  total training steps (`epochs = 4000 * batch_size / images`, using
  whatever `batch_size` is already in the template), clamped to between 10
  and 300 so a very small or very large dataset doesn't produce a nonsensical
  epoch count.

If the dataset has zero crops, or the bundled template is missing/invalid,
nothing is written and a warning is printed — the rest of the run is
unaffected.

To (re)generate `training_config.json` for a dataset you already produced,
without touching crops or captions, use `--training-config-only`:

```bash
./traincrop -o ./dataset --training-config-only
```

It reads `manifest.jsonl` for the tier counts and crop total and writes the
config the same way `--training-config` would — useful after tweaking
`traincrop.onetrainer.example.json`, or if you skipped `--training-config`
on the original run. Like `--caption-only`, it only reads `--output`.

## Useful flags

```
-p, --padding PCT       percent added to every side (default 20)
-s, --sizes LIST        candidate sizes (default 512,768,1024)
    --min-ratio F       fraction of a size a face must reach (default 0.8)
    --min-sharpness F   drop blurry crops below this score, 0 disables (default 10)
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
    --training-config       write a filled-in OneTrainer config alongside the crops
    --training-config-only  write it for an existing --output dataset, nothing else
```

`./traincrop --help` lists all of them.

## Configuration Files

Settings can live in a TOML file instead of on the command line. By default, **`traincrop.toml` in the current directory is auto-loaded** — you don't need to pass `--config` every time:

```bash
# Auto-loads ./traincrop.toml if it exists:
./traincrop

# Or override specific settings:
./traincrop --padding 40 --min-sharpness 0
```

**To set up your config**:
1. Copy the example: `cp traincrop.example.toml traincrop.toml`
2. Edit `traincrop.toml`: set `input` to your photos folder and `output` to where you want crops
3. Adjust other settings as needed (padding, min_sharpness, trigger word, etc.)

Command-line flags always win over the config file, so you can tune a single setting without editing the file:

```bash
./traincrop --min-sharpness 15       # override just this, use rest from traincrop.toml
./traincrop --config other.toml      # use a different config file
```

**Personal config files** (`traincrop.toml`, `*.local.toml`, and generated `training_config.json`) are in `.gitignore` — they're yours to customize and won't be committed to version control.

## System Information

**Python:** Requires 3.11 or newer.  
**Platform:** macOS (Apple Silicon & Intel), Linux, Windows (untested but should work).

**Image formats:** Reads JPEG, PNG, WebP, TIFF, BMP, GIF, and **HEIC/HEIF/AVIF** (iPhone photos). EXIF rotation is applied before detection, so sideways-stored photos are detected and cropped upright.

**Models:**
- **Detector weights** (YuNet: 230 KB, YOLOX: 35 MB) — downloaded once, cached in `~/.cache/traincrop/`
- **Caption model** (SmolVLM by default: 4.5 GB) — downloaded once, cached in Hugging Face's cache (`~/.cache/huggingface/`), fully offline after first download
- Model downloads resume on interruption; partial files are never cached

**GPU support:** NVIDIA CUDA and Apple Silicon (MPS) are auto-detected and used if available. CPU is fine too.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

## License

[MIT](LICENSE)

# RoboCrop

[![Tests](https://github.com/wernjien/robocrop/actions/workflows/tests.yml/badge.svg)](https://github.com/wernjien/robocrop/actions/workflows/tests.yml)

RoboCrop turns a folder of photos into a dataset for training an image model.
It finds the faces (or bodies, animals or other objects) in every photo, crops
each one to a 256, 512, 768 or 1024 px square by default, and writes a caption
beside it. You can choose other sizes or keep whole photos.

```
dataset/
  0001.png          crop
  0001.txt          its caption
  ...
  manifest.jsonl    one row per crop
  manifest.json     run settings, totals, and skipped images with the reason
```

It reads JPEG, PNG, WebP, TIFF, BMP, GIF, HEIC and AVIF, and everything runs on
your own computer.

## Contents

- [Install](#install)
- [Usage](#usage)
- [Reference](#reference)
  - [Settings file](#settings-file)
  - [Crop size](#crop-size)
  - [Keeping whole photos](#keeping-whole-photos)
  - [Keeping native size](#keeping-native-size)
  - [Padding and framing](#padding-and-framing)
  - [Blurry crops](#blurry-crops)
  - [Captions](#captions)
  - [Several faces in one photo](#several-faces-in-one-photo)
  - [Detectors](#detectors)
  - [Resuming and overwriting](#resuming-and-overwriting)
  - [Masking faces, backgrounds and clothing](#masking-faces-backgrounds-and-clothing)
  - [Main flags](#main-flags)
- [Updating](#updating)
- [More documentation](#more-documentation)

## Install

Allow about 6 GB of free disk space for the default setup and caption model,
plus space for your photos and dataset. Linux libraries, larger caption models
and optional masks need more space. Default captioning also needs roughly
6–8 GB of free memory; `--captioner template` avoids loading a caption model.
See [download and memory requirements](docs/install.md#what-gets-downloaded).
If you already have Python 3.11 or newer and Git, skip to step 2.

### 1. Install Python and Git

On Mac, the current dependencies require Apple silicon and macOS 14 or newer;
see [platform requirements](docs/install.md#what-gets-downloaded).

**Windows**

1. Install Python from <https://www.python.org/downloads/>. On the first
   screen, tick **Add python.exe to PATH**.
2. Install Git from <https://git-scm.com/downloads/win> with the default
   options.
3. Open a new **PowerShell** window (click Start and type `PowerShell`), and
   use it for all the commands below.

**Mac**

Open **Terminal** (press Cmd + Space and type `Terminal`), then:

1. If `brew --version` doesn't print a version, install Homebrew. It asks for
   your Mac password, and nothing shows while you type it. When it finishes,
   run the commands it prints under **Next steps**.

   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. Install Python and Git. The second line lets you type `python`, since a Mac
   only has `python3` by default.

   ```bash
   brew install python git
   echo 'export PATH="$(brew --prefix)/opt/python/libexec/bin:$PATH"' >> ~/.zprofile
   ```

3. Quit Terminal and open it again.

**Linux** (Ubuntu 24.04+ or Debian 12+)

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python-is-python3 libgl1 libglib2.0-0
```

On any system, `python --version` should now show 3.11 or newer, and
`git --version` should print a version.

### 2. Download RoboCrop

```bash
cd ~
git clone https://github.com/wernjien/robocrop.git
cd robocrop
```

### 3. Run the first-time setup

```bash
./robocrop --setup
```

On Windows, use PowerShell:

```powershell
.\robocrop.cmd --setup
```

This creates a private environment in `robocrop/.venv` and downloads about
1 GB of libraries (several GB on Linux). It can look stuck for up to 15
minutes. It's done when it prints `robocrop: setup complete`.

It also creates `robocrop.toml`, your [settings file](#settings-file).

**Using Anaconda or Miniconda?** The launchers use their own `.venv`, even
when a Conda environment is active. To use the Python and CUDA-enabled PyTorch
in your Conda environment, follow [the Anaconda setup](docs/install.md#use-an-anaconda-or-miniconda-environment-on-windows)
and run `python -m robocrop` instead of `.\robocrop.cmd`.

### 4. Try it

Put a few photos of a person in `~/Pictures/test-photos`, then run:

```bash
./robocrop --input ~/Pictures/test-photos --output ~/Pictures/robocrop-test
```

On Windows, type `.\robocrop.cmd` instead of `./robocrop`, here and in every
other example. Replace the example paths with your own folders; Windows
Pictures folders may be under OneDrive. Put quotes around paths with spaces.

The first run that writes captions downloads the caption model, about 4.5 GB.
To skip it, add `--captioner template` for simpler captions without a model.
Each crop is saved as a `.png` with its caption in a `.txt` file of the same
name. Open a few crops and captions to check the framing and descriptions
before processing your full collection.

If something goes wrong, see the [FAQ](docs/faq.md).

## Usage

Run RoboCrop from the `robocrop` folder. Every run loads `robocrop.toml` if it
exists, so settings you always want can go there instead of on the command
line (see [Settings file](#settings-file)). Flags win over the file.
`./robocrop --help` lists every option.

```bash
# Crop faces
./robocrop -i ~/Pictures/portraits -o ./dataset

# Start every caption with a trigger word
./robocrop -i ~/Pictures/portraits -o ./dataset --trigger "my subject"

# Crop whole bodies, or dogs and cats
./robocrop -i ~/Pictures/bodies -o ./dataset --detector yolox --detector-opt classes=person
./robocrop -i ~/Pictures/pets -o ./dataset --detector yolox --detector-opt classes=dog,cat

# Keep whole photos, resized, instead of cropping them
./robocrop -i ~/Pictures/portraits -o ./dataset --no-crop

# Keep whole photos at their own size, shrinking any over 1536 px
./robocrop -i ~/Pictures/portraits -o ./dataset --keep-size

# Mask faces out of training, so the model learns the outfit, not the person
./robocrop -i ~/Pictures/bodies -o ./dataset --detector yolox --detector-opt classes=person --mask-faces --training-config

# Preview crop counts and skips without saving a dataset
./robocrop -i ~/Pictures/portraits -o ./dataset --dry-run
```

Examples are separate runs: choose a new output folder for each trial.
To check the actual images, add `--limit 10 --captioner template` and inspect
the small dataset first. A dry run performs detection, resizing and blur
checks, plus any requested masks, but saves no dataset and skips captioning.
It can still download processing models and the launcher can install libraries.

The reference below explains the main settings. Use `./robocrop --help` for
the complete option list.

## Reference

### Settings file

`./robocrop --setup` creates `robocrop.toml`, a copy of
`robocrop.example.toml`, if it doesn't exist yet. Running it again never
overwrites your edits. Set `input`, `output` and anything else you want in it,
and then run just `./robocrop`. It loads `robocrop.toml` from the current
folder. Flags on the command line win over the file, and `--config other.toml`
loads a different file.

Keys are the flag names with dashes turned into underscores, except `multi`
(`--multi-face`) and `detector_opts` (`--detector-opt`). `robocrop.toml` is
listed in `.gitignore`.

For example, use these settings in `robocrop.toml` for the default crop sizes
and simple captions:

```toml
input = "~/Pictures/portraits"
output = "./dataset"
sizes = [256, 512, 768, 1024]
captioner = "template"
```

`padding` and `mask_margin` are percentages in both the file and CLI (for
example, `padding = 20`). Relative paths are resolved from the folder where
you run the command, even when the settings file is elsewhere. To turn off a
setting such as `mask_faces = true`, edit it to `false`; the CLI has no
`--no-mask-faces` flag.

### Crop size

Each detection produces one crop at the largest eligible size, rather than a
copy at every size. By default, eligibility uses the detection's longer side
before padding: it must reach `--min-ratio` (default 0.8) of the output size.

| Output size | Minimum detection size with the defaults |
|---|---|
| 256 × 256 | 204.8 px |
| 512 × 512 | 409.6 px |
| 768 × 768 | 614.4 px |
| 1024 × 1024 | 819.2 px |

For example, a 700 px detection gets a 768 px crop. Smaller detections that
cannot reach 256 px are skipped and listed in `manifest.json`. With the
default ratio, qualifying crops need at most about 1.25× enlargement.

All four sizes follow the same selection rule. The default list is equivalent to:

```bash
./robocrop -i ./photos -o ./dataset --sizes 256,512,768,1024
```

This admits detections down to 204.8 px at the default ratio. Use `--sizes 768`
for only 768 px crops, or `--sizes 512,768,1024` to omit 256 px. Sizes must be a
comma-separated list of positive integers. `--per-size-dirs` groups outputs
in folders named for the sizes actually produced: `256/`, `512/`, `768/` or `1024/`.
The generated OneTrainer config uses the smallest size produced, so a single
256 px crop sets its training resolution to 256.

A crop that doesn't quite fill its size is enlarged with a plain resize.
`--upscale` uses FSRCNN, a small AI upscaler, instead. FSRCNN enlarges 2x in one
pass. The standard install includes `opencv-contrib-python`, so `--upscale` is
ready to use without swapping OpenCV packages. Keep `--min-ratio` above 0.5;
RoboCrop warns when `--upscale` is enabled and the ratio is below 0.5.

### Keeping whole photos

For a new dataset without `manifest.jsonl`, use `--skip-detection` to import
whole images and generate captions without detecting or cropping subjects:

```bash
./robocrop -i ./photos -o ./dataset --skip-detection --captioner vlm --training-config
```

This writes numbered images, matching captions, a manifest, and the requested
OneTrainer config into the output folder. Images keep their dimensions by
default, including small images and photos without faces. They are decoded
and saved in the selected output format, rather than copied byte for byte.
Blur filtering is bypassed. Add `--no-crop` to resize to `--sizes`, or
`--keep-size` to apply `--min-side` and `--max-side`. Mask options cannot be
combined with this mode. Template captions describe lighting and colour
without estimating head pose. For native dimensions, the training config
keeps the template's resolution; check your trainer's aspect-ratio buckets.

`--no-crop` keeps each whole photo instead of cropping around the subject. Its
shape is kept, and its long side is resized to the largest size it reaches, so
a 4000×3000 photo becomes 1024×768. Photos with no detection are still skipped,
and each photo gives one output however many people are in it. Padding and
framing options don't apply, and the blur check scores the largest retained
detection. `--sizes` and `--min-ratio` are checked against the whole photo's
long side, so small photos can be skipped or slightly enlarged. Multi-subject
filters still apply: `--multi-face skip` can skip a group photo.

### Keeping native size

`--keep-size` keeps each whole photo at its own size and shape:

- a photo whose long side is between `--min-side` (default 256) and
  `--max-side` (default 1536) is kept as it is
- a larger photo is shrunk so its long side is `--max-side`
- a smaller photo is skipped

```bash
./robocrop -i ./photos -o ./dataset --keep-size --min-side 256 --max-side 1536
```

As with `--no-crop`, each photo gives one output, and photos with no detection
are skipped. Padding, framing, `--sizes` and `--min-ratio` don't determine the
output dimensions in this mode. `--keep-size` and `--no-crop` cannot be used
together. Because the sizes vary, `--training-config` keeps the template's
`resolution` instead of setting it; check it's what you want to train at.

### Padding and framing

`--padding` (default 20) adds that percentage of the detection's longer side
to every side. With `--padding 10`, a 600 px detection uses a 720 px square
window before resizing to its chosen output size.

Face detectors shift the crop up slightly to keep the hair in frame.
`--offset-y` changes the shift: a more negative value gives more headroom, and
0 turns it off.

When a crop runs off the edge of the photo, it slides back inside. If the
square is larger than the image can hold, it shrinks to fit. `--edge extend`
grows the canvas instead and fills the gap with `--fill blur`, `edge`,
`reflect` or `color`; `--edge skip` drops the crop.

### Blurry crops

`--min-sharpness` (default 10) drops blurry crops. The score is the variance of
the Laplacian over the centre of the crop, reducing the effect of padding
and soft backgrounds. It is measured after resizing, and scores vary with
image content and output size. Check your own crops before raising the cutoff.

Kept crops record their score in `manifest.jsonl` (`sharpness`), and dropped
ones are listed in `manifest.json` with the reason `blurry`. `--min-sharpness 0`
turns the check off.

### Captions

Captions are written by a vision-language model that runs on your computer.
`--trigger` puts a word first in every caption, for the trained model to link
to the subject:

```bash
./robocrop -i ./photos -o ./dataset --trigger "my subject"
```

```
my subject, a smiling person facing right with short blonde hair and a blue T-shirt under a denim jacket. Soft light falls across the person against a white wall
```

The default prompt asks for a concise training caption describing only what
is clearly visible in each crop. It covers the subject, appearance, actions,
setting, lighting, and composition. For people, it includes visible clothing,
pose, and facial details when clear. It asks the model to omit uncertain details
and avoid guessing. Check generated captions against their crops before training.

Use `--caption-prompt` to replace the entire default prompt with your own
instructions for a run:

```bash
./robocrop -i ./photos -o ./dataset --caption-prompt "Describe only the visible subject, clothing, and background. Return only the caption."
```

To save your instructions for future runs, set `caption_prompt` in `robocrop.toml`:

```toml
caption_prompt = "Describe only the visible subject, clothing, and background. Return only the caption."
```

The command-line prompt takes precedence over the settings file. To return to
the default, omit `--caption-prompt` and remove or leave empty `caption_prompt`
in your settings file. Custom prompts work with instruction-following VLMs;
`blip` ignores them, and the `template` and `none` captioners do not use them.

| `--caption-model` | Download | Notes |
|---|---|---|
| `smolvlm` | ~4.5 GB | The default |
| `smolvlm-small` | ~1.0 GB | The lightest, with rougher captions |
| `qwen` | ~7.5 GB | Qwen2.5-VL 3B; detailed captions, needs 16 GB or more of free memory |
| `joycaption` | ~17 GB | JoyCaption Beta One; designed for uncensored diffusion training captions; allow 24 GB or more of free memory |
| `joycaption-nf4` | ~17 GB | Same model loaded in 4-bit NF4; NVIDIA CUDA and optional bitsandbytes dependency required; start with batch 1 |
| `huihui-qwen3-vl-4b` | ~9 GB | Qwen3-VL 4B with refusal tuning reduced; allow 16 GB or more of free memory |
| `blip` | ~1.9 GB | The fastest; short, generic captions, and ignores `--caption-prompt` |

You can also supply a compatible Hugging Face image-text-to-text model ID.
Models that need custom remote code are not supported. `./robocrop --list-models`
shows the bundled presets. If memory is tight, try `--caption-batch 1`, the
smaller model, or template captions. Review generated captions before training.

For uncensored captioning, try [JoyCaption Beta One](https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava)
or [Huihui Qwen3-VL 4B](https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated).
JoyCaption is trained specifically for image captions with SFW and NSFW coverage;
Huihui modifies Qwen3-VL to reduce refusals. Neither guarantees factual accuracy
or refusal-free output. Both support the default instruction and custom prompts.
Start with one image per batch:

```bash
./robocrop -i ./photos -o ./dataset --caption-model joycaption --caption-batch 1
./robocrop -o ./dataset --caption-only --caption-model huihui-qwen3-vl-4b --caption-batch 1
```

Memory figures are starting estimates for GPU or Apple silicon inference;
CPU inference uses float32 weights and needs substantially more memory.
These presets require Transformers 4.57.1 or newer. The launcher installs the
updated requirement automatically; if you run Python directly in an existing
environment, upgrade with `python -m pip install -r requirements-caption.txt`.

For JoyCaption on a smaller NVIDIA GPU, use `joycaption-nf4`. It downloads the
same original weights and quantizes the language model during loading; the
vision tower and projector stay in floating point. Upstream documents an
[8 GB VRAM setup](https://github.com/fpgaminer/joycaption/blob/main/gradio-app/README.md),
though actual memory use depends on batch size and caption length. Start with
batch 1 on an RTX 5070 Ti. Quantization can change caption quality.

Install the optional dependencies in the environment you use to run RoboCrop:

```bash
# Launcher environment on macOS/Linux:
./.venv/bin/python -m pip install -r requirements-caption-nf4.txt
# Launcher environment on Windows:
.\.venv\Scripts\python.exe -m pip install -r requirements-caption-nf4.txt
# Anaconda or another environment used with python -m robocrop:
python -m pip install -r requirements-caption-nf4.txt
```

First install a CUDA-enabled PyTorch build using the
[official PyTorch installer](https://pytorch.org/get-started/locally/).
RTX 50 series GPUs need a Blackwell-compatible build (PyTorch 2.7 or newer,
with CUDA 12.8 or newer) and compatible NVIDIA drivers. Installing bitsandbytes
alone does not make a CPU-only PyTorch installation CUDA-capable. The regular
launcher does not install the NF4 extras automatically.

```bash
./robocrop -i ./photos -o ./dataset --caption-model joycaption-nf4 --caption-device cuda --caption-batch 1
```

On Windows, use `.\robocrop.cmd` with the same options. In your settings file,
set `caption_model = "joycaption-nf4"` and `caption_batch = 1`.
This preset requires NVIDIA CUDA; it reports an error on CPU or Apple silicon
instead of loading the full-precision model.

- `--captioner template` uses no model. It builds captions from the head
  direction, lighting and colour it measures, instantly and offline. Head
  direction is omitted when facial landmarks are missing; it cannot describe
  expressions, clothing, or actions.
- `--captioner none` writes no captions.
- `--caption-prompt` replaces the default prompt with your own instructions.
- `--caption-prefix` adds text after the trigger word, and `--caption-suffix`
  adds text at the end.
- `--caption-drop REGEX` removes matching text, and can be given more than once.

`--caption-only` captions a dataset previously made by RoboCrop, using its
`manifest.jsonl`; it does not scan an arbitrary folder of images. It rewrites
every caption, or with `--resume` retries missing or empty caption files:

```bash
./robocrop -o ./dataset --caption-only --captioner vlm
```

To regenerate captions for just one or a few dataset images with a custom
prompt, add `--caption-images`:

```bash
./robocrop -o ./dataset --caption-only --captioner vlm \
    --caption-images 0001.png 0003.png \
    --caption-prompt "Describe the visible clothing, pose, and background."
```

Use image paths relative to `--output` (such as `512/0001.png` when using
size subdirectories), or absolute paths to dataset images. You can repeat
`--caption-images`; repeated images are captioned once. Every selected image
must appear in `manifest.jsonl`, and an unknown selection stops the run before
any captions are written. Other captions are kept. Omit `--resume` to replace
existing captions; with it, only missing or empty captions among the selected
images are retried. `--dry-run` previews the selected count without generating
captions. Custom prompts require a model that supports them, such as the
default `smolvlm`; `blip` ignores the prompt.

### Several faces in one photo

Each face becomes its own crop. `--multi-face largest` keeps only the biggest
face in each photo, and `--multi-face skip` skips photos with more than one.
Despite the flag name, these settings also apply to bodies and objects.

### Detectors

| `--detector` | Finds |
|---|---|
| `yunet` (default) | Faces, with landmarks that give the head direction in captions |
| `haar` | Faces looking at the camera; a fallback |
| `yolox` | Any of the 80 COCO classes |

With `yolox`, choose the classes with `--detector-opt classes=`:

```bash
./robocrop -i ./photos -o ./dataset --detector yolox --detector-opt classes=person
./robocrop -i ./photos -o ./dataset --detector yolox --detector-opt classes=dog,cat,bird
```

Write spaces in class names as underscores, as in `sports_ball` or
`dining_table`. Without a class option, `yolox` finds people only.
`classes=all` crops every class. The 80 classes are:

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

### Resuming and overwriting

`--resume` continues an interrupted run. Completed photos in `manifest.jsonl`
are skipped, partly saved photos get their remaining crops, and the numbering
carries on. Missing or empty caption files are retried; existing captions are
kept. Resume with the same crop and detector settings as the interrupted run.
Keep the input path, output layout and mask settings the same too. Photos
skipped without producing a crop can be checked again during resume.

If the output folder already holds a run, RoboCrop won't start unless you pass
`--resume` or `--overwrite`. `--overwrite` first deletes the previous run's
crops, captions and masks, and leaves any other files alone.
For masks in a separate folder, pass the same `--mask-dir` to remove the old
masks there too. An existing `training_config.json` is kept unless you add
`--training-config`; regenerate it if the dataset changes.

The output folder can be inside the input folder, because the scan skips it.

### Masking faces, backgrounds and clothing

A mask tells the trainer which parts of an image to learn. Each crop gets a
greyscale mask, where white is learned, black is ignored, and grey is learned
partly. The crop itself is left untouched. The three mask options below can be
combined, and each pixel takes the lowest weight.

#### Mask folders for each trainer

Trainers look for masks in different places.

**OneTrainer** (the default) reads masks from beside the crops, named
`NAME-masklabel.png`:

```
dataset/0001.png             the crop
dataset/0001-masklabel.png   its mask
```

Add `--training-config` to have masked training turned on for you.

**ai-toolkit** and **kohya's sd-scripts** can read masks from a separate folder.
`--mask-dir` writes them as PNGs with the same base name as their crop:

```bash
./robocrop -i ./photos -o ./dataset --mask-clothing --mask-dir ./dataset-masks
```

```
dataset/0001.png             the crop
dataset-masks/0001.png       its mask
```

The mask folder must be outside the dataset folder, because ai-toolkit trains
on every image it finds there, subfolders included. In ai-toolkit's config,
point `mask_path` at it:

```yaml
datasets:
  - folder_path: /path/to/dataset
    mask_path: /path/to/dataset-masks
```

Leave ai-toolkit's `mask_min_value` at 0; raising it gives the masked areas
some weight back. See its [dataset settings](https://github.com/ostris/ai-toolkit/blob/main/toolkit/config_modules.py).
For kohya's sd-scripts, set `conditioning_data_dir` to the mask folder in
your dataset subset and enable `--masked_loss`; see its
[masked-loss guide](https://github.com/kohya-ss/sd-scripts/blob/main/docs/masked_loss_README.md).
RoboCrop writes masks as PNGs even when crops are JPG or WebP. Configure your
trainer's filename matching accordingly. `--training-config` writes a
OneTrainer config, so it can't be combined with `--mask-dir`.

#### Masking faces

`--mask-faces` lets a model learn a body, outfit or style without the person's
identity. [SegFace](https://github.com/Kartik-3004/SegFace) parses each detected
face separately at 512 x 512, masking the skin, facial features, ears and
glasses while keeping hair and neck learned. Its Swin-B checkpoint (about
1.1 GB) downloads the first time. The image itself is left alone, because
blurring or covering the face would teach the model a blob for a face.

```bash
./robocrop -i ./photos -o ./dataset --detector yolox --detector-opt classes=person --mask-faces --training-config
```

- `--face-mask oval` masks an oval over the whole head instead, hair and ears
  included. In outline mode, a detected face that the parser cannot outline
  gets an oval fallback. A face missed by the detector cannot get that fallback.
- `--mask-margin` (default 35) is the percentage of the face box added around
  the oval.
- `--mask-missing` (default `skip`) drops crops where no face is detected.
  `keep` keeps them without a face mask; background and clothing masks still
  apply if enabled. This option does not catch a missed face when another
  face in the crop was detected.
- `--mask-min-score` (default 0.5) is the face confidence needed to mask a
  face. It's lower than `--min-score` on purpose, because missing a face costs
  more than masking a patch of background.

This needs a body or object detector, not a face detector, unless you use
`--no-crop` or `--keep-size` to keep whole photos. Skin tone and hair colour
still show, so masking reduces identity leakage but doesn't remove it.

#### Masking the background

`--mask-background` stops a model from learning the backdrop of a shoot. A
matting model ([BiRefNet-matting](https://github.com/ZhengPeng7/BiRefNet))
outlines foreground connected to the detected person, and everything else
drops to `--background-weight` (default 0.1; 0 ignores it completely).

```bash
./robocrop -i ./photos -o ./dataset --mask-background --training-config
```

Separate foreground regions outside the selected detection count as
background. Touching people or objects can remain in the same foreground
region. Whole-photo modes keep foreground for all retained detections.
When no matching outline is found, the detection box is used instead, and
the crop's row in `manifest.jsonl` shows `background_mask: "box"`.

#### Masking clothing

`--mask-clothing` lets a model learn the person without their outfits. A
clothes parser
([FASHN Human Parser](https://huggingface.co/fashn-ai/fashn-human-parser)) finds
hats, glasses, tops, skirts, trousers, dresses, belts, bags, scarves and
jewelry, and drops them to `--clothing-weight` (default 0). The mask reaches
slightly past each garment's edge, so the outline isn't learned either.

```bash
./robocrop -i ./photos -o ./dataset --mask-clothing --training-config
```

`masked_clothing` in `manifest.jsonl` is the fraction of each crop that was
masked as clothing. FASHN has no separate shoe class; its feet class stays
learned so bare feet are not accidentally masked.

Inspect a few masks alongside their crops before training, especially for
profiles, small faces and photos with several people. Mask generation does
not enable masked training in your trainer by itself; use the generated
OneTrainer config or configure your trainer to read the masks.

The background and clothing masks both look for people, so with `yolox` the
class list must include `person`.

The background model downloads about 930 MB and the clothing model about
257 MB on first use. Each mask model is shared across crop workers to bound
memory use. BiRefNet and FASHN use ONNX Runtime CPU; SegFace uses CUDA or
Apple MPS when available, otherwise CPU. These models favor accuracy and
cost more memory and processing time than the earlier mask models. See
[mask model details](docs/mask-models.md) for pinned exports, class mappings
and license terms, including FASHN's non-commercial restriction.

#### The OneTrainer config

`--training-config` writes `training_config.json` into the output folder. It's
a copy of `robocrop.onetrainer.example.json` with these fields filled in from
the dataset:

- `resolution` is the smallest output size recorded in the dataset. Native
  sizes keep the template's resolution; non-square photos may still need
  resizing for your trainer's aspect-ratio buckets.
- `epochs` gives about 4000 training steps in total
  (`4000 * batch_size / images`), kept between 10 and 300.
- `masked_training`, `unmasked_probability` and `unmasked_weight` are only set
  when the dataset has masks. They make every pixel count exactly as its mask
  says.

Everything else comes from the template, which is set up for Krea-2 LoRA
training. RoboCrop does not start training or add the dataset as a OneTrainer
concept. Set up the dataset concept and check the base model, output paths,
LoRA rank and optimizer in OneTrainer before training. For a full
fine-tune, set `training_method` to `FINE_TUNE` and lower `learning_rate` well
below the LoRA default of `1e-4`.

A dataset with no crops gets no config. To write the config for a dataset you
already made with RoboCrop, without cropping again, run:

```bash
./robocrop -o ./dataset --training-config-only
```

This reads `manifest.jsonl` and replaces `training_config.json` with a fresh
copy of the template. Pass your original `--sizes` or `--keep-size` settings
so its resolution handling matches the dataset. Save any manual config edits
before regenerating it.

### Main flags

```
-i, --input DIR         folder of photos, searched with all its subfolders
-o, --output DIR        where the crops go (default ./dataset)
-p, --padding PCT       percent added to every side (default 20)
-s, --sizes LIST        crop sizes (default 256,512,768,1024)
    --min-ratio F       how much of a size a detection must fill (default 0.8)
    --min-score F       drop detections below this confidence (default 0.8)
    --min-sharpness F   drop crops below this sharpness, 0 to disable (default 10)
    --upscale           enlarge small crops with AI upscaling
    --no-crop           keep whole photos, resized, instead of cropping
    --keep-size         keep photos at native size, shrunk past --max-side
    --min-side PX       with --keep-size, minimum long side (default 256)
    --max-side PX       with --keep-size, maximum long side (default 1536)
-f, --format FMT        png, jpg or webp (default png)
    --per-size-dirs     group crops in subfolders by their output size
-d, --detector NAME     yunet, haar or yolox (default yunet)
    --detector-opt K=V  detector setting, e.g. classes=person (repeatable)
    --multi-face MODE   all, largest or skip (default all)
-c, --captioner NAME    vlm, template or none (default vlm)
    --caption-model M   caption model preset or compatible Hugging Face ID
-t, --trigger WORD      word placed first in every caption
    --caption-only      caption an existing dataset without cropping again
    --skip-detection    import whole images without detection or cropping
    --mask-faces        mask faces out of training
    --face-mask SHAPE   outline (the face's own shape) or oval (default outline)
    --mask-background   weight the background down
    --mask-clothing     mask clothing out of training
    --mask-dir DIR      put the masks in DIR, for ai-toolkit and kohya
    --training-config   write a OneTrainer config beside the crops
    --training-config-only  regenerate config for an existing RoboCrop dataset
    --exclude GLOB      skip matching paths (repeatable)
    --limit N           stop after N photos
-n, --dry-run           preview counts and skips without saving a dataset
-j, --workers N         crop worker threads (0 chooses automatically)
    --resume            continue a previous run
    --overwrite         replace a previous run
    --config FILE       read settings from a TOML file
-q, --quiet             only report warnings and errors
-v, --verbose           log every crop
```

## Updating

```bash
cd ~/robocrop
git pull
```

That's all. The next run installs any new libraries by itself, and your
`robocrop.toml` and datasets are left alone. New settings appear in
`robocrop.example.toml`; copy any you want into `robocrop.toml`. The standard
install already includes the libraries for `--upscale`; no extra setup is needed.

If an older `robocrop.toml` lists `sizes = [512, 768, 1024]`, change it to
`sizes = [256, 512, 768, 1024]` to use all current default sizes. Explicit
settings continue to override the built-in defaults.

## More documentation

- [FAQ](docs/faq.md): troubleshooting common errors
- [Install help](docs/install.md): what gets downloaded, GPU setup, and
  uninstalling
- [Mask models](docs/mask-models.md): model details, limitations and licenses
- [Development](docs/development.md): tests, launchers, and adding a detector

## License

[MIT](LICENSE)

Downloaded models have their own licenses. In particular, the clothing parser
has a non-commercial restriction; see [mask model licenses](docs/mask-models.md).

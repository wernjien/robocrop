# RoboCrop

[![Tests](https://github.com/wernjien/robocrop/actions/workflows/tests.yml/badge.svg)](https://github.com/wernjien/robocrop/actions/workflows/tests.yml)

RoboCrop turns a folder of photos into a dataset for training an image model.
It finds the faces (or bodies, animals or other objects) in every photo, crops
each one to a 512, 768 or 1024 px square, and writes a caption beside it.

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

You need about 6 GB of free disk space. If you already have Python 3.11 or
newer and Git, skip to step 2.

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
sudo apt install -y git python3 python3-venv python-is-python3 libgl1
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
./robocrop --setup          # Mac and Linux
.\robocrop.cmd --setup      # Windows
```

This creates a private environment in `robocrop/.venv` and downloads about
1 GB of libraries (several GB on Linux). It can look stuck for up to 15
minutes. It's done when it prints `robocrop: setup complete`.

It also creates `robocrop.toml`, your [settings file](#settings-file).

### 4. Try it

Put a few photos of a person in `~/Pictures/test-photos`, then run:

```bash
./robocrop --input ~/Pictures/test-photos --output ~/Pictures/robocrop-test
```

On Windows, type `.\robocrop.cmd` instead of `./robocrop`, here and in every
other example.

The first run that writes captions downloads the caption model, about 4.5 GB.
To skip it, add `--captioner template` for simpler captions without a model.
Each crop is saved as a `.png` with its caption in a `.txt` file of the same
name.

If something goes wrong, see the [FAQ](docs/faq.md).

## Usage

Run RoboCrop from the `robocrop` folder. Every run loads `robocrop.toml`, so
settings you always want can go there instead of on the command line (see
[Settings file](#settings-file)). Flags win over the file.
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

# Keep photos at their own size, cutting any over 1536 px around the subject
./robocrop -i ~/Pictures/portraits -o ./dataset --keep-size

# Mask faces out of training, so the model learns the outfit, not the person
./robocrop -i ~/Pictures/bodies -o ./dataset --detector yolox --detector-opt classes=person --mask-faces --training-config

# Show what would be cropped, without writing anything
./robocrop -i ~/Pictures/portraits -o ./dataset --dry-run
```

The rest of this README goes deeper into each of these, plus every other
setting.

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

### Crop size

Each crop gets the largest size (512, 768 or 1024 by default) that the detected
object fills to at least `--min-ratio` (default 0.8), measured before padding.
Objects too small for the smallest size are skipped and listed in
`manifest.json`, so no crop is enlarged more than 1.25x.

`--sizes` changes the sizes: `--sizes 256,512,768,1024` also allows 256 px
crops, and `--sizes 768,1024` drops 512. The generated OneTrainer config trains
at the smallest size in the dataset, so a single 256 px crop lowers it to 256.

A crop that doesn't quite fill its size is enlarged with a plain resize.
`--upscale` uses FSRCNN, a small AI upscaler, instead. FSRCNN enlarges 2x in one
pass. The standard install includes `opencv-contrib-python`, so `--upscale` is
ready to use without swapping OpenCV packages. Keep `--min-ratio` above 0.5;
RoboCrop warns you if it isn't.

### Keeping whole photos

`--no-crop` keeps each whole photo instead of cropping around the subject. Its
shape is kept, and its long side is resized to the largest size it reaches, so
a 4000×3000 photo becomes 1024×768. Photos with no detection are still skipped,
and each photo gives one output however many people are in it. Padding and
framing options don't apply, and the blur check scores only the detected
subject.

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
are skipped. Because the sizes vary, `--training-config` keeps the template's
`resolution` instead of setting it; check it's what you want to train at.

### Padding and framing

`--padding` (default 20) adds that percentage of the object's size to every
side, so `--padding 10` turns a 400 px face into a 480 px crop.

Face detectors shift the crop up slightly to keep the hair in frame.
`--offset-y` changes the shift: a more negative value gives more headroom, and
0 turns it off.

When a crop runs off the edge of the photo, it slides back inside. `--edge
extend` grows the canvas instead and fills the gap (`--fill
blur|edge|reflect|color`), and `--edge skip` drops the crop.

### Blurry crops

`--min-sharpness` (default 10) drops blurry crops. The score is the variance of
the Laplacian over the centre of the crop, so a soft background doesn't count
against a sharp face. In testing, sharp photos scored 15–96 and mildly blurred
ones 3–6.

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
my subject, facing right, smiling, blonde short hair, blue t shirt over denim jacket, soft natural light, white wall
```

| `--caption-model` | Download | Notes |
|---|---|---|
| `smolvlm` | ~4.5 GB | The default |
| `smolvlm-small` | ~1.0 GB | The lightest, with rougher captions |
| `qwen` | ~7.5 GB | The most detailed; needs 16 GB or more of free memory |
| `blip` | ~1.9 GB | The fastest; short, generic captions, and ignores `--caption-prompt` |

Any Hugging Face image-text-to-text model ID also works.

- `--captioner template` uses no model. It builds captions from the head
  direction, lighting and colour it measures, instantly and offline.
- `--captioner none` writes no captions.
- `--caption-prompt` tells the model what to describe.
- `--caption-prefix` adds text after the trigger word, and `--caption-suffix`
  adds text at the end.
- `--caption-drop REGEX` removes matching text, and can be given more than once.

`--caption-only` captions a dataset you already cropped, without cropping again.
It rewrites every caption, or with `--resume` only adds the missing ones:

```bash
./robocrop -o ./dataset --caption-only --captioner vlm
```

### Several faces in one photo

Each face becomes its own crop. `--multi-face largest` keeps only the biggest
face in each photo, and `--multi-face skip` skips photos with more than one.

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
`dining_table`. `classes=all` crops every class. The 80 classes are:

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

If the output folder already holds a run, RoboCrop won't start unless you pass
`--resume` or `--overwrite`. `--overwrite` first deletes the previous run's
crops, captions and masks, and leaves any other files alone.

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

**ai-toolkit** and **kohya's sd-scripts** read masks from a separate folder,
named exactly like their crop. `--mask-dir` writes them that way:

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
some weight back. `--training-config` writes a OneTrainer config, so it can't
be combined with `--mask-dir`.

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
  included. A face the parser can't find, such as a strong profile, gets the
  oval either way.
- `--mask-margin` (default 35) is the percentage of the face box added around
  the oval.
- `--mask-missing` (default `skip`) drops crops where no face is found, since
  an unmasked face would be learned. `keep` keeps them unmasked.
- `--mask-min-score` (default 0.5) is the face confidence needed to mask a
  face. It's lower than `--min-score` on purpose, because missing a face costs
  more than masking a patch of background.

This needs a body or object detector, not a face detector, unless you use
`--no-crop` or `--keep-size` to keep whole photos. Skin tone and hair colour
still show, so masking reduces identity leakage but doesn't remove it.

#### Masking the background

`--mask-background` stops a model from learning the backdrop of a shoot. A
matting model ([BiRefNet-matting](https://github.com/ZhengPeng7/BiRefNet))
outlines the detected person, and everything else drops to
`--background-weight` (default 0.1, and 0
ignores the background completely).

```bash
./robocrop -i ./photos -o ./dataset --mask-background --training-config
```

Other people who don't overlap the detection count as background. When no
outline is found, the detection box is used instead, and the crop's row in
`manifest.jsonl` shows `background_mask: "box"`.

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

- `resolution` is the smallest crop size in the dataset, so OneTrainer never
  enlarges a crop.
- `epochs` gives about 4000 training steps in total
  (`4000 * batch_size / images`), kept between 10 and 300.
- `masked_training`, `unmasked_probability` and `unmasked_weight` are only set
  when the dataset has masks. They make every pixel count exactly as its mask
  says.

Everything else comes from the template, which is set up for LoRA training.
Edit it for your base model, LoRA rank, optimizer and so on. For a full
fine-tune, set `training_method` to `FINE_TUNE` and lower `learning_rate` well
below the LoRA default of `1e-4`.

A dataset with no crops gets no config. To write the config for a dataset you
already made, without cropping again, run:

```bash
./robocrop -o ./dataset --training-config-only
```

### Main flags

```
-i, --input DIR         folder of photos, searched with all its subfolders
-o, --output DIR        where the crops go (default ./dataset)
-p, --padding PCT       percent added to every side (default 20)
-s, --sizes LIST        crop sizes (default 512,768,1024)
    --min-ratio F       how much of a size a detection must fill (default 0.8)
    --min-score F       drop detections below this confidence (default 0.8)
    --min-sharpness F   drop crops below this sharpness, 0 to disable (default 10)
    --upscale           enlarge small crops with AI upscaling
    --no-crop           keep whole photos, resized, instead of cropping
    --keep-size         keep photos at native size, shrunk past --max-side
-f, --format FMT        png, jpg or webp (default png)
    --per-size-dirs     write into 512/, 768/ and 1024/ subfolders
-d, --detector NAME     yunet, haar or yolox (default yunet)
    --multi-face MODE   all, largest or skip (default all)
-c, --captioner NAME    vlm, template or none (default vlm)
-t, --trigger WORD      word placed first in every caption
    --caption-only      caption an existing dataset without cropping again
    --mask-faces        mask faces out of training
    --face-mask SHAPE   outline (the face's own shape) or oval (default outline)
    --mask-background   weight the background down
    --mask-clothing     mask clothing out of training
    --mask-dir DIR      put the masks in DIR, for ai-toolkit and kohya
    --training-config   write a OneTrainer config beside the crops
    --exclude GLOB      skip matching paths (repeatable)
    --limit N           stop after N photos
-n, --dry-run           report what would happen, and write nothing
    --resume            continue a previous run
    --overwrite         replace a previous run
    --config FILE       read settings from a TOML file
-q, --quiet             only report errors
-v, --verbose           log every crop
```

## Updating

```bash
cd ~/robocrop
git pull
```

That's all. The next run installs any new libraries by itself, and your
`robocrop.toml` and datasets are left alone. New settings appear in
`robocrop.example.toml`; copy any you want into `robocrop.toml`. If you set up
[`--upscale`](#crop-size), repeat its two install commands afterwards.

## More documentation

- [FAQ](docs/faq.md): troubleshooting common errors
- [Install help](docs/install.md): what gets downloaded, GPU setup, and
  uninstalling
- [Development](docs/development.md): tests, launchers, and adding a detector

## License

[MIT](LICENSE)

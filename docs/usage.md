# Usage guide

On Windows, type `.\robocrop.cmd` wherever this guide says `./robocrop`.
`./robocrop --help` lists every option.

## Settings file

To keep your settings in a file rather than typing them each time, copy the
example:

```bash
cp robocrop.example.toml robocrop.toml
```

Set `input`, `output` and anything else you want in `robocrop.toml`, and then
run just `./robocrop`. It loads `robocrop.toml` from the current folder. Flags
on the command line win over the file, and `--config other.toml` loads a
different file.

Keys are the flag names with dashes turned into underscores, except `multi`
(`--multi-face`) and `detector_opts` (`--detector-opt`). `robocrop.toml` is
listed in `.gitignore`.

## Crop size

Each crop gets the largest size (512, 768 or 1024 by default) that the detected
object fills to at least `--min-ratio` (default 0.8), measured before padding.
Objects too small for the smallest size are skipped and listed in
`manifest.json`, so no crop is enlarged more than 1.25x.

`--sizes` changes the sizes: `--sizes 256,512,768,1024` also allows 256 px
crops, and `--sizes 768,1024` drops 512. The generated OneTrainer config trains
at the smallest size in the dataset, so a single 256 px crop lowers it to 256.

A crop that doesn't quite fill its size is enlarged with a plain resize.
`--upscale` uses FSRCNN, a small AI upscaler, instead. FSRCNN enlarges 2x in one
pass, so keep `--min-ratio` above 0.5; RoboCrop warns you if it isn't.

`--upscale` needs `opencv-contrib-python` in place of `opencv-python`. Swap them
once from the `robocrop` folder (on Windows, use `.venv\Scripts\python`):

```bash
.venv/bin/python -m pip uninstall -y opencv-python
.venv/bin/python -m pip install -r requirements-upscale.txt
```

An update can put `opencv-python` back, so repeat this if `--upscale` stops
working.

## Keeping whole photos

`--no-crop` keeps each whole photo instead of cropping around the subject. Its
shape is kept, and its long side is resized to the largest size it reaches, so
a 4000×3000 photo becomes 1024×768. Photos with no detection are still skipped,
and each photo gives one output however many people are in it. Padding and
framing options don't apply, and the blur check scores only the detected
subject.

## Keeping native size

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

## Padding and framing

`--padding` (default 20) adds that percentage of the object's size to every
side, so `--padding 10` turns a 400 px face into a 480 px crop.

Face detectors shift the crop up slightly to keep the hair in frame.
`--offset-y` changes the shift: a more negative value gives more headroom, and
0 turns it off.

When a crop runs off the edge of the photo, it slides back inside. `--edge
extend` grows the canvas instead and fills the gap (`--fill
blur|edge|reflect|color`), and `--edge skip` drops the crop.

## Blurry crops

`--min-sharpness` (default 10) drops blurry crops. The score is the variance of
the Laplacian over the centre of the crop, so a soft background doesn't count
against a sharp face. In testing, sharp photos scored 15–96 and mildly blurred
ones 3–6.

Kept crops record their score in `manifest.jsonl` (`sharpness`), and dropped
ones are listed in `manifest.json` with the reason `blurry`. `--min-sharpness 0`
turns the check off.

## Captions

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

## Several faces in one photo

Each face becomes its own crop. `--multi-face largest` keeps only the biggest
face in each photo, and `--multi-face skip` skips photos with more than one.

## Detectors

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

## Resuming and overwriting

`--resume` continues an interrupted run. Photos already in `manifest.jsonl` are
skipped, and the numbering carries on.

If the output folder already holds a run, RoboCrop won't start unless you pass
`--resume` or `--overwrite`. `--overwrite` first deletes the previous run's
crops, captions and masks, and leaves any other files alone.

The output folder can be inside the input folder, because the scan skips it.

## Main flags

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

The masks and the OneTrainer config are explained in
[Training masks](training.md).

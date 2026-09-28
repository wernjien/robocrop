# Training masks and the OneTrainer config

On Windows, type `.\robocrop.cmd` wherever this page says `./robocrop`.

## How masks work

A mask tells the trainer which parts of an image to learn. The mask options
write a greyscale `NAME-masklabel.png` beside each crop, where white is learned,
black is ignored, and grey is learned partly:

```
dataset/0001.png             the crop, untouched
dataset/0001-masklabel.png   its mask
```

OneTrainer reads this naming directly. Other trainers may need the masks moved
or renamed; kohya's sd-scripts, for example, reads the same filenames from a
separate folder. Check that your trainer uses the grey levels as loss weights
rather than rounding them to black and white.

The three mask options can be combined into one mask, and each pixel takes the
lowest weight. Add `--training-config` to have masked training turned on in
the generated config.

## Masking faces

`--mask-faces` lets a model learn a body, outfit or style without the person's
identity. Every face in each crop gets a soft black oval. The image itself is
left alone, because blurring or covering the face would teach the model a blob
for a face.

```bash
./robocrop -i ./photos -o ./dataset --detector yolox --detector-opt classes=person --mask-faces --training-config
```

- `--mask-margin` (default 35) is the percentage of the face box added around
  the oval, so that the hair, ears and jaw are covered.
- `--mask-missing` (default `skip`) drops crops where no face is found, since
  an unmasked face would be learned. `keep` keeps them unmasked.
- `--mask-min-score` (default 0.5) is the face confidence needed to mask a
  face. It's lower than `--min-score` on purpose, because missing a face costs
  more than masking a patch of background.

This needs a body or object detector, not a face detector. Skin tone and hair
colour still show, so masking reduces identity leakage but doesn't remove it.

## Masking the background

`--mask-background` stops a model from learning the backdrop of a shoot. A
matting model ([MODNet](https://github.com/ZHKKKe/MODNet)) outlines the detected
person, and everything else drops to `--background-weight` (default 0.1, and 0
ignores the background completely).

```bash
./robocrop -i ./photos -o ./dataset --mask-background --training-config
```

Other people who don't overlap the detection count as background. When no
outline is found, the detection box is used instead, and the crop's row in
`manifest.jsonl` shows `background_mask: "box"`.

## Masking clothing

`--mask-clothing` lets a model learn the person without their outfits. A
clothes parser
([SegFormer-B2](https://huggingface.co/mattmdjaga/segformer_b2_clothes)) finds
hats, sunglasses, tops, skirts, trousers, dresses, belts, shoes, bags and
scarves, and drops them to `--clothing-weight` (default 0). The mask reaches
slightly past each garment's edge, so the outline isn't learned either.

```bash
./robocrop -i ./photos -o ./dataset --mask-clothing --training-config
```

`masked_clothing` in `manifest.jsonl` is the fraction of each crop that was
masked as clothing.

The background and clothing masks both look for people, so with `yolox` the
class list must include `person`.

## The OneTrainer config

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

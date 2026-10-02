# Mask models

The standard mask backends are BiRefNet-matting for background, SegFace
Swin-B for face outlines, and FASHN Human Parser for clothing. MODNet,
SegFormer-B2/ATR and LBF are no longer loaded or downloaded. The existing
mask flags, loss weights, file names and explicit oval face mode still work.

Weights are downloaded only when their mask is requested. URLs are pinned
to repository revisions and checked against SHA-256 in `models.py`.
Downloading all three requires about 2.3 GB. Each pipeline shares one copy
of each model across its workers, and each backend serializes inference to
bound memory usage. Ordinary detection and cropping do not import PyTorch.

## Background: BiRefNet-matting

- [Original model](https://huggingface.co/ZhengPeng7/BiRefNet-matting), MIT.
- [Pinned ONNX export](https://huggingface.co/Grazier/birefnet-matting-deformconv/tree/dcf5090bd129e29d527362bdc1234e23fbc215b1):
  `birefnet_matting_deformconv.onnx`, about 930 MB.
- RGB, PIL bilinear resize of the full image to 1024 x 1024, ImageNet
  normalization. The export produces logits; sigmoid yields the alpha matte.
- ONNX Runtime CPU 1.30 or newer is used for the pinned native `DeformConv`
  export; earlier releases have incomplete operator-version support.
- Connected components intersecting the selected person's detection are kept;
  if none remain, the existing detection-box fallback is used.

This is a foreground matting model. It may include salient objects alongside
the person, and connected-component filtering cannot separate touching people.
CPU inference is substantially slower than the earlier 512px portrait model.

## Face: SegFace

- [Original model and code](https://github.com/Kartik-3004/SegFace), MIT.
- [Pinned checkpoint](https://huggingface.co/kartiknarayan/SegFace/tree/5e093b03c0523f7f32a9845bbbc75ecb027c8bee/swinb_celeba_512):
  `model_299.pt`, about 1.1 GB including unused training state.
- The Swin-B inference architecture is adapted locally in `_segface.py`;
  no unpinned remote Python code or ImageNet checkpoint is loaded. Checkpoints
  load with `weights_only=True`, strict state matching and memory mapping.
- Each YuNet face gets its own square head crop, padded at image boundaries,
  RGB bicubic resize to 512 x 512 and ImageNet normalization.
- The union of skin, ears, brows, eyes, nose, mouth, lips and glasses is masked.
  Neck, hair, hats, earrings, necklaces and clothing are excluded. A small
  outward feather covers the face boundary.
- If parsing does not cover enough of a detected face, that face uses the
  oval fallback. Missing face detections still follow `--mask-missing`.
- PyTorch selects CUDA, then Apple MPS, then CPU. Model code is imported only
  for outline masks. Attribution and licenses are in `third_party/`.

## Clothing: FASHN Human Parser

- [Original model](https://huggingface.co/fashn-ai/fashn-human-parser),
  SegFormer-B4 with 18 classes.
- [Pinned float32 ONNX export](https://huggingface.co/faisal-shohag/fashn-human-parser-onnx/tree/153ad2092f3eadd6f9b06b32471f4fa0a4a899d9):
  `onnx/model.onnx`, about 257 MB. Runs on ONNX Runtime CPU.
- RGB resize to 384 x 576 with OpenCV `INTER_AREA`, matching the official
  FASHN package, then ImageNet normalization.
- Clothing probabilities combine top, dress, skirt, pants, belt, bag, hat,
  scarf, glasses and jewelry. Scores below 0.3 become unmasked, scores above
  0.7 become fully masked, and the boundary between stays soft. This prevents
  small residual class probabilities from reducing the loss on bare skin.
  The mask grows slightly to cover garment edges.
- Face, hair, arms, hands, legs, feet and torso stay learned. There is no
  separate shoe class, so shoes are no longer explicitly masked.
- The model is optimized for clear, single-person fashion photographs. Small
  subjects and multi-person scenes may need closer inspection.
- FASHN inherits the [NVIDIA SegFormer license](https://github.com/NVlabs/SegFormer/blob/master/LICENSE),
  which restricts use to non-commercial purposes.

## Verification

Unit tests cover RGB normalization, class mappings, logit-to-alpha conversion,
face-crop projection and edge padding, fallbacks and combined loss masks.
They use stub inference so CI does not download gigabytes of weights.
When changing any backend, also run its actual pinned weights on photos and
inspect the generated masks, including small faces, profiles, necks, hair,
bare skin and garment boundaries. Passing unit tests does not establish a
model's accuracy on a user's photo collection.

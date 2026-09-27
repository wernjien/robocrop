# Quick Start

## 30 seconds

```bash
# 1. Check you have Python 3.11+
python3 --version

# 2. Make the launcher executable (if needed)
chmod +x robocrop

# 3. Run it (faces by default)
./robocrop --input ~/Pictures/portraits --output ./dataset

# First run downloads models (~5 min), then processes your photos.
# Results go to ./dataset/
```

Works with faces, bodies, animals, vehicles, or anything in the COCO dataset (80 classes).

## See what would happen first

```bash
./robocrop -i ~/Pictures/portraits -o ./dataset --dry-run
```

No downloads, no writes. Just tells you what you'd get.

## Customize

### Quick one-off changes:

```bash
# Trigger word for the subject (faces)
./robocrop -i ./photos -o ./dataset --trigger "my subject"

# Crop dogs instead of faces
./robocrop -i ./photos -o ./dataset --detector yolox --detector-opt classes=dog

# Crop cats and birds
./robocrop -i ./photos -o ./dataset --detector yolox --detector-opt classes=cat,bird

# Tighter framing (less padding)
./robocrop -i ./photos -o ./dataset --padding 15

# JPEG output instead of PNG
./robocrop -i ./photos -o ./dataset --format jpg

# Offline captions (no model download)
./robocrop -i ./photos -o ./dataset --captioner template

# Crop only, add captions later
./robocrop -i ./photos -o ./dataset --captioner none

# ...then caption that dataset without re-cropping
./robocrop -o ./dataset --caption-only --captioner vlm

# (Re)generate just the OneTrainer config for an existing dataset
./robocrop -o ./dataset --training-config-only
```

### Persistent settings (recommended):

```bash
cp robocrop.example.toml robocrop.toml
```

Edit `robocrop.toml` with your paths and preferences, then just run:

```bash
./robocrop    # auto-loads ./robocrop.toml
```

### More options:

```bash
# Full help
./robocrop --help
```

## Next Steps

1. Try `--dry-run` to preview the output
2. Check the example config: `cat robocrop.example.toml`
3. Read the README for detector options, face handling, resume, etc.

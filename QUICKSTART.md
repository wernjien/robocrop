# Quick Start

## 30 seconds

```bash
# 1. Check you have Python 3.11+
python3 --version

# 2. Make the launcher executable (if needed)
chmod +x traincrop

# 3. Run it (faces by default)
./traincrop --input ~/Pictures/portraits --output ./dataset

# First run downloads models (~5 min), then processes your photos.
# Results go to ./dataset/
```

Works with faces, bodies, animals, vehicles, or anything in the COCO dataset (80 classes).

## See what would happen first

```bash
./traincrop -i ~/Pictures/portraits -o ./dataset --dry-run
```

No downloads, no writes. Just tells you what you'd get.

## Customize

### Quick one-off changes:

```bash
# Trigger word for your LoRA (faces)
./traincrop -i ./photos -o ./dataset --trigger "my subject"

# Crop dogs instead of faces
./traincrop -i ./photos -o ./dataset --detector yolox --detector-opt classes=dog

# Crop cats and birds
./traincrop -i ./photos -o ./dataset --detector yolox --detector-opt classes=cat,bird

# Tighter framing (less padding)
./traincrop -i ./photos -o ./dataset --padding 15

# JPEG output instead of PNG
./traincrop -i ./photos -o ./dataset --format jpg

# Offline captions (no model download)
./traincrop -i ./photos -o ./dataset --captioner template

# Crop only, add captions later
./traincrop -i ./photos -o ./dataset --captioner none

# ...then caption that dataset without re-cropping
./traincrop -o ./dataset --caption-only --captioner vlm

# (Re)generate just the OneTrainer config for an existing dataset
./traincrop -o ./dataset --training-config-only
```

### Persistent settings (recommended):

```bash
cp traincrop.example.toml traincrop.toml
```

Edit `traincrop.toml` with your paths and preferences, then just run:

```bash
./traincrop    # auto-loads ./traincrop.toml
```

### More options:

```bash
# Full help
./traincrop --help
```

## Next Steps

1. Try `--dry-run` to preview the output
2. Check the example config: `cat traincrop.example.toml`
3. Read the README for detector options, face handling, resume, etc.

# Installation Guide

## TL;DR

```bash
# Check Python version
python3 --version  # needs 3.11+

# Run it (everything else is automatic)
./robocrop --input ~/Pictures/portraits --output ./dataset

# That's it. On first run, it will:
# 1. Create .venv/ locally (nothing system-wide)
# 2. Download ~4.5 GB model (only once, cached locally)
# 3. Start processing
```

## What Gets Installed

### First Run (Automatic)

| Component | Size | Where | When |
|-----------|------|-------|------|
| Python packages (venv) | 1.1 GB | `./.venv/` | First run |
| Vision-language model | 4.5–7.5 GB | `~/.cache/huggingface/` | First caption run |
| Detector weights | 50 MB | `~/.cache/robocrop/` | First detection |

**Total disk needed for first run:** ~5.6 GB  
**After first run:** Already cached, no re-downloads

### Can I Avoid the 4.5 GB Model?

Yes. Use the template captioner:

```bash
# No model download, instant, offline
./robocrop -i ./photos -o ./dataset --captioner template

# OR: Crops only, no captions
./robocrop -i ./photos -o ./dataset --captioner none
```

Or skip it if already installed:
```bash
ROBOCROP_NO_VLM=1 ./robocrop -i ./photos -o ./dataset
```

### Want Sharper Enlarged Crops? (`--upscale`)

Undersized crops are enlarged with a plain resize by default. `--upscale`
switches that to AI upscaling (FSRCNN, a small neural network trained to
enlarge images), but its `dnn_superres` module isn't in the `opencv-python`
this project installs automatically — it needs `opencv-contrib-python`
instead, and the two can't both be installed:

```bash
.venv/bin/pip uninstall -y opencv-python
.venv/bin/pip install -r requirements-upscale.txt
```

Not required for normal use — leave it out and `--upscale` is simply
unavailable (`robocrop` will say so if you pass it anyway).

## System Requirements

- **Python:** 3.11+ (check: `python3 --version`)
- **Disk:** ~6 GB (first run only)
- **RAM:** 
  - Detection only: 500 MB
  - With captions: 6–8 GB (varies by model)
  - Apple Silicon/GPU optional but recommended
- **Network:** First run downloads models (~4.5 GB), then fully offline

## Common Issues

### "No Python 3.11+"
Install via Homebrew (Mac) or your OS package manager:
```bash
# macOS
brew install python@3.11

# Ubuntu/Debian
sudo apt install python3.11

# Windows: https://www.python.org/downloads/
```

### "Not enough disk space"
The main cost is the caption model (~4.5 GB, downloaded once to `~/.cache/huggingface/`). 

Skip it:
```bash
./robocrop -i ./photos -o ./dataset --captioner template
```

### "Download hangs or fails"
Model downloads resume. If it gets stuck, just interrupt and retry — partial files are never cached.

If you're behind a proxy, set your OS proxy settings; the tool inherits them automatically.

### "Warning: You are sending unauthenticated requests to the HF Hub"
Harmless — the model still downloads. This just means you're hitting Hugging
Face's public (lower) rate limit. To raise it and speed up the first-run
download, set a free access token:

1. Create an account at [huggingface.co/join](https://huggingface.co/join)
   and a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   (the default **Read** role is enough)
2. Either export it each session:
   ```bash
   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
   ```
   or save it once, permanently:
   ```bash
   .venv/bin/pip install -q huggingface_hub[cli]
   .venv/bin/huggingface-cli login
   ```

Not required — only useful if downloads feel slow or you're rate-limited.

### "Model config: pad_token_id must be ... got 128002"
Also harmless, and fixed as of this version — some vision-language model
configs on Hugging Face (SmolVLM included) carry a stale, mismatched
`pad_token_id` left over from whatever base checkpoint they were built from.
`transformers` warns about it at config-load time, but `robocrop` always
passes the tokenizer's actual, valid pad token to generation regardless, so
captions were never affected. The warning itself is now suppressed.

### "Slow on first run"
Normal — downloading 4.5 GB and extracting takes a few minutes. After that, runs are fast.

Check progress with:
```bash
./robocrop -i ./photos -o ./dataset --dry-run
```
This runs detection without downloading the caption model.

## Uninstalling

Everything is in `.venv/`, `~/.cache/robocrop/`, and `~/.cache/huggingface/`. Just delete those:

```bash
rm -rf .venv ~/.cache/robocrop ~/.cache/huggingface/hub/models--*
```

The `robocrop` script itself stays wherever you put it.

## GPU Acceleration

NVIDIA CUDA and Apple Silicon (MPS) are auto-detected. No setup needed — they "just work" if available.

To force CPU-only:
```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 ./robocrop -i ./photos -o ./dataset
```

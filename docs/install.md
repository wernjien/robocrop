# Install help

On Windows, type `.\robocrop.cmd` wherever this page says `./robocrop`. For
setup steps and errors during install, see the [README](../README.md#install)
and the [FAQ](faq.md).

## What gets downloaded

| What | Size | Where |
|---|---|---|
| Python libraries | ~1 GB (several GB on Linux) | `robocrop/.venv` |
| Caption model | 4.5 GB by default (1–7.5 GB) | `~/.cache/huggingface` |
| Detector and mask models | 0.2–175 MB, each on first use | `~/.cache/robocrop` |

Each is downloaded once. After that, RoboCrop works offline, and your photos
never leave your computer.

Captioning needs 6–8 GB of free memory, and cropping alone needs about 500 MB.
A GPU is optional and only speeds up captioning. Apple-silicon and NVIDIA GPUs
are used automatically, but an NVIDIA GPU on Windows needs the step below. An
Intel Mac can crop but can't run the caption models.

## Optional setup

Run these from the `robocrop` folder.

### Use an NVIDIA GPU on Windows

On Windows, pip installs a CPU-only PyTorch. To use the GPU:

1. Remove the CPU build:

   ```powershell
   .venv\Scripts\python -m pip uninstall -y torch torchvision
   ```

2. On <https://pytorch.org/get-started/locally/>, choose **Stable**,
   **Windows**, **Pip**, **Python** and the newest **CUDA**.
3. Run the command it shows, with `pip3` replaced by
   `.venv\Scripts\python -m pip`.

Captioning runs then print `loading caption model … on cuda` rather than
`on cpu`. To force the CPU on any system, add `--caption-device cpu`.

### Crop without the caption libraries

`ROBOCROP_NO_VLM=1` skips installing torch and transformers (about 900 MB). Set
it on every run, since a run without it installs them, and use
`--captioner template` or `--captioner none`:

```bash
ROBOCROP_NO_VLM=1 ./robocrop -i ./photos -o ./dataset --captioner template
```

On Windows, run `$env:ROBOCROP_NO_VLM = "1"` once in each PowerShell window.

### Faster model downloads

If a run prints `You are sending unauthenticated requests to the HF Hub`,
Hugging Face is limiting the download speed. Create a free **Read** token at
<https://huggingface.co/settings/tokens>, and save it with
`.venv/bin/hf auth login` (on Windows, `.venv\Scripts\hf auth login`).

## Uninstalling

Move any datasets you want to keep out of the `robocrop` folder, and then
delete:

- the `robocrop` folder
- `~/.cache/robocrop`
- the caption models you used, in `~/.cache/huggingface/hub`
  (`models--HuggingFaceTB--SmolVLM-Instruct` is the default)

Leave the rest of `~/.cache/huggingface` alone, because other apps keep their
models there too.

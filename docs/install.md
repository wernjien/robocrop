# Install help

On Windows, type `.\robocrop.cmd` wherever this page says `./robocrop`. For
setup steps and errors during install, see the [README](../README.md#install)
and the [FAQ](faq.md).

## What gets downloaded

| What | Size | Where |
|---|---|---|
| Python libraries | ~1 GB (several GB on Linux) | `robocrop/.venv` |
| Caption model | 4.5 GB by default (1–7.5 GB) | `~/.cache/huggingface` |
| Detector and upscaler models | 0.2–35 MB, each on first use | `~/.cache/robocrop` |
| Specialist mask models | ~2.3 GB for all three, each on first use | `~/.cache/robocrop` |

Each is downloaded once. After that, RoboCrop works offline, and your photos
never leave your computer.

Captioning needs roughly 6–8 GB of free memory. Cropping memory depends on image
size and worker count, and the specialist mask models need several additional
GB. A GPU is optional: captioning and SegFace use Apple-silicon or NVIDIA GPUs
automatically. Background and clothing masks currently run on CPU. An NVIDIA
GPU on Windows needs the step below.

The current core dependencies require an Apple-silicon Mac with macOS 14 or
newer. Intel Macs are unsupported by the current PyTorch and ONNX Runtime
wheels ([PyTorch announcement](https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690),
[ONNX Runtime release files](https://pypi.org/project/onnxruntime/1.30.0/#files)).

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

`ROBOCROP_NO_VLM=1` skips the caption dependencies (transformers and accelerate).
Torch and torchvision remain installed for SegFace outline masks. Set
it on every run, since a run without it installs the caption libraries, and use
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

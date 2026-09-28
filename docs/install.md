# Install help

On Windows, type `.\robocrop.cmd` wherever this page says `./robocrop`.

## Troubleshooting

### Windows asks which app to open `robocrop` with

Type `.\robocrop.cmd`. The file without `.cmd` is the launcher for Mac and
Linux.

### `python` isn't found, or the Microsoft Store opens

- **Windows:** Python isn't on your PATH. Run the Python installer again,
  choose **Modify**, click **Next**, tick **Add Python to environment
  variables**, and install. Then open a new PowerShell window.
- **Mac:** Run the `echo 'export PATH=…'` line from the install steps again,
  and reopen Terminal. If `python3 --version` shows 3.11 or newer, RoboCrop
  works anyway, because it falls back to `python3`.
- **Linux:** Run `sudo apt install python-is-python3`.

### `Python 3.11 or newer is required`

Install a newer Python. Ubuntu 22.04 ships 3.10, so add 3.12 alongside it:

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt install -y python3.12 python3.12-venv
ROBOCROP_PYTHON=python3.12 ./robocrop --setup
```

You only need the last line once.

### `git clone` asks for a username and password

The address is mistyped. RoboCrop is public and needs no login.

### `permission denied: ./robocrop`

Run `chmod +x robocrop`. This happens when RoboCrop was downloaded as a ZIP
instead of with `git clone`.

### `could not create a virtualenv` (Linux)

Run `sudo apt install python3-venv`.

### `dependency install failed`, or a download hangs

This is usually a dropped connection. Press Ctrl + C and run the command
again. If it keeps failing, delete the `.venv` folder inside `robocrop` and
try once more.

### `caption dependency install failed` on an Intel Mac

PyTorch no longer supports Intel Macs. Follow
[Crop without the caption libraries](#crop-without-the-caption-libraries), and
use `--captioner template`.

### `ImportError: libGL.so.1` or `libgthread-2.0.so.0` (Linux)

Run `sudo apt install -y libgl1 libglib2.0-0`.

### `input directory not found`

Check the path. On Windows, the Pictures folder is often inside OneDrive, so
try `~/OneDrive/Pictures/...`. To avoid typing a path, drag the folder into
Terminal on a Mac, or use **Copy as path** from its right-click menu on
Windows. A path with spaces needs quotes around it.

### `... exists from a previous run`

The output folder already holds a run. Add `--resume` to continue it, add
`--overwrite` to replace it, or choose another folder.

### `crops written 0`

The `skipped` line shows why. Usually the faces are too small in the photos,
so use photos taken closer or at a higher resolution.

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

# FAQ

On Windows, type `.\robocrop.cmd` wherever this page says `./robocrop`.

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

### `git pull` says your local changes would be overwritten

You've edited one of RoboCrop's own files. Run `git stash` to set your edits
aside, then `git pull` again. Your settings belong in `robocrop.toml`, which
`git pull` never touches.

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

### `Torch not compiled with CUDA enabled`, but CUDA works in Anaconda

The CUDA test and RoboCrop may be using different Python environments.
`.\robocrop.cmd` uses its private `.venv` even when you have activated Conda.
Use [the Anaconda setup](install.md#use-an-anaconda-or-miniconda-environment-on-windows)
to run `python -m robocrop` with your active environment, or install
[CUDA-enabled PyTorch in the launcher's environment](install.md#use-an-nvidia-gpu-on-windows).

`--caption-device cuda` requests the GPU; it cannot add CUDA support to a
CPU-only PyTorch build. The `loading caption model ... on cuda` message prints
before the model moves to the GPU, so it alone does not confirm successful
GPU loading.

### Dependency installation fails on an Intel Mac

The current core requirements include PyTorch and ONNX Runtime versions
without Intel Mac wheels. `ROBOCROP_NO_VLM=1` skips caption libraries but
does not remove those core requirements, so it is not an Intel Mac workaround.
Use a supported system; see [platform requirements](install.md#what-gets-downloaded).

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

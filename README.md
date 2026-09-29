# RoboCrop

RoboCrop turns a folder of photos into a dataset for training an image model.
It finds the faces (or bodies, animals or other objects) in every photo, crops
each one to a 512, 768 or 1024 px square, and writes a caption beside it.

```
dataset/
  0001.png          crop
  0001.txt          its caption
  ...
  manifest.jsonl    one row per crop
  manifest.json     run settings, totals, and skipped images with the reason
```

It reads JPEG, PNG, WebP, TIFF, BMP, GIF, HEIC and AVIF, and everything runs on
your own computer.

## Install

You need about 6 GB of free disk space. If you already have Python 3.11 or
newer and Git, skip to step 2.

### 1. Install Python and Git

**Windows**

1. Install Python from <https://www.python.org/downloads/>. On the first
   screen, tick **Add python.exe to PATH**.
2. Install Git from <https://git-scm.com/downloads/win> with the default
   options.
3. Open a new **PowerShell** window (click Start and type `PowerShell`), and
   use it for all the commands below.

**Mac**

Open **Terminal** (press Cmd + Space and type `Terminal`), then:

1. If `brew --version` doesn't print a version, install Homebrew. It asks for
   your Mac password, and nothing shows while you type it. When it finishes,
   run the commands it prints under **Next steps**.

   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. Install Python and Git. The second line lets you type `python`, since a Mac
   only has `python3` by default.

   ```bash
   brew install python git
   echo 'export PATH="$(brew --prefix)/opt/python/libexec/bin:$PATH"' >> ~/.zprofile
   ```

3. Quit Terminal and open it again.

**Linux** (Ubuntu 24.04+ or Debian 12+)

```bash
sudo apt update
sudo apt install -y git python3 python3-venv python-is-python3 libgl1
```

On any system, `python --version` should now show 3.11 or newer, and
`git --version` should print a version.

### 2. Download RoboCrop

```bash
cd ~
git clone https://github.com/wernjien/robocrop.git
cd robocrop
```

### 3. Run the first-time setup

```bash
./robocrop --setup          # Mac and Linux
.\robocrop.cmd --setup      # Windows
```

This creates a private environment in `robocrop/.venv` and downloads about
1 GB of libraries (several GB on Linux). It can look stuck for up to 15
minutes. It's done when it prints `robocrop: setup complete`.

### 4. Try it

Put a few photos of a person in `~/Pictures/test-photos`, then run:

```bash
./robocrop --input ~/Pictures/test-photos --output ~/Pictures/robocrop-test
```

On Windows, type `.\robocrop.cmd` instead of `./robocrop`, here and in every
other example.

The first run that writes captions downloads the caption model, about 4.5 GB.
To skip it, add `--captioner template` for simpler captions without a model.
Each crop is saved as a `.png` with its caption in a `.txt` file of the same
name.

If something goes wrong, see [Troubleshooting](docs/install.md#troubleshooting).

## Usage

Run RoboCrop from the `robocrop` folder:

```bash
# Crop faces
./robocrop -i ~/Pictures/portraits -o ./dataset

# Start every caption with a trigger word
./robocrop -i ~/Pictures/portraits -o ./dataset --trigger "my subject"

# Crop whole bodies, or dogs and cats
./robocrop -i ~/Pictures/bodies -o ./dataset --detector yolox --detector-opt classes=person
./robocrop -i ~/Pictures/pets -o ./dataset --detector yolox --detector-opt classes=dog,cat

# Keep whole photos, resized, instead of cropping them
./robocrop -i ~/Pictures/portraits -o ./dataset --no-crop

# Show what would be cropped, without writing anything
./robocrop -i ~/Pictures/portraits -o ./dataset --dry-run
```

To update RoboCrop, run `git pull` in the `robocrop` folder. The next run
installs anything new.

## Documentation

- [Usage guide](docs/usage.md): settings files, crop sizes, padding, blurry
  crops, captions, detectors, resuming, and all the main flags
- [Training masks](docs/training.md): masking faces, backgrounds and clothing
  out of training, mask folders for OneTrainer, ai-toolkit and kohya, and the
  generated OneTrainer config
- [Install help](docs/install.md): troubleshooting, what gets downloaded, GPU
  setup, and uninstalling
- [Development](docs/development.md): tests, launchers, and adding a detector

## License

[MIT](LICENSE)

"""Download-once, cache-forever model files.

Model weights are not shipped with the inference libraries, so they are fetched
on first use into ``~/.cache/robocrop/models`` (override with
``ROBOCROP_CACHE``).

Every file is checked against a pinned SHA-256 before it is cached, so a
truncated download, an HTML error page, or a file swapped upstream is
refused rather than loaded for inference.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

_OPENCV_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
_OPENCV_DATA = "https://raw.githubusercontent.com/opencv/opencv/4.x/data"
_FSRCNN = "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models"
# Pinned to a commit rather than main, so the file cannot change under us.
_BIREFNET = "https://huggingface.co/Grazier/birefnet-matting-deformconv/resolve/dcf5090bd129e29d527362bdc1234e23fbc215b1"
_FASHN = "https://huggingface.co/faisal-shohag/fashn-human-parser-onnx/resolve/153ad2092f3eadd6f9b06b32471f4fa0a4a899d9/onnx"
_SEGFACE = "https://huggingface.co/kartiknarayan/SegFace/resolve/5e093b03c0523f7f32a9845bbbc75ecb027c8bee/swinb_celeba_512"


class Model(NamedTuple):
    url: str
    sha256: str
    filename: str = ""
    """Cache filename, when the URL's own ends in something too generic to
    share a cache directory with (Hugging Face exports are all model.onnx)."""

    @property
    def cache_name(self) -> str:
        return self.filename or Path(self.url).name


REGISTRY: dict[str, Model] = {
    "yunet": Model(
        f"{_OPENCV_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    ),
    "haar_frontalface": Model(
        f"{_OPENCV_DATA}/haarcascades/haarcascade_frontalface_default.xml",
        "0f7d4527844eb514d4a4948e822da90fbb16a34a0bbbbc6adc6498747a5aafb0",
    ),
    "haar_profileface": Model(
        f"{_OPENCV_DATA}/haarcascades/haarcascade_profileface.xml",
        "b39a4a3be45539db146a7fc1d3e761a292c196eb88421185e6a615b3055e612d",
    ),
    "yolox": Model(
        f"{_OPENCV_ZOO}/object_detection_yolox/object_detection_yolox_2022nov.onnx",
        "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
    ),
    "fsrcnn_x2": Model(
        f"{_FSRCNN}/FSRCNN_x2.pb",
        "366b33f0084c7b3f2bf6724f0a2c77bca94fcec9d7b6d72389d330073b380d5c",
    ),
    "birefnet_matting": Model(
        f"{_BIREFNET}/birefnet_matting_deformconv.onnx",
        "5aae0819fdb4521ec4c8914a4f898a8d83d2fae5edb91753a306712e6ea2bd42",
        "birefnet_matting.onnx",
    ),
    "fashn_clothes": Model(
        f"{_FASHN}/model.onnx",
        "b05324a0b4a249530da089c1259099bf4823f6ae654dd7b550f6f36ccbe23cda",
        "fashn_human_parser.onnx",
    ),
    "segface": Model(
        f"{_SEGFACE}/model_299.pt",
        "320b1c167191804913323b229a5256b1fc71f1987161fc54dc440e59af45dae8",
        "segface_swin_celeba.pt",
    ),
}

#: Serialises downloads. Every crop worker builds its own detector on first
#: use, so on a first run eight threads would otherwise fetch the same 35 MB
#: file at once; with the lock, one downloads and the rest find it cached.
_download_lock = threading.Lock()

#: Files already checksummed in this process, so each is hashed once per run
#: rather than once per worker thread.
_verified: set[Path] = set()


class ModelUnavailable(RuntimeError):
    """A model file could not be fetched and is not already cached."""


def cache_dir() -> Path:
    root = os.environ.get("ROBOCROP_CACHE")
    base = Path(root).expanduser() if root else Path.home() / ".cache" / "robocrop"
    path = base / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure(name: str, *, quiet: bool = False) -> Path:
    """Return a local path to ``name``, downloading it if necessary."""
    try:
        model = REGISTRY[name]
    except KeyError:
        raise ModelUnavailable(f"unknown model {name!r}") from None

    target = cache_dir() / model.cache_name
    with _download_lock:
        if target in _verified:
            return target
        if not (target.exists() and _sha256(target) == model.sha256):
            _download(name, model, target, quiet=quiet)
        _verified.add(target)
    return target


def _download(name: str, model: Model, target: Path, *, quiet: bool) -> None:
    if not quiet:
        print(f"  fetching model {name} ({target.name}) ...", flush=True)

    # Download to a sibling temp file and rename, so a cancelled or failed
    # fetch can never leave a half-written model behind for the next run.
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, suffix=".part")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        req = urllib.request.Request(model.url, headers={"User-Agent": "robocrop"})
        with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out)
        digest = _sha256(tmp)
        if digest != model.sha256:
            raise ModelUnavailable(
                f"download for {name!r} failed its checksum ({tmp.stat().st_size} "
                f"bytes, sha256 {digest[:12]}...); check your network or proxy"
            )
        tmp.replace(target)
    except OSError as exc:
        # URLError, but also timeouts and resets mid-transfer, which are not.
        raise ModelUnavailable(f"could not download {name!r} from {model.url}: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

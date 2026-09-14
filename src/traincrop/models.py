"""Download-once, cache-forever model files.

Detector weights are a few hundred kilobytes each and are not shipped with
OpenCV 5 or MediaPipe 1.x, so they are fetched on first use into
``~/.cache/traincrop/models`` (override with ``TRAINCROP_CACHE``).
"""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

_OPENCV_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
_OPENCV_DATA = "https://raw.githubusercontent.com/opencv/opencv/4.x/data"

#: Logical name -> (url, minimum plausible size in bytes).
#: The size acts as a cheap integrity check: an HTML error page or a truncated
#: download is far smaller than the real weights and must not be cached.
REGISTRY: dict[str, tuple[str, int]] = {
    "yunet": (
        f"{_OPENCV_ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        200_000,
    ),
    "haar_frontalface": (
        f"{_OPENCV_DATA}/haarcascades/haarcascade_frontalface_default.xml",
        500_000,
    ),
    "haar_profileface": (
        f"{_OPENCV_DATA}/haarcascades/haarcascade_profileface.xml",
        400_000,
    ),
    "yolox": (
        f"{_OPENCV_ZOO}/object_detection_yolox/object_detection_yolox_2022nov.onnx",
        20_000_000,
    ),
}


class ModelUnavailable(RuntimeError):
    """A model file could not be fetched and is not already cached."""


def cache_dir() -> Path:
    root = os.environ.get("TRAINCROP_CACHE")
    base = Path(root).expanduser() if root else Path.home() / ".cache" / "traincrop"
    path = base / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure(name: str, *, quiet: bool = False) -> Path:
    """Return a local path to ``name``, downloading it if necessary."""
    try:
        url, min_bytes = REGISTRY[name]
    except KeyError:
        raise ModelUnavailable(f"unknown model {name!r}") from None

    target = cache_dir() / Path(url).name
    if target.exists() and target.stat().st_size >= min_bytes:
        return target

    if not quiet:
        print(f"  fetching model {name} ({Path(url).name}) ...", flush=True)

    # Download to a sibling temp file and rename, so a cancelled or failed
    # fetch can never leave a half-written model behind for the next run.
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, suffix=".part")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "traincrop"})
        with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out)
        size = tmp.stat().st_size
        if size < min_bytes:
            raise ModelUnavailable(
                f"download for {name!r} was only {size} bytes "
                f"(expected >= {min_bytes}); check your network or proxy"
            )
        tmp.replace(target)
    except urllib.error.URLError as exc:
        raise ModelUnavailable(f"could not download {name!r} from {url}: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)

    return target

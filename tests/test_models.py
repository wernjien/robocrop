"""Model cache: checksums, and one download however many threads ask."""

import hashlib
import io
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest  # noqa: E402

from traincrop import models  # noqa: E402

PAYLOAD = b"not really a model, but it hashes"


@pytest.fixture
def fake_model(tmp_path, monkeypatch):
    """A registry entry whose pinned hash matches PAYLOAD, and a fake server."""
    monkeypatch.setenv("TRAINCROP_CACHE", str(tmp_path))
    monkeypatch.setattr(models, "_verified", set())
    monkeypatch.setitem(models.REGISTRY, "fake", models.Model(
        "https://example.invalid/fake.onnx", hashlib.sha256(PAYLOAD).hexdigest()
    ))
    served = {"body": PAYLOAD, "count": 0}

    def fake_urlopen(req, timeout):
        served["count"] += 1
        return io.BytesIO(served["body"])

    monkeypatch.setattr(models.urllib.request, "urlopen", fake_urlopen)
    return served


def test_a_download_that_matches_its_checksum_is_cached(fake_model):
    path = models.ensure("fake", quiet=True)

    assert path.read_bytes() == PAYLOAD
    assert models.ensure("fake", quiet=True) == path
    assert fake_model["count"] == 1


def test_a_download_that_fails_its_checksum_is_refused(fake_model):
    fake_model["body"] = b"<html>rate limited</html>"

    with pytest.raises(models.ModelUnavailable, match="checksum"):
        models.ensure("fake", quiet=True)
    assert not list(models.cache_dir().iterdir())   # nothing half-cached


def test_a_corrupt_cached_file_is_fetched_again(fake_model):
    (models.cache_dir() / "fake.onnx").write_bytes(b"truncated")

    assert models.ensure("fake", quiet=True).read_bytes() == PAYLOAD
    assert fake_model["count"] == 1


def test_concurrent_first_use_downloads_once(fake_model):
    threads = [threading.Thread(target=models.ensure, args=("fake",), kwargs={"quiet": True})
               for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert fake_model["count"] == 1


def test_a_network_error_mid_transfer_is_reported_cleanly(fake_model, monkeypatch):
    def timeout(req, timeout):
        raise TimeoutError("read timed out")

    monkeypatch.setattr(models.urllib.request, "urlopen", timeout)
    with pytest.raises(models.ModelUnavailable, match="timed out"):
        models.ensure("fake", quiet=True)


def test_every_registered_model_pins_a_sha256():
    for name, model in models.REGISTRY.items():
        assert len(model.sha256) == 64, name

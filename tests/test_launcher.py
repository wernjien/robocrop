"""Dependency synchronization and OpenCV repair in the POSIX launcher."""

import json
import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(os.name == "nt", reason="POSIX launcher")
@pytest.mark.parametrize("uninstall_fails", [False, True])
def test_launcher_reinstalls_shared_opencv_files_and_only_stamps_success(tmp_path, uninstall_fails):
    venv = tmp_path / "venv with spaces"
    python = venv / "bin" / "python"
    python.parent.mkdir(parents=True)
    log = tmp_path / "commands.jsonl"
    python.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['REVIEW_PIP_LOG'], 'a') as log:\n"
        "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if 'uninstall' in sys.argv and os.environ['REVIEW_UNINSTALL_FAILS'] == '1':\n"
        "    sys.exit(1)\n"
    )
    python.chmod(0o755)
    env = {**os.environ, "ROBOCROP_VENV": str(venv), "ROBOCROP_NO_VLM": "1",
           "ROBOCROP_SKIP_SYNC": "0", "REVIEW_PIP_LOG": str(log),
           "REVIEW_UNINSTALL_FAILS": str(int(uninstall_fails))}
    launcher = Path(__file__).resolve().parents[1] / "robocrop"
    result = subprocess.run(["bash", str(launcher), "--help"], env=env, capture_output=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls[0] == ["-m", "pip", "uninstall", "--yes", "opencv-python",
                        "opencv-python-headless", "opencv-contrib-python-headless",
                        "opencv-contrib-python"]
    if uninstall_fails:
        assert result.returncode == 1
        assert len(calls) == 1
        assert not (venv / ".robocrop-deps").exists()
    else:
        assert result.returncode == 0
        assert calls[1][:4] == ["-m", "pip", "install", "--quiet"]
        assert (venv / ".robocrop-deps").read_text().startswith("v3:core:")
        # Once synced, normal launches do not touch pip.
        result = subprocess.run(["bash", str(launcher), "--help"], env=env, capture_output=True)
        assert result.returncode == 0
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        assert len(calls) == 4
        assert calls[-1] == ["-m", "robocrop", "--help"]

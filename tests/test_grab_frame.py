import os
import subprocess
import sys
from pathlib import Path

import cv2
import pytest

from camera_mcp.config import CONFIG_ENV_VAR

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "grab_frame.py"
SAMPLE_VIDEO = REPO_ROOT / "tests" / "fixtures" / "sample.mp4"


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        "cameras:\n"
        "  - id: sample\n"
        "    name: Sample\n"
        f"    source: {{type: file, path: '{SAMPLE_VIDEO.as_posix()}', loop: true}}\n"
        "  - id: broken\n"
        "    name: Broken\n"
        "    source: {type: file, path: absent.mp4}\n",
        encoding="utf-8",
    )
    return path


def run_script(config_path: Path, cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        env={**os.environ, CONFIG_ENV_VAR: str(config_path)},
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_saves_frame_to_default_location(config_path: Path, tmp_path: Path) -> None:
    result = run_script(config_path, tmp_path, "--camera", "sample")

    assert result.returncode == 0, result.stderr
    image = cv2.imread(str(tmp_path / "recordings" / "sample.jpg"))
    assert image is not None
    assert image.shape == (240, 320, 3)


def test_saves_frame_to_given_path(config_path: Path, tmp_path: Path) -> None:
    target = tmp_path / "nested" / "out.jpg"

    result = run_script(config_path, tmp_path, "--camera", "sample", "--output", str(target))

    assert result.returncode == 0, result.stderr
    assert target.stat().st_size > 0


def test_unknown_camera_lists_available_cameras(config_path: Path, tmp_path: Path) -> None:
    result = run_script(config_path, tmp_path, "--camera", "nope")

    assert result.returncode == 1
    assert "broken, sample" in result.stderr
    assert "Traceback" not in result.stderr


def test_invalid_config_gives_clear_message(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("cameras: []\n", encoding="utf-8")

    result = run_script(config, tmp_path, "--camera", "sample")

    assert result.returncode == 1
    assert "Invalid configuration" in result.stderr
    assert "Traceback" not in result.stderr

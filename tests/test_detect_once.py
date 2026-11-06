import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest
from fakes import FakeDetector

from camera_mcp.config import CONFIG_ENV_VAR
from camera_mcp.detector import DetectorError
from camera_mcp.schemas import Detection

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
FIXTURES = REPO_ROOT / "tests" / "fixtures"


@pytest.fixture
def detect_once(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("detect_once", SCRIPTS / "detect_once.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        "cameras:\n"
        "  - id: sample\n"
        "    name: Sample\n"
        f"    source: {{type: file, path: '{(FIXTURES / 'sample.mp4').as_posix()}', loop: true}}\n"
        "zones:\n"
        "  sample:\n"
        "    - {name: left, polygon: [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(CONFIG_ENV_VAR, str(path))
    return path


PERSON = Detection(label="person", confidence=0.9, bbox=(0.1, 0.2, 0.3, 0.4))


def fake_factory(_config: object) -> FakeDetector:
    return FakeDetector([PERSON])


@pytest.mark.usefixtures("config_path")
def test_image_gives_json_on_stdout(detect_once: ModuleType, capsys) -> None:
    code = detect_once.run(["--image", str(FIXTURES / "sample.png")], fake_factory)

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["camera_id"] == "image"
    assert payload["frame_size"] == [320, 240]
    assert payload["counts"] == {"person": 1}


@pytest.mark.usefixtures("config_path")
def test_camera_gives_json_on_stdout(detect_once: ModuleType, capsys) -> None:
    code = detect_once.run(["--camera", "sample"], fake_factory)

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["camera_id"] == "sample"
    assert payload["frame_size"] == [320, 240]
    assert payload["detections"][0]["zones"] == ["left"]


@pytest.mark.usefixtures("config_path")
def test_image_file_has_no_zones(detect_once: ModuleType, capsys) -> None:
    detect_once.run(["--image", str(FIXTURES / "sample.png")], fake_factory)

    assert json.loads(capsys.readouterr().out)["detections"][0]["zones"] == []


@pytest.mark.usefixtures("config_path")
def test_missing_image_reports_on_stderr_only(detect_once: ModuleType, capsys, tmp_path) -> None:
    code = detect_once.run(["--image", str(tmp_path / "absent.jpg")], fake_factory)

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "could not read image file" in captured.err


@pytest.mark.usefixtures("config_path")
def test_unknown_camera_reports_available_cameras(detect_once: ModuleType, capsys) -> None:
    code = detect_once.run(["--camera", "nope"], fake_factory)

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "Available cameras: sample" in captured.err


@pytest.mark.usefixtures("config_path")
def test_detector_failure_reports_on_stderr_only(detect_once: ModuleType, capsys) -> None:
    def failing_factory(_config: object) -> FakeDetector:
        raise DetectorError("model is broken")

    code = detect_once.run(["--camera", "sample"], failing_factory)

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "model is broken" in captured.err


@pytest.mark.usefixtures("config_path")
def test_model_loading_does_not_count_towards_frame_age(detect_once: ModuleType, capsys) -> None:
    load_time_s = 0.5

    def slow_factory(_config: object) -> FakeDetector:
        time.sleep(load_time_s)
        return FakeDetector()

    detect_once.run(["--image", str(FIXTURES / "sample.png")], slow_factory)

    assert json.loads(capsys.readouterr().out)["frame_age_s"] < load_time_s


@pytest.mark.parametrize("argv", [[], ["--camera", "a", "--image", "b.jpg"]])
def test_exactly_one_target_is_required(detect_once: ModuleType, argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        detect_once.run(argv, fake_factory)


def run_script(config: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / "detect_once.py"), *args],
        env={**os.environ, CONFIG_ENV_VAR: str(config)},
        cwd=REPO_ROOT,  # The model file is kept in the ignored project root.
        capture_output=True,
        encoding="utf-8",  # Ultralytics writes UTF-8 progress bars to stderr.
        errors="replace",
        timeout=600,
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    "target",
    [["--camera", "sample"], ["--image", str(FIXTURES / "sample.png")]],
    ids=["read-only-camera-frame", "image-file"],
)
def test_real_model_writes_only_json_to_stdout(config_path: Path, target: list[str]) -> None:
    """Checks the mechanics only: synthetic shapes say nothing about detection quality."""
    result = run_script(config_path, *target)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["model"] == "yolo26n.pt"
    assert payload["device"] in {"cpu", "cuda"}
    assert payload["inference_ms"] > 0
    assert payload["frame_size"] == [320, 240]

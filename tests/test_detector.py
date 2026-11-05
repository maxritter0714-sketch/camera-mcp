import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
from fakes import FakeDetector

import camera_mcp.detector as detector_module
from camera_mcp.camera_manager import Frame
from camera_mcp.config import DetectorConfig
from camera_mcp.detector import DetectorError, UltralyticsDetector, detect_in_frame
from camera_mcp.schemas import Detection


def make_frame(width: int = 64, height: int = 48) -> Frame:
    return Frame(
        image=np.zeros((height, width, 3), dtype=np.uint8),
        captured_at=datetime.now().astimezone(),
        captured_monotonic=time.monotonic(),
    )


PERSON = Detection(label="person", confidence=0.91234, bbox=(0.123456, 0.08, 0.41, 0.97))
CUP_LOW = Detection(label="cup", confidence=0.5, bbox=(0.7, 0.55, 0.76, 0.66))
CUP_HIGH = Detection(label="cup", confidence=0.95, bbox=(0.2, 0.55, 0.26, 0.66))


def run(detector: FakeDetector, **kwargs: object):
    return detect_in_frame(detector, make_frame(), "cam", default_confidence=0.4, **kwargs)


def test_response_contains_all_fields_of_the_specified_format() -> None:
    response = run(FakeDetector([PERSON]))

    payload = json.loads(response.model_dump_json())

    assert set(payload) == {
        "camera_id",
        "timestamp",
        "frame_age_s",
        "frame_size",
        "model",
        "device",
        "inference_ms",
        "counts",
        "detections",
    }
    assert payload["camera_id"] == "cam"
    assert payload["frame_size"] == [64, 48]
    assert payload["model"] == "fake.pt"
    assert payload["device"] == "cpu"
    assert payload["inference_ms"] == 12.3
    assert payload["counts"] == {"person": 1}
    assert payload["detections"] == [
        {
            "label": "person",
            "confidence": 0.912,
            "bbox": [0.1235, 0.08, 0.41, 0.97],
            "zones": [],
        }
    ]


def test_timestamp_is_iso_8601_with_timezone() -> None:
    payload = json.loads(run(FakeDetector()).model_dump_json())

    assert datetime.fromisoformat(payload["timestamp"]).utcoffset() is not None


def test_detections_are_counted_and_sorted_by_confidence() -> None:
    response = run(FakeDetector([CUP_LOW, PERSON, CUP_HIGH]))

    assert response.counts == {"cup": 2, "person": 1}
    assert [d.confidence for d in response.detections] == [0.95, 0.912, 0.5]


def test_no_detections_gives_empty_result() -> None:
    response = run(FakeDetector())

    assert response.counts == {}
    assert response.detections == []


def test_classes_filter_detections_and_counts() -> None:
    response = run(FakeDetector([PERSON, CUP_HIGH]), classes=["cup"])

    assert response.counts == {"cup": 1}
    assert [d.label for d in response.detections] == ["cup"]


def test_empty_classes_mean_no_filter() -> None:
    assert run(FakeDetector([PERSON]), classes=[]).counts == {"person": 1}


def test_unknown_class_lists_valid_classes_without_running_inference() -> None:
    detector = FakeDetector()

    with pytest.raises(DetectorError, match=r"Unknown classes: unicorn\. Valid classes: person"):
        run(detector, classes=["person", "unicorn"])

    assert detector.calls == []


def test_list_of_valid_classes_is_shortened() -> None:
    detector = FakeDetector()
    detector.class_names = tuple(f"class{i}" for i in range(80))

    with pytest.raises(DetectorError) as excinfo:
        run(detector, classes=["unicorn"])

    assert "class19," in str(excinfo.value)
    assert "class20" not in str(excinfo.value)
    assert "80 in total" in str(excinfo.value)


def test_default_confidence_is_used_unless_overridden() -> None:
    detector = FakeDetector()

    run(detector)
    run(detector, min_confidence=0.7)
    run(detector, min_confidence=0.0)

    assert detector.calls == [0.4, 0.7, 0.0]


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_invalid_confidence_is_rejected(value: float) -> None:
    detector = FakeDetector()

    with pytest.raises(DetectorError, match="min_confidence"):
        run(detector, min_confidence=value)

    assert detector.calls == []


class FakeTensor:
    def __init__(self, values: list) -> None:
        self._values = values

    def tolist(self) -> list:
        return self._values


class FakeBoxes:
    def __init__(self, xyxyn: list, conf: list, cls: list) -> None:
        self.xyxyn = FakeTensor(xyxyn)
        self.conf = FakeTensor(conf)
        self.cls = FakeTensor(cls)


class FakeResult:
    def __init__(self, boxes: FakeBoxes) -> None:
        self.boxes = boxes


class FakeModel:
    names = {0: "person", 1: "cup"}

    def __init__(self, boxes: FakeBoxes | None = None) -> None:
        self.boxes = boxes or FakeBoxes([], [], [])
        self.kwargs: dict = {}
        self.active = 0
        self.max_active = 0
        self.guard = threading.Lock()

    def predict(self, image: np.ndarray, **kwargs: object) -> list[FakeResult]:
        self.kwargs = kwargs
        with self.guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.01)
        with self.guard:
            self.active -= 1
        return [FakeResult(self.boxes)]


@pytest.fixture
def no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detector_module, "_cuda_available", lambda: False)


def make_detector(model: FakeModel, **config: object) -> UltralyticsDetector:
    return UltralyticsDetector(DetectorConfig(**config), load_model=lambda _name: model)


@pytest.mark.usefixtures("no_cuda")
def test_ultralytics_results_are_converted() -> None:
    boxes = FakeBoxes([[0.1, 0.2, 0.3, 0.4]], [0.9], [1.0])
    detector = make_detector(FakeModel(boxes))

    output = detector.detect(make_frame().image, 0.5)

    assert output.detections == [Detection(label="cup", confidence=0.9, bbox=(0.1, 0.2, 0.3, 0.4))]
    assert output.inference_ms > 0
    assert detector.class_names == ("person", "cup")


@pytest.mark.usefixtures("no_cuda")
def test_coordinates_are_clipped_to_the_image() -> None:
    boxes = FakeBoxes([[-0.01, 0.2, 1.02, 0.4]], [0.9], [0.0])

    output = make_detector(FakeModel(boxes)).detect(make_frame().image, 0.5)

    assert output.detections[0].bbox == (0.0, 0.2, 1.0, 0.4)


@pytest.mark.usefixtures("no_cuda")
def test_prediction_is_silent_and_uses_configuration() -> None:
    model = FakeModel()

    make_detector(model, imgsz=320).detect(make_frame().image, 0.25)

    assert model.kwargs == {"conf": 0.25, "imgsz": 320, "device": "cpu", "verbose": False}


@pytest.mark.usefixtures("no_cuda")
def test_concurrent_calls_are_serialized() -> None:
    model = FakeModel()
    detector = make_detector(model)
    image = make_frame().image

    threads = [threading.Thread(target=detector.detect, args=(image, 0.5)) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert model.max_active == 1


@pytest.mark.parametrize(
    ("requested", "available", "expected"),
    [
        ("auto", True, "cuda"),
        ("auto", False, "cpu"),
        ("cuda", True, "cuda"),
        ("cpu", True, "cpu"),
        ("cpu", False, "cpu"),
    ],
)
def test_device_selection(
    monkeypatch: pytest.MonkeyPatch, requested: str, available: bool, expected: str
) -> None:
    monkeypatch.setattr(detector_module, "_cuda_available", lambda: available)

    assert make_detector(FakeModel(), device=requested).device == expected


def test_cuda_without_gpu_gives_clear_error(no_cuda: None) -> None:
    with pytest.raises(DetectorError, match="CUDA is not available"):
        make_detector(FakeModel(), device="cuda")


@pytest.mark.usefixtures("no_cuda")
def test_model_load_failure_gives_clear_error() -> None:
    def failing_load(_name: str) -> None:
        raise FileNotFoundError("secret internal path")

    with pytest.raises(DetectorError, match=r"'broken\.pt' \(FileNotFoundError\)") as excinfo:
        UltralyticsDetector(DetectorConfig(model="broken.pt"), load_model=failing_load)

    assert "secret internal path" not in str(excinfo.value)


def test_first_ultralytics_import_keeps_stdout_clean_and_disables_network_features(
    tmp_path: Path,
) -> None:
    # A fresh config directory triggers the settings file message of a first run.
    code = (
        "from camera_mcp.detector import import_yolo\n"
        "import_yolo()\n"
        "from ultralytics.utils import AUTOINSTALL, LOGGER, ONLINE\n"
        "from ultralytics.utils.events import events\n"
        "assert ONLINE is False\n"
        "assert AUTOINSTALL is False\n"
        "assert events.enabled is False\n"
        "import sys\n"
        "assert all(h.stream is sys.stderr for h in LOGGER.handlers)\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "YOLO_CONFIG_DIR": str(tmp_path)},
        capture_output=True,
        encoding="utf-8",  # Ultralytics writes UTF-8 symbols to stderr.
        errors="replace",
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""

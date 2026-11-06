import importlib.util
import json
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pytest
from fakes import COCO_SUBSET, FakeDetector

from camera_mcp.config import CONFIG_ENV_VAR
from camera_mcp.schemas import Detection

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
SAMPLE_VIDEO = REPO_ROOT / "tests" / "fixtures" / "sample.mp4"
BOX = (0.1, 0.2, 0.3, 0.4)


@pytest.fixture
def ev(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    path = SCRIPTS / "evaluate_detection.py"
    spec = importlib.util.spec_from_file_location("evaluate_detection", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def det(label: str, confidence: float) -> Detection:
    return Detection(label=label, confidence=confidence, bbox=BOX)


def write_csv(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8-sig")
    return path


def test_ground_truth_reads_excel_csv_with_blanks_as_zero(ev: ModuleType, tmp_path: Path) -> None:
    path = write_csv(
        tmp_path / "counts.csv",
        "datei;person;car;notiz\r\na.jpg;2;;\r\nb.jpg;;;4 wrist watches\r\n",
    )

    truths = ev.load_ground_truth(path, COCO_SUBSET)

    assert truths[0] == ev.GroundTruth("a.jpg", {"person": 2, "car": 0}, "")
    assert truths[1].note == "4 wrist watches"
    assert truths[1].is_negative
    assert not truths[0].is_negative


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("datei;unicorn\na.jpg;1\n", "not classes of the model: unicorn"),
        ("datei;person\na.jpg;zwei\n", "line 2: 'zwei' for person"),
        ("datei;person\na.jpg;-1\n", "not a whole number"),
        ("datei;person\n;1\n", "line 2: no image name"),
        ("person\n1\n", "needs a 'datei' column"),
        ("datei;person\n", "contains no images"),
    ],
)
def test_invalid_ground_truth_is_rejected(
    ev: ModuleType, tmp_path: Path, text: str, message: str
) -> None:
    path = write_csv(tmp_path / "counts.csv", text)

    with pytest.raises(ev.EvaluationError, match=message):
        ev.load_ground_truth(path, COCO_SUBSET)


def test_evaluate_counts_missed_extra_and_other_classes(ev: ModuleType) -> None:
    truths = [
        ev.GroundTruth("a.jpg", {"person": 2, "car": 0}, ""),
        ev.GroundTruth("b.jpg", {"person": 0, "car": 1}, ""),
        ev.GroundTruth("c.jpg", {"person": 0, "car": 0}, "4 wrist watches"),
    ]
    detections = {
        "a.jpg": [det("person", 0.9), det("person", 0.3)],
        "b.jpg": [det("car", 0.8), det("car", 0.45), det("dog", 0.7)],
        "c.jpg": [det("car", 0.3)],
    }

    low = ev.evaluate(truths, detections, 0.25, 0.25)
    high = ev.evaluate(truths, detections, 0.5, 0.25)

    # At 0.25: a exact, b and c one car too many each.
    assert (low.images_exact, low.images_total) == (1, 3)
    assert (low.pairs_exact, low.pairs_total) == (1, 3)
    assert low.mean_abs_error == pytest.approx(2 / 3)
    assert (low.truth_total, low.detected_total, low.missed, low.extra) == (3, 5, 0, 2)
    assert low.outside_classes == 1
    # At 0.5: a misses one person, b and c exact. The compared pairs stay the same.
    assert (high.images_exact, high.missed, high.extra) == (2, 1, 0)
    assert (high.pairs_exact, high.pairs_total) == (2, 3)


def test_evaluate_classes_lists_only_classes_that_occur(ev: ModuleType) -> None:
    truths = [ev.GroundTruth("a.jpg", {"person": 1, "car": 0, "cup": 0}, "")]
    detections = {"a.jpg": [det("car", 0.9)]}

    result = ev.evaluate_classes(truths, detections, 0.4)

    assert result == [
        ev.ClassResult("person", 1, 0, 1, 0),
        ev.ClassResult("car", 0, 1, 0, 1),
    ]


def test_video_is_sampled_once_per_second(ev: ModuleType) -> None:
    capture = cv2.VideoCapture(str(SAMPLE_VIDEO))
    seconds = capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    detector = FakeDetector([det("person", 0.9)])

    timeline = ev.sample_video(detector, SAMPLE_VIDEO, 0.4)

    assert [second for second, _ in timeline] == list(range(len(timeline)))
    assert len(timeline) in {int(seconds), int(seconds) + 1}
    assert all(found == {"person": 1} for _, found in timeline)


def test_unreadable_video_is_reported(ev: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(ev.EvaluationError, match="Cannot open video"):
        ev.sample_video(FakeDetector(), tmp_path / "absent.mp4", 0.4)


@pytest.fixture
def eval_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config = tmp_path / "config.yaml"
    config.write_text(
        "cameras:\n"
        f"  - {{id: s, name: S, source: {{type: file, path: '{SAMPLE_VIDEO.as_posix()}'}}}}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(CONFIG_ENV_VAR, str(config))
    folder = tmp_path / "eval"
    (folder / "images").mkdir(parents=True)
    for name in ("a.jpg", "b.jpg"):
        ok, data = cv2.imencode(".jpg", np.zeros((40, 60, 3), dtype=np.uint8))
        assert ok
        (folder / "images" / name).write_bytes(data.tobytes())
    write_csv(folder / "counts.csv", "datei;person;car;notiz\na.jpg;1;;\nb.jpg;;;1 helicopter\n")
    return folder


def test_run_prints_report_and_writes_raw_results(
    ev: ModuleType, eval_dir: Path, tmp_path: Path, capsys
) -> None:
    raw_path = tmp_path / "raw.json"
    detector = FakeDetector([det("person", 0.9), det("dog", 0.3)])

    code = ev.run(
        ["--dir", str(eval_dir), "--video", str(SAMPLE_VIDEO), "--json", str(raw_path)],
        lambda _config: detector,
    )

    out = capsys.readouterr().out
    assert code == 0
    assert out.isascii()  # The Windows console cannot print every character.
    assert "| 0.25 | 1/2 (50 %)" in out  # b.jpg gets a person that is not there.
    assert "| b.jpg | - | person 1 | 1 helicopter |" in out
    assert "### Video, one frame per second" in out
    assert detector.calls[:2] == [0.25, 0.25]  # One inference per image.
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    assert [t["threshold"] for t in raw["thresholds"]] == [0.25, 0.4, 0.5, 0.6]
    assert raw["images"]["b.jpg"]["note"] == "1 helicopter"
    assert raw["video"]


def test_missing_image_is_reported_on_stderr_only(ev: ModuleType, eval_dir: Path, capsys) -> None:
    (eval_dir / "images" / "b.jpg").unlink()

    code = ev.run(["--dir", str(eval_dir)], lambda _config: FakeDetector())

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "Cannot decode image" in captured.err or "Cannot read image" in captured.err

"""Compare detected object counts with hand counts for docs/EVALUATION.md (KONZEPT.md 10.3).

Input is a folder with ``images/`` and ``counts.csv``: one row per image, one column per counted
COCO class (empty means 0) and a ``notiz`` column for objects outside those classes. Each image
is run through the detector once at the lowest threshold; higher thresholds filter that result.
Only counts are compared, not box positions.

Optionally, ``--video`` samples one frame per second of a video and lists the counts over time,
next to the events noted by hand in ``--events``.
"""

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from camera_mcp.config import ConfigError, DetectorConfig, load_config, resolve_config_path
from camera_mcp.detector import Detector, DetectorError, UltralyticsDetector
from camera_mcp.logging_setup import setup_logging
from camera_mcp.schemas import Detection

THRESHOLDS = (0.25, 0.4, 0.5, 0.6)
NAME_COLUMN = "datei"
NOTE_COLUMN = "notiz"
CSV_DELIMITER = ";"


class EvaluationError(Exception):
    """The input cannot be evaluated; the message says why."""


@dataclass(frozen=True)
class GroundTruth:
    image: str
    counts: dict[str, int]  # every counted class, 0 included
    note: str

    @property
    def is_negative(self) -> bool:
        """No object of the counted classes; only objects outside them, if any."""
        return not any(self.counts.values())


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    images_exact: int
    images_total: int
    pairs_exact: int
    pairs_total: int
    mean_abs_error: float
    truth_total: int
    detected_total: int
    missed: int
    extra: int
    outside_classes: int


@dataclass(frozen=True)
class ClassResult:
    label: str
    truth: int
    detected: int
    missed: int
    extra: int


def load_ground_truth(path: Path, valid_classes: Sequence[str]) -> list[GroundTruth]:
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise EvaluationError(f"Cannot read {path}.") from exc
    reader = csv.DictReader(text.splitlines(), delimiter=CSV_DELIMITER)
    fields = reader.fieldnames or []
    if NAME_COLUMN not in fields:
        raise EvaluationError(f"{path} needs a '{NAME_COLUMN}' column.")
    classes = [f for f in fields if f not in (NAME_COLUMN, NOTE_COLUMN)]
    unknown = [c for c in classes if c not in valid_classes]
    if unknown:
        raise EvaluationError(f"Columns are not classes of the model: {', '.join(unknown)}.")

    truths = []
    for line, row in enumerate(reader, start=2):
        name = (row.get(NAME_COLUMN) or "").strip()
        if not name:
            raise EvaluationError(f"{path}, line {line}: no image name.")
        counts = {}
        for label in classes:
            value = (row.get(label) or "").strip()
            if value and not (value.isdigit()):
                raise EvaluationError(
                    f"{path}, line {line}: '{value}' for {label} is not a whole number >= 0."
                )
            counts[label] = int(value) if value else 0
        truths.append(GroundTruth(name, counts, (row.get(NOTE_COLUMN) or "").strip()))
    if not truths:
        raise EvaluationError(f"{path} contains no images.")
    return truths


def read_image(path: Path) -> np.ndarray:
    # Decoded from bytes: cv2.imread cannot handle non-ASCII paths on Windows.
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise EvaluationError(f"Cannot read image {path}.") from exc
    image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    if image is None:
        raise EvaluationError(f"Cannot decode image {path}.")
    return image


def above(detections: Sequence[Detection], threshold: float) -> Counter[str]:
    return Counter(d.label for d in detections if d.confidence >= threshold)


def evaluate(
    truths: Sequence[GroundTruth],
    detections: Mapping[str, Sequence[Detection]],
    threshold: float,
    pair_threshold: float,
) -> ThresholdResult:
    """Count-based comparison; missed and extra objects are differences of counts per class.

    Pairs of image and class are compared where the hand count or the detection at
    ``pair_threshold`` is above 0. Pairs where both are always 0 would inflate the accuracy, and
    a fixed set keeps the results of all thresholds comparable.
    """
    images_exact = pairs_exact = pairs_total = 0
    errors: list[int] = []
    truth_total = detected_total = missed = extra = outside = 0
    for truth in truths:
        found = above(detections[truth.image], threshold)
        candidates = above(detections[truth.image], pair_threshold)
        outside += sum(n for label, n in found.items() if label not in truth.counts)
        image_exact = True
        for label, expected in truth.counts.items():
            got = found.get(label, 0)
            truth_total += expected
            detected_total += got
            missed += max(expected - got, 0)
            extra += max(got - expected, 0)
            if expected or candidates.get(label, 0):
                pairs_total += 1
                pairs_exact += expected == got
                errors.append(abs(expected - got))
            image_exact &= expected == got
        images_exact += image_exact
    return ThresholdResult(
        threshold=threshold,
        images_exact=images_exact,
        images_total=len(truths),
        pairs_exact=pairs_exact,
        pairs_total=pairs_total,
        mean_abs_error=statistics.fmean(errors) if errors else 0.0,
        truth_total=truth_total,
        detected_total=detected_total,
        missed=missed,
        extra=extra,
        outside_classes=outside,
    )


def evaluate_classes(
    truths: Sequence[GroundTruth],
    detections: Mapping[str, Sequence[Detection]],
    threshold: float,
) -> list[ClassResult]:
    totals: dict[str, list[int]] = {label: [0, 0, 0, 0] for label in truths[0].counts}
    for truth in truths:
        found = above(detections[truth.image], threshold)
        for label, expected in truth.counts.items():
            got = found.get(label, 0)
            entry = totals[label]
            entry[0] += expected
            entry[1] += got
            entry[2] += max(expected - got, 0)
            entry[3] += max(got - expected, 0)
    return [
        ClassResult(label, *values) for label, values in totals.items() if values[0] or values[1]
    ]


def sample_video(
    detector: Detector, path: Path, threshold: float
) -> list[tuple[int, Counter[str]]]:
    """Counts on one frame per second."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise EvaluationError(f"Cannot open video {path}.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not fps or fps <= 0 or frames <= 0:
            raise EvaluationError(f"Video {path} reports no frame rate or length.")
        timeline = []
        for second in range(int(frames / fps) + 1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, min(round(second * fps), frames - 1))
            ok, frame = capture.read()
            if not ok:
                break
            found = Counter(d.label for d in detector.detect(frame, threshold).detections)
            timeline.append((second, found))
        return timeline
    finally:
        capture.release()


def _summary(counts: Mapping[str, int]) -> str:
    parts = [f"{label} {n}" for label, n in sorted(counts.items()) if n]
    return ", ".join(parts) if parts else "-"


def _percent(part: int, whole: int) -> str:
    return f"{part}/{whole} ({100 * part / whole:.0f} %)" if whole else "-"


def render_markdown(
    model: str,
    device: str,
    default_threshold: float,
    truths: Sequence[GroundTruth],
    detections: Mapping[str, Sequence[Detection]],
    results: Sequence[ThresholdResult],
    timeline: Sequence[tuple[int, Counter[str]]] | None,
    events: str | None,
) -> str:
    negatives = [t for t in truths if t.is_negative]
    lines = [
        f"## Detection quality ({model}, {device})",
        "",
        f"- Images: {len(truths)}, of which {len(negatives)} contain none of the counted classes",
        f"- Counted classes: {', '.join(truths[0].counts)}",
        "- Compared: object counts per image and class, not box positions",
        "",
        "### By confidence threshold",
        "",
        "| Threshold | Images exact | Class counts exact | Mean abs. error | Hand count "
        "| Detected | Missed | Extra | Other classes |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.threshold:.2f} | {_percent(r.images_exact, r.images_total)} "
            f"| {_percent(r.pairs_exact, r.pairs_total)} | {r.mean_abs_error:.2f} "
            f"| {r.truth_total} | {r.detected_total} | {r.missed} | {r.extra} "
            f"| {r.outside_classes} |"
        )
    lines += [
        "",
        "Class counts exact: the same pairs of image and class for every threshold, namely those "
        f"where the hand count or the detection at {min(THRESHOLDS):.2f} is above 0. "
        "Missed/extra: differences of counts. Other classes: detections of classes that were "
        "not counted by hand.",
        "",
        f"### By class (threshold {default_threshold:.2f})",
        "",
        "| Class | Hand count | Detected | Missed | Extra |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in evaluate_classes(truths, detections, default_threshold):
        lines.append(f"| {c.label} | {c.truth} | {c.detected} | {c.missed} | {c.extra} |")
    lines += [
        "",
        f"### Images with differences (threshold {default_threshold:.2f})",
        "",
        "| Image | Hand count | Detected | Note |",
        "|---|---|---|---|",
    ]
    for truth in truths:
        found = above(detections[truth.image], default_threshold)
        if any(found.get(label, 0) != n for label, n in truth.counts.items()) or any(
            label not in truth.counts for label in found
        ):
            lines.append(
                f"| {truth.image} | {_summary(truth.counts)} | {_summary(found)} "
                f"| {truth.note or '-'} |"
            )
    if timeline is not None:
        labels = sorted({label for _, found in timeline for label in found})
        lines += [
            "",
            f"### Video, one frame per second (threshold {default_threshold:.2f})",
            "",
            "| Second | " + " | ".join(labels) + " |",
            "|---:|" + "---:|" * len(labels),
        ]
        for second, found in timeline:
            lines.append(f"| {second} | " + " | ".join(str(found.get(x, 0)) for x in labels) + " |")
        if events:
            lines += ["", "Events noted by hand:", "", "```", events.strip(), "```"]
    return "\n".join(lines)


def run(argv: list[str], make_detector: Callable[[DetectorConfig], Detector]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dir", type=Path, default=Path("recordings/eval"), help="folder with images/, counts.csv"
    )
    parser.add_argument("--video", type=Path, help="video to sample one frame per second from")
    parser.add_argument("--events", type=Path, help="text file with events noted for the video")
    parser.add_argument("--json", type=Path, help="also write the raw results to this file")
    args = parser.parse_args(argv)

    setup_logging()
    try:
        config = load_config(resolve_config_path())
        detector = make_detector(config.detector)
        truths = load_ground_truth(args.dir / "counts.csv", detector.class_names)
        detections = {}
        for truth in truths:
            image = read_image(args.dir / "images" / truth.image)
            detections[truth.image] = detector.detect(image, min(THRESHOLDS)).detections
        results = [evaluate(truths, detections, t, min(THRESHOLDS)) for t in THRESHOLDS]
        default = config.detector.default_confidence
        timeline = sample_video(detector, args.video, default) if args.video else None
        events = args.events.read_text(encoding="utf-8") if args.events else None
    except (ConfigError, DetectorError, EvaluationError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1

    print(
        render_markdown(
            detector.model_name,
            detector.device,
            default,
            truths,
            detections,
            results,
            timeline,
            events,
        )
    )
    if args.json:
        raw = {
            "model": detector.model_name,
            "device": detector.device,
            "thresholds": [asdict(r) for r in results],
            "images": {
                t.image: {
                    "hand": t.counts,
                    "note": t.note,
                    "detections": [d.model_dump() for d in detections[t.image]],
                }
                for t in truths
            },
            "video": None
            if timeline is None
            else [{"second": s, "counts": dict(found)} for s, found in timeline],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:], UltralyticsDetector))

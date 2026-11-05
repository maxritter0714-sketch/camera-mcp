"""Test doubles shared between test modules."""

from collections.abc import Sequence

from camera_mcp.detector import DetectorOutput
from camera_mcp.schemas import Detection
from camera_mcp.sources import Image

COCO_SUBSET = ("person", "bicycle", "car", "dog", "cup", "cell phone")


class FakeDetector:
    """Returns fixed detections and applies the confidence threshold like a real model."""

    model_name = "fake.pt"
    device = "cpu"
    class_names: Sequence[str] = COCO_SUBSET

    def __init__(self, detections: list[Detection] | None = None) -> None:
        self.detections = detections or []
        self.calls: list[float] = []

    def detect(self, image: Image, min_confidence: float) -> DetectorOutput:
        self.calls.append(min_confidence)
        kept = [d for d in self.detections if d.confidence >= min_confidence]
        return DetectorOutput(detections=kept, inference_ms=12.345)

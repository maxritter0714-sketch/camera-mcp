"""Object detection: the Detector protocol, its Ultralytics implementation and shared logic."""

import contextlib
import os
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from camera_mcp.camera_manager import Frame
from camera_mcp.config import DetectorConfig, ZoneConfig
from camera_mcp.schemas import Detection, DetectionResponse
from camera_mcp.sources import Image
from camera_mcp.zones import zones_for

MAX_LISTED_CLASSES = 20

# Must be set before the first ultralytics import, which reads them once:
# - YOLO_OFFLINE stops the usage analytics sent on every prediction; downloading weights still
#   works. Settings are not touched, as they would change the user's global Ultralytics file.
# - YOLO_AUTOINSTALL stops ultralytics from installing packages into the environment.
# - YOLO_VERBOSE limits its logger to errors.
_ULTRALYTICS_ENV = {
    "YOLO_OFFLINE": "true",
    "YOLO_AUTOINSTALL": "false",
    "YOLO_VERBOSE": "false",
}


class DetectorError(Exception):
    """A detection request or the detector itself failed. Written for the calling model."""


@dataclass(frozen=True)
class DetectorOutput:
    detections: list[Detection]
    inference_ms: float


class Detector(Protocol):
    """A replaceable object detector. Must be safe to call from several threads."""

    @property
    def model_name(self) -> str: ...

    @property
    def device(self) -> str: ...

    @property
    def class_names(self) -> Sequence[str]: ...

    def detect(self, image: Image, min_confidence: float) -> DetectorOutput:
        """Return all objects in a BGR image with at least the given confidence."""
        ...


def import_yolo() -> Any:
    """Import the YOLO class with its output sent to stderr.

    The ultralytics logger binds to ``sys.stdout`` during import and announces the creation of
    its settings file on a fresh machine, which would corrupt the MCP protocol channel.
    """
    os.environ.update(_ULTRALYTICS_ENV)
    with contextlib.redirect_stdout(sys.stderr):
        from ultralytics import YOLO
    return YOLO


def _load_model(model_name: str) -> Any:
    yolo = import_yolo()
    # The weights download reports its progress on sys.stdout as well.
    with contextlib.redirect_stdout(sys.stderr):
        return yolo(model_name)


def _cuda_available() -> bool:
    import torch

    return torch.cuda.is_available()


def _resolve_device(requested: str, cuda_available: bool) -> str:
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    if requested == "cuda" and not cuda_available:
        raise DetectorError(
            "Detector device 'cuda' is configured, but CUDA is not available. "
            "Set detector.device to 'auto' or 'cpu', or install a CUDA build of PyTorch."
        )
    return requested


class UltralyticsDetector:
    """YOLO model loaded once. Calls are serialized, as GPU models need not be thread-safe."""

    def __init__(
        self, config: DetectorConfig, load_model: Callable[[str], Any] = _load_model
    ) -> None:
        self._model_name = config.model
        self._imgsz = config.imgsz
        self._device = _resolve_device(config.device, _cuda_available())
        self._lock = threading.Lock()
        try:
            self._model = load_model(config.model)
        except Exception as exc:
            # Library boundary: download, file and format problems surface as many exception
            # types, and any of them means the same to the caller.
            raise DetectorError(
                f"Could not load detector model '{config.model}' ({type(exc).__name__}). "
                "Check the model name and, for the first download, the internet connection."
            ) from exc
        self._class_names = tuple(self._model.names.values())

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device

    @property
    def class_names(self) -> Sequence[str]:
        return self._class_names

    def detect(self, image: Image, min_confidence: float) -> DetectorOutput:
        with self._lock:
            started = time.perf_counter()
            results = self._model.predict(
                image,
                conf=min_confidence,
                imgsz=self._imgsz,
                device=self._device,
                verbose=False,
            )
            inference_ms = (time.perf_counter() - started) * 1000
        return DetectorOutput(_to_detections(results[0], self._model.names), inference_ms)


def _to_detections(result: Any, names: dict[int, str]) -> list[Detection]:
    boxes = result.boxes
    return [
        Detection(
            label=names[int(class_id)],
            confidence=confidence,
            bbox=tuple(min(max(value, 0.0), 1.0) for value in bbox),
        )
        for bbox, confidence, class_id in zip(
            boxes.xyxyn.tolist(), boxes.conf.tolist(), boxes.cls.tolist(), strict=True
        )
    ]


def _check_request(
    detector: Detector, classes: Sequence[str] | None, min_confidence: float
) -> None:
    """Reject bad requests before spending time on inference."""
    if not 0.0 <= min_confidence <= 1.0:
        raise DetectorError(f"min_confidence must be between 0 and 1, got {min_confidence}.")
    valid = detector.class_names
    unknown = [name for name in classes or () if name not in valid]
    if unknown:
        listed = ", ".join(valid[:MAX_LISTED_CLASSES])
        suffix = f", ... ({len(valid)} in total)" if len(valid) > MAX_LISTED_CLASSES else ""
        raise DetectorError(
            f"Unknown classes: {', '.join(unknown)}. Valid classes: {listed}{suffix}."
        )


def _finished(detection: Detection, zones: Sequence[ZoneConfig]) -> Detection:
    """Round the values and assign zones, based on the rounded box the caller sees."""
    bbox = tuple(round(value, 4) for value in detection.bbox)
    return detection.model_copy(
        update={
            "confidence": round(detection.confidence, 3),
            "bbox": bbox,
            "zones": zones_for(bbox, zones),
        }
    )


def detect_in_frame(
    detector: Detector,
    frame: Frame,
    camera_id: str,
    default_confidence: float,
    classes: Sequence[str] | None = None,
    min_confidence: float | None = None,
    zones: Sequence[ZoneConfig] = (),
) -> DetectionResponse:
    """Run detection and build the response. No or empty ``classes`` means all classes.

    Classes are filtered here rather than in the model call, so that every detector shares the
    same logic: the network runs in full either way. Each detection lists the ``zones`` that
    contain its foot point.

    Raises:
        DetectorError: If the request is invalid.
    """
    confidence = default_confidence if min_confidence is None else min_confidence
    _check_request(detector, classes, confidence)

    output = detector.detect(frame.image, confidence)

    wanted = set(classes) if classes else None
    detections = sorted(
        (_finished(d, zones) for d in output.detections if wanted is None or d.label in wanted),
        key=lambda d: d.confidence,
        reverse=True,
    )
    height, width = frame.image.shape[:2]
    return DetectionResponse(
        camera_id=camera_id,
        timestamp=frame.captured_at,
        # Measured after inference: this is how old the result is when the caller receives it.
        frame_age_s=round(frame.age_s, 3),
        frame_size=(width, height),
        model=detector.model_name,
        device=detector.device,
        inference_ms=round(output.inference_ms, 1),
        counts=dict(Counter(d.label for d in detections)),
        detections=detections,
    )

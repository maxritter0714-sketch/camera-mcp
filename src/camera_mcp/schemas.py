"""Response models of the tools. Coordinates are normalized, timestamps carry a timezone."""

from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

Size = tuple[int, int]
BoundingBox = tuple[float, float, float, float]


class _Response(BaseModel):
    model_config = ConfigDict(frozen=True)


class CameraInfo(_Response):
    id: str
    name: str
    source_type: str
    status: Literal["online", "offline"]
    resolution: Size | None
    fps_measured: float | None
    last_frame_age_s: float | None
    zones: list[str]


class CameraInfoResponse(_Response):
    cameras: list[CameraInfo]


class Detection(_Response):
    label: str
    confidence: float
    bbox: BoundingBox = Field(description="Normalized [x1, y1, x2, y2].")
    zones: list[str] = Field(default_factory=list)


class DetectionResponse(_Response):
    camera_id: str
    timestamp: AwareDatetime
    frame_age_s: float
    frame_size: Size
    model: str
    device: str
    inference_ms: float
    counts: dict[str, int]
    detections: list[Detection]

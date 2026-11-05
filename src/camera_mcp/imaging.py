"""Image helpers: downscaling, JPEG encoding and drawing detections."""

from collections.abc import Iterable

import cv2

from camera_mcp.schemas import Detection
from camera_mcp.sources import Image

_BOX_COLOR = (0, 255, 0)
_TEXT_COLOR = (0, 0, 0)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_REFERENCE_WIDTH = 640  # Line and font sizes are scaled relative to this width.


class ImagingError(Exception):
    """An image could not be processed."""


def resize_to_width(image: Image, max_width: int) -> Image:
    """Downscale to ``max_width`` keeping the aspect ratio. Never enlarges an image."""
    if max_width <= 0:
        raise ImagingError(f"max_width must be positive, got {max_width}.")
    height, width = image.shape[:2]
    if width <= max_width:
        return image
    new_height = max(1, round(height * max_width / width))
    return cv2.resize(image, (max_width, new_height), interpolation=cv2.INTER_AREA)


def encode_jpeg(image: Image, quality: int) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ImagingError("JPEG encoding failed.")
    return encoded.tobytes()


def annotate(image: Image, detections: Iterable[Detection]) -> Image:
    """Return a copy with boxes and labels drawn. Frames are shared and read-only."""
    annotated = image.copy()
    height, width = annotated.shape[:2]
    scale = max(width / _REFERENCE_WIDTH, 0.5)
    thickness = max(1, round(2 * scale))
    font_scale = 0.5 * scale

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        left, top = round(x1 * width), round(y1 * height)
        right, bottom = round(x2 * width), round(y2 * height)
        cv2.rectangle(annotated, (left, top), (right, bottom), _BOX_COLOR, thickness)

        text = f"{detection.label} {detection.confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(text, _FONT, font_scale, thickness)
        label_height = text_height + baseline
        # Keep the label inside the image if the box touches the top edge.
        label_top = top - label_height if top >= label_height else top
        cv2.rectangle(
            annotated,
            (left, label_top),
            (left + text_width, label_top + label_height),
            _BOX_COLOR,
            cv2.FILLED,
        )
        cv2.putText(
            annotated,
            text,
            (left, label_top + text_height),
            _FONT,
            font_scale,
            _TEXT_COLOR,
            thickness,
            cv2.LINE_AA,
        )
    return annotated

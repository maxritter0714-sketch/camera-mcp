import cv2
import numpy as np
import pytest

from camera_mcp.imaging import ImagingError, annotate, encode_jpeg, resize_to_width
from camera_mcp.schemas import Detection


def blank(width: int = 640, height: int = 480) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_resize_downscales_and_keeps_aspect_ratio() -> None:
    assert resize_to_width(blank(1280, 720), 640).shape == (360, 640, 3)


@pytest.mark.parametrize("max_width", [640, 1000])
def test_resize_never_enlarges(max_width: int) -> None:
    image = blank(640, 480)

    assert resize_to_width(image, max_width) is image


def test_resize_keeps_extreme_aspect_ratio_valid() -> None:
    assert resize_to_width(blank(1000, 1), 10).shape == (1, 10, 3)


@pytest.mark.parametrize("max_width", [0, -5])
def test_resize_rejects_non_positive_width(max_width: int) -> None:
    with pytest.raises(ImagingError, match="positive"):
        resize_to_width(blank(), max_width)


def test_resize_accepts_read_only_frame() -> None:
    image = blank(1280, 720)
    image.flags.writeable = False

    assert resize_to_width(image, 640).shape == (360, 640, 3)


def test_encode_jpeg_produces_decodable_image() -> None:
    data = encode_jpeg(blank(320, 240), quality=80)

    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert data.startswith(b"\xff\xd8")
    assert decoded.shape == (240, 320, 3)


def test_higher_jpeg_quality_gives_larger_file() -> None:
    noise = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)

    assert len(encode_jpeg(noise, quality=95)) > len(encode_jpeg(noise, quality=20))


def box(bbox: tuple[float, float, float, float]) -> Detection:
    return Detection(label="person", confidence=0.91, bbox=bbox)


def test_annotate_draws_at_normalized_position_on_a_copy() -> None:
    image = blank()
    image.flags.writeable = False

    result = annotate(image, [box((0.25, 0.5, 0.75, 1.0))])

    assert image.sum() == 0
    assert result[360, 160].sum() > 0  # left edge of the box
    assert result[20, 20].sum() == 0  # outside of box and label


def test_annotate_without_detections_returns_equal_image() -> None:
    image = blank()

    assert np.array_equal(annotate(image, []), image)


@pytest.mark.parametrize(
    "bbox", [(0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.0), (0.9, 0.99, 1.0, 1.0)]
)
def test_annotate_handles_boxes_at_image_borders(bbox: tuple[float, float, float, float]) -> None:
    assert annotate(blank(), [box(bbox)]).shape == (480, 640, 3)

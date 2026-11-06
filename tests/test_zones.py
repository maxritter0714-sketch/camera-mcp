import time
from datetime import datetime

import numpy as np
import pytest
from fakes import FakeDetector

from camera_mcp.camera_manager import Frame
from camera_mcp.config import ZoneConfig
from camera_mcp.detector import detect_in_frame
from camera_mcp.schemas import Detection
from camera_mcp.zones import ZoneError, contains, count_in_zone, find_zone, foot_point, zones_for

SQUARE = [(0.2, 0.2), (0.6, 0.2), (0.6, 0.6), (0.2, 0.6)]
# An L shape: the square from (0, 0) to (1, 1) without its upper right quarter.
L_SHAPE = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)]
TRIANGLE = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
WHOLE_IMAGE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def test_foot_point_is_the_middle_of_the_bottom_edge() -> None:
    assert foot_point((0.2, 0.1, 0.4, 0.9)) == pytest.approx((0.3, 0.9))


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ((0.4, 0.4), True),  # inside
        ((0.1, 0.4), False),  # left of it
        ((0.4, 0.7), False),  # below it
        ((0.9, 0.9), False),  # far away
    ],
)
def test_point_inside_and_outside(point: tuple[float, float], expected: bool) -> None:
    assert contains(SQUARE, point) is expected


@pytest.mark.parametrize(
    "point",
    [
        (0.4, 0.2),  # on the top edge
        (0.6, 0.4),  # on the right edge
        (0.4, 0.6),  # on the bottom edge
        (0.2, 0.4),  # on the left edge
        (0.2, 0.2),  # on a corner
        (0.6, 0.6),  # on the opposite corner
    ],
)
def test_boundary_counts_as_inside(point: tuple[float, float]) -> None:
    assert contains(SQUARE, point)


def test_just_outside_the_boundary_is_outside() -> None:
    assert not contains(SQUARE, (0.6 + 1e-6, 0.4))


def test_diagonal_edge() -> None:
    assert contains(TRIANGLE, (0.5, 0.5))  # on the hypotenuse
    assert contains(TRIANGLE, (0.2, 0.2))
    assert not contains(TRIANGLE, (0.6, 0.6))


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ((0.25, 0.25), True),  # upper left arm
        ((0.75, 0.75), True),  # lower right arm
        ((0.75, 0.25), False),  # the missing quarter
        ((0.5, 0.25), True),  # inner edge
        ((0.5, 0.5), True),  # inner corner
    ],
)
def test_concave_polygon(point: tuple[float, float], expected: bool) -> None:
    assert contains(L_SHAPE, point) is expected


def test_ray_through_a_vertex_is_counted_once() -> None:
    # The horizontal rays from both points pass exactly through the vertices at height 0.5.
    diamond = [(0.5, 0.2), (0.8, 0.5), (0.5, 0.8), (0.2, 0.5)]

    assert contains(diamond, (0.3, 0.5))
    assert not contains(diamond, (0.1, 0.5))


def test_foot_at_the_bottom_of_the_image_is_in_a_zone_that_reaches_it() -> None:
    zone = ZoneConfig(name="floor", polygon=[(0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0)])

    assert zones_for((0.4, 0.3, 0.6, 1.0), [zone]) == ["floor"]


def test_overlapping_zones_are_all_listed_in_configuration_order() -> None:
    zones = [
        ZoneConfig(name="all", polygon=WHOLE_IMAGE),
        ZoneConfig(name="left", polygon=[(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]),
        ZoneConfig(name="right", polygon=[(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)]),
    ]

    assert zones_for((0.1, 0.1, 0.3, 0.9), zones) == ["all", "left"]
    assert zones_for((0.4, 0.1, 0.6, 0.9), zones) == ["all", "left", "right"]  # foot on the seam


def test_box_center_does_not_decide_but_the_foot_point() -> None:
    # The box mostly covers the zone, but stands below it.
    zone = ZoneConfig(name="table", polygon=SQUARE)

    assert zones_for((0.2, 0.2, 0.6, 0.8), [zone]) == []


def test_find_zone_returns_the_zone() -> None:
    zone = ZoneConfig(name="door", polygon=SQUARE)

    assert find_zone([zone], "cam", "door") is zone


def test_unknown_zone_lists_the_available_ones() -> None:
    zones = [ZoneConfig(name=name, polygon=SQUARE) for name in ("window", "door")]

    with pytest.raises(ZoneError, match=r"Unknown zone 'desk' for camera 'cam'\. .*door, window"):
        find_zone(zones, "cam", "desk")


def test_camera_without_zones_says_so() -> None:
    with pytest.raises(ZoneError, match="Camera 'cam' has no zones.*get_camera_info"):
        find_zone([], "cam", "door")


def make_frame() -> Frame:
    return Frame(
        image=np.zeros((48, 64, 3), dtype=np.uint8),
        captured_at=datetime.now().astimezone(),
        captured_monotonic=time.monotonic(),
    )


def test_count_keeps_only_the_detections_in_the_zone() -> None:
    zones = [ZoneConfig(name="left", polygon=[(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)])]
    detections = [
        Detection(label="person", confidence=0.9, bbox=(0.1, 0.1, 0.3, 0.9)),
        Detection(label="person", confidence=0.8, bbox=(0.6, 0.1, 0.8, 0.9)),
        Detection(label="cup", confidence=0.7, bbox=(0.2, 0.5, 0.3, 0.6)),
    ]
    response = detect_in_frame(
        FakeDetector(detections), make_frame(), "cam", default_confidence=0.4, zones=zones
    )

    counted = count_in_zone(response, "left")

    assert counted.camera_id == "cam"
    assert counted.zone == "left"
    assert counted.timestamp == response.timestamp
    assert counted.frame_age_s == response.frame_age_s
    assert counted.counts == {"person": 1, "cup": 1}
    assert [d.confidence for d in counted.detections] == [0.9, 0.7]
    assert all(d.zones == ["left"] for d in counted.detections)


def test_empty_zone_gives_empty_counts() -> None:
    zones = [ZoneConfig(name="corner", polygon=[(0.9, 0.9), (1.0, 0.9), (1.0, 1.0)])]
    detections = [Detection(label="person", confidence=0.9, bbox=(0.1, 0.1, 0.3, 0.5))]
    response = detect_in_frame(
        FakeDetector(detections), make_frame(), "cam", default_confidence=0.4, zones=zones
    )

    counted = count_in_zone(response, "corner")

    assert counted.counts == {}
    assert counted.detections == []

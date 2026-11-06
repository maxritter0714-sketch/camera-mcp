"""Zones: named polygons in normalized coordinates, and which detections lie in them.

A detection belongs to a zone if its foot point, the middle of the bottom edge of its box, lies
inside the polygon or on its boundary (see DESIGN.md).
"""

from collections import Counter
from collections.abc import Sequence

from camera_mcp.config import ZoneConfig
from camera_mcp.schemas import BoundingBox, DetectionResponse, ZoneCountResponse

Point = tuple[float, float]

# Tolerance for "on the boundary". Far below the precision of normalized pixel coordinates,
# far above float rounding errors.
_EPSILON = 1e-9


class ZoneError(Exception):
    """A requested zone does not exist. The message is meant for the model."""


def foot_point(bbox: BoundingBox) -> Point:
    """Middle of the bottom edge of a normalized ``[x1, y1, x2, y2]`` box."""
    x1, _y1, x2, y2 = bbox
    return ((x1 + x2) / 2, y2)


def _on_segment(point: Point, start: Point, end: Point) -> bool:
    (px, py), (ax, ay), (bx, by) = point, start, end
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if abs(cross) > _EPSILON:
        return False
    within_x = min(ax, bx) - _EPSILON <= px <= max(ax, bx) + _EPSILON
    within_y = min(ay, by) - _EPSILON <= py <= max(ay, by) + _EPSILON
    return within_x and within_y


def contains(polygon: Sequence[Point], point: Point) -> bool:
    """Whether ``point`` lies inside ``polygon`` or on its boundary.

    Ray casting with the even-odd rule, so concave polygons work and self-intersecting ones
    count overlapping parts as outside. The boundary is checked first and counts as inside.
    """
    edges = list(zip(polygon, [*polygon[1:], polygon[0]], strict=True))
    if any(_on_segment(point, start, end) for start, end in edges):
        return True

    px, py = point
    inside = False
    for (ax, ay), (bx, by) in edges:
        # Half-open rule: a vertex exactly at the height of the ray counts for one edge only.
        if (ay > py) != (by > py):
            crossing_x = ax + (py - ay) * (bx - ax) / (by - ay)
            if px < crossing_x:
                inside = not inside
    return inside


def zones_for(bbox: BoundingBox, zones: Sequence[ZoneConfig]) -> list[str]:
    """Names of the zones that contain the foot point of ``bbox``, in configuration order."""
    point = foot_point(bbox)
    return [zone.name for zone in zones if contains(zone.polygon, point)]


def find_zone(zones: Sequence[ZoneConfig], camera_id: str, name: str) -> ZoneConfig:
    """Look up a zone of a camera by name.

    Raises:
        ZoneError: If the camera has no zone with that name.
    """
    for zone in zones:
        if zone.name == name:
            return zone
    if not zones:
        raise ZoneError(
            f"Camera '{camera_id}' has no zones. Use get_camera_info to see the zones of each "
            "camera."
        )
    available = ", ".join(sorted(zone.name for zone in zones))
    raise ZoneError(
        f"Unknown zone '{name}' for camera '{camera_id}'. Available zones: {available}."
    )


def count_in_zone(response: DetectionResponse, zone: str) -> ZoneCountResponse:
    """Reduce a detection result to the detections in ``zone`` and count them per class."""
    detections = [d for d in response.detections if zone in d.zones]
    return ZoneCountResponse(
        camera_id=response.camera_id,
        zone=zone,
        timestamp=response.timestamp,
        frame_age_s=response.frame_age_s,
        counts=dict(Counter(d.label for d in detections)),
        detections=detections,
    )

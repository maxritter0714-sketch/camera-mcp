"""MCP server: lifespan and read-only camera tools. Run with ``camera-mcp`` (stdio)."""

import logging
import sys
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np
from mcp.server.mcpserver import Context, Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

from camera_mcp.camera_manager import CameraError, CameraManager, CameraStatus, Frame
from camera_mcp.config import (
    AppConfig,
    ConfigError,
    DetectorConfig,
    load_config,
    resolve_config_path,
)
from camera_mcp.detector import Detector, DetectorError, UltralyticsDetector, detect_in_frame
from camera_mcp.imaging import ImagingError, annotate, encode_jpeg, resize_to_width
from camera_mcp.logging_setup import setup_logging
from camera_mcp.schemas import (
    CameraInfo,
    CameraInfoResponse,
    DetectionResponse,
    ZoneCountResponse,
)
from camera_mcp.zones import ZoneError, count_in_zone, find_zone

logger = logging.getLogger(__name__)

SERVER_NAME = "camera-mcp"
_READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True)

DetectorFactory = Callable[[DetectorConfig], Detector]


@dataclass(frozen=True)
class _App:
    """Everything the tools need; created once per server run by the lifespan."""

    config: AppConfig
    cameras: CameraManager
    detector: Detector


def _warm_up(detector: Detector, config: DetectorConfig) -> None:
    """Run one inference, as the first call is far slower than later ones (see DESIGN.md)."""
    blank = np.zeros((config.imgsz, config.imgsz, 3), dtype=np.uint8)
    detector.detect(blank, config.default_confidence)


@contextmanager
def _model_readable_errors() -> Iterator[None]:
    """Report anticipated failures to the model. The SDK withholds the text of anything else."""
    try:
        yield
    except (CameraError, DetectorError, ImagingError, ZoneError) as exc:
        raise ToolError(str(exc)) from exc


def _camera_info(status: CameraStatus, config: AppConfig) -> CameraInfo:
    age = status.last_frame_age_s
    fps = status.fps_measured
    return CameraInfo(
        id=status.id,
        name=status.name,
        source_type=status.source_type,
        status=status.status,
        resolution=status.resolution,
        fps_measured=None if fps is None else round(fps, 1),
        last_frame_age_s=None if age is None else round(age, 2),
        zones=[zone.name for zone in config.zones.get(status.id, [])],
    )


def _jpeg(app: _App, image: Any, max_width: int) -> tuple[Image, tuple[int, int]]:
    """Encode an image within the configured width limit; returns its delivered size."""
    limits = app.config.limits
    resized = resize_to_width(image, min(max_width, limits.max_image_width))
    data = encode_jpeg(resized, limits.jpeg_quality)
    return Image(data=data, format="jpeg"), (resized.shape[1], resized.shape[0])


def _detect(
    app: _App,
    frame: Frame,
    camera_id: str,
    classes: list[str] | None = None,
    min_confidence: float | None = None,
) -> DetectionResponse:
    return detect_in_frame(
        app.detector,
        frame,
        camera_id,
        default_confidence=app.config.detector.default_confidence,
        classes=classes,
        min_confidence=min_confidence,
        zones=app.config.zones.get(camera_id, []),
    )


def create_server(make_detector: DetectorFactory = UltralyticsDetector) -> MCPServer[_App]:
    """Build the server. Tests pass a factory for a fake detector."""

    @asynccontextmanager
    async def lifespan(_server: MCPServer[_App]) -> AsyncIterator[_App]:
        config = load_config(resolve_config_path())
        detector = make_detector(config.detector)
        _warm_up(detector, config.detector)
        cameras = CameraManager(config.cameras, config.limits)
        cameras.start()
        logger.info("Server ready: %d camera(s), device %s.", len(config.cameras), detector.device)
        try:
            yield _App(config=config, cameras=cameras, detector=detector)
        finally:
            cameras.stop()

    server: MCPServer[_App] = MCPServer(SERVER_NAME, lifespan=lifespan)

    # Plain functions on purpose: the SDK runs them in a worker thread, so that inference and
    # encoding do not block the protocol. Docstrings are the descriptions from KONZEPT.md
    # section 5 and part of the design; a test compares them.

    @server.tool(annotations=_READ_ONLY)
    def get_camera_info(ctx: Context, camera_id: str | None = None) -> CameraInfoResponse:
        """Listet die verfügbaren Kameras mit Status (online/offline), Auflösung, Alter des letzten Bildes und den definierten Zonen. Ohne `camera_id` werden alle Kameras geliefert. Nutze dieses Tool zuerst, wenn du nicht weißt, welche Kameras oder Zonen es gibt."""  # noqa: E501
        app = ctx.request_context.lifespan_context
        with _model_readable_errors():
            statuses = (
                app.cameras.statuses() if camera_id is None else [app.cameras.status(camera_id)]
            )
        return CameraInfoResponse(cameras=[_camera_info(s, app.config) for s in statuses])

    @server.tool(annotations=_READ_ONLY, structured_output=False)
    def get_frame(
        ctx: Context, camera_id: str, max_width: int = 1024, annotate: bool = False
    ) -> list[Image | str]:
        """Liefert das aktuelle Bild einer Kamera. Nur nutzen, wenn du die Szene wirklich sehen musst (z.B. um etwas zu beschreiben, das die Objekterkennung nicht abdeckt). Für Fragen wie „ist jemand da?“ oder „wie viele …?“ ist `detect_objects` schneller und günstiger. Mit `annotate=true` werden erkannte Objekte eingezeichnet."""  # noqa: E501
        app = ctx.request_context.lifespan_context
        with _model_readable_errors():
            frame = app.cameras.latest_frame(camera_id)
            image = frame.image
            if annotate:
                image = _draw(image, _detect(app, frame, camera_id))
            jpeg, (width, height) = _jpeg(app, image, max_width)
        original_height, original_width = frame.image.shape[:2]
        text = (
            f"timestamp: {frame.captured_at.isoformat()}, "
            f"original resolution: {original_width}x{original_height}, "
            f"delivered resolution: {width}x{height}, "
            f"frame age: {frame.age_s:.2f} s"
        )
        return [jpeg, text]

    @server.tool(annotations=_READ_ONLY, structured_output=False)
    def detect_objects(
        ctx: Context,
        camera_id: str,
        classes: list[str] | None = None,
        min_confidence: float | None = None,
        include_image: bool = False,
    ) -> list[Image | str]:
        """Erkennt Objekte im aktuellen Bild (COCO-Klassen wie person, car, bicycle, dog, cell phone …). Liefert strukturierte Ergebnisse mit Klasse, Konfidenz, Position und Zone. Mit `classes` auf bestimmte Klassen filtern. Ein Bild nur mit `include_image=true` anfordern, wenn die Ergebnisse visuell geprüft werden sollen."""  # noqa: E501
        app = ctx.request_context.lifespan_context
        with _model_readable_errors():
            frame = app.cameras.latest_frame(camera_id)
            response = _detect(app, frame, camera_id, classes, min_confidence)
            content: list[Image | str] = [response.model_dump_json()]
            if include_image:
                limit = app.config.limits.max_image_width
                content.append(_jpeg(app, _draw(frame.image, response), limit)[0])
        return content

    @server.tool(name="count_in_zone", annotations=_READ_ONLY)
    def count_in_zone_tool(
        ctx: Context,
        camera_id: str,
        zone: str,
        classes: list[str] | None = None,
        min_confidence: float | None = None,
    ) -> ZoneCountResponse:
        """Zählt erkannte Objekte innerhalb einer benannten Zone (z.B. „tuer“). Ein Objekt zählt zur Zone, wenn der Mittelpunkt der Unterkante seiner Box (Fußpunkt) in der Zone liegt. Verfügbare Zonen liefert `get_camera_info`."""  # noqa: E501
        app = ctx.request_context.lifespan_context
        with _model_readable_errors():
            frame = app.cameras.latest_frame(camera_id)
            # Checked before inference, which is the expensive part.
            find_zone(app.config.zones.get(camera_id, []), camera_id, zone)
            response = _detect(app, frame, camera_id, classes, min_confidence)
        return count_in_zone(response, zone)

    return server


def _draw(image: Any, response: DetectionResponse) -> Any:
    # Module-level alias: the `annotate` parameter of get_frame shadows the imaging function.
    return annotate(image, response.detections)


setup_logging()
# Found by name by `mcp dev`. Nothing is loaded until the server actually runs.
mcp = create_server()


def _log_startup_errors(group: BaseExceptionGroup) -> None:
    for exc in group.exceptions:
        if isinstance(exc, BaseExceptionGroup):
            _log_startup_errors(exc)
        else:
            logger.error("%s", exc)


def main() -> None:
    """Run the server on stdio. Unusable configuration or models end it with status 1."""
    failed = False
    try:
        # Checked here as well, to fail before any protocol traffic starts.
        load_config(resolve_config_path())
        mcp.run()
    except* (ConfigError, DetectorError) as group:
        _log_startup_errors(group)
        failed = True
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

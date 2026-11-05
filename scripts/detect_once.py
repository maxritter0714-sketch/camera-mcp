"""Run detection once on a camera frame or an image file and print JSON, without MCP."""

import argparse
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from grab_frame import grab

from camera_mcp.camera_manager import CameraOfflineError, Frame, UnknownCameraError
from camera_mcp.config import ConfigError, DetectorConfig, load_config, resolve_config_path
from camera_mcp.detector import Detector, DetectorError, UltralyticsDetector, detect_in_frame
from camera_mcp.logging_setup import setup_logging
from camera_mcp.sources import ImageFileFrameSource, SourceEnded, SourceError

IMAGE_CAMERA_ID = "image"


def read_image_file(path: Path) -> Frame:
    source = ImageFileFrameSource(path, loop=False)
    source.open()
    try:
        image = source.read()
    finally:
        source.close()
    return Frame(
        image=image,
        captured_at=datetime.now().astimezone(),
        captured_monotonic=time.monotonic(),
    )


def run(argv: list[str], make_detector: Callable[[DetectorConfig], Detector]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--camera", help="camera id from the configuration")
    target.add_argument("--image", type=Path, help="image file to run detection on")
    args = parser.parse_args(argv)

    setup_logging()
    try:
        config = load_config(resolve_config_path())
        # Loaded before reading the frame, so that loading does not count towards its age.
        detector = make_detector(config.detector)
        if args.image:
            camera_id, frame = IMAGE_CAMERA_ID, read_image_file(args.image)
        else:
            camera_id, frame = args.camera, grab(config, args.camera)
        response = detect_in_frame(
            detector,
            frame,
            camera_id,
            default_confidence=config.detector.default_confidence,
        )
    except (
        ConfigError,
        UnknownCameraError,
        CameraOfflineError,
        SourceError,
        SourceEnded,
        DetectorError,
    ) as exc:
        print(exc, file=sys.stderr)
        return 1

    print(response.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:], UltralyticsDetector))

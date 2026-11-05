"""Save one frame of a configured camera as JPEG, without MCP."""

import argparse
import sys
import time
from pathlib import Path

from camera_mcp.camera_manager import (
    CameraManager,
    CameraOfflineError,
    Frame,
    UnknownCameraError,
)
from camera_mcp.config import AppConfig, ConfigError, load_config, resolve_config_path
from camera_mcp.imaging import ImagingError, encode_jpeg
from camera_mcp.logging_setup import setup_logging

FIRST_FRAME_TIMEOUT_S = 15.0
POLL_INTERVAL_S = 0.05


def wait_for_frame(manager: CameraManager, camera_id: str) -> Frame:
    deadline = time.monotonic() + FIRST_FRAME_TIMEOUT_S
    while True:
        try:
            return manager.latest_frame(camera_id)
        except CameraOfflineError:
            if time.monotonic() >= deadline:
                raise
        time.sleep(POLL_INTERVAL_S)


def grab(config: AppConfig, camera_id: str) -> Frame:
    """Run only the requested camera, so that other unreachable cameras cause no delay."""
    selected = [camera for camera in config.cameras if camera.id == camera_id]
    if not selected:
        raise UnknownCameraError(camera_id, [camera.id for camera in config.cameras])
    manager = CameraManager(selected, config.limits)
    manager.start()
    try:
        return wait_for_frame(manager, camera_id)
    finally:
        manager.stop()


def save_jpeg(frame: Frame, path: Path, quality: int) -> None:
    data = encode_jpeg(frame.image, quality)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", required=True, help="camera id from the configuration")
    parser.add_argument(
        "--output", type=Path, help="target file (default: recordings/<camera>.jpg)"
    )
    args = parser.parse_args()

    setup_logging()
    output: Path = args.output or Path("recordings") / f"{args.camera}.jpg"
    try:
        config = load_config(resolve_config_path())
        frame = grab(config, args.camera)
        save_jpeg(frame, output, config.limits.jpeg_quality)
    except (ConfigError, UnknownCameraError, CameraOfflineError, ImagingError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1

    height, width = frame.image.shape[:2]
    print(f"Saved {width}x{height} frame from '{args.camera}' to {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""The real server with a fake detector, started as a subprocess by the end-to-end tests."""

from fakes import FakeDetector

from camera_mcp.schemas import Detection
from camera_mcp.server import create_server

DETECTIONS = [Detection(label="person", confidence=0.9, bbox=(0.1, 0.2, 0.3, 0.4))]

if __name__ == "__main__":
    create_server(lambda _config: FakeDetector(DETECTIONS)).run()

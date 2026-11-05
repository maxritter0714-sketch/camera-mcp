"""Tool behaviour through a real MCP client connected in-process, with a fake detector."""

import base64
import json
import re
import threading
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path

import anyio
import cv2
import numpy as np
import pytest
from fakes import FakeDetector
from mcp.client import Client
from mcp_types import CallToolResult, ImageContent, TextContent

from camera_mcp.config import CONFIG_ENV_VAR
from camera_mcp.schemas import Detection
from camera_mcp.server import create_server

pytestmark = pytest.mark.anyio

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_VIDEO = (REPO_ROOT / "tests" / "fixtures" / "sample.mp4").as_posix()
MAX_IMAGE_WIDTH = 200
TOOLS = {"get_camera_info", "get_frame", "detect_objects"}

PERSON = Detection(label="person", confidence=0.9, bbox=(0.1, 0.2, 0.3, 0.4))
CUP = Detection(label="cup", confidence=0.5, bbox=(0.5, 0.5, 0.6, 0.6))


@pytest.fixture
def detector() -> FakeDetector:
    return FakeDetector([PERSON, CUP])


@pytest.fixture
async def client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, detector: FakeDetector
) -> AsyncIterator[Client]:
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
cameras:
  - {{id: sample, name: Sample, source: {{type: file, path: '{SAMPLE_VIDEO}', loop: true}}}}
  - {{id: broken, name: Broken, source: {{type: file, path: absent.mp4}}}}
zones:
  sample:
    - {{name: door, polygon: [[0, 0], [1, 0], [1, 1]]}}
limits:
  max_image_width: {MAX_IMAGE_WIDTH}
""",
        encoding="utf-8",
    )
    monkeypatch.setenv(CONFIG_ENV_VAR, str(config))
    async with Client(create_server(lambda _config: detector)) as connected:
        await wait_for_camera(connected)
        yield connected


async def wait_for_camera(client: Client) -> None:
    with anyio.fail_after(10):
        while True:
            info = payload(await client.call_tool("get_camera_info", {"camera_id": "sample"}))
            if info["cameras"][0]["status"] == "online":
                return
            await anyio.sleep(0.02)


def texts(result: CallToolResult) -> list[str]:
    return [block.text for block in result.content if isinstance(block, TextContent)]


def images(result: CallToolResult) -> list[ImageContent]:
    return [block for block in result.content if isinstance(block, ImageContent)]


def payload(result: CallToolResult) -> dict:
    assert not result.is_error, texts(result)
    return json.loads(texts(result)[0])


def decode(block: ImageContent) -> np.ndarray:
    raw = np.frombuffer(base64.b64decode(block.data), dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def error_text(result: CallToolResult) -> str:
    assert result.is_error
    return texts(result)[0]


def specified_descriptions() -> dict[str, str]:
    """The tool descriptions as written in KONZEPT.md section 5."""
    concept = (REPO_ROOT / "KONZEPT.md").read_text(encoding="utf-8")
    pattern = r"#### `(\w+)\(.*?\n.*?\n?> (.+?)\n"
    return dict(re.findall(pattern, concept, flags=re.DOTALL))


async def test_only_the_milestone_tools_are_offered(client: Client) -> None:
    tools = (await client.list_tools()).tools

    assert {tool.name for tool in tools} == TOOLS


async def test_descriptions_match_the_specification_exactly(client: Client) -> None:
    specified = specified_descriptions()

    for tool in (await client.list_tools()).tools:
        assert tool.description == specified[tool.name]


async def test_tools_are_marked_read_only_and_idempotent(client: Client) -> None:
    for tool in (await client.list_tools()).tools:
        assert tool.annotations is not None
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.idempotent_hint is True


async def test_parameters_match_the_specification(client: Client) -> None:
    schemas = {tool.name: tool.input_schema for tool in (await client.list_tools()).tools}

    assert set(schemas["get_camera_info"]["properties"]) == {"camera_id"}
    assert set(schemas["get_frame"]["properties"]) == {"camera_id", "max_width", "annotate"}
    assert set(schemas["detect_objects"]["properties"]) == {
        "camera_id",
        "classes",
        "min_confidence",
        "include_image",
    }


async def test_camera_info_lists_all_cameras(client: Client) -> None:
    cameras = payload(await client.call_tool("get_camera_info"))["cameras"]

    assert [camera["id"] for camera in cameras] == ["sample", "broken"]
    sample, broken = cameras
    assert sample["name"] == "Sample"
    assert sample["source_type"] == "file"
    assert sample["status"] == "online"
    assert sample["resolution"] == [320, 240]
    assert sample["last_frame_age_s"] < 5
    assert sample["zones"] == ["door"]
    assert broken["status"] == "offline"
    assert broken["resolution"] is None
    assert broken["last_frame_age_s"] is None
    assert broken["zones"] == []


async def test_camera_info_filters_by_id(client: Client) -> None:
    result = payload(await client.call_tool("get_camera_info", {"camera_id": "broken"}))

    assert [camera["id"] for camera in result["cameras"]] == ["broken"]


async def test_camera_info_unknown_camera_lists_known_ones(client: Client) -> None:
    result = await client.call_tool("get_camera_info", {"camera_id": "nope"})

    assert "Available cameras: broken, sample" in error_text(result)


async def test_get_frame_returns_jpeg_and_description(client: Client) -> None:
    result = await client.call_tool("get_frame", {"camera_id": "sample"})

    assert not result.is_error
    image = images(result)[0]
    assert image.mime_type == "image/jpeg"
    assert decode(image).shape[1] == MAX_IMAGE_WIDTH
    text = texts(result)[0]
    assert "original resolution: 320x240" in text
    assert f"delivered resolution: {MAX_IMAGE_WIDTH}x150" in text
    assert "frame age:" in text
    timestamp = re.search(r"timestamp: (\S+),", text)
    assert timestamp is not None
    assert datetime.fromisoformat(timestamp.group(1)).utcoffset() is not None


async def test_get_frame_uses_smaller_requested_width(client: Client) -> None:
    result = await client.call_tool("get_frame", {"camera_id": "sample", "max_width": 100})

    assert decode(images(result)[0]).shape[1] == 100


async def test_get_frame_limits_width_to_configuration(client: Client) -> None:
    result = await client.call_tool("get_frame", {"camera_id": "sample", "max_width": 5000})

    assert decode(images(result)[0]).shape[1] == MAX_IMAGE_WIDTH


@pytest.mark.parametrize("width", [0, -10])
async def test_get_frame_rejects_invalid_width(client: Client, width: int) -> None:
    result = await client.call_tool("get_frame", {"camera_id": "sample", "max_width": width})

    assert "max_width must be positive" in error_text(result)


async def test_get_frame_runs_detection_only_when_annotating(
    client: Client, detector: FakeDetector
) -> None:
    calls_after_warm_up = len(detector.calls)

    await client.call_tool("get_frame", {"camera_id": "sample"})
    assert len(detector.calls) == calls_after_warm_up

    result = await client.call_tool("get_frame", {"camera_id": "sample", "annotate": True})
    assert not result.is_error
    assert len(detector.calls) == calls_after_warm_up + 1


async def test_annotation_changes_the_image(client: Client, detector: FakeDetector) -> None:
    full_box = Detection(label="person", confidence=0.9, bbox=(0.0, 0.0, 1.0, 1.0))
    detector.detections = [full_box]

    result = await client.call_tool("get_frame", {"camera_id": "sample", "annotate": True})

    border = decode(images(result)[0])[75, 0]
    assert border[1] > 150  # green channel of the box at the left edge


async def test_offline_camera_gives_clear_error(client: Client) -> None:
    for tool in ("get_frame", "detect_objects"):
        result = await client.call_tool(tool, {"camera_id": "broken"})
        message = error_text(result)
        assert "Camera 'broken' is offline" in message
        assert "get_camera_info" in message
        assert "Traceback" not in message


async def test_detect_objects_returns_specified_json_without_image(client: Client) -> None:
    result = await client.call_tool("detect_objects", {"camera_id": "sample"})

    response = payload(result)
    assert images(result) == []
    assert response["camera_id"] == "sample"
    assert response["frame_size"] == [320, 240]
    assert response["model"] == "fake.pt"
    assert response["counts"] == {"person": 1, "cup": 1}
    assert response["detections"][0] == {
        "label": "person",
        "confidence": 0.9,
        "bbox": [0.1, 0.2, 0.3, 0.4],
        "zones": [],
    }


async def test_detect_objects_filters_classes(client: Client) -> None:
    result = await client.call_tool("detect_objects", {"camera_id": "sample", "classes": ["cup"]})

    assert payload(result)["counts"] == {"cup": 1}


async def test_detect_objects_applies_confidence(client: Client) -> None:
    result = await client.call_tool(
        "detect_objects", {"camera_id": "sample", "min_confidence": 0.8}
    )

    assert payload(result)["counts"] == {"person": 1}


async def test_detect_objects_unknown_class_lists_valid_classes(client: Client) -> None:
    result = await client.call_tool(
        "detect_objects", {"camera_id": "sample", "classes": ["unicorn"]}
    )

    message = error_text(result)
    assert "Unknown classes: unicorn" in message
    assert "person" in message


async def test_detect_objects_invalid_confidence_gives_clear_error(client: Client) -> None:
    result = await client.call_tool("detect_objects", {"camera_id": "sample", "min_confidence": 2})

    assert "min_confidence" in error_text(result)


async def test_detect_objects_adds_limited_image_on_request(client: Client) -> None:
    result = await client.call_tool(
        "detect_objects", {"camera_id": "sample", "include_image": True}
    )

    assert payload(result)["counts"] == {"person": 1, "cup": 1}
    assert decode(images(result)[0]).shape[1] == MAX_IMAGE_WIDTH


async def test_unexpected_failure_does_not_reach_the_client(
    client: Client, detector: FakeDetector
) -> None:
    def crash(_image: object, _confidence: float) -> None:
        raise RuntimeError("rtsp://user:s3cret@camera.invalid")

    detector.detect = crash

    result = await client.call_tool("detect_objects", {"camera_id": "sample"})

    message = error_text(result)
    assert "s3cret" not in message
    assert "Traceback" not in message


async def test_model_is_warmed_up_once_before_the_first_call(
    client: Client, detector: FakeDetector
) -> None:
    assert detector.calls == [0.4]


async def test_cameras_stop_when_the_server_ends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, detector: FakeDetector
) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        f"cameras:\n  - {{id: a, name: A, source: {{type: file, path: '{SAMPLE_VIDEO}'}}}}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(CONFIG_ENV_VAR, str(config))

    async with Client(create_server(lambda _config: detector)):
        assert any(thread.name == "camera-a" for thread in threading.enumerate())

    assert not any(thread.name == "camera-a" for thread in threading.enumerate())

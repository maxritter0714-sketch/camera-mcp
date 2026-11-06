"""A real MCP client talking to a server subprocess over stdio."""

import json
import os
import subprocess
import sys
from pathlib import Path

import anyio
import pytest
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp_types import ImageContent, TextContent

from camera_mcp.config import CONFIG_ENV_VAR

pytestmark = pytest.mark.anyio

REPO_ROOT = Path(__file__).resolve().parent.parent
FAKE_SERVER = REPO_ROOT / "tests" / "fake_mcp_server.py"
SAMPLE_VIDEO = (REPO_ROOT / "tests" / "fixtures" / "sample.mp4").as_posix()
TIMEOUT_S = 120
SERVER_MODULE = ["-m", "camera_mcp.server"]


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        "cameras:\n"
        f"  - {{id: sample, name: Sample, source: {{type: file, path: '{SAMPLE_VIDEO}'}}}}\n"
        "zones:\n"
        "  sample:\n"
        "    - {name: all, polygon: [[0, 0], [1, 0], [1, 1], [0, 1]]}\n",
        encoding="utf-8",
    )
    return path


def server_parameters(config: Path, *args: str) -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=list(args),
        env={CONFIG_ENV_VAR: str(config)},
        cwd=REPO_ROOT,  # The real model file is kept in the ignored project root.
    )


async def wait_for_camera(client: Client) -> None:
    with anyio.fail_after(30):
        while True:
            result = await client.call_tool("get_camera_info", {"camera_id": "sample"})
            text = result.content[0]
            assert isinstance(text, TextContent)
            if json.loads(text.text)["cameras"][0]["status"] == "online":
                return
            await anyio.sleep(0.05)


async def check_tools(client: Client, expected_model: str) -> None:
    """Every tool works, which also proves that nothing but protocol data was on stdout."""
    tools = {tool.name for tool in (await client.list_tools()).tools}
    assert tools == {"get_camera_info", "get_frame", "detect_objects", "count_in_zone"}

    await wait_for_camera(client)

    frame = await client.call_tool("get_frame", {"camera_id": "sample", "annotate": True})
    assert not frame.is_error
    assert any(isinstance(block, ImageContent) for block in frame.content)

    detected = await client.call_tool("detect_objects", {"camera_id": "sample"})
    assert not detected.is_error
    text = detected.content[0]
    assert isinstance(text, TextContent)
    assert json.loads(text.text)["model"] == expected_model

    counted = await client.call_tool("count_in_zone", {"camera_id": "sample", "zone": "all"})
    assert not counted.is_error
    text = counted.content[0]
    assert isinstance(text, TextContent)
    assert json.loads(text.text)["zone"] == "all"


async def test_all_tools_work_over_stdio(config_path: Path) -> None:
    parameters = server_parameters(config_path, str(FAKE_SERVER))

    with anyio.fail_after(TIMEOUT_S):
        async with Client(stdio_client(parameters)) as client:
            await check_tools(client, "fake.pt")


@pytest.mark.integration
async def test_all_tools_work_over_stdio_with_the_real_model(config_path: Path) -> None:
    parameters = server_parameters(config_path, *SERVER_MODULE)

    with anyio.fail_after(TIMEOUT_S):
        async with Client(stdio_client(parameters)) as client:
            await check_tools(client, "yolo26n.pt")


def run_server_expecting_failure(config: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *SERVER_MODULE],
        env={**os.environ, CONFIG_ENV_VAR: str(config)},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=TIMEOUT_S,
    )


def test_unavailable_device_ends_server_with_clear_message(tmp_path: Path) -> None:
    """Runs where CUDA is missing; a machine with CUDA would load the model instead."""
    import torch

    if torch.cuda.is_available():
        pytest.skip("needs a machine without CUDA")
    config = tmp_path / "config.yaml"
    config.write_text(
        "detector: {device: cuda}\n"
        f"cameras:\n  - {{id: a, name: A, source: {{type: file, path: '{SAMPLE_VIDEO}'}}}}\n",
        encoding="utf-8",
    )

    result = run_server_expecting_failure(config)

    assert result.returncode == 1
    assert result.stdout == ""
    assert "CUDA is not available" in result.stderr
    assert "Traceback" not in result.stderr

"""Measure tool latency and response sizes for docs/EVALUATION.md (KONZEPT.md 10.1 and 10.2).

Latency is measured end to end: the server runs as a subprocess and is called over stdio by a
real MCP client, as Claude would call it. Image sizes are measured in-process with the same
imaging functions the server uses, but without the configured width limit, so that widths above
``limits.max_image_width`` can be compared as well.
"""

import argparse
import base64
import json
import math
import platform
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import anyio
import cv2
import numpy as np
from grab_frame import grab
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp_types import CallToolResult, ImageContent, TextContent

from camera_mcp.camera_manager import CameraOfflineError, UnknownCameraError
from camera_mcp.config import CONFIG_ENV_VAR, ConfigError, load_config, resolve_config_path
from camera_mcp.imaging import encode_jpeg, resize_to_width
from camera_mcp.logging_setup import setup_logging

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_ARGS = ["-m", "camera_mcp.server"]
IMAGE_WIDTHS = (512, 1024, 1600)
WARMUP_CALLS = 3
ONLINE_TIMEOUT_S = 30.0
SESSION_TIMEOUT_S = 1800.0

# Image cost as documented by Anthropic (platform.claude.com/docs/en/build-with-claude/vision,
# "Resolution and token cost", checked 2026-11-06): one visual token per 28x28 px patch. Larger
# images are first scaled down to the largest aspect-preserving size within the tier's limits.
PATCH_PX = 28
BYTES_PER_TEXT_TOKEN = 4  # Rough rule of thumb for JSON and English text.


@dataclass(frozen=True)
class Tier:
    name: str
    max_edge_px: int
    max_tokens: int


STANDARD = Tier("standard (up to Claude 4.6)", 1568, 1568)
HIGH_RES = Tier("high-res (Claude 4.7 and later)", 2576, 4784)
TIERS = (STANDARD, HIGH_RES)
# Tool responses are counted for the tier of current models; at the default width limit of
# 1024 px both tiers give the same result.
RESPONSE_TIER = HIGH_RES


class BenchmarkError(Exception):
    """The benchmark could not run; the message says why."""


@dataclass(frozen=True)
class Case:
    name: str
    tool: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LatencyResult:
    name: str
    median_ms: float
    p95_ms: float
    inference_median_ms: float | None
    response_bytes: int
    response_tokens: int


@dataclass(frozen=True)
class ImageSizeResult:
    requested_width: int
    delivered: tuple[int, int]
    jpeg_bytes: int
    base64_bytes: int
    seen_by_claude: dict[str, tuple[int, int]]  # by tier name
    tokens: dict[str, int]  # by tier name


def p95(values: Sequence[float]) -> float:
    """95th percentile with linear interpolation between the closest ranks."""
    if len(values) < 2:
        raise ValueError("p95 needs at least two values.")
    return statistics.quantiles(values, n=100, method="inclusive")[94]


def estimate_text_tokens(n_bytes: int) -> int:
    return math.ceil(n_bytes / BYTES_PER_TEXT_TOKEN)


def count_image_tokens(width: int, height: int) -> int:
    return math.ceil(width / PATCH_PX) * math.ceil(height / PATCH_PX)


def image_as_seen_by_claude(width: int, height: int, tier: Tier) -> tuple[int, int]:
    """Size after Claude's own downscaling; follows Anthropic's reference implementation."""

    def fits(w: int, h: int) -> bool:
        return (
            math.ceil(w / PATCH_PX) * PATCH_PX <= tier.max_edge_px
            and math.ceil(h / PATCH_PX) * PATCH_PX <= tier.max_edge_px
            and count_image_tokens(w, h) <= tier.max_tokens
        )

    if fits(width, height):
        return width, height
    if height > width:
        seen_height, seen_width = image_as_seen_by_claude(height, width, tier)
        return seen_width, seen_height
    # Binary search along the long edge for the largest aspect-preserving size that fits.
    aspect_ratio = width / height
    lo, hi = 1, width  # lo always fits, hi never does
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fits(mid, max(round(mid / aspect_ratio), 1)):
            lo = mid
        else:
            hi = mid
    return lo, max(round(lo / aspect_ratio), 1)


def estimate_image_tokens(width: int, height: int, tier: Tier) -> int:
    return count_image_tokens(*image_as_seen_by_claude(width, height, tier))


def _image_size(block: ImageContent) -> tuple[int, int]:
    data = np.frombuffer(base64.b64decode(block.data), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise BenchmarkError("A tool returned an image that cannot be decoded.")
    return image.shape[1], image.shape[0]


def response_size(result: CallToolResult) -> tuple[int, int]:
    """Bytes and estimated tokens of the content blocks, which is what the model reads."""
    n_bytes = tokens = 0
    for block in result.content:
        if isinstance(block, TextContent):
            size = len(block.text.encode("utf-8"))
            n_bytes += size
            tokens += estimate_text_tokens(size)
        elif isinstance(block, ImageContent):
            n_bytes += len(block.data)
            tokens += estimate_image_tokens(*_image_size(block), RESPONSE_TIER)
    return n_bytes, tokens


def _json_payload(result: CallToolResult) -> dict[str, Any]:
    first = result.content[0] if result.content else None
    if not isinstance(first, TextContent):
        raise BenchmarkError("Expected a JSON text block as the first content block.")
    return json.loads(first.text)


def _text(result: CallToolResult) -> str:
    return " ".join(block.text for block in result.content if isinstance(block, TextContent))


async def _call(client: Client, case: Case) -> CallToolResult:
    result = await client.call_tool(case.tool, case.arguments)
    if result.is_error:
        raise BenchmarkError(f"{case.name} failed: {_text(result)}")
    return result


async def _wait_until_online(client: Client, camera_id: str) -> dict[str, Any]:
    deadline = time.monotonic() + ONLINE_TIMEOUT_S
    while True:
        result = await _call(
            client, Case("get_camera_info", "get_camera_info", {"camera_id": camera_id})
        )
        camera = _json_payload(result)["cameras"][0]
        if camera["status"] == "online":
            return camera
        if time.monotonic() >= deadline:
            raise BenchmarkError(
                f"Camera '{camera_id}' did not come online within {ONLINE_TIMEOUT_S:.0f} s."
            )
        await anyio.sleep(0.1)


def build_cases(camera_id: str, zones: Sequence[str]) -> list[Case]:
    cases = [
        Case("get_camera_info", "get_camera_info", {"camera_id": camera_id}),
        Case("get_frame (1024 px)", "get_frame", {"camera_id": camera_id, "max_width": 1024}),
        Case("detect_objects", "detect_objects", {"camera_id": camera_id}),
        Case(
            "detect_objects (include_image)",
            "detect_objects",
            {"camera_id": camera_id, "include_image": True},
        ),
    ]
    if zones:
        cases.append(
            Case(
                f"count_in_zone ({zones[0]})",
                "count_in_zone",
                {"camera_id": camera_id, "zone": zones[0]},
            )
        )
    return cases


async def _measure(client: Client, case: Case, runs: int) -> LatencyResult:
    for _ in range(WARMUP_CALLS):
        await _call(client, case)
    durations: list[float] = []
    inference: list[float] = []
    result: CallToolResult | None = None
    for _ in range(runs):
        start = time.perf_counter()
        result = await _call(client, case)
        durations.append((time.perf_counter() - start) * 1000)
        if case.tool == "detect_objects":  # count_in_zone does not report inference time.
            inference.append(_json_payload(result)["inference_ms"])
    assert result is not None
    n_bytes, tokens = response_size(result)
    return LatencyResult(
        name=case.name,
        median_ms=statistics.median(durations),
        p95_ms=p95(durations),
        inference_median_ms=statistics.median(inference) if inference else None,
        response_bytes=n_bytes,
        response_tokens=tokens,
    )


async def measure_latency(
    server_args: Sequence[str], config_path: Path, camera_id: str, runs: int
) -> tuple[dict[str, Any], list[LatencyResult]]:
    """Returns the camera info and detection metadata, and one result per case."""
    parameters = StdioServerParameters(
        command=sys.executable,
        args=list(server_args),
        env={CONFIG_ENV_VAR: str(config_path.resolve())},
        cwd=REPO_ROOT,  # The model file is kept in the ignored project root.
    )
    with anyio.fail_after(SESSION_TIMEOUT_S):
        async with Client(stdio_client(parameters)) as client:
            camera = await _wait_until_online(client, camera_id)
            probe = await _call(
                client, Case("detect_objects", "detect_objects", {"camera_id": camera_id})
            )
            payload = _json_payload(probe)
            meta = {"camera": camera, "model": payload["model"], "device": payload["device"]}
            results = [
                await _measure(client, case, runs)
                for case in build_cases(camera_id, camera["zones"])
            ]
    return meta, results


def measure_image_sizes(image: np.ndarray, jpeg_quality: int) -> list[ImageSizeResult]:
    results = []
    for width in IMAGE_WIDTHS:
        resized = resize_to_width(image, width)
        data = encode_jpeg(resized, jpeg_quality)
        delivered = (resized.shape[1], resized.shape[0])
        results.append(
            ImageSizeResult(
                requested_width=width,
                delivered=delivered,
                jpeg_bytes=len(data),
                base64_bytes=len(base64.b64encode(data)),
                seen_by_claude={t.name: image_as_seen_by_claude(*delivered, t) for t in TIERS},
                tokens={t.name: estimate_image_tokens(*delivered, t) for t in TIERS},
            )
        )
    return results


def _size(size: tuple[int, int]) -> str:
    return f"{size[0]}x{size[1]}"


def render_markdown(
    header: dict[str, Any], latency: Sequence[LatencyResult], images: Sequence[ImageSizeResult]
) -> str:
    lines = [
        f"## Benchmark: {header['camera_id']} ({header['device']})",
        "",
        f"- Date: {header['date']}",
        f"- Device: {header['device']}, model: {header['model']}",
        f"- Camera: {header['camera_id']}, resolution {header['resolution']}",
        f"- Platform: {header['platform']}, Python {header['python']}",
        f"- Calls per case: {header['runs']} (after {WARMUP_CALLS} warm-up calls)",
        "",
        "### Latency per tool call (MCP over stdio)",
        "",
        "| Call | Median ms | p95 ms | Inference ms (median) | Response bytes | ~Tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in latency:
        inference = "-" if r.inference_median_ms is None else f"{r.inference_median_ms:.1f}"
        lines.append(
            f"| {r.name} | {r.median_ms:.1f} | {r.p95_ms:.1f} | {inference} "
            f"| {r.response_bytes} | {r.response_tokens} |"
        )
    lines += [
        "",
        f"### Image size by width (JPEG quality {header['jpeg_quality']}, without width limit)",
        "",
        "| Requested px | Delivered | JPEG bytes | Base64 bytes "
        "| Standard: seen | Standard: tokens | High-res: seen | High-res: tokens |",
        "|---:|---|---:|---:|---|---:|---|---:|",
    ]
    for r in images:
        cells = [str(r.requested_width), _size(r.delivered), str(r.jpeg_bytes), str(r.base64_bytes)]
        for tier in TIERS:
            cells += [_size(r.seen_by_claude[tier.name]), str(r.tokens[tier.name])]
        lines.append("| " + " | ".join(cells) + " |")
    tiers = "; ".join(
        f"{t.name}: long edge <= {t.max_edge_px} px, <= {t.max_tokens} tokens" for t in TIERS
    )
    lines += [
        "",
        f"Token estimates: text ~ bytes / {BYTES_PER_TEXT_TOKEN}; image = "
        f"ceil(w/{PATCH_PX}) * ceil(h/{PATCH_PX}) after Claude's downscaling ({tiers}). "
        f"Response tokens above use the {RESPONSE_TIER.name} tier.",
    ]
    return "\n".join(lines)


def run(argv: list[str], server_args: Sequence[str] = SERVER_ARGS) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--camera", required=True, help="camera id from the configuration")
    parser.add_argument("--runs", type=int, default=50, help="measured calls per case (at least 2)")
    parser.add_argument("--json", type=Path, help="also write the raw results to this file")
    args = parser.parse_args(argv)
    if args.runs < 2:
        parser.error("--runs must be at least 2.")

    setup_logging()
    config_path = resolve_config_path()
    try:
        config = load_config(config_path)
        # Read before the server starts, so that a missing camera fails fast.
        frame = grab(config, args.camera)
        images = measure_image_sizes(frame.image, config.limits.jpeg_quality)
        meta, latency = anyio.run(measure_latency, server_args, config_path, args.camera, args.runs)
    except (ConfigError, UnknownCameraError, CameraOfflineError, BenchmarkError) as exc:
        print(exc, file=sys.stderr)
        return 1

    height, width = frame.image.shape[:2]
    header = {
        "date": datetime.now().astimezone().isoformat(timespec="seconds"),
        "camera_id": args.camera,
        "resolution": f"{width}x{height}",
        "device": meta["device"],
        "model": meta["model"],
        "platform": platform.platform(),
        "python": platform.python_version(),
        "runs": args.runs,
        "jpeg_quality": config.limits.jpeg_quality,
    }
    print(render_markdown(header, latency, images))
    if args.json:
        raw = {
            "header": header,
            "latency": [asdict(r) for r in latency],
            "image_sizes": [asdict(r) for r in images],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(run(sys.argv[1:]))

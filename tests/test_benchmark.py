import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from camera_mcp.config import CONFIG_ENV_VAR

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
FAKE_SERVER = REPO_ROOT / "tests" / "fake_mcp_server.py"
SAMPLE_VIDEO = (REPO_ROOT / "tests" / "fixtures" / "sample.mp4").as_posix()


@pytest.fixture
def benchmark(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("benchmark", SCRIPTS / "benchmark.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, zones: bool) -> Path:
    text = (
        "cameras:\n"
        f"  - {{id: sample, name: Sample, source: {{type: file, path: '{SAMPLE_VIDEO}'}}}}\n"
    )
    if zones:
        text += "zones:\n  sample:\n    - {name: all, polygon: [[0, 0], [1, 0], [1, 1], [0, 1]]}\n"
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    monkeypatch.setenv(CONFIG_ENV_VAR, str(path))
    return path


def test_p95_interpolates_between_ranks(benchmark: ModuleType) -> None:
    assert benchmark.p95(list(range(1, 101))) == pytest.approx(95.05)
    assert benchmark.p95([10.0, 20.0]) == pytest.approx(19.5)


def test_p95_needs_two_values(benchmark: ModuleType) -> None:
    with pytest.raises(ValueError):
        benchmark.p95([1.0])


def test_text_tokens_are_rounded_up(benchmark: ModuleType) -> None:
    assert benchmark.estimate_text_tokens(0) == 0
    assert benchmark.estimate_text_tokens(5) == 2


@pytest.mark.parametrize(
    ("size", "tier_name", "seen", "tokens"),
    [
        # Examples from the Anthropic vision documentation.
        ((200, 200), "STANDARD", (200, 200), 64),
        ((1000, 1000), "STANDARD", (1000, 1000), 1296),
        ((1920, 1080), "STANDARD", (1456, 819), 1560),
        ((1920, 1080), "HIGH_RES", (1920, 1080), 2691),
        # The docs table says 1269x952; their reference implementation, which is followed
        # here, gives 1270x952. Both cost the same.
        ((2000, 1500), "STANDARD", (1270, 952), 1564),
        ((3840, 2160), "HIGH_RES", (2576, 1449), 4784),
        ((1075, 1520), "STANDARD", (924, 1307), 33 * 47),
    ],
)
def test_image_cost_matches_the_documented_examples(
    benchmark: ModuleType,
    size: tuple[int, int],
    tier_name: str,
    seen: tuple[int, int],
    tokens: int,
) -> None:
    tier = getattr(benchmark, tier_name)

    assert benchmark.image_as_seen_by_claude(*size, tier) == seen
    assert benchmark.estimate_image_tokens(*size, tier) == tokens


def test_long_edge_is_limited(benchmark: ModuleType) -> None:
    seen = benchmark.image_as_seen_by_claude(4000, 200, benchmark.STANDARD)

    assert max(seen) <= benchmark.STANDARD.max_edge_px


def test_image_sizes_ignore_the_width_limit_but_never_enlarge(benchmark: ModuleType) -> None:
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)

    results = benchmark.measure_image_sizes(image, jpeg_quality=80)

    assert [r.delivered for r in results] == [(512, 288), (1024, 576), (1600, 900)]
    assert all(r.base64_bytes > r.jpeg_bytes for r in results)
    small = benchmark.measure_image_sizes(np.zeros((240, 320, 3), dtype=np.uint8), 80)
    assert {r.delivered for r in small} == {(320, 240)}


def test_benchmark_runs_all_cases_against_the_fake_server(
    benchmark: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    write_config(tmp_path, monkeypatch, zones=True)
    raw_path = tmp_path / "out" / "raw.json"

    code = benchmark.run(
        ["--camera", "sample", "--runs", "2", "--json", str(raw_path)], [str(FAKE_SERVER)]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert out.isascii()  # The Windows console cannot print every character.
    for name in [
        "get_camera_info",
        "get_frame (1024 px)",
        "| detect_objects |",
        "detect_objects (include_image)",
        "count_in_zone (all)",
    ]:
        assert name in out
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    assert raw["header"]["model"] == "fake.pt"
    assert len(raw["latency"]) == 5
    assert raw["latency"][2]["inference_median_ms"] is not None
    assert len(raw["image_sizes"]) == 3


def test_camera_without_zones_skips_count_in_zone(
    benchmark: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    write_config(tmp_path, monkeypatch, zones=False)

    code = benchmark.run(["--camera", "sample", "--runs", "2"], [str(FAKE_SERVER)])

    assert code == 0
    assert "count_in_zone" not in capsys.readouterr().out


def test_unknown_camera_fails_before_the_server_starts(
    benchmark: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    write_config(tmp_path, monkeypatch, zones=False)

    code = benchmark.run(["--camera", "nope"], [str(FAKE_SERVER)])

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert "Available cameras: sample" in captured.err


def test_too_few_runs_are_rejected(benchmark: ModuleType) -> None:
    with pytest.raises(SystemExit):
        benchmark.run(["--camera", "sample", "--runs", "1"])

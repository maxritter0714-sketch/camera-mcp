import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from camera_mcp.config import CONFIG_ENV_VAR
from camera_mcp.logging_setup import setup_logging


@pytest.fixture
def restore_root_logger():
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


@pytest.mark.usefixtures("restore_root_logger")
def test_logs_go_to_stderr_only(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging()

    logging.getLogger("camera_mcp.test").info("hello")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "hello" in captured.err


def run_main(config_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "camera_mcp.server"],
        env={**os.environ, CONFIG_ENV_VAR: str(config_path)},
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_main_with_valid_config_exits_cleanly_without_stdout(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "cameras:\n  - {id: desk, name: Desk, source: {type: webcam, index: 0}}\n",
        encoding="utf-8",
    )

    result = run_main(config)

    assert result.returncode == 0
    assert result.stdout == ""
    assert "1 camera(s)" in result.stderr


def test_main_with_invalid_config_reports_on_stderr_and_exits_nonzero(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("cameras: []\n", encoding="utf-8")

    result = run_main(config)

    assert result.returncode == 1
    assert result.stdout == ""
    assert "Invalid configuration" in result.stderr
    assert "cameras" in result.stderr
    assert "Traceback" not in result.stderr

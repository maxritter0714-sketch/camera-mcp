import logging
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest

from camera_mcp.camera_manager import CameraManager, CameraOfflineError, UnknownCameraError
from camera_mcp.config import CameraConfig, FileSource, LimitsConfig, WebcamSource
from camera_mcp.sources import Image, SourceEnded, SourceError

FIXTURES = Path(__file__).resolve().parent / "fixtures"
FAST_LIMITS = LimitsConfig(max_frame_age_s=5, reconnect_interval_s=0.01)
FRAME_DELAY_S = 0.002
TIMEOUT_S = 5.0


def wait_until(condition: Callable[[], bool]) -> None:
    deadline = time.monotonic() + TIMEOUT_S
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.005)


class FakeSource:
    """Each step of ``reads`` is delivered per read; an exception step is raised instead."""

    def __init__(
        self,
        fail_opens: int = 0,
        reads: list[Exception] | None = None,
        block_after: int | None = None,
    ) -> None:
        self.fail_opens = fail_opens
        self.reads = reads or []
        self.block_after = block_after
        self.opens = 0
        self.closes = 0
        self.frames_served = 0
        self.release = threading.Event()

    def open(self) -> None:
        self.opens += 1
        if self.opens <= self.fail_opens:
            raise SourceError("could not open fake")

    def read(self) -> Image:
        if self.block_after is not None and self.frames_served >= self.block_after:
            self.release.wait(TIMEOUT_S)
            raise SourceError("released")
        if self.reads:
            raise self.reads.pop(0)
        time.sleep(FRAME_DELAY_S)
        self.frames_served += 1
        return np.full((48, 64, 3), self.frames_served % 256, dtype=np.uint8)

    def close(self) -> None:
        self.closes += 1


def camera(camera_id: str = "cam") -> CameraConfig:
    return CameraConfig(id=camera_id, name=camera_id, source=WebcamSource(type="webcam"))


@pytest.fixture
def run_manager() -> Iterator[Callable[..., CameraManager]]:
    started: list[CameraManager] = []

    def run(
        *sources: FakeSource,
        limits: LimitsConfig = FAST_LIMITS,
        ids: tuple[str, ...] = ("cam",),
    ) -> CameraManager:
        remaining = iter(sources)
        manager = CameraManager(
            [camera(camera_id) for camera_id in ids],
            limits,
            source_factory=lambda _config: next(remaining),
        )
        manager.start()
        started.append(manager)
        return manager

    yield run
    for manager in started:
        manager.stop()


def has_frame(manager: CameraManager, camera_id: str = "cam") -> bool:
    return manager.status(camera_id).status == "online"


def test_status_reports_online_camera(run_manager) -> None:
    manager = run_manager(FakeSource())
    wait_until(lambda: (manager.status("cam").fps_measured or 0) > 0)

    status = manager.status("cam")

    assert status.id == "cam"
    assert status.source_type == "webcam"
    assert status.status == "online"
    assert status.resolution == (64, 48)
    assert status.last_frame_age_s is not None
    assert status.last_frame_age_s < 5


def test_latest_frame_is_replaced_by_newer_frames(run_manager) -> None:
    manager = run_manager(FakeSource())
    wait_until(lambda: has_frame(manager))

    first = manager.latest_frame("cam")
    wait_until(lambda: manager.latest_frame("cam").captured_monotonic > first.captured_monotonic)

    assert manager.latest_frame("cam") is not first


def test_frame_is_read_only_and_timestamp_has_timezone(run_manager) -> None:
    manager = run_manager(FakeSource())
    wait_until(lambda: has_frame(manager))

    frame = manager.latest_frame("cam")

    assert frame.image.flags.writeable is False
    assert frame.captured_at.utcoffset() is not None


def test_unknown_camera_error_lists_known_cameras(run_manager) -> None:
    manager = run_manager(FakeSource(), FakeSource(), ids=("a", "b"))

    with pytest.raises(UnknownCameraError, match="'x'.*a, b"):
        manager.latest_frame("x")
    with pytest.raises(UnknownCameraError):
        manager.status("x")


def test_camera_without_frame_is_offline(run_manager) -> None:
    manager = run_manager(FakeSource(fail_opens=10**6))

    status = manager.status("cam")

    assert status.status == "offline"
    assert status.resolution is None
    assert status.fps_measured is None
    assert status.last_frame_age_s is None
    with pytest.raises(CameraOfflineError, match="no frame received yet"):
        manager.latest_frame("cam")


def test_reconnects_after_failed_opens(run_manager) -> None:
    source = FakeSource(fail_opens=3)
    manager = run_manager(source)

    wait_until(lambda: has_frame(manager))

    assert source.opens == 4


def test_reconnects_after_lost_connection_and_closes_source(run_manager) -> None:
    source = FakeSource(reads=[SourceError("lost")])
    manager = run_manager(source)

    wait_until(lambda: has_frame(manager))

    assert source.opens == 2
    assert source.closes == 1


def test_repeated_failures_are_logged_once(run_manager, caplog) -> None:
    source = FakeSource(fail_opens=5)
    with caplog.at_level(logging.INFO, logger="camera_mcp.camera_manager"):
        manager = run_manager(source)
        wait_until(lambda: has_frame(manager))

    messages = [record.getMessage() for record in caplog.records]
    assert sum("unavailable" in message for message in messages) == 1
    assert sum("reconnected" in message for message in messages) == 1


def test_unexpected_exception_keeps_thread_alive_and_stays_out_of_log(run_manager, caplog) -> None:
    source = FakeSource(reads=[RuntimeError("rtsp://user:s3cret@camera.invalid")])
    with caplog.at_level(logging.WARNING, logger="camera_mcp.camera_manager"):
        manager = run_manager(source)
        wait_until(lambda: has_frame(manager))

    assert "RuntimeError" in caplog.text
    assert "s3cret" not in caplog.text


def test_stale_frame_makes_camera_offline(run_manager) -> None:
    source = FakeSource(block_after=1)
    limits = LimitsConfig(max_frame_age_s=0.05, reconnect_interval_s=0.01)
    manager = run_manager(source, limits=limits)
    wait_until(lambda: source.frames_served >= 1)

    wait_until(lambda: manager.status("cam").status == "offline")

    status = manager.status("cam")
    assert status.resolution == (64, 48)
    assert status.fps_measured is None
    assert status.last_frame_age_s is not None
    with pytest.raises(CameraOfflineError, match="last frame"):
        manager.latest_frame("cam")
    source.release.set()


def test_source_end_stops_capture_without_reconnect(run_manager) -> None:
    source = FakeSource(reads=[SourceEnded("done")])
    run_manager(source)
    wait_until(lambda: source.closes == 1)

    time.sleep(FAST_LIMITS.reconnect_interval_s * 5)

    assert source.opens == 1


def test_failing_camera_does_not_affect_others(run_manager) -> None:
    manager = run_manager(FakeSource(fail_opens=10**6), FakeSource(), ids=("bad", "good"))

    wait_until(lambda: has_frame(manager, "good"))

    assert [status.status for status in manager.statuses()] == ["offline", "online"]


def test_stop_closes_source_and_ends_capture() -> None:
    source = FakeSource()
    manager = CameraManager([camera()], FAST_LIMITS, source_factory=lambda _config: source)
    manager.start()
    wait_until(lambda: source.frames_served > 0)

    manager.stop()

    served = source.frames_served
    assert source.closes == 1
    time.sleep(FRAME_DELAY_S * 5)
    assert source.frames_served == served


def test_stop_returns_even_if_source_is_stuck(caplog, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("camera_mcp.camera_manager.STOP_TIMEOUT_S", 0.05)
    source = FakeSource(block_after=0)
    manager = CameraManager([camera()], FAST_LIMITS, source_factory=lambda _config: source)
    manager.start()
    wait_until(lambda: source.opens == 1)

    with caplog.at_level(logging.WARNING, logger="camera_mcp.camera_manager"):
        manager.stop()

    assert "did not stop in time" in caplog.text
    source.release.set()


def test_stop_without_start_is_safe() -> None:
    CameraManager([camera()], FAST_LIMITS, source_factory=lambda _config: FakeSource()).stop()


def test_real_video_file_end_to_end() -> None:
    config = CameraConfig(
        id="sample",
        name="Sample",
        source=FileSource(type="file", path=FIXTURES / "sample.mp4"),
    )
    manager = CameraManager([config], FAST_LIMITS)
    manager.start()
    try:
        wait_until(lambda: (manager.status("sample").fps_measured or 0) > 0)

        status = manager.status("sample")
        assert status.resolution == (320, 240)
        assert manager.latest_frame("sample").image.shape == (240, 320, 3)
        # The measured rate must reflect pacing, not the speed of decoding.
        assert status.fps_measured < 100
    finally:
        manager.stop()

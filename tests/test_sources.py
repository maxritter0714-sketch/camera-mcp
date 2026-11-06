import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from camera_mcp.config import FileSource, RtspSource, WebcamSource
from camera_mcp.sources import (
    STREAM_TIMEOUT_MS,
    ImageFileFrameSource,
    SourceEnded,
    SourceError,
    StreamFrameSource,
    VideoFileFrameSource,
    WebcamFrameSource,
    make_source,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SECRET_URL = "rtsp://user:s3cret@camera.invalid:8554/stream"


def write_video(path: Path, frames: int, fps: float) -> Path:
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (64, 48))
    for index in range(frames):
        writer.write(np.full((48, 64, 3), index * 10, dtype=np.uint8))
    writer.release()
    return path


def test_sample_video_fixture_is_readable() -> None:
    source = VideoFileFrameSource(FIXTURES / "sample.mp4", loop=False)
    source.open()
    try:
        assert source.read().shape == (240, 320, 3)
    finally:
        source.close()


def test_video_without_loop_ends(tmp_path: Path) -> None:
    source = VideoFileFrameSource(write_video(tmp_path / "v.mp4", 3, 200), loop=False)
    source.open()
    try:
        for _ in range(3):
            source.read()
        with pytest.raises(SourceEnded):
            source.read()
    finally:
        source.close()


def test_video_with_loop_restarts_from_the_beginning(tmp_path: Path) -> None:
    source = VideoFileFrameSource(write_video(tmp_path / "v.mp4", 3, 200), loop=True)
    source.open()
    try:
        first = source.read()
        for _ in range(2):
            source.read()
        assert abs(int(source.read().mean()) - int(first.mean())) <= 2
    finally:
        source.close()


def test_video_is_paced_at_its_own_frame_rate(tmp_path: Path) -> None:
    source = VideoFileFrameSource(write_video(tmp_path / "v.mp4", 6, 20), loop=False)
    source.open()
    try:
        started = time.monotonic()
        for _ in range(6):
            source.read()
        # Six reads at 20 fps span five intervals of 50 ms; the lower bound has some tolerance.
        assert time.monotonic() - started >= 0.2
    finally:
        source.close()


def test_missing_video_file_raises_source_error(tmp_path: Path) -> None:
    source = VideoFileFrameSource(tmp_path / "absent.mp4", loop=True)

    with pytest.raises(SourceError, match="file not found"):
        source.open()


def test_corrupt_video_file_raises_source_error(tmp_path: Path) -> None:
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"this is not a video")
    source = VideoFileFrameSource(path, loop=True)

    with pytest.raises(SourceError, match="could not open"):
        source.open()


def test_image_with_loop_repeats() -> None:
    source = ImageFileFrameSource(FIXTURES / "sample.png", loop=True)
    source.open()
    try:
        first = source.read()
        assert first.shape == (240, 320, 3)
        assert np.array_equal(source.read(), first)
    finally:
        source.close()


def test_image_without_loop_is_served_once_per_open() -> None:
    source = ImageFileFrameSource(FIXTURES / "sample.png", loop=False)
    source.open()
    try:
        source.read()
        with pytest.raises(SourceEnded):
            source.read()
    finally:
        source.close()


@pytest.mark.parametrize("content", [None, b"", b"this is not an image"])
def test_unusable_image_file_raises_source_error(tmp_path: Path, content: bytes | None) -> None:
    path = tmp_path / "broken.png"
    if content is not None:
        path.write_bytes(content)
    source = ImageFileFrameSource(path, loop=True)

    with pytest.raises(SourceError, match="image file"):
        source.open()


def test_read_before_open_raises_source_error() -> None:
    with pytest.raises(SourceError, match="not open"):
        ImageFileFrameSource(FIXTURES / "sample.png", loop=True).read()
    with pytest.raises(SourceError, match="not open"):
        VideoFileFrameSource(FIXTURES / "sample.mp4", loop=True).read()


def test_close_is_safe_without_open_and_source_can_reopen(tmp_path: Path) -> None:
    source = VideoFileFrameSource(write_video(tmp_path / "v.mp4", 2, 200), loop=False)
    source.close()

    source.open()
    source.read()
    source.close()
    source.open()
    try:
        assert source.read() is not None
    finally:
        source.close()


class FakeCapture:
    def __init__(self, opened: bool, frames: list[np.ndarray]) -> None:
        self._opened = opened
        self._frames = frames
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802 - mirrors cv2.VideoCapture
        return self._opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        return (True, self._frames.pop(0)) if self._frames else (False, None)

    def release(self) -> None:
        self.released = True


def test_stream_uses_ffmpeg_backend_with_timeouts_and_reports_lost_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    capture = FakeCapture(opened=True, frames=[frame])
    calls: list[tuple] = []

    def fake_capture(*args: object) -> FakeCapture:
        calls.append(args)
        return capture

    monkeypatch.setattr(cv2, "VideoCapture", fake_capture)
    source = StreamFrameSource(SECRET_URL)

    source.open()
    assert source.read() is frame
    with pytest.raises(SourceError) as excinfo:
        source.read()
    source.close()

    timeout_ms = STREAM_TIMEOUT_MS
    assert calls == [
        (
            SECRET_URL,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                timeout_ms,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                timeout_ms,
            ],
        )
    ]
    assert "s3cret" not in str(excinfo.value)
    assert capture.released


def test_stream_that_cannot_open_is_released_without_leaking_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = FakeCapture(opened=False, frames=[])
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: capture)
    source = StreamFrameSource(SECRET_URL)

    with pytest.raises(SourceError) as excinfo:
        source.open()

    assert "s3cret" not in str(excinfo.value)
    assert capture.released


def test_webcam_opens_configured_index(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple] = []

    def fake_capture(*args: object) -> FakeCapture:
        calls.append(args)
        return FakeCapture(opened=True, frames=[])

    monkeypatch.setattr(cv2, "VideoCapture", fake_capture)

    WebcamFrameSource(3).open()

    assert calls == [(3,)]


def test_failed_open_writes_nothing_to_stdout(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "broken.mp4"
    path.write_bytes(b"this is not a video")

    with pytest.raises(SourceError):
        VideoFileFrameSource(path, loop=True).open()

    assert capfd.readouterr().out == ""


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (FileSource(type="file", path=Path("a.mp4")), VideoFileFrameSource),
        (FileSource(type="file", path=Path("a.PNG")), ImageFileFrameSource),
        (WebcamSource(type="webcam", index=0), WebcamFrameSource),
        (RtspSource(type="rtsp", url=SECRET_URL), StreamFrameSource),
    ],
)
def test_make_source_selects_implementation(config: object, expected: type) -> None:
    assert isinstance(make_source(config), expected)

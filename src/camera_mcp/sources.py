"""Frame sources: a common blocking read interface for webcams, network streams and files."""

import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import numpy.typing as npt

from camera_mcp.config import FileSource, RtspSource, Source, WebcamSource

Image = npt.NDArray[np.uint8]

DEFAULT_FILE_FPS = 10.0
IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})
# Limits for opening and reading a network stream. OpenCV's default of 30 s let a single
# reconnect attempt to an RTSP camera that is off hang for 30 s (see docs/EVALUATION.md).
STREAM_TIMEOUT_MS = 5000
_STREAM_PARAMS = [
    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
    STREAM_TIMEOUT_MS,
    cv2.CAP_PROP_READ_TIMEOUT_MSEC,
    STREAM_TIMEOUT_MS,
]


class SourceError(Exception):
    """The source is unavailable but may recover. Messages never contain stream URLs."""


class SourceEnded(Exception):
    """A finite source has delivered its last frame and will not produce more."""


class FrameSource(Protocol):
    """A source of frames. ``read`` blocks until a frame is due, like a real camera."""

    def open(self) -> None:
        """Connect to the source. Can be called again after ``close``.

        Raises:
            SourceError: If the source cannot be opened.
        """
        ...

    def read(self) -> Image:
        """Return the next frame as a BGR image.

        Raises:
            SourceError: If the connection is lost.
            SourceEnded: If a finite source is exhausted.
        """
        ...

    def close(self) -> None:
        """Release the source. Safe to call when it is not open."""
        ...


class _Pacer:
    """Limits a loop to a frame rate without bursting to catch up after a delay."""

    def __init__(self, fps: float) -> None:
        self._interval = 1.0 / fps
        self._next_due: float | None = None

    def wait(self) -> None:
        now = time.monotonic()
        due = now if self._next_due is None else self._next_due
        if due > now:
            time.sleep(due - now)
        self._next_due = max(due, now) + self._interval


class _CaptureFrameSource:
    """Shared ``cv2.VideoCapture`` handling. ``label`` appears in messages, so no secrets."""

    def __init__(self, label: str, open_capture: Callable[[], cv2.VideoCapture]) -> None:
        self._label = label
        self._open_capture = open_capture
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        self.close()
        capture = self._open_capture()
        if not capture.isOpened():
            capture.release()
            raise SourceError(f"could not open {self._label}")
        self._capture = capture

    def read(self) -> Image:
        frame = self._grab()
        if frame is None:
            raise SourceError(f"lost connection to {self._label}")
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _grab(self) -> Image | None:
        if self._capture is None:
            raise SourceError(f"{self._label} is not open")
        ok, frame = self._capture.read()
        return frame if ok else None


class WebcamFrameSource(_CaptureFrameSource):
    def __init__(self, index: int) -> None:
        super().__init__(f"webcam {index}", lambda: cv2.VideoCapture(index))


class StreamFrameSource(_CaptureFrameSource):
    """RTSP or HTTP stream. The URL may contain credentials and stays out of all messages."""

    def __init__(self, url: str) -> None:
        super().__init__(
            "network stream", lambda: cv2.VideoCapture(url, cv2.CAP_FFMPEG, _STREAM_PARAMS)
        )


class VideoFileFrameSource(_CaptureFrameSource):
    """Plays a video file at its own frame rate, so that it behaves like a live camera."""

    def __init__(self, path: Path, loop: bool) -> None:
        super().__init__(f"video file {path}", lambda: cv2.VideoCapture(str(path)))
        self._path = path
        self._loop = loop
        self._pacer = _Pacer(DEFAULT_FILE_FPS)

    def open(self) -> None:
        if not self._path.is_file():
            raise SourceError(f"file not found: {self._path}")
        super().open()
        assert self._capture is not None
        self._pacer = _Pacer(_usable_fps(self._capture.get(cv2.CAP_PROP_FPS)))

    def read(self) -> Image:
        frame = self._grab()
        if frame is None:
            if not self._loop:
                raise SourceEnded(f"end of {self._label}")
            assert self._capture is not None
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame = self._grab()
            if frame is None:
                raise SourceError(f"{self._label} contains no readable frames")
        self._pacer.wait()
        return frame


class ImageFileFrameSource:
    """Serves a still image as a camera: repeated if looping, otherwise exactly once."""

    def __init__(self, path: Path, loop: bool) -> None:
        self._path = path
        self._loop = loop
        self._image: Image | None = None
        self._served = False
        self._pacer = _Pacer(DEFAULT_FILE_FPS)

    def open(self) -> None:
        # Decoded from bytes: cv2.imread cannot handle non-ASCII paths on Windows.
        try:
            data = np.fromfile(self._path, dtype=np.uint8)
        except OSError as exc:
            raise SourceError(f"could not read image file {self._path}") from exc
        image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        if image is None:
            raise SourceError(f"could not decode image file {self._path}")
        self._image = image
        self._served = False
        self._pacer = _Pacer(DEFAULT_FILE_FPS)

    def read(self) -> Image:
        if self._image is None:
            raise SourceError(f"image file {self._path} is not open")
        if self._served and not self._loop:
            raise SourceEnded(f"end of image file {self._path}")
        self._pacer.wait()
        self._served = True
        return self._image

    def close(self) -> None:
        self._image = None


def _usable_fps(fps: float) -> float:
    """Containers can report 0 or NaN as frame rate; fall back to a fixed rate then."""
    return fps if math.isfinite(fps) and fps > 0 else DEFAULT_FILE_FPS


def make_source(config: Source) -> FrameSource:
    """Create the frame source described by a camera's source configuration."""
    match config:
        case FileSource(path=path, loop=loop) if path.suffix.lower() in IMAGE_SUFFIXES:
            return ImageFileFrameSource(path, loop)
        case FileSource(path=path, loop=loop):
            return VideoFileFrameSource(path, loop)
        case WebcamSource(index=index):
            return WebcamFrameSource(index)
        case RtspSource(url=url):
            return StreamFrameSource(url)

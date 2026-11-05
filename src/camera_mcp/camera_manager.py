"""Background capture: one thread per camera keeps only the newest frame."""

import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from camera_mcp.config import CameraConfig, LimitsConfig, Source
from camera_mcp.sources import FrameSource, Image, SourceEnded, SourceError, make_source

logger = logging.getLogger(__name__)

FPS_WINDOW_FRAMES = 30
STOP_TIMEOUT_S = 2.0


class CameraError(Exception):
    """A camera request failed. The message is written for the calling model."""


class UnknownCameraError(CameraError):
    def __init__(self, camera_id: str, known_ids: Iterable[str]) -> None:
        known = ", ".join(sorted(known_ids)) or "none"
        super().__init__(f"Unknown camera '{camera_id}'. Available cameras: {known}.")


class CameraOfflineError(CameraError):
    def __init__(self, camera_id: str, last_frame_age_s: float | None) -> None:
        if last_frame_age_s is None:
            detail = "no frame received yet"
        else:
            detail = f"last frame {last_frame_age_s:.0f} s ago"
        super().__init__(f"Camera '{camera_id}' is offline, {detail}. Check get_camera_info.")


@dataclass(frozen=True)
class Frame:
    """A captured frame. ``image`` is read-only because callers share the same array."""

    image: Image
    captured_at: datetime
    captured_monotonic: float

    @property
    def age_s(self) -> float:
        return time.monotonic() - self.captured_monotonic


@dataclass(frozen=True)
class CameraStatus:
    id: str
    name: str
    source_type: str
    status: Literal["online", "offline"]
    resolution: tuple[int, int] | None
    fps_measured: float | None
    last_frame_age_s: float | None


class _CameraWorker:
    """Owns one source and its capture thread; survives every failure of the source."""

    def __init__(self, config: CameraConfig, source: FrameSource, limits: LimitsConfig) -> None:
        self.config = config
        self._source = source
        self._limits = limits
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._frame_times: deque[float] = deque(maxlen=FPS_WINDOW_FRAMES)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"camera-{config.id}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def request_stop(self) -> None:
        self._stop.set()

    def join(self, timeout_s: float) -> bool:
        """Wait for the thread to end. A source stuck in a blocking call may outlast this."""
        if self._thread.is_alive():
            self._thread.join(timeout_s)
        return not self._thread.is_alive()

    def latest(self) -> Frame | None:
        with self._lock:
            return self._latest

    def is_online(self, frame: Frame | None) -> bool:
        return frame is not None and frame.age_s <= self._limits.max_frame_age_s

    def status(self) -> CameraStatus:
        with self._lock:
            frame = self._latest
            times = list(self._frame_times)
        online = self.is_online(frame)
        return CameraStatus(
            id=self.config.id,
            name=self.config.name,
            source_type=self.config.source.type,
            status="online" if online else "offline",
            resolution=(frame.image.shape[1], frame.image.shape[0]) if frame else None,
            fps_measured=_measure_fps(times) if online else None,
            last_frame_age_s=frame.age_s if frame else None,
        )

    def _run(self) -> None:
        camera_id = self.config.id
        healthy = True  # Log only state changes, not every failed reconnect attempt.
        while not self._stop.is_set():
            try:
                try:
                    self._source.open()
                    if not healthy:
                        logger.info("Camera '%s' reconnected.", camera_id)
                    healthy = True
                    self._capture_until_stopped()
                finally:
                    self._source.close()
            except SourceEnded:
                logger.info("Camera '%s' reached the end of its source.", camera_id)
                return
            except SourceError as exc:
                if healthy:
                    logger.warning("Camera '%s' unavailable: %s", camera_id, exc)
                healthy = False
            except Exception as exc:
                # Thread boundary: nothing may escape. Only the type is logged, because the
                # text of foreign exceptions can contain stream URLs with credentials.
                if healthy:
                    logger.warning("Camera '%s' failed: %s", camera_id, type(exc).__name__)
                healthy = False
            self._stop.wait(self._limits.reconnect_interval_s)

    def _capture_until_stopped(self) -> None:
        while not self._stop.is_set():
            self._publish(self._source.read())

    def _publish(self, image: Image) -> None:
        image.flags.writeable = False
        now = time.monotonic()
        frame = Frame(image=image, captured_at=datetime.now().astimezone(), captured_monotonic=now)
        with self._lock:
            self._latest = frame
            self._frame_times.append(now)


def _measure_fps(times: list[float]) -> float | None:
    if len(times) < 2 or times[-1] <= times[0]:
        return None
    return (len(times) - 1) / (times[-1] - times[0])


class CameraManager:
    """Runs all configured cameras in the background. Reads never block on a camera."""

    def __init__(
        self,
        cameras: Iterable[CameraConfig],
        limits: LimitsConfig,
        source_factory: Callable[[Source], FrameSource] = make_source,
    ) -> None:
        self._workers = {
            camera.id: _CameraWorker(camera, source_factory(camera.source), limits)
            for camera in cameras
        }

    def start(self) -> None:
        for worker in self._workers.values():
            worker.start()

    def stop(self) -> None:
        """Stop all cameras. Threads stuck in a blocking read are left behind as daemons."""
        workers = list(self._workers.values())
        for worker in workers:
            worker.request_stop()
        deadline = time.monotonic() + STOP_TIMEOUT_S
        for worker in workers:
            if not worker.join(max(0.0, deadline - time.monotonic())):
                logger.warning("Camera '%s' did not stop in time.", worker.config.id)

    def statuses(self) -> list[CameraStatus]:
        return [worker.status() for worker in self._workers.values()]

    def status(self, camera_id: str) -> CameraStatus:
        return self._worker(camera_id).status()

    def latest_frame(self, camera_id: str) -> Frame:
        """Return the newest frame without waiting.

        Raises:
            UnknownCameraError: If the camera is not configured.
            CameraOfflineError: If there is no frame or the newest one is too old.
        """
        worker = self._worker(camera_id)
        frame = worker.latest()
        if frame is None or not worker.is_online(frame):
            raise CameraOfflineError(camera_id, frame.age_s if frame else None)
        return frame

    def _worker(self, camera_id: str) -> _CameraWorker:
        try:
            return self._workers[camera_id]
        except KeyError:
            raise UnknownCameraError(camera_id, self._workers) from None

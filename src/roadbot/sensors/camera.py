"""Picamera2 camera adapter.

Image capture belongs here; image interpretation belongs in perception.  The
Picamera2 import is deliberately delayed until :meth:`open` so the rest of the
pipeline (and its tests) can run on non-Raspberry Pi machines.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from roadbot.clock import now_ns
from roadbot.messages.common import MessageHeader
from roadbot.messages.sensors import CameraFrame


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or read."""


class Picamera2Camera:
    """Synchronous camera source which emits pipeline ``CameraFrame`` objects."""

    def __init__(
        self,
        resolution: tuple[int, int] = (640, 480),
        pixel_format: str = "RGB888",
        fps: float = 30.0,
        *,
        horizontal_flip: bool = False,
        vertical_flip: bool = False,
        camera_factory: Callable[[], Any] | None = None,
        clock: Callable[[], int] = now_ns,
    ) -> None:
        if any(value <= 0 for value in resolution):
            raise ValueError("camera resolution must be positive")
        if fps <= 0:
            raise ValueError("camera fps must be positive")
        self.resolution = resolution
        self.pixel_format = pixel_format
        self.fps = fps
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self._camera_factory = camera_factory
        self._clock = clock
        self._camera: Any | None = None
        self._sequence = 0

    def open(self) -> None:
        if self._camera is not None:
            return
        try:
            if self._camera_factory is None:
                from libcamera import Transform
                from picamera2 import Picamera2

                camera = Picamera2()
                transform: Any = Transform(
                    hflip=self.horizontal_flip,
                    vflip=self.vertical_flip,
                )
            else:
                camera = self._camera_factory()
                # Fakes used by unit tests need not have libcamera installed.
                transform = {
                    "hflip": self.horizontal_flip,
                    "vflip": self.vertical_flip,
                }

            configuration = camera.create_video_configuration(
                main={"size": self.resolution, "format": self.pixel_format},
                controls={"FrameRate": self.fps},
                transform=transform,
                buffer_count=4,
            )
            camera.configure(configuration)
            camera.start()
        except (ImportError, OSError, RuntimeError) as error:
            try:
                camera.close()  # type: ignore[possibly-undefined]
            except (AttributeError, UnboundLocalError):
                pass
            raise CameraError(f"unable to start Picamera2 camera: {error}") from error

        self._camera = camera
        self._sequence = 0

    def read(self) -> CameraFrame:
        if self._camera is None:
            raise CameraError("camera is not open")

        request: Any | None = None
        try:
            # The pixels and metadata must come from the same completed request.
            request = self._camera.capture_request()
            image = request.make_array("main")
            metadata = dict(request.get_metadata())
        except (OSError, RuntimeError, ValueError) as error:
            raise CameraError(f"unable to capture camera frame: {error}") from error
        finally:
            if request is not None:
                request.release()

        sensor_timestamp = metadata.get("SensorTimestamp")
        timestamp_ns = (
            int(sensor_timestamp)
            if isinstance(sensor_timestamp, (int, float)) and sensor_timestamp > 0
            else self._clock()
        )
        frame = CameraFrame(
            header=MessageHeader(
                timestamp_ns=timestamp_ns,
                sequence=self._sequence,
                source="camera",
            ),
            image=image,
            metadata=metadata,
        )
        self._sequence += 1
        return frame

    def close(self) -> None:
        camera, self._camera = self._camera, None
        if camera is None:
            return
        try:
            camera.stop()
        finally:
            camera.close()

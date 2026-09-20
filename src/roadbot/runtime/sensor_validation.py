"""Boundary validation for live camera and IMU pipeline messages."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from roadbot.clock import now_ns
from roadbot.messages.common import MessageHeader, Vector3
from roadbot.messages.sensors import CameraFrame, ImuSample


@dataclass(frozen=True, slots=True)
class FeedSummary:
    """Observed health of one input feed."""

    name: str
    valid_samples: int
    rate_hz: float
    issues: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.issues


@dataclass(frozen=True, slots=True)
class PipelineReport:
    camera: FeedSummary
    imu: FeedSummary

    @property
    def ok(self) -> bool:
        return self.camera.ok and self.imu.ok


@dataclass(slots=True)
class _FeedState:
    count: int = 0
    first_timestamp_ns: int | None = None
    last_timestamp_ns: int | None = None
    last_sequence: int | None = None
    issues: list[str] | None = None

    def __post_init__(self) -> None:
        self.issues = []


class SensorPipelineValidator:
    """Validate sensor messages at the point where they enter the runtime pipeline.

    The class is thread-safe because camera and IMU acquisition normally run at
    different rates in separate workers.
    """

    def __init__(
        self,
        camera_resolution: tuple[int, int],
        *,
        max_sample_age_s: float = 0.5,
        clock: Callable[[], int] = now_ns,
    ) -> None:
        if any(value <= 0 for value in camera_resolution):
            raise ValueError("camera resolution must be positive")
        if max_sample_age_s <= 0:
            raise ValueError("maximum sample age must be positive")
        self._camera_resolution = camera_resolution
        self._max_age_ns = int(max_sample_age_s * 1_000_000_000)
        self._clock = clock
        self._camera = _FeedState()
        self._imu = _FeedState()
        self._lock = threading.Lock()

    def _check_header(self, name: str, header: MessageHeader, state: _FeedState) -> list[str]:
        issues: list[str] = []
        current_ns = self._clock()
        if header.source != name:
            issues.append(f"{name}: source is {header.source!r}, expected {name!r}")
        if header.sequence < 0:
            issues.append(f"{name}: sequence is negative")
        if header.timestamp_ns <= 0:
            issues.append(f"{name}: timestamp is not positive")
        elif header.timestamp_ns > current_ns + 100_000_000:
            issues.append(f"{name}: timestamp is in the future")
        elif current_ns - header.timestamp_ns > self._max_age_ns:
            age_s = (current_ns - header.timestamp_ns) / 1_000_000_000
            issues.append(f"{name}: sample is stale ({age_s:.3f}s old)")
        if state.last_sequence is not None and header.sequence <= state.last_sequence:
            issues.append(f"{name}: sequence did not increase")
        if state.last_timestamp_ns is not None and header.timestamp_ns <= state.last_timestamp_ns:
            issues.append(f"{name}: timestamp did not increase")
        return issues

    @staticmethod
    def _finite_vector(name: str, vector: Vector3) -> list[str]:
        if not all(math.isfinite(value) for value in (vector.x, vector.y, vector.z)):
            return [f"imu: {name} contains a non-finite value"]
        return []

    @staticmethod
    def _record(state: _FeedState, header: MessageHeader) -> None:
        if state.first_timestamp_ns is None:
            state.first_timestamp_ns = header.timestamp_ns
        state.last_timestamp_ns = header.timestamp_ns
        state.last_sequence = header.sequence
        state.count += 1

    def accept_camera(self, frame: CameraFrame) -> bool:
        """Validate and record a camera message, returning whether it is usable."""
        with self._lock:
            issues = self._check_header("camera", frame.header, self._camera)
            image = np.asarray(frame.image)
            expected_width, expected_height = self._camera_resolution
            if image.size == 0:
                issues.append("camera: image is empty")
            elif image.ndim not in (2, 3):
                issues.append(f"camera: expected a 2D or 3D image, got {image.ndim} dimensions")
            elif image.shape[:2] != (expected_height, expected_width):
                issues.append(
                    "camera: frame size is "
                    f"{image.shape[1]}x{image.shape[0]}, expected {expected_width}x{expected_height}"
                )
            if not isinstance(frame.metadata, dict):
                issues.append("camera: metadata is not a dictionary")
            if issues:
                assert self._camera.issues is not None
                for issue in issues:
                    if issue not in self._camera.issues:
                        self._camera.issues.append(issue)
                return False
            self._record(self._camera, frame.header)
            return True

    def accept_imu(self, sample: ImuSample) -> bool:
        """Validate and record an IMU message, returning whether it is usable."""
        with self._lock:
            issues = self._check_header("imu", sample.header, self._imu)
            issues.extend(self._finite_vector("acceleration", sample.acceleration_mps2))
            issues.extend(self._finite_vector("angular velocity", sample.angular_velocity_rad_s))
            if sample.temperature_c is not None and not math.isfinite(sample.temperature_c):
                issues.append("imu: temperature is not finite")
            if issues:
                assert self._imu.issues is not None
                for issue in issues:
                    if issue not in self._imu.issues:
                        self._imu.issues.append(issue)
                return False
            self._record(self._imu, sample.header)
            return True

    def record_issue(self, feed: str, message: str) -> None:
        """Attach an acquisition error to a feed's final report."""
        if feed not in {"camera", "imu"}:
            raise ValueError(f"unknown sensor feed: {feed}")
        with self._lock:
            state = self._camera if feed == "camera" else self._imu
            assert state.issues is not None
            issue = f"{feed}: {message}"
            if issue not in state.issues:
                state.issues.append(issue)

    @staticmethod
    def _summary(name: str, state: _FeedState, minimum_rate_hz: float) -> FeedSummary:
        issues = list(state.issues or [])
        if state.count < 2:
            rate_hz = 0.0
            issues.append(f"{name}: fewer than two valid samples arrived")
        else:
            assert state.first_timestamp_ns is not None and state.last_timestamp_ns is not None
            elapsed_s = (state.last_timestamp_ns - state.first_timestamp_ns) / 1_000_000_000
            rate_hz = (state.count - 1) / elapsed_s if elapsed_s > 0 else 0.0
            if rate_hz < minimum_rate_hz:
                issues.append(
                    f"{name}: feed rate {rate_hz:.1f} Hz is below {minimum_rate_hz:.1f} Hz"
                )
        return FeedSummary(name, state.count, rate_hz, tuple(issues))

    def report(self, *, minimum_camera_hz: float, minimum_imu_hz: float) -> PipelineReport:
        """Return a final immutable report with minimum-rate checks applied."""
        if minimum_camera_hz <= 0 or minimum_imu_hz <= 0:
            raise ValueError("minimum feed rates must be positive")
        with self._lock:
            return PipelineReport(
                camera=self._summary("camera", self._camera, minimum_camera_hz),
                imu=self._summary("imu", self._imu, minimum_imu_hz),
            )

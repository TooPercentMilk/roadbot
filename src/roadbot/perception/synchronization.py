"""Bounded, camera-anchored approximate-time association on acquisition stamps.

Each source must arrive in timestamp/sequence order. The three sources may
interleave arbitrarily. A single sensor sample may serve multiple camera frames.
This associates measurements; it neither interpolates nor performs state fusion.
"""

from __future__ import annotations

import math
import threading
from collections import Counter, deque
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from roadbot.config import SynchronizationConfig
from roadbot.messages.common import MessageHeader
from roadbot.messages.sensors import CameraFrame, ImuSample, WheelEncoderSample

SensorMessage = CameraFrame | ImuSample | WheelEncoderSample


@dataclass(frozen=True, slots=True)
class SynchronizedFrame:
    camera: CameraFrame
    imu: ImuSample
    encoders: WheelEncoderSample
    imu_offset_ns: int
    encoder_offset_ns: int
    imu_window: tuple[ImuSample, ...]
    previous_camera_timestamp_ns: int | None
    # True only when all samples from the previously emitted camera to this one
    # remain available, with no observed sequence gaps across the boundaries.
    imu_window_complete: bool

    @property
    def header(self) -> MessageHeader:
        return self.camera.header


class SensorSynchronizer:
    """Thread-safe intake with bounded wait, stale rejection and drop counters."""

    def __init__(self, config: SynchronizationConfig | None = None) -> None:
        self.config = config or SynchronizationConfig()
        self._camera: deque[tuple[CameraFrame, int]] = deque()
        self._imu: deque[ImuSample] = deque()
        self._encoders: deque[WheelEncoderSample] = deque()
        self._last: dict[str, MessageHeader] = {}
        self._previous_camera_ns: int | None = None
        self._counters: Counter[str] = Counter()
        self._lock = threading.Lock()

    def reset(self) -> None:
        """Explicitly start a new session after replay seek or sensor restart."""
        with self._lock:
            self._camera.clear()
            self._imu.clear()
            self._encoders.clear()
            self._last.clear()
            self._previous_camera_ns = None
            self._counters.clear()

    def diagnostics(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters) | {
                "camera_buffered": len(self._camera),
                "imu_buffered": len(self._imu),
                "encoders_buffered": len(self._encoders),
            }

    def _valid(self, name: str, message: SensorMessage, now_ns: int) -> bool:
        header = message.header
        reason = None
        if header.clock_domain != self.config.clock_domain:
            reason = "clock_domain"
        elif (
            type(header.timestamp_ns) is not int
            or header.timestamp_ns <= 0
            or type(header.sequence) is not int
            or header.sequence < 0
        ):
            reason = "header"
        elif header.timestamp_ns > now_ns:
            reason = "future"
        elif now_ns - header.timestamp_ns > self.config.max_age_ms * 1e6:
            reason = "stale"
        elif name in self._last and (
            header.timestamp_ns <= self._last[name].timestamp_ns
            or header.sequence <= self._last[name].sequence
        ):
            reason = "out_of_order"
        elif isinstance(message, CameraFrame):
            image = np.asarray(message.image)
            channels = {"bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4}
            if (
                image.ndim != 3
                or image.size == 0
                or image.dtype != np.uint8
                or image.shape[2] != channels.get(message.encoding)
            ):
                reason = "image"
            elif (
                self.config.require_camera_sensor_timestamp
                and header.timestamp_source != "sensor_boottime"
            ):
                reason = "timestamp_source"
        elif isinstance(message, ImuSample):
            a, g = message.acceleration_mps2, message.angular_velocity_rad_s
            values = [a.x, a.y, a.z, g.x, g.y, g.z]
            if message.temperature_c is not None:
                values.append(message.temperature_c)
            if not all(math.isfinite(v) for v in values):
                reason = "nonfinite"
        elif isinstance(message, WheelEncoderSample):
            if type(message.left_count) is not int or type(message.right_count) is not int:
                reason = "counts"
            elif not all(
                v is None or math.isfinite(v) for v in (message.left_rad_s, message.right_rad_s)
            ):
                reason = "nonfinite"
        if reason:
            self._counters[f"{name}_rejected_{reason}"] += 1
            return False
        self._last[name] = header
        self._counters[f"{name}_accepted"] += 1
        return True

    def add(self, message: SensorMessage, *, now_ns: int) -> bool:
        """Use current time from the same domain, or an injected replay clock."""
        with self._lock:
            if isinstance(message, CameraFrame):
                name, buffer, capacity = "camera", self._camera, self.config.camera_buffer_size
                item = (message, now_ns)
            elif isinstance(message, ImuSample):
                name, buffer, capacity = "imu", self._imu, self.config.imu_buffer_size
                item = message
            elif isinstance(message, WheelEncoderSample):
                name, buffer, capacity = "encoders", self._encoders, self.config.encoder_buffer_size
                item = message
            else:
                raise TypeError("unsupported sensor message")
            if not self._valid(name, message, now_ns):
                return False
            if len(buffer) == capacity:
                buffer.popleft()
                self._counters[f"{name}_overflow"] += 1
            buffer.append(item)
            return True

    def poll(self, *, now_ns: int) -> SynchronizedFrame | None:
        """Return one ready frame; call periodically even when a sensor stops.

        Wait for both streams to cross the camera timestamp, which makes nearest
        selection definitive for ordered streams. At the deadline, use the best
        available sample within tolerance or drop the frame. Ties prefer earlier.
        """
        with self._lock:
            while self._camera:
                camera, arrival = self._camera[0]
                stamp = camera.header.timestamp_ns
                if now_ns - stamp > self.config.max_age_ms * 1e6:
                    self._camera.popleft()
                    self._counters["camera_dropped_stale"] += 1
                    continue
                crossed = all(
                    buffer and buffer[-1].header.timestamp_ns >= stamp
                    for buffer in (self._imu, self._encoders)
                )
                if not crossed and now_ns - arrival < self.config.max_wait_ms * 1e6:
                    return None
                self._camera.popleft()
                imu = min(self._imu, key=lambda s: abs(s.header.timestamp_ns - stamp), default=None)
                enc = min(
                    self._encoders, key=lambda s: abs(s.header.timestamp_ns - stamp), default=None
                )
                if (
                    imu is None
                    or enc is None
                    or abs(imu.header.timestamp_ns - stamp) > self.config.imu_tolerance_ms * 1e6
                    or abs(enc.header.timestamp_ns - stamp) > self.config.encoder_tolerance_ms * 1e6
                    or now_ns - imu.header.timestamp_ns > self.config.max_age_ms * 1e6
                    or now_ns - enc.header.timestamp_ns > self.config.max_age_ms * 1e6
                ):
                    self._counters["camera_dropped_unsynchronized"] += 1
                    continue
                previous = self._previous_camera_ns
                window = tuple(
                    s
                    for s in self._imu
                    if (previous is None or s.header.timestamp_ns > previous)
                    and s.header.timestamp_ns <= stamp
                )
                before = [
                    s
                    for s in self._imu
                    if previous is not None and s.header.timestamp_ns <= previous
                ]
                after = next((s for s in self._imu if s.header.timestamp_ns > stamp), None)
                coverage = ([before[-1]] if before else []) + list(window)
                if after is not None:
                    coverage.append(after)
                complete = (
                    bool(before)
                    and bool(coverage)
                    and (coverage[-1].header.timestamp_ns >= stamp)
                    and all(
                        b.header.sequence == a.header.sequence + 1 for a, b in pairwise(coverage)
                    )
                )
                self._previous_camera_ns = stamp
                self._counters["synchronized"] += 1
                return SynchronizedFrame(
                    camera,
                    imu,
                    enc,
                    imu.header.timestamp_ns - stamp,
                    enc.header.timestamp_ns - stamp,
                    window,
                    previous,
                    complete,
                )
            return None

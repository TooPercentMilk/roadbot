"""Read-only, timestamped encoder intake; no motor configuration or commands."""

from __future__ import annotations

import math
from collections.abc import Callable

from roadbot.clock import now_ns
from roadbot.hardware.i2c import I2CBus, SMBus2Bus
from roadbot.hardware.motor_controller import HiwonderMotorController
from roadbot.messages.common import MessageHeader
from roadbot.messages.sensors import WheelEncoderSample


def counter_delta(current: int, previous: int) -> int:
    """Signed difference across a 32-bit counter wrap."""
    return (current - previous + 2**31) % 2**32 - 2**31


class WheelEncoders:
    def __init__(
        self,
        bus_number: int = 1,
        address: int = 0x34,
        left_channel: int = 1,
        right_channel: int = 3,
        counts_per_revolution: float | None = None,
        *,
        bus: I2CBus | None = None,
        clock: Callable[[], int] = now_ns,
    ) -> None:
        if not 1 <= left_channel <= 4 or not 1 <= right_channel <= 4:
            raise ValueError("encoder channels must be in 1..4")
        if left_channel == right_channel:
            raise ValueError("encoder channels must differ")
        if counts_per_revolution is not None and (
            not math.isfinite(counts_per_revolution) or counts_per_revolution <= 0
        ):
            raise ValueError("counts_per_revolution must be positive or null")
        self.bus_number, self.address = bus_number, address
        self.left_channel, self.right_channel = left_channel, right_channel
        self.counts_per_revolution = counts_per_revolution
        self._bus, self._owns_bus = bus, bus is None
        self._clock = clock
        self._controller: HiwonderMotorController | None = None
        self._previous: WheelEncoderSample | None = None
        self._sequence = 0

    def open(self) -> None:
        if self._controller is not None:
            return
        if self._bus is None:
            self._bus = SMBus2Bus(self.bus_number)
        self._controller = HiwonderMotorController(self._bus, self.address)
        self._previous = None
        self._sequence = 0

    def read(self) -> WheelEncoderSample:
        if self._controller is None:
            raise RuntimeError("encoders are not open")
        started_ns = self._clock()
        counts = self._controller.read_encoder_counts()
        received_ns = self._clock()
        timestamp_ns = (started_ns + received_ns) // 2
        left, right = counts.channel(self.left_channel), counts.channel(self.right_channel)
        left_speed = right_speed = None
        previous = self._previous
        if previous is not None and self.counts_per_revolution is not None:
            dt = (timestamp_ns - previous.header.timestamp_ns) / 1e9
            if dt <= 0:
                raise ValueError("encoder clock did not advance")
            scale = math.tau / self.counts_per_revolution / dt
            left_speed = counter_delta(left, previous.left_count) * scale
            right_speed = counter_delta(right, previous.right_count) * scale
        sample = WheelEncoderSample(
            MessageHeader(
                timestamp_ns,
                self._sequence,
                "encoders",
                timestamp_source="host_read_midpoint",
                received_timestamp_ns=received_ns,
                timestamp_uncertainty_ns=(received_ns - started_ns + 1) // 2,
            ),
            left,
            right,
            left_speed,
            right_speed,
        )
        self._previous = sample
        self._sequence += 1
        return sample

    def close(self) -> None:
        self._controller = None
        if self._owns_bus and self._bus is not None:
            self._bus.close()
            self._bus = None

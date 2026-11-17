import math

import pytest

from roadbot.sensors.encoders import WheelEncoders


class ReadOnlyBus:
    def __init__(self):
        self.values = (2**31 - 1, 0, 10, 0)

    def write_then_read(self, address, write, read_length):
        assert address == 0x34 and write == bytes([60]) and read_length == 16
        return b"".join(v.to_bytes(4, "little", signed=True) for v in self.values)


def test_encoder_read_only_midpoint_timestamps_wrap_and_unknown_speed():
    times = iter([1_000_000_000, 1_002_000_000, 2_000_000_000, 2_002_000_000])
    bus = ReadOnlyBus()
    source = WheelEncoders(bus=bus, clock=lambda: next(times), counts_per_revolution=100)
    source.open()
    first = source.read()
    assert first.header.timestamp_ns == 1_001_000_000
    assert first.header.received_timestamp_ns == 1_002_000_000
    assert first.header.timestamp_uncertainty_ns == 1_000_000
    assert first.left_rad_s is None
    bus.values = (-(2**31), 0, 9, 0)
    second = source.read()
    assert second.header.sequence == 1
    assert second.left_rad_s == pytest.approx(math.tau / 100)
    assert second.right_rad_s == pytest.approx(-math.tau / 100)
    source.close()
    source = WheelEncoders(bus=bus, clock=lambda: 10)
    source.open()
    assert source.read().right_rad_s is None
    assert source.read().left_rad_s is None
    source.close()

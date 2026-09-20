import math

import pytest

from roadbot.sensors.imu import (
    ACCEL_CHIP_ID,
    ACCEL_CHIP_ID_REGISTER,
    ACCEL_DATA_REGISTER,
    ACCEL_TEMP_REGISTER,
    GYRO_CHIP_ID,
    GYRO_DATA_REGISTER,
    GYRO_RANGE_REGISTER,
    GYRO_RANGE_REGISTER_VALUE,
    Bmi088Imu,
)


class FakeBus:
    def __init__(self) -> None:
        self.writes: list[tuple[int, int, int]] = []
        self.closed = False

    def read_byte_data(self, address: int, register: int) -> int:
        if register != ACCEL_CHIP_ID_REGISTER:
            raise AssertionError("unexpected identity register")
        return ACCEL_CHIP_ID if address == 0x19 else GYRO_CHIP_ID

    def write_byte_data(self, address: int, register: int, value: int) -> None:
        self.writes.append((address, register, value))

    def write_then_read(self, address: int, write: bytes, read_length: int) -> bytes:
        register = write[0]
        if address == 0x19 and register == ACCEL_DATA_REGISTER:
            # x=0, y=0, z=16384: half of the configured +/-6 g range.
            return bytes.fromhex("000000000040")
        if address == 0x69 and register == GYRO_DATA_REGISTER:
            # x=16384, y=0, z=0: half of the configured +/-250 dps range.
            return bytes.fromhex("004000000000")
        if address == 0x19 and register == ACCEL_TEMP_REGISTER:
            return bytes((0, 0))
        raise AssertionError("unexpected block read")

    def close(self) -> None:
        self.closed = True


def test_imu_initializes_and_converts_samples_to_si_units(monkeypatch) -> None:
    monkeypatch.setattr("roadbot.sensors.imu.time.sleep", lambda _duration: None)
    bus = FakeBus()
    imu = Bmi088Imu(bus=bus, clock=lambda: 456)

    imu.open()
    sample = imu.read()
    imu.close()

    assert len(bus.writes) == 7
    assert (0x69, GYRO_RANGE_REGISTER, GYRO_RANGE_REGISTER_VALUE) in bus.writes
    assert not bus.closed  # The caller retains ownership of an injected bus.
    assert sample.header.timestamp_ns == 456
    assert sample.header.sequence == 0
    assert sample.acceleration_mps2.z == pytest.approx(3.0 * 9.80665)
    assert sample.angular_velocity_rad_s.x == pytest.approx(math.radians(125.0))
    assert sample.temperature_c == 23.0

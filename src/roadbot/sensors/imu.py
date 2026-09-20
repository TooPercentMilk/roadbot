"""BMI088 accelerometer and gyroscope adapter."""

from __future__ import annotations

import math
import time
from collections.abc import Callable

from roadbot.clock import now_ns
from roadbot.hardware.i2c import I2CBus, SMBus2Bus
from roadbot.messages.common import MessageHeader, Vector3
from roadbot.messages.sensors import ImuSample

ACCEL_CHIP_ID_REGISTER = 0x00
ACCEL_CHIP_ID = 0x1E
ACCEL_DATA_REGISTER = 0x12
ACCEL_TEMP_REGISTER = 0x22
ACCEL_CONFIG_REGISTER = 0x40
ACCEL_RANGE_REGISTER = 0x41
ACCEL_POWER_CONFIG_REGISTER = 0x7C
ACCEL_POWER_CONTROL_REGISTER = 0x7D

GYRO_CHIP_ID_REGISTER = 0x00
GYRO_CHIP_ID = 0x0F
GYRO_DATA_REGISTER = 0x02
GYRO_RANGE_REGISTER = 0x0F
GYRO_BANDWIDTH_REGISTER = 0x10
GYRO_POWER_REGISTER = 0x11

STANDARD_GRAVITY_MPS2 = 9.80665
ACCEL_RANGE_G = 6.0
GYRO_RANGE_DPS = 2000.0
ACCEL_ODR_VALUES = {
    100.0: 0x08,
    200.0: 0x09,
    400.0: 0x0A,
}
GYRO_BANDWIDTH_VALUES = {
    100.0: 0x07,
    200.0: 0x06,
    400.0: 0x03,
}


class ImuError(RuntimeError):
    """Raised when the BMI088 cannot be initialized or sampled."""


def _decode_xyz(payload: bytes, scale: float) -> Vector3:
    if len(payload) != 6:
        raise ImuError(f"expected 6 data bytes, received {len(payload)}")
    values = [
        int.from_bytes(payload[offset : offset + 2], "little", signed=True) * scale
        for offset in range(0, 6, 2)
    ]
    return Vector3(*values)


def _decode_temperature(payload: bytes) -> float:
    if len(payload) != 2:
        raise ImuError(f"expected 2 temperature bytes, received {len(payload)}")
    raw = (payload[0] << 3) | (payload[1] >> 5)
    if raw >= 1024:
        raw -= 2048
    return raw * 0.125 + 23.0


class Bmi088Imu:
    """Synchronous BMI088 source which emits SI-unit ``ImuSample`` objects."""

    def __init__(
        self,
        bus_number: int = 1,
        accelerometer_address: int = 0x19,
        gyroscope_address: int = 0x69,
        sample_rate_hz: float = 100.0,
        *,
        bus: I2CBus | None = None,
        clock: Callable[[], int] = now_ns,
    ) -> None:
        if bus_number < 0:
            raise ValueError("I2C bus number must be non-negative")
        if not 0 <= accelerometer_address <= 0x7F:
            raise ValueError("accelerometer address must be a 7-bit value")
        if not 0 <= gyroscope_address <= 0x7F:
            raise ValueError("gyroscope address must be a 7-bit value")
        if sample_rate_hz not in ACCEL_ODR_VALUES:
            supported = ", ".join(f"{rate:g}" for rate in ACCEL_ODR_VALUES)
            raise ValueError(f"unsupported BMI088 sample rate; choose one of: {supported} Hz")
        self.bus_number = bus_number
        self.accelerometer_address = accelerometer_address
        self.gyroscope_address = gyroscope_address
        self.sample_rate_hz = sample_rate_hz
        self._bus = bus
        self._owns_bus = bus is None
        self._clock = clock
        self._open = False
        self._sequence = 0

    def _read(self, address: int, register: int, length: int) -> bytes:
        assert self._bus is not None
        return self._bus.write_then_read(address, bytes([register]), length)

    def open(self) -> None:
        if self._open:
            return
        try:
            if self._bus is None:
                self._bus = SMBus2Bus(self.bus_number)

            accel_id = self._bus.read_byte_data(
                self.accelerometer_address, ACCEL_CHIP_ID_REGISTER
            )
            gyro_id = self._bus.read_byte_data(self.gyroscope_address, GYRO_CHIP_ID_REGISTER)
            if accel_id != ACCEL_CHIP_ID:
                raise ImuError(
                    f"unexpected accelerometer chip ID 0x{accel_id:02X} "
                    f"(expected 0x{ACCEL_CHIP_ID:02X})"
                )
            if gyro_id != GYRO_CHIP_ID:
                raise ImuError(
                    f"unexpected gyroscope chip ID 0x{gyro_id:02X} "
                    f"(expected 0x{GYRO_CHIP_ID:02X})"
                )

            # Accelerometer: active mode, configured ODR, normal bandwidth, +/-6 g.
            self._bus.write_byte_data(
                self.accelerometer_address, ACCEL_POWER_CONFIG_REGISTER, 0x00
            )
            time.sleep(0.005)
            self._bus.write_byte_data(
                self.accelerometer_address, ACCEL_POWER_CONTROL_REGISTER, 0x04
            )
            accel_config = 0xA0 | ACCEL_ODR_VALUES[self.sample_rate_hz]
            self._bus.write_byte_data(
                self.accelerometer_address, ACCEL_CONFIG_REGISTER, accel_config
            )
            self._bus.write_byte_data(self.accelerometer_address, ACCEL_RANGE_REGISTER, 0x01)

            # Gyroscope: normal mode, configured ODR/bandwidth, +/-2000 deg/s.
            self._bus.write_byte_data(self.gyroscope_address, GYRO_POWER_REGISTER, 0x00)
            self._bus.write_byte_data(self.gyroscope_address, GYRO_RANGE_REGISTER, 0x00)
            self._bus.write_byte_data(
                self.gyroscope_address,
                GYRO_BANDWIDTH_REGISTER,
                GYRO_BANDWIDTH_VALUES[self.sample_rate_hz],
            )
            time.sleep(0.05)
        except ImuError:
            self.close()
            raise
        except (OSError, ValueError) as error:
            self.close()
            raise ImuError(f"unable to initialize BMI088: {error}") from error

        self._sequence = 0
        self._open = True

    def read(self) -> ImuSample:
        if not self._open:
            raise ImuError("IMU is not open")
        try:
            acceleration = _decode_xyz(
                self._read(self.accelerometer_address, ACCEL_DATA_REGISTER, 6),
                ACCEL_RANGE_G * STANDARD_GRAVITY_MPS2 / 32768.0,
            )
            angular_velocity = _decode_xyz(
                self._read(self.gyroscope_address, GYRO_DATA_REGISTER, 6),
                math.radians(GYRO_RANGE_DPS) / 32768.0,
            )
            temperature = _decode_temperature(
                self._read(self.accelerometer_address, ACCEL_TEMP_REGISTER, 2)
            )
        except OSError as error:
            raise ImuError(f"unable to read BMI088: {error}") from error

        sample = ImuSample(
            header=MessageHeader(
                timestamp_ns=self._clock(),
                sequence=self._sequence,
                source="imu",
            ),
            acceleration_mps2=acceleration,
            angular_velocity_rad_s=angular_velocity,
            temperature_c=temperature,
        )
        self._sequence += 1
        return sample

    def close(self) -> None:
        self._open = False
        if self._owns_bus and self._bus is not None:
            self._bus.close()
            self._bus = None

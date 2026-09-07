"""Hiwonder I2C motor controller and encoder telemetry adapter."""

from dataclasses import dataclass
from typing import cast

from roadbot.hardware.i2c import I2CBus

DEFAULT_I2C_ADDRESS = 0x34
ENCODER_TOTAL_REGISTER = 60
MOTOR_CHANNEL_COUNT = 4
ENCODER_RESPONSE_SIZE = MOTOR_CHANNEL_COUNT * 4


class MotorControllerError(RuntimeError):
    """Raised when the motor controller returns invalid telemetry."""


@dataclass(frozen=True, slots=True)
class EncoderCounts:
    """Cumulative signed encoder counts for motor channels M1 through M4."""

    values: tuple[int, int, int, int]

    def channel(self, channel_number: int) -> int:
        if not 1 <= channel_number <= MOTOR_CHANNEL_COUNT:
            raise ValueError("channel number must be in the range 1..4")
        return self.values[channel_number - 1]


def decode_encoder_counts(payload: bytes) -> EncoderCounts:
    """Decode the controller's four little-endian signed 32-bit counters."""
    if len(payload) != ENCODER_RESPONSE_SIZE:
        raise MotorControllerError(
            f"expected {ENCODER_RESPONSE_SIZE} encoder bytes, received {len(payload)}"
        )

    decoded = tuple(
        int.from_bytes(payload[offset : offset + 4], byteorder="little", signed=True)
        for offset in range(0, ENCODER_RESPONSE_SIZE, 4)
    )
    return EncoderCounts(cast(tuple[int, int, int, int], decoded))


class HiwonderMotorController:
    """Read-only telemetry interface for the four-channel controller."""

    def __init__(self, bus: I2CBus, address: int = DEFAULT_I2C_ADDRESS) -> None:
        if not 0 <= address <= 0x7F:
            raise ValueError("I2C address must be a 7-bit value")
        self._bus = bus
        self.address = address

    def read_encoder_counts(self) -> EncoderCounts:
        try:
            payload = self._bus.write_then_read(
                self.address,
                bytes([ENCODER_TOTAL_REGISTER]),
                ENCODER_RESPONSE_SIZE,
            )
        except OSError as error:
            raise MotorControllerError(
                f"unable to read controller at 0x{self.address:02X}: {error}"
            ) from error
        return decode_encoder_counts(payload)

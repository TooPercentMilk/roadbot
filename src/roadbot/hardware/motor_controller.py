"""Hiwonder I2C motor controller and encoder telemetry adapter."""

from dataclasses import dataclass
from typing import cast

from roadbot.hardware.i2c import I2CBus

DEFAULT_I2C_ADDRESS = 0x34
MOTOR_TYPE_REGISTER = 20
MOTOR_ENCODER_POLARITY_REGISTER = 21
MOTOR_FIXED_SPEED_REGISTER = 51
ENCODER_TOTAL_REGISTER = 60
MOTOR_CHANNEL_COUNT = 4
ENCODER_RESPONSE_SIZE = MOTOR_CHANNEL_COUNT * 4
MINIMUM_CLOSED_LOOP_SPEED = -50
MAXIMUM_CLOSED_LOOP_SPEED = 50
MOTOR_TYPE_WITHOUT_ENCODER = 0
MOTOR_TYPE_TT = 1
MOTOR_TYPE_N20 = 2
MOTOR_TYPE_JGB = 3


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
    """I2C control and telemetry interface for the four-channel controller."""

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

    def configure_encoder(self, motor_type: int, polarity: int) -> None:
        """Configure the controller's closed-loop encoder mode."""
        if motor_type not in range(MOTOR_TYPE_WITHOUT_ENCODER, MOTOR_TYPE_JGB + 1):
            raise ValueError("motor type must be in the range 0..3")
        if polarity not in (0, 1):
            raise ValueError("encoder polarity must be 0 or 1")
        try:
            self._bus.write_byte_data(self.address, MOTOR_TYPE_REGISTER, motor_type)
            self._bus.write_byte_data(
                self.address,
                MOTOR_ENCODER_POLARITY_REGISTER,
                polarity,
            )
        except OSError as error:
            raise MotorControllerError(
                f"unable to configure controller at 0x{self.address:02X}: {error}"
            ) from error

    def set_motor_speeds(self, speeds: tuple[int, int, int, int]) -> None:
        """Set four closed-loop targets in encoder pulses per 10 milliseconds."""
        if len(speeds) != MOTOR_CHANNEL_COUNT:
            raise ValueError(f"exactly {MOTOR_CHANNEL_COUNT} motor speeds are required")
        if any(
            speed < MINIMUM_CLOSED_LOOP_SPEED or speed > MAXIMUM_CLOSED_LOOP_SPEED
            for speed in speeds
        ):
            raise ValueError(
                "closed-loop motor speeds must be between "
                f"{MINIMUM_CLOSED_LOOP_SPEED} and {MAXIMUM_CLOSED_LOOP_SPEED}"
            )
        payload = bytes(speed & 0xFF for speed in speeds)
        try:
            self._bus.write_block_data(self.address, MOTOR_FIXED_SPEED_REGISTER, payload)
        except OSError as error:
            raise MotorControllerError(
                f"unable to command controller at 0x{self.address:02X}: {error}"
            ) from error

    def stop(self) -> None:
        """Command all four motor channels to zero speed."""
        self.set_motor_speeds((0, 0, 0, 0))

import pytest

from roadbot.hardware.motor_controller import (
    ENCODER_TOTAL_REGISTER,
    MOTOR_ENCODER_POLARITY_REGISTER,
    MOTOR_FIXED_SPEED_REGISTER,
    MOTOR_TYPE_JGB,
    MOTOR_TYPE_REGISTER,
    HiwonderMotorController,
    MotorControllerError,
    decode_encoder_counts,
)


class FakeBus:
    def __init__(self, response: bytes) -> None:
        self.response = response
        self.last_transaction: tuple[int, bytes, int] | None = None
        self.byte_writes: list[tuple[int, int, int]] = []
        self.block_writes: list[tuple[int, int, bytes]] = []

    def write_byte_data(self, address: int, register: int, value: int) -> None:
        self.byte_writes.append((address, register, value))

    def write_block_data(self, address: int, register: int, data: bytes) -> None:
        self.block_writes.append((address, register, data))

    def write_then_read(self, address: int, write: bytes, read_length: int) -> bytes:
        self.last_transaction = (address, write, read_length)
        return self.response


def encode_counts(*values: int) -> bytes:
    return b"".join(value.to_bytes(4, byteorder="little", signed=True) for value in values)


def test_decode_encoder_counts_preserves_sign_and_channel_order() -> None:
    counts = decode_encoder_counts(encode_counts(123, -456, 2_147_483_647, -2_147_483_648))

    assert counts.values == (123, -456, 2_147_483_647, -2_147_483_648)


def test_decode_encoder_counts_rejects_short_response() -> None:
    with pytest.raises(MotorControllerError, match="expected 16"):
        decode_encoder_counts(b"\x00" * 12)


def test_controller_uses_documented_register_and_address() -> None:
    bus = FakeBus(encode_counts(1, 2, 3, 4))
    controller = HiwonderMotorController(bus, address=0x34)  # type: ignore[arg-type]

    result = controller.read_encoder_counts()

    assert result.values == (1, 2, 3, 4)
    assert bus.last_transaction == (0x34, bytes([ENCODER_TOTAL_REGISTER]), 16)


def test_channel_numbers_are_one_based() -> None:
    counts = decode_encoder_counts(encode_counts(10, 20, 30, 40))

    assert counts.channel(1) == 10
    assert counts.channel(4) == 40
    with pytest.raises(ValueError):
        counts.channel(0)


def test_controller_configures_jgb_encoder_mode() -> None:
    bus = FakeBus(encode_counts(0, 0, 0, 0))
    controller = HiwonderMotorController(bus, address=0x34)  # type: ignore[arg-type]

    controller.configure_encoder(MOTOR_TYPE_JGB, polarity=0)

    assert bus.byte_writes == [
        (0x34, MOTOR_TYPE_REGISTER, MOTOR_TYPE_JGB),
        (0x34, MOTOR_ENCODER_POLARITY_REGISTER, 0),
    ]


def test_controller_encodes_signed_speed_targets_and_stop() -> None:
    bus = FakeBus(encode_counts(0, 0, 0, 0))
    controller = HiwonderMotorController(bus, address=0x34)  # type: ignore[arg-type]

    controller.set_motor_speeds((8, 0, -8, 0))
    controller.stop()

    assert bus.block_writes == [
        (0x34, MOTOR_FIXED_SPEED_REGISTER, bytes((8, 0, 248, 0))),
        (0x34, MOTOR_FIXED_SPEED_REGISTER, bytes(4)),
    ]


def test_controller_rejects_out_of_range_speed() -> None:
    bus = FakeBus(encode_counts(0, 0, 0, 0))
    controller = HiwonderMotorController(bus, address=0x34)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="between -50 and 50"):
        controller.set_motor_speeds((51, 0, 0, 0))

    assert bus.block_writes == []


import importlib.util
import sys
from pathlib import Path

from roadbot.hardware.motor_controller import ENCODER_TOTAL_REGISTER
from roadbot.sensors.imu import ACCEL_CHIP_ID, GYRO_CHIP_ID

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "tools" / "check_i2c.py"
SPEC = importlib.util.spec_from_file_location("check_i2c", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
check_i2c = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = check_i2c
SPEC.loader.exec_module(check_i2c)


class FakeBus:
    def __init__(self, *, failed_address: int | None = None) -> None:
        self.failed_address = failed_address
        self.byte_reads: list[tuple[int, int]] = []
        self.block_reads: list[tuple[int, bytes, int]] = []
        self.configuration_writes: list[tuple[int, int, int]] = []

    def read_byte_data(self, address: int, register: int) -> int:
        self.byte_reads.append((address, register))
        if address == self.failed_address:
            raise OSError("no acknowledgement")
        return ACCEL_CHIP_ID if address == 0x19 else GYRO_CHIP_ID

    def write_byte_data(self, address: int, register: int, value: int) -> None:
        self.configuration_writes.append((address, register, value))

    def write_then_read(self, address: int, write: bytes, read_length: int) -> bytes:
        self.block_reads.append((address, write, read_length))
        if address == self.failed_address:
            raise OSError("no acknowledgement")
        return b"".join(value.to_bytes(4, "little", signed=True) for value in (1, 2, 3, 4))

    def close(self) -> None:
        pass


def settings():
    return check_i2c.I2CSettings(1, 0x19, 0x69, 0x34)


def test_probe_verifies_all_devices_without_configuration_writes() -> None:
    bus = FakeBus()

    results = check_i2c.probe_devices(bus, settings())

    assert all(result.ok for result in results)
    assert [result.address for result in results] == [0x19, 0x69, 0x34]
    assert bus.byte_reads == [(0x19, 0x00), (0x69, 0x00)]
    assert bus.block_reads == [(0x34, bytes([ENCODER_TOTAL_REGISTER]), 16)]
    assert bus.configuration_writes == []


def test_probe_reports_one_failure_and_still_checks_other_devices() -> None:
    bus = FakeBus(failed_address=0x69)

    results = check_i2c.probe_devices(bus, settings())

    assert [result.ok for result in results] == [True, False, True]
    assert "no acknowledgement" in results[1].detail
    assert bus.block_reads  # The motor controller was still checked.

"""Read-only communication check for every configured Roadbot I2C device.

The BMI088 accelerometer and gyroscope are checked through their chip-ID
registers. The Hiwonder motor controller is checked by reading its encoder
telemetry. No motor speed, direction, PWM, or IMU configuration commands are
sent.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Permit ``python tools/check_i2c.py`` from an uninstalled checkout.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from roadbot.hardware.i2c import I2CBus, SMBus2Bus
from roadbot.hardware.motor_controller import HiwonderMotorController, MotorControllerError
from roadbot.sensors.imu import (
    ACCEL_CHIP_ID,
    ACCEL_CHIP_ID_REGISTER,
    GYRO_CHIP_ID,
    GYRO_CHIP_ID_REGISTER,
)

DEFAULT_CONFIG = REPOSITORY_ROOT / "config" / "robot.yaml"


@dataclass(frozen=True, slots=True)
class I2CSettings:
    bus: int
    accelerometer_address: int
    gyroscope_address: int
    motor_controller_address: int


@dataclass(frozen=True, slots=True)
class ProbeResult:
    name: str
    address: int
    ok: bool
    detail: str


def parse_address(value: str) -> int:
    """Parse and validate a decimal or 0x-prefixed 7-bit I2C address."""
    try:
        address = int(value, 0)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid I2C address: {value}") from error
    if not 0 <= address <= 0x7F:
        raise argparse.ArgumentTypeError("I2C address must be between 0x00 and 0x7F")
    return address


def _address(value: Any, name: str) -> int:
    address = int(value)
    if not 0 <= address <= 0x7F:
        raise ValueError(f"{name} must be a 7-bit I2C address")
    return address


def load_settings(path: Path) -> I2CSettings:
    """Load the shared-bus device addresses from robot.yaml."""
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        imu = document["imu"]
        motor = document["motor_controller"]
        imu_bus = int(imu.get("bus", 1))
        motor_bus = int(motor.get("bus", 1))
        settings = I2CSettings(
            bus=imu_bus,
            accelerometer_address=_address(
                imu.get("accelerometer_address", 0x19), "accelerometer address"
            ),
            gyroscope_address=_address(
                imu.get("gyroscope_address", 0x69), "gyroscope address"
            ),
            motor_controller_address=_address(
                motor.get("address", 0x34), "motor-controller address"
            ),
        )
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read I2C settings from {path}: {error}") from error

    if imu_bus < 0 or motor_bus < 0:
        raise ValueError("I2C bus numbers must be non-negative")
    if imu_bus != motor_bus:
        raise ValueError(
            f"this check requires one shared bus, but IMU uses bus {imu_bus} "
            f"and motor controller uses bus {motor_bus}"
        )

    addresses = (
        settings.accelerometer_address,
        settings.gyroscope_address,
        settings.motor_controller_address,
    )
    if len(set(addresses)) != len(addresses):
        raise ValueError("configured I2C device addresses must be unique")
    return settings


def _probe_chip_id(
    bus: I2CBus,
    *,
    name: str,
    address: int,
    register: int,
    expected: int,
) -> ProbeResult:
    try:
        observed = bus.read_byte_data(address, register)
    except OSError as error:
        return ProbeResult(name, address, False, f"read failed: {error}")
    if observed != expected:
        return ProbeResult(
            name,
            address,
            False,
            f"wrong chip ID 0x{observed:02X}; expected 0x{expected:02X}",
        )
    return ProbeResult(name, address, True, f"chip ID 0x{observed:02X}")


def probe_devices(bus: I2CBus, settings: I2CSettings) -> tuple[ProbeResult, ...]:
    """Probe all configured devices and continue after individual failures."""
    results = [
        _probe_chip_id(
            bus,
            name="BMI088 accelerometer",
            address=settings.accelerometer_address,
            register=ACCEL_CHIP_ID_REGISTER,
            expected=ACCEL_CHIP_ID,
        ),
        _probe_chip_id(
            bus,
            name="BMI088 gyroscope",
            address=settings.gyroscope_address,
            register=GYRO_CHIP_ID_REGISTER,
            expected=GYRO_CHIP_ID,
        ),
    ]

    try:
        counts = HiwonderMotorController(
            bus, settings.motor_controller_address
        ).read_encoder_counts()
        detail = "encoder telemetry readable: " + ", ".join(
            f"M{index}={count}" for index, count in enumerate(counts.values, start=1)
        )
        results.append(
            ProbeResult(
                "Hiwonder motor controller",
                settings.motor_controller_address,
                True,
                detail,
            )
        )
    except (MotorControllerError, OSError, ValueError) as error:
        results.append(
            ProbeResult(
                "Hiwonder motor controller",
                settings.motor_controller_address,
                False,
                f"telemetry read failed: {error}",
            )
        )
    return tuple(results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--bus", type=int, help="override the I2C bus from robot.yaml")
    parser.add_argument("--accelerometer-address", type=parse_address)
    parser.add_argument("--gyroscope-address", type=parse_address)
    parser.add_argument("--motor-address", type=parse_address)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        configured = load_settings(args.config)
        bus_number = configured.bus if args.bus is None else args.bus
        if bus_number < 0:
            parser.error("--bus must be non-negative")
        settings = I2CSettings(
            bus=bus_number,
            accelerometer_address=(
                configured.accelerometer_address
                if args.accelerometer_address is None
                else args.accelerometer_address
            ),
            gyroscope_address=(
                configured.gyroscope_address
                if args.gyroscope_address is None
                else args.gyroscope_address
            ),
            motor_controller_address=(
                configured.motor_controller_address
                if args.motor_address is None
                else args.motor_address
            ),
        )

        print(f"Checking shared I2C bus {settings.bus} (read-only device probes)...")
        print("No motor speed, direction, PWM, or IMU configuration commands will be sent.")
        with SMBus2Bus(settings.bus) as bus:
            results = probe_devices(bus, settings)
    except (ImportError, OSError, ValueError) as error:
        print(f"ERROR: unable to open or configure the I2C bus: {error}", file=sys.stderr)
        print("Check that I2C is enabled and /dev/i2c-1 exists.", file=sys.stderr)
        return 1

    for result in results:
        status = "PASS" if result.ok else "FAIL"
        stream = sys.stdout if result.ok else sys.stderr
        print(
            f"{status} 0x{result.address:02X} {result.name}: {result.detail}",
            file=stream,
        )

    if all(result.ok for result in results):
        print("PASS: all configured devices communicate over the shared SDA/SCL bus.")
        return 0

    print(
        "FAIL: one or more devices did not respond correctly. Check power, common ground, "
        "SDA/SCL continuity, and address configuration.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


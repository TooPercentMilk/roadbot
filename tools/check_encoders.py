"""Read-only Hiwonder motor-encoder diagnostic.

This tool reads cumulative encoder counters. It never writes motor speed or PWM
commands. For a motion check, safely raise the driven wheels and rotate each one
by hand while the tool is running.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import TextIO

import yaml

from roadbot.hardware.i2c import SMBus2Bus
from roadbot.hardware.motor_controller import (
    DEFAULT_I2C_ADDRESS,
    HiwonderMotorController,
    MotorControllerError,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "robot.yaml"
UINT32_MODULUS = 1 << 32
INT32_HALF_RANGE = 1 << 31


def parse_address(value: str) -> int:
    """Parse a decimal or 0x-prefixed I2C address."""
    address = int(value, 0)
    if not 0 <= address <= 0x7F:
        raise argparse.ArgumentTypeError("I2C address must be between 0x00 and 0x7F")
    return address


def counter_delta(current: int, previous: int) -> int:
    """Calculate a signed delta that tolerates a 32-bit counter rollover."""
    delta = (current - previous) % UINT32_MODULUS
    if delta >= INT32_HALF_RANGE:
        delta -= UINT32_MODULUS
    return delta


def load_defaults(path: Path) -> tuple[int, int, tuple[int, int]]:
    """Read I2C and driven-channel defaults from robot.yaml."""
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        controller = document["motor_controller"]
        bus = int(controller.get("bus", 1))
        address = int(controller.get("address", DEFAULT_I2C_ADDRESS))
        channels = (
            int(controller.get("left_channel", 1)),
            int(controller.get("right_channel", 3)),
        )
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read motor-controller settings from {path}: {error}") from error

    if any(channel not in range(1, 5) for channel in channels):
        raise ValueError("configured motor channels must be in the range 1..4")
    return bus, address, channels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--bus", type=int, help="Linux I2C bus number; defaults to robot.yaml")
    parser.add_argument(
        "--address",
        type=parse_address,
        help="controller address in decimal or hex; defaults to robot.yaml",
    )
    parser.add_argument("--interval", type=float, default=0.25, help="seconds between samples")
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="test duration in seconds; use 0 to run until Ctrl+C",
    )
    parser.add_argument(
        "--expect-motion",
        action="store_true",
        help="fail unless both configured drive encoders register activity",
    )
    parser.add_argument(
        "--minimum-activity",
        type=int,
        default=1,
        help="minimum accumulated absolute count change per driven channel",
    )
    parser.add_argument("--output", type=Path, help="optional destination for CSV samples")
    return parser


def validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.bus is not None and args.bus < 0:
        parser.error("--bus must be non-negative")
    if args.interval <= 0:
        parser.error("--interval must be greater than zero")
    if args.duration < 0:
        parser.error("--duration must be zero or greater")
    if args.minimum_activity < 1:
        parser.error("--minimum-activity must be at least one")


def open_csv(path: Path | None):
    if path is None:
        return nullcontext(None)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", newline="", encoding="utf-8")


def print_sample(
    elapsed: float,
    counts: tuple[int, int, int, int],
    deltas: tuple[int, int, int, int],
    rates: tuple[float, float, float, float],
) -> None:
    channel_text = "  ".join(
        f"M{index}={count:>11d} delta={delta:>7d} rate={rate:>9.1f} count/s"
        for index, (count, delta, rate) in enumerate(zip(counts, deltas, rates), start=1)
    )
    print(f"{elapsed:7.2f}s  {channel_text}")


def _make_csv_writer(csv_file: TextIO | None) -> csv.writer | None:
    if csv_file is None:
        return None
    writer = csv.writer(csv_file)
    writer.writerow(
        [
            "elapsed_s",
            "m1_count",
            "m2_count",
            "m3_count",
            "m4_count",
            "m1_delta",
            "m2_delta",
            "m3_delta",
            "m4_delta",
            "m1_count_per_s",
            "m2_count_per_s",
            "m3_count_per_s",
            "m4_count_per_s",
        ]
    )
    return writer


def run(args: argparse.Namespace) -> int:
    config_bus, config_address, drive_channels = load_defaults(args.config)
    bus_number = config_bus if args.bus is None else args.bus
    address = config_address if args.address is None else args.address

    print(f"Reading encoder totals from I2C bus {bus_number}, address 0x{address:02X}")
    print(f"Configured drive channels: M{drive_channels[0]} and M{drive_channels[1]}")
    print("This is read-only: no motor or steering commands will be sent.")
    if args.expect_motion:
        print("Safely raise the drive wheels, then rotate both by hand during the test.")

    sample_count = 0
    activity = [0, 0, 0, 0]
    start_time = time.monotonic()
    previous_time = start_time

    try:
        with SMBus2Bus(bus_number) as bus, open_csv(args.output) as csv_file:
            controller = HiwonderMotorController(bus, address)
            previous = controller.read_encoder_counts().values
            csv_writer = _make_csv_writer(csv_file)

            while args.duration == 0 or time.monotonic() - start_time < args.duration:
                time.sleep(args.interval)
                now = time.monotonic()
                current = controller.read_encoder_counts().values
                elapsed_interval = now - previous_time
                deltas = tuple(counter_delta(value, old) for value, old in zip(current, previous))
                rates = tuple(delta / elapsed_interval for delta in deltas)

                for index, delta in enumerate(deltas):
                    activity[index] += abs(delta)

                elapsed = now - start_time
                print_sample(elapsed, current, deltas, rates)
                if csv_writer is not None:
                    csv_writer.writerow([elapsed, *current, *deltas, *rates])

                sample_count += 1
                previous = current
                previous_time = now
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except (MotorControllerError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        print(
            "Check controller power, common ground, SDA/SCL wiring, and I2C enablement.",
            file=sys.stderr,
        )
        return 1

    if sample_count == 0:
        print("ERROR: no samples were collected", file=sys.stderr)
        return 1

    print(f"\nRead {sample_count} samples successfully.")
    print("Accumulated activity: " + ", ".join(f"M{i}={value}" for i, value in enumerate(activity, 1)))

    inactive = [
        channel for channel in drive_channels if activity[channel - 1] < args.minimum_activity
    ]
    if args.expect_motion and inactive:
        names = ", ".join(f"M{channel}" for channel in inactive)
        print(f"FAIL: expected encoder activity was not observed on {names}.", file=sys.stderr)
        return 2

    if args.expect_motion:
        print("PASS: both configured drive encoders reported motion.")
    else:
        print("PASS: controller telemetry was readable (motion was not required).")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_arguments(parser, args)
    try:
        return run(args)
    except ValueError as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Bench-test the configured drive motors and encoder feedback.

This command intentionally moves hardware. Raise the driven wheels clear of the
ground, keep hands and loose items away, and keep the motor power switch within
reach. Each configured drive channel is exercised individually at low speed in
both directions. All channels are commanded to stop between steps and again on
every normal or error exit path.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from roadbot.hardware.i2c import SMBus2Bus
from roadbot.hardware.motor_controller import (
    DEFAULT_I2C_ADDRESS,
    MOTOR_TYPE_JGB,
    HiwonderMotorController,
    MotorControllerError,
)

DEFAULT_CONFIG = REPOSITORY_ROOT / "config" / "robot.yaml"
UINT32_MODULUS = 1 << 32
INT32_HALF_RANGE = 1 << 31


@dataclass(frozen=True, slots=True)
class MotorTestSettings:
    bus: int
    address: int
    channels: tuple[int, int]
    motor_type: int
    encoder_polarity: int


@dataclass(frozen=True, slots=True)
class MotorStepResult:
    channel: int
    target: int
    encoder_delta: int
    ok: bool


def counter_delta(current: int, previous: int) -> int:
    """Calculate a signed change while tolerating 32-bit counter rollover."""
    delta = (current - previous) % UINT32_MODULUS
    if delta >= INT32_HALF_RANGE:
        delta -= UINT32_MODULUS
    return delta


def load_settings(path: Path) -> MotorTestSettings:
    """Load motor-controller settings used by the bench test."""
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        motor = document["motor_controller"]
        settings = MotorTestSettings(
            bus=int(motor.get("bus", 1)),
            address=int(motor.get("address", DEFAULT_I2C_ADDRESS)),
            channels=(int(motor["left_channel"]), int(motor["right_channel"])),
            motor_type=int(motor.get("motor_type", MOTOR_TYPE_JGB)),
            encoder_polarity=int(motor.get("encoder_polarity", 0)),
        )
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read motor-test settings from {path}: {error}") from error

    if settings.bus < 0:
        raise ValueError("I2C bus number must be non-negative")
    if not 0 <= settings.address <= 0x7F:
        raise ValueError("motor-controller address must be a 7-bit I2C address")
    if any(channel not in range(1, 5) for channel in settings.channels):
        raise ValueError("drive motor channels must be in the range 1..4")
    if settings.channels[0] == settings.channels[1]:
        raise ValueError("left and right drive motor channels must be different")
    if settings.motor_type not in range(4):
        raise ValueError("motor type must be in the range 0..3")
    if settings.encoder_polarity not in (0, 1):
        raise ValueError("encoder polarity must be 0 or 1")
    return settings


def _targets(channel: int, target: int) -> tuple[int, int, int, int]:
    values = [0, 0, 0, 0]
    values[channel - 1] = target
    return tuple(values)  # type: ignore[return-value]


def run_motor_sequence(
    controller: HiwonderMotorController,
    channels: tuple[int, int],
    *,
    speed: int,
    duration_s: float,
    settle_s: float,
    minimum_counts: int,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[MotorStepResult, ...]:
    """Exercise each channel in both directions, stopping between commands."""
    results: list[MotorStepResult] = []
    try:
        for channel in channels:
            for target in (speed, -speed):
                before = controller.read_encoder_counts().channel(channel)
                controller.set_motor_speeds(_targets(channel, target))
                sleep(duration_s)
                controller.stop()
                sleep(settle_s)
                after = controller.read_encoder_counts().channel(channel)
                delta = counter_delta(after, before)
                results.append(
                    MotorStepResult(
                        channel=channel,
                        target=target,
                        encoder_delta=delta,
                        ok=abs(delta) >= minimum_counts,
                    )
                )
    finally:
        controller.stop()
    return tuple(results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--enable-motors",
        action="store_true",
        help="required acknowledgement that the raised wheels will move",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=8,
        help="closed-loop target in encoder pulses per 10 ms (default: 8; maximum: 20)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="seconds per direction on each motor (default: 1.0; maximum: 2.0)",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=0.25,
        help="seconds to wait after each stop before reading the encoder",
    )
    parser.add_argument(
        "--minimum-counts",
        type=int,
        default=1,
        help="minimum absolute encoder change required per step",
    )
    return parser


def validate_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.enable_motors:
        parser.error("--enable-motors is required because this test moves the wheels")
    if not 1 <= args.speed <= 20:
        parser.error("--speed must be in the range 1..20")
    if not 0.1 <= args.duration <= 2.0:
        parser.error("--duration must be in the range 0.1..2.0 seconds")
    if not 0 <= args.settle_time <= 2.0:
        parser.error("--settle-time must be in the range 0..2.0 seconds")
    if args.minimum_counts < 1:
        parser.error("--minimum-counts must be at least one")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    validate_arguments(parser, args)

    try:
        settings = load_settings(args.config)
    except ValueError as error:
        parser.error(str(error))

    channel_names = " and ".join(f"M{channel}" for channel in settings.channels)
    print("WARNING: THIS TEST WILL MOVE THE DRIVE WHEELS.")
    print(f"Configured channels: {channel_names}; test target: +/-{args.speed}.")
    print("Raise the driven wheels, clear the area, and keep the power switch within reach.")
    print("Starting in 3 seconds. Press Ctrl+C now to cancel.", flush=True)
    try:
        for remaining in (3, 2, 1):
            print(f"  {remaining}...", flush=True)
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nCancelled before motor commands were sent.")
        return 130

    try:
        with SMBus2Bus(settings.bus) as bus:
            controller = HiwonderMotorController(bus, settings.address)
            try:
                controller.stop()
                controller.configure_encoder(settings.motor_type, settings.encoder_polarity)
                time.sleep(2.0)
                results = run_motor_sequence(
                    controller,
                    settings.channels,
                    speed=args.speed,
                    duration_s=args.duration,
                    settle_s=args.settle_time,
                    minimum_counts=args.minimum_counts,
                )
            finally:
                # This must run while the I2C bus is still open.
                controller.stop()
    except KeyboardInterrupt:
        print("\nInterrupted; stop command sent.", file=sys.stderr)
        return 130
    except (ImportError, MotorControllerError, OSError, ValueError) as error:
        print(f"ERROR: motor test failed: {error}", file=sys.stderr)
        print("Verify that the wheels stopped; switch off motor power if unsure.", file=sys.stderr)
        return 1

    for result in results:
        status = "PASS" if result.ok else "FAIL"
        direction = "positive" if result.target > 0 else "negative"
        print(
            f"{status} M{result.channel} {direction}: "
            f"target={result.target:+d}, encoder delta={result.encoder_delta:+d}"
        )

    all_steps_moved = all(result.ok for result in results)
    opposite_directions = True
    for channel in settings.channels:
        deltas = [result.encoder_delta for result in results if result.channel == channel]
        if len(deltas) != 2 or deltas[0] * deltas[1] >= 0:
            opposite_directions = False
            print(
                f"FAIL M{channel}: encoder direction did not reverse between commands.",
                file=sys.stderr,
            )

    if all_steps_moved and opposite_directions:
        print("PASS: both drive motors moved, reversed, stopped, and reported encoder feedback.")
        return 0
    print("FAIL: motor control or encoder-direction validation failed.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

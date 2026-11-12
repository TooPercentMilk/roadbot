"""Interactive bench test for the Roadbot steering servo.

The test centers the steering, applies a conservative left pulse, applies a
conservative right pulse, and returns to center after each movement. Because
the steering servo has no position sensor, the operator confirms each observed
wheel position. PWM is disabled when the test exits.
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

from roadbot.hardware.steering_servo import (
    SERVO_MAXIMUM_PULSE_US,
    SERVO_MINIMUM_PULSE_US,
    PwmSteeringServo,
    SteeringServoError,
)

DEFAULT_CONFIG = REPOSITORY_ROOT / "config" / "robot.yaml"


@dataclass(frozen=True, slots=True)
class SteeringTestSettings:
    gpio_bcm: int
    frequency_hz: float
    neutral_pulse_us: int
    minimum_pulse_us: int
    maximum_pulse_us: int


@dataclass(frozen=True, slots=True)
class SteeringObservation:
    name: str
    pulse_us: int
    confirmed: bool


def _optional_int(value: object, fallback: int) -> int:
    return fallback if value is None else int(value)


def load_settings(path: Path) -> SteeringTestSettings:
    """Load PWM settings, using manufacturer limits until calibration exists."""
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        steering = document["steering"]
        settings = SteeringTestSettings(
            gpio_bcm=int(steering["gpio_bcm"]),
            frequency_hz=float(steering.get("frequency_hz", 50)),
            neutral_pulse_us=int(steering.get("neutral_pulse_us", 1500)),
            minimum_pulse_us=_optional_int(
                steering.get("minimum_pulse_us"), SERVO_MINIMUM_PULSE_US
            ),
            maximum_pulse_us=_optional_int(
                steering.get("maximum_pulse_us"), SERVO_MAXIMUM_PULSE_US
            ),
        )
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read steering settings from {path}: {error}") from error

    if not 0 <= settings.gpio_bcm <= 27:
        raise ValueError("steering BCM GPIO must be in the range 0..27")
    if settings.frequency_hz <= 0:
        raise ValueError("steering PWM frequency must be positive")
    if not (
        0
        < settings.minimum_pulse_us
        < settings.neutral_pulse_us
        < settings.maximum_pulse_us
    ):
        raise ValueError("steering pulses must satisfy 0 < minimum < neutral < maximum")
    return settings


def run_steering_sequence(
    servo: PwmSteeringServo,
    *,
    offset_us: int,
    hold_s: float,
    center_hold_s: float,
    sleep: Callable[[float], None] = time.sleep,
    confirm: Callable[[str, int], bool],
) -> tuple[SteeringObservation, ...]:
    """Command center/left/right poses and collect physical observations."""
    neutral = servo.neutral_pulse_us
    poses = (
        ("center", neutral),
        ("left", neutral - offset_us),
        ("right", neutral + offset_us),
    )
    observations: list[SteeringObservation] = []
    try:
        for name, pulse_us in poses:
            servo.set_pulse_us(pulse_us)
            sleep(hold_s)
            if name != "center":
                servo.center()
                sleep(center_hold_s)
            confirmed = confirm(name, pulse_us)
            observations.append(SteeringObservation(name, pulse_us, confirmed))
            if not confirmed:
                break
    finally:
        servo.center()
        sleep(center_hold_s)
    return tuple(observations)


def _confirm_position(name: str, pulse_us: int) -> bool:
    expected = {
        "center": "point straight ahead without binding or continuous buzzing",
        "left": "move left and then return to center without binding",
        "right": "move right and then return to center without binding",
    }[name]
    try:
        answer = input(
            f"Did the front wheels {expected}? "
            f"(command was {name}, {pulse_us} us) [y/N]: "
        )
    except EOFError:
        return False
    return answer.strip().lower() in {"y", "yes"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--enable-steering",
        action="store_true",
        help="required acknowledgement that the steering linkage will move",
    )
    parser.add_argument(
        "--offset-us",
        type=int,
        default=250,
        help="pulse offset from neutral for left/right checks (default: 250; maximum: 500)",
    )
    parser.add_argument(
        "--hold-time",
        type=float,
        default=1.0,
        help="seconds to hold each commanded position (default: 1.0; maximum: 3.0)",
    )
    return parser


def validate_arguments(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    settings: SteeringTestSettings,
) -> None:
    if not args.enable_steering:
        parser.error("--enable-steering is required because this test moves the linkage")
    if not 50 <= args.offset_us <= 500:
        parser.error("--offset-us must be in the range 50..500")
    if not 0.2 <= args.hold_time <= 3.0:
        parser.error("--hold-time must be in the range 0.2..3.0 seconds")
    if settings.neutral_pulse_us - args.offset_us < settings.minimum_pulse_us:
        parser.error("left test pulse would exceed the configured minimum")
    if settings.neutral_pulse_us + args.offset_us > settings.maximum_pulse_us:
        parser.error("right test pulse would exceed the configured maximum")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        settings = load_settings(args.config)
    except ValueError as error:
        parser.error(str(error))
    validate_arguments(parser, args, settings)

    print("WARNING: THIS TEST WILL MOVE THE FRONT STEERING LINKAGE.")
    print("Raise the front wheels, stop the drive motors, and keep hands clear.")
    print("Power the servo from its rated supply; connect only its signal to the GPIO.")
    print(
        f"BCM GPIO {settings.gpio_bcm}, neutral {settings.neutral_pulse_us} us, "
        f"test offset +/-{args.offset_us} us."
    )
    print("Starting in 3 seconds. Press Ctrl+C now to cancel.", flush=True)
    try:
        for remaining in (3, 2, 1):
            print(f"  {remaining}...", flush=True)
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nCancelled before steering commands were sent.")
        return 130

    servo = PwmSteeringServo(
        settings.gpio_bcm,
        frequency_hz=settings.frequency_hz,
        neutral_pulse_us=settings.neutral_pulse_us,
        minimum_pulse_us=settings.minimum_pulse_us,
        maximum_pulse_us=settings.maximum_pulse_us,
    )
    try:
        with servo:
            observations = run_steering_sequence(
                servo,
                offset_us=args.offset_us,
                hold_s=args.hold_time,
                center_hold_s=0.5,
                confirm=_confirm_position,
            )
    except KeyboardInterrupt:
        print("\nInterrupted; steering was returned to center and PWM disabled.", file=sys.stderr)
        return 130
    except (SteeringServoError, OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: steering test failed: {error}", file=sys.stderr)
        print("Switch off servo power if the linkage is not centered.", file=sys.stderr)
        return 1

    for observation in observations:
        status = "PASS" if observation.confirmed else "FAIL"
        print(f"{status} {observation.name}: command={observation.pulse_us} us")

    if len(observations) == 3 and all(item.confirmed for item in observations):
        print("PASS: steering centered, moved left and right, and returned to center.")
        return 0
    print("FAIL: the observed steering motion did not match every command.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

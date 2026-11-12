import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "tools" / "check_steering.py"
SPEC = importlib.util.spec_from_file_location("check_steering", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
check_steering = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = check_steering
SPEC.loader.exec_module(check_steering)


class FakeServo:
    neutral_pulse_us = 1500

    def __init__(self) -> None:
        self.pulses: list[int] = []

    def set_pulse_us(self, pulse_us: int) -> None:
        self.pulses.append(pulse_us)

    def center(self) -> None:
        self.set_pulse_us(self.neutral_pulse_us)


def test_sequence_checks_center_left_right_and_finishes_centered() -> None:
    servo = FakeServo()
    confirmations: list[tuple[str, int]] = []

    def confirm(name: str, pulse_us: int) -> bool:
        confirmations.append((name, pulse_us))
        return True

    observations = check_steering.run_steering_sequence(
        servo,  # type: ignore[arg-type]
        offset_us=250,
        hold_s=1.0,
        center_hold_s=0.5,
        sleep=lambda _duration: None,
        confirm=confirm,
    )

    assert confirmations == [("center", 1500), ("left", 1250), ("right", 1750)]
    assert all(item.confirmed for item in observations)
    assert servo.pulses == [1500, 1250, 1500, 1750, 1500, 1500]


def test_sequence_returns_to_center_when_confirmation_is_interrupted() -> None:
    servo = FakeServo()

    with pytest.raises(RuntimeError, match="interrupted"):
        check_steering.run_steering_sequence(
            servo,  # type: ignore[arg-type]
            offset_us=250,
            hold_s=1.0,
            center_hold_s=0.5,
            sleep=lambda _duration: None,
            confirm=lambda _name, _pulse: (_ for _ in ()).throw(RuntimeError("interrupted")),
        )

    assert servo.pulses[-1] == 1500

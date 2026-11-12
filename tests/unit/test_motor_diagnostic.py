import importlib.util
import sys
from pathlib import Path

import pytest

from roadbot.hardware.motor_controller import EncoderCounts

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "tools" / "check_motors.py"
SPEC = importlib.util.spec_from_file_location("check_motors", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
check_motors = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = check_motors
SPEC.loader.exec_module(check_motors)


class FakeController:
    def __init__(self) -> None:
        self.counts = [0, 0, 0, 0]
        self.commands: list[tuple[int, int, int, int]] = []

    def read_encoder_counts(self) -> EncoderCounts:
        return EncoderCounts(tuple(self.counts))  # type: ignore[arg-type]

    def set_motor_speeds(self, speeds: tuple[int, int, int, int]) -> None:
        self.commands.append(speeds)
        for index, speed in enumerate(speeds):
            self.counts[index] += speed * 10

    def stop(self) -> None:
        self.set_motor_speeds((0, 0, 0, 0))


def test_sequence_exercises_both_channels_both_directions_and_stops() -> None:
    controller = FakeController()

    results = check_motors.run_motor_sequence(
        controller,  # type: ignore[arg-type]
        (1, 3),
        speed=8,
        duration_s=1.0,
        settle_s=0.0,
        minimum_counts=1,
        sleep=lambda _duration: None,
    )

    assert [(result.channel, result.target) for result in results] == [
        (1, 8),
        (1, -8),
        (3, 8),
        (3, -8),
    ]
    assert [result.encoder_delta for result in results] == [80, -80, 80, -80]
    assert all(result.ok for result in results)
    assert controller.commands[-1] == (0, 0, 0, 0)


def test_sequence_stops_all_channels_when_a_step_is_interrupted() -> None:
    controller = FakeController()

    with pytest.raises(RuntimeError, match="interrupted"):
        check_motors.run_motor_sequence(
            controller,  # type: ignore[arg-type]
            (1, 3),
            speed=8,
            duration_s=1.0,
            settle_s=0.0,
            minimum_counts=1,
            sleep=lambda _duration: (_ for _ in ()).throw(RuntimeError("interrupted")),
        )

    assert controller.commands[-1] == (0, 0, 0, 0)

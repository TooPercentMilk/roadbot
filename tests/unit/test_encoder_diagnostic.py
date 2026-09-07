import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "tools" / "check_encoders.py"
SPEC = importlib.util.spec_from_file_location("check_encoders", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
check_encoders = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_encoders)


def test_counter_delta_handles_forward_rollover() -> None:
    assert check_encoders.counter_delta(-2_147_483_648, 2_147_483_647) == 1


def test_counter_delta_handles_reverse_rollover() -> None:
    assert check_encoders.counter_delta(2_147_483_647, -2_147_483_648) == -1

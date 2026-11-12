import pytest

from roadbot.hardware.steering_servo import PwmSteeringServo, SteeringServoError


class FakePwm:
    def __init__(
        self,
        pin: int,
        *,
        active_high: bool,
        initial_value: float,
        frequency: float,
    ) -> None:
        self.pin = pin
        self.active_high = active_high
        self.frequency = frequency
        self.values = [initial_value]
        self.closed = False

    @property
    def value(self) -> float:
        return self.values[-1]

    @value.setter
    def value(self, value: float) -> None:
        self.values.append(value)

    def close(self) -> None:
        self.closed = True


def test_servo_converts_microseconds_to_pwm_duty_cycle_and_centers() -> None:
    pwm: FakePwm | None = None

    def factory(*args, **kwargs):
        nonlocal pwm
        pwm = FakePwm(*args, **kwargs)
        return pwm

    servo = PwmSteeringServo(12, pwm_factory=factory)

    servo.open()
    servo.center()
    servo.set_pulse_us(1250)
    servo.close()

    assert pwm is not None
    assert pwm.pin == 12
    assert pwm.frequency == 50.0
    assert pwm.values == [0.0, pytest.approx(0.075), pytest.approx(0.0625), 0.0]
    assert pwm.closed


def test_servo_rejects_pulses_outside_limits_without_changing_output() -> None:
    pwm = FakePwm(12, active_high=True, initial_value=0.0, frequency=50.0)
    servo = PwmSteeringServo(12, pwm_factory=lambda *_args, **_kwargs: pwm)
    servo.open()

    with pytest.raises(ValueError, match="between 500 and 2500"):
        servo.set_pulse_us(499)

    assert pwm.values == [0.0]


def test_servo_requires_open_before_command() -> None:
    servo = PwmSteeringServo(12, pwm_factory=FakePwm)

    with pytest.raises(SteeringServoError, match="not open"):
        servo.center()

"""PWM steering-servo adapter with explicit pulse-width safety limits."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Self

DEFAULT_FREQUENCY_HZ = 50.0
DEFAULT_NEUTRAL_PULSE_US = 1500
SERVO_MINIMUM_PULSE_US = 500
SERVO_MAXIMUM_PULSE_US = 2500


class SteeringServoError(RuntimeError):
    """Raised when the steering PWM output cannot be controlled safely."""


class PwmSteeringServo:
    """Control a conventional PWM servo using GPIO Zero on a BCM GPIO pin."""

    def __init__(
        self,
        gpio_bcm: int,
        *,
        frequency_hz: float = DEFAULT_FREQUENCY_HZ,
        neutral_pulse_us: int = DEFAULT_NEUTRAL_PULSE_US,
        minimum_pulse_us: int = SERVO_MINIMUM_PULSE_US,
        maximum_pulse_us: int = SERVO_MAXIMUM_PULSE_US,
        pwm_factory: Callable[..., Any] | None = None,
    ) -> None:
        if not 0 <= gpio_bcm <= 27:
            raise ValueError("BCM GPIO number must be in the range 0..27")
        if frequency_hz <= 0:
            raise ValueError("PWM frequency must be positive")
        if not 0 < minimum_pulse_us < neutral_pulse_us < maximum_pulse_us:
            raise ValueError("pulse limits must satisfy 0 < minimum < neutral < maximum")
        if maximum_pulse_us * frequency_hz >= 1_000_000:
            raise ValueError("maximum pulse width must be shorter than one PWM period")
        self.gpio_bcm = gpio_bcm
        self.frequency_hz = float(frequency_hz)
        self.neutral_pulse_us = neutral_pulse_us
        self.minimum_pulse_us = minimum_pulse_us
        self.maximum_pulse_us = maximum_pulse_us
        self._pwm_factory = pwm_factory
        self._pwm: Any | None = None

    @property
    def is_open(self) -> bool:
        return self._pwm is not None

    def open(self) -> None:
        if self._pwm is not None:
            return
        try:
            if self._pwm_factory is None:
                from gpiozero import PWMOutputDevice

                factory = PWMOutputDevice
            else:
                factory = self._pwm_factory
            self._pwm = factory(
                self.gpio_bcm,
                active_high=True,
                initial_value=0.0,
                frequency=self.frequency_hz,
            )
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            self._pwm = None
            raise SteeringServoError(
                f"unable to open steering PWM on BCM GPIO {self.gpio_bcm}: {error}"
            ) from error

    def set_pulse_us(self, pulse_us: int) -> None:
        if self._pwm is None:
            raise SteeringServoError("steering servo is not open")
        if not self.minimum_pulse_us <= pulse_us <= self.maximum_pulse_us:
            raise ValueError(
                f"steering pulse must be between {self.minimum_pulse_us} and "
                f"{self.maximum_pulse_us} us"
            )
        duty_cycle = pulse_us * self.frequency_hz / 1_000_000.0
        try:
            self._pwm.value = duty_cycle
        except (OSError, RuntimeError, ValueError) as error:
            raise SteeringServoError(f"unable to set steering pulse: {error}") from error

    def center(self) -> None:
        self.set_pulse_us(self.neutral_pulse_us)

    def disable(self) -> None:
        if self._pwm is None:
            return
        try:
            self._pwm.value = 0.0
        except (OSError, RuntimeError, ValueError) as error:
            raise SteeringServoError(f"unable to disable steering PWM: {error}") from error

    def close(self) -> None:
        pwm, self._pwm = self._pwm, None
        if pwm is None:
            return
        try:
            pwm.value = 0.0
            pwm.close()
        except (OSError, RuntimeError, ValueError) as error:
            raise SteeringServoError(f"unable to close steering PWM: {error}") from error

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


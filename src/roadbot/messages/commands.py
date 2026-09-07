"""Planning and actuator command messages."""

from dataclasses import dataclass

from roadbot.messages.common import MessageHeader


@dataclass(frozen=True, slots=True)
class DriveCommand:
    header: MessageHeader
    target_speed_mps: float
    target_steering_rad: float
    valid_until_ns: int


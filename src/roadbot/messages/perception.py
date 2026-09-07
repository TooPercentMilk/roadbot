"""Outputs from lane, object, and traffic-light perception."""

from dataclasses import dataclass
from enum import Enum, auto

from roadbot.messages.common import MessageHeader


class TrafficLightState(Enum):
    UNKNOWN = auto()
    RED = auto()
    YELLOW = auto()
    GREEN = auto()


@dataclass(frozen=True, slots=True)
class LaneEstimate:
    header: MessageHeader
    lateral_offset_m: float
    heading_error_rad: float
    confidence: float


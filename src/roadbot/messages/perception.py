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


@dataclass(frozen=True, slots=True)
class ObjectDetection:
    class_id: int
    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]  # original camera pixels


@dataclass(frozen=True, slots=True)
class ObjectDetections:
    header: MessageHeader  # retains camera acquisition stamp and sequence
    frame_id: str
    detections: tuple[ObjectDetection, ...]

"""Fields shared by all timestamped messages."""

from dataclasses import dataclass
from enum import Enum, auto


class HealthStatus(Enum):
    """Coarse subsystem health for supervisor decisions."""

    STARTING = auto()
    HEALTHY = auto()
    DEGRADED = auto()
    FAILED = auto()
    STOPPED = auto()


@dataclass(frozen=True, slots=True)
class MessageHeader:
    timestamp_ns: int
    sequence: int
    source: str


@dataclass(frozen=True, slots=True)
class Vector3:
    x: float
    y: float
    z: float


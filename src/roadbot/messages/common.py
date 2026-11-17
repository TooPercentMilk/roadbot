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
    """Acquisition time in a named clock domain, never inference completion time.

    ``timestamp_uncertainty_ns`` describes host clock/read measurement only;
    device sampling latency may add error and is not inferred here.
    """

    timestamp_ns: int
    sequence: int
    source: str
    clock_domain: str = "monotonic"
    timestamp_source: str = "unspecified"
    received_timestamp_ns: int | None = None
    timestamp_uncertainty_ns: int | None = None


@dataclass(frozen=True, slots=True)
class Vector3:
    x: float
    y: float
    z: float

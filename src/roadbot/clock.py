"""Clock utilities shared by sensor and control messages."""

import time
from collections.abc import Callable


def now_ns() -> int:
    """Return a monotonic timestamp suitable for elapsed-time calculations."""
    return time.monotonic_ns()


def boottime_ns() -> int:
    """libcamera's Linux clock; intentionally unavailable on other platforms."""
    return time.clock_gettime_ns(time.CLOCK_BOOTTIME)


def map_sensor_timestamp(
    timestamp_ns: int,
    *,
    sensor_clock: Callable[[], int] = boottime_ns,
    clock: Callable[[], int] = now_ns,
) -> tuple[int, int]:
    """Map a same-host sensor clock to monotonic using a bracketed clock pair.

    Resample per frame so BOOTTIME's suspend time is not mistaken for MONOTONIC
    time. This is not a clock-drift estimator for an external device.
    """
    before = clock()
    sensor_now = sensor_clock()
    after = clock()
    return timestamp_ns + (before + after) // 2 - sensor_now, (after - before + 1) // 2

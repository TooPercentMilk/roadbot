"""Clock utilities shared by sensor and control messages."""

from time import monotonic_ns


def now_ns() -> int:
    """Return a monotonic timestamp suitable for elapsed-time calculations."""
    return monotonic_ns()


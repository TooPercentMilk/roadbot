"""Common lifecycle contract for sensor implementations."""

from typing import Protocol, TypeVar

SampleT_co = TypeVar("SampleT_co", covariant=True)


class Sensor(Protocol[SampleT_co]):
    """Minimal interface implemented by synchronous sensor drivers."""

    def open(self) -> None: ...

    def read(self) -> SampleT_co: ...

    def close(self) -> None: ...

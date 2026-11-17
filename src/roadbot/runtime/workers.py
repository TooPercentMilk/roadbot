"""Long-running sensor, perception, state, and control workers."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from typing import Self

from roadbot.clock import now_ns
from roadbot.config import PerceptionConfig
from roadbot.perception.preprocessing import PreparedFrame, YoloPreprocessor
from roadbot.perception.synchronization import SensorMessage, SensorSynchronizer
from roadbot.sensors.base import Sensor


class PerceptionRuntime:
    """Acquire three feeds independently; preprocess only on the consumer thread.

    Sensor workers own their device lifecycle. They never wait for inference.
    The synchronizer drops oldest queued data at capacity, keeping memory bounded.
    Each hardware source must bound its own I/O calls for timely shutdown.
    """

    def __init__(
        self,
        sources: dict[str, tuple[Sensor[SensorMessage], float]],
        config: PerceptionConfig | None = None,
        *,
        clock: Callable[[], int] = now_ns,
    ) -> None:
        if set(sources) != {"camera", "imu", "encoders"}:
            raise ValueError("camera, imu, and encoders sources are required")
        if any(not math.isfinite(rate) or rate <= 0 for _, rate in sources.values()):
            raise ValueError("sensor rates must be finite and positive")
        self.config = config or PerceptionConfig()
        self.synchronizer = SensorSynchronizer(self.config.synchronization)
        self.preprocessor = YoloPreprocessor(self.config.preprocessing)
        self.sources, self._clock = sources, clock
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._errors: dict[str, Exception] = {}
        self._error_lock = threading.Lock()
        self.prepared_count = 0
        self.preprocessing_stale_count = 0

    def _fail(self, name: str, error: Exception) -> None:
        with self._error_lock:
            self._errors.setdefault(name, error)
        self._stop.set()

    def _sample(self, name: str, source: Sensor[SensorMessage], rate: float) -> None:
        try:
            source.open()
            deadline = time.monotonic()
            while not self._stop.is_set():
                message = source.read()
                self.synchronizer.add(message, now_ns=self._clock())
                deadline = max(deadline + 1 / rate, time.monotonic())
                self._stop.wait(max(0, deadline - time.monotonic()))
        except Exception as error:  # noqa: BLE001 - propagate worker failures to the consumer
            self._fail(name, error)
        finally:
            try:
                source.close()
            except Exception as error:  # noqa: BLE001 - report device cleanup failures
                self._fail(name, error)

    def start(self) -> None:
        if self._threads:
            raise RuntimeError("runtime already started")
        self._stop.clear()
        with self._error_lock:
            self._errors.clear()
        self.synchronizer.reset()
        self.prepared_count = self.preprocessing_stale_count = 0
        for name, (source, rate) in self.sources.items():
            thread = threading.Thread(
                target=self._sample,
                args=(name, source, rate),
                name=f"perception-{name}",
                daemon=True,
            )
            self._threads.append(thread)
            thread.start()

    def check_health(self) -> None:
        with self._error_lock:
            if self._errors:
                name, error = next(iter(self._errors.items()))
                raise RuntimeError(f"{name} acquisition failed: {error}") from error

    def read(self, timeout_s: float = 1.0) -> PreparedFrame | None:
        """Wait for a usable prepared frame; None means no match before timeout."""
        if not self._threads:
            raise RuntimeError("runtime is not started")
        if not math.isfinite(timeout_s) or timeout_s < 0:
            raise ValueError("timeout_s must be finite and nonnegative")
        deadline = time.monotonic() + timeout_s
        while True:
            self.check_health()
            if self._stop.is_set():
                return None
            packet = self.synchronizer.poll(now_ns=self._clock())
            if packet is not None:
                prepared = self.preprocessor.prepare(packet)
                oldest_ns = min(
                    packet.header.timestamp_ns,
                    packet.imu.header.timestamp_ns,
                    packet.encoders.header.timestamp_ns,
                )
                if self._clock() - oldest_ns <= self.config.synchronization.max_age_ms * 1e6:
                    self.prepared_count += 1
                    return prepared
                self.preprocessing_stale_count += 1
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            self._stop.wait(min(0.005, remaining))

    def close(self) -> None:
        self._stop.set()
        deadline = time.monotonic() + 2.0
        for thread in self._threads:
            thread.join(max(0, deadline - time.monotonic()))
        alive = [t.name for t in self._threads if t.is_alive()]
        if alive:
            raise RuntimeError(f"sensor I/O did not stop within 2 seconds: {alive}")
        self._threads.clear()
        self.check_health()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()

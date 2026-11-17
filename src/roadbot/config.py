"""Typed configuration loading for the robot."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class SynchronizationConfig:
    imu_tolerance_ms: float = 15.0
    encoder_tolerance_ms: float = 30.0
    max_wait_ms: float = 60.0
    max_age_ms: float = 250.0
    camera_buffer_size: int = 3
    imu_buffer_size: int = 256
    encoder_buffer_size: int = 128
    require_camera_sensor_timestamp: bool = True
    clock_domain: str = "monotonic"

    def __post_init__(self) -> None:
        for name in ("imu_tolerance_ms", "encoder_tolerance_ms", "max_wait_ms", "max_age_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_wait_ms > self.max_age_ms:
            raise ValueError("max_wait_ms must not exceed max_age_ms")
        for name in ("camera_buffer_size", "imu_buffer_size", "encoder_buffer_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.require_camera_sensor_timestamp) is not bool:
            raise ValueError("require_camera_sensor_timestamp must be boolean")
        if not isinstance(self.clock_domain, str) or not self.clock_domain.strip():
            raise ValueError("clock_domain must be named")


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    input_size: tuple[int, int] = (640, 640)  # height, width
    stride: int = 32
    scale_up: bool = True
    padding_value: int = 114

    def __post_init__(self) -> None:
        if type(self.stride) is not int or self.stride <= 0:
            raise ValueError("stride must be a positive integer")
        if len(self.input_size) != 2 or any(
            type(v) is not int or v <= 0 or v % self.stride for v in self.input_size
        ):
            raise ValueError(
                "input_size must contain two positive stride multiples (height, width)"
            )
        if type(self.padding_value) is not int or not 0 <= self.padding_value <= 255:
            raise ValueError("padding_value must be a uint8 value")
        if type(self.scale_up) is not bool:
            raise ValueError("scale_up must be boolean")


@dataclass(frozen=True, slots=True)
class PerceptionConfig:
    synchronization: SynchronizationConfig = field(default_factory=SynchronizationConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)


def load_perception_config(path: Path) -> PerceptionConfig:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise TypeError("robot configuration must be a mapping")
    settings = document.get("perception", {})
    if not isinstance(settings, dict) or set(settings) - {"synchronization", "preprocessing"}:
        raise ValueError("invalid perception configuration sections")
    preprocess = dict(settings.get("preprocessing", {}))
    if "input_size" in preprocess:
        preprocess["input_size"] = tuple(preprocess["input_size"])
    return PerceptionConfig(
        SynchronizationConfig(**settings.get("synchronization", {})),
        PreprocessingConfig(**preprocess),
    )

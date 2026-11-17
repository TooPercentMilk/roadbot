"""Raw and calibrated sensor messages."""

from dataclasses import dataclass
from typing import Any

from roadbot.messages.common import MessageHeader, Vector3


@dataclass(frozen=True, slots=True)
class CameraFrame:
    header: MessageHeader
    image: Any
    metadata: dict[str, Any]
    encoding: str = "bgr8"
    frame_id: str = "camera_optical_frame"


@dataclass(frozen=True, slots=True)
class ImuSample:
    header: MessageHeader
    acceleration_mps2: Vector3
    angular_velocity_rad_s: Vector3
    temperature_c: float | None = None
    frame_id: str = "imu_link"


@dataclass(frozen=True, slots=True)
class WheelEncoderSample:
    header: MessageHeader
    left_count: int
    right_count: int
    left_rad_s: float | None = None
    right_rad_s: float | None = None
    frame_id: str = "base_link"

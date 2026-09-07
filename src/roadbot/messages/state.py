"""Estimated vehicle-state messages."""

from dataclasses import dataclass

from roadbot.messages.common import MessageHeader


@dataclass(frozen=True, slots=True)
class VehicleState:
    header: MessageHeader
    speed_mps: float
    yaw_rate_rad_s: float
    heading_rad: float
    confidence: float


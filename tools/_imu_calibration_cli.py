"""Shared hardware recording helpers for the IMU calibration commands."""

from __future__ import annotations

import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from roadbot.sensors.imu import Bmi088Imu

DEFAULT_CONFIG = REPOSITORY_ROOT / "config" / "robot.yaml"


@dataclass(frozen=True, slots=True)
class ImuSettings:
    bus: int
    accelerometer_address: int
    gyroscope_address: int
    sample_rate_hz: float
    calibration_file: Path


@dataclass(frozen=True, slots=True)
class Recording:
    timestamps_ns: np.ndarray
    acceleration_mps2: np.ndarray
    angular_velocity_rad_s: np.ndarray
    temperature_c: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(self.timestamps_ns.size)

    @property
    def effective_rate_hz(self) -> float:
        if self.sample_count < 2:
            return float("nan")
        elapsed_s = (self.timestamps_ns[-1] - self.timestamps_ns[0]) / 1_000_000_000
        return (self.sample_count - 1) / elapsed_s if elapsed_s > 0 else float("nan")


def _parse_integer(value: object) -> int:
    if isinstance(value, str):
        return int(value, 0)
    return int(value)


def load_imu_settings(config_path: Path, calibration_override: Path | None) -> ImuSettings:
    try:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        section = document["imu"]
        configured_calibration = Path(section.get("calibration_file", "calibration/imu.yaml"))
        calibration_file = configured_calibration
        if not calibration_file.is_absolute():
            calibration_file = config_path.resolve().parent.parent / calibration_file
        if calibration_override is not None:
            calibration_file = calibration_override.resolve()
        settings = ImuSettings(
            bus=int(section.get("bus", 1)),
            accelerometer_address=_parse_integer(section.get("accelerometer_address", 0x19)),
            gyroscope_address=_parse_integer(section.get("gyroscope_address", 0x69)),
            sample_rate_hz=float(section.get("sample_rate_hz", 100.0)),
            calibration_file=calibration_file,
        )
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read IMU settings from {config_path}: {error}") from error
    if settings.bus < 0 or settings.sample_rate_hz <= 0:
        raise ValueError("IMU bus must be non-negative and sample rate must be positive")
    return settings


def create_imu(settings: ImuSettings) -> Bmi088Imu:
    return Bmi088Imu(
        bus_number=settings.bus,
        accelerometer_address=settings.accelerometer_address,
        gyroscope_address=settings.gyroscope_address,
        sample_rate_hz=settings.sample_rate_hz,
    )


def wait_for_confirmation(message: str, assume_yes: bool) -> None:
    print(message)
    if assume_yes:
        print("Starting in 3 seconds...")
        time.sleep(3.0)
    else:
        input("Press Enter when ready, or Ctrl+C to cancel: ")


def record(
    imu: Bmi088Imu,
    *,
    duration_s: float,
    sample_rate_hz: float,
    warmup_s: float = 0.0,
) -> Recording:
    if duration_s <= 0 or sample_rate_hz <= 0 or warmup_s < 0:
        raise ValueError("duration/rate must be positive and warmup cannot be negative")
    samples = []
    imu.open()
    try:
        if warmup_s:
            print(f"Warming up for {warmup_s:g} seconds; keep the robot stationary...")
            warmup_deadline = time.monotonic() + warmup_s
            while time.monotonic() < warmup_deadline:
                imu.read()
                time.sleep(1.0 / sample_rate_hz)

        target_count = max(2, math.ceil(duration_s * sample_rate_hz))
        print(f"Recording approximately {target_count} samples for {duration_s:g} seconds...")
        start = time.monotonic()
        next_read = start
        next_progress = start + max(1.0, duration_s / 10.0)
        while len(samples) < target_count:
            now = time.monotonic()
            if now < next_read:
                time.sleep(next_read - now)
            samples.append(imu.read())
            next_read += 1.0 / sample_rate_hz
            now = time.monotonic()
            if now >= next_progress:
                print(f"  {min(now - start, duration_s):.1f}/{duration_s:.1f} seconds")
                next_progress += max(1.0, duration_s / 10.0)
    finally:
        imu.close()

    timestamps = np.array([sample.header.timestamp_ns for sample in samples], dtype=np.int64)
    acceleration = np.array(
        [
            [
                sample.acceleration_mps2.x,
                sample.acceleration_mps2.y,
                sample.acceleration_mps2.z,
            ]
            for sample in samples
        ]
    )
    angular_velocity = np.array(
        [
            [
                sample.angular_velocity_rad_s.x,
                sample.angular_velocity_rad_s.y,
                sample.angular_velocity_rad_s.z,
            ]
            for sample in samples
        ]
    )
    temperature = np.array(
        [float("nan") if sample.temperature_c is None else sample.temperature_c for sample in samples]
    )
    return Recording(timestamps, acceleration, angular_velocity, temperature)


def save_recording(path: Path, recording: Recording) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "timestamp_ns",
                "accel_x_mps2",
                "accel_y_mps2",
                "accel_z_mps2",
                "gyro_x_rad_s",
                "gyro_y_rad_s",
                "gyro_z_rad_s",
                "temperature_c",
            ]
        )
        for index in range(recording.sample_count):
            writer.writerow(
                [
                    int(recording.timestamps_ns[index]),
                    *recording.acceleration_mps2[index],
                    *recording.angular_velocity_rad_s[index],
                    recording.temperature_c[index],
                ]
            )


def write_allan_csv(
    csv_file: TextIO,
    cluster_times_s: np.ndarray,
    accel_deviation: np.ndarray,
    gyro_deviation: np.ndarray,
) -> None:
    writer = csv.writer(csv_file)
    writer.writerow(
        [
            "cluster_time_s",
            "accel_x_mps2",
            "accel_y_mps2",
            "accel_z_mps2",
            "gyro_x_rad_s",
            "gyro_y_rad_s",
            "gyro_z_rad_s",
        ]
    )
    for index, cluster_time in enumerate(cluster_times_s):
        writer.writerow(
            [
                cluster_time,
                *accel_deviation[index],
                *gyro_deviation[index],
            ]
        )

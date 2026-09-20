"""Smoke-test live camera and IMU data at the runtime pipeline boundary.

Both devices are sampled concurrently.  The command fails when a device cannot
be read, a message is stale or malformed, timestamps/sequences do not advance,
or either feed falls below the requested fraction of its configured rate.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from roadbot.runtime.sensor_validation import PipelineReport, SensorPipelineValidator
from roadbot.sensors.base import Sensor
from roadbot.sensors.camera import Picamera2Camera
from roadbot.sensors.imu import Bmi088Imu

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "robot.yaml"


def load_settings(path: Path) -> dict[str, Any]:
    """Load and validate only the camera/IMU settings used by this check."""
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        camera = document["camera"]
        imu = document["imu"]
        resolution_values = camera["resolution"]
        resolution = (int(resolution_values[0]), int(resolution_values[1]))
        settings = {
            "resolution": resolution,
            "pixel_format": str(camera.get("format", "RGB888")),
            "camera_fps": float(camera.get("fps", 30)),
            "horizontal_flip": bool(camera.get("horizontal_flip", False)),
            "vertical_flip": bool(camera.get("vertical_flip", False)),
            "bus": int(imu.get("bus", 1)),
            "accelerometer_address": int(imu.get("accelerometer_address", 0x19)),
            "gyroscope_address": int(imu.get("gyroscope_address", 0x69)),
            "imu_rate_hz": float(imu.get("sample_rate_hz", 100)),
        }
    except (OSError, KeyError, IndexError, TypeError, ValueError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read camera/IMU settings from {path}: {error}") from error

    if any(value <= 0 for value in settings["resolution"]):
        raise ValueError("camera resolution values must be positive")
    if settings["camera_fps"] <= 0 or settings["imu_rate_hz"] <= 0:
        raise ValueError("camera and IMU rates must be positive")
    if settings["bus"] < 0:
        raise ValueError("IMU bus number must be non-negative")
    return settings


def _sample_feed(
    name: str,
    sensor: Sensor[Any],
    accept: Callable[[Any], bool],
    requested_rate_hz: float,
    duration_s: float,
    start: threading.Event,
    validator: SensorPipelineValidator,
) -> None:
    start.wait()
    deadline = time.monotonic() + duration_s
    period_s = 1.0 / requested_rate_hz
    next_read = time.monotonic()
    while time.monotonic() < deadline:
        try:
            accept(sensor.read())
        except (OSError, RuntimeError, ValueError) as error:
            validator.record_issue(name, f"read failed: {error}")
            return
        next_read += period_s
        remaining = min(next_read, deadline) - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)


def run_probe(
    camera: Sensor[Any],
    imu: Sensor[Any],
    *,
    resolution: tuple[int, int],
    camera_fps: float,
    imu_rate_hz: float,
    duration_s: float,
    minimum_rate_ratio: float,
    maximum_sample_age_s: float,
) -> PipelineReport:
    """Open, concurrently sample, validate, and close both sensor sources."""
    validator = SensorPipelineValidator(
        resolution,
        max_sample_age_s=maximum_sample_age_s,
    )
    opened: list[Sensor[Any]] = []
    try:
        camera.open()
        opened.append(camera)
        imu.open()
        opened.append(imu)

        start = threading.Event()
        threads = [
            threading.Thread(
                target=_sample_feed,
                args=(
                    "camera",
                    camera,
                    validator.accept_camera,
                    camera_fps,
                    duration_s,
                    start,
                    validator,
                ),
                name="camera-pipeline-check",
                daemon=True,
            ),
            threading.Thread(
                target=_sample_feed,
                args=(
                    "imu",
                    imu,
                    validator.accept_imu,
                    imu_rate_hz,
                    duration_s,
                    start,
                    validator,
                ),
                name="imu-pipeline-check",
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()
        start.set()
        for thread in threads:
            thread.join(duration_s + maximum_sample_age_s + 2.0)
            if thread.is_alive():
                validator.record_issue(thread.name.split("-")[0], "read did not finish in time")
    finally:
        for sensor in reversed(opened):
            try:
                sensor.close()
            except (OSError, RuntimeError, ValueError) as error:
                name = "camera" if sensor is camera else "imu"
                validator.record_issue(name, f"close failed: {error}")

    return validator.report(
        minimum_camera_hz=camera_fps * minimum_rate_ratio,
        minimum_imu_hz=imu_rate_hz * minimum_rate_ratio,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--duration", type=float, default=5.0, help="sample duration in seconds")
    parser.add_argument(
        "--minimum-rate-ratio",
        type=float,
        default=0.7,
        help="minimum fraction of each configured rate (default: 0.7)",
    )
    parser.add_argument(
        "--maximum-sample-age",
        type=float,
        default=0.5,
        help="maximum accepted message age in seconds (default: 0.5)",
    )
    return parser


def _print_report(report: PipelineReport) -> None:
    for summary in (report.camera, report.imu):
        status = "PASS" if summary.ok else "FAIL"
        print(
            f"{status} {summary.name}: {summary.valid_samples} valid samples, "
            f"{summary.rate_hz:.1f} Hz"
        )
        for issue in summary.issues:
            print(f"  - {issue}", file=sys.stderr)


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error("--duration must be greater than zero")
    if not 0 < args.minimum_rate_ratio <= 1:
        parser.error("--minimum-rate-ratio must be in the range (0, 1]")
    if args.maximum_sample_age <= 0:
        parser.error("--maximum-sample-age must be greater than zero")

    try:
        settings = load_settings(args.config)
        camera = Picamera2Camera(
            resolution=settings["resolution"],
            pixel_format=settings["pixel_format"],
            fps=settings["camera_fps"],
            horizontal_flip=settings["horizontal_flip"],
            vertical_flip=settings["vertical_flip"],
        )
        imu = Bmi088Imu(
            bus_number=settings["bus"],
            accelerometer_address=settings["accelerometer_address"],
            gyroscope_address=settings["gyroscope_address"],
            sample_rate_hz=settings["imu_rate_hz"],
        )
        print(
            f"Checking camera at {settings['resolution'][0]}x{settings['resolution'][1]} "
            f"@ {settings['camera_fps']:g} Hz and BMI088 @ {settings['imu_rate_hz']:g} Hz...",
            flush=True,
        )
        report = run_probe(
            camera,
            imu,
            resolution=settings["resolution"],
            camera_fps=settings["camera_fps"],
            imu_rate_hz=settings["imu_rate_hz"],
            duration_s=args.duration,
            minimum_rate_ratio=args.minimum_rate_ratio,
            maximum_sample_age_s=args.maximum_sample_age,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    _print_report(report)
    if report.ok:
        print("PASS: camera and IMU data reached the pipeline correctly.")
        return 0
    print("FAIL: one or more sensor feeds did not pass validation.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Smoke-test live camera and IMU data at the runtime pipeline boundary.

Both devices are sampled concurrently.  The command fails when a device cannot
be read, a message is stale or malformed, timestamps/sequences do not advance,
or either feed falls below the requested fraction of its configured rate.
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Permit ``python tools/check_sensor_pipeline.py`` from an uninstalled checkout,
# independent of the caller's working directory.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from roadbot.messages.sensors import CameraFrame, ImuSample
from roadbot.runtime.sensor_validation import (
    PipelineReport,
    SensorPipelineValidator,
)
from roadbot.sensors.base import Sensor
from roadbot.sensors.camera import Picamera2Camera
from roadbot.sensors.imu import Bmi088Imu

DEFAULT_CONFIG = REPOSITORY_ROOT / "config" / "robot.yaml"
WINDOW_NAME = "Roadbot camera + IMU"


class _LiveSamples:
    """Latest-value exchange between acquisition workers and the display loop."""

    def __init__(self) -> None:
        self.frames: queue.Queue[CameraFrame] = queue.Queue(maxsize=1)
        self._imu: ImuSample | None = None
        self._lock = threading.Lock()

    def publish_frame(self, frame: CameraFrame) -> None:
        try:
            self.frames.put_nowait(frame)
        except queue.Full:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                pass
            self.frames.put_nowait(frame)

    def publish_imu(self, sample: ImuSample) -> None:
        with self._lock:
            self._imu = sample

    def latest_imu(self) -> ImuSample | None:
        with self._lock:
            return self._imu


def _wall_timestamp(timestamp_ns: int, wall_clock_offset_ns: int) -> str:
    wall_seconds = (timestamp_ns + wall_clock_offset_ns) / 1_000_000_000
    return datetime.fromtimestamp(wall_seconds).astimezone().isoformat(timespec="milliseconds")


def _format_imu_sample(sample: ImuSample, wall_clock_offset_ns: int) -> str:
    accel = sample.acceleration_mps2
    gyro = sample.angular_velocity_rad_s
    temperature = "n/a" if sample.temperature_c is None else f"{sample.temperature_c:.2f} C"
    return (
        f"IMU {_wall_timestamp(sample.header.timestamp_ns, wall_clock_offset_ns)} "
        f"timestamp_ns={sample.header.timestamp_ns} seq={sample.header.sequence}  "
        f"accel_mps2[x={accel.x:+8.3f} y={accel.y:+8.3f} z={accel.z:+8.3f}]  "
        f"gyro_rad_s[x={gyro.x:+8.3f} y={gyro.y:+8.3f} z={gyro.z:+8.3f}]  "
        f"temp={temperature}"
    )


def _display_image(frame: CameraFrame, pixel_format: str, cv2: Any) -> np.ndarray:
    image = np.asarray(frame.image).copy()
    if pixel_format == "BGR888":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if pixel_format == "XRGB8888":
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if pixel_format == "XBGR8888":
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return image


def _draw_overlay(
    image: np.ndarray,
    frame: CameraFrame,
    imu: ImuSample | None,
    wall_clock_offset_ns: int,
    cv2: Any,
) -> None:
    overlay = image.copy()
    panel_height = min(image.shape[0], 142)
    cv2.rectangle(overlay, (0, 0), (image.shape[1], panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, image, 0.38, 0, image)

    camera_time = _wall_timestamp(frame.header.timestamp_ns, wall_clock_offset_ns)
    lines = [f"CAM {camera_time}  seq={frame.header.sequence}"]
    if imu is None:
        lines.append("IMU waiting for first sample...")
    else:
        accel = imu.acceleration_mps2
        gyro = imu.angular_velocity_rad_s
        imu_time = _wall_timestamp(imu.header.timestamp_ns, wall_clock_offset_ns)
        lines.extend(
            [
                f"IMU {imu_time}  seq={imu.header.sequence}",
                f"Accel m/s^2  X {accel.x:+8.3f}  Y {accel.y:+8.3f}  Z {accel.z:+8.3f}",
                f"Gyro rad/s  X {gyro.x:+8.3f}  Y {gyro.y:+8.3f}  Z {gyro.z:+8.3f}",
                "Press Q or Esc to stop",
            ]
        )
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (10, 22 + index * 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _load_cv2() -> Any:
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        raise RuntimeError(
            "no graphical desktop was detected; run from a Pi desktop terminal, "
            "use SSH X forwarding, or pass --no-display"
        )
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "live display requires OpenCV; install it with "
            "'sudo apt install python3-opencv' and ensure the virtual environment "
            "was created with --system-site-packages"
        ) from error
    return cv2


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
    duration_s: float | None,
    start: threading.Event,
    stop: threading.Event,
    validator: SensorPipelineValidator,
    on_sample: Callable[[Any], None],
) -> None:
    start.wait()
    deadline = None if duration_s is None else time.monotonic() + duration_s
    period_s = 1.0 / requested_rate_hz
    next_read = time.monotonic()
    while not stop.is_set() and (deadline is None or time.monotonic() < deadline):
        try:
            sample = sensor.read()
            if accept(sample):
                on_sample(sample)
        except (OSError, RuntimeError, ValueError) as error:
            validator.record_issue(name, f"read failed: {error}")
            stop.set()
            return
        next_read += period_s
        wake_time = next_read if deadline is None else min(next_read, deadline)
        remaining = wake_time - time.monotonic()
        if remaining > 0:
            stop.wait(remaining)


def run_probe(
    camera: Sensor[Any],
    imu: Sensor[Any],
    *,
    resolution: tuple[int, int],
    camera_fps: float,
    imu_rate_hz: float,
    duration_s: float | None,
    minimum_rate_ratio: float,
    maximum_sample_age_s: float,
    pixel_format: str = "RGB888",
    display: bool = False,
    imu_output_rate_hz: float = 0.0,
    cv2_module: Any | None = None,
) -> PipelineReport:
    """Open, concurrently sample, display, validate, and close both sources."""
    validator = SensorPipelineValidator(
        resolution,
        max_sample_age_s=maximum_sample_age_s,
    )
    live = _LiveSamples()
    stop = threading.Event()
    wall_clock_offset_ns = time.time_ns() - time.monotonic_ns()
    output_period_s = 0.0 if imu_output_rate_hz == 0 else 1.0 / imu_output_rate_hz
    next_imu_output = 0.0

    def publish_imu(sample: ImuSample) -> None:
        nonlocal next_imu_output
        live.publish_imu(sample)
        current_time = time.monotonic()
        if output_period_s and current_time >= next_imu_output:
            print(_format_imu_sample(sample, wall_clock_offset_ns), flush=True)
            next_imu_output = current_time + output_period_s

    cv2 = None
    if display:
        cv2 = cv2_module if cv2_module is not None else _load_cv2()

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
                    stop,
                    validator,
                    live.publish_frame,
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
                    stop,
                    validator,
                    publish_imu,
                ),
                name="imu-pipeline-check",
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()
        start.set()

        last_frame: CameraFrame | None = None
        try:
            while any(thread.is_alive() for thread in threads):
                if not display:
                    stop.wait(0.05)
                    continue
                assert cv2 is not None
                try:
                    last_frame = live.frames.get(timeout=0.05)
                except queue.Empty:
                    pass
                if last_frame is not None:
                    image = _display_image(last_frame, pixel_format, cv2)
                    _draw_overlay(
                        image,
                        last_frame,
                        live.latest_imu(),
                        wall_clock_offset_ns,
                        cv2,
                    )
                    cv2.imshow(WINDOW_NAME, image)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    stop.set()
        except KeyboardInterrupt:
            print("\nStopped by user.")
            stop.set()
        finally:
            stop.set()
            if cv2 is not None:
                cv2.destroyAllWindows()

        join_timeout = maximum_sample_age_s + 2.0
        for thread in threads:
            thread.join(join_timeout)
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
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="test duration in seconds (default: 5)",
    )
    duration.add_argument(
        "--continuous",
        action="store_true",
        help="run until Q, Esc, or Ctrl+C",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="disable the video window (useful over headless SSH)",
    )
    parser.add_argument(
        "--imu-output-rate",
        type=float,
        default=10.0,
        metavar="HZ",
        help="terminal IMU update rate; use 0 to disable (default: 10)",
    )
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
    if args.imu_output_rate < 0:
        parser.error("--imu-output-rate must be zero or greater")
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
        if not args.no_display:
            print("A live window will open; press Q or Esc to stop it.", flush=True)
        if args.continuous:
            print("Continuous mode enabled; press Ctrl+C to stop from the terminal.", flush=True)
        report = run_probe(
            camera,
            imu,
            resolution=settings["resolution"],
            camera_fps=settings["camera_fps"],
            imu_rate_hz=settings["imu_rate_hz"],
            duration_s=None if args.continuous else args.duration,
            minimum_rate_ratio=args.minimum_rate_ratio,
            maximum_sample_age_s=args.maximum_sample_age,
            pixel_format=settings["pixel_format"],
            display=not args.no_display,
            imu_output_rate_hz=args.imu_output_rate,
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

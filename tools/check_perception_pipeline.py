"""Exercise camera/IMU/encoder synchronization and YOLO preprocessing, without weights."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from roadbot.config import PerceptionConfig, load_perception_config
from roadbot.messages.common import MessageHeader, Vector3
from roadbot.messages.sensors import CameraFrame, ImuSample, WheelEncoderSample
from roadbot.perception.preprocessing import PreparedFrame, YoloPreprocessor
from roadbot.perception.synchronization import SensorSynchronizer
from roadbot.runtime.workers import PerceptionRuntime
from roadbot.sensors.camera import Picamera2Camera
from roadbot.sensors.encoders import WheelEncoders
from roadbot.sensors.imu import Bmi088Imu
from roadbot.sensors.imu_calibration import CalibratedImu, ImuCalibration


def synthetic_frames(config: PerceptionConfig, duration: float, sync: SensorSynchronizer):
    """Replay deterministic multi-rate events with camera delivery latency."""
    preprocess = YoloPreprocessor(config.preprocessing)
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:, :, 2] = 255  # BGR red; useful for inspecting channel conversion
    events = []
    for name, rate, delay_ns in (
        ("camera", 30, 20_000_000),
        ("imu", 100, 1_000_000),
        ("encoders", 50, 2_000_000),
    ):
        for sequence in range(max(1, math.ceil(duration * rate))):
            stamp = 1_000_000_000 + round(sequence / rate * 1e9)
            header = MessageHeader(stamp, sequence, name, timestamp_source="simulated")
            if name == "camera":
                message = CameraFrame(header, image, {})
            elif name == "imu":
                message = ImuSample(header, Vector3(0, 0, 9.80665), Vector3(0, 0, 0))
            else:
                message = WheelEncoderSample(header, sequence, sequence)
            events.append((stamp + delay_ns, message))
    events.sort(key=lambda event: event[0])
    for now, message in events:
        sync.add(message, now_ns=now)
        while (packet := sync.poll(now_ns=now)) is not None:
            yield preprocess.prepare(packet)
    now = events[-1][0] + round(config.synchronization.max_wait_ms * 1e6)
    while (packet := sync.poll(now_ns=now)) is not None:
        yield preprocess.prepare(packet)


def live_runtime(path: Path, config: PerceptionConfig) -> PerceptionRuntime:
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    camera, imu = settings["camera"], settings["imu"]
    motor, encoders = settings["motor_controller"], settings.get("encoders", {})
    camera_source = Picamera2Camera(
        tuple(camera["resolution"]),
        camera["format"],
        camera["fps"],
        horizontal_flip=camera.get("horizontal_flip", False),
        vertical_flip=camera.get("vertical_flip", False),
        frame_id=camera.get("frame_id", "camera_optical_frame"),
    )
    imu_source = Bmi088Imu(
        imu["bus"], imu["accelerometer_address"], imu["gyroscope_address"], imu["sample_rate_hz"]
    )
    calibration_path = Path(imu["calibration_file"])
    if not calibration_path.is_absolute():
        calibration_path = path.resolve().parent.parent / calibration_path
    calibration = ImuCalibration.load(calibration_path)
    if calibration.calibrated:
        imu_source = CalibratedImu(imu_source, calibration)
    else:
        print("IMU calibration incomplete: retaining raw SI values in imu_link.", file=sys.stderr)
    encoder_source = WheelEncoders(
        motor["bus"],
        motor["address"],
        motor["left_channel"],
        motor["right_channel"],
        encoders.get("counts_per_revolution"),
    )
    return PerceptionRuntime(
        {
            "camera": (camera_source, camera["fps"]),
            "imu": (imu_source, imu["sample_rate_hz"]),
            "encoders": (encoder_source, encoders.get("sample_rate_hz", 50)),
        },
        config,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "config/robot.yaml")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument(
        "--synthetic", action="store_true", help="run deterministic data without hardware"
    )
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("duration must be finite and positive")
    count = 0
    max_imu_skew = max_encoder_skew = 0
    last: PreparedFrame | None = None

    def observe(frame: PreparedFrame) -> None:
        nonlocal count, max_imu_skew, max_encoder_skew, last
        count += 1
        max_imu_skew = max(max_imu_skew, abs(frame.sensors.imu_offset_ns))
        max_encoder_skew = max(max_encoder_skew, abs(frame.sensors.encoder_offset_ns))
        last = frame

    try:
        config = load_perception_config(args.config)
        if args.synthetic:
            sync = SensorSynchronizer(
                replace(config.synchronization, require_camera_sensor_timestamp=False)
            )
            for frame in synthetic_frames(config, args.duration, sync):
                observe(frame)
            diagnostics = sync.diagnostics()
        else:
            with live_runtime(args.config, config) as runtime:
                deadline = time.monotonic() + args.duration
                while time.monotonic() < deadline:
                    frame = runtime.read(timeout_s=min(0.1, max(0, deadline - time.monotonic())))
                    if frame is not None:
                        observe(frame)
                diagnostics = runtime.synchronizer.diagnostics()
                diagnostics["preprocessing_stale"] = runtime.preprocessing_stale_count
        print(
            json.dumps(
                {
                    "prepared_frames": count,
                    "tensor_shape": list(last.tensor.shape) if last is not None else None,
                    "max_imu_skew_ms": max_imu_skew / 1e6,
                    "max_encoder_skew_ms": max_encoder_skew / 1e6,
                    "diagnostics": diagnostics,
                },
                indent=2,
            )
        )
        if not count:
            print(
                "FAIL: no synchronized frames; inspect feed and timestamp diagnostics.",
                file=sys.stderr,
            )
            return 1
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, RuntimeError, ValueError, TypeError, ImportError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

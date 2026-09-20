import importlib.util
import time
from pathlib import Path

import numpy as np

from roadbot.messages.common import MessageHeader, Vector3
from roadbot.messages.sensors import CameraFrame, ImuSample

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "tools" / "check_sensor_pipeline.py"
SPEC = importlib.util.spec_from_file_location("check_sensor_pipeline", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
check_sensor_pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_sensor_pipeline)


class FakeCamera:
    def __init__(self) -> None:
        self.sequence = 0
        self.opened = False
        self.closed = False

    def open(self) -> None:
        self.opened = True

    def read(self) -> CameraFrame:
        frame = CameraFrame(
            MessageHeader(time.monotonic_ns(), self.sequence, "camera"),
            np.zeros((48, 64, 3), dtype=np.uint8),
            {},
        )
        self.sequence += 1
        return frame

    def close(self) -> None:
        self.closed = True


class FakeImu:
    def __init__(self) -> None:
        self.sequence = 0
        self.opened = False
        self.closed = False

    def open(self) -> None:
        self.opened = True

    def read(self) -> ImuSample:
        sample = ImuSample(
            MessageHeader(time.monotonic_ns(), self.sequence, "imu"),
            Vector3(1.0, 2.0, 3.0),
            Vector3(4.0, 5.0, 6.0),
            24.5,
        )
        self.sequence += 1
        return sample

    def close(self) -> None:
        self.closed = True


def test_headless_probe_streams_and_validates_both_sensors() -> None:
    camera = FakeCamera()
    imu = FakeImu()

    report = check_sensor_pipeline.run_probe(
        camera,
        imu,
        resolution=(64, 48),
        camera_fps=20.0,
        imu_rate_hz=40.0,
        duration_s=0.2,
        minimum_rate_ratio=0.1,
        maximum_sample_age_s=0.5,
    )

    assert report.ok
    assert report.camera.valid_samples >= 2
    assert report.imu.valid_samples >= 2
    assert camera.opened and camera.closed
    assert imu.opened and imu.closed


def test_imu_output_contains_timestamp_and_all_axes() -> None:
    timestamp_ns = 1_700_000_000_000_000_000
    sample = ImuSample(
        MessageHeader(timestamp_ns, 7, "imu"),
        Vector3(1.0, 2.0, 3.0),
        Vector3(4.0, 5.0, 6.0),
        24.5,
    )

    output = check_sensor_pipeline._format_imu_sample(sample, 0)

    assert f"timestamp_ns={timestamp_ns}" in output
    assert "accel_mps2[x=" in output
    assert "y=" in output and "z=" in output
    assert "gyro_rad_s[x=" in output


def test_live_frame_exchange_drops_old_frames() -> None:
    live = check_sensor_pipeline._LiveSamples()
    first = CameraFrame(
        MessageHeader(1, 1, "camera"),
        np.zeros((1, 1, 3), dtype=np.uint8),
        {},
    )
    second = CameraFrame(
        MessageHeader(2, 2, "camera"),
        np.zeros((1, 1, 3), dtype=np.uint8),
        {},
    )

    live.publish_frame(first)
    live.publish_frame(second)

    assert live.frames.get_nowait() is second

import numpy as np

from roadbot.messages.common import MessageHeader, Vector3
from roadbot.messages.sensors import CameraFrame, ImuSample
from roadbot.runtime.sensor_validation import SensorPipelineValidator


def camera_frame(timestamp_ns: int, sequence: int, shape=(48, 64, 3)) -> CameraFrame:
    return CameraFrame(
        header=MessageHeader(timestamp_ns, sequence, "camera"),
        image=np.zeros(shape, dtype=np.uint8),
        metadata={"ExposureTime": 1_000},
    )


def imu_sample(timestamp_ns: int, sequence: int) -> ImuSample:
    return ImuSample(
        header=MessageHeader(timestamp_ns, sequence, "imu"),
        acceleration_mps2=Vector3(0.0, 0.0, 9.80665),
        angular_velocity_rad_s=Vector3(0.0, 0.0, 0.0),
        temperature_c=24.0,
    )


def test_accepts_well_formed_feeds_at_required_rates() -> None:
    current = [1_000_000_000]
    validator = SensorPipelineValidator((64, 48), clock=lambda: current[0])

    assert validator.accept_camera(camera_frame(current[0], 0))
    assert validator.accept_imu(imu_sample(current[0], 0))
    current[0] += 100_000_000
    assert validator.accept_camera(camera_frame(current[0], 1))
    assert validator.accept_imu(imu_sample(current[0], 1))

    report = validator.report(minimum_camera_hz=10.0, minimum_imu_hz=10.0)

    assert report.ok
    assert report.camera.rate_hz == 10.0
    assert report.imu.rate_hz == 10.0


def test_rejects_wrong_camera_dimensions() -> None:
    validator = SensorPipelineValidator((64, 48), clock=lambda: 1_000_000_000)

    assert not validator.accept_camera(camera_frame(1_000_000_000, 0, (24, 32, 3)))

    report = validator.report(minimum_camera_hz=1.0, minimum_imu_hz=1.0)
    assert not report.ok
    assert any("frame size" in issue for issue in report.camera.issues)


def test_rejects_stale_and_non_finite_imu_data() -> None:
    validator = SensorPipelineValidator(
        (64, 48),
        max_sample_age_s=0.5,
        clock=lambda: 2_000_000_000,
    )
    sample = ImuSample(
        header=MessageHeader(1_000_000_000, 0, "imu"),
        acceleration_mps2=Vector3(float("nan"), 0.0, 0.0),
        angular_velocity_rad_s=Vector3(0.0, 0.0, 0.0),
    )

    assert not validator.accept_imu(sample)
    report = validator.report(minimum_camera_hz=1.0, minimum_imu_hz=1.0)
    assert any("stale" in issue for issue in report.imu.issues)
    assert any("non-finite" in issue for issue in report.imu.issues)


def test_rejects_non_increasing_headers() -> None:
    current = [1_000_000_000]
    validator = SensorPipelineValidator((64, 48), clock=lambda: current[0])
    assert validator.accept_camera(camera_frame(current[0], 4))
    current[0] += 1

    assert not validator.accept_camera(camera_frame(current[0], 4))

    report = validator.report(minimum_camera_hz=1.0, minimum_imu_hz=1.0)
    assert any("sequence did not increase" in issue for issue in report.camera.issues)

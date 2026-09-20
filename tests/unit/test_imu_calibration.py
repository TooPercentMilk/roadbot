from pathlib import Path

import numpy as np
import pytest

from roadbot.messages.common import MessageHeader, Vector3
from roadbot.messages.sensors import ImuSample
from roadbot.sensors.imu import STANDARD_GRAVITY_MPS2
from roadbot.sensors.imu_calibration import (
    FrameCalibration,
    ImuCalibration,
    TriadCalibration,
    allan_deviation,
    fit_accelerometer_six_position,
    fit_temperature_model,
    parse_axis_mapping,
)


def test_six_position_fit_recovers_bias_and_correction() -> None:
    correction = np.array(
        [
            [1.02, 0.01, -0.005],
            [0.004, 0.98, 0.006],
            [-0.003, 0.008, 1.01],
        ]
    )
    bias = np.array([0.14, -0.09, 0.21])
    targets = STANDARD_GRAVITY_MPS2 * np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    measured = (np.linalg.inv(correction) @ targets.T).T + bias

    fit = fit_accelerometer_six_position(measured)

    assert fit.bias_mps2 == pytest.approx(bias)
    assert fit.correction_matrix == pytest.approx(correction)
    assert fit.maximum_residual_mps2 < 1e-10


def test_temperature_fit_recovers_intercept_and_slope() -> None:
    temperatures = np.linspace(10.0, 40.0, 100)
    reference = 25.0
    intercept = np.array([0.1, -0.2, 0.3])
    slope = np.array([0.01, -0.02, 0.005])
    values = intercept + (temperatures - reference)[:, None] * slope

    fit = fit_temperature_model(
        temperatures,
        values,
        reference_temperature_c=reference,
    )

    assert fit.intercept == pytest.approx(intercept)
    assert fit.coefficient_per_c == pytest.approx(slope)
    assert fit.temperature_span_c == 30.0


def test_calibration_applies_temperature_then_body_rotation() -> None:
    calibration = ImuCalibration(
        accelerometer=TriadCalibration(
            calibrated=True,
            bias=np.array([1.0, 2.0, 3.0]),
            correction_matrix=np.diag([2.0, 3.0, 4.0]),
            reference_temperature_c=20.0,
            temperature_coefficient=np.array([0.1, 0.0, 0.0]),
        ),
        gyroscope=TriadCalibration(
            calibrated=True,
            bias=np.array([0.1, 0.2, 0.3]),
        ),
        frame=FrameCalibration(
            calibrated=True,
            rotation_matrix=parse_axis_mapping("+y,-x,+z"),
        ),
    )
    raw = ImuSample(
        MessageHeader(1, 2, "imu"),
        Vector3(3.0, 4.0, 5.0),
        Vector3(1.1, 2.2, 3.3),
        30.0,
    )

    corrected = calibration.apply(raw)

    assert corrected.acceleration_mps2 == Vector3(6.0, -2.0, 8.0)
    assert corrected.angular_velocity_rad_s.x == pytest.approx(2.0)
    assert corrected.angular_velocity_rad_s.y == pytest.approx(-1.0)
    assert corrected.angular_velocity_rad_s.z == pytest.approx(3.0)
    assert corrected.frame_id == "base_link"

    batch_acceleration, batch_angular_velocity = calibration.apply_batch(
        np.array([[3.0, 4.0, 5.0]]),
        np.array([[1.1, 2.2, 3.3]]),
        np.array([30.0]),
    )
    assert batch_acceleration[0] == pytest.approx([6.0, -2.0, 8.0])
    assert batch_angular_velocity[0] == pytest.approx([2.0, -1.0, 3.0])


def test_calibration_yaml_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "imu.yaml"
    calibration = ImuCalibration()
    calibration.gyroscope.bias = np.array([0.1, 0.2, 0.3])
    calibration.gyroscope.calibrated = True
    calibration.noise = {"sample_rate_hz": 100.0}
    calibration.save(path)

    loaded = ImuCalibration.load(path)

    assert loaded.gyroscope.bias == pytest.approx([0.1, 0.2, 0.3])
    assert loaded.gyroscope.calibrated
    assert loaded.noise["sample_rate_hz"] == 100.0


def test_axis_mapping_rejects_reflection() -> None:
    assert parse_axis_mapping("+x,-y,-z") == pytest.approx(np.diag([1.0, -1.0, -1.0]))
    with pytest.raises(ValueError, match="right-handed"):
        parse_axis_mapping("+x,+y,-z")


def test_allan_deviation_returns_finite_three_axis_summary() -> None:
    random = np.random.default_rng(7)
    samples = random.normal(0.0, 0.01, size=(2000, 3))

    result = allan_deviation(samples, 100.0)

    assert result.deviation.shape[1] == 3
    assert np.all(np.diff(result.cluster_times_s) > 0)
    assert np.all(np.isfinite(result.white_noise_coefficient))
    assert np.all(result.bias_instability > 0)

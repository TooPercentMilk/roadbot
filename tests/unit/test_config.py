"""Tests for configuration parsing and validation."""

from pathlib import Path

import pytest

from roadbot.config import PreprocessingConfig, SynchronizationConfig, load_perception_config


@pytest.mark.parametrize(
    "kwargs",
    [
        {"imu_tolerance_ms": 0},
        {"max_age_ms": float("nan")},
        {"max_wait_ms": 300},
        {"camera_buffer_size": 0},
        {"imu_buffer_size": 1.5},
        {"require_camera_sensor_timestamp": "false"},
    ],
)
def test_invalid_sync_settings(kwargs):
    with pytest.raises(ValueError):
        SynchronizationConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [{"input_size": (640, 641)}, {"stride": 0}, {"padding_value": 256}, {"scale_up": "false"}],
)
def test_invalid_preprocessing_settings(kwargs):
    with pytest.raises(ValueError):
        PreprocessingConfig(**kwargs)


def test_checked_in_perception_configuration():
    config = load_perception_config(Path(__file__).resolve().parents[2] / "config/robot.yaml")
    assert config.preprocessing.input_size == (640, 640)
    assert config.synchronization.require_camera_sensor_timestamp

"""IMU calibration models, fitting routines, and calibrated sensor wrapper."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from roadbot.messages.common import Vector3
from roadbot.messages.sensors import ImuSample
from roadbot.sensors.base import Sensor
from roadbot.sensors.imu import STANDARD_GRAVITY_MPS2

CALIBRATION_VERSION = 1
IDENTITY_MATRIX = np.eye(3, dtype=float)
ZERO_VECTOR = np.zeros(3, dtype=float)


def _vector(values: Any, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite values")
    return result


def _matrix(values: Any, *, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    return result


def _vector3(values: np.ndarray) -> Vector3:
    return Vector3(*(float(value) for value in values))


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def parse_axis_mapping(mapping: str) -> np.ndarray:
    """Parse body X/Y/Z as a signed permutation of sensor X/Y/Z.

    For example, ``"+x,-y,-z"`` means body X equals sensor +X, body Y
    equals sensor -Y, and body Z equals sensor -Z.
    """
    tokens = [token.strip().lower() for token in mapping.split(",")]
    if len(tokens) != 3:
        raise ValueError("axis mapping must contain three comma-separated axes")
    axes = {"x": 0, "y": 1, "z": 2}
    rotation = np.zeros((3, 3), dtype=float)
    used_axes: set[int] = set()
    for body_axis, token in enumerate(tokens):
        if token.startswith(("+", "-")):
            sign = -1.0 if token[0] == "-" else 1.0
            name = token[1:]
        else:
            sign = 1.0
            name = token
        if name not in axes:
            raise ValueError(f"invalid mapped axis {token!r}; use signed x, y, or z")
        sensor_axis = axes[name]
        if sensor_axis in used_axes:
            raise ValueError("axis mapping must use each sensor axis exactly once")
        used_axes.add(sensor_axis)
        rotation[body_axis, sensor_axis] = sign
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-9):
        raise ValueError("axis mapping must be right-handed (determinant +1)")
    return rotation


@dataclass(slots=True)
class TriadCalibration:
    """Bias, scale/cross-axis, and temperature model for one three-axis sensor."""

    calibrated: bool = False
    bias: np.ndarray = field(default_factory=lambda: ZERO_VECTOR.copy())
    correction_matrix: np.ndarray = field(default_factory=lambda: IDENTITY_MATRIX.copy())
    reference_temperature_c: float | None = None
    temperature_coefficient: np.ndarray = field(default_factory=lambda: ZERO_VECTOR.copy())

    def __post_init__(self) -> None:
        self.bias = _vector(self.bias, name="bias")
        self.correction_matrix = _matrix(
            self.correction_matrix,
            name="correction_matrix",
        )
        self.temperature_coefficient = _vector(
            self.temperature_coefficient,
            name="temperature_coefficient",
        )
        if self.reference_temperature_c is not None and not math.isfinite(
            self.reference_temperature_c
        ):
            raise ValueError("reference_temperature_c must be finite")
        if abs(float(np.linalg.det(self.correction_matrix))) < 1e-9:
            raise ValueError("correction_matrix must be invertible")

    def bias_at(self, temperature_c: float | None) -> np.ndarray:
        if temperature_c is None or self.reference_temperature_c is None:
            return self.bias
        return self.bias + self.temperature_coefficient * (
            temperature_c - self.reference_temperature_c
        )

    def apply(self, values: np.ndarray, temperature_c: float | None) -> np.ndarray:
        return self.correction_matrix @ (values - self.bias_at(temperature_c))

    def apply_batch(
        self,
        values: np.ndarray,
        temperatures_c: np.ndarray,
    ) -> np.ndarray:
        measurements = np.asarray(values, dtype=float)
        temperatures = np.asarray(temperatures_c, dtype=float)
        if measurements.ndim != 2 or measurements.shape[1] != 3:
            raise ValueError("batch measurements must have shape (sample_count, 3)")
        if temperatures.shape != (measurements.shape[0],):
            raise ValueError("temperature count must match batch measurement count")
        biases = np.broadcast_to(self.bias, measurements.shape).copy()
        if self.reference_temperature_c is not None:
            temperature_delta = np.where(
                np.isfinite(temperatures),
                temperatures - self.reference_temperature_c,
                0.0,
            )
            biases += temperature_delta[:, None] * self.temperature_coefficient
        return (self.correction_matrix @ (measurements - biases).T).T


@dataclass(slots=True)
class FrameCalibration:
    """Fixed, proper rotation from the IMU sensor frame to the robot body frame."""

    calibrated: bool = False
    source: str = "imu_link"
    target: str = "base_link"
    rotation_matrix: np.ndarray = field(default_factory=lambda: IDENTITY_MATRIX.copy())

    def __post_init__(self) -> None:
        self.rotation_matrix = _matrix(self.rotation_matrix, name="rotation_matrix")
        should_be_identity = self.rotation_matrix @ self.rotation_matrix.T
        if not np.allclose(should_be_identity, IDENTITY_MATRIX, atol=1e-6):
            raise ValueError("frame rotation_matrix must be orthonormal")
        if not math.isclose(float(np.linalg.det(self.rotation_matrix)), 1.0, abs_tol=1e-6):
            raise ValueError("frame rotation_matrix must be right-handed (determinant +1)")
        if not self.source or not self.target:
            raise ValueError("frame source and target names cannot be empty")


@dataclass(slots=True)
class ImuCalibration:
    """Complete persistent calibration for a BMI088 installation."""

    accelerometer: TriadCalibration = field(default_factory=TriadCalibration)
    gyroscope: TriadCalibration = field(default_factory=TriadCalibration)
    frame: FrameCalibration = field(default_factory=FrameCalibration)
    temperature_calibrated: bool = False
    noise: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def calibrated(self) -> bool:
        return (
            self.accelerometer.calibrated
            and self.gyroscope.calibrated
            and self.frame.calibrated
        )

    def apply(self, sample: ImuSample) -> ImuSample:
        acceleration = np.array(
            [
                sample.acceleration_mps2.x,
                sample.acceleration_mps2.y,
                sample.acceleration_mps2.z,
            ],
            dtype=float,
        )
        angular_velocity = np.array(
            [
                sample.angular_velocity_rad_s.x,
                sample.angular_velocity_rad_s.y,
                sample.angular_velocity_rad_s.z,
            ],
            dtype=float,
        )
        acceleration = self.frame.rotation_matrix @ self.accelerometer.apply(
            acceleration,
            sample.temperature_c,
        )
        angular_velocity = self.frame.rotation_matrix @ self.gyroscope.apply(
            angular_velocity,
            sample.temperature_c,
        )
        return ImuSample(
            header=sample.header,
            acceleration_mps2=_vector3(acceleration),
            angular_velocity_rad_s=_vector3(angular_velocity),
            temperature_c=sample.temperature_c,
            frame_id=self.frame.target,
        )

    def apply_batch(
        self,
        acceleration_mps2: np.ndarray,
        angular_velocity_rad_s: np.ndarray,
        temperatures_c: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calibrate arrays and rotate them into the configured robot frame."""
        acceleration = self.accelerometer.apply_batch(acceleration_mps2, temperatures_c)
        angular_velocity = self.gyroscope.apply_batch(
            angular_velocity_rad_s,
            temperatures_c,
        )
        return (
            (self.frame.rotation_matrix @ acceleration.T).T,
            (self.frame.rotation_matrix @ angular_velocity.T).T,
        )

    def to_dict(self) -> dict[str, Any]:
        def triad(sensor: TriadCalibration, coefficient_name: str) -> dict[str, Any]:
            return {
                "calibrated": sensor.calibrated,
                "bias": sensor.bias.tolist(),
                "correction_matrix": sensor.correction_matrix.tolist(),
                "reference_temperature_c": sensor.reference_temperature_c,
                coefficient_name: sensor.temperature_coefficient.tolist(),
            }

        return {
            "version": CALIBRATION_VERSION,
            "sensor_model": "BMI088",
            "calibrated": self.calibrated,
            "accelerometer": triad(
                self.accelerometer,
                "bias_temperature_coefficient_mps2_per_c",
            ),
            "gyroscope": triad(
                self.gyroscope,
                "bias_temperature_coefficient_rad_s_per_c",
            ),
            "temperature": {"calibrated": self.temperature_calibrated},
            "frame": {
                "calibrated": self.frame.calibrated,
                "source": self.frame.source,
                "target": self.frame.target,
                "rotation_matrix": self.frame.rotation_matrix.tolist(),
            },
            "noise": self.noise,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, document: dict[str, Any] | None) -> ImuCalibration:
        document = document or {}
        version = int(document.get("version", CALIBRATION_VERSION))
        if version != CALIBRATION_VERSION:
            raise ValueError(
                f"unsupported IMU calibration version {version}; expected {CALIBRATION_VERSION}"
            )

        def triad(section_name: str, coefficient_name: str) -> TriadCalibration:
            section = document.get(section_name, {}) or {}
            if not isinstance(section, dict):
                raise TypeError(f"{section_name} calibration must be a mapping")
            return TriadCalibration(
                calibrated=bool(section.get("calibrated", False)),
                bias=section.get("bias", ZERO_VECTOR),
                correction_matrix=section.get("correction_matrix", IDENTITY_MATRIX),
                reference_temperature_c=section.get("reference_temperature_c"),
                temperature_coefficient=section.get(coefficient_name, ZERO_VECTOR),
            )

        frame_section = document.get("frame", {}) or {}
        if not isinstance(frame_section, dict):
            raise TypeError("frame calibration must be a mapping")
        temperature_section = document.get("temperature", {}) or {}
        noise = document.get("noise", {}) or {}
        metadata = document.get("metadata", {}) or {}
        if not all(isinstance(value, dict) for value in (temperature_section, noise, metadata)):
            raise ValueError("temperature, noise, and metadata must be mappings")
        return cls(
            accelerometer=triad(
                "accelerometer",
                "bias_temperature_coefficient_mps2_per_c",
            ),
            gyroscope=triad(
                "gyroscope",
                "bias_temperature_coefficient_rad_s_per_c",
            ),
            frame=FrameCalibration(
                calibrated=bool(frame_section.get("calibrated", False)),
                source=str(frame_section.get("source", "imu_link")),
                target=str(frame_section.get("target", "base_link")),
                rotation_matrix=frame_section.get("rotation_matrix", IDENTITY_MATRIX),
            ),
            temperature_calibrated=bool(temperature_section.get("calibrated", False)),
            noise=dict(noise),
            metadata=dict(metadata),
        )

    @classmethod
    def load(cls, path: Path) -> ImuCalibration:
        if not path.exists():
            return cls()
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(f"cannot read IMU calibration from {path}: {error}") from error
        if document is not None and not isinstance(document, dict):
            raise ValueError(f"IMU calibration in {path} must be a YAML mapping")
        return cls.from_dict(document)

    def save(self, path: Path) -> None:
        """Atomically replace a calibration file after a successful fit."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        contents = yaml.safe_dump(self.to_dict(), sort_keys=False)
        try:
            temporary_path.write_text(contents, encoding="utf-8")
            temporary_path.replace(path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def mark_updated(self, operation: str) -> None:
        self.metadata["last_updated_utc"] = _timestamp()
        self.metadata["last_operation"] = operation


@dataclass(frozen=True, slots=True)
class AccelerometerFit:
    bias_mps2: np.ndarray
    correction_matrix: np.ndarray
    residual_rms_mps2: float
    maximum_residual_mps2: float


def fit_accelerometer_six_position(
    pose_means_mps2: np.ndarray,
    gravity_mps2: float = STANDARD_GRAVITY_MPS2,
) -> AccelerometerFit:
    """Fit a full affine calibration from +X/-X/+Y/-Y/+Z/-Z means."""
    means = np.asarray(pose_means_mps2, dtype=float)
    if means.shape != (6, 3) or not np.all(np.isfinite(means)):
        raise ValueError("pose_means_mps2 must be a finite 6x3 array")
    if not math.isfinite(gravity_mps2) or gravity_mps2 <= 0:
        raise ValueError("gravity_mps2 must be positive and finite")

    targets = gravity_mps2 * np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    design = np.column_stack((means, np.ones(6)))
    coefficients, _, rank, _ = np.linalg.lstsq(design, targets, rcond=None)
    if rank < 4:
        raise ValueError("six-position observations do not span three dimensions")
    correction = coefficients[:3, :].T
    if abs(float(np.linalg.det(correction))) < 1e-9:
        raise ValueError("fitted accelerometer correction is singular")
    offset = coefficients[3, :]
    bias = np.linalg.solve(correction, -offset)
    corrected = (correction @ (means - bias).T).T
    residual_norms = np.linalg.norm(corrected - targets, axis=1)
    return AccelerometerFit(
        bias_mps2=bias,
        correction_matrix=correction,
        residual_rms_mps2=float(np.sqrt(np.mean(residual_norms**2))),
        maximum_residual_mps2=float(np.max(residual_norms)),
    )


@dataclass(frozen=True, slots=True)
class TemperatureFit:
    reference_temperature_c: float
    intercept: np.ndarray
    coefficient_per_c: np.ndarray
    residual_rms: np.ndarray
    temperature_span_c: float


def fit_temperature_model(
    temperatures_c: np.ndarray,
    measurements: np.ndarray,
    *,
    reference_temperature_c: float | None = None,
) -> TemperatureFit:
    """Fit one linear measurement-versus-temperature model for each axis."""
    temperatures = np.asarray(temperatures_c, dtype=float)
    values = np.asarray(measurements, dtype=float)
    if temperatures.ndim != 1 or values.shape != (temperatures.size, 3):
        raise ValueError("measurements must have shape (len(temperatures_c), 3)")
    if temperatures.size < 6 or not np.all(np.isfinite(temperatures)):
        raise ValueError("at least six finite temperature samples are required")
    if not np.all(np.isfinite(values)):
        raise ValueError("measurements must be finite")
    reference = (
        float(np.median(temperatures))
        if reference_temperature_c is None
        else float(reference_temperature_c)
    )
    design = np.column_stack((np.ones(temperatures.size), temperatures - reference))
    coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
    predicted = design @ coefficients
    return TemperatureFit(
        reference_temperature_c=reference,
        intercept=coefficients[0],
        coefficient_per_c=coefficients[1],
        residual_rms=np.sqrt(np.mean((values - predicted) ** 2, axis=0)),
        temperature_span_c=float(np.ptp(temperatures)),
    )


@dataclass(frozen=True, slots=True)
class AllanResult:
    cluster_times_s: np.ndarray
    deviation: np.ndarray
    white_noise_coefficient: np.ndarray
    bias_instability: np.ndarray


def allan_deviation(
    measurements: np.ndarray,
    sample_rate_hz: float,
    *,
    point_count: int = 40,
) -> AllanResult:
    """Calculate non-overlapping Allan deviation and useful summary coefficients."""
    values = np.asarray(measurements, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] < 20:
        raise ValueError("measurements must contain at least 20 three-axis samples")
    if not np.all(np.isfinite(values)):
        raise ValueError("measurements must be finite")
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive and finite")
    maximum_cluster = values.shape[0] // 4
    clusters = np.unique(
        np.logspace(0, math.log10(maximum_cluster), max(2, point_count)).astype(int)
    )
    deviations: list[np.ndarray] = []
    valid_clusters: list[int] = []
    for cluster_size in clusters:
        cluster_count = values.shape[0] // cluster_size
        if cluster_count < 2:
            continue
        truncated = values[: cluster_count * cluster_size]
        averages = truncated.reshape(cluster_count, cluster_size, 3).mean(axis=1)
        difference = np.diff(averages, axis=0)
        deviations.append(np.sqrt(0.5 * np.mean(difference**2, axis=0)))
        valid_clusters.append(int(cluster_size))
    deviation = np.asarray(deviations)
    cluster_times = np.asarray(valid_clusters, dtype=float) / sample_rate_hz
    initial_count = min(5, cluster_times.size)
    white_noise = np.median(
        deviation[:initial_count] * np.sqrt(cluster_times[:initial_count, None]),
        axis=0,
    )
    bias_instability = np.min(deviation, axis=0) / 0.664
    return AllanResult(
        cluster_times_s=cluster_times,
        deviation=deviation,
        white_noise_coefficient=white_noise,
        bias_instability=bias_instability,
    )


class CalibratedImu:
    """Sensor wrapper applying intrinsic, temperature, and body-frame calibration."""

    def __init__(
        self,
        source: Sensor[ImuSample],
        calibration: ImuCalibration,
        *,
        require_complete: bool = True,
    ) -> None:
        if require_complete and not calibration.calibrated:
            raise ValueError(
                "IMU calibration is incomplete; accelerometer, gyroscope, and frame "
                "must all be calibrated"
            )
        self._source = source
        self.calibration = calibration

    def open(self) -> None:
        self._source.open()

    def read(self) -> ImuSample:
        return self.calibration.apply(self._source.read())

    def close(self) -> None:
        self._source.close()

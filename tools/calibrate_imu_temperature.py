"""Fit stationary linear BMI088 bias-versus-temperature models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from _imu_calibration_cli import (
    DEFAULT_CONFIG,
    create_imu,
    load_imu_settings,
    record,
    save_recording,
    wait_for_confirmation,
)

from roadbot.sensors.imu_calibration import ImuCalibration, fit_temperature_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--calibration", type=Path, help="override calibration YAML path")
    parser.add_argument(
        "--duration",
        type=float,
        default=900.0,
        help="recording duration while the cold sensor warms up",
    )
    parser.add_argument("--minimum-span-c", type=float, default=4.0)
    parser.add_argument(
        "--maximum-gyro-residual",
        type=float,
        default=0.02,
        help="maximum per-axis temperature-fit residual RMS in rad/s",
    )
    parser.add_argument(
        "--maximum-accel-residual",
        type=float,
        default=0.20,
        help="maximum per-axis temperature-fit residual RMS in m/s^2",
    )
    parser.add_argument("--raw-output", type=Path, help="optional raw sample CSV")
    parser.add_argument("--force", action="store_true", help="save despite quality failures")
    parser.add_argument("--yes", action="store_true", help="skip the Enter prompt")
    return parser


def run(args: argparse.Namespace) -> int:
    settings = load_imu_settings(args.config, args.calibration)
    wait_for_confirmation(
        "Cold-soak the powered-off robot beforehand, then run this command promptly after "
        "power-on. Place it in one fixed orientation on a rigid surface and do not move it "
        "while the electronics warm naturally.",
        args.yes,
    )
    recording = record(
        create_imu(settings),
        duration_s=args.duration,
        sample_rate_hz=settings.sample_rate_hz,
    )
    if args.raw_output:
        save_recording(args.raw_output, recording)
    finite = np.isfinite(recording.temperature_c)
    gyro_fit = fit_temperature_model(
        recording.temperature_c[finite],
        recording.angular_velocity_rad_s[finite],
    )
    accel_fit = fit_temperature_model(
        recording.temperature_c[finite],
        recording.acceleration_mps2[finite],
    )
    print(
        f"Temperature range: {np.min(recording.temperature_c[finite]):.2f} to "
        f"{np.max(recording.temperature_c[finite]):.2f} C "
        f"(span {gyro_fit.temperature_span_c:.2f} C)"
    )
    print(
        "Gyro bias coefficient [rad/s/C]: "
        + ", ".join(f"{value:+.9g}" for value in gyro_fit.coefficient_per_c)
    )
    print(
        "Accel bias coefficient [m/s^2/C]: "
        + ", ".join(f"{value:+.9g}" for value in accel_fit.coefficient_per_c)
    )
    print(
        "Gyro fit residual RMS [rad/s]: "
        + ", ".join(f"{value:.9g}" for value in gyro_fit.residual_rms)
    )
    print(
        "Accel fit residual RMS [m/s^2]: "
        + ", ".join(f"{value:.9g}" for value in accel_fit.residual_rms)
    )
    quality_failed = (
        gyro_fit.temperature_span_c < args.minimum_span_c
        or np.max(gyro_fit.residual_rms) > args.maximum_gyro_residual
        or np.max(accel_fit.residual_rms) > args.maximum_accel_residual
    )
    if quality_failed and not args.force:
        print(
            "ERROR: temperature span is insufficient or motion made the fit residual too "
            "large; calibration was not changed. Cool the robot further and keep it still.",
            file=sys.stderr,
        )
        return 2

    calibration = ImuCalibration.load(settings.calibration_file)
    # A stationary gyro measures bias directly, so the fitted intercept becomes
    # its bias at the fit reference temperature.
    calibration.gyroscope.bias = gyro_fit.intercept
    calibration.gyroscope.reference_temperature_c = gyro_fit.reference_temperature_c
    calibration.gyroscope.temperature_coefficient = gyro_fit.coefficient_per_c
    calibration.gyroscope.calibrated = True
    # Acceleration contains gravity in this fixed pose. Its slope is usable, but
    # its intercept is not an accelerometer bias and must not replace the six-pose fit.
    calibration.accelerometer.temperature_coefficient = accel_fit.coefficient_per_c
    if calibration.accelerometer.reference_temperature_c is None:
        calibration.accelerometer.reference_temperature_c = accel_fit.reference_temperature_c
    calibration.temperature_calibrated = True
    calibration.metadata["temperature_span_c"] = gyro_fit.temperature_span_c
    calibration.metadata["gyro_temperature_fit_rms_rad_s"] = gyro_fit.residual_rms.tolist()
    calibration.metadata["accel_temperature_fit_rms_mps2"] = accel_fit.residual_rms.tolist()
    calibration.mark_updated("temperature")
    calibration.save(settings.calibration_file)
    print(f"Updated {settings.calibration_file}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if (
        args.duration <= 0
        or args.minimum_span_c <= 0
        or args.maximum_gyro_residual <= 0
        or args.maximum_accel_residual <= 0
    ):
        parser.error("duration, temperature span, and residual limits must be positive")
    try:
        return run(args)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

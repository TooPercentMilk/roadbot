"""Measure stationary BMI088 gyroscope bias and update calibration/imu.yaml."""

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

from roadbot.sensors.imu_calibration import ImuCalibration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--calibration", type=Path, help="override calibration YAML path")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--warmup", type=float, default=10.0)
    parser.add_argument(
        "--maximum-stddev",
        type=float,
        default=0.01,
        help="maximum stationary per-axis standard deviation in rad/s",
    )
    parser.add_argument("--raw-output", type=Path, help="optional raw sample CSV")
    parser.add_argument("--force", action="store_true", help="save despite stationarity failure")
    parser.add_argument("--yes", action="store_true", help="skip the Enter prompt")
    return parser


def run(args: argparse.Namespace) -> int:
    settings = load_imu_settings(args.config, args.calibration)
    wait_for_confirmation(
        "Place the fully assembled robot on a rigid surface. Do not touch it, run motors, "
        "or disturb the table during warmup and recording.",
        args.yes,
    )
    recording = record(
        create_imu(settings),
        duration_s=args.duration,
        sample_rate_hz=settings.sample_rate_hz,
        warmup_s=args.warmup,
    )
    if args.raw_output:
        save_recording(args.raw_output, recording)

    bias = np.mean(recording.angular_velocity_rad_s, axis=0)
    stddev = np.std(recording.angular_velocity_rad_s, axis=0, ddof=1)
    temperature = float(np.nanmean(recording.temperature_c))
    print(f"Samples: {recording.sample_count}; effective rate: {recording.effective_rate_hz:.2f} Hz")
    print(f"Mean temperature: {temperature:.2f} C")
    print("Gyro bias [rad/s]: " + ", ".join(f"{value:+.8f}" for value in bias))
    print("Gyro stddev [rad/s]: " + ", ".join(f"{value:.8f}" for value in stddev))
    if np.max(stddev) > args.maximum_stddev and not args.force:
        print(
            "ERROR: recording was not stationary enough; calibration was not changed. "
            "Use a more stable surface or --force to override.",
            file=sys.stderr,
        )
        return 2

    calibration = ImuCalibration.load(settings.calibration_file)
    calibration.gyroscope.bias = bias
    calibration.gyroscope.reference_temperature_c = temperature
    calibration.gyroscope.calibrated = True
    calibration.mark_updated("gyroscope_bias")
    calibration.save(settings.calibration_file)
    print(f"Updated {settings.calibration_file}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.duration <= 0 or args.warmup < 0 or args.maximum_stddev <= 0:
        parser.error("duration/stddev must be positive and warmup cannot be negative")
    try:
        return run(args)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

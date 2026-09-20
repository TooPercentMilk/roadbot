"""Run a guided six-position BMI088 accelerometer calibration."""

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

from roadbot.sensors.imu_calibration import ImuCalibration, fit_accelerometer_six_position

POSES = (
    ("+X", "sensor +X arrow pointing straight upward"),
    ("-X", "sensor +X arrow pointing straight downward"),
    ("+Y", "sensor +Y arrow pointing straight upward"),
    ("-Y", "sensor +Y arrow pointing straight downward"),
    ("+Z", "sensor +Z axis pointing straight upward"),
    ("-Z", "sensor +Z axis pointing straight downward"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--calibration", type=Path, help="override calibration YAML path")
    parser.add_argument("--duration-per-pose", type=float, default=8.0)
    parser.add_argument("--warmup", type=float, default=10.0)
    parser.add_argument(
        "--maximum-stddev",
        type=float,
        default=0.15,
        help="maximum stationary per-axis standard deviation in m/s^2",
    )
    parser.add_argument(
        "--maximum-fit-residual",
        type=float,
        default=0.15,
        help="maximum allowed fitted pose residual in m/s^2",
    )
    parser.add_argument(
        "--maximum-correction",
        type=float,
        default=0.20,
        help="maximum absolute correction-matrix departure from identity",
    )
    parser.add_argument(
        "--raw-output-dir",
        type=Path,
        help="optional directory for one raw CSV per pose",
    )
    parser.add_argument("--force", action="store_true", help="save despite quality failures")
    parser.add_argument("--yes", action="store_true", help="use countdowns instead of Enter prompts")
    return parser


def run(args: argparse.Namespace) -> int:
    settings = load_imu_settings(args.config, args.calibration)
    means = []
    temperatures = []
    imu = create_imu(settings)
    for index, (name, instruction) in enumerate(POSES):
        wait_for_confirmation(
            f"Pose {index + 1}/6 ({name}): securely place the assembled robot with the "
            f"{instruction}. Keep it motionless.",
            args.yes,
        )
        recording = record(
            imu,
            duration_s=args.duration_per_pose,
            sample_rate_hz=settings.sample_rate_hz,
            warmup_s=args.warmup if index == 0 else 0.0,
        )
        if args.raw_output_dir:
            pose_filename = f"accel_pose_{index + 1}_{name.replace('+', 'plus').replace('-', 'minus')}.csv"
            save_recording(args.raw_output_dir / pose_filename, recording)
        mean = np.mean(recording.acceleration_mps2, axis=0)
        stddev = np.std(recording.acceleration_mps2, axis=0, ddof=1)
        print(f"  mean: {mean}; stddev: {stddev}")
        if np.max(stddev) > args.maximum_stddev and not args.force:
            print(
                f"ERROR: {name} was not stationary enough; calibration was not changed.",
                file=sys.stderr,
            )
            return 2
        means.append(mean)
        temperatures.append(float(np.nanmean(recording.temperature_c)))

    fit = fit_accelerometer_six_position(np.asarray(means))
    print("Accelerometer bias [m/s^2]: " + ", ".join(f"{v:+.8f}" for v in fit.bias_mps2))
    print("Correction matrix:")
    for row in fit.correction_matrix:
        print("  [" + ", ".join(f"{value:+.9f}" for value in row) + "]")
    print(f"Fit RMS residual: {fit.residual_rms_mps2:.6f} m/s^2")
    print(f"Fit maximum residual: {fit.maximum_residual_mps2:.6f} m/s^2")
    correction_departure = float(np.max(np.abs(fit.correction_matrix - np.eye(3))))
    print(f"Maximum correction-matrix departure from identity: {correction_departure:.6f}")
    if (
        fit.maximum_residual_mps2 > args.maximum_fit_residual
        or correction_departure > args.maximum_correction
    ) and not args.force:
        print(
            "ERROR: fit residual or correction is too large; check pose labels/alignment and "
            "repeat. Calibration was not changed.",
            file=sys.stderr,
        )
        return 2

    calibration = ImuCalibration.load(settings.calibration_file)
    calibration.accelerometer.bias = fit.bias_mps2
    calibration.accelerometer.correction_matrix = fit.correction_matrix
    calibration.accelerometer.reference_temperature_c = float(np.mean(temperatures))
    calibration.accelerometer.calibrated = True
    calibration.metadata["accelerometer_fit_rms_mps2"] = fit.residual_rms_mps2
    calibration.metadata["accelerometer_fit_maximum_mps2"] = fit.maximum_residual_mps2
    calibration.mark_updated("accelerometer_six_position")
    calibration.save(settings.calibration_file)
    print(f"Updated {settings.calibration_file}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if (
        args.duration_per_pose <= 0
        or args.warmup < 0
        or args.maximum_stddev <= 0
        or args.maximum_fit_residual <= 0
        or args.maximum_correction <= 0
    ):
        parser.error("durations and quality thresholds must be positive")
    try:
        return run(args)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

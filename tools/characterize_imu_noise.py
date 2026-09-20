"""Record a stationary BMI088 and calculate covariance and Allan deviation."""

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
    write_allan_csv,
)

from roadbot.sensors.imu_calibration import ImuCalibration, allan_deviation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--calibration", type=Path, help="override calibration YAML path")
    parser.add_argument("--duration", type=float, default=1800.0)
    parser.add_argument("--warmup", type=float, default=60.0)
    parser.add_argument("--maximum-gyro-stddev", type=float, default=0.02)
    parser.add_argument("--maximum-accel-norm-stddev", type=float, default=0.20)
    parser.add_argument("--raw-output", type=Path, help="optional raw sample CSV")
    parser.add_argument(
        "--allan-output",
        type=Path,
        default=Path("data/recordings/imu_allan.csv"),
    )
    parser.add_argument("--force", action="store_true", help="analyze despite detected motion")
    parser.add_argument("--yes", action="store_true", help="skip the Enter prompt")
    return parser


def run(args: argparse.Namespace) -> int:
    settings = load_imu_settings(args.config, args.calibration)
    calibration = ImuCalibration.load(settings.calibration_file)
    if not calibration.calibrated:
        raise ValueError(
            "accelerometer, gyroscope, and frame calibration must be completed before "
            "body-frame noise characterization"
        )
    wait_for_confirmation(
        "Place the robot on a rigid, vibration-free surface in a thermally stable room. "
        "Do not touch it or run motors for the entire recording.",
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
    acceleration, angular_velocity = calibration.apply_batch(
        recording.acceleration_mps2,
        recording.angular_velocity_rad_s,
        recording.temperature_c,
    )
    gyro_stddev = np.std(angular_velocity, axis=0, ddof=1)
    accel_norm_stddev = float(np.std(np.linalg.norm(acceleration, axis=1), ddof=1))
    print("Gyro stddev [rad/s]: " + ", ".join(f"{value:.9g}" for value in gyro_stddev))
    print(f"Acceleration-norm stddev [m/s^2]: {accel_norm_stddev:.9g}")
    if (
        np.max(gyro_stddev) > args.maximum_gyro_stddev
        or accel_norm_stddev > args.maximum_accel_norm_stddev
    ) and not args.force:
        print(
            "ERROR: motion or vibration was detected; noise calibration was not changed. "
            "Repeat on a more stable surface or use --force to analyze deliberately.",
            file=sys.stderr,
        )
        return 2
    rate = recording.effective_rate_hz
    acceleration_covariance = np.cov(acceleration, rowvar=False, ddof=1)
    angular_velocity_covariance = np.cov(angular_velocity, rowvar=False, ddof=1)
    accel_allan = allan_deviation(acceleration, rate)
    gyro_allan = allan_deviation(angular_velocity, rate)
    if not np.allclose(accel_allan.cluster_times_s, gyro_allan.cluster_times_s):
        raise RuntimeError("accelerometer and gyroscope Allan cluster times differ")

    args.allan_output.parent.mkdir(parents=True, exist_ok=True)
    with args.allan_output.open("w", newline="", encoding="utf-8") as csv_file:
        write_allan_csv(
            csv_file,
            accel_allan.cluster_times_s,
            accel_allan.deviation,
            gyro_allan.deviation,
        )
    calibration.noise = {
        "calibrated": True,
        "frame": calibration.frame.target,
        "sample_rate_hz": float(rate),
        "sample_count": recording.sample_count,
        "duration_s": args.duration,
        "accelerometer_covariance_mps2_squared": acceleration_covariance.tolist(),
        "gyroscope_covariance_rad_s_squared": angular_velocity_covariance.tolist(),
        "accelerometer_allan_white_noise_mps2_sqrt_s": (
            accel_allan.white_noise_coefficient.tolist()
        ),
        "gyroscope_allan_white_noise_rad_s_sqrt_s": (
            gyro_allan.white_noise_coefficient.tolist()
        ),
        "accelerometer_bias_instability_mps2": accel_allan.bias_instability.tolist(),
        "gyroscope_bias_instability_rad_s": gyro_allan.bias_instability.tolist(),
        "allan_curve_csv": str(args.allan_output),
    }
    calibration.mark_updated("noise_characterization")
    calibration.save(settings.calibration_file)
    print(f"Effective sample rate: {rate:.3f} Hz")
    print("Gyroscope covariance:")
    for row in angular_velocity_covariance:
        print("  [" + ", ".join(f"{value:.9g}" for value in row) + "]")
    print(
        "Gyro Allan white-noise coefficient [rad/s*sqrt(s)]: "
        + ", ".join(f"{value:.9g}" for value in gyro_allan.white_noise_coefficient)
    )
    print(
        "Gyro bias instability [rad/s]: "
        + ", ".join(f"{value:.9g}" for value in gyro_allan.bias_instability)
    )
    print(f"Wrote Allan curve to {args.allan_output}")
    print(f"Updated {settings.calibration_file}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if (
        args.duration <= 0
        or args.warmup < 0
        or args.maximum_gyro_stddev <= 0
        or args.maximum_accel_norm_stddev <= 0
    ):
        parser.error("duration/limits must be positive and warmup cannot be negative")
    try:
        return run(args)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

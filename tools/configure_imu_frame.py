"""Configure the fixed signed-axis rotation from the BMI088 to the robot body."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _imu_calibration_cli import DEFAULT_CONFIG, load_imu_settings

from roadbot.sensors.imu_calibration import (
    FrameCalibration,
    ImuCalibration,
    parse_axis_mapping,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mapping",
        required=True,
        help=(
            "body X,Y,Z expressed as sensor axes, e.g. '+x,+y,+z' or '+x,-y,-z'; "
            "when the value starts with '-', use --mapping=..."
        ),
    )
    parser.add_argument("--source-frame", default="imu_link")
    parser.add_argument("--target-frame", default="base_link")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--calibration", type=Path, help="override calibration YAML path")
    return parser


def run(args: argparse.Namespace) -> int:
    settings = load_imu_settings(args.config, args.calibration)
    rotation = parse_axis_mapping(args.mapping)
    calibration = ImuCalibration.load(settings.calibration_file)
    calibration.frame = FrameCalibration(
        calibrated=True,
        source=args.source_frame,
        target=args.target_frame,
        rotation_matrix=rotation,
    )
    calibration.mark_updated("frame_mapping")
    calibration.save(settings.calibration_file)
    print("Sensor-to-body rotation matrix:")
    for row in rotation:
        print("  [" + ", ".join(f"{value:+.0f}" for value in row) + "]")
    print(f"Updated {settings.calibration_file}")
    return 0


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run(args)
    except (OSError, TypeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

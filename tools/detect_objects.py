"""Run pretrained YOLO26n .pt inference on synchronized perception frames."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

from check_perception_pipeline import REPOSITORY_ROOT, live_runtime, synthetic_frames

from roadbot.clock import now_ns
from roadbot.config import load_perception_config
from roadbot.perception.object_detector import UltralyticsDetector
from roadbot.perception.synchronization import SensorSynchronizer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "config/robot.yaml")
    parser.add_argument(
        "--weights", type=Path, default=REPOSITORY_ROOT / "models/deployment/yolo26n.pt"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("duration must be finite and positive")
    if args.weights.suffix.lower() != ".pt":
        parser.error("this runner requires .pt weights")
    config = load_perception_config(args.config)
    detector = UltralyticsDetector.from_weights(
        args.weights, device=args.device, confidence=args.confidence
    )
    # Initialize the predictor before starting hardware acquisition.
    import torch

    detector.model.predict(
        source=torch.zeros(1, 3, *config.preprocessing.input_size),
        device=args.device,
        verbose=False,
    )
    processed = expired = 0

    def detect(frame):
        nonlocal processed, expired
        started = time.monotonic()
        result = detector.detect(frame)
        processed += 1
        stale = (
            not args.synthetic
            and now_ns() - result.header.timestamp_ns > config.synchronization.max_age_ms * 1e6
        )
        expired += int(stale)
        print(
            json.dumps(
                {
                    "result": asdict(result),
                    "stale": stale,
                    "inference_ms": (time.monotonic() - started) * 1000,
                }
            ),
            flush=True,
        )

    if args.synthetic:
        sync = SensorSynchronizer(
            replace(config.synchronization, require_camera_sensor_timestamp=False)
        )
        for frame in synthetic_frames(config, args.duration, sync):
            detect(frame)
    else:
        with live_runtime(args.config, config) as runtime:
            deadline = time.monotonic() + args.duration
            while time.monotonic() < deadline:
                frame = runtime.read(timeout_s=min(0.1, max(0, deadline - time.monotonic())))
                if frame is not None:
                    detect(frame)
            sync = runtime.synchronizer
    print(
        json.dumps(
            {
                "processed_frames": processed,
                "expired_results": expired,
                "diagnostics": sync.diagnostics(),
            }
        )
    )
    return 0 if processed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None

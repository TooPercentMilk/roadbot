# Roadbot

Python software for a Raspberry Pi 5 autonomous Ackermann-steering robot.

The first milestone is hardware bring-up: verify the camera, BMI088 IMU, I2C bus,
and chassis encoder telemetry independently before adding perception or control.

## Raspberry Pi environment

Picamera2 and OpenCV should be installed from Raspberry Pi OS packages so they
remain compatible with libcamera. Create the project virtual environment with
access to system packages:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e '.[hardware,dev]'
```

## Camera and IMU pipeline check

With the camera and BMI088 connected, verify that both feeds reach the runtime
pipeline with valid messages and adequate sample rates:

```bash
python tools/check_sensor_pipeline.py --duration 10
```

The check runs both inputs concurrently and fails on device/read errors, stale
or non-monotonic messages, incorrect camera dimensions, non-finite IMU values,
or an observed rate below 70% of the rate in `config/robot.yaml`. Use
`--minimum-rate-ratio` to choose a different threshold.

## Encoder diagnostic

The encoder diagnostic is read-only: it never commands the motors. On the Pi,
first confirm I2C is enabled and the motor-controller board is powered. Safely
raise the driven wheels, then run:

```bash
python tools/check_encoders.py --duration 15 --expect-motion
```

Rotate each driven wheel by hand during the test. The script reads all four
cumulative counters and requires activity from the left and right channels set
in `config/robot.yaml`. To save the samples:

```bash
python tools/check_encoders.py --duration 15 --expect-motion \
  --output data/recordings/encoder_check.csv
```

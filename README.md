# Roadbot

Python software for a Raspberry Pi 5 autonomous Ackermann-steering robot.

The first milestone is hardware bring-up: verify the camera, BMI088 IMU, I2C bus,
and chassis encoder telemetry independently before adding perception or control.

## Raspberry Pi environment

Picamera2 and OpenCV should be installed from Raspberry Pi OS packages so they
remain compatible with libcamera. Create the project virtual environment with
access to system packages:

```bash
sudo apt install -y python3-picamera2 python3-opencv
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

The check displays the live camera feed with timestamped accelerometer and
gyroscope X/Y/Z values overlaid, and prints timestamped IMU samples to the
terminal at 10 Hz. Press Q or Esc in the video window to stop early. Run until
stopped manually with:

```bash
python tools/check_sensor_pipeline.py --continuous
```

For a headless SSH session, use `--no-display`; change terminal output frequency
with `--imu-output-rate HZ`. The check runs both inputs concurrently and fails
on device/read errors, stale or non-monotonic messages, incorrect camera
dimensions, non-finite IMU values, or an observed rate below 70% of the rate in
`config/robot.yaml`.

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

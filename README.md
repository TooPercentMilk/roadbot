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

Most hardware scripts in `tools/` are placeholders until the individual drivers
are implemented and wiring has been verified.

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

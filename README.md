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

## IMU calibration

The BMI088 driver uses a +/-250 degree/second gyroscope range. Perform IMU
calibration after the sensor and its PCB are permanently mounted. All commands
below update `calibration/imu.yaml` incrementally and leave the other completed
sections intact.

First record the stationary gyroscope bias on a rigid, vibration-free surface:

```bash
python tools/calibrate_gyro_bias.py --duration 20 --warmup 10 \
  --raw-output data/recordings/imu_gyro_bias.csv
```

Next run the guided six-position accelerometer calibration. Each instruction
refers to the axes printed on the sensor board. Secure the entire assembled
robot in every pose; do not calibrate the loose sensor separately.

```bash
python tools/calibrate_accelerometer.py --duration-per-pose 8 \
  --raw-output-dir data/recordings/imu_accelerometer
```

Configure the fixed sensor-to-robot rotation. The mapping lists robot body
X-forward, Y-left, Z-up as signed sensor axes. An aligned installation uses the
identity example below. A board whose sensor X is forward, Y is right, and Z is
down uses `--mapping=+x,-y,-z`.

```bash
python tools/configure_imu_frame.py --mapping=+x,+y,+z
```

For temperature calibration, cold-soak the powered-off robot, then power it on
and run the command promptly. Leave it stationary while it warms naturally.
The command requires at least a 4 C change by default; a controlled temperature
chamber covering the expected operating range produces a better result.

```bash
python tools/calibrate_imu_temperature.py --duration 900 \
  --raw-output data/recordings/imu_temperature.csv
```

Finally, let the robot reach thermal equilibrium and characterize noise. Thirty
minutes is a useful minimum; several hours gives a more informative Allan
deviation curve.

```bash
python tools/characterize_imu_noise.py --warmup 60 --duration 1800 \
  --raw-output data/recordings/imu_noise.csv \
  --allan-output data/recordings/imu_allan.csv
```

The noise command writes calibrated body-frame covariance and Allan summary
values to `calibration/imu.yaml`. It writes the complete curve to the requested
CSV. Calibration commands reject visibly moving or poor-quality datasets rather
than overwriting a good calibration; `--force` is available for deliberate
experiments.

`Bmi088Imu` continues to publish SI-unit data in `imu_link`, which keeps raw
diagnostics available. Runtime consumers should wrap it with `CalibratedImu`;
that layer applies temperature, bias, scale/cross-axis, and sensor-to-body
rotation corrections and publishes the result in `base_link`. Gravity removal
and online residual-bias estimation remain localization responsibilities.

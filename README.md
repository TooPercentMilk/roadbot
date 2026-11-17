# Roadbot

Python software for a Raspberry Pi 5 autonomous Ackermann-steering robot.

Hardware diagnostics verify the camera, BMI088 IMU, I2C bus, and chassis encoder
telemetry independently. The perception intake now associates all three sensor
streams by timestamp and prepares fixed-shape inputs for YOLO26.

## Perception intake and YOLO26 preparation

Pretrained YOLO26n PyTorch inference is available with:

```bash
python tools/detect_objects.py --duration 10
```

The runner loads `models/deployment/yolo26n.pt` once, warms up the CPU predictor,
then runs detection on synchronized frames. It emits JSON records containing
camera timestamps, class names, confidence, original-image boxes, inference time,
and a stale-result flag. All pretrained COCO classes are enabled. Stale results
are diagnostic output and must not be used for control.

Test real model execution without the Pi with
`python tools/detect_objects.py --synthetic --duration 0.1`. Synthetic images are
blank color fields, so empty detections are expected. Use `--device` to select
another installed PyTorch device or `--confidence` to change the threshold.
Install dependencies with `python -m pip install -e '.[yolo,dev]'`. On a new
machine, download the official pretrained weights once with:

```bash
python -c "from ultralytics import YOLO; YOLO('models/deployment/yolo26n.pt')"
```

Weights are intentionally excluded from Git. This uses the `.pt` runtime;
no export or retraining is needed.

Run the complete intake on the Pi without model weights or detection targets:

```bash
python tools/check_perception_pipeline.py --duration 10
```

This reads the camera, IMU, and encoders concurrently, associates measurements
with each camera acquisition timestamp, and reports synchronized frame counts,
timestamp offsets, and rejected/dropped data. It does not command the motors.
Incomplete IMU calibration leaves samples explicitly in `imu_link`; encoder
angular speeds remain unknown until counts per wheel revolution is configured.

For a hardware-free smoke test on a development machine:

```bash
python -m pip install -e '.[perception,dev]'
python tools/check_perception_pipeline.py --synthetic --duration 2
```

The output is a contiguous RGB float32 tensor shaped `[1, 3, 640, 640]`, with
letterboxing and normalization, accompanied by original sensor messages and an
inverse image transform. Settings live under `perception` in `config/robot.yaml`.
No weights or target classes are selected automatically. See
[the perception architecture and integration guide](docs/perception.md) for
clock semantics, queue policies, YOLO26/ONNX integration, and hardware checks.

## Raspberry Pi environment

Picamera2 and OpenCV should be installed from Raspberry Pi OS packages so they
remain compatible with libcamera. Create the project virtual environment with
access to system packages:

```bash
sudo apt install -y python3-picamera2 python3-opencv python3-gpiozero python3-lgpio
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

## Shared I2C communication check

After connecting the BMI088 IMU and Hiwonder motor controller to the same SDA
and SCL lines, probe all three configured I2C endpoints:

```bash
python tools/check_i2c.py
```

The check reads the BMI088 accelerometer and gyroscope chip IDs at `0x19` and
`0x69`, then reads encoder telemetry from the motor controller at `0x34`. It
does not send motor speed, direction, PWM, or IMU configuration commands. A
successful run prints one PASS line per address and a final all-devices PASS.

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

## Drive-motor bench test

This test intentionally moves the rear drive wheels. Raise both driven wheels
completely off the ground, clear hands and loose items from the drivetrain, and
keep the motor-power switch within reach. Then run:

```bash
python tools/check_motors.py --enable-motors
```

The command configures the documented JGB encoder mode, then exercises M1 and
M3 individually at a low closed-loop target in both directions. It stops all
four channels between steps and on Ctrl+C or an error. The test passes only if
both motors move in both directions, their encoder counts change and reverse
sign, and the final stop command succeeds. For the first run, keep the default
speed and one-second duration.

## Steering bench test

This interactive test intentionally moves the front steering linkage. Raise
the front wheels, stop the drive motors, keep hands clear, and power the servo
from its rated external supply with a common ground to the Pi. Do not power the
servo from a GPIO signal pin. Run:

```bash
python tools/check_steering.py --enable-steering
```

The test commands the configured 1500 us center, then conservative 1250 us
left and 1750 us right positions. It returns to center after every movement
and asks you to confirm the observed position because the servo provides no
position feedback. On exit it centers the linkage and disables PWM. Stop and
switch off servo power if you hear continuous buzzing or see mechanical
binding. The larger documented 1000/2000 us steering positions should wait
until the safe endpoints are calibrated.

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

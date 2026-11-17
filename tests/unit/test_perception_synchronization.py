from dataclasses import replace

import numpy as np
import pytest

from roadbot.config import SynchronizationConfig
from roadbot.messages.common import MessageHeader, Vector3
from roadbot.messages.sensors import CameraFrame, ImuSample, WheelEncoderSample
from roadbot.perception.synchronization import SensorSynchronizer


def stamp(ms):
    return 1_000_000_000 + round(ms * 1e6)


def message(name, ms, sequence=None):
    header = MessageHeader(
        stamp(ms),
        round(ms) if sequence is None else sequence,
        name,
        timestamp_source="sensor_boottime" if name == "camera" else "host_read_midpoint",
    )
    if name == "camera":
        return CameraFrame(header, np.zeros((48, 64, 3), np.uint8), {})
    if name == "imu":
        return ImuSample(header, Vector3(0, 0, 9.8), Vector3(0, 0, 0))
    return WheelEncoderSample(header, 1, 2)


def add(sync, name, ms, sequence=None, now=120):
    sample = message(name, ms, sequence)
    assert sync.add(sample, now_ns=stamp(max(now, ms)))
    return sample


def test_waits_for_crossing_then_selects_nearest_preserving_acquisition_headers():
    sync = SensorSynchronizer()
    camera = add(sync, "camera", 100)
    add(sync, "imu", 90)
    add(sync, "encoders", 95)
    assert sync.poll(now_ns=stamp(120)) is None
    imu = add(sync, "imu", 102)
    encoders = add(sync, "encoders", 104)
    packet = sync.poll(now_ns=stamp(121))
    assert packet.camera is camera and packet.imu is imu and packet.encoders is encoders
    assert packet.header is camera.header
    assert packet.imu_offset_ns == 2_000_000 and packet.encoder_offset_ns == 4_000_000
    assert sync.poll(now_ns=stamp(121)) is None


def test_timeout_allows_nearby_past_samples_without_restamping():
    sync = SensorSynchronizer()
    add(sync, "camera", 100)
    add(sync, "imu", 85)
    add(sync, "encoders", 70)
    assert sync.poll(now_ns=stamp(179)) is None
    packet = sync.poll(now_ns=stamp(180))
    assert packet.imu.header.timestamp_ns == stamp(85)
    assert packet.encoders.header.timestamp_ns == stamp(70)


@pytest.mark.parametrize("missing", ["imu", "encoders"])
def test_missing_stream_drops_camera_at_deadline(missing):
    sync = SensorSynchronizer()
    add(sync, "camera", 100)
    add(sync, "encoders" if missing == "imu" else "imu", 100)
    assert sync.poll(now_ns=stamp(180)) is None
    assert sync.diagnostics()["camera_dropped_unsynchronized"] == 1
    assert sync.diagnostics()["camera_buffered"] == 0


def test_outside_tolerance_and_stale_frames_are_dropped():
    sync = SensorSynchronizer()
    add(sync, "camera", 100)
    add(sync, "imu", 116)
    add(sync, "encoders", 100)
    assert sync.poll(now_ns=stamp(120)) is None
    assert sync.diagnostics()["camera_dropped_unsynchronized"] == 1
    add(sync, "camera", 200, now=200)
    assert sync.poll(now_ns=stamp(451)) is None
    assert sync.diagnostics()["camera_dropped_stale"] == 1


def test_queued_sensor_cannot_be_reused_after_its_age_limit():
    sync = SensorSynchronizer()
    add(sync, "camera", 100)
    add(sync, "imu", 90)
    add(sync, "encoders", 100)
    # Camera remains within 250 ms, but the buffered IMU sample has expired.
    assert sync.poll(now_ns=stamp(345)) is None
    assert sync.diagnostics()["camera_dropped_unsynchronized"] == 1


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"clock_domain": "wall"}, "clock_domain"),
        ({"timestamp_ns": stamp(121)}, "future"),
        ({"timestamp_ns": stamp(-200)}, "stale"),
        ({"timestamp_ns": 0}, "header"),
        ({"timestamp_source": "host_receive"}, "timestamp_source"),
    ],
)
def test_rejects_wrong_clock_and_untrustworthy_camera_times(change, reason):
    sync = SensorSynchronizer()
    sample = message("camera", 100)
    sample = replace(sample, header=replace(sample.header, **change))
    assert not sync.add(sample, now_ns=stamp(120))
    assert sync.diagnostics()[f"camera_rejected_{reason}"] == 1


def test_rejects_duplicates_reordered_sequences_and_nonfinite_values():
    sync = SensorSynchronizer()
    first = add(sync, "imu", 100, 4)
    assert not sync.add(first, now_ns=stamp(120))
    assert not sync.add(message("imu", 101, 3), now_ns=stamp(120))
    bad = replace(message("imu", 102, 5), acceleration_mps2=Vector3(float("nan"), 0, 0))
    assert not sync.add(bad, now_ns=stamp(120))
    assert sync.diagnostics()["imu_rejected_out_of_order"] == 2
    assert sync.diagnostics()["imu_rejected_nonfinite"] == 1


def test_buffers_are_bounded_and_reset_allows_new_session():
    sync = SensorSynchronizer(
        SynchronizationConfig(camera_buffer_size=2, imu_buffer_size=3, encoder_buffer_size=2)
    )
    for ms in range(10):
        for name in ("camera", "imu", "encoders"):
            add(sync, name, ms)
    stats = sync.diagnostics()
    assert (stats["camera_buffered"], stats["imu_buffered"], stats["encoders_buffered"]) == (
        2,
        3,
        2,
    )
    assert (stats["camera_overflow"], stats["imu_overflow"], stats["encoders_overflow"]) == (
        8,
        7,
        8,
    )
    sync.reset()
    add(sync, "camera", 0, 0)


@pytest.mark.parametrize("gap", [False, True])
def test_retains_imu_interval_and_reports_missing_samples(gap):
    sync = SensorSynchronizer()
    add(sync, "imu", 100, 0)
    add(sync, "encoders", 100)
    add(sync, "camera", 100)
    assert not sync.poll(now_ns=stamp(120)).imu_window_complete
    add(sync, "imu", 110, 1)
    add(sync, "imu", 120, 3 if gap else 2)
    add(sync, "imu", 130, 4 if gap else 3, now=130)
    add(sync, "encoders", 130, now=130)
    add(sync, "camera", 125, now=140)
    packet = sync.poll(now_ns=stamp(140))
    assert [s.header.timestamp_ns for s in packet.imu_window] == [stamp(110), stamp(120)]
    assert packet.imu_window_complete is not gap
    assert packet.previous_camera_timestamp_ns == stamp(100)


def test_ties_choose_earlier_and_sensor_samples_can_be_reused():
    sync = SensorSynchronizer()
    add(sync, "imu", 90)
    add(sync, "imu", 110)
    add(sync, "encoders", 100)
    add(sync, "camera", 100)
    assert sync.poll(now_ns=stamp(120)).imu.header.timestamp_ns == stamp(90)
    add(sync, "camera", 105)
    packet = sync.poll(now_ns=stamp(180))
    assert packet.imu.header.timestamp_ns == stamp(110)
    assert packet.encoders.header.timestamp_ns == stamp(100)

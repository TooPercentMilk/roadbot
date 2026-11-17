import numpy as np
import pytest

from roadbot.clock import now_ns
from roadbot.config import PerceptionConfig, PreprocessingConfig, SynchronizationConfig
from roadbot.messages.common import MessageHeader, Vector3
from roadbot.messages.sensors import CameraFrame, ImuSample, WheelEncoderSample
from roadbot.runtime.workers import PerceptionRuntime


class FakeSource:
    def __init__(self, name, *, fail=False):
        self.name, self.fail = name, fail
        self.sequence = 0
        self.closed = False

    def open(self):
        pass

    def read(self):
        if self.fail:
            raise OSError("disconnected")
        header = MessageHeader(now_ns(), self.sequence, self.name)
        self.sequence += 1
        if self.name == "camera":
            return CameraFrame(header, np.zeros((48, 64, 3), np.uint8), {})
        if self.name == "imu":
            return ImuSample(header, Vector3(0, 0, 9.8), Vector3(0, 0, 0))
        return WheelEncoderSample(header, 0, 0)

    def close(self):
        self.closed = True


def make_runtime(fail=False):
    sources = {
        name: (FakeSource(name, fail=fail and name == "imu"), 100)
        for name in ("camera", "imu", "encoders")
    }
    config = PerceptionConfig(
        SynchronizationConfig(require_camera_sensor_timestamp=False, max_age_ms=1000),
        PreprocessingConfig((64, 64)),
    )
    return PerceptionRuntime(sources, config)


def test_three_workers_produce_prepared_frames_and_close_all_sources():
    runtime = make_runtime()
    with runtime:
        result = runtime.read(timeout_s=1)
        assert result is not None
        assert result.tensor.shape == (1, 3, 64, 64)
        assert result.sensors.encoders.left_count == 0
        assert runtime.prepared_count == 1
    assert all(source.closed for source, _ in runtime.sources.values())
    assert not runtime._threads


def test_worker_failure_reaches_consumer_and_all_sources_close():
    runtime = make_runtime(fail=True)
    runtime.start()
    with pytest.raises(RuntimeError, match="imu acquisition failed.*disconnected"):
        runtime.read()
    with pytest.raises(RuntimeError, match="imu acquisition failed"):
        runtime.close()
    assert all(source.closed for source, _ in runtime.sources.values())


def test_runtime_drops_frame_that_expires_during_preprocessing():
    runtime = make_runtime()
    prepare = runtime.preprocessor.prepare

    def expire(sensors):
        output = prepare(sensors)
        # Deliberately advance the acquisition clock beyond the latency budget.
        runtime._clock = lambda: sensors.header.timestamp_ns + 2_000_000_000
        return output

    runtime.preprocessor.prepare = expire
    with runtime:
        assert runtime.read(timeout_s=0.05) is None
        assert runtime.preprocessing_stale_count >= 1
        assert runtime.prepared_count == 0

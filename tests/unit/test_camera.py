import numpy as np

from roadbot.clock import map_sensor_timestamp
from roadbot.sensors.camera import Picamera2Camera


class FakeRequest:
    def __init__(self) -> None:
        self.released = False

    def make_array(self, stream: str) -> np.ndarray:
        assert stream == "main"
        return np.zeros((48, 64, 3), dtype=np.uint8)

    def get_metadata(self) -> dict[str, int]:
        return {"ExposureTime": 1000}

    def release(self) -> None:
        self.released = True


class FakePicamera2:
    def __init__(self) -> None:
        self.request = FakeRequest()
        self.configuration = None
        self.started = False
        self.stopped = False
        self.closed = False

    def create_video_configuration(self, **configuration):
        return configuration

    def configure(self, configuration) -> None:
        self.configuration = configuration

    def start(self) -> None:
        self.started = True

    def capture_request(self) -> FakeRequest:
        return self.request

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.closed = True


def test_camera_emits_frame_and_matching_metadata() -> None:
    backend = FakePicamera2()
    camera = Picamera2Camera(
        (64, 48),
        camera_factory=lambda: backend,
        clock=lambda: 123,
    )

    camera.open()
    frame = camera.read()
    camera.close()

    assert backend.started and backend.stopped and backend.closed
    assert backend.request.released
    assert frame.header.timestamp_ns == 123
    assert frame.header.sequence == 0
    assert frame.image.shape == (48, 64, 3)
    assert frame.metadata == {"ExposureTime": 1000}
    assert frame.encoding == "bgr8"
    assert frame.header.timestamp_source == "host_receive"


def test_camera_maps_boot_clock_and_owns_pixels_before_release():
    backend = FakePicamera2()
    pixels = np.ones((48, 64, 3), dtype=np.uint8)
    backend.request.make_array = lambda _stream: pixels
    backend.request.get_metadata = lambda: {"SensorTimestamp": 5_000_000_090}
    backend.request.release = lambda: pixels.fill(0)
    camera = Picamera2Camera(
        (64, 48),
        pixel_format="BGR888",
        camera_factory=lambda: backend,
        clock=lambda: 100,
        sensor_clock=lambda: 5_000_000_100,
    )
    camera.open()
    frame = camera.read()
    camera.close()
    assert frame.header.timestamp_ns == 90
    assert frame.header.received_timestamp_ns == 100
    assert frame.header.timestamp_source == "sensor_boottime"
    assert frame.encoding == "rgb8"
    assert frame.image.all() and not pixels.any()
    assert backend.configuration["queue"] is False


def test_clock_mapping_reports_bracket_uncertainty():
    clock = iter([100, 110])
    assert map_sensor_timestamp(1010, sensor_clock=lambda: 1020, clock=lambda: next(clock)) == (
        95,
        5,
    )

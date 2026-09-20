import numpy as np

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

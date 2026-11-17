import sys
from types import SimpleNamespace

import numpy as np
import pytest

from roadbot.config import PreprocessingConfig
from roadbot.messages.common import MessageHeader, Vector3
from roadbot.messages.sensors import CameraFrame, ImuSample, WheelEncoderSample
from roadbot.perception.object_detector import UltralyticsDetector
from roadbot.perception.preprocessing import YoloPreprocessor
from roadbot.perception.synchronization import SynchronizedFrame


def packet(image, encoding="bgr8"):
    header = MessageHeader(123, 4, "camera")
    camera = CameraFrame(header, image, {}, encoding)
    return SynchronizedFrame(
        camera,
        ImuSample(header, Vector3(0, 0, 0), Vector3(0, 0, 0)),
        WheelEncoderSample(header, 0, 0),
        0,
        0,
        (),
        None,
        False,
    )


@pytest.mark.parametrize(
    "encoding,pixel",
    [
        ("bgr8", [0, 0, 255]),
        ("rgb8", [255, 0, 0]),
        ("bgra8", [0, 0, 255, 42]),
        ("rgba8", [255, 0, 0, 42]),
    ],
)
def test_letterbox_color_layout_normalization_and_original_ownership(encoding, pixel):
    image = np.tile(np.array(pixel, dtype=np.uint8), (48, 64, 1))
    original = image.copy()
    sensors = packet(image, encoding)
    prepared = YoloPreprocessor(PreprocessingConfig((64, 64))).prepare(sensors)
    assert prepared.tensor.shape == (1, 3, 64, 64)
    assert prepared.tensor.dtype == np.float32 and prepared.tensor.flags.c_contiguous
    np.testing.assert_array_equal(prepared.tensor[0, :, 8, 0], [1, 0, 0])
    np.testing.assert_allclose(prepared.tensor[0, :, 0, 0], 114 / 255)
    assert prepared.transform.padding == (0, 8, 0, 8)
    assert prepared.sensors is sensors
    np.testing.assert_array_equal(image, original)


def test_odd_padding_rounding_and_box_round_trip_clipping():
    prepared = YoloPreprocessor(PreprocessingConfig((64, 96))).prepare(
        packet(np.zeros((37, 91, 3), dtype=np.uint8))
    )
    geometry = prepared.transform
    assert geometry.resized_size == (39, 96)
    assert geometry.padding == (0, 12, 0, 13)
    boxes = geometry.to_camera_boxes(np.array([[0, 12, 96, 51], [-20, -20, 300, 300]]))
    np.testing.assert_allclose(boxes, [[0, 0, 91, 37], [0, 0, 91, 37]])
    no_upscale = YoloPreprocessor(PreprocessingConfig((64, 96), scale_up=False)).prepare(
        packet(np.zeros((37, 91, 3), dtype=np.uint8))
    )
    assert no_upscale.transform.resized_size == (37, 91)


@pytest.mark.parametrize(
    "image,encoding",
    [
        (np.zeros((0, 4, 3), np.uint8), "bgr8"),
        (np.zeros((4, 4, 3), np.float32), "bgr8"),
        (np.zeros((4, 4, 4), np.uint8), "bgr8"),
        (np.zeros((4, 4, 3), np.uint8), "YUV420"),
    ],
)
def test_rejects_ambiguous_or_invalid_images(image, encoding):
    with pytest.raises(ValueError):
        YoloPreprocessor().prepare(packet(image, encoding))


def test_yolo_adapter_passes_tensor_once_and_restores_source_geometry(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(from_numpy=lambda array: array))
    prepared = YoloPreprocessor(PreprocessingConfig((64, 64))).prepare(
        packet(np.zeros((48, 64, 3), dtype=np.uint8))
    )

    def tensor(array):
        return SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.array(array)))

    result = SimpleNamespace(
        boxes=SimpleNamespace(xyxy=tensor([[0, 8, 64, 56]]), conf=tensor([0.9]), cls=tensor([7])),
        names={7: "chosen target"},
    )
    calls = []

    def predict(**kwargs):
        calls.append(kwargs)
        return [result]

    detector = UltralyticsDetector(SimpleNamespace(predict=predict), classes=[7])
    output = detector.detect(prepared)
    assert calls[0]["source"] is prepared.tensor
    assert calls[0]["rect"] is False and calls[0]["classes"] == [7]
    assert output.header is prepared.sensors.header
    assert output.detections[0].xyxy == (0, 0, 64, 48)
    assert output.detections[0].label == "chosen target"
    result.boxes = None
    assert detector.detect(prepared).detections == ()


def test_weights_must_exist_locally(tmp_path):
    with pytest.raises(FileNotFoundError):
        UltralyticsDetector.from_weights(tmp_path / "missing.pt")

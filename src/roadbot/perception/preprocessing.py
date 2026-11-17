"""Fixed-shape YOLO input without importing a model or ML runtime."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roadbot.config import PreprocessingConfig
from roadbot.perception.synchronization import SynchronizedFrame


@dataclass(frozen=True, slots=True)
class LetterboxTransform:
    original_size: tuple[int, int]  # height, width
    input_size: tuple[int, int]
    resized_size: tuple[int, int]
    padding: tuple[int, int, int, int]  # left, top, right, bottom

    def to_camera_boxes(self, xyxy: np.ndarray) -> np.ndarray:
        """Undo padding/resize and clip boxes to source-image pixel coordinates."""
        boxes = np.asarray(xyxy, dtype=np.float32).copy()
        if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
            raise ValueError("boxes must be a finite Nx4 array")
        height, width = self.original_size
        resized_height, resized_width = self.resized_size
        left, top, _, _ = self.padding
        boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - left) * width / resized_width, 0, width)
        boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - top) * height / resized_height, 0, height)
        return boxes


@dataclass(frozen=True, slots=True)
class PreparedFrame:
    sensors: SynchronizedFrame
    tensor: np.ndarray  # contiguous RGB float32, [1, 3, H, W], [0, 1]
    transform: LetterboxTransform


class YoloPreprocessor:
    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()

    def prepare(self, sensors: SynchronizedFrame) -> PreparedFrame:
        try:
            import cv2
        except ImportError as error:
            raise RuntimeError(
                "preprocessing requires OpenCV (python3-opencv on the Pi)"
            ) from error
        frame = sensors.camera
        image = np.asarray(frame.image)
        channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}
        if (
            image.dtype != np.uint8
            or image.ndim != 3
            or image.size == 0
            or image.shape[2] != channels.get(frame.encoding)
        ):
            raise ValueError(
                "expected nonempty uint8 RGB/BGR/RGBA/BGRA image with matching encoding"
            )
        rgb = image[:, :, :3]
        if frame.encoding in ("bgr8", "bgra8"):
            rgb = rgb[:, :, ::-1]
        height, width = image.shape[:2]
        target_height, target_width = self.config.input_size
        ratio = min(target_height / height, target_width / width)
        if not self.config.scale_up:
            ratio = min(ratio, 1.0)
        resized_width, resized_height = max(1, round(width * ratio)), max(1, round(height * ratio))
        if (resized_height, resized_width) != (height, width):
            rgb = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        left = (target_width - resized_width) // 2
        top = (target_height - resized_height) // 2
        right, bottom = target_width - resized_width - left, target_height - resized_height - top
        padded = np.full(
            (target_height, target_width, 3), self.config.padding_value, dtype=np.uint8
        )
        padded[top : top + resized_height, left : left + resized_width] = rgb
        tensor = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)
        tensor /= 255.0
        return PreparedFrame(
            sensors,
            tensor,
            LetterboxTransform(
                (height, width),
                self.config.input_size,
                (resized_height, resized_width),
                (left, top, right, bottom),
            ),
        )

"""Optional model adapter; target classes and weights are chosen by the caller."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from roadbot.messages.perception import ObjectDetection, ObjectDetections
from roadbot.perception.preprocessing import PreparedFrame


class ObjectDetector(Protocol):
    def detect(self, frame: PreparedFrame) -> ObjectDetections: ...


class UltralyticsDetector:
    """Feed prepared RGB tensors to YOLO26 without applying preprocessing twice.

    Accepts an already loaded Ultralytics model to keep dependency/device/model
    selection outside the intake pipeline. Tensor results use model-input pixel
    coordinates; this adapter maps them back to original camera pixels.
    """

    def __init__(
        self,
        model: Any,
        *,
        confidence: float = 0.25,
        classes: list[int] | None = None,
        device: str = "cpu",
    ) -> None:
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        self.model, self.confidence, self.classes, self.device = model, confidence, classes, device

    @classmethod
    def from_weights(cls, path: Path, **kwargs: Any) -> UltralyticsDetector:
        if not path.is_file():
            raise FileNotFoundError(f"provide local YOLO26 weights: {path}")
        from ultralytics import YOLO

        return cls(YOLO(str(path), task="detect"), **kwargs)

    def detect(self, frame: PreparedFrame) -> ObjectDetections:
        import torch

        result = self.model.predict(
            source=torch.from_numpy(frame.tensor),
            imgsz=frame.transform.input_size,
            rect=False,
            conf=self.confidence,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )[0]
        detections = []
        if result.boxes is not None:
            boxes = frame.transform.to_camera_boxes(result.boxes.xyxy.cpu().numpy())
            scores = result.boxes.conf.cpu().numpy()
            ids = result.boxes.cls.cpu().numpy()
            for box, score, class_id in zip(boxes, scores, ids):
                detections.append(
                    ObjectDetection(
                        int(class_id),
                        result.names[int(class_id)],
                        float(score),
                        tuple(float(v) for v in box),
                    )
                )
        return ObjectDetections(
            frame.sensors.header, frame.sensors.camera.frame_id, tuple(detections)
        )

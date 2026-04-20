from __future__ import annotations
import time
import yaml
from typing import Any, Dict, List, Optional
from ultralytics import YOLO


class YOLOWorldDetector:
    def __init__(
        self,
        model_path: str = "yolov8s-world.pt",
        conf_threshold: float = 0.25,
    ) -> None:
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self._current_vocab: List[str] = []

    def set_vocabulary(self, categories: List[str]) -> float:
        """
        Set text vocabulary and return encoding latency in ms.
        This is the key instrumented method for VocabDriftMOT.
        """
        t0 = time.perf_counter()
        self.model.set_classes(categories)
        latency_ms = (time.perf_counter() - t0) * 1000
        self._current_vocab = categories
        return latency_ms

    def detect(self, frame: Any) -> tuple[List[Dict[str, Any]], float]:
        """
        Run detection on frame.
        Returns (detections, inference_latency_ms)
        """
        t0 = time.perf_counter()
        results = self.model.track(
            frame,
            verbose=False,
            conf=self.conf_threshold,
            persist=True,
            tracker="bytetrack.yaml",
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        tracked: List[Dict[str, Any]] = []
        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue
            for box, track_id in zip(result.boxes, result.boxes.id):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                tracked.append({
                    "track_id": int(track_id.item()),
                    "bbox": [x1, y1, x2, y2],
                    "class_id": int(box.cls[0].item()),
                    "confidence": float(box.conf[0].item()),
                    "label": self._current_vocab[int(box.cls[0].item())]
                    if int(box.cls[0].item()) < len(self._current_vocab)
                    else "unknown",
                })
        return tracked, latency_ms

    @property
    def current_vocab(self) -> List[str]:
        return self._current_vocab
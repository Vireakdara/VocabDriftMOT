from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np
import torch
from ultralytics import YOLO


class YOLOWorldDetector:
    def __init__(
        self,
        model_path: str = "yolov8s-world.pt",
        conf_threshold: float = 0.25,
    ) -> None:
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model = YOLO(model_path)
        self.model.to(self.device)

        self.conf_threshold = conf_threshold
        self._current_vocab: List[str] = []

        # Warmup text encoder
        self.set_vocabulary(["person"])

        # Warmup detection + ByteTrack path
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.track(
            dummy,
            verbose=False,
            conf=self.conf_threshold,
            persist=True,
            tracker="bytetrack.yaml",
            device=self.device,
        )

    def set_vocabulary(self, categories: List[str]) -> float:
        t0 = time.perf_counter()

        self.model.to(self.device)
        self.model.set_classes(categories)
        self.model.to(self.device)

        latency_ms = (time.perf_counter() - t0) * 1000
        self._current_vocab = categories
        return latency_ms

    def detect(self, frame: Any) -> tuple[List[Dict[str, Any]], float]:
        t0 = time.perf_counter()

        results = self.model.track(
            frame,
            verbose=False,
            conf=self.conf_threshold,
            persist=True,
            tracker="bytetrack.yaml",
            device=self.device,
        )

        latency_ms = (time.perf_counter() - t0) * 1000

        tracked: List[Dict[str, Any]] = []

        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue

            for box, track_id in zip(result.boxes, result.boxes.id):
                class_id = int(box.cls[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                tracked.append(
                    {
                        "track_id": int(track_id.item()),
                        "bbox": [x1, y1, x2, y2],
                        "class_id": class_id,
                        "confidence": float(box.conf[0].item()),
                        "label": self._current_vocab[class_id]
                        if class_id < len(self._current_vocab)
                        else "unknown",
                    }
                )

        return tracked, latency_ms

    @property
    def current_vocab(self) -> List[str]:
        return self._current_vocab
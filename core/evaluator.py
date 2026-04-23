from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute IoU between two boxes in x1y1x2y2 format.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


class IDFTracker:
    """
    Tracks identity matches between predicted and ground truth tracks
    to compute IDF1 and IDF1-AT (IDF1 Across Transition).

    IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
    where:
        IDTP = correctly matched ID pairs across frames
        IDFP = predicted IDs with no GT match
        IDFN = GT IDs with no predicted match
    """

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = iou_threshold
        self.idtp = 0
        self.idfp = 0
        self.idfn = 0

        # For IDF1-AT: track per-transition window
        self.transition_windows: Dict[str, Dict] = {}
        self.current_transition: str = None
        self.window_idtp = 0
        self.window_idfp = 0
        self.window_idfn = 0

        # Track ID mapping: pred_id -> gt_id
        self.id_map: Dict[int, int] = {}

    def open_transition_window(
        self, transition_type: str, frame_idx: int
    ) -> None:
        """Start measuring a transition window."""
        self.current_transition = f"{transition_type}@{frame_idx}"
        self.window_idtp = 0
        self.window_idfp = 0
        self.window_idfn = 0

    def close_transition_window(self) -> None:
        """Close current window and store results."""
        if self.current_transition is None:
            return
        self.transition_windows[self.current_transition] = {
            "idtp": self.window_idtp,
            "idfp": self.window_idfp,
            "idfn": self.window_idfn,
            "idf1": self._idf1(
                self.window_idtp, self.window_idfp, self.window_idfn
            ),
        }
        self.current_transition = None

    def update(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
    ) -> None:
        """
        Match predictions to GT for one frame and update counts.
        predictions: list of {track_id, bbox, ...}
        ground_truth: list of {track_id, bbox}
        """
        matched_gt = set()
        matched_pred = set()

        # Match predictions to GT by IoU
        for pred in predictions:
            best_iou = self.iou_threshold
            best_gt = None
            for gt in ground_truth:
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt

            if best_gt is not None:
                # Check ID consistency
                pred_id = pred["track_id"]
                gt_id = best_gt["track_id"]

                if pred_id not in self.id_map:
                    self.id_map[pred_id] = gt_id

                if self.id_map[pred_id] == gt_id:
                    self.idtp += 1
                    if self.current_transition:
                        self.window_idtp += 1
                else:
                    # ID switch — pred matched to wrong GT identity
                    self.idfp += 1
                    self.idfn += 1
                    if self.current_transition:
                        self.window_idfp += 1
                        self.window_idfn += 1
                    self.id_map[pred_id] = gt_id

                matched_gt.add(id(best_gt))
                matched_pred.add(id(pred))

        # Unmatched predictions = false positives
        unmatched_pred = len(predictions) - len(matched_pred)
        self.idfp += unmatched_pred
        if self.current_transition:
            self.window_idfp += unmatched_pred

        # Unmatched GT = false negatives
        unmatched_gt = len(ground_truth) - len(matched_gt)
        self.idfn += unmatched_gt
        if self.current_transition:
            self.window_idfn += unmatched_gt

    def idf1(self) -> float:
        """Overall IDF1 for the full sequence."""
        return self._idf1(self.idtp, self.idfp, self.idfn)

    def idf1_at(self) -> Dict[str, float]:
        """IDF1-AT per transition window."""
        return {
            k: v["idf1"] for k, v in self.transition_windows.items()
        }

    def _idf1(self, idtp: int, idfp: int, idfn: int) -> float:
        denom = 2 * idtp + idfp + idfn
        return (2 * idtp / denom) if denom > 0 else 0.0


class ScoreJitterTracker:
    """
    Tracks SJS (Score Jitter under Synonym).
    Measures confidence score variance around transition frames.
    """

    def __init__(self, window_size: int = 10) -> None:
        self.window_size = window_size
        self.pre_scores: List[float] = []
        self.post_scores: List[float] = []
        self.recording_pre = False
        self.recording_post = False
        self.results: Dict[str, float] = {}
        self.current_key: str = None

    def start_pre_window(self, key: str) -> None:
        self.current_key = key
        self.pre_scores = []
        self.recording_pre = True
        self.recording_post = False

    def start_post_window(self) -> None:
        self.post_scores = []
        self.recording_pre = False
        self.recording_post = True

    def update(self, detections: List[Dict]) -> None:
        if not detections:
            return
        avg_conf = np.mean([d["confidence"] for d in detections])
        if self.recording_pre:
            self.pre_scores.append(avg_conf)
        elif self.recording_post:
            self.post_scores.append(avg_conf)

    def close_window(self) -> None:
        if not self.current_key:
            return
        pre_mean = np.mean(self.pre_scores) if self.pre_scores else 0.0
        post_mean = np.mean(self.post_scores) if self.post_scores else 0.0
        sjs = abs(float(pre_mean) - float(post_mean))
        self.results[self.current_key] = {
            "pre_mean_conf": round(float(pre_mean), 4),
            "post_mean_conf": round(float(post_mean), 4),
            "sjs": round(sjs, 4),
        }
        self.recording_post = False
        self.current_key = None

    def sjs_results(self) -> Dict:
        return self.results

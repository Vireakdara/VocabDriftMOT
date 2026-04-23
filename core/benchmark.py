from __future__ import annotations
import cv2
import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional

class MOTSequence:
    """
    Load a MOT17 sequence from disk.
    Provides frames and ground truch track data.
    """

    def __init__(self, sequence_dir: str) -> None:
        self.sequence_dir = Path(sequence_dir)
        self.img_dir = self.sequence_dir / "img1"
        self.gt_file = self.sequence_dir / "gt" / "gt.txt"

        # Get sorted frame paths
        self.frame_paths = sorted(self.img_dir.glob("*.jpg"))
        self.total_frames = len(self.frame_paths)

        # Load ground truth
        self.ground_truth = self._load_ground_truth()

        print(f"Loaded sequence: {self.sequence_dir.name}")
        print(f"  Total frames: {self.total_frames}")
        print(f"  GT tracks: {len(set(v['track_id'] for vs in self.ground_truth.values() for v in vs))}")

    def _load_ground_truth(self) -> Dict[int, List[Dict]]:
        """
        Load gt.txt into a dict keyed by frame index.
            gt.txt format: frame, id, x, y, w, h, active, class, visibility
            We only keep active (conf=1) pedestrian (class=1) objects.
        """
        gt: Dict[int, List[Dict]] = {}
        if not self.gt_file.exists():
            print(f"Warning: no gt.txt at {self.gt_file}")
            return gt

        with open(self.gt_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 7:
                    continue
                frame = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h= float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                active = int(parts[6])
                obj_class = int(parts[7]) if len(parts) > 7 else 1

                #Only keep active pedestrians
                if active != 1 or obj_class != 1:
                    continue

                if frame not in gt:
                    gt[frame] = []
                gt[frame].append({
                    "track_id": track_id,
                    "bbox": [x, y, x+w, y+h],
                })
        return gt
    
    def get_frame(self, frame_idx: int) -> Optional[Any]:
        """Load frame by index (0-based)."""
        if frame_idx < 0 or frame_idx >= self.total_frames:
            return None
        return cv2.imread(str(self.frame_paths[frame_idx]))
    
    def get_gt_for_frame(self, frame_idx:int) -> List[Dict]:
        """
        Get ground truth for frame (1-based in gt.txt, 0-based here).
        """
        return self.ground_truth.get(frame_idx + 1, [])

class TransitionSchedule:
    """
    Defines when vocabulary transitions happen in a sequence.
    """

    def __init__(self, transitions: List[Dict]) -> None:
        self.transitions = transitions
        self.transition_map = {t["frame"]: t for t in transitions}

    @classmethod
    def from_json(cls, path: str) -> "TransitionSchedule":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(data["transitions"])

    @classmethod
    def build(
        cls,
        total_frames: int,
        initial_vocab: List[str],
        transition_type_sequence: List[str],
    ) -> "TransitionSchedule":
        vocab_map = {
            "synonym": ["pedestrian", "car", "bicycle"],
            "hypernym_expand": ["person", "vehicle"],
            "hyponym_narrow": ["man", "woman", "child"],
            "sibling_swap": ["cyclist", "car", "bus"],
            "disjoint": ["airplane", "dog", "cat"],
        }

        n = len(transition_type_sequence)
        start = int(total_frames * 0.2)
        spacing = int((total_frames * 0.7) / max(n, 1))

        transitions = []
        for i, t_type in enumerate(transition_type_sequence):
            transitions.append({
                "frame": start + i * spacing,
                "vocab": vocab_map[t_type],
                "transition_type": t_type,
            })

        return cls(transitions)

    def get_transition_at(self, frame_idx: int) -> Optional[Dict]:
        return self.transition_map.get(frame_idx)

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({"transitions": self.transitions}, f, indent=2)
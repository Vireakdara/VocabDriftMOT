from __future__ import annotations
import cv2
import yaml
import json
from pathlib import Path
from core.detector import YOLOWorldDetector


def load_config(path: str = "configs/base.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    # Setup
    model_path = config["model"]["path"]
    conf_threshold = config["model"]["conf_threshold"]
    video_path = config["video"]["path"]
    initial_vocab = config["vocabulary"]["initial"]
    transitions = config["transitions"]
    log_file = config["logging"]["log_file"]

    Path("outputs").mkdir(exist_ok=True)

    # Init detector
    detector = YOLOWorldDetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
    )

    print("Detector initialized with internal warmup vocabulary: ['person']")

    # Set initial vocabulary
    encode_latency = detector.set_vocabulary(initial_vocab)
    print(f"Initial vocab set: {initial_vocab}")
    print(f"Encode latency: {encode_latency:.2f} ms")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open video at {video_path}")
        return

    frame_idx = 0
    log_records = []

    # Build transition lookup: frame_idx -> transition
    transition_map = {t["frame"]: t for t in transitions}

    print("Starting tracker. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if vocab transition fires this frame
        encode_latency_ms = 0.0
        transition_event = None

        if frame_idx in transition_map:
            t = transition_map[frame_idx]
            encode_latency_ms = detector.set_vocabulary(t["vocab"])
            transition_event = t["transition_type"]
            print(f"[Frame {frame_idx}] Transition: {transition_event}")
            print(f"  New vocab: {t['vocab']}")
            print(f"  Encode latency: {encode_latency_ms:.2f} ms")

        # Detect and track
        detections, infer_latency_ms = detector.detect(frame)

        # Log record
        record = {
            "frame": frame_idx,
            "vocab": detector.current_vocab,
            "transition_type": transition_event,
            "encode_latency_ms": round(encode_latency_ms, 3),
            "infer_latency_ms": round(infer_latency_ms, 3),
            "detection_count": len(detections),
            "track_ids": [d["track_id"] for d in detections],
        }
        log_records.append(record)

        # Draw boxes
        for obj in detections:
            x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
            label = f"ID:{obj['track_id']} {obj['label']} {obj['confidence']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Show frame index and current vocab on frame
        vocab_preview = detector.current_vocab[:3]
        vocab_suffix = "..." if len(detector.current_vocab) > 3 else ""
        cv2.putText(
            frame,
            f"Frame: {frame_idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"Vocab: {vocab_preview}{vocab_suffix}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )

        cv2.imshow("VocabDriftMOT", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # Save logs
    with open(log_file, "w") as f:
        for record in log_records:
            f.write(json.dumps(record) + "\n")

    print(f"\nDone. Processed {frame_idx} frames.")
    print(f"Log saved to {log_file}")


if __name__ == "__main__":
    main()
from __future__ import annotations
import cv2
import json
import yaml
import time
from pathlib import Path
from core.detector import YOLOWorldDetector
from core.benchmark import MOTSequence, TransitionSchedule
from core.evaluator import IDFTracker, ScoreJitterTracker


def load_config(path: str = "configs/base.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_sequence(
    sequence_dir: str,
    initial_vocab: list,
    transition_types: list,
    model_path: str = "yolov8s-world.pt",
    conf_threshold: float = 0.25,
    output_dir: str = "outputs",
    visualize: bool = True,
) -> dict:

    Path(output_dir).mkdir(exist_ok=True)
    sequence_name = Path(sequence_dir).name

    # Load sequence
    sequence = MOTSequence(sequence_dir)

    # Build transition schedule
    schedule = TransitionSchedule.build(
        total_frames=sequence.total_frames,
        initial_vocab=initial_vocab,
        transition_type_sequence=transition_types,
    )

    # Save transition schedule
    schedule_path = f"{output_dir}/{sequence_name}_schedule.json"
    schedule.to_json(schedule_path)
    print(f"Transition schedule saved: {schedule_path}")

    # Init detector
    detector = YOLOWorldDetector(
        model_path=model_path,
        conf_threshold=conf_threshold,
    )
    encode_latency = detector.set_vocabulary(initial_vocab)
    print(f"Initial vocab: {initial_vocab}")
    print(f"Initial encode latency: {encode_latency:.2f} ms")

    # Init evaluators
    idf_tracker = IDFTracker(iou_threshold=0.5)
    sjs_tracker = ScoreJitterTracker(window_size=10)

    # Logging
    log_records = []
    transition_log = []

    # SJS pre-window starts immediately
    sjs_tracker.start_pre_window("pre_first_transition")

    print(f"\nRunning benchmark on {sequence_name}...")
    print(f"Total frames: {sequence.total_frames}")
    print(f"Transitions: {len(schedule.transitions)}")
    print("-" * 50)

    TRANSITION_WINDOW_SIZE = 30

    for frame_idx in range(sequence.total_frames):

        frame = sequence.get_frame(frame_idx)
        if frame is None:
            break

        gt = sequence.get_gt_for_frame(frame_idx)

        # Check transition
        transition = schedule.get_transition_at(frame_idx)
        encode_latency_ms = 0.0
        transition_type = None

        if transition is not None:
            # Close previous SJS window
            sjs_tracker.close_window()

            # Switch vocabulary
            encode_latency_ms = detector.set_vocabulary(transition["vocab"])
            transition_type = transition["transition_type"]

            # Open IDF1-AT window
            idf_tracker.open_transition_window(transition_type, frame_idx)

            # Open SJS post window
            sjs_tracker.start_post_window()

            print(f"[Frame {frame_idx}] {transition_type}")
            print(f"  Vocab: {transition['vocab']}")
            print(f"  Encode latency: {encode_latency_ms:.2f} ms")

            transition_log.append({
                "frame": frame_idx,
                "transition_type": transition_type,
                "vocab": transition["vocab"],
                "encode_latency_ms": round(encode_latency_ms, 3),
            })

        # Close IDF1-AT window after TRANSITION_WINDOW_SIZE frames
        for t in schedule.transitions:
            if frame_idx == t["frame"] + TRANSITION_WINDOW_SIZE:
                idf_tracker.close_transition_window()
                # Start new SJS pre window for next transition
                sjs_tracker.start_pre_window(
                    f"pre_{frame_idx}"
                )

        # Detect and track
        t0 = time.perf_counter()
        detections, infer_latency_ms = detector.detect(frame)
        
        # Update evaluators
        idf_tracker.update(detections, gt)
        sjs_tracker.update(detections)

        # Log
        record = {
            "frame": frame_idx,
            "vocab": detector.current_vocab,
            "transition_type": transition_type,
            "encode_latency_ms": round(encode_latency_ms, 3),
            "infer_latency_ms": round(infer_latency_ms, 3),
            "detection_count": len(detections),
            "gt_count": len(gt),
            "track_ids": [d["track_id"] for d in detections],
            "gt_ids": [g["track_id"] for g in gt],
        }
        log_records.append(record)

        # Visualize
        if visualize:
            for obj in detections:
                x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
                label = f"ID:{obj['track_id']} {obj['label']} {obj['confidence']:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            for g in gt:
                x1, y1, x2, y2 = [int(v) for v in g["bbox"]]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

            cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Vocab: {detector.current_vocab[:2]}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)
            cv2.putText(frame, f"IDF1: {idf_tracker.idf1():.3f}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1)

            cv2.imshow("VocabDriftMOT Benchmark", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()

    # Close any open windows
    idf_tracker.close_transition_window()
    sjs_tracker.close_window()

    # Compile results
    results = {
        "sequence": sequence_name,
        "total_frames": sequence.total_frames,
        "transitions": transition_log,
        "metrics": {
            "idf1_overall": round(idf_tracker.idf1(), 4),
            "idf1_at": idf_tracker.idf1_at(),
            "sjs": sjs_tracker.sjs_results(),
        },
    }

    # Save results
    results_path = f"{output_dir}/{sequence_name}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    log_path = f"{output_dir}/{sequence_name}_framelog.jsonl"
    with open(log_path, "w") as f:
        for record in log_records:
            f.write(json.dumps(record) + "\n")

    print("\n" + "=" * 50)
    print(f"RESULTS: {sequence_name}")
    print(f"  IDF1 overall:  {results['metrics']['idf1_overall']:.4f}")
    print(f"  IDF1-AT per transition:")
    for k, v in results["metrics"]["idf1_at"].items():
        print(f"    {k}: {v:.4f}")
    print(f"  SJS per transition:")
    for k, v in results["metrics"]["sjs"].items():
        print(f"    {k}: {v}")
    print(f"\nResults saved: {results_path}")
    print(f"Frame log saved: {log_path}")

    return results


def main():
    config = load_config()

    sequences = [
        "E:\\Dataset\\MOT17\\train\\MOT17-02-FRCNN",
        "E:\\Dataset\\MOT17\\train\\MOT17-09-FRCNN",
        "E:\\Dataset\\MOT17\\train\\MOT17-13-FRCNN",
    ]

    initial_vocab = ["person", "car", "bicycle", "motorcycle", "bus", "truck"]
    transition_types = [
        "synonym",
        "hypernym_expand",
        "hyponym_narrow",
        "sibling_swap",
        "disjoint",
    ]

    all_results = []
    for seq_dir in sequences:
        results = run_sequence(
            sequence_dir=seq_dir,
            initial_vocab=initial_vocab,
            transition_types=transition_types,
            model_path=config["model"]["path"],
            conf_threshold=config["model"]["conf_threshold"],
            output_dir="outputs",
            visualize=True,
        )
        all_results.append(results)

    # Save combined results
    with open("outputs/benchmark_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nAll results saved: outputs/benchmark_results.json")


if __name__ == "__main__":
    main()
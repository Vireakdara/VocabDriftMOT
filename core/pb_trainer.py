from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
from core.prompt_bridge_v2 import PromptBridgeV2


def load_benchmark_data(output_dir: str = "outputs") -> list:
    """
    Load real IDF1-AT measurements from benchmark results.
    Returns list of (old_vocab, new_vocab, transition_type, idf1_at) tuples.
    """
    samples = []
    for f in Path(output_dir).glob("*_results.json"):
        if f.name == "benchmark_results.json":
            continue
        with open(f) as fp:
            result = json.load(fp)

        # Get transition schedule
        schedule_file = f.parent / f.name.replace("_results.json", "_schedule.json")
        if not schedule_file.exists():
            continue
        with open(schedule_file) as fp:
            schedule = json.load(fp)

        # Map frame -> vocab
        frame_to_vocab = {t["frame"]: t["vocab"] for t in schedule["transitions"]}
        frame_to_type = {t["frame"]: t["transition_type"] for t in schedule["transitions"]}

        # Initial vocab from config
        initial_vocab = ["person", "car", "bicycle", "motorcycle", "bus", "truck"]

        # Build old/new vocab pairs with IDF1-AT labels
        sorted_frames = sorted(frame_to_vocab.keys())
        for i, frame in enumerate(sorted_frames):
            key = f"{frame_to_type[frame]}@{frame}"
            if key not in result["metrics"]["idf1_at"]:
                continue
            idf1_at = result["metrics"]["idf1_at"][key]

            old_vocab = (
                frame_to_vocab[sorted_frames[i - 1]]
                if i > 0
                else initial_vocab
            )
            new_vocab = frame_to_vocab[frame]
            t_type = frame_to_type[frame]

            samples.append({
                "old_vocab": old_vocab,
                "new_vocab": new_vocab,
                "transition_type": t_type,
                "idf1_at": idf1_at,
            })

    print(f"Loaded {len(samples)} real transition samples")
    return samples


def fit_parameters(
    samples: list,
    pb: PromptBridgeV2,
    output_dir: str = "outputs",
) -> dict:
    """
    Fit w1, w2, b to minimize MSE between gate predictions and real IDF1-AT.
    """
    print("Computing CLIP signals for all samples...")
    signals = []
    labels = []

    for s in samples:
        old_emb = pb.encode(s["old_vocab"])
        new_emb = pb.encode(s["new_vocab"])
        cosine_sim, drift_magnitude = pb.compute_signals(old_emb, new_emb)
        signals.append([cosine_sim, drift_magnitude])
        labels.append(s["idf1_at"])

    signals = np.array(signals)
    labels = np.array(labels)

    print(f"Fitting parameters on {len(labels)} samples...")
    print(f"Label distribution: min={labels.min():.4f} max={labels.max():.4f} "
          f"mean={labels.mean():.4f}")

    def loss(params):
        w1, w2, b = params
        logits = w1 * signals[:, 0] + w2 * signals[:, 1] + b
        preds = 1 / (1 + np.exp(-logits))
        # Weight positive samples more — they are underrepresented
        weights = np.where(labels > 0.1, 5.0, 1.0)
        return np.mean(weights * (preds - labels) ** 2)

    # Fit
    result = minimize(
        loss,
        x0=[5.0, -3.0, 0.0],
        method="Nelder-Mead",
        options={"maxiter": 10000, "xatol": 1e-6, "fatol": 1e-6},
    )

    w1, w2, b = result.x
    final_loss = result.fun

    # Compute predictions with fitted params
    logits = w1 * signals[:, 0] + w2 * signals[:, 1] + b
    preds = 1 / (1 + np.exp(-logits))

    # Correlation
    correlation = float(np.corrcoef(preds, labels)[0, 1])

    print(f"\nFitted parameters:")
    print(f"  w1 (cosine weight):   {w1:.4f}")
    print(f"  w2 (drift weight):    {w2:.4f}")
    print(f"  b  (bias):            {b:.4f}")
    print(f"  MSE loss:             {final_loss:.6f}")
    print(f"  Pred-label corr:      {correlation:.4f}")

    # Per transition type analysis
    print(f"\nPer transition type:")
    t_types = list(set(s["transition_type"] for s in samples))
    for t in sorted(t_types):
        idxs = [i for i, s in enumerate(samples) if s["transition_type"] == t]
        t_labels = labels[idxs]
        t_preds = preds[idxs]
        print(f"  {t:<20} "
              f"true={t_labels.mean():.4f} "
              f"pred={t_preds.mean():.4f} "
              f"err={abs(t_preds.mean()-t_labels.mean()):.4f}")

    fitted = {
        "w1": float(w1),
        "w2": float(w2),
        "b": float(b),
        "mse": float(final_loss),
        "correlation": correlation,
        "n_samples": len(samples),
    }

    out_path = f"{output_dir}/prompt_bridge_v2_params.json"
    with open(out_path, "w") as f:
        json.dump(fitted, f, indent=2)
    print(f"\nParameters saved: {out_path}")

    return fitted


def main():
    pb = PromptBridgeV2(device="cuda")
    samples = load_benchmark_data("outputs")

    if not samples:
        print("No benchmark data found. Run run_benchmark.py first.")
        return

    fitted = fit_parameters(samples, pb, "outputs")
    print(f"\nFinal: w1={fitted['w1']:.4f} w2={fitted['w2']:.4f} "
          f"b={fitted['b']:.4f} corr={fitted['correlation']:.4f}")


if __name__ == "__main__":
    main()
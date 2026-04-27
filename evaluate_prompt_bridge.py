from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from core.prompt_bridge_v2 import PromptBridgeV2
from core.pb_trainer import load_benchmark_data, fit_parameters


def evaluate(
    output_dir: str = "outputs",
    device: str = "cuda",
) -> None:

    # Load fitted parameters
    params_path = Path(output_dir) / "prompt_bridge_v2_params.json"
    if not params_path.exists():
        print("No fitted parameters found. Running trainer first...")
        pb = PromptBridgeV2(device=device)
        samples = load_benchmark_data(output_dir)
        fitted = fit_parameters(samples, pb, output_dir)
    else:
        with open(params_path) as f:
            fitted = json.load(f)
        pb = PromptBridgeV2(
            device=device,
            w1=fitted["w1"],
            w2=fitted["w2"],
            b=fitted["b"],
        )

    print("=" * 60)
    print("PROMPTBRIDGE V2 — EVALUATION REPORT")
    print("=" * 60)
    print(f"Fitted parameters: w1={fitted['w1']:.4f} "
          f"w2={fitted['w2']:.4f} b={fitted['b']:.4f}")
    print(f"Training MSE: {fitted['mse']:.6f}")
    print(f"Training correlation: {fitted['correlation']:.4f}")
    print()

    # Evaluate on all 5 transition types
    transition_pairs = [
        ("synonym",         ["person"],       ["pedestrian"]),
        ("synonym",         ["car"],          ["automobile"]),
        ("hypernym_expand", ["car", "bus"],   ["vehicle"]),
        ("hypernym_expand", ["person"],       ["human"]),
        ("hyponym_narrow",  ["person"],       ["man", "woman"]),
        ("hyponym_narrow",  ["vehicle"],      ["car", "truck"]),
        ("sibling_swap",    ["person"],       ["bicycle"]),
        ("sibling_swap",    ["car"],          ["motorcycle"]),
        ("disjoint",        ["person"],       ["airplane"]),
        ("disjoint",        ["car"],          ["dog"]),
    ]

    print("GATE DECISIONS PER TRANSITION TYPE")
    print("-" * 60)
    print(f"{'Transition':<20} {'Old':<15} {'New':<15} "
          f"{'Cosine':>7} {'Drift':>7} {'Gate':>7} {'Decision'}")
    print("-" * 60)

    results_by_type = {}
    for t_type, old_vocab, new_vocab in transition_pairs:
        gate_value, info = pb.gate(old_vocab, new_vocab)
        print(f"{t_type:<20} {str(old_vocab):<15} {str(new_vocab):<15} "
              f"{info['cosine_sim']:>7.4f} "
              f"{info['drift_magnitude']:>7.4f} "
              f"{info['gate_value']:>7.4f} "
              f"{info['decision'].upper()}")

        if t_type not in results_by_type:
            results_by_type[t_type] = []
        results_by_type[t_type].append({
            "gate_value": gate_value,
            "decision": info["decision"],
        })

    print()
    print("SUMMARY BY TRANSITION TYPE")
    print("-" * 60)

    ORDER = ["synonym", "hypernym_expand", "hyponym_narrow",
             "sibling_swap", "disjoint"]
    EXPECTED = {
        "synonym": "preserve",
        "hypernym_expand": "preserve",
        "hyponym_narrow": "preserve",
        "sibling_swap": "terminate",
        "disjoint": "terminate",
    }
    BASELINE_IDF1AT = {
        "synonym": 0.008,
        "hypernym_expand": 0.560,
        "hyponym_narrow": 0.082,
        "sibling_swap": 0.000,
        "disjoint": 0.000,
    }

    for t_type in ORDER:
        if t_type not in results_by_type:
            continue
        entries = results_by_type[t_type]
        mean_gate = np.mean([e["gate_value"] for e in entries])
        correct = sum(
            1 for e in entries
            if e["decision"] == EXPECTED[t_type]
        )
        accuracy = correct / len(entries)
        baseline = BASELINE_IDF1AT[t_type]

        print(f"  {t_type:<20} "
              f"mean_gate={mean_gate:.4f} "
              f"correct={correct}/{len(entries)} "
              f"({accuracy:.0%}) "
              f"baseline_IDF1-AT={baseline:.3f}")

    print()
    print("INTERPRETATION")
    print("-" * 60)
    print("  PromptBridge V2 uses fitted geometric signals from")
    print("  real IDF1-AT measurements to gate track preservation.")
    print()
    print("  Expected improvements over naive re-encoding baseline:")
    print("  - Synonym: gate→preserve prevents premature track loss")
    print("  - Hypernym: gate→preserve maintains continuity")
    print("  - Sibling/Disjoint: gate→terminate correct behavior")
    print()

    # Save evaluation results
    eval_results = {
        "fitted_params": fitted,
        "transition_evaluation": {
            t: {
                "mean_gate": float(np.mean([e["gate_value"] for e in entries])),
                "correct_decisions": sum(
                    1 for e in entries if e["decision"] == EXPECTED[t]
                ),
                "total": len(entries),
                "baseline_idf1at": BASELINE_IDF1AT[t],
            }
            for t, entries in results_by_type.items()
        }
    }

    out_path = f"{output_dir}/prompt_bridge_v2_eval.json"
    with open(out_path, "w") as f:
        json.dump(eval_results, f, indent=2)
    print(f"Evaluation saved: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate(output_dir="outputs", device="cuda")
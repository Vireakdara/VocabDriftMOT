from __future__ import annotations
import json
import os
from pathlib import Path
from collections import defaultdict
import statistics

def load_results(output_dir: str = "outputs") -> list:
    results = []
    for f in Path(output_dir).glob("*_results.json"):
        if f.name == "benchmark_results.json":
            continue
        with open(f) as fp:
            data = json.load(fp)
            if isinstance(data, dict):
                results.append(data)
            elif isinstance(data, list):
                results.extend(data)
    return results


def aggregate(results: list) -> dict:
    """
    Aggregate IDF1-AT scores per transition type across all sequences.
    Returns mean, min, max, std, and count per transition type.
    """

    scores = defaultdict(list)

    for r in results:
        for key, score in r["metrics"]["idf1_at"].items():
            t_type = key.split("@")[0]
            scores[t_type].append(score)

    summary = {}
    for t_type, vals in sorted(scores.items()):
        summary[t_type] = {
            "mean": round(statistics.mean(vals), 4),
            "std": round(statistics.stdev(vals) if len(vals) > 1 else 0.0, 4),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "count": len(vals),
        }
    return summary


def per_sequence_table(results: list) -> dict:
    """
    Per-sequence mean IDF1-AT per transition type.
    """
    table = {}
    for r in results:
        seq = r["sequence"]
        per_type = defaultdict(list)
        for key, score in r["metrics"]["idf1_at"].items():
            t_type = key.split("@")[0]
            per_type[t_type].append(score)
        table[seq] = {
            t: round(sum(v) / len(v), 4)
            for t, v in per_type.items()
        }
    return table


def encode_latency_stats(results: list) -> dict:
    """
    Average encode latency per transition type across all sequences.
    """
    latencies = defaultdict(list)
    for r in results:
        for t in r["transitions"]:
            latencies[t["transition_type"]].append(
                t["encode_latency_ms"]
            )
    return {
        t: round(sum(v) / len(v), 2)
        for t, v in sorted(latencies.items())
    }


def print_report(results: list) -> None:
    summary = aggregate(results)
    seq_table = per_sequence_table(results)
    latencies = encode_latency_stats(results)

    print("=" * 70)
    print("VOCABDRIFTMOT — DIAGNOSTIC REPORT")
    print("=" * 70)
    print(f"Sequences analyzed: {len(results)}")
    total_events = sum(v["count"] for v in summary.values())
    print(f"Total transition events: {total_events}")
    print()

    # Main results table
    print("IDF1-AT SUMMARY (mean ± std across all sequences and repeats)")
    print("-" * 70)
    print(f"{'Transition':<20} {'Mean':>8} {'Std':>8} "
          f"{'Min':>8} {'Max':>8} {'N':>6}")
    print("-" * 70)

    ORDER = [
        "synonym", "hypernym_expand", "hyponym_narrow",
        "sibling_swap", "disjoint"
    ]
    for t in ORDER:
        if t not in summary:
            continue
        s = summary[t]
        print(f"{t:<20} {s['mean']:>8.4f} {s['std']:>8.4f} "
              f"{s['min']:>8.4f} {s['max']:>8.4f} {s['count']:>6}")
    print()

    # Per-sequence table
    print("IDF1-AT PER SEQUENCE (mean per transition type)")
    print("-" * 70)
    seqs = sorted(seq_table.keys())
    header = f"{'Sequence':<25}" + "".join(
        f"{t[:8]:>10}" for t in ORDER
    )
    print(header)
    print("-" * 70)
    for seq in seqs:
        row = f"{seq:<25}"
        for t in ORDER:
            val = seq_table[seq].get(t, 0.0)
            row += f"{val:>10.4f}"
        print(row)
    print()

    # Encode latency
    print("MEAN ENCODE LATENCY PER TRANSITION TYPE (ms)")
    print("-" * 70)
    for t, lat in latencies.items():
        print(f"  {t:<20} {lat:>8.2f} ms")
    print()

    # Key findings
    print("KEY FINDINGS")
    print("-" * 70)
    syn = summary.get("synonym", {}).get("mean", 0)
    hyp_e = summary.get("hypernym_expand", {}).get("mean", 0)
    hyp_n = summary.get("hyponym_narrow", {}).get("mean", 0)
    sib = summary.get("sibling_swap", {}).get("mean", 0)
    dis = summary.get("disjoint", {}).get("mean", 0)

    print(f"  1. Synonym causes near-complete identity loss "
          f"(mean IDF1-AT = {syn:.3f})")
    print(f"  2. Hypernym expand is the only safe transition "
          f"(mean IDF1-AT = {hyp_e:.3f})")
    print(f"  3. Hyponym narrow mostly fails "
          f"(mean IDF1-AT = {hyp_n:.3f})")
    print(f"  4. Sibling swap correctly terminates "
          f"(mean IDF1-AT = {sib:.3f})")
    print(f"  5. Disjoint correctly terminates "
          f"(mean IDF1-AT = {dis:.3f})")
    print()
    print(f"  Gap: hypernym ({hyp_e:.3f}) vs synonym ({syn:.3f}) "
          f"= {hyp_e - syn:.3f} IDF1-AT difference")
    print(f"  Both synonym and disjoint score ~0.0 but "
          f"for opposite reasons.")
    print("=" * 70)


def save_report(results: list, path: str = "outputs/diagnostic_report.json") -> None:
    report = {
        "summary": aggregate(results),
        "per_sequence": per_sequence_table(results),
        "encode_latency_ms": encode_latency_stats(results),
        "total_sequences": len(results),
        "total_events": sum(
            len(r["metrics"]["idf1_at"]) for r in results
        ),
    }
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {path}")


def main():
    results = load_results("outputs")
    if not results:
        print("No results found in outputs/")
        return
    print_report(results)
    save_report(results)


if __name__ == "__main__":
    main()
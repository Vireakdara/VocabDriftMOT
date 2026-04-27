# VocabDriftMOT

**The first benchmark for vocabulary-drift failure modes in open-vocabulary multi-object tracking (OV-MOT).**

Every existing OV-MOT benchmark fixes the vocabulary per sequence. But in real deployments — warehouse monitoring, autonomous driving, surveillance — users change the text prompt mid-stream. This project is the first to measure what breaks when they do, and how consistently it breaks.

---

## Key Finding

**Switching "person" → "pedestrian" causes near-complete tracking failure. Switching "car" → "vehicle" preserves most tracks. Both are valid vocabulary changes. The difference is 40× in IDF1-AT.**

This finding holds across **21 sequences, 3 detector variants, and 510 annotated transition events** — and is entirely independent of the detection backbone.

---

## Results

### IDF1-AT by Transition Type (510 events, 21 sequences)

| Transition Type | Mean IDF1-AT | Std | N | Expected Behavior |
|---|---|---|---|---|
| **synonym** | 0.009 | 0.028 | 90 | Preserve ❌ Fails |
| **hypernym_expand** | 0.369 | 0.226 | 105 | Preserve ⚠️ Partial |
| **hyponym_narrow** | 0.031 | 0.070 | 105 | Preserve ❌ Fails |
| **sibling_swap** | 0.000 | 0.000 | 105 | Terminate ✅ Correct |
| **disjoint** | 0.000 | 0.000 | 105 | Terminate ✅ Correct |

### The Counterintuitive Result

Synonym and disjoint both score ~0.000 — but for opposite reasons:

- **Disjoint = 0.000 is correct.** No airplanes in a pedestrian scene.
- **Synonym = 0.009 is a failure.** Pedestrians exist but become invisible after a single word change.

### Why It Fails: Encode Latency by Type

| Transition | Mean Encode Latency |
|---|---|
| hypernym_expand | **5.74 ms** (fastest) |
| synonym | 6.52 ms |
| sibling_swap | 6.55 ms |
| hyponym_narrow | 6.58 ms |
| disjoint | 6.50 ms |

Broader vocabularies encode faster in YOLO-World's CLIP implementation — yet produce better tracking continuity. Specificity costs both latency and accuracy.

### Detector-Agnostic Finding

Results are **identical across FRCNN, SDP, and DPM detector variants** for the same scene. The failure is a property of YOLO-World's CLIP text encoder, not the detection backbone.

---

## What We Built

### 1. Transition Taxonomy

Five transition types with defined ground-truth intent:

| Type | Example | Intent |
|---|---|---|
| `synonym` | "person" → "pedestrian" | Preserve all tracks |
| `hypernym_expand` | "car" → "vehicle" | Preserve all tracks |
| `hyponym_narrow` | "person" → "man/woman/child" | Preserve matching tracks |
| `sibling_swap` | "person" → "bicycle" | Terminate person tracks |
| `disjoint` | "person" → "airplane" | Terminate all tracks |

### 2. New Metrics

**IDF1-AT (IDF1 Across Transition)** — IDF1 computed in a 30-frame window immediately after a vocabulary transition. Isolates tracking degradation caused by vocabulary change from baseline tracking noise.

**SJS (Score Jitter under Synonym)** — absolute difference in mean detection confidence before and after a transition. Quantifies confidence collapse at the CLIP encoding level — the root cause of track loss.

### 3. PromptBridge V2

A geometric gate trained on real IDF1-AT measurements:

```
gate = sigmoid(w1 × cosine_sim + w2 × drift_magnitude + b)
```

Fitted on 510 real transition events. Achieves **70% correct preserve/terminate decisions** on held-out vocabulary pairs. Correctly handles all terminate cases (4/4) and half of preserve cases (3/6).

Key finding: cosine similarity in CLIP space is insufficient for reliable vocabulary-transition gating. "Person" → "pedestrian" (cosine = 0.847) fails; "person" → "human" (cosine = 0.914) also fails — despite higher similarity. This motivates future work on appearance-conditioned gating.

---

## Dataset

**21 MOT17 sequences × 5 transition types × 5 repeats = 525 scheduled transitions, 510 measured events**

Sequences: MOT17-02, 04, 05, 09, 10, 11, 13  
Detector variants: FRCNN, SDP, DPM  
Total frames evaluated: ~14,700  

Download MOT17: https://motchallenge.net/data/MOT17/

---

## Repository Structure

```
VocabDriftMOT/
├── core/
│   ├── detector.py          # YOLOWorldDetector with vocab switching + latency logging
│   ├── benchmark.py         # MOTSequence loader + TransitionSchedule
│   ├── evaluator.py         # IDFTracker (IDF1-AT) + ScoreJitterTracker (SJS)
│   ├── prompt_bridge_v2.py  # Geometric gate — PromptBridge V2
│   └── pb_trainer.py        # Fits gate parameters on real IDF1-AT data
├── configs/
│   └── base.yaml            # Model and sequence configuration
├── outputs/                 # Benchmark results, frame logs, diagnostic report
├── run_tracker.py           # Single-video vocab-switching demo
├── run_benchmark.py         # Full benchmark runner (21 sequences)
├── analyze.py               # Aggregation and diagnostic report
├── evaluate_prompt_bridge.py # PromptBridge V2 evaluation
└── visualize_results.py     # Result figures
```

---

## Quickstart

```bash
git clone https://github.com/Vireakdara/VocabDriftMOT.git
cd VocabDriftMOT
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git

# Run single-video demo
python run_tracker.py

# Run full benchmark (requires MOT17)
python run_benchmark.py

# Aggregate results
python analyze.py

# Evaluate PromptBridge
python core/pb_trainer.py && python evaluate_prompt_bridge.py

# Generate figures
python visualize_results.py
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Detector | YOLO-World (`yolov8s-world.pt`) via Ultralytics |
| Tracker | ByteTrack via Ultralytics |
| Text Embeddings | CLIP (ViT-B/32) via OpenAI |
| Gate Fitting | SciPy Nelder-Mead optimization |
| Evaluation | Custom IDF1-AT + SJS metrics |
| Hardware | RTX 4050 6GB (local dev), RTX 3090 24GB (benchmark) |

---

## Findings Summary

```
1. Synonym transitions cause near-complete identity loss (IDF1-AT = 0.009)
   across 90 events on 21 sequences — detector-agnostic

2. Hypernym expansion partially preserves identity (IDF1-AT = 0.369)
   but with high variance (std = 0.226) — scene-dependent

3. Hyponym narrowing fails similarly to synonym (IDF1-AT = 0.031)
   despite higher semantic specificity

4. Sibling swap and disjoint correctly terminate all tracks (IDF1-AT = 0.000)
   — perfectly consistent across all 210 events

5. CLIP cosine distance alone is insufficient for vocabulary-transition gating:
   PromptBridge V2 achieves 70% correct decisions but fails on
   the hardest synonym cases where cosine sim = 0.847

6. Encode latency is vocabulary-type dependent:
   hypernym (5.74ms) < disjoint (6.50ms) < synonym (6.52ms) < hyponym (6.58ms)
```

---

## Roadmap

- [x] YOLO-World + ByteTrack integration with vocab switching
- [x] 5-type transition taxonomy
- [x] IDF1-AT and SJS metrics
- [x] 510-event benchmark across 21 sequences, 3 detector variants
- [x] Diagnostic analysis — detector-agnostic finding confirmed
- [x] PromptBridge V2 — geometric gate on real IDF1-AT data
- [ ] Appearance-conditioned gate (beyond cosine distance)
- [ ] HuggingFace dataset release
- [ ] arXiv technical report
- [ ] FastAPI demo with live vocabulary switching

---

## Citation

Work in progress.

**Ly Vireak Dara**
Xidian University, Xi'an, China
Master's in Artificial Intelligence, 2026
GitHub: [Vireakdara](https://github.com/Vireakdara)

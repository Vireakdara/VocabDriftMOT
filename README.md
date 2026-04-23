# VocabDriftMOT

The first benchmark for **vocabulary-drift failure modes in open-vocabulary 
multi-object tracking (OV-MOT)**. We study what happens when a user changes 
the text prompt mid-stream in a YOLO-World-based tracker — and show that 
semantically preserving transitions cause near-complete identity loss while 
broader category expansions preserve tracking continuity.

---

## The Problem

Every existing OV-MOT benchmark fixes the vocabulary per sequence. But in 
real deployments, users change prompts mid-stream. A warehouse system 
switches from "forklift" to "heavy machinery." An autonomous vehicle adapts 
its vocabulary based on scene type. Every time this happens, tracking breaks 
silently — no error thrown, just fragmented trajectories and inflated object 
counts.

Nobody had measured this. We did.

---

## Key Findings

We ran YOLO-World × ByteTrack across 3 MOT17 sequences with 5 transition 
types injected mid-stream. Results measured using IDF1-AT (IDF1 Across 
Transition) — our new metric that isolates tracking degradation specifically 
around vocabulary transition frames.

| Transition Type | MOT17-02 | MOT17-09 | MOT17-13 | Mean IDF1-AT |
|---|---|---|---|---|
| synonym | 0.0000 | 0.0000 | 0.1031 | **0.034** |
| hypernym_expand | 0.3490 | 0.7082 | 0.2736 | **0.444** |
| hyponym_narrow | 0.0335 | 0.0500 | 0.0000 | **0.028** |
| sibling_swap | 0.0000 | 0.0000 | 0.0000 | **0.000** |
| disjoint | 0.0000 | 0.0000 | 0.0000 | **0.000** |

### What This Means

**Synonym transitions cause near-complete identity loss (mean IDF1-AT = 
0.034).** Switching "person" → "pedestrian" — semantically identical — drops 
all tracks to zero. YOLO-World's CLIP encoder treats these words as different 
embeddings, confidence collapses, and ByteTrack loses every track.

**Hyponym narrowing is equally catastrophic (mean IDF1-AT = 0.028).** 
Switching "person" → "man/woman/child" should preserve tracks under more 
specific labels. It does not. Specificity creates uncertainty in CLIP space 
that destroys detection confidence.

**Hypernym expansion is the only safe transition (mean IDF1-AT = 0.444).** 
Broader categories maintain enough CLIP alignment to preserve detection 
confidence. Tracks survive at 70%+ fidelity on clean sequences.

**Sibling swap and disjoint correctly terminate all tracks (IDF1-AT = 
0.000).** These are correct behaviors — the system terminates tracks when 
the vocabulary is incompatible. Consistent across all sequences.

### The Counterintuitive Result

Synonym and disjoint both score near 0.000 — but for opposite reasons.
- Disjoint = 0.000 is **correct**. No airplanes in a pedestrian scene.
- Synonym = 0.034 is a **failure**. Pedestrians exist but are invisible 
  to the tracker after a word change.

---

## New Metrics

**IDF1-AT (IDF1 Across Transition)** — measures track identity preservation 
in a 30-frame window immediately following a vocabulary transition. Isolates 
degradation caused by vocabulary change from baseline tracking noise.

**SJS (Score Jitter under Synonym)** — measures detection confidence 
variance before and after a transition. Quantifies why tracks break: 
confidence collapse at the CLIP encoding level.

---

## Transition Taxonomy

| Type | Example | Expected | Actual |
|---|---|---|---|
| synonym | "person" → "pedestrian" | Preserve | Fails (0.034) |
| hypernym_expand | "car" → "vehicle" | Preserve | Works (0.444) |
| hyponym_narrow | "person" → "man/woman/child" | Preserve | Fails (0.028) |
| sibling_swap | "person" → "cyclist" | Terminate | Correct (0.000) |
| disjoint | "person" → "airplane" | Terminate | Correct (0.000) |

---

## Repository Structure

```
VocabDriftMOT/
├── core/
│   ├── detector.py        # YOLOWorldDetector with vocab switching
│   ├── benchmark.py       # MOTSequence loader + TransitionSchedule
│   └── evaluator.py       # IDFTracker (IDF1-AT) + ScoreJitterTracker (SJS)
├── configs/
│   └── base.yaml          # Model and video configuration
├── outputs/               # Benchmark results and frame logs
├── run_tracker.py         # Single video vocab-switching demo
├── run_benchmark.py       # Full benchmark runner
└── requirements.txt
```

---

## Quickstart

```bash
git clone https://github.com/Vireakdara/VocabDriftMOT.git
cd VocabDriftMOT
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run demo on single video
python run_tracker.py

# Run full benchmark (requires MOT17 dataset)
python run_benchmark.py
```

---

## Dataset

Benchmark results use MOT17 sequences:
- MOT17-02-FRCNN — 600 frames, 62 GT tracks, static camera
- MOT17-09-FRCNN — 525 frames, 26 GT tracks, static camera
- MOT17-13-FRCNN — 750 frames, 110 GT tracks, moving camera

Download MOT17: https://motchallenge.net/data/MOT17/

---

## Tech Stack

| Component | Tool |
|---|---|
| Detector | YOLO-World (yolov8s-world.pt) |
| Tracker | ByteTrack via Ultralytics |
| Text embeddings | CLIP via OpenAI |
| Evaluation | Custom IDF1-AT + SJS metrics |
| Hardware tested | RTX 4050 6GB (local), RTX 3090 24GB (server) |

---

## Roadmap

- [x] YOLO-World + ByteTrack integration
- [x] Vocabulary transition controller
- [x] MOT17 benchmark harness
- [x] IDF1-AT and SJS metrics
- [x] Full 5-transition diagnostic sweep
- [ ] PromptBridge adapter (in progress)
- [ ] HuggingFace dataset release
- [ ] Docker demo with live vocab switching

---

## Citation

Work in progress. Pre-print coming soon.

**Ly Vireak Dara**  
Xidian University, Xi'an, China  
Master's in Artificial Intelligence, 2026
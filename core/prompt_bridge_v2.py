from __future__ import annotations
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
import clip


class PromptBridgeV2:
    """
    Geometric gate for vocabulary transition handling.
    Uses cosine similarity and drift magnitude between CLIP embeddings.
    Parameters fitted on real IDF1-AT measurements from VocabDriftMOT benchmark.

    Gate = sigmoid(w1 * cosine_sim + w2 * drift_magnitude + b)
    gate > threshold → preserve tracks
    gate < threshold → terminate tracks
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B/32",
        device: str = "cuda",
        preserve_threshold: float = 0.5,
        w1: float = 2.0,
        w2: float = -1.0,
        b: float = 0.0,
    ) -> None:
        self.device = device
        self.preserve_threshold = preserve_threshold
        self.w1 = w1
        self.w2 = w2
        self.b = b

        print("Loading CLIP for PromptBridgeV2...")
        self.clip_model, _ = clip.load(clip_model_name, device=device)
        self.clip_model.eval()

    def encode(self, vocab: List[str]) -> torch.Tensor:
        with torch.no_grad():
            tokens = clip.tokenize(vocab).to(self.device)
            embs = self.clip_model.encode_text(tokens).float()
            embs = embs / embs.norm(dim=-1, keepdim=True)
            return embs.mean(dim=0)

    def compute_signals(
        self,
        old_emb: torch.Tensor,
        new_emb: torch.Tensor,
    ) -> Tuple[float, float]:
        old_norm = old_emb / old_emb.norm()
        new_norm = new_emb / new_emb.norm()
        cosine_sim = float(
            F.cosine_similarity(
                old_norm.unsqueeze(0),
                new_norm.unsqueeze(0)
            ).item()
        )
        drift_magnitude = float((new_emb - old_emb).norm().item())
        return cosine_sim, drift_magnitude

    def gate(
        self,
        old_vocab: List[str],
        new_vocab: List[str],
    ) -> Tuple[float, dict]:
        old_emb = self.encode(old_vocab)
        new_emb = self.encode(new_vocab)
        cosine_sim, drift_magnitude = self.compute_signals(old_emb, new_emb)
        logit = self.w1 * cosine_sim + self.w2 * drift_magnitude + self.b
        gate_value = float(1 / (1 + np.exp(-logit)))
        decision = "preserve" if gate_value > self.preserve_threshold else "terminate"
        info = {
            "cosine_sim": round(cosine_sim, 4),
            "drift_magnitude": round(drift_magnitude, 4),
            "logit": round(logit, 4),
            "gate_value": round(gate_value, 4),
            "decision": decision,
        }
        return gate_value, info

    def set_parameters(self, w1: float, w2: float, b: float) -> None:
        self.w1 = w1
        self.w2 = w2
        self.b = b

    def analyze_transition(
        self,
        old_vocab: List[str],
        new_vocab: List[str],
    ) -> None:
        gate_value, info = self.gate(old_vocab, new_vocab)
        print(f"\nTransition: {old_vocab} → {new_vocab}")
        print(f"  Cosine similarity:  {info['cosine_sim']:.4f}")
        print(f"  Drift magnitude:    {info['drift_magnitude']:.4f}")
        print(f"  Gate value:         {info['gate_value']:.4f}")
        print(f"  Decision:           {info['decision'].upper()}")
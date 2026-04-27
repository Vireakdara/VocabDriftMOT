from __future__ import annotations
import torch
import clip
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict


# Gate labels per transition type
# 1 = preserve identity, 0 = terminate
GATE_LABELS = {
    "synonym": 1,
    "hypernym_expand": 1,
    "hyponym_narrow": 1,
    "sibling_swap": 0,
    "disjoint": 0,
}

# Vocabulary pairs per transition type
TRANSITION_PAIRS = [
    ("synonym", ["person"], ["pedestrian"]),
    ("synonym", ["car"], ["automobile"]),
    ("synonym", ["bicycle"], ["bike"]),
    ("synonym", ["truck"], ["lorry"]),
    ("synonym", ["motorcycle"], ["motorbike"]),
    ("hypernym_expand", ["car", "bus"], ["vehicle"]),
    ("hypernym_expand", ["person", "man"], ["human"]),
    ("hypernym_expand", ["car"], ["object"]),
    ("hyponym_narrow", ["person"], ["man", "woman"]),
    ("hyponym_narrow", ["vehicle"], ["car", "truck"]),
    ("hyponym_narrow", ["person"], ["child", "adult"]),
    ("sibling_swap", ["person"], ["bicycle"]),
    ("sibling_swap", ["car"], ["motorcycle"]),
    ("sibling_swap", ["bus"], ["truck"]),
    ("disjoint", ["person"], ["airplane"]),
    ("disjoint", ["car"], ["dog"]),
    ("disjoint", ["person", "car"], ["cat", "bird"]),
    ("disjoint", ["bicycle"], ["airplane", "boat"]),
]


class PromptBridgeDataset(torch.utils.data.Dataset):
    """
    Synthetic training dataset for PromptBridge.

    Each sample:
        old_emb: CLIP embedding of old vocabulary [clip_dim]
        new_emb: CLIP embedding of new vocabulary [clip_dim]
        track_emb: Simulated track appearance embedding [clip_dim]
        gate_label: 1=preserve, 0=terminate [1]
    """

    def __init__(
        self,
        clip_model_name: str = "ViT-B/32",
        device: str = "cuda",
        samples_per_pair: int = 50,
        noise_std: float = 0.05,
    ) -> None:
        self.device = device
        self.samples_per_pair = samples_per_pair
        self.noise_std = noise_std

        print("Loading CLIP model...")
        self.clip_model, _ = clip.load(clip_model_name, device=device)
        self.clip_model.eval()
        self.clip_dim = 512

        print("Generating training samples...")
        self.samples = self._generate()
        print(f"Dataset size: {len(self.samples)} samples")

    def _encode_vocab(self, vocab: List[str]) -> torch.Tensor:
        """Encode vocabulary list to mean CLIP embedding."""
        with torch.no_grad():
            tokens = clip.tokenize(vocab).to(self.device)
            embs = self.clip_model.encode_text(tokens).float()
            embs = embs / embs.norm(dim=-1, keepdim=True)
            return embs.mean(dim=0)

    def _generate(self) -> List[Dict]:
        samples = []
        for t_type, old_vocab, new_vocab in TRANSITION_PAIRS:
            old_emb = self._encode_vocab(old_vocab).cpu()
            new_emb = self._encode_vocab(new_vocab).cpu()
            gate_label = float(GATE_LABELS[t_type])

            for i in range(self.samples_per_pair):
                # Add more realistic track embedding variation
                # Track embeddings should reflect visual appearance
                # not just text embeddings

                # Vary noise level across samples
                noise_level = self.noise_std * (1 + i % 5)

                if gate_label == 1.0:
                    # Preserve case: track looks like old vocab
                    # but with varying levels of noise
                    base = old_emb if i % 3 != 0 else (old_emb + new_emb) / 2
                    track_emb = base + torch.randn_like(old_emb) * noise_level
                else:
                    # Terminate case: track looks like old vocab
                    # new vocab is completely different domain
                    # add random direction to make it harder
                    random_dir = torch.randn_like(old_emb)
                    random_dir = random_dir / random_dir.norm()
                    track_emb = old_emb + random_dir * noise_level * 2

                track_emb = track_emb / track_emb.norm()

                samples.append({
                    "old_emb": old_emb,
                    "new_emb": new_emb,
                    "track_emb": track_emb,
                    "gate_label": torch.tensor([gate_label]),
                    "transition_type": t_type,
                })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def get_dataloader(
    dataset: PromptBridgeDataset,
    batch_size: int = 32,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: {
            "old_emb": torch.stack([b["old_emb"] for b in batch]),
            "new_emb": torch.stack([b["new_emb"] for b in batch]),
            "track_emb": torch.stack([b["track_emb"] for b in batch]),
            "gate_label": torch.stack([b["gate_label"] for b in batch]),
        }
    )
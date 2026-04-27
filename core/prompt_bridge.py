from __future__ import annotations
import torch
import torch.nn as nn
from typing import Tuple


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation.
    Conditions visual features on text embedding difference.
    gamma and beta are predicted from the conditioning vector.
    output = gamma * x + beta
    """

    def __init__(self, feature_dim: int, condition_dim: int) -> None:
        super().__init__()
        self.gamma_net = nn.Linear(condition_dim, feature_dim)
        self.beta_net = nn.Linear(condition_dim, feature_dim)

    def forward(
        self, x: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        gamma = self.gamma_net(condition)
        beta = self.beta_net(condition)
        return gamma * x + beta


class PromptBridge(nn.Module):
    """
    Lightweight FiLM-conditioned adapter for vocabulary-drift mitigation.

    Takes:
        old_emb: CLIP text embedding of old vocabulary [B, clip_dim]
        new_emb: CLIP text embedding of new vocabulary [B, clip_dim]
        track_emb: Track appearance embedding [B, track_dim]

    Outputs:
        gate: Per-track preserve/terminate decision [B, 1] in [0, 1]
        blended_emb: Smoothed class embedding for transition window [B, clip_dim]

    Architecture:
        1. Compute drift vector: delta = new_emb - old_emb
        2. Project drift to condition space
        3. FiLM-condition track embedding on drift
        4. MLP gate: outputs preserve (1) or terminate (0)
        5. Blend old/new embeddings weighted by gate
    """

    def __init__(
        self,
        clip_dim: int = 512,
        track_dim: int = 512,
        hidden_dim: int = 256,
        condition_dim: int = 128,
    ) -> None:
        super().__init__()

        self.clip_dim = clip_dim
        self.track_dim = track_dim

        # Project drift vector to condition space
        self.drift_projector = nn.Sequential(
            nn.Linear(clip_dim, condition_dim),
            nn.ReLU(),
            nn.Linear(condition_dim, condition_dim),
        )

        # Project track embedding to feature space
        self.track_projector = nn.Sequential(
            nn.Linear(track_dim, hidden_dim),
            nn.ReLU(),
        )

        # FiLM layer — condition track features on drift
        self.film = FiLMLayer(
            feature_dim=hidden_dim,
            condition_dim=condition_dim,
        )

        # Gate MLP — outputs preserve/terminate decision
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Embedding blender — smooth transition
        self.blend_projector = nn.Sequential(
            nn.Linear(clip_dim * 2 + 1, clip_dim),
            nn.ReLU(),
            nn.Linear(clip_dim, clip_dim),
        )

    def forward(
        self,
        old_emb: torch.Tensor,
        new_emb: torch.Tensor,
        track_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # Compute drift vector
        delta = new_emb - old_emb  # [B, clip_dim]

        # Project drift to condition space
        condition = self.drift_projector(delta)  # [B, condition_dim]

        # Project track embedding
        track_features = self.track_projector(track_emb)  # [B, hidden_dim]

        # FiLM conditioning
        conditioned = self.film(track_features, condition)  # [B, hidden_dim]

        # Gate prediction
        gate = self.gate_mlp(conditioned)  # [B, 1]

        # Blend embeddings weighted by gate
        blend_input = torch.cat([old_emb, new_emb, gate], dim=-1)
        blended_emb = self.blend_projector(blend_input)  # [B, clip_dim]

        return gate, blended_emb

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
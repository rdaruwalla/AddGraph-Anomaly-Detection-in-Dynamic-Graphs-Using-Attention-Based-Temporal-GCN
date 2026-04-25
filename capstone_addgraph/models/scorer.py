from typing import Sequence

import torch
import torch.nn as nn

from capstone_addgraph.data.types import ScoredEdge


class AnomalyScorer(nn.Module):
    def __init__(self, hidden_dim: int, beta: float, mu: float):
        super().__init__()
        self.a = nn.Parameter(torch.ones(hidden_dim))
        self.b = nn.Parameter(torch.ones(hidden_dim))
        self.beta = beta
        self.mu = mu

    def forward(self, h: torch.Tensor, weighted_edges: Sequence[ScoredEdge]) -> torch.Tensor:
        if not weighted_edges:
            return torch.empty(0, device=h.device)

        us = torch.tensor([u for u, _, _ in weighted_edges], dtype=torch.long, device=h.device)
        vs = torch.tensor([v for _, v, _ in weighted_edges], dtype=torch.long, device=h.device)
        ws = torch.tensor([w for _, _, w in weighted_edges], dtype=torch.float32, device=h.device)

        h_u = h[us]
        h_v = h[vs]
        combined = self.a * h_u + self.b * h_v
        norm_sq = torch.sum(combined * combined, dim=1)
        return ws * torch.sigmoid(self.beta * (norm_sq - self.mu))

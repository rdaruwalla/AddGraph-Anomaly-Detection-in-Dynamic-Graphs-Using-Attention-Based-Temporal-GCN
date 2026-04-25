from typing import Sequence

import torch
import torch.nn.functional as F

from capstone_addgraph.data.types import ScoredEdge


def pairwise_margin_loss(model, margin: float, h_t: torch.Tensor, positives: Sequence[ScoredEdge], negatives: Sequence[ScoredEdge]) -> torch.Tensor:
    pos_scores = model.scorer(h_t, positives)
    neg_scores = model.scorer(h_t, negatives)

    if pos_scores.numel() == 0:
        return torch.tensor(0.0, device=h_t.device)

    return F.relu(margin + pos_scores - neg_scores).mean()

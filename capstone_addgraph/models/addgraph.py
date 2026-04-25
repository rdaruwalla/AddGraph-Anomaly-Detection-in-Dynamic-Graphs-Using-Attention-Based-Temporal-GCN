from collections import deque
from typing import Deque

import torch
import torch.nn as nn

from capstone_addgraph.models.attention import ContextualAttentionBlock
from capstone_addgraph.models.config import AddGraphConfig
from capstone_addgraph.models.gcn import MultiLayerGCN
from capstone_addgraph.models.scorer import AnomalyScorer


class AddGraph(nn.Module):
    def __init__(self, config: AddGraphConfig):
        super().__init__()
        self.config = config
        self.num_nodes = config.num_nodes
        self.hidden_dim = config.hidden_dim
        self.window_size = config.window_size

        self.h0 = nn.Parameter(torch.randn(self.num_nodes, self.hidden_dim) * 0.01)
        self.gcn = MultiLayerGCN(config.hidden_dim, config.gcn_layers, config.dropout)
        self.cab = ContextualAttentionBlock(config.hidden_dim)
        self.gru = nn.GRUCell(config.hidden_dim, config.hidden_dim)
        self.scorer = AnomalyScorer(config.hidden_dim, config.beta, config.mu)

    def init_history(self, device: str) -> Deque[torch.Tensor]:
        h0 = self.h0.to(device)
        return deque([h0.clone() for _ in range(self.window_size)], maxlen=self.window_size)

    def step(self, a_hat: torch.Tensor, history: Deque[torch.Tensor]) -> torch.Tensor:
        h_prev = history[-1]
        current_t = self.gcn(a_hat, h_prev)
        history_tensor = torch.stack(list(history), dim=0)
        short_t = self.cab(history_tensor)
        return self.gru(current_t, short_t)

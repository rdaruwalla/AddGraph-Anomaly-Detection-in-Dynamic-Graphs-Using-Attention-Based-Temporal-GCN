import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLayerGCN(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = dropout

    def forward(self, a_hat: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        z = h_prev
        for idx, layer in enumerate(self.layers):
            z = a_hat @ z
            z = layer(z)
            z = F.relu(z)
            if self.dropout > 0 and idx < len(self.layers) - 1:
                z = F.dropout(z, p=self.dropout, training=self.training)
        return z

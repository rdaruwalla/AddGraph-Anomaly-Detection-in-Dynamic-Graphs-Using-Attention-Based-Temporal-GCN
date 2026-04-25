import torch
import torch.nn as nn


class ContextualAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.q_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.r = nn.Parameter(torch.randn(hidden_dim) * 0.02)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.size(0) == 1:
            return history[0]

        transformed = torch.tanh(self.q_h(history))
        scores = torch.einsum("wnd,d->wn", transformed, self.r)
        attn = torch.softmax(scores, dim=0)
        short_t = torch.einsum("wn,wnd->nd", attn, history)
        return short_t


class MeanHistoryBlock(nn.Module):
    def forward(self, history: torch.Tensor) -> torch.Tensor:
        return history.mean(dim=0)

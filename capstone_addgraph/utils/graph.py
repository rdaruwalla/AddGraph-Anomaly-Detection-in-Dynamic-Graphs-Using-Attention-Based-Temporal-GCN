from typing import Sequence

import torch

from capstone_addgraph.data.types import Edge


def normalize_adjacency(num_nodes: int, edges: Sequence[Edge], device: str) -> torch.Tensor:
    a = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    for u, v, w, _ in edges:
        a[u, v] += float(w)
        a[v, u] += float(w)

    a_tilde = a + torch.eye(num_nodes, device=device)
    degree = a_tilde.sum(dim=1)
    d_inv_sqrt = torch.pow(degree.clamp(min=1.0), -0.5)
    d_mat = torch.diag(d_inv_sqrt)
    return d_mat @ a_tilde @ d_mat

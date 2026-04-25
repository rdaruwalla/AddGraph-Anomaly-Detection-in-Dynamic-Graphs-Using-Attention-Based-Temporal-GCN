from collections import defaultdict
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import torch

from capstone_addgraph.data.types import Edge, ScoredEdge


def degrees_from_edges(edges: Sequence[Edge]) -> Dict[int, float]:
    deg: Dict[int, float] = defaultdict(float)
    for u, v, w, _ in edges:
        deg[u] += float(w)
        deg[v] += float(w)
    return deg


def existing_edge_set(edges: Sequence[Edge]) -> Set[Tuple[int, int]]:
    return {(u, v) for u, v, _, _ in edges}


def bernoulli_negative_sample(
    u: int,
    v: int,
    w: float,
    num_nodes: int,
    degrees: Dict[int, float],
    rng: np.random.Generator,
    forbidden_edges: Set[Tuple[int, int]] | None = None,
    max_tries: int = 50,
) -> ScoredEdge:
    forbidden_edges = forbidden_edges or set()
    d_u = max(degrees.get(u, 1.0), 1.0)
    d_v = max(degrees.get(v, 1.0), 1.0)
    p_replace_u = d_u / (d_u + d_v)
    
    for _ in range(max_tries):
        if rng.random() < p_replace_u:
            u_neg = int(rng.integers(0, num_nodes))
            if u_neg != u and (u_neg, v) not in forbidden_edges:
                return u_neg, v, w
        else:
            v_neg = int(rng.integers(0, num_nodes))
            if v_neg != v and (u, v_neg) not in forbidden_edges:
                return u, v_neg, w

    for offset in range(1, num_nodes):
        u_neg = (u + offset) % num_nodes
        if u_neg != u and (u_neg, v) not in forbidden_edges:
            return u_neg, v, w
    return (u + 1) % num_nodes, v, w


@torch.no_grad()
def filter_selective_pairs(model, h_t: torch.Tensor, positives: Sequence[ScoredEdge], negatives: Sequence[ScoredEdge]) -> Tuple[List[ScoredEdge], List[ScoredEdge]]:
    if not positives:
        return [], []

    pos_scores = model.scorer(h_t, positives)
    neg_scores = model.scorer(h_t, negatives)

    keep_pos: List[ScoredEdge] = []
    keep_neg: List[ScoredEdge] = []
    for idx in range(len(positives)):
        if pos_scores[idx] <= neg_scores[idx]:
            keep_pos.append(positives[idx])
            keep_neg.append(negatives[idx])
    return keep_pos, keep_neg

from typing import Dict, List, Sequence

import numpy as np
import torch

from capstone_addgraph.data.types import SnapshotBatch
from capstone_addgraph.training.losses import pairwise_margin_loss
from capstone_addgraph.training.negative_sampling import (
    bernoulli_negative_sample,
    degrees_from_edges,
    existing_edge_set,
    filter_selective_pairs,
)
from capstone_addgraph.utils.graph import normalize_adjacency


class GenericTrainer:
    def __init__(self, model, config):
        self.model = model.to(config.device)
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

    def fit(self, snapshots: Sequence[SnapshotBatch], train_until: int, epochs: int = 10) -> List[Dict[str, float | int]]:
        history_rows: List[Dict[str, float | int]] = []

        for epoch in range(epochs):
            self.model.train()
            history = self.model.init_history(self.config.device)
            total_loss = 0.0
            steps = 0
            skipped_no_benign = 0
            skipped_no_pairs = 0

            print(f"Starting epoch {epoch + 1}/{epochs}")

            for t in range(train_until):
                if t % 10 == 0:
                    print(f"  epoch {epoch + 1}: snapshot {t + 1}/{train_until}")

                edges_t = snapshots[t].edges
                if not edges_t:
                    continue

                a_hat = normalize_adjacency(self.config.num_nodes, edges_t, self.config.device)
                h_t = self.model.step(a_hat, history)

                positives = [(u, v, w) for (u, v, w, label) in edges_t if label == 0]
                if not positives:
                    skipped_no_benign += 1
                    history.append(h_t.detach())
                    continue

                degrees = degrees_from_edges(edges_t)
                forbidden = existing_edge_set(edges_t)
                negatives = [
                    bernoulli_negative_sample(u, v, w, self.config.num_nodes, degrees, self.rng, forbidden)
                    for (u, v, w) in positives
                ]

                if self.config.filter_training_pairs:
                    positives, negatives = filter_selective_pairs(self.model, h_t, positives, negatives)

                if not positives:
                    skipped_no_pairs += 1
                    history.append(h_t.detach())
                    continue

                loss = pairwise_margin_loss(self.model, self.config.margin, h_t, positives, negatives)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    refreshed = self.model.step(a_hat, history)
                    history.append(refreshed.detach())

                total_loss += float(loss.item())
                steps += 1

            mean_loss = total_loss / max(steps, 1)
            print(f"Finished epoch {epoch + 1}/{epochs} mean_loss={mean_loss:.4f}")
            
            row = {
                "epoch": epoch + 1,
                "mean_loss": mean_loss,
                "steps": steps,
                "skipped_no_benign": skipped_no_benign,
                "skipped_no_pairs": skipped_no_pairs,
            }
            history_rows.append(row)
            print(
                f"epoch={epoch + 1:02d} mean_loss={mean_loss:.4f} "
                f"steps={steps} skipped_no_benign={skipped_no_benign} skipped_no_pairs={skipped_no_pairs}"
            )

        return history_rows

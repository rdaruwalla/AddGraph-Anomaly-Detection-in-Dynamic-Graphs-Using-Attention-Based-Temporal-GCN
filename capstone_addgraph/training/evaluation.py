from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from capstone_addgraph.data.types import SnapshotBatch
from capstone_addgraph.training.negative_sampling import (
    bernoulli_negative_sample,
    degrees_from_edges,
    existing_edge_set,
    filter_selective_pairs,
)
from capstone_addgraph.utils.graph import normalize_adjacency


@dataclass
class EvaluationResult:
    overall_auc: float
    overall_pr_auc: float
    threshold_precision: float
    threshold_recall: float
    threshold_f1: float
    threshold_used: float
    per_snapshot_auc: List[Dict[str, float | int | str]]
    total_edges_scored: int
    total_attack_edges_scored: int
    score_trend_summary: Dict[str, float | int]



def _safe_snapshot_metrics(labels: List[int], scores: List[float]) -> Dict[str, float | int | str]:
    benign_scores = [score for score, label in zip(scores, labels) if label == 0]
    attack_scores = [score for score, label in zip(scores, labels) if label == 1]

    row: Dict[str, float | int | str] = {
        "edges_scored": len(labels),
        "attack_edges_scored": int(sum(labels)),
        "benign_edges_scored": int(len(labels) - sum(labels)),
        "mean_score": float(mean(scores)) if scores else 0.0,
        "max_score": float(max(scores)) if scores else 0.0,
        "min_score": float(min(scores)) if scores else 0.0,
        "mean_attack_score": float(mean(attack_scores)) if attack_scores else 0.0,
        "mean_benign_score": float(mean(benign_scores)) if benign_scores else 0.0,
    }
    if len(set(labels)) >= 2:
        row["auc"] = float(roc_auc_score(labels, scores))
        row["pr_auc"] = float(average_precision_score(labels, scores))
    else:
        row["auc"] = "NA"
        row["pr_auc"] = "NA"
    return row



def evaluate_model(
    model,
    config,
    rng,
    snapshots: Sequence[SnapshotBatch],
    train_until: int,
    decision_threshold: float = 0.5,
) -> EvaluationResult:
    import torch
    model.eval()
    history = model.init_history(config.device)

    all_scores: List[float] = []
    all_labels: List[int] = []
    per_snapshot: List[Dict[str, float | int | str]] = []

    for t in range(len(snapshots)):
        edges_t = snapshots[t].edges
        if not edges_t:
            continue

        a_hat = normalize_adjacency(config.num_nodes, edges_t, config.device)
        h_t = model.step(a_hat, history)

        if t >= train_until:
            eval_edges = [(u, v, w) for (u, v, w, _) in edges_t]
            labels    = [label for (_, _, _, label) in edges_t]

            # score every real edge — no negative sampling, no filtering
            with torch.no_grad():
                scores = model.scorer(h_t, eval_edges).cpu().numpy().tolist()

            snapshot_row = {
                "snapshot_index": t,
                "start_time": str(snapshots[t].start_time),
                "end_time":   str(snapshots[t].end_time),
            }
            snapshot_row.update(_safe_snapshot_metrics(labels, scores))
            per_snapshot.append(snapshot_row)
            all_scores.extend(scores)
            all_labels.extend(labels)

        history.append(h_t.detach())

    if len(set(all_labels)) < 2:
        raise ValueError("Evaluation set needs both benign and attack labels for AUC.")

    binary_preds = (np.asarray(all_scores) >= decision_threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, binary_preds, average="binary", zero_division=0,
    )

    snapshot_mean_scores    = [float(r["mean_score"])   for r in per_snapshot if r["edges_scored"] > 0]
    snapshot_attack_scores  = [float(r["mean_attack_score"]) for r in per_snapshot if r["attack_edges_scored"] > 0]
    snapshot_benign_scores  = [float(r["mean_benign_score"]) for r in per_snapshot if r["benign_edges_scored"] > 0]

    score_trend_summary = {
        "num_eval_snapshots":        len(per_snapshot),
        "avg_snapshot_mean_score":   float(mean(snapshot_mean_scores))   if snapshot_mean_scores   else 0.0,
        "avg_snapshot_attack_score": float(mean(snapshot_attack_scores)) if snapshot_attack_scores else 0.0,
        "avg_snapshot_benign_score": float(mean(snapshot_benign_scores)) if snapshot_benign_scores else 0.0,
        "max_snapshot_mean_score":   float(max(snapshot_mean_scores))    if snapshot_mean_scores   else 0.0,
        "min_snapshot_mean_score":   float(min(snapshot_mean_scores))    if snapshot_mean_scores   else 0.0,
    }

    return EvaluationResult(
        overall_auc=float(roc_auc_score(all_labels, all_scores)),
        overall_pr_auc=float(average_precision_score(all_labels, all_scores)),
        threshold_precision=float(precision),
        threshold_recall=float(recall),
        threshold_f1=float(f1),
        threshold_used=float(decision_threshold),
        per_snapshot_auc=per_snapshot,
        total_edges_scored=len(all_labels),
        total_attack_edges_scored=int(sum(all_labels)),
        score_trend_summary=score_trend_summary,
    )
from typing import Dict, List, Sequence

from capstone_addgraph.data.types import SnapshotBatch


def summarize_snapshots(snapshots: Sequence[SnapshotBatch], train_until: int) -> Dict[str, int | List[dict]]:
    train_benign = 0
    train_attack = 0
    test_benign = 0
    test_attack = 0
    per_snapshot = []

    for idx, snapshot in enumerate(snapshots):
        benign = sum(1 for _, _, _, label in snapshot.edges if label == 0)
        attack = sum(1 for _, _, _, label in snapshot.edges if label == 1)
        per_snapshot.append({
            "snapshot_index": idx,
            "edges": len(snapshot.edges),
            "benign_edges": benign,
            "attack_edges": attack,
            "start_time": snapshot.start_time,
            "end_time": snapshot.end_time,
            "split": "train" if idx < train_until else "test",
        })

        if idx < train_until:
            train_benign += benign
            train_attack += attack
        else:
            test_benign += benign
            test_attack += attack

    return {
        "num_snapshots": len(snapshots),
        "train_snapshots": train_until,
        "test_snapshots": len(snapshots) - train_until,
        "train_benign_edges": train_benign,
        "train_attack_edges": train_attack,
        "test_benign_edges": test_benign,
        "test_attack_edges": test_attack,
        "per_snapshot": per_snapshot,
    }

import argparse
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from capstone_addgraph.data.dataset_stats import summarize_snapshots
from capstone_addgraph.data.loaders import load_cicids_graph_snapshots
from capstone_addgraph.models.addgraph import AddGraph
from capstone_addgraph.models.config import AddGraphConfig
from capstone_addgraph.models.static_gcn_baseline import StaticGCNBaseline
from capstone_addgraph.models.temporal_no_attention import TemporalGCNNoAttention
from capstone_addgraph.training.evaluation import evaluate_model
from capstone_addgraph.training.trainer import GenericTrainer
from capstone_addgraph.utils.io_utils import ensure_dir, save_csv, save_json
from capstone_addgraph.utils.seed import set_global_seed

MODEL_REGISTRY = {
    "addgraph": AddGraph,
    "static_gcn": StaticGCNBaseline,
    "temporal_no_attention": TemporalGCNNoAttention,
}


def _parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--bucket", type=str, default="1H")
    parser.add_argument("--max_rows_per_file", type=int, default=None)
    parser.add_argument("--min_edge_weight", type=int, default=1)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--gcn_layers", type=int, default=3)
    parser.add_argument("--window_size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.6)
    parser.add_argument("--weight_decay", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--mu", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, choices=sorted(MODEL_REGISTRY), default="addgraph")
    parser.add_argument("--disable_training_pair_filter", action="store_true")
    parser.add_argument("--decision_threshold", type=float, default=0.5)
    parser.add_argument("--attack_types", type=str, default="")
    parser.add_argument("--allowed_files", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="/mnt/data/capstone_addgraph/results")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    set_global_seed(args.seed)

    selected_attacks = _parse_csv_list(args.attack_types)
    selected_files = _parse_csv_list(args.allowed_files)

    snapshots, node_map, flow_df, loader_info = load_cicids_graph_snapshots(
        data_dir=args.data_dir,
        bucket=args.bucket,
        max_rows_per_file=args.max_rows_per_file,
        min_edge_weight=args.min_edge_weight,
        attack_types=selected_attacks,
        allowed_filenames=selected_files,
    )

    if len(snapshots) < 4:
        raise ValueError("Need more snapshots. Try a smaller bucket like 15min, 5min, or 1min.")

    train_until = max(1, min(len(snapshots) - 1, int(math.floor(len(snapshots) * args.train_ratio))))
    print(f"Training snapshots: 0 .. {train_until - 1}")
    print(f"Testing snapshots:  {train_until} .. {len(snapshots) - 1}")

    dataset_stats = summarize_snapshots(snapshots, train_until)
    dataset_stats["loader_info"] = loader_info

    print(
        f"Train benign edges={dataset_stats['train_benign_edges']:,} "
        f"train attack edges={dataset_stats['train_attack_edges']:,} "
        f"test benign edges={dataset_stats['test_benign_edges']:,} "
        f"test attack edges={dataset_stats['test_attack_edges']:,}"
    )

    cfg = AddGraphConfig(
        num_nodes=len(node_map),
        hidden_dim=args.hidden_dim,
        gcn_layers=args.gcn_layers,
        window_size=args.window_size,
        beta=args.beta,
        mu=args.mu,
        margin=args.margin,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        filter_training_pairs=not args.disable_training_pair_filter,
        device=args.device,
        seed=args.seed,
    )

    model = MODEL_REGISTRY[args.model_name](cfg)
    trainer = GenericTrainer(model, cfg)
    train_rows = trainer.fit(snapshots, train_until=train_until, epochs=args.epochs)
    eval_result = evaluate_model(
        trainer.model,
        cfg,
        trainer.rng,
        snapshots,
        train_until,
        decision_threshold=args.decision_threshold,
    )

    out_dir = ensure_dir(Path(args.output_dir) / args.model_name / f"seed_{args.seed}")
    save_json(out_dir / "config.json", vars(args))
    save_json(out_dir / "dataset_stats.json", dataset_stats)
    save_csv(out_dir / "training_history.csv", train_rows)
    save_csv(out_dir / "per_snapshot_metrics.csv", eval_result.per_snapshot_auc)
    save_json(out_dir / "summary.json", {
        "model_name": args.model_name,
        "overall_auc": eval_result.overall_auc,
        "overall_pr_auc": eval_result.overall_pr_auc,
        "threshold_used": eval_result.threshold_used,
        "threshold_precision": eval_result.threshold_precision,
        "threshold_recall": eval_result.threshold_recall,
        "threshold_f1": eval_result.threshold_f1,
        "total_edges_scored": eval_result.total_edges_scored,
        "total_attack_edges_scored": eval_result.total_attack_edges_scored,
        "num_nodes": len(node_map),
        "num_snapshots": len(snapshots),
        "train_until": train_until,
        "rows_after_cleaning": int(len(flow_df)),
        "loader_info": loader_info,
        "score_trend_summary": eval_result.score_trend_summary,
    })

    print(
        f"model={args.model_name} test_auc={eval_result.overall_auc:.4f} "
        f"pr_auc={eval_result.overall_pr_auc:.4f} "
        f"precision@{eval_result.threshold_used:.2f}={eval_result.threshold_precision:.4f} "
        f"recall@{eval_result.threshold_used:.2f}={eval_result.threshold_recall:.4f} "
        f"f1@{eval_result.threshold_used:.2f}={eval_result.threshold_f1:.4f}"
    )


if __name__ == "__main__":
    main()

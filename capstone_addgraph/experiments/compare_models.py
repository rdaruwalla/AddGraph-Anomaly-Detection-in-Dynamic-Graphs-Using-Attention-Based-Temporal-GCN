import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "capstone_addgraph" / "experiments" / "run_experiment.py"


MODEL_NAMES = ["addgraph", "static_gcn", "temporal_no_attention"]



def _parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/mnt/data/capstone_addgraph/results")
    parser.add_argument("--bucket", type=str, default="5min")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--attack_types", type=str, default="")
    parser.add_argument("--allowed_files", type=str, default="")
    args = parser.parse_args()

    seeds = [int(seed) for seed in _parse_csv_list(args.seeds)]
    summary_rows: list[dict] = []

    for model_name in MODEL_NAMES:
        for seed in seeds:
            cmd = [
                sys.executable,
                str(RUNNER),
                "--data_dir", args.data_dir,
                "--model_name", model_name,
                "--bucket", args.bucket,
                "--epochs", str(args.epochs),
                "--train_ratio", str(args.train_ratio),
                "--device", args.device,
                "--seed", str(seed),
                "--output_dir", args.output_dir,
            ]
            if args.attack_types:
                cmd.extend(["--attack_types", args.attack_types])
            if args.allowed_files:
                cmd.extend(["--allowed_files", args.allowed_files])

            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)

            summary_path = Path(args.output_dir) / model_name / f"seed_{seed}" / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows)
    compare_dir = Path(args.output_dir) / "comparisons"
    compare_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(compare_dir / "all_model_runs.csv", index=False)

    grouped = (
        summary_df.groupby("model_name", as_index=False)
        .agg(
            mean_auc=("overall_auc", "mean"),
            std_auc=("overall_auc", "std"),
            mean_pr_auc=("overall_pr_auc", "mean"),
            std_pr_auc=("overall_pr_auc", "std"),
            mean_f1=("threshold_f1", "mean"),
            std_f1=("threshold_f1", "std"),
        )
        .sort_values("mean_auc", ascending=False)
    )
    grouped.to_csv(compare_dir / "model_comparison_summary.csv", index=False)
    print(grouped.to_string(index=False))


if __name__ == "__main__":
    main()

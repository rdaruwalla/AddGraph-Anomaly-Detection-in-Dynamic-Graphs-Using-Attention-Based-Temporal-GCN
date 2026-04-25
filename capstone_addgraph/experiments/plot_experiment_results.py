import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _collect_summary_rows(results_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for summary_path in results_dir.rglob("summary.json"):
        try:
            row = json.loads(summary_path.read_text(encoding="utf-8"))
            row["summary_path"] = str(summary_path)
            rows.append(row)
        except Exception:
            continue
    return pd.DataFrame(rows)


def _collect_snapshot_rows(results_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for snapshot_path in results_dir.rglob("per_snapshot_metrics.csv"):
        try:
            df = pd.read_csv(snapshot_path)
            model_name = snapshot_path.parts[-3]
            seed_name = snapshot_path.parts[-2]
            df["model_name"] = model_name
            df["seed_name"] = seed_name
            df["source_path"] = str(snapshot_path)
            rows.append(df)
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_df = _collect_summary_rows(results_dir)
    snapshot_df = _collect_snapshot_rows(results_dir)

    if not summary_df.empty:
        summary_df.to_csv(results_dir / "all_summaries.csv", index=False)

        compare = (
            summary_df.groupby("model_name", as_index=False)
            .agg(
                mean_auc=("overall_auc", "mean"),
                mean_pr_auc=("overall_pr_auc", "mean"),
                mean_f1=("threshold_f1", "mean"),
            )
            .sort_values("mean_auc", ascending=False)
        )
        compare.to_csv(results_dir / "model_summary_table.csv", index=False)

        plt.figure(figsize=(8, 5))
        plt.bar(compare["model_name"], compare["mean_auc"])
        plt.ylabel("Mean AUC")
        plt.title("Model Comparison by Mean AUC")
        plt.tight_layout()
        plt.savefig(plots_dir / "model_mean_auc.png", dpi=180)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.bar(compare["model_name"], compare["mean_pr_auc"])
        plt.ylabel("Mean PR-AUC")
        plt.title("Model Comparison by Mean PR-AUC")
        plt.tight_layout()
        plt.savefig(plots_dir / "model_mean_pr_auc.png", dpi=180)
        plt.close()

    if not snapshot_df.empty:
        numeric_snapshot = snapshot_df.copy()
        numeric_snapshot["auc"] = pd.to_numeric(numeric_snapshot["auc"], errors="coerce")
        numeric_snapshot["mean_score"] = pd.to_numeric(numeric_snapshot.get("mean_score"), errors="coerce")
        numeric_snapshot["mean_attack_score"] = pd.to_numeric(numeric_snapshot.get("mean_attack_score"), errors="coerce")
        numeric_snapshot["mean_benign_score"] = pd.to_numeric(numeric_snapshot.get("mean_benign_score"), errors="coerce")

        auc_by_snapshot = (
            numeric_snapshot.groupby(["model_name", "snapshot_index"], as_index=False)["auc"]
            .mean()
            .dropna()
        )
        plt.figure(figsize=(9, 5))
        for model_name, sub in auc_by_snapshot.groupby("model_name"):
            plt.plot(sub["snapshot_index"], sub["auc"], marker="o", label=model_name)
        plt.xlabel("Snapshot Index")
        plt.ylabel("AUC")
        plt.title("Per-Snapshot AUC")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "per_snapshot_auc.png", dpi=180)
        plt.close()

        if "mean_score" in numeric_snapshot.columns:
            score_by_snapshot = (
                numeric_snapshot.groupby(["model_name", "snapshot_index"], as_index=False)["mean_score"]
                .mean()
                .dropna()
            )
            plt.figure(figsize=(9, 5))
            for model_name, sub in score_by_snapshot.groupby("model_name"):
                plt.plot(sub["snapshot_index"], sub["mean_score"], marker="o", label=model_name)
            plt.xlabel("Snapshot Index")
            plt.ylabel("Mean Anomaly Score")
            plt.title("Mean Anomaly Score Over Time")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / "mean_anomaly_score_over_time.png", dpi=180)
            plt.close()

            attack_benign = (
                numeric_snapshot.groupby(["model_name", "snapshot_index"], as_index=False)[["mean_attack_score", "mean_benign_score"]]
                .mean()
                .dropna(how="all")
            )
            for model_name, sub in attack_benign.groupby("model_name"):
                plt.figure(figsize=(9, 5))
                plt.plot(sub["snapshot_index"], sub["mean_attack_score"], marker="o", label="attack")
                plt.plot(sub["snapshot_index"], sub["mean_benign_score"], marker="o", label="benign")
                plt.xlabel("Snapshot Index")
                plt.ylabel("Mean Score")
                plt.title(f"Attack vs Benign Mean Scores: {model_name}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(plots_dir / f"attack_vs_benign_scores_{model_name}.png", dpi=180)
                plt.close()


if __name__ == "__main__":
    main()

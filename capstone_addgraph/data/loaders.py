from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from capstone_addgraph.data.types import SnapshotBatch


def parse_cicids_timestamp(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    if parsed.isna().all():
        parsed = pd.to_datetime(series, errors="coerce")
    return parsed


def read_csv_with_fallback(
    csv_file: Path,
    max_rows_per_file: int | None = None,
) -> pd.DataFrame:
    encodings = ["utf-8", "cp1252", "latin1"]
    last_error: Exception | None = None

    for encoding in encodings:
        try:
            return pd.read_csv(
                csv_file,
                low_memory=False,
                nrows=max_rows_per_file,
                encoding=encoding,
            )
        except UnicodeDecodeError as error:
            last_error = error

    raise UnicodeDecodeError(
        "fallback",
        b"",
        0,
        1,
        f"Could not read {csv_file.name} with tried encodings: {encodings}. Last error: {last_error}",
    )


def load_cicids_graph_snapshots(
    data_dir: str,
    bucket: str = "1H",
    max_rows_per_file: int | None = None,
    min_edge_weight: int = 1,
    attack_types: Sequence[str] | None = None,
    allowed_filenames: Sequence[str] | None = None,
) -> Tuple[List[SnapshotBatch], Dict[str, int], pd.DataFrame, dict]:
    data_path = Path(data_dir)
    csv_files = sorted(data_path.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {data_dir}")

    allowed_names = None
    if allowed_filenames:
        allowed_names = {name.strip() for name in allowed_filenames if name.strip()}

    if allowed_names is not None:
        csv_files = [file for file in csv_files if file.name in allowed_names]

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files matched the allowed filename filter in: {data_dir}"
        )

    selected_attack_types = None
    if attack_types:
        selected_attack_types = {
            attack.strip().upper() for attack in attack_types if attack.strip()
        }

    frames: List[pd.DataFrame] = []
    per_file_summary: List[dict] = []
    original_label_counts: Dict[str, int] = {}

    for csv_file in csv_files:
        print(f"Reading {csv_file.name} ...")

        df = read_csv_with_fallback(
            csv_file,
            max_rows_per_file=max_rows_per_file,
        )
        df.columns = [col.strip() for col in df.columns]

        timestamp_col = "Timestamp"
        src_col = "Source IP"
        dst_col = "Destination IP"
        label_col = "Label"

        current_df = pd.DataFrame(
            {
                "timestamp": parse_cicids_timestamp(df[timestamp_col]),
                "src": df[src_col].astype(str).str.strip(),
                "dst": df[dst_col].astype(str).str.strip(),
                "label": df[label_col].astype(str).str.strip(),
                "source_file": csv_file.name,
            }
        )

        current_df = current_df.dropna(subset=["timestamp", "src", "dst", "label"])
        current_df = current_df[current_df["src"] != current_df["dst"]].copy()
        current_df["label_upper"] = current_df["label"].str.upper()

        file_label_counts = current_df["label_upper"].value_counts().to_dict()
        for label, count in file_label_counts.items():
            original_label_counts[label] = original_label_counts.get(label, 0) + int(count)

        if selected_attack_types is None:
            current_df["is_attack"] = (current_df["label_upper"] != "BENIGN").astype(int)
        else:
            labels_to_keep = {"BENIGN"} | selected_attack_types
            current_df = current_df[current_df["label_upper"].isin(labels_to_keep)].copy()
            current_df["is_attack"] = current_df["label_upper"].isin(
                selected_attack_types
            ).astype(int)

        if current_df.empty:
            per_file_summary.append(
                {
                    "file": csv_file.name,
                    "rows_after_cleaning": 0,
                    "benign_rows": 0,
                    "attack_rows": 0,
                }
            )
            continue

        per_file_summary.append(
            {
                "file": csv_file.name,
                "rows_after_cleaning": int(len(current_df)),
                "benign_rows": int((current_df["is_attack"] == 0).sum()),
                "attack_rows": int((current_df["is_attack"] == 1).sum()),
            }
        )

        frames.append(current_df)

    if not frames:
        raise ValueError("No usable rows remained after cleaning and filtering.")

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.sort_values("timestamp").reset_index(drop=True)

    if all_df.empty:
        raise ValueError("No usable rows after cleaning.")

    all_df["bucket"] = all_df["timestamp"].dt.floor(bucket)

    unique_nodes = pd.Index(pd.concat([all_df["src"], all_df["dst"]], axis=0).unique())
    node_map = {node: idx for idx, node in enumerate(unique_nodes)}

    all_df["u"] = all_df["src"].map(node_map)
    all_df["v"] = all_df["dst"].map(node_map)

    grouped = (
        all_df.groupby(["bucket", "u", "v"], as_index=False)
        .agg(
            weight=("u", "size"),
            label=("is_attack", "max"),
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
        )
    )

    grouped = grouped[grouped["weight"] >= min_edge_weight].copy()
    grouped = grouped.sort_values(["bucket", "u", "v"]).reset_index(drop=True)

    snapshots: List[SnapshotBatch] = []

    for _, group in grouped.groupby("bucket", sort=True):
        edges = [
            (int(row.u), int(row.v), float(row.weight), int(row.label))
            for row in group.itertuples(index=False)
        ]

        snapshots.append(
            SnapshotBatch(
                edges=edges,
                start_time=group["start_time"].min(),
                end_time=group["end_time"].max(),
            )
        )

    total_edges = sum(len(snapshot.edges) for snapshot in snapshots)
    attack_edges = sum(label for snapshot in snapshots for (_, _, _, label) in snapshot.edges)

    loader_info = {
        "num_csv_files": len(csv_files),
        "selected_attack_types": sorted(selected_attack_types) if selected_attack_types else None,
        "original_label_counts": original_label_counts,
        "per_file_summary": per_file_summary,
        "total_cleaned_flows": int(len(all_df)),
        "unique_ip_nodes": int(len(node_map)),
        "num_snapshots": int(len(snapshots)),
        "total_aggregated_edges": int(total_edges),
        "attack_labeled_edges": int(attack_edges),
    }

    print(f"Loaded {len(csv_files)} CSV files")
    print(f"Total cleaned flows: {len(all_df):,}")
    print(f"Unique IP nodes: {len(node_map):,}")
    print(f"Snapshots: {len(snapshots):,}")
    print(f"Aggregated edges: {total_edges:,}")
    print(f"Attack-labeled edges: {attack_edges:,}")

    return snapshots, node_map, all_df, loader_info
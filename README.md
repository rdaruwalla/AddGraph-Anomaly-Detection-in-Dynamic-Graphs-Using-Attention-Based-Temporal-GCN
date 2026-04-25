# capstone_addgraph

Reproduction study of [AddGraph (IJCAI 2019)](https://arxiv.org/abs/1908.05653) applied to the CICIDS2017 network intrusion dataset. The project implements three model variants — full AddGraph, a temporal GCN without attention, and a static GCN baseline — and compares them across several experimental settings as part of a CS5100 capstone at Northeastern University.

---

## Project Structure

```
capstone_addgraph/
├── data/
│   ├── loaders.py          # loads and preprocesses CICIDS2017 into graph snapshots
│   ├── types.py            # Edge and SnapshotBatch type definitions
│   └── dataset_stats.py    # snapshot-level statistics and train/test summaries
├── models/
│   ├── config.py           # AddGraphConfig dataclass
│   ├── addgraph.py         # full AddGraph model (GCN + CAB + GRU)
│   ├── gcn.py              # multi-layer GCN
│   ├── attention.py        # contextual attention block and mean pooling block
│   ├── scorer.py           # anomaly scorer
│   ├── static_gcn_baseline.py      # static GCN baseline (no temporal modeling)
│   └── temporal_no_attention.py    # temporal GCN with mean pooling instead of attention
├── training/
│   ├── trainer.py          # snapshot-by-snapshot training loop
│   ├── evaluation.py       # evaluation loop and metrics
│   ├── losses.py           # pairwise margin loss
│   └── negative_sampling.py        # Bernoulli negative sampler and selective filter
├── experiments/
│   └── run_experiment.py   # main entry point for running a single experiment
└── utils/
    ├── graph.py            # adjacency matrix normalization
    ├── seed.py             # global seed setter
    └── io_utils.py         # directory creation, JSON and CSV saving
```

---

## Setup

You need Python 3.10+ and the following packages:

```
torch
numpy
pandas
scikit-learn
```

Install them however you prefer, e.g.:

```bash
pip install torch numpy pandas scikit-learn
```

You also need the CICIDS2017 dataset CSV files. Place them all in a single directory, e.g. `CICIDS2017/`. The loader will pick up any `.csv` files in that directory, and you can filter which files to use via `--allowed_files`.

---

## Running an Experiment

```bash
python capstone_addgraph/experiments/run_experiment.py \
  --data_dir CICIDS2017 \
  --model_name addgraph \
  --bucket 5min \
  --epochs 3 \
  --train_ratio 0.7 \
  --device cpu \
  --seed 42 \
  --hidden_dim 32 \
  --gcn_layers 2 \
  --window_size 2 \
  --max_rows_per_file 300000 \
  --allowed_files "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv,Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv,Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv" \
  --disable_selective_sampling \
  --disable_training_pair_filter \
  --output_dir ./results/setting_a/
```

`--model_name` can be `addgraph`, `static_gcn`, or `temporal_no_attention`.

Results are saved to `--output_dir/<model_name>/seed_<seed>/` and include:
- `config.json` — the full argument config used for the run
- `dataset_stats.json` — snapshot-level breakdown of edges and labels
- `summary.json` — overall ROC-AUC, PR-AUC, and score trend summary
- `training_history.csv` — per-epoch loss
- `per_snapshot_metrics.csv` — per-snapshot AUC and score statistics

---

## Results

Results from all experimental settings are stored in `results/`. The main settings are:

- `setting_a/` — window size 2, 3 files, seeds 42 and 43
- `setting_b/` — window size 1, 3 files, seeds 42 and 43
- `validation_4file/` — window size 2, 4 files, seed 42
- `selective_sampling_on/` — AddGraph with selective negative sampling enabled, 2 files, seed 42

---

## Key Findings

Temporal modeling consistently improves ROC-AUC over the static GCN baseline across all settings and seeds. The attention mechanism does not show a clear consistent advantage over mean pooling on this dataset. PR-AUC is heavily influenced by edge weight for high-volume attacks like DDoS and PortScan. See the report for full analysis.

# Reproducing AddGraph: Anomaly Detection in Dynamic Graphs Using Attention-Based Temporal GCN

CS5100 Foundations of Artificial Intelligence — Northeastern University, Spring 2026

**Demo video:** https://drive.google.com/file/d/1rbrZPsGgA9119nHxdaiAO4ssw6fnNgzs/view?usp=sharing  
**Dataset:** [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) (download separately, not included)

---

## Overview

This project reproduces [AddGraph (Zheng et al., IJCAI 2019)](https://www.ijcai.org/proceedings/2019/614), a semi-supervised framework for anomalous edge detection in dynamic graphs. Instead of replicating the original experiments on UCI Message and Digg, the method is applied to the CICIDS2017 network intrusion dataset to test whether the core architectural claims hold on real labeled attack traffic.

Three model variants are implemented and compared:

- **AddGraph** — full model with GCN, Contextual Attention Block, and GRU
- **Temporal No-Attention** — same as AddGraph but with mean pooling instead of attention
- **Static GCN** — no temporal modeling, structural baseline only

---

## Repo Structure

```
capstone_addgraph/
├── data/
│   ├── loaders.py          # loads CICIDS2017 CSVs and builds graph snapshots
│   ├── types.py            # Edge and SnapshotBatch type definitions
│   └── dataset_stats.py    # snapshot-level train/test statistics
├── models/
│   ├── config.py           # AddGraphConfig dataclass
│   ├── addgraph.py         # full AddGraph model
│   ├── gcn.py              # multi-layer GCN
│   ├── attention.py        # contextual attention block and mean pooling
│   ├── scorer.py           # edge anomaly scorer
│   ├── static_gcn_baseline.py      # static GCN baseline
│   └── temporal_no_attention.py    # temporal GCN without attention
├── training/
│   ├── trainer.py          # snapshot-by-snapshot training loop
│   ├── evaluation.py       # evaluation loop, ROC-AUC, PR-AUC
│   ├── losses.py           # pairwise margin loss
│   └── negative_sampling.py        # Bernoulli sampler and selective filter
├── experiments/
│   └── run_experiment.py   # main entry point
└── utils/
    ├── graph.py            # adjacency matrix normalization
    ├── seed.py             # global seed setter
    └── io_utils.py         # JSON and CSV saving
results/                    # all experiment outputs (JSON + CSV)
```

---

## Setup

Python 3.10+ required. Install dependencies:

```bash
pip install torch numpy pandas scikit-learn
```

Download the CICIDS2017 CSV files from the [UNB dataset page](https://www.unb.ca/cic/datasets/ids-2017.html) and place them in a folder called `CICIDS2017/` in the project root.

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

`--model_name` accepts `addgraph`, `static_gcn`, or `temporal_no_attention`.

To enable selective negative sampling, omit the two `--disable` flags.

---

## Output Files

Each run saves to `--output_dir/<model_name>/seed_<seed>/`:

| File | Contents |
|------|----------|
| `config.json` | Full run configuration |
| `dataset_stats.json` | Per-snapshot edge and label breakdown |
| `summary.json` | Overall ROC-AUC, PR-AUC, score trend summary |
| `training_history.csv` | Per-epoch loss |
| `per_snapshot_metrics.csv` | Per-snapshot AUC and score statistics |

Pre-computed results for all experimental settings are included in `results/`.

---

## Experimental Settings

| Setting | Files | Seeds | Window |
|---------|-------|-------|--------|
| Setting A (primary ablation) | DDoS, PortScan, WebAttacks | 42, 43 | 2 |
| Setting B (attention check) | DDoS, PortScan, WebAttacks | 42, 43 | 1 |
| 4-file Validation | + Infiltration | 42 | 2 |
| Selective Sampling On/Off | DDoS, PortScan | 42 | 2 |

---

## Key Results

Temporal modeling consistently improves ROC-AUC over the static GCN baseline across all settings and seeds. The attention mechanism does not show a clear consistent advantage over mean pooling on this dataset. PR-AUC is dominated by edge weight on high-volume attacks like DDoS and PortScan, making it an unreliable architectural discriminator in this setting. See the report for full analysis.

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd

# Edge format: (u, v, weight, label)
# label: 0 = benign, 1 = attack/anomalous
Edge = Tuple[int, int, float, int]
ScoredEdge = Tuple[int, int, float]


@dataclass
class SnapshotBatch:
    edges: List[Edge]
    start_time: pd.Timestamp | None = None
    end_time: pd.Timestamp | None = None

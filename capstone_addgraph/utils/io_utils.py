import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def save_csv(path: str | Path, rows) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)

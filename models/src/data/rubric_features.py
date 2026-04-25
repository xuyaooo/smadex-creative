"""
Loads precomputed LLM rubric scores from parquet and aligns them to creative_ids.

Used at training time AND at inference time — but both just read the cached
parquet. No LLM call ever happens at runtime.
"""
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.training.openrouter_rubric import RUBRIC_NAMES


def load_rubric(parquet_path: Path) -> pd.DataFrame | None:
    """Returns a DataFrame indexed by creative_id with columns RUBRIC_NAMES, or None if not extracted yet."""
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        return None
    df = pd.read_parquet(parquet_path)
    if "creative_id" not in df.columns:
        return None
    df = df.set_index("creative_id")
    cols = [c for c in RUBRIC_NAMES if c in df.columns]
    return df[cols]


def align_rubric(
    parquet_path: Path, creative_ids: List[int]
) -> Tuple[np.ndarray, List[str]]:
    """Returns (n_creatives, n_rubric) feature matrix aligned to `creative_ids`.
    Missing rows are zero-filled — model still works without rubric extracted."""
    rubric_df = load_rubric(parquet_path)
    if rubric_df is None or rubric_df.empty:
        return np.zeros((len(creative_ids), 0), dtype=np.float32), []

    cols = list(rubric_df.columns)
    out = np.zeros((len(creative_ids), len(cols)), dtype=np.float32)
    cid_to_row = {int(cid): i for i, cid in enumerate(rubric_df.index)}
    for i, cid in enumerate(creative_ids):
        j = cid_to_row.get(int(cid))
        if j is not None:
            out[i] = rubric_df.iloc[j].fillna(0).values.astype(np.float32)
    return out, cols

"""Small, shared data-loading helpers with strict competition-data checks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import ID_COLUMN, MAX_TARGET_HOURS, TARGET_COLUMN


def load_training_data(path: str | Path) -> tuple[pd.DataFrame, np.ndarray]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Training file not found: {path}")

    frame = pd.read_csv(path, dtype={ID_COLUMN: "string"})
    if TARGET_COLUMN not in frame.columns:
        raise ValueError(f"Training file must contain {TARGET_COLUMN!r}.")
    if ID_COLUMN not in frame.columns:
        raise ValueError(f"Training file must contain {ID_COLUMN!r}.")

    target = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    valid = target.notna() & np.isfinite(target) & (target >= 0)
    frame = frame.loc[valid].reset_index(drop=True)
    target = target.loc[valid].to_numpy(dtype=float)

    if len(frame) < 100:
        raise ValueError(f"Only {len(frame)} valid training rows were found.")
    if frame[ID_COLUMN].isna().any() or frame[ID_COLUMN].astype(str).str.strip().eq("").any():
        raise ValueError("Training data contains a missing or blank project_id.")
    if frame[ID_COLUMN].astype(str).duplicated().any():
        raise ValueError("Training data contains duplicate project_id values.")
    if not np.isfinite(target).all():
        raise ValueError("Training target contains invalid numeric values.")
    if (target > MAX_TARGET_HOURS).any():
        raise ValueError(
            f"Training target exceeds the official {MAX_TARGET_HOURS}-hour cap."
        )
    return frame, target
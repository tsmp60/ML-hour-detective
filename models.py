"""CPU-friendly model factories and RMSLE helpers."""
from __future__ import annotations

from typing import Iterable

import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from config import (
    DEFAULT_LIGHTGBM_LEARNING_RATE,
    DEFAULT_LIGHTGBM_MAX_BIN,
    DEFAULT_LIGHTGBM_MIN_CHILD_SAMPLES,
    DEFAULT_LIGHTGBM_NUM_LEAVES,
    DEFAULT_LIGHTGBM_REG_ALPHA,
    DEFAULT_LIGHTGBM_REG_LAMBDA,
    MAX_TARGET_HOURS,
    MAX_THREADS,
)


def build_catboost(iterations: int, seed: int) -> CatBoostRegressor:
    """Create the selected CatBoost regressor for the log-hours target."""
    return CatBoostRegressor(
        iterations=max(100, int(iterations)),
        depth=6,
        learning_rate=0.025,
        loss_function="RMSE",
        eval_metric="RMSE",
        l2_leaf_reg=10.0,
        random_strength=0.7,
        random_seed=int(seed),
        thread_count=max(1, int(MAX_THREADS)),
        allow_writing_files=False,
        verbose=False,
    )


def build_lightgbm(
    iterations: int,
    seed: int,
    *,
    num_leaves: int = DEFAULT_LIGHTGBM_NUM_LEAVES,
    min_child_samples: int = DEFAULT_LIGHTGBM_MIN_CHILD_SAMPLES,
    learning_rate: float = DEFAULT_LIGHTGBM_LEARNING_RATE,
    reg_lambda: float = DEFAULT_LIGHTGBM_REG_LAMBDA,
    reg_alpha: float = DEFAULT_LIGHTGBM_REG_ALPHA,
    max_bin: int = DEFAULT_LIGHTGBM_MAX_BIN,
) -> LGBMRegressor:
    """Create a deterministic, regularized LightGBM CPU regressor."""
    seed = int(seed)
    return LGBMRegressor(
        objective="regression",
        n_estimators=max(100, int(iterations)),
        learning_rate=float(learning_rate),
        num_leaves=max(3, int(num_leaves)),
        max_depth=-1,
        min_child_samples=max(5, int(min_child_samples)),
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=float(reg_lambda),
        reg_alpha=float(reg_alpha),
        max_bin=max(31, int(max_bin)),
        random_state=seed,
        bagging_seed=seed,
        feature_fraction_seed=seed,
        data_random_seed=seed,
        deterministic=True,
        force_col_wise=True,
        n_jobs=max(1, int(MAX_THREADS)),
        verbosity=-1,
    )


def rmsle_from_log(actual_log: Iterable[float], predicted_log: Iterable[float]) -> float:
    """Compute RMSLE when inputs are already in log1p-hours space."""
    actual = np.asarray(actual_log, dtype=float)
    predicted = np.asarray(predicted_log, dtype=float)
    predicted = np.clip(predicted, 0.0, np.log1p(MAX_TARGET_HOURS))
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def log_prediction_to_hours(predicted_log: Iterable[float]) -> np.ndarray:
    """Convert log1p predictions to valid, capped hour values."""
    predicted = np.asarray(predicted_log, dtype=float)
    predicted = np.clip(predicted, 0.0, np.log1p(MAX_TARGET_HOURS))
    hours = np.expm1(predicted)
    return np.clip(hours, 0.0, MAX_TARGET_HOURS)
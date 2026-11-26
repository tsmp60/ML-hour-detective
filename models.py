"""Model factories and metric helpers."""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from config import (
    CATBOOST_DEFAULT_ITERATIONS,
    LIGHTGBM_DEFAULT_ITERATIONS,
    MAX_THREADS,
    RANDOM_SEED,
)


def thread_count() -> int:
    return max(1, min(MAX_THREADS, os.cpu_count() or 1))


def build_catboost(
    iterations: int = CATBOOST_DEFAULT_ITERATIONS, seed: int = RANDOM_SEED
) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=max(100, int(iterations)),
        depth=6,
        learning_rate=0.03,
        loss_function="RMSE",
        eval_metric="RMSE",
        l2_leaf_reg=8.0,
        random_strength=0.7,
        random_seed=int(seed),
        thread_count=thread_count(),
        allow_writing_files=False,
        verbose=False,
    )


def build_lightgbm(
    iterations: int = LIGHTGBM_DEFAULT_ITERATIONS, seed: int = RANDOM_SEED
) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        n_estimators=max(100, int(iterations)),
        learning_rate=0.02,
        num_leaves=15,
        max_depth=-1,
        min_child_samples=25,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        reg_alpha=0.1,
        random_state=int(seed),
        n_jobs=thread_count(),
        verbosity=-1,
    )


def rmsle_from_log(actual_log: Iterable[float], predicted_log: Iterable[float]) -> float:
    """RMSLE when the model already predicts log1p(hours)."""
    actual = np.asarray(actual_log, dtype=float)
    predicted = np.asarray(predicted_log, dtype=float)
    predicted = np.maximum(predicted, 0.0)  # hours cannot be negative
    return float(np.sqrt(np.mean(np.square(actual - predicted))))


def log_prediction_to_hours(predicted_log: Iterable[float]) -> np.ndarray:
    predicted = np.asarray(predicted_log, dtype=float)
    predicted = np.maximum(predicted, 0.0)
    hours = np.expm1(predicted)
    return np.maximum(hours, 0.0)
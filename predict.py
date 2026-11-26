"""Haggle submission entry point: train an ensemble and write predictions.csv."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CATBOOST_DEFAULT_ITERATIONS,
    CATBOOST_DEFAULT_WEIGHT,
    ID_COLUMN,
    LIGHTGBM_DEFAULT_ITERATIONS,
    LIGHTGBM_DEFAULT_WEIGHT,
    RANDOM_SEED,
    TARGET_COLUMN,
)
from features import make_train_test_features, to_lightgbm_categories
from models import build_catboost, build_lightgbm, log_prediction_to_hours
from train import load_training_data


def load_model_config() -> dict:
    path = Path("model_config.json")
    if not path.exists():
        return {
            "catboost_iterations": CATBOOST_DEFAULT_ITERATIONS,
            "lightgbm_iterations": LIGHTGBM_DEFAULT_ITERATIONS,
            "catboost_weight": CATBOOST_DEFAULT_WEIGHT,
            "lightgbm_weight": LIGHTGBM_DEFAULT_WEIGHT,
            "random_seed": RANDOM_SEED,
        }
    try:
        config = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read model_config.json: {exc}") from exc
    return config


def main() -> None:
    train_path = Path("train.csv")
    test_path = Path("test.csv")
    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            "Expected train.csv and test.csv beside predict.py. "
            "Haggle provides both files when it runs a submission."
        )

    train_frame, target_hours = load_training_data(train_path)
    test_frame = pd.read_csv(test_path)
    if ID_COLUMN not in test_frame.columns:
        raise ValueError(f"test.csv is missing {ID_COLUMN!r}.")
    if len(test_frame) == 0:
        raise ValueError("test.csv contains no rows.")

    x_train, x_test, categorical_columns = make_train_test_features(
        train_frame, test_frame
    )
    target_log = np.log1p(target_hours)
    settings = load_model_config()
    seed = int(settings.get("random_seed", RANDOM_SEED))
    cat_iterations = int(settings.get("catboost_iterations", CATBOOST_DEFAULT_ITERATIONS))
    lgb_iterations = int(settings.get("lightgbm_iterations", LIGHTGBM_DEFAULT_ITERATIONS))
    cat_weight = float(settings.get("catboost_weight", CATBOOST_DEFAULT_WEIGHT))
    lgb_weight = float(settings.get("lightgbm_weight", LIGHTGBM_DEFAULT_WEIGHT))

    if cat_weight < 0 or lgb_weight < 0 or not np.isclose(cat_weight + lgb_weight, 1.0):
        raise ValueError("model_config.json blend weights must be non-negative and sum to 1.")

    print(f"Training ensemble on {len(x_train):,} rows and {x_train.shape[1]} features...")
    cat_model = build_catboost(cat_iterations, seed)
    cat_model.fit(
        x_train,
        target_log,
        cat_features=categorical_columns,
        verbose=False,
    )
    cat_log = np.maximum(cat_model.predict(x_test), 0.0)

    lgb_train, lgb_test = to_lightgbm_categories(
        x_train, x_test, categorical_columns
    )
    lgb_model = build_lightgbm(lgb_iterations, seed)
    lgb_model.fit(
        lgb_train,
        target_log,
        categorical_feature=categorical_columns,
    )
    lgb_log = np.maximum(lgb_model.predict(lgb_test), 0.0)

    ensemble_log = cat_weight * cat_log + lgb_weight * lgb_log
    predictions = log_prediction_to_hours(ensemble_log)
    if len(predictions) != len(test_frame) or not np.isfinite(predictions).all():
        raise RuntimeError("Model produced missing, non-finite, or incomplete predictions.")

    output = pd.DataFrame(
        {
            ID_COLUMN: test_frame[ID_COLUMN].astype(str).to_numpy(),
            TARGET_COLUMN: predictions.astype(float),
        }
    )
    output.to_csv("predictions.csv", index=False)
    print(f"Created predictions.csv with {len(output):,} rows.")
    print(f"Ensemble weights: CatBoost={cat_weight:.2f}, LightGBM={lgb_weight:.2f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"predict.py failed: {exc}", file=sys.stderr)
        raise
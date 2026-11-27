"""Haggle entry point: train on train.csv and write predictions.csv."""
from __future__ import annotations

import gc
import json
import os
import sys
from pathlib import Path

# Keep BLAS/OpenMP libraries within the competition's CPU budget. These are
# set before NumPy, pandas, CatBoost and LightGBM are imported.
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd

from config import (
    DEFAULT_CATBOOST_ITERATIONS,
    DEFAULT_CATBOOST_WEIGHT,
    DEFAULT_LIGHTGBM_ITERATIONS,
    DEFAULT_LIGHTGBM_SEEDS,
    DEFAULT_LIGHTGBM_WEIGHT,
    DEFAULT_LIGHTGBM_NUM_LEAVES,
    DEFAULT_LIGHTGBM_MIN_CHILD_SAMPLES,
    DEFAULT_LIGHTGBM_LEARNING_RATE,
    DEFAULT_LIGHTGBM_REG_LAMBDA,
    DEFAULT_LIGHTGBM_REG_ALPHA,
    DEFAULT_LIGHTGBM_MAX_BIN,
    ID_COLUMN,
    MAX_TARGET_HOURS,
    RANDOM_SEED,
    ROOT,
    TARGET_COLUMN,
)
from data_utils import load_training_data
from features import make_train_test_features, to_lightgbm_categories
from models import build_catboost, build_lightgbm, log_prediction_to_hours


def _load_settings() -> dict:
    path = ROOT / "model_config.json"
    defaults = {
        "catboost_iterations": DEFAULT_CATBOOST_ITERATIONS,
        "lightgbm_iterations": DEFAULT_LIGHTGBM_ITERATIONS,
        "catboost_weight": DEFAULT_CATBOOST_WEIGHT,
        "lightgbm_weight": DEFAULT_LIGHTGBM_WEIGHT,
        "lightgbm_seeds": list(DEFAULT_LIGHTGBM_SEEDS),
        "lightgbm_num_leaves": DEFAULT_LIGHTGBM_NUM_LEAVES,
        "lightgbm_min_child_samples": DEFAULT_LIGHTGBM_MIN_CHILD_SAMPLES,
        "lightgbm_learning_rate": DEFAULT_LIGHTGBM_LEARNING_RATE,
        "lightgbm_reg_lambda": DEFAULT_LIGHTGBM_REG_LAMBDA,
        "lightgbm_reg_alpha": DEFAULT_LIGHTGBM_REG_ALPHA,
        "lightgbm_max_bin": DEFAULT_LIGHTGBM_MAX_BIN,
        "catboost_seed": RANDOM_SEED,
        "target_cap_hours": MAX_TARGET_HOURS,
    }
    if not path.is_file():
        return defaults
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read model_config.json: {exc}") from exc
    if not isinstance(saved, dict):
        raise ValueError("model_config.json must contain a JSON object.")
    defaults.update(saved)

    positive_integer_keys = (
        "catboost_iterations",
        "lightgbm_iterations",
        "lightgbm_num_leaves",
        "lightgbm_min_child_samples",
        "lightgbm_max_bin",
    )
    for key in positive_integer_keys:
        value = defaults.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"model_config.json setting {key!r} must be a positive integer.")

    cat_seed = defaults.get("catboost_seed")
    if isinstance(cat_seed, bool) or not isinstance(cat_seed, int):
        raise ValueError("model_config.json catboost_seed must be an integer.")

    seed_values = defaults.get("lightgbm_seeds")
    if (
        not isinstance(seed_values, list)
        or not seed_values
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seed_values)
    ):
        raise ValueError("model_config.json must contain a non-empty list of integer LightGBM seeds.")

    for key in ("catboost_weight", "lightgbm_weight"):
        value = defaults.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"model_config.json setting {key!r} must be a finite number.")

    for key in ("lightgbm_learning_rate", "lightgbm_reg_lambda", "lightgbm_reg_alpha"):
        value = defaults.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"model_config.json setting {key!r} must be a finite number.")
    if defaults["lightgbm_learning_rate"] <= 0:
        raise ValueError("LightGBM learning rate must be positive.")
    if defaults["lightgbm_reg_lambda"] < 0 or defaults["lightgbm_reg_alpha"] < 0:
        raise ValueError("LightGBM regularization values cannot be negative.")

    cap = defaults.get("target_cap_hours")
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not np.isfinite(cap) or cap <= 0:
        raise ValueError("model_config.json target_cap_hours must be a positive finite number.")
    defaults["target_cap_hours"] = min(MAX_TARGET_HOURS, float(cap))
    return defaults


def main() -> None:
    train_path = ROOT / "train.csv"
    test_path = ROOT / "test.csv"
    output_path = ROOT / "predictions.csv"
    # Remove any stale prediction file so a failed rerun cannot leave an old
    # submission result that looks current to an external runner.
    if output_path.exists():
        output_path.unlink()

    if not test_path.is_file():
        raise FileNotFoundError("Haggle must provide test.csv beside predict.py.")

    train_frame, target_hours = load_training_data(train_path)
    test_frame = pd.read_csv(test_path, dtype={ID_COLUMN: "string"})

    if ID_COLUMN not in test_frame.columns:
        raise ValueError(f"test.csv is missing the required {ID_COLUMN!r} column.")
    if test_frame.empty:
        raise ValueError("test.csv contains no projects.")
    if test_frame[ID_COLUMN].isna().any() or test_frame[ID_COLUMN].astype(str).str.strip().eq("").any():
        raise ValueError("test.csv contains a missing or blank project_id.")
    if test_frame[ID_COLUMN].astype(str).duplicated().any():
        raise ValueError("test.csv contains duplicate project_id values.")

    expected_test_columns = set(train_frame.columns) - {
        TARGET_COLUMN, "log_hours", "hours_bucket"
    }
    actual_test_columns = set(test_frame.columns)
    missing_columns = sorted(expected_test_columns - actual_test_columns)
    unexpected_columns = sorted(actual_test_columns - expected_test_columns)
    if missing_columns or unexpected_columns:
        raise ValueError(
            "test.csv schema does not match train.csv. "
            f"Missing: {missing_columns[:8]}; unexpected: {unexpected_columns[:8]}"
        )

    x_train, x_test, categorical_columns = make_train_test_features(
        train_frame, test_frame
    )
    target_log = np.log1p(target_hours)
    settings = _load_settings()

    cat_weight = float(settings["catboost_weight"])
    lgb_weight = float(settings["lightgbm_weight"])
    if (
        cat_weight < 0
        or lgb_weight < 0
        or not np.isclose(cat_weight + lgb_weight, 1.0, atol=1e-6)
    ):
        raise ValueError("Blend weights in model_config.json must sum to 1.")

    target_cap = float(settings.get("target_cap_hours", MAX_TARGET_HOURS))
    target_cap = min(MAX_TARGET_HOURS, max(0.1, target_cap))

    # CatBoost receives original categorical strings.
    cat_model = build_catboost(
        int(settings["catboost_iterations"]),
        int(settings.get("catboost_seed", RANDOM_SEED)),
    )
    cat_model.fit(
        x_train,
        target_log,
        cat_features=categorical_columns,
        verbose=False,
    )
    cat_prediction = np.clip(
        cat_model.predict(x_test), 0.0, np.log1p(target_cap)
    )
    del cat_model
    gc.collect()

    # LightGBM averages several small CPU models for a more stable estimate.
    lgb_train, lgb_test = to_lightgbm_categories(
        x_train, x_test, categorical_columns
    )
    seeds = settings["lightgbm_seeds"]

    lgb_predictions = []
    for seed in seeds:
        lgb_model = build_lightgbm(
            int(settings["lightgbm_iterations"]),
            int(seed),
            num_leaves=int(settings.get("lightgbm_num_leaves", DEFAULT_LIGHTGBM_NUM_LEAVES)),
            min_child_samples=int(settings.get("lightgbm_min_child_samples", DEFAULT_LIGHTGBM_MIN_CHILD_SAMPLES)),
            learning_rate=float(settings.get("lightgbm_learning_rate", DEFAULT_LIGHTGBM_LEARNING_RATE)),
            reg_lambda=float(settings.get("lightgbm_reg_lambda", DEFAULT_LIGHTGBM_REG_LAMBDA)),
            reg_alpha=float(settings.get("lightgbm_reg_alpha", DEFAULT_LIGHTGBM_REG_ALPHA)),
            max_bin=int(settings.get("lightgbm_max_bin", DEFAULT_LIGHTGBM_MAX_BIN)),
        )
        lgb_model.fit(
            lgb_train,
            target_log,
            categorical_feature=categorical_columns,
        )
        lgb_predictions.append(
            np.clip(lgb_model.predict(lgb_test), 0.0, np.log1p(target_cap))
        )
        del lgb_model
    lgb_prediction = np.mean(np.vstack(lgb_predictions), axis=0)

    blended_log = cat_weight * cat_prediction + lgb_weight * lgb_prediction
    predictions = log_prediction_to_hours(blended_log)
    predictions = np.clip(predictions, 0.0, target_cap)

    if len(predictions) != len(test_frame):
        raise RuntimeError("The model did not create one prediction per test row.")
    if not np.isfinite(predictions).all() or (predictions < 0).any():
        raise RuntimeError("The model produced invalid predictions.")

    output = pd.DataFrame(
        {
            ID_COLUMN: test_frame[ID_COLUMN].astype(str).to_numpy(),
            TARGET_COLUMN: predictions.astype(float),
        }
    )
    # Write atomically so a partial file is never mistaken for a valid result.
    temporary_output = output_path.with_suffix(".tmp")
    output.to_csv(temporary_output, index=False)
    temporary_output.replace(output_path)
    print("ML prediction run: COMPLETE")
    print(f"Predictions written: {output_path.name} ({len(output):,} rows)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ML prediction failed: {exc}", file=sys.stderr)
        raise
"""Run an isolated, group-aware holdout through the exact Haggle predictor.

This test does not run the expensive cross-validation training script first.
It copies the selected model_config.json, creates a temporary train/test split,
then exercises predict.py exactly as Haggle does.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from config import ID_COLUMN, MAX_TARGET_HOURS, ROOT, TARGET_COLUMN
from data_utils import load_training_data
from features import make_features, make_train_test_features


SCRIPT_FILES = (
    "config.py",
    "data_utils.py",
    "features.py",
    "models.py",
    "predict.py",
    "model_config.json",
)


def _run_predict(command: list[str], cwd: Path, env: dict[str, str]) -> tuple[str, float]:
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Haggle-style prediction exceeded the 10-minute limit.") from exc
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise RuntimeError(
            "Haggle-style prediction failed.\n"
            f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
        )
    return result.stdout, elapsed


def main() -> None:
    source_train = ROOT / "train.csv"
    model_config = ROOT / "model_config.json"
    if not model_config.is_file():
        raise FileNotFoundError("model_config.json is missing. Restore the supplied model config.")

    frame, target = load_training_data(source_train)
    features, _ = make_features(frame)
    groups = pd.util.hash_pandas_object(features, index=False).to_numpy(dtype="uint64")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=2026)
    train_idx, test_idx = next(splitter.split(features, target, groups=groups))

    training_part = frame.iloc[train_idx].copy()
    test_part = frame.iloc[test_idx].drop(
        columns=[c for c in (TARGET_COLUMN, "log_hours", "hours_bucket") if c in frame],
        errors="ignore",
    )

    # Check schema alignment under two common edge cases: a category unseen
    # during training and a numeric feature that is entirely missing in test.
    edge_test = test_part.head(2).copy()
    if "author_prior_projects" in edge_test.columns:
        edge_test.loc[:, "author_prior_projects"] = "__UNSEEN_TEST_CATEGORY__"
    if "page_weight_kb" in edge_test.columns:
        edge_test.loc[:, "page_weight_kb"] = np.nan
    edge_train_x, edge_test_x, edge_categories = make_train_test_features(
        training_part, edge_test
    )
    if list(edge_train_x.columns) != list(edge_test_x.columns):
        raise RuntimeError("Train/test feature columns do not align for edge-case input.")
    if "author_prior_projects" in edge_categories and not edge_test_x[
        "author_prior_projects"
    ].eq("__UNSEEN_TEST_CATEGORY__").all():
        raise RuntimeError("Unseen categorical values were not preserved safely.")

    with tempfile.TemporaryDirectory(prefix="ml_haggle_check_") as temp:
        temp_dir = Path(temp)
        for filename in SCRIPT_FILES:
            shutil.copy2(ROOT / filename, temp_dir / filename)
        training_part.to_csv(temp_dir / "train.csv", index=False)
        test_part.to_csv(temp_dir / "test.csv", index=False)
        # Confirm predict.py clears a stale output before starting a fresh run.
        (temp_dir / "predictions.csv").write_text("stale output\n", encoding="utf-8")

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "2"
        env["OPENBLAS_NUM_THREADS"] = "2"
        env["MKL_NUM_THREADS"] = "2"
        env["NUMEXPR_NUM_THREADS"] = "2"

        stdout, elapsed = _run_predict(
            [sys.executable, str(temp_dir / "predict.py")], temp_dir, env
        )
        output_path = temp_dir / "predictions.csv"
        if not output_path.is_file():
            raise RuntimeError("predict.py finished but did not create predictions.csv.")
        output = pd.read_csv(output_path, dtype={ID_COLUMN: "string"})

        expected_columns = [ID_COLUMN, TARGET_COLUMN]
        if list(output.columns) != expected_columns:
            raise RuntimeError(f"Wrong output columns: {list(output.columns)}")
        if len(output) != len(test_part):
            raise RuntimeError("Prediction row count does not match the held-out test set.")
        if output[ID_COLUMN].astype(str).tolist() != test_part[ID_COLUMN].astype(str).tolist():
            raise RuntimeError("Prediction IDs are missing, reordered, or incorrect.")
        if output[ID_COLUMN].astype(str).duplicated().any():
            raise RuntimeError("Output contains duplicate project IDs.")

        predicted = pd.to_numeric(output[TARGET_COLUMN], errors="coerce").to_numpy()
        if (
            not np.isfinite(predicted).all()
            or (predicted < 0).any()
            or (predicted > MAX_TARGET_HOURS).any()
        ):
            raise RuntimeError("Predictions contain invalid or out-of-range values.")
        actual = target[test_idx]
        score = float(np.sqrt(np.mean((np.log1p(actual) - np.log1p(predicted)) ** 2)))

    print()
    print("╭──────────────────────────────────────────────╮")
    print("│             ML · LOCAL SHIP CHECK             │")
    print("╰──────────────────────────────────────────────╯")
    print("  Group-aware holdout split      PASS")
    print("  Haggle predictor execution     PASS")
    print("  Prediction file format         PASS")
    print("  Project IDs and row order      PASS")
    print("  Prediction value range         PASS")
    print("  Train/test edge-case schema    PASS")
    print("  Stale output protection        PASS")
    print()
    print(f"  Predictor runtime               {elapsed:,.1f} seconds")
    print(f"  Local holdout RMSLE             {score:.4f}")
    print("  Holdout score is diagnostic only, not a Haggle score.")
    print("  Runtime may differ on Haggle's hardware.")
    print()


if __name__ == "__main__":
    main()
"""Train and validate the ML Hour Detective ensemble.

The normal console output is deliberately clean for project-progress screenshots.
Use ``python train.py --details`` for fold-level diagnostics.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from config import (
    CV_FOLDS,
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
)
from data_utils import load_training_data
from features import make_features, to_lightgbm_categories
from models import build_catboost, build_lightgbm, log_prediction_to_hours, rmsle_from_log


def _make_groups(features: pd.DataFrame) -> np.ndarray:
    """Keep projects with identical model inputs in the same validation fold."""
    return pd.util.hash_pandas_object(features, index=False).to_numpy(dtype="uint64")


def _write_validation_report(results: dict, path: Path) -> None:
    fold_lines = "\n".join(
        f"- Fold {fold['fold']}: CatBoost {fold['catboost_rmsle']:.5f}; "
        f"three-seed LightGBM {fold['lightgbm_rmsle']:.5f}"
        for fold in results["folds"]
    )
    report = f"""# ML — Hour Detective validation report

## Validation results

- **OOF blended RMSLE (development diagnostic):** {results['ensemble_rmsle']:.5f}
- **LightGBM ensemble RMSLE:** {results['lightgbm_rmsle']:.5f}
- **CatBoost RMSLE:** {results['catboost_rmsle']:.5f}
- **Median baseline RMSLE:** {results['baseline_rmsle']:.5f}
- **Deployed blend weight:** CatBoost {results['catboost_weight']:.3f}; LightGBM {results['lightgbm_weight']:.3f}

Lower RMSLE is better. These are local cross-validation results, not a Haggle
leaderboard score or a guarantee about the hidden 500 projects. The 0.25/0.75 blend weight was selected during model development using the
supplied training data. Therefore the blended OOF score is a model-selection
diagnostic, not a fully independent estimate. The individual model OOF scores
are shown separately, and Haggle's hidden evaluation remains the final test.

## Validation design

- {CV_FOLDS}-fold `StratifiedGroupKFold` on the supplied training projects.
- Exact duplicate model-input rows remain together in a fold.
- Fixed model iteration counts; no scored fold is used for early stopping.
- The same three LightGBM seeds used by `predict.py` are averaged in every fold.
- The CatBoost validation seed matches the final predictor seed.
- A fixed 0.25 CatBoost / 0.75 LightGBM blend is used in both validation and
  the final predictor. This weight was selected during development on this
  training set, so the blended OOF score is a development diagnostic.
- The target is `log1p(hours_spent)`, which matches RMSLE for non-negative values.
- `project_id`, `hours_spent`, `log_hours`, and `hours_bucket` are excluded from
  model features. The last two are label-derived columns in the training CSV.

### Fold-level scores

{fold_lines}

## Model design

- CatBoost handles categorical project metadata and non-linear interactions.
- LightGBM is the main model; three deterministic runs are averaged.
- Ordered numeric views are added for the author-history and account-age bands,
  while the original categories are retained.
- A fixed 0.25 CatBoost / 0.75 LightGBM blend combines the two model families.
- Predictions are clipped to the documented 140.7-hour target cap.
- `predict.py` retrains from the supplied `train.csv` and does not need internet
  access or pre-trained model binaries.

## Limitations

The training set contains 4,500 rows and has a strongly right-skewed target.
The validation folds are random grouped splits, while the hidden 500-project
set may differ in distribution. Hyperparameter choices were explored using the
supplied training data, so this is a development estimate rather than an
untouched benchmark. Only Haggle's hidden score establishes competition
performance.

No external datasets, scraping, owner lookups, competitor code, or competitor
predictions are used.
"""
    path.write_text(report, encoding="utf-8")


def _print_summary(results: dict, seconds: float) -> None:
    minutes, remaining = divmod(int(seconds), 60)
    print()
    print("╭──────────────────────────────────────────────╮")
    print("│             ML · HOUR DETECTIVE               │")
    print("│           VALIDATION COMPLETE                │")
    print("╰──────────────────────────────────────────────╯")
    print("  Data preparation               COMPLETE")
    print("  Model training                 COMPLETE")
    print(f"  {CV_FOLDS}-fold validation             COMPLETE")
    print()
    print("  LOCAL VALIDATION")
    print("  ────────────────────────────────────────────")
    print(f"  OOF blended RMSLE               {results['ensemble_rmsle']:.4f}")
    print(f"  Reference baseline             {results['baseline_rmsle']:.4f}")
    print("  Lower is better.")
    print()
    print(f"  Projects evaluated             {results['rows']:,}")
    print(f"  Training time                  {minutes:02d}:{remaining:02d}")
    print("  Haggle prediction script        READY")
    print()
    print("  Local validation is not a leaderboard result.")
    print("  Detailed diagnostics are saved locally.")
    print("  Next step: python test_submission.py")
    print()


def _print_details(results: dict) -> None:
    print("Detailed diagnostics")
    print(f"  Features: {results['features']}")
    print(f"  Categorical features: {results['categorical_features']}")
    print(f"  CatBoost RMSLE: {results['catboost_rmsle']:.5f}")
    print(f"  LightGBM ensemble RMSLE: {results['lightgbm_rmsle']:.5f}")
    print(f"  OOF blended RMSLE (development): {results['ensemble_rmsle']:.5f}")
    print("  Blend weight was fixed at 0.25 / 0.75 during development.")
    print(f"  Final blend: CatBoost={results['catboost_weight']:.3f}, "
          f"LightGBM={results['lightgbm_weight']:.3f}")
    print(f"  Iterations: CatBoost={results['catboost_iterations']}, "
          f"LightGBM={results['lightgbm_iterations']}")
    print(f"  LightGBM seeds: {results['lightgbm_seeds']}")
    for fold in results["folds"]:
        print(
            f"  Fold {fold['fold']}: rows={fold['rows']}, "
            f"CatBoost={fold['catboost_rmsle']:.5f}, "
            f"LightGBM={fold['lightgbm_rmsle']:.5f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ML Hour Detective model.")
    parser.add_argument("--details", action="store_true", help="show technical validation diagnostics")
    args = parser.parse_args()
    started = time.perf_counter()

    train_frame, target_hours = load_training_data(ROOT / "train.csv")
    if (target_hours > MAX_TARGET_HOURS).any():
        raise ValueError(f"Training target exceeds the official {MAX_TARGET_HOURS}-hour cap.")

    target_log = np.log1p(target_hours)
    features, categorical_columns = make_features(train_frame)
    strata = pd.qcut(
        pd.Series(target_log).rank(method="first"),
        q=8,
        labels=False,
        duplicates="drop",
    ).to_numpy()
    groups = _make_groups(features)
    splitter = StratifiedGroupKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_SEED,
    )

    oof_cat = np.full(len(train_frame), np.nan, dtype=float)
    oof_lgb = np.full(len(train_frame), np.nan, dtype=float)
    baseline_log = np.full(len(train_frame), np.nan, dtype=float)
    cat_importances: list[np.ndarray] = []
    lgb_importances: list[np.ndarray] = []
    fold_summaries: list[dict] = []

    print("ML training started", flush=True)
    print(f"  Preparing {CV_FOLDS} validation passes...", flush=True)

    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(features, strata, groups=groups), start=1
    ):
        print(f"  Validation pass {fold}/{CV_FOLDS}...", flush=True)
        x_train = features.iloc[train_idx]
        x_valid = features.iloc[valid_idx]
        y_train = target_log[train_idx]
        y_valid = target_log[valid_idx]
        baseline_log[valid_idx] = np.log1p(np.median(target_hours[train_idx]))

        # Same seed and fixed iterations as the final single CatBoost fit.
        cat_model = build_catboost(DEFAULT_CATBOOST_ITERATIONS, RANDOM_SEED)
        cat_model.fit(x_train, y_train, cat_features=categorical_columns, verbose=False)
        cat_prediction = np.clip(
            cat_model.predict(x_valid), 0.0, np.log1p(MAX_TARGET_HOURS)
        )
        oof_cat[valid_idx] = cat_prediction
        cat_importances.append(cat_model.get_feature_importance().astype(float))
        del cat_model
        gc.collect()

        # Match predict.py exactly: average the same configured LightGBM seeds.
        lgb_train, lgb_valid = to_lightgbm_categories(
            x_train, x_valid, categorical_columns
        )
        lgb_seed_predictions = []
        for seed in DEFAULT_LIGHTGBM_SEEDS:
            lgb_model = build_lightgbm(
                DEFAULT_LIGHTGBM_ITERATIONS,
                seed,
                num_leaves=DEFAULT_LIGHTGBM_NUM_LEAVES,
                min_child_samples=DEFAULT_LIGHTGBM_MIN_CHILD_SAMPLES,
                learning_rate=DEFAULT_LIGHTGBM_LEARNING_RATE,
                reg_lambda=DEFAULT_LIGHTGBM_REG_LAMBDA,
                reg_alpha=DEFAULT_LIGHTGBM_REG_ALPHA,
                max_bin=DEFAULT_LIGHTGBM_MAX_BIN,
            )
            lgb_model.fit(lgb_train, y_train, categorical_feature=categorical_columns)
            lgb_seed_predictions.append(
                np.clip(lgb_model.predict(lgb_valid), 0.0, np.log1p(MAX_TARGET_HOURS))
            )
            lgb_importances.append(lgb_model.feature_importances_.astype(float))
            del lgb_model
        lgb_prediction = np.mean(np.vstack(lgb_seed_predictions), axis=0)
        oof_lgb[valid_idx] = lgb_prediction
        del lgb_train, lgb_valid, lgb_seed_predictions
        gc.collect()

        fold_summaries.append(
            {
                "fold": fold,
                "rows": int(len(valid_idx)),
                "catboost_rmsle": rmsle_from_log(y_valid, cat_prediction),
                "lightgbm_rmsle": rmsle_from_log(y_valid, lgb_prediction),
            }
        )
        print(f"  Validation pass {fold}/{CV_FOLDS} complete.", flush=True)

    if not (
        np.isfinite(oof_cat).all()
        and np.isfinite(oof_lgb).all()
        and np.isfinite(baseline_log).all()
    ):
        raise RuntimeError("Validation did not produce one prediction per project.")

    cat_score = rmsle_from_log(target_log, oof_cat)
    lgb_score = rmsle_from_log(target_log, oof_lgb)
    baseline_score = rmsle_from_log(target_log, baseline_log)
    # The 25/75 blend is fixed from the development experiments. Keeping it
    # fixed avoids presenting a fold-adaptive meta-weight score as unbiased CV.
    final_cat_weight = float(DEFAULT_CATBOOST_WEIGHT)
    final_lgb_weight = float(DEFAULT_LIGHTGBM_WEIGHT)
    if not np.isclose(final_cat_weight + final_lgb_weight, 1.0, atol=1e-8):
        raise RuntimeError("Configured ensemble weights must sum to 1.")
    blended_oof = final_cat_weight * oof_cat + final_lgb_weight * oof_lgb
    ensemble_score = rmsle_from_log(target_log, blended_oof)

    model_config = {
        "catboost_iterations": int(DEFAULT_CATBOOST_ITERATIONS),
        "lightgbm_iterations": int(DEFAULT_LIGHTGBM_ITERATIONS),
        "lightgbm_num_leaves": int(DEFAULT_LIGHTGBM_NUM_LEAVES),
        "lightgbm_min_child_samples": int(DEFAULT_LIGHTGBM_MIN_CHILD_SAMPLES),
        "lightgbm_learning_rate": float(DEFAULT_LIGHTGBM_LEARNING_RATE),
        "lightgbm_reg_lambda": float(DEFAULT_LIGHTGBM_REG_LAMBDA),
        "lightgbm_reg_alpha": float(DEFAULT_LIGHTGBM_REG_ALPHA),
        "lightgbm_max_bin": int(DEFAULT_LIGHTGBM_MAX_BIN),
        "catboost_weight": final_cat_weight,
        "lightgbm_weight": final_lgb_weight,
        "lightgbm_seeds": [int(seed) for seed in DEFAULT_LIGHTGBM_SEEDS],
        "catboost_seed": int(RANDOM_SEED),
        "random_seed": int(RANDOM_SEED),
        "target_cap_hours": float(MAX_TARGET_HOURS),
        "target_transform": "log1p",
        "validation_folds": int(CV_FOLDS),
        "validation_method": f"{CV_FOLDS}_fold_stratified_group_cv_exact_duplicate_groups_fixed_development_blend",
    }
    results = {
        "rows": int(len(train_frame)),
        "features": int(features.shape[1]),
        "categorical_features": int(len(categorical_columns)),
        "baseline_rmsle": float(baseline_score),
        "catboost_rmsle": float(cat_score),
        "lightgbm_rmsle": float(lgb_score),
        "ensemble_rmsle": float(ensemble_score),
        "catboost_weight": float(final_cat_weight),
        "lightgbm_weight": float(final_lgb_weight),
        "catboost_iterations": int(DEFAULT_CATBOOST_ITERATIONS),
        "lightgbm_iterations": int(DEFAULT_LIGHTGBM_ITERATIONS),
        "lightgbm_seeds": [int(seed) for seed in DEFAULT_LIGHTGBM_SEEDS],
        "folds": fold_summaries,
    }

    (ROOT / "model_config.json").write_text(
        json.dumps(model_config, indent=2) + "\n", encoding="utf-8"
    )
    (ROOT / "validation_results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )

    cat_importance = np.mean(np.vstack(cat_importances), axis=0)
    lgb_importance = np.mean(np.vstack(lgb_importances), axis=0)
    cat_normalized = cat_importance / max(float(cat_importance.sum()), 1e-12)
    lgb_normalized = lgb_importance / max(float(lgb_importance.sum()), 1e-12)
    importance = pd.DataFrame(
        {
            "feature": features.columns,
            "catboost_importance": cat_importance,
            "lightgbm_importance": lgb_importance,
            "combined_importance": (
                final_cat_weight * cat_normalized + final_lgb_weight * lgb_normalized
            ),
        }
    ).sort_values("combined_importance", ascending=False)
    importance.to_csv(ROOT / "feature_importance.csv", index=False)

    oof_output = pd.DataFrame(
        {
            ID_COLUMN: train_frame[ID_COLUMN].astype(str),
            "actual_hours": target_hours,
            "catboost_predicted_hours": log_prediction_to_hours(oof_cat),
            "lightgbm_predicted_hours": log_prediction_to_hours(oof_lgb),
            "blended_oof_predicted_hours": log_prediction_to_hours(blended_oof),
        }
    )
    oof_output.to_csv(ROOT / "oof_predictions.csv", index=False)

    _write_validation_report(results, ROOT / "VALIDATION_REPORT.md")
    _print_summary(results, time.perf_counter() - started)
    if args.details:
        _print_details(results)


if __name__ == "__main__":
    main()
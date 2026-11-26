# ML — Haggle Hour Detective

A tabular regression system for predicting `hours_spent` from Hack Club project metadata. The model is an ensemble of CatBoost and LightGBM regressors trained on `log1p(hours_spent)`, which directly aligns the objective with RMSLE.

## Project structure

```text
ML/
├── train.csv                 # Local training data; intentionally git-ignored
├── config.py                 # Shared model/CV settings
├── features.py               # Leakage-safe preprocessing + engineered features
├── models.py                 # CatBoost/LightGBM factories and metric helpers
├── train.py                  # Group-aware cross-validation and model selection
├── predict.py                # Haggle entry point; writes predictions.csv
├── model_config.json         # Chosen iteration counts and ensemble weights
├── requirements.txt
└── README.md
```

## Important data rules

- `project_id` is an identifier, not a predictive feature.
- `hours_spent` is the target.
- `log_hours` and `hours_bucket` are derived from the target and are explicitly excluded to prevent label leakage.
- The training CSV is included in this local starter archive for convenience, but `*.csv` is ignored by Git. **Do not commit `train.csv` or any copy of the competition data to GitHub.**

## Setup (VS Code terminal)

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Train and validate

```bash
python train.py
```

The training script:

1. Creates interpretable ratios from commits, code structure, assets, and README features.
2. Keeps duplicate feature rows together using `StratifiedGroupKFold`.
3. Trains CatBoost and LightGBM on `log1p(hours_spent)`.
4. Reports out-of-fold RMSLE for each model and their ensemble.
5. Selects an ensemble blend from out-of-fold predictions and writes `model_config.json`.
6. Writes local analysis files: `validation_results.json`, `feature_importance.csv`, and `oof_predictions.csv`.

Use the out-of-fold score as the local estimate. It is not a guarantee of the hidden leaderboard score.

## Test the Haggle prediction interface locally

After `train.py` has completed, create a temporary test file from held-out-looking rows:

```bash
python - <<'PY'
import pandas as pd
train = pd.read_csv("train.csv")
test = train.drop(columns=["hours_spent", "log_hours", "hours_bucket"]).sample(25, random_state=17)
test.to_csv("test.csv", index=False)
PY

python predict.py
```

The script should create `predictions.csv` with exactly these columns:

```csv
project_id,hours_spent
535ce026f7,12.5
```

The example row is illustrative. Your actual predictions will differ.

## Haggle execution interface

Haggle's current first-ship guide says it places `train.csv` and hidden `test.csv` beside `predict.py`, runs `python predict.py` without internet, and reads `predictions.csv`. The required columns are `project_id,hours_spent`; the script must finish within 10 minutes. The guide says pandas, scikit-learn, XGBoost, LightGBM, and CatBoost are already installed. See <https://haggle.hackclub.com/guides/first-ship/>.

## Before shipping

- Run `python train.py` and save the actual validation results.
- Run the local `predict.py` smoke test and inspect `predictions.csv`.
- Push source code and `model_config.json`, not `train.csv`, `test.csv`, or generated prediction CSVs.
- Haggle's first-ship guide also requires a public repository, a playable web URL, a project description, a screenshot, and tracked Hackatime hours. See <https://haggle.hackclub.com/guides/first-ship/>.

## Validation snapshot from the uploaded dataset

The provided 4,500-row `train.csv` was run through the current 3-fold, duplicate-aware validation pipeline. That run produced:

| Model | Out-of-fold RMSLE |
|---|---:|
| Median baseline | 0.96672 |
| CatBoost | 0.67624 |
| LightGBM | 0.66470 |
| Weighted ensemble | **0.66454** |

The selected blend for this run is 10% CatBoost and 90% LightGBM. The blend weight was selected from these same out-of-fold predictions, so the result is a useful development measurement—not an independent test score, not a public leaderboard score, and not a guarantee on Haggle's hidden test set. Re-run `python train.py` after any feature or parameter changes.

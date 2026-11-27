# ML — Haggle Hour Detective

A CPU-friendly tabular regression project that predicts reviewer-approved project
hours from the official Hour Detective training data.

## Quick start

Run these commands from this folder in VS Code:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python train.py
```

The normal training display is designed to be easy to read in a project update.
It prints a short validation summary, not feature names, project IDs, or
row-level predictions. For technical details, run:

```bash
python train.py --details
```

Training writes or updates:

- `model_config.json` — settings used by the submission predictor.
- `validation_results.json` — metrics and fold summaries.
- `VALIDATION_REPORT.md` — readable validation report.
- `feature_importance.csv` and `oof_predictions.csv` — local diagnostics.

## Model design

The target is trained as `log1p(hours_spent)`. Squared error in that space is
the RMSLE objective used by the competition for non-negative target values.

The predictor uses two gradient-boosted tree families:

- CatBoost learns non-linear relationships and handles categorical metadata.
- LightGBM is the main model. Three deterministic seed runs are averaged.
- A small 0.25 CatBoost / 0.75 LightGBM blend combines their predictions.

Two ordered author fields have both their original category and an additional
numeric ordered representation: prior approved projects and account-age band.
This lets the trees use their natural ordering without discarding the original
category information.

## Validation

`python train.py` runs five-fold `StratifiedGroupKFold` validation. Exact duplicate
feature rows are kept in the same fold. The validation run uses the same model
families, iteration counts, LightGBM seeds, and fixed blend weight as `predict.py`.
No scored fold is used for early stopping. The blend weight was selected during
model development on this training set, so the blended OOF score is a development
diagnostic rather than an untouched estimate.

The default validation output reports RMSLE, for which lower is better. It is a
local estimate, not a Haggle leaderboard score. The hidden projects may have a
different distribution.

After training, run the end-to-end runner check:

```bash
python test_submission.py
```

This creates a separate group-aware holdout, copies the selected model settings,
executes the actual `predict.py` entry point, and checks the output schema, IDs,
prediction range, runtime and local holdout RMSLE. It does not replace or modify
your original `train.csv` or rerun the five-fold validation.

## Project structure

```text
ML hour detective/
├── train.csv                 # Official training data; never publish it
├── config.py                 # Shared paths and model settings
├── data_utils.py             # Input validation
├── features.py               # Feature creation and train/test alignment
├── models.py                 # CatBoost and LightGBM factories
├── train.py                  # Five-fold grouped validation
├── predict.py                # Haggle's required prediction entry point
├── test_submission.py        # End-to-end local runner check
├── model_config.json         # Selected model settings
├── validation_results.json   # Recorded validation and smoke-test evidence
├── VALIDATION_REPORT.md      # Readable validation report
├── AUDIT_REPORT.md           # Code audit and remaining risks
├── requirements.txt
├── README.md
└── .gitignore
```

## Haggle runner contract

Haggle puts `train.csv` and `test.csv` beside `predict.py`, then runs:

```bash
python predict.py
```

The run has no internet access, is CPU-only, and must finish within 10 minutes.
`test.csv` contains the same feature columns as `train.csv`, except for the
three label columns: `hours_spent`, `log_hours`, and `hours_bucket`. The script
writes `predictions.csv` with exactly these columns and one row per test row,
in the same order:

```csv
project_id,hours_spent
3f9a1c0b2e,12.5
```

The predictor resolves files relative to its own location, preserves IDs as
strings, validates the input schema, clips outputs to the official 140.7-hour
cap, and writes the prediction file atomically.

Haggle creates the playable page for a scored ship; you do not need to build a
separate website. A ship still needs the public GitHub repository, README,
description, screenshot, Hackatime project, and Lapse backup required by the
current guide.

## Rules and data integrity

- Use only Haggle's supplied `train.csv`. Do not use additional datasets,
  scraping, project-owner lookups, or other participants' code or predictions.
- Never commit `train.csv`, `test.csv`, or generated CSVs. `.gitignore` excludes
  `*.csv`.
- `project_id`, `hours_spent`, `log_hours`, and `hours_bucket` are never model
  inputs. The last two are derived from the target and would leak label data.
- No saved model binaries are required. `predict.py` trains from the supplied
  training file on each run.

## Research references

- [Haggle Hour Detective data dictionary](https://haggle.hackclub.com/competitions/hour-detective/data/)
- [Haggle first-ship guide and runner contract](https://haggle.hackclub.com/guides/first-ship/)
- [Haggle competition rules](https://haggle.hackclub.com/competitions/hour-detective/rules/)
- [CatBoost: unbiased boosting with categorical features, NeurIPS 2018](https://papers.neurips.cc/paper/7898-catboost-unbiased-boosting-with-categorical-features)
- [LightGBM, NeurIPS 2017](https://papers.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
- [scikit-learn grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
# ML — Hour Detective validation report

## Validation results

- **OOF blended RMSLE (development diagnostic):** 0.65834
- **LightGBM ensemble RMSLE:** 0.66012
- **CatBoost RMSLE:** 0.66604
- **Median baseline RMSLE:** 0.96672
- **Deployed blend weight:** CatBoost 0.250; LightGBM 0.750

Lower RMSLE is better. These are local cross-validation results, not a Haggle
leaderboard score or a guarantee about the hidden 500 projects. The 0.25/0.75 blend weight was selected during model development using the
supplied training data. Therefore the blended OOF score is a model-selection
diagnostic, not a fully independent estimate. The individual model OOF scores
are shown separately, and Haggle's hidden evaluation remains the final test.

## Validation design

- 5-fold `StratifiedGroupKFold` on the supplied training projects.
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

- Fold 1: CatBoost 0.64364; three-seed LightGBM 0.65122
- Fold 2: CatBoost 0.64217; three-seed LightGBM 0.64097
- Fold 3: CatBoost 0.68668; three-seed LightGBM 0.67756
- Fold 4: CatBoost 0.69310; three-seed LightGBM 0.67438
- Fold 5: CatBoost 0.66291; three-seed LightGBM 0.65573

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

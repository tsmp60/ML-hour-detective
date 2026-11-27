"""Shared settings for the Haggle Hour Detective model."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent

ID_COLUMN = "project_id"
TARGET_COLUMN = "hours_spent"
TARGET_DERIVED_COLUMNS = {TARGET_COLUMN, "log_hours", "hours_bucket"}

# The official Hour Detective data dictionary documents this target cap.
MAX_TARGET_HOURS = 140.7

RANDOM_SEED = 42
CV_FOLDS = 5
MAX_THREADS = 2

# Chosen from grouped validation on the supplied 4,500-row training set.
DEFAULT_CATBOOST_ITERATIONS = 1177
DEFAULT_LIGHTGBM_ITERATIONS = 500
DEFAULT_LIGHTGBM_SEEDS = (42, 2027, 2028)
DEFAULT_CATBOOST_WEIGHT = 0.25
DEFAULT_LIGHTGBM_WEIGHT = 0.75

# LightGBM settings selected from paired fold comparisons.
DEFAULT_LIGHTGBM_NUM_LEAVES = 31
DEFAULT_LIGHTGBM_MIN_CHILD_SAMPLES = 35
DEFAULT_LIGHTGBM_LEARNING_RATE = 0.02
DEFAULT_LIGHTGBM_REG_LAMBDA = 5.0
DEFAULT_LIGHTGBM_REG_ALPHA = 0.2
DEFAULT_LIGHTGBM_MAX_BIN = 127
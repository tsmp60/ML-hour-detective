"""Leakage-safe feature creation for Hour Detective."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_object_dtype, is_string_dtype

from config import ID_COLUMN, TARGET_DERIVED_COLUMNS


LOG_COLUMNS = (
    "num_commits", "active_commit_days", "project_span_days",
    "lines_added_total", "lines_deleted_total", "median_lines_per_commit",
    "max_lines_in_one_commit", "lines_of_code", "num_source_files",
    "num_directories", "max_dir_depth", "num_languages", "num_dependencies",
    "num_test_files", "num_image_files", "num_audio_files",
    "num_3d_model_files", "num_font_files", "asset_size_mb",
    "readme_word_count", "readme_num_headings", "readme_num_images",
    "readme_num_code_blocks", "page_weight_kb", "num_js_files_loaded",
    "num_internal_pages", "load_time_ms", "description_word_count",
    "stars", "forks", "num_issues", "num_pull_requests", "num_releases",
)

WEB_MEASUREMENT_COLUMNS = (
    "page_weight_kb", "num_js_files_loaded", "num_internal_pages",
    "is_mobile_responsive", "load_time_ms",
)


def _number(frame: pd.DataFrame, column: str) -> pd.Series:
    """Read a numeric column safely; absent or invalid values become NaN."""
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """A stable ratio for non-negative counts and durations."""
    return numerator / denominator.clip(lower=1)


def _detect_boolean_text_columns(frame: pd.DataFrame) -> set[str]:
    """Find string columns whose observed values are exclusively True/False."""
    detected: set[str] = set()
    for column in frame.columns:
        series = frame[column]
        if not (is_object_dtype(series) or is_string_dtype(series)):
            continue
        values = series.dropna().astype(str).str.strip().str.lower()
        if len(values) and values.isin({"true", "false"}).all():
            detected.add(column)
    return detected


def _normalise_boolean_text(
    frame: pd.DataFrame, boolean_text_columns: set[str] | None = None
) -> pd.DataFrame:
    """Convert schema-known string booleans to numeric flags."""
    result = frame.copy()
    columns = (
        _detect_boolean_text_columns(result)
        if boolean_text_columns is None
        else boolean_text_columns
    )
    for column in columns:
        if column not in result.columns:
            continue
        mapped = result[column].astype("string").str.strip().str.lower().map(
            {"true": 1.0, "false": 0.0}
        )
        result[column] = mapped.astype("float32")
    return result


def make_features(
    frame: pd.DataFrame,
    *,
    boolean_text_columns: set[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop identifiers/target leakage, normalize types, and engineer features.

    Missingness flags are created consistently for every numeric source column.
    This avoids the old train/test bug where a flag could be present in the
    training frame but absent or NaN in the test frame.
    """
    drop = [c for c in (TARGET_DERIVED_COLUMNS | {ID_COLUMN}) if c in frame.columns]
    base = _normalise_boolean_text(
        frame.drop(columns=drop).copy(), boolean_text_columns
    )

    engineered: dict[str, pd.Series] = {}

    # Missing-value flags are fixed by schema, not by whichever split is being
    # processed. A test-only missing value therefore still has a valid flag.
    numeric_sources = [c for c in base.columns if pd.api.types.is_numeric_dtype(base[c])]
    for column in numeric_sources:
        engineered[f"{column}__missing"] = base[column].isna().astype("int8")

    get = lambda name: _number(base, name)

    # Git activity and code-change intensity.
    engineered["commits_per_project_day"] = _safe_ratio(
        get("num_commits"), get("project_span_days")
    )
    engineered["active_days_per_project_day"] = _safe_ratio(
        get("active_commit_days"), get("project_span_days")
    )
    engineered["commits_per_active_day"] = _safe_ratio(
        get("num_commits"), get("active_commit_days")
    )
    engineered["active_day_share_of_span"] = _safe_ratio(
        get("active_commit_days"), get("project_span_days")
    )

    total_changed = get("lines_added_total").fillna(0) + get(
        "lines_deleted_total"
    ).fillna(0)
    engineered["lines_changed_total"] = total_changed
    engineered["lines_changed_per_commit"] = _safe_ratio(
        total_changed, get("num_commits")
    )
    engineered["lines_added_per_commit"] = _safe_ratio(
        get("lines_added_total"), get("num_commits")
    )
    engineered["deletion_share"] = _safe_ratio(
        get("lines_deleted_total").fillna(0), total_changed
    )

    # Repository structure and testing.
    engineered["lines_per_source_file"] = _safe_ratio(
        get("lines_of_code"), get("num_source_files")
    )
    engineered["test_files_per_source_file"] = _safe_ratio(
        get("num_test_files"), get("num_source_files")
    )
    engineered["files_per_directory"] = _safe_ratio(
        get("num_source_files"), get("num_directories")
    )
    engineered["commits_per_source_file"] = _safe_ratio(
        get("num_commits"), get("num_source_files")
    )
    engineered["lines_per_commit_author"] = _safe_ratio(
        get("lines_of_code"), get("num_commit_authors")
    )
    engineered["commits_per_author"] = _safe_ratio(
        get("num_commits"), get("num_commit_authors")
    )

    # Art assets and documentation intensity.
    asset_columns = (
        "num_image_files", "num_audio_files", "num_3d_model_files", "num_font_files"
    )
    asset_count = sum((_number(base, c).fillna(0) for c in asset_columns))
    engineered["asset_file_count"] = asset_count
    engineered["assets_per_source_file"] = _safe_ratio(
        asset_count, get("num_source_files")
    )
    engineered["assets_per_code_line_1k"] = asset_count / (
        get("lines_of_code").clip(lower=0) / 1000 + 1
    )

    readme_words = get("readme_word_count").fillna(0).clip(lower=0) + 1
    for source, output in (
        ("readme_num_headings", "readme_headings_per_100_words"),
        ("readme_num_images", "readme_images_per_100_words"),
        ("readme_num_code_blocks", "readme_code_blocks_per_100_words"),
    ):
        engineered[output] = get(source).fillna(0) * 100.0 / readme_words
    engineered["description_to_readme_ratio"] = (
        get("description_word_count").fillna(0) + 1
    ) / readme_words

    # Missing live-site measurements are informative, but the count is not a
    # proxy for target information: it only records whether a measurement exists.
    web_missing = sum(
        (_number(base, c).isna().astype("int8") for c in WEB_MEASUREMENT_COLUMNS)
    )
    engineered["web_measurements_missing"] = web_missing.astype("int8")

    # Log versions of skewed non-negative counts/sizes. Raw values stay present.
    for column in LOG_COLUMNS:
        if column in base.columns:
            engineered[f"{column}__log1p"] = np.log1p(
                _number(base, column).clip(lower=0)
            )

    features = pd.concat([base, pd.DataFrame(engineered, index=base.index)], axis=1)

    # These two source fields are ordinal bands, not arbitrary categories.
    # Append ordered numeric views while retaining the original categories.
    ordinal_maps = {
        "author_prior_projects": {"0": 0, "1-2": 1, "3-5": 2, "6+": 3},
        "author_account_age_years": {"<1": 0, "1-2": 1, "3-5": 2, "6+": 3},
    }
    ordinal_features: dict[str, pd.Series] = {}
    for column, mapping in ordinal_maps.items():
        if column in base.columns:
            ordinal_features[f"{column}__ordinal"] = (
                base[column].astype("string").str.strip().map(mapping).astype("float32")
            )
    if ordinal_features:
        features = pd.concat(
            [features, pd.DataFrame(ordinal_features, index=base.index)], axis=1
        )

    categorical_columns: list[str] = []
    for column in features.columns:
        series = features[column]
        if is_bool_dtype(series):
            features[column] = series.astype("int8")
        elif is_object_dtype(series) or is_string_dtype(series) or isinstance(
            series.dtype, pd.CategoricalDtype
        ):
            categorical_columns.append(column)
            features[column] = (
                series.astype("object").where(series.notna(), "__MISSING__").astype(str)
            )
        else:
            features[column] = pd.to_numeric(series, errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )

    if features.empty or features.shape[1] == 0:
        raise ValueError("No usable model features remain after preprocessing.")
    if not features.columns.is_unique:
        raise ValueError("Feature names are not unique after feature engineering.")
    return features, categorical_columns


def make_train_test_features(
    train_frame: pd.DataFrame, test_frame: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Create train/test matrices with the same columns and compatible types."""
    train_raw = train_frame.drop(
        columns=[c for c in (TARGET_DERIVED_COLUMNS | {ID_COLUMN}) if c in train_frame],
        errors="ignore",
    )
    # The training schema is authoritative. A new value in test.csv must not
    # change a source column from numeric to categorical (or vice versa).
    # For a training boolean-text column, any unseen test value safely maps to
    # missing; for a training categorical column, test values remain categories.
    boolean_text_columns = _detect_boolean_text_columns(train_raw)
    boolean_text_columns.update(
        column for column in train_raw.columns if is_bool_dtype(train_raw[column])
    )
    train_x, categorical_columns = make_features(
        train_frame, boolean_text_columns=boolean_text_columns
    )
    test_x, _ = make_features(
        test_frame, boolean_text_columns=boolean_text_columns
    )
    test_x = test_x.reindex(columns=train_x.columns)

    for column in categorical_columns:
        test_x[column] = (
            test_x[column]
            .astype("object")
            .where(test_x[column].notna(), "__MISSING__")
            .astype(str)
        )
    for column in train_x.columns:
        if column not in categorical_columns:
            test_x[column] = pd.to_numeric(test_x[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )

    return train_x, test_x, categorical_columns


def to_lightgbm_categories(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame | None,
    categorical_columns: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Use training categories; unseen categories safely become missing."""
    train_lgb = train_x.copy()
    test_lgb = test_x.copy() if test_x is not None else None
    for column in categorical_columns:
        levels = pd.Index(train_lgb[column].astype(str).unique())
        train_lgb[column] = pd.Categorical(
            train_lgb[column].astype(str), categories=levels
        )
        if test_lgb is not None:
            test_values = test_lgb[column].astype(str)
            test_lgb[column] = pd.Categorical(test_values, categories=levels)
    return train_lgb, test_lgb
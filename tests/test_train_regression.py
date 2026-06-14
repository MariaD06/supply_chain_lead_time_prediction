"""Tests for core regression pipeline helpers used before full model training."""

import pandas as pd
import pytest

from src.models.train_regression import (
    build_preprocessor,
    chronological_split,
    validate_no_leakage_features,
)


NUMERIC_FEATURES = [
    "distance_km",
    "weight_mt",
    "fuel_price_index",
    "geopolitical_risk_score",
    "carrier_reliability_score",
    "copper__usd_per_mt",
]

CATEGORICAL_FEATURES = [
    "origin_port",
    "destination_port",
    "transport_mode",
    "product_category",
    "weather_condition",
]

FEATURE_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_COL = "lead_time_days"
DATE_COL = "date"

LEAKAGE_COLS = {
    TARGET_COL,
    DATE_COL,
    "month",
    "shipment_id",
    "disruption_occurred",
}


def make_tiny_modeling_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01"]
            ),
            "distance_km": [100.0, 200.0, 300.0, 400.0],
            "weight_mt": [10.0, 20.0, 30.0, 40.0],
            "fuel_price_index": [1.1, 1.2, 1.3, 1.4],
            "geopolitical_risk_score": [0.1, 0.2, 0.3, 0.4],
            "carrier_reliability_score": [0.95, 0.90, 0.85, 0.80],
            "copper__usd_per_mt": [8000.0, 8100.0, 8200.0, 8300.0],
            "origin_port": ["Hamburg", "Hamburg", "Rotterdam", "Rotterdam"],
            "destination_port": ["Shanghai", "New York", "Shanghai", "New York"],
            "transport_mode": ["sea", "air", "sea", "rail"],
            "product_category": ["electronics", "metals", "electronics", "metals"],
            "weather_condition": ["clear", "rain", "clear", "storm"],
            "lead_time_days": [20.0, 10.0, 25.0, 15.0],
        }
    )

# Checks that numeric scaling and categorical one-hot encoding run on valid input.
def test_build_preprocessor_fit_transform_tiny_dataframe():
    df = make_tiny_modeling_df()

    preprocessor = build_preprocessor(
        numeric_features=NUMERIC_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )

    transformed = preprocessor.fit_transform(df[FEATURE_COLS])

    assert transformed.shape[0] == len(df)
    assert transformed.shape[1] >= len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)


# Checks that the target column cannot accidentally be used as a feature.
def test_validate_no_leakage_features_raises_for_target_column():
    bad_feature_cols = FEATURE_COLS + [TARGET_COL]

    with pytest.raises(ValueError, match="Leakage columns found"):
        validate_no_leakage_features(bad_feature_cols, LEAKAGE_COLS)


# Checks that outcome-like information is blocked from the feature list.
def test_validate_no_leakage_features_raises_for_outcome_like_column():
    bad_feature_cols = FEATURE_COLS + ["disruption_occurred"]

    with pytest.raises(ValueError, match="Leakage columns found"):
        validate_no_leakage_features(bad_feature_cols, LEAKAGE_COLS)


# Checks that the final holdout split respects chronological order.
def test_chronological_split_uses_rows_before_test_start_for_training():
    df = make_tiny_modeling_df()

    train_df, test_df = chronological_split(
        df=df,
        date_col=DATE_COL,
        test_start_date="2023-03-01",
    )

    assert len(train_df) == 2
    assert len(test_df) == 2

    assert train_df[DATE_COL].max() < pd.Timestamp("2023-03-01")
    assert test_df[DATE_COL].min() >= pd.Timestamp("2023-03-01")

    assert train_df[TARGET_COL].tolist() == [20.0, 10.0]
    assert test_df[TARGET_COL].tolist() == [25.0, 15.0]


# Checks that invalid split dates fail clearly instead of producing empty data.
def test_chronological_split_raises_if_train_or_test_split_is_empty():
    df = make_tiny_modeling_df()

    with pytest.raises(ValueError, match="empty train or test data"):
        chronological_split(
            df=df,
            date_col=DATE_COL,
            test_start_date="2025-01-01",
        )
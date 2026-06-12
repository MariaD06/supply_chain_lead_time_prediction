"""Reusable EDA validation and summary helpers for the lead time regression notebook."""

from __future__ import annotations

import pandas as pd


def missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing-value counts and percentages for all columns."""
    return (
        df.isna()
        .sum()
        .to_frame("missing_count")
        .assign(missing_pct=lambda x: 100 * x["missing_count"] / len(df))
        .sort_values("missing_count", ascending=False)
    )


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
) -> None:
    """Fail if required columns are missing from the dataset."""
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def validate_no_missing_required_values(
    df: pd.DataFrame,
    required_columns: list[str],
) -> None:
    """Fail if required modeling columns contain missing values."""
    missing_counts = df[required_columns].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]

    if not missing_counts.empty:
        raise ValueError(
            "Required modeling columns contain missing values:\n"
            f"{missing_counts}"
        )


def validate_positive_target(df: pd.DataFrame, target_col: str) -> None:
    """Fail if the target contains missing or non-positive values."""
    if df[target_col].isna().any():
        raise ValueError(f"Target column {target_col!r} contains missing values.")

    if (df[target_col] <= 0).any():
        raise ValueError(f"Target column {target_col!r} contains non-positive values.")


def long_delay_summary(
    df: pd.DataFrame,
    target_col: str,
    numeric_cols: list[str],
    quantile: float = 0.90,
) -> tuple[float, pd.DataFrame]:
    """Compare numeric variables for long-delay vs non-long-delay shipments."""
    threshold = df[target_col].quantile(quantile)

    summary = (
        df.assign(long_delay=df[target_col] >= threshold)
        .groupby("long_delay")[numeric_cols + [target_col]]
        .agg(["count", "median", "mean", "max"])
    )

    return threshold, summary


def categorical_target_summary(
    df: pd.DataFrame,
    category_col: str,
    target_col: str,
) -> pd.DataFrame:
    """Summarize target values by category."""
    return (
        df.groupby(category_col)[target_col]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .sort_values("count", ascending=False)
    )


def split_summary_by_date(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    test_start_date: str | pd.Timestamp,
) -> pd.DataFrame:
    """Summarize chronological development/test split."""
    test_start_date = pd.to_datetime(test_start_date)
    df_sorted = df.sort_values(date_col).copy()

    dev_df = df_sorted[df_sorted[date_col] < test_start_date]
    test_df = df_sorted[df_sorted[date_col] >= test_start_date]

    return pd.DataFrame(
        {
            "period": ["development", "final_test"],
            "n_rows": [len(dev_df), len(test_df)],
            "date_min": [dev_df[date_col].min(), test_df[date_col].min()],
            "date_max": [dev_df[date_col].max(), test_df[date_col].max()],
            "target_mean": [dev_df[target_col].mean(), test_df[target_col].mean()],
            "target_median": [dev_df[target_col].median(), test_df[target_col].median()],
        }
    )
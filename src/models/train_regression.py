"""Train regression models for shipment lead-time prediction."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.model_registry import build_model


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def make_one_hot_encoder() -> OneHotEncoder:
    """Create OneHotEncoder compatible with different scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", make_one_hot_encoder(), categorical_features),
        ]
    )


def evaluate(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mse),
        "r2": r2_score(y_true, y_pred),
    }


def validate_inputs(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    missing_values = df[required_cols].isna().sum()
    missing_values = missing_values[missing_values > 0]

    if not missing_values.empty:
        raise ValueError(f"Missing values found:\n{missing_values}")


def validate_no_leakage_features(feature_cols: list[str], leakage_cols: set[str]) -> None:
    leakage_features = sorted(set(feature_cols) & leakage_cols)

    if leakage_features:
        raise ValueError(f"Leakage columns found in feature list: {leakage_features}")


def chronological_split(
    df: pd.DataFrame,
    date_col: str,
    test_start_date: str,
):
    test_start_date = pd.Timestamp(test_start_date)

    train_df = df.loc[df[date_col] < test_start_date].copy()
    test_df = df.loc[df[date_col] >= test_start_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Chronological split produced empty train or test data.")

    return train_df, test_df


def safe_to_csv(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    """Write DataFrame to CSV; on PermissionError write to a timestamped fallback file.

    Returns the path that was actually written.
    """
    try:
        df.to_csv(path, index=index)
        return path
    except PermissionError as exc:
        raise PermissionError(
            f"Permission denied writing {path}. Close any program using the file and retry."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/modeling.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)

    project_name = config["project_name"]
    target_col = config["data"]["target_col"]
    date_col = config["data"]["date_col"]
    test_start_date = config["data"]["test_start_date"]

    numeric_features = config["features"]["numeric"]
    categorical_features = config["features"]["categorical"]
    feature_cols = numeric_features + categorical_features
    required_cols = feature_cols + [target_col, date_col]

    leakage_cols = {target_col, date_col, "month", "shipment_id", "disruption_occurred"}
    validate_no_leakage_features(feature_cols, leakage_cols)

    cv_n_splits = config.get("cv", {}).get("n_splits", 4)
    cv_gap = config.get("cv", {}).get("gap", 0)

    results_dir = Path(config["output"]["results_dir"]) / project_name
    model_dir = Path(config["output"]["model_dir"])

    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(config["data"]["input_path"])
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    validate_inputs(df, required_cols)

    train_df, test_df = chronological_split(
        df=df,
        date_col=date_col,
        test_start_date=test_start_date,
    )

    X_dev = train_df[feature_cols]
    y_dev = train_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    print("\nSplit sizes:")
    print(f"Dev (train + CV): {len(train_df)}")
    print(f"Test:             {len(test_df)}")

    tscv = TimeSeriesSplit(n_splits=cv_n_splits, gap=cv_gap)

    comparison_rows = []
    cv_fold_rows = []
    fitted_grids = {}

    for model_config_path in config["models"]:
        model_config = load_yaml(model_config_path)
        model_name = model_config["name"]
        param_grid = model_config.get("param_grid", {})

        print(f"\nTraining {model_name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
                ("model", build_model(model_config)),
            ]
        )

        # Prefix keys so GridSearchCV targets the pipeline's model step
        prefixed_grid = {f"model__{k}": v for k, v in param_grid.items()}

        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=prefixed_grid if prefixed_grid else {},
            cv=tscv,
            scoring="neg_mean_absolute_error",
            refit=True,
            n_jobs=-1,
        )

        grid.fit(X_dev, y_dev)

        clean_best_params = {
            key.replace("model__", ""): value
            for key, value in grid.best_params_.items()
        }

        best_pipeline = grid.best_estimator_

        train_pred = best_pipeline.predict(X_dev)
        test_pred = best_pipeline.predict(X_test)

        train_metrics = evaluate(y_dev, train_pred)
        test_metrics = evaluate(y_test, test_pred)

        fold_metrics = []

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_dev), start=1):
            fold_model = clone(best_pipeline)

            X_train_fold = X_dev.iloc[train_idx]
            y_train_fold = y_dev.iloc[train_idx]
            X_val_fold = X_dev.iloc[val_idx]
            y_val_fold = y_dev.iloc[val_idx]

            fold_model.fit(X_train_fold, y_train_fold)
            val_pred = fold_model.predict(X_val_fold)
            metrics = evaluate(y_val_fold, val_pred)

            fold_row = {
                "model": model_name,
                "fold": fold_idx,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "best_params": str(clean_best_params),
            }

            cv_fold_rows.append(fold_row)
            fold_metrics.append(metrics)

        cv_metrics = pd.DataFrame(fold_metrics)

        # Add per-fold rows already appended; now append mean and std summary rows for this model
        mean_row = {
            "model": model_name,
            "fold": "mean",
            "mae": cv_metrics["mae"].mean(),
            "rmse": cv_metrics["rmse"].mean(),
            "r2": cv_metrics["r2"].mean(),
            "n_train": np.nan,
            "n_val": np.nan,
            "best_params": str(clean_best_params),
        }

        std_row = {
            "model": model_name,
            "fold": "std",
            "mae": cv_metrics["mae"].std(),
            "rmse": cv_metrics["rmse"].std(),
            "r2": cv_metrics["r2"].std(),
            "n_train": np.nan,
            "n_val": np.nan,
            "best_params": str(clean_best_params),
        }

        cv_fold_rows.append(mean_row)
        cv_fold_rows.append(std_row)

        # Comparison rows: only train and test (single-evaluation rows)
        comparison_rows.extend([
            {
                "model": model_name,
                "split": "train",
                "mae": train_metrics["mae"],
                "rmse": train_metrics["rmse"],
                "r2": train_metrics["r2"],
                "n_rows": len(y_dev),
                "best_params": str(clean_best_params),
            },
            {
                "model": model_name,
                "split": "test",
                "mae": test_metrics["mae"],
                "rmse": test_metrics["rmse"],
                "r2": test_metrics["r2"],
                "n_rows": len(y_test),
                "best_params": str(clean_best_params),
            },
        ])

        fitted_grids[model_name] = grid

        print(
            f"{model_name}: "
            f"best_params={clean_best_params}  "
            f"train_MAE={train_metrics['mae']:.3f}  "
            f"train_R²={train_metrics['r2']:.3f}  "
            f"cv_MAE={cv_metrics['mae'].mean():.3f} ± {cv_metrics['mae'].std():.3f}  "
            f"cv_R²={cv_metrics['r2'].mean():.3f} ± {cv_metrics['r2'].std():.3f}  "
            f"test_MAE={test_metrics['mae']:.3f}  "
            f"test_R²={test_metrics['r2']:.3f}"
        )


    comparison_results = pd.DataFrame(comparison_rows)
    comparison_cols = ["model", "split", "mae", "rmse", "r2", "n_rows", "best_params"]
    comparison_results = comparison_results.reindex(columns=comparison_cols)
    comparison_path = results_dir / "model_comparison.csv"
    written_comparison = safe_to_csv(comparison_results, comparison_path, index=False)

    cv_fold_results = pd.DataFrame(cv_fold_rows)
    cv_fold_cols = ["model", "fold", "mae", "rmse", "r2", "n_train", "n_val", "best_params"]
    cv_fold_results = cv_fold_results.reindex(columns=cv_fold_cols)
    cv_fold_path = results_dir / "cv_fold_metrics.csv"
    written_cv = safe_to_csv(cv_fold_results, cv_fold_path, index=False)

    # Select best model by CV score — test set is never used for selection
    best_model_name = max(fitted_grids, key=lambda n: fitted_grids[n].best_score_)
    best_model = fitted_grids[best_model_name].best_estimator_

    model_path = model_dir / "lead_time_model.joblib"
    joblib.dump(best_model, model_path)

    print("\nBest model:", best_model_name)
    print("Saved:")
    print(f"- {comparison_path}")
    print(f"- {cv_fold_path}")
    print(f"- {model_path}")

if __name__ == "__main__":
    main()
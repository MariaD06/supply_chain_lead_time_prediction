"""Train regression models for shipment lead-time prediction."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
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

    rows = []
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

        best_cv_mae = -grid.best_score_
        test_pred = grid.predict(X_test)
        test_metrics = evaluate(y_test, test_pred)

        train_pred = grid.predict(X_dev)

        train_metrics = evaluate(y_dev, train_pred)  # replaces the single train_mae line

        rows.append({
            "model": model_name,
            "best_params": str(grid.best_params_),
            "train_mae": train_metrics["mae"],
            "train_r2": train_metrics["r2"],
            "cv_mae": best_cv_mae,
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
        })

        fitted_grids[model_name] = grid

        print(
            f"{model_name}: "
            f"best_params={grid.best_params_}  "
            f"train_MAE={train_metrics['mae']:.3f}  "
            f"train_R²={train_metrics['r2']:.3f}  "
            f"cv_MAE={best_cv_mae:.3f}  "
            f"test_MAE={test_metrics['mae']:.3f}  "
            f"test_R²={test_metrics['r2']:.3f}"
        )

    results = pd.DataFrame(rows)
    results_path = results_dir / "model_comparison.csv"
    results.to_csv(results_path, index=False)

    # Select best model by CV score — test set is never used for selection
    best_model_name = min(fitted_grids, key=lambda n: -fitted_grids[n].best_score_)
    best_model = fitted_grids[best_model_name].best_estimator_

    model_path = model_dir / "lead_time_model.joblib"
    joblib.dump(best_model, model_path)

    test_predictions = test_df[["shipment_id", date_col, target_col]].copy()
    test_predictions["prediction"] = best_model.predict(X_test)
    predictions_path = results_dir / "predictions.csv"
    test_predictions.to_csv(predictions_path, index=False)

    print("\nBest model:", best_model_name)
    print("Saved:")
    print(f"- {results_path}")
    print(f"- {predictions_path}")
    print(f"- {model_path}")


if __name__ == "__main__":
    main()
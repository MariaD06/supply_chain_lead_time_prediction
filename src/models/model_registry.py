"""Model registry for regression models."""

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


def build_model(model_config: dict):
    """Build a regression model from config."""
    estimator = model_config["estimator"]
    params = model_config.get("params", {})

    if estimator == "dummy":
        return DummyRegressor(**params)

    if estimator == "ridge":
        return Ridge(**params)

    if estimator == "random_forest":
        return RandomForestRegressor(**params)

    if estimator == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError(
                "xgboost is not installed. Install it or remove the XGBoost config."
            ) from exc

        return XGBRegressor(**params)

    raise ValueError(f"Unknown estimator: {estimator}")
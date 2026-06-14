"""FastAPI app for shipment lead-time prediction."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


MODEL_PATH = Path("models/lead_time_model.joblib")

FEATURE_COLUMNS = [
    "distance_km",
    "weight_mt",
    "fuel_price_index",
    "geopolitical_risk_score",
    "carrier_reliability_score",
    "copper__usd_per_mt",
    "origin_port",
    "destination_port",
    "transport_mode",
    "product_category",
    "weather_condition",
]


# FastAPI validates incoming JSON against this schema.
class ShipmentFeatures(BaseModel):
    """Input schema for one shipment prediction."""

    distance_km: float = Field(14285.36, ge=0)
    weight_mt: float = Field(237.24, ge=0)
    fuel_price_index: float = Field(2.3)
    geopolitical_risk_score: float = Field(7.5)
    carrier_reliability_score: float = Field(0.592)
    copper__usd_per_mt: float = Field(9464.43)

    origin_port: str = Field("Singapore")
    destination_port: str = Field("Shanghai")
    transport_mode: str = Field("Rail")
    product_category: str = Field("Automotive")
    weather_condition: str = Field("Storm")


# ASGI servers such as Uvicorn use this object to run the API.
app = FastAPI(title="Supply Chain Lead Time Prediction API")


def load_model(model_path: Path = MODEL_PATH) -> Any:
    """Load the trained sklearn pipeline."""
    return joblib.load(model_path)


@app.get("/health")
def health() -> dict[str, str]:
    """Return API health status."""
    return {"status": "ok"}


@app.post("/predict")
def predict(features: ShipmentFeatures) -> dict[str, float]:
    """Predict shipment lead time in days."""
    model = load_model()

    # Convert the validated request into the one-row DataFrame expected by sklearn.
    input_df = pd.DataFrame([features.model_dump()])

    # Enforce the same feature column order used during model training.
    input_df = input_df[FEATURE_COLUMNS]

    prediction = model.predict(input_df)[0]

    return {"predicted_lead_time_days": round(float(prediction), 2)}
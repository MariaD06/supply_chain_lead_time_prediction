# Supply Chain Lead Time Prediction

![CI](https://github.com/MariaD06/supply_chain_lead_time_prediction/actions/workflows/ci.yml/badge.svg)

Shipment lead-time regression pipeline: DuckDB SQL preparation → scikit-learn / XGBoost training with chronological validation → FastAPI prediction endpoint.

## Business Problem

Unexpectedly long shipment lead times create planning problems, stock risks, and delivery delays. This project predicts shipment `lead_time_days` from information available at or before shipment planning.

The use case is shipment-level tabular regression, not time-series forecasting.

## Project Overview

1. Clean raw supply-chain and commodity-price files into SQL-ready CSVs
2. Use DuckDB SQL to validate, join, and export the modeling dataset
3. Run EDA and leakage diagnostics
4. Train and evaluate regression models with chronological validation
5. Serve predictions through a FastAPI endpoint

## Quickstart

**Requirements:** Python 3.10+, [DuckDB CLI](https://duckdb.org/docs/installation/) for the SQL pipeline step.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare cleaned CSVs from raw data
python src/data/prepare_sql_inputs.py

# 3. Run the DuckDB SQL pipeline (validate, join, export modeling dataset)
duckdb < sql/01_load_and_prepare.sql
duckdb < sql/02_validate.sql
duckdb < sql/03_export.sql

# 4. Train models and save the best pipeline
python -m src.models.train_regression

# 5. Start the prediction API
uvicorn src.api.main:app --reload
```

## Testing

```bash
pytest tests/
```

## Data and Features

The modeling dataset is built from raw supply-chain shipment records joined with World Bank monthly commodity prices via DuckDB SQL.

Target:

* `lead_time_days`

Numeric features:

* `distance_km`
* `weight_mt`
* `fuel_price_index`
* `geopolitical_risk_score`
* `carrier_reliability_score`
* `copper__usd_per_mt`

Categorical features:

* `origin_port`
* `destination_port`
* `transport_mode`
* `product_category`
* `weather_condition`

Excluded from model inputs:

* `shipment_id`, `date`, `month` — identifiers and temporal join keys
* `lead_time_days` — target variable
* `disruption_occurred` — excluded as outcome-like information not assumed available before shipment completion

## Modeling Approach

* Chronological final holdout test set based on `data.test_start_date`
* `TimeSeriesSplit` cross-validation on the development period
* `GridSearchCV` with mean absolute error as the selection metric
* Numeric scaling and one-hot encoding inside the sklearn pipeline
* Final evaluation once on the held-out test period
* Best model selected by CV performance, not test performance

Models compared:

* DummyRegressor (baseline)
* Ridge regression
* Random Forest regression
* XGBoost regression

## Diagnostics

Tree-based models achieved very high performance. To check whether this was caused by leakage or data structure, the project includes a diagnostic notebook with leakage checks and feature ablations.

Ablations showed performance is mainly driven by `distance_km`, `transport_mode`, and `weather_condition`, suggesting the dataset contains strong nonlinear, rule-like structure. Results should be interpreted as performance on a simulated dataset, not as evidence of equal performance in a noisy real-world logistics system.

## Prediction API

The FastAPI endpoint serves predictions from the trained scikit-learn pipeline. The model is loaded once at server startup.

Run from the project root:

```bash
uvicorn src.api.main:app --reload
```

Interactive API docs (while the server is running):

```
http://127.0.0.1:8000/docs
```

Endpoints:

* `GET /health` — checks whether the API is running
* `POST /predict` — returns a predicted shipment lead time in days

Example request:

```json
{
    "distance_km": 14285.36,
    "weight_mt": 237.24,
    "fuel_price_index": 2.3,
    "geopolitical_risk_score": 7.5,
    "carrier_reliability_score": 0.592,
    "copper__usd_per_mt": 9464.43,
    "origin_port": "Singapore",
    "destination_port": "Shanghai",
    "transport_mode": "Rail",
    "product_category": "Automotive",
    "weather_condition": "Storm"
}
```

Example response:

```json
{
  "predicted_lead_time_days": 38.14
}
```

## Docker

Build and run the prediction API in a container:

```bash
docker build -t supply-chain-api .
docker run --rm -p 8000:8000 supply-chain-api
```

The API will be available at `http://localhost:8000`. Use `http://localhost:8000/docs` for the interactive Swagger UI.

To stop the container, press `Ctrl+C` in the terminal where it is running.

> **Note:** The container serves predictions only — it requires `models/lead_time_model.joblib` to be present (already committed to the repo). To retrain the model, follow the Quickstart steps locally first.

## Outputs

Training produces:

```
results/lead_time_regression/model_comparison.csv
results/lead_time_regression/cv_fold_metrics.csv
models/lead_time_model.joblib
```

## Stack

* Python
* pandas · numpy · scikit-learn · XGBoost
* DuckDB (CLI)
* Jupyter
* FastAPI · Uvicorn
* pytest

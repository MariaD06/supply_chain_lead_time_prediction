# Supply Chain Lead Time Prediction

## Business Problem

Unexpectedly long shipment lead times can create planning problems, stock risks, and delivery delays. This project predicts shipment `lead_time_days` from information assumed to be available at or before shipment planning.

The use case is shipment-level tabular regression, not time-series forecasting.

## Project Overview

The current workflow includes:

1. Clean raw supply-chain and commodity-price files into SQL-ready CSV files
2. Use DuckDB SQL to validate, join, and export the modeling dataset
3. Run EDA and leakage diagnostics
4. Train and evaluate regression models
5. Serve predictions through FastAPI

## Data and Features

The modeling dataset is:

```text
data/processed/supply_chain_enriched_overlap.csv
```

SQL-derived variables include:

* `month`, derived from `date` for joining and temporal structure
* `copper__usd_per_mt`, added through the monthly commodity-price join

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

* `shipment_id`
* `date`
* `month`
* `lead_time_days`
* `disruption_occurred`

`disruption_occurred` is excluded because it is treated as likely outcome-like information that should not be assumed known before shipment completion.

## Modeling Approach

The modeling pipeline uses:

* chronological final holdout test set based on `data.test_start_date`
* `TimeSeriesSplit` on the development period
* `GridSearchCV` with mean absolute error as the selection metric
* numeric scaling and one-hot encoding inside the pipeline
* final evaluation once on the held-out test period

Models included:

* DummyRegressor baseline
* Ridge regression
* Random Forest regression
* XGBoost regression

The best model is selected by cross-validation performance, not by final test performance.

## Diagnostics

Tree-based models achieved very high performance. To check whether this was caused by leakage or data structure, the project includes a diagnostic notebook with leakage checks and feature ablations.

The ablations showed that performance is mainly driven by:

* `distance_km`
* `transport_mode`
* `weather_condition`

This suggests that the dataset contains strong nonlinear, rule-like structure. Results should therefore be interpreted as performance on a simulated dataset, not as evidence of equal performance in a noisy real-world logistics system.

## Outputs

The training script saves:

```text
results/lead_time_regression/model_comparison.csv
results/lead_time_regression/cv_fold_metrics.csv
models/lead_time_model.joblib
```


## Prediction API

This project includes a minimal FastAPI endpoint for serving predictions from the trained scikit-learn pipeline. The API loads the fitted model from:

```text
models/lead_time_model.joblib
```

The API expects shipment information that would be available before or during shipment planning and returns the predicted shipment lead time in days.

Run the API from the project root:

```bash
uvicorn src.api.main:app --reload
```

Then open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Available endpoints:

* `GET /health`: checks whether the API is running
* `POST /predict`: returns a predicted shipment lead time in days

Example prediction request:

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
    "weather_condition": "Storm",
}
```

Example response:

```json
{
  "predicted_lead_time_days": 38.14
}
```

The local documentation URL only works while the API is running on your machine.

## Current Status

Completed:

* SQL data preparation
* enriched modeling dataset
* EDA notebook
* TimeSeriesSplit regression pipeline
* model comparison and CV fold outputs
* leakage and ablation diagnostics
* conservative tree-model grids
* add minimal tests
* summarize final results
* add FastAPI endpoint


To Dos: 
* optionally add Docker setup

## Stack

* Python
* pandas
* scikit-learn
* XGBoost
* DuckDB / SQL
* Jupyter
* pytest
* FastAPI, planned
* Docker, optional

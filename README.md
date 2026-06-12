# Supply Chain Lead Time Prediction

## Business Problem

Unexpectedly long shipment lead times can create planning problems, stock risks, and delivery delays. This project predicts shipment `lead_time_days` from information assumed to be available at or before shipment planning.

The use case is shipment-level tabular regression, not time-series forecasting.

## Project Overview

The current workflow includes:

1. Preparing the modeling dataset with DuckDB SQL
2. Joining monthly commodity-price data
3. Running EDA checks for target validity, missingness, feature plausibility, and temporal split feasibility
4. Training regression models with a chronological validation design
5. Diagnosing unusually high tree-model performance with leakage checks and feature ablations

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

## Current Status

Completed:

* SQL data preparation
* enriched modeling dataset
* EDA notebook
* TimeSeriesSplit regression pipeline
* model comparison and CV fold outputs
* leakage and ablation diagnostics
* conservative tree-model grids

Next steps:

* add minimal tests
* summarize final results
* optionally add FastAPI endpoint
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

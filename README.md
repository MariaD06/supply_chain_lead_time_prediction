# Supply Chain Lead Time Prediction

## Business Problem

Unexpectedly long shipment lead times can create planning problems, stock risks, and delivery delays. A data-driven lead-time estimate can help logistics teams identify shipments that may need additional buffer time or operational attention.

The goal of this project is to predict shipment `lead_time_days` using information assumed to be available at or before shipment planning.

## Technical Approach

This project builds a small end-to-end regression workflow:

1. Load prepared supply-chain and commodity-price data
2. Use DuckDB SQL to validate, join, and export the modeling dataset
3. Explore missing values, outliers, and target distribution in an EDA notebook
4. Train and evaluate regression models with scikit-learn
5. Serve predictions through a FastAPI endpoint

The model output is the predicted lead time in days. In a later extension, the API may be expanded with a lightweight decision assistant that adds simple risk explanations and operational recommendations.

## Current Status

- DuckDB SQL pipeline for loading, validation, joining, and export
- Enriched dataset created with monthly commodity price data
- EDA notebook for missing values, outliers, and target checks
- Target selected: `lead_time_days`

## Next Steps

- Finalize input features
- Build regression modeling pipeline
- Evaluate baseline and machine-learning models
- Add FastAPI prediction endpoint
- Optionally add Docker and cloud deployment

## Stack

- Python
- pandas
- scikit-learn
- DuckDB / SQL
- Jupyter
- FastAPI
- pytest
- Docker
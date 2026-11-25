
# Telecom Intelligence

## AI-Powered Revenue-Aware Customer Retention

GCI World 2026 — Final Assignment

## Project Overview

An end-to-end data science and machine learning proof
of concept for telecom customer churn prediction,
explainability and financially prioritized retention.

## Dataset

- Client.csv
- Record.csv
- Total customers: 100,000

## Final Model

- Champion: Tuned XGBoost
- Test ROC-AUC: 0.6937
- Test F1: 0.6887
- Calibration: Uncalibrated
- Decision threshold: 0.3300

## Project Structure

- data/raw: Original assignment datasets.
- notebooks: Complete analysis and model development.
- models: Serialized prediction system.
- outputs/figures: Analytical visualizations.
- outputs/reports: Evaluation and financial reports.
- outputs/predictions: Customer intelligence results.
- outputs/monitoring: Drift monitoring reports.
- retention_system.py: Standalone prediction engine.
- requirements.txt: Python dependencies.

## Reproduction

1. Create a Python virtual environment.
2. Install requirements.txt.
3. Place the supplied datasets inside data/raw.
4. Open the notebook.
5. Select the corresponding Python kernel.
6. Execute all notebook cells in order.

## Limitations

The project is a historical-data proof of concept.
Financial impact is estimated under explicit assumptions.
Production deployment requires external validation
and controlled retention experiments.

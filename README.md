# ML Monitoring System

An end-to-end ML monitoring system that tracks data drift, concept drift, and model performance degradation in production — with a Streamlit dashboard, Evidently reports, and scheduled CI checks.

## Architecture

```
Data layer  →  Drift detection engine  →  Observability layer
(reference,     (PSI · KS test ·           (Streamlit dashboard ·
 production,     chi-squared ·              Alert system ·
 ground truth)   perf. degradation)         Evidently reports)
```

## Project structure

```
ml-monitoring/
├── data/
│   ├── reference_data.csv          # training-time distribution
│   ├── production_data.csv         # simulated live predictions
│   └── generate_production_data.py # generates drifted production data
├── model/
│   └── train.py                    # trains and saves baseline model
├── monitoring/
│   ├── drift_detector.py           # PSI, KS test, chi-squared
│   ├── performance_tracker.py      # accuracy, F1, AUC over time
│   └── alerts.py                   # threshold-based alerting logic
├── dashboard/
│   └── app.py                      # Streamlit monitoring dashboard
├── reports/
│   └── generate_report.py          # Evidently HTML reports (with fallback)
├── notebooks/
│   └── exploration.ipynb           # EDA + drift analysis walkthrough
├── tests/
│   ├── test_drift_detector.py
│   ├── test_performance_tracker.py
│   └── test_alerts.py
├── .github/workflows/
│   └── monitor.yml                 # CI: daily drift checks + report upload
├── requirements.txt
└── Dockerfile
```

## Quickstart

```bash
pip install -r requirements.txt

# 1. Train the baseline model and generate reference data
python model/train.py

# 2. Simulate production data with drift
python data/generate_production_data.py

# 3. Run drift checks
python monitoring/drift_detector.py

# 4. Generate HTML report (Evidently if installed, plain HTML fallback)
python reports/generate_report.py

# 5. Launch the dashboard
streamlit run dashboard/app.py
```

## Dashboard tabs

| Tab | Contents |
|-----|----------|
| **Overview** | KPI cards (features monitored, drifted count, avg PSI) + overlay histograms |
| **Data Drift** | Per-feature PSI / KS / chi-squared table + bar charts |
| **Performance** | Accuracy, F1, AUC on production data + embedded Evidently report |
| **Alerts** | Active drift and performance alerts |

## Drift detection methods

| Method | When used |
|--------|-----------|
| **PSI** (Population Stability Index) | Numerical features — flags distribution shift |
| **KS test** (Kolmogorov-Smirnov) | Numerical features — non-parametric distribution test |
| **Chi-squared** | Categorical features — tests frequency table independence |

Default thresholds: PSI > 0.1, KS p-value < 0.05 — all configurable in the dashboard sidebar.

## Docker

```bash
docker build -t ml-monitoring .
docker run -p 8501:8501 ml-monitoring
# open http://localhost:8501
```

## CI

The GitHub Actions workflow (`.github/workflows/monitor.yml`) runs daily at 06:00 UTC:
1. Trains the model
2. Generates production data
3. Runs drift detection
4. Generates a monitoring report
5. Uploads the report as a workflow artifact (retained 30 days)

## Running tests

```bash
pytest tests/ -v
```

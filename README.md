# MugenFinity — Urban Dengue Early-Warning Models

Machine-learning notebook for ward-level dengue early warning in Bangladesh, developed for the **EDU AI Engineering Hackathon 2026 (Phase 1)**. It uses historical dengue, weather, and ward-vulnerability data to estimate dengue activity **2, 3, and 4 weeks ahead**.

## What it does

- Audits raw data for missing values and duplicates.
- Creates ward-specific weather and dengue-case lag features, rolling rainfall totals, calendar fields, and future targets.
- Visualizes rainfall lag correlations and ward vulnerability patterns.
- Separates training (2016–2023), validation (2024), and held-out test (2025) data chronologically.
- Compares binary outbreak classifiers (whether future cases are at least one) and case-count regressors for each forecast horizon.
- Produces estimated case counts, outbreak probabilities, 80% nominal prediction intervals, and risk categories.
- Saves reusable trained pipelines and a unified model bundle.

## Repository layout

```text
.
├── dengue_preprocessing.ipynb
├── README.md
├── architecture.md
├── hackathon_preli_dataset_-_bangladesh_dengue_large_dataset.csv  # required input (not included here)
├── bangladesh_dengue_preprocessed.csv                        # generated
└── models/                                                   # generated
    ├── dengue_reg_2w.pkl, dengue_reg_3w.pkl, dengue_reg_4w.pkl
    ├── dengue_clf_2w.pkl, dengue_clf_3w.pkl, dengue_clf_4w.pkl
    ├── dengue_xgb_classifier.pkl, dengue_lgbm_classifier.pkl
    ├── dengue_hgb_classifier.pkl, dengue_rf_classifier.pkl
    ├── dengue_lr_classifier.pkl, dengue_nb_classifier.pkl
    └── dengue_forecast_bundle.pkl
```

The notebook also supports placing the input CSV under a `Trial/` subdirectory. In that case, generated CSV and model files are placed there too.

## Setup and execution

1. Install Python 3.10+ and Jupyter Notebook or JupyterLab.
2. Install the packages:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm joblib notebook
   ```

3. Put `hackathon_preli_dataset_-_bangladesh_dengue_large_dataset.csv` next to the notebook (or in `Trial/`).
4. Open `dengue_preprocessing.ipynb` and **Run All** cells in order. The notebook reads the CSV, trains multiple models, generates charts, and writes its output files.

> Training runs numerous models over three horizons and can take time. The notebook is a modeling pipeline, not a web server or live-data ingestion service.

## Prediction outputs

| Output | Meaning |
| --- | --- |
| `target_2w`, `target_3w`, `target_4w` | Observed dengue cases 2, 3, or 4 weeks after the input week (training targets) |
| Classification | Probability and prediction of at least **1** future case |
| Regression | Predicted number of future cases (negative predictions clipped to zero) |
| Interval | Lower and upper estimates using separately fitted 10th- and 90th-quantile models; check empirical coverage before interpreting |
| Risk tier | **Low:** fewer than 2 predicted cases; **Medium:** 2 to fewer than 5; **High:** 5 to fewer than 12; **Critical:** 12 or more |

Classification models are selected using **2024 validation F1**; case-count regressors are selected using **2024 validation RMSE**. The notebook also reports 2025 held-out test metrics, confusion matrices, and selected forecast examples.

## Important limitations

- Predictions describe patterns in historical data; they are **not confirmed future case counts** or medical guidance.
- The notebook uses historical records, not a real-time surveillance feed.
- The nominal 80% intervals come from quantile estimators; actual coverage is measured in the notebook and is not guaranteed to equal 80%.
- Features and results here belong to this notebook's multi-horizon pipeline. A separately supplied HTML dashboard uses its own embedded logistic-regression scoring implementation and should not be assumed to load the saved `.pkl` bundle.

See [architecture.md](architecture.md) for the detailed pipeline and saved-model structure.

# Disaster-Aware Insurance Risk Pipeline

    pip install -r requirements.txt
    python train.py      # trains both models, prints the comparison report, writes ./artifacts
    python app.py        # open http://127.0.0.1:5000

| File | What it does |
|---|---|
| `config.py` | Paths, time windows, **state -> region mapping (edit here)** |
| `disaster_model.py` | Model 1: event table -> state-year features -> LogReg / RF / XGBoost, walk-forward validation, state + regional risk |
| `insurance_model.py` | Model 2: Baseline (A) vs Disaster-aware (B) vs Risk-instead-of-region (C), paired CV comparison, diagnostics, estimator |
| `train.py` | Runs everything, saves models + `results.json` |
| `app.py`, `templates/index.html` | Flask API + interactive UI (Estimator, Risk Map, Model Lab) |

Notes
- Without `xgboost` installed the code falls back to scikit-learn HistGradientBoosting and says so.
- `LAST_RELIABLE_YEAR` in `config.py` is 2011 because the FEMA file thins out after 2012.
- Insurance models are compared using identical 5-fold cross-validation splits repeated 3 times; the XGBoost regressor uses a shallow, regularized configuration selected for lower MAE/RMSE and higher R².
- Disaster prediction uses chronological walk-forward validation. Its small dataset and incomplete later-year event history limit how much the scores can be improved reliably.

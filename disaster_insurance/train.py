"""Run the whole pipeline end-to-end:  python train.py

1. Disaster model  -> compare LogReg / RandomForest / XGBoost (walk-forward)
2. State risk -> regional risk (average of member states)
3. Insurance model -> compare families, then Baseline (A) vs Disaster-aware (B)
4. Save everything the website needs into ./artifacts
"""
import json

import joblib
import numpy as np
import pandas as pd

from common import HAS_XGB
from config import ARTIFACT_DIR, LAST_RELIABLE_YEAR
from disaster_model import run_disaster_pipeline
from insurance_model import (compare_insurance_models, diagnostics, fit_final,
                             load_insurance)


def _json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))


def main():
    ARTIFACT_DIR.mkdir(exist_ok=True)
    pd.options.display.float_format = "{:.3f}".format
    if not HAS_XGB:
        print("NOTE: xgboost not installed -> using sklearn HistGradientBoosting as the "
              "'XGBoost' stand-in. Run `pip install xgboost` for the real thing.\n")

    # ---------------- Model 1 ----------------
    d = run_disaster_pipeline()
    i = d["info"]
    print("=" * 70, "\nMODEL 1 - DISASTER RISK\n" + "=" * 70)
    print(f"{i['n_events']} state-level events, {i['panel_rows']} state-year rows, "
          f"{i['positive_rate']:.1%} of state-years had a disaster next year")
    print(f"Events per year: {i['events_per_year']}")
    print(f"Walk-forward test years {i['test_years'][0]}-{i['test_years'][1]}\n")
    print(pd.DataFrame(d["table"]).set_index("model").to_string())
    print(f"\nSelected: {i['selected_model']}  (by ROC-AUC)")
    print("Top features:", {k: round(v, 3) for k, v in list(d["importances"].items())[:5]})
    print(f"\nPredicted risk of >=1 major disaster in {i['predicts_year']}:")
    print(d["region_risk"].sort_values(ascending=False).round(3).to_string())
    print("\nTop 8 states:\n", d["state_risk"].head(8).to_string(index=False))

    # ---------------- Model 2 ----------------
    df = load_insurance(d["region_risk"])
    summary, paired, best = compare_insurance_models(df)
    diag = diagnostics(df, best)
    print("\n" + "=" * 70, "\nMODEL 2 - INSURANCE COST (5-fold CV x3 repeats)\n" + "=" * 70)
    print(pd.DataFrame(summary)[["family", "spec", "mae", "rmse", "r2"]].to_string(index=False))
    print("\nPaired B vs A (RMSE gain, positive = disaster-aware better):")
    print(pd.DataFrame(paired).to_string(index=False))
    print(f"\nBest family: {best}")
    print("\nDIAGNOSTICS")
    print(f"  risk values per region: {diag['risk_values_per_region']}")
    if diag["risk_is_function_of_region"]:
        print("  !! disaster_risk is a deterministic function of region -> Model B holds no "
              "information that Model A lacks.")
    print(pd.DataFrame(diag["regions"]).to_string(index=False))

    # ---------------- Save ----------------
    models = fit_final(df, best)
    joblib.dump(models, ARTIFACT_DIR / "insurance_models.joblib")
    joblib.dump(d["model"], ARTIFACT_DIR / "disaster_model.joblib")
    d["state_risk"].to_csv(ARTIFACT_DIR / "state_risk.csv", index=False)
    d["region_risk"].rename("risk").to_csv(ARTIFACT_DIR / "region_risk.csv")
    results = {
        "has_xgboost": HAS_XGB, "last_reliable_year": LAST_RELIABLE_YEAR,
        "disaster": {"table": d["table"], "info": i, "importances": d["importances"]},
        "insurance": {"summary": summary, "paired": paired, "best_family": best,
                      "diagnostics": diag, "n_rows": int(len(df))},
    }
    (ARTIFACT_DIR / "results.json").write_text(json.dumps(results, default=_json, indent=2))
    print(f"\nSaved models, risk tables and results.json to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()

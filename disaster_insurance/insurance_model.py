"""MODEL 2 - Insurance cost prediction (baseline vs disaster-aware).

Feature sets compared on identical cross-validation folds:
    A  baseline        : age, sex, bmi, children, smoker, region
    B  disaster-aware  : A + disaster_risk
    C  risk-not-region : age, sex, bmi, children, smoker, disaster_risk  (no region)
    D  demographics    : age, sex, bmi, children, smoker  (used for diagnostics only)

IMPORTANT LIMITATION (the code measures and reports it, see `diagnostics`):
insurance.csv only knows the 4-level `region`, so `disaster_risk` is a
deterministic function of region (one value per region). B therefore contains
exactly the same information as A, and C is just a re-coding of region. Model B
cannot be expected to beat Model A on this dataset. To truly test the research
question you need insurance records with a finer location (state / ZIP / county).
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import xgb_regressor
from config import INSURANCE_CSV, SEED

NUM = ["age", "bmi", "children"]
SPECS = {                                   # name -> (numeric cols, categorical cols)
    "A": (NUM, ["sex", "smoker", "region"]),
    "B": (NUM + ["disaster_risk"], ["sex", "smoker", "region"]),
    "C": (NUM + ["disaster_risk"], ["sex", "smoker"]),
    "D": (NUM, ["sex", "smoker"]),
}
SPEC_LABELS = {"A": "A: Baseline (region)", "B": "B: Disaster-aware (region + risk)",
               "C": "C: Risk instead of region"}


def candidate_regressors():
    return {
        "Linear Regression": LinearRegression,
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=SEED, n_jobs=-1),
        "XGBoost": xgb_regressor,
    }


def build_pipeline(spec, regressor):
    num, cat = SPECS[spec]
    pre = ColumnTransformer([("num", StandardScaler(), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    return Pipeline([("pre", pre), ("model", regressor)])


def load_insurance(region_risk, path=INSURANCE_CSV):
    df = pd.read_csv(path)
    df["region"] = df["region"].str.lower().str.strip()
    df["disaster_risk"] = df["region"].map(region_risk)   # from Model 1 (regional average)
    if df["disaster_risk"].isna().any():
        raise ValueError("Some insurance regions have no disaster-risk score")
    return df


# --------------------------------------------------------------------------
# Cross-validated comparison
# --------------------------------------------------------------------------
def _fold_metrics(y, p):
    return {"mae": mean_absolute_error(y, p),
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "r2": r2_score(y, p)}


def cv_scores(df, spec, make_reg, splits):
    rows = []
    for tr, te in splits:
        m = build_pipeline(spec, make_reg()).fit(df.iloc[tr], df["charges"].iloc[tr])
        rows.append(_fold_metrics(df["charges"].iloc[te], m.predict(df.iloc[te])))
    return pd.DataFrame(rows)


def compare_insurance_models(df, n_splits=5, n_repeats=3):
    splits = list(RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                random_state=SEED).split(df))
    summary, folds = [], {}
    for fam, make in candidate_regressors().items():
        for spec in ("A", "B", "C"):
            f = cv_scores(df, spec, make, splits)
            folds[(fam, spec)] = f
            summary.append({"family": fam, "spec": spec, "label": SPEC_LABELS[spec],
                            **{k: float(f[k].mean()) for k in ("mae", "rmse", "r2")},
                            **{k + "_std": float(f[k].std()) for k in ("mae", "rmse", "r2")}})

    # Paired comparison B vs A on the same folds (positive gain = B better).
    paired = []
    for fam in candidate_regressors():
        gain = folds[(fam, "A")]["rmse"] - folds[(fam, "B")]["rmse"]
        se = float(gain.std(ddof=1) / np.sqrt(len(gain)))
        mean = float(gain.mean())
        verdict = "gain" if mean > 2 * se else ("worse" if mean < -2 * se else "none")
        paired.append({"family": fam, "rmse_gain_B_vs_A": mean, "se": se,
                       "folds_B_better": int((gain > 0).sum()), "n_folds": int(len(gain)),
                       "verdict": verdict})
    best = min((s for s in summary if s["spec"] == "A"), key=lambda s: s["rmse"])["family"]
    return summary, paired, best


def diagnostics(df, family):
    """Evidence for the 'risk is just region in disguise' limitation + a descriptive
    look at whether regional cost levels line up with regional risk (n = 4 only)."""
    per_region = df.groupby("region")["disaster_risk"].nunique()
    oof = cross_val_predict(build_pipeline("D", candidate_regressors()[family]()), df,
                            df["charges"], cv=KFold(5, shuffle=True, random_state=SEED))
    d = df.assign(pred_demo=oof).groupby("region").agg(
        risk=("disaster_risk", "first"), mean_charges=("charges", "mean"),
        actual=("charges", "sum"), predicted=("pred_demo", "sum"))
    d["cost_relativity"] = d["actual"] / d["predicted"]   # >1: costlier than demographics explain
    rho = spearmanr(d["risk"], d["cost_relativity"])[0] if len(d) > 2 else float("nan")
    return {
        "risk_values_per_region": {k: int(v) for k, v in per_region.items()},
        "risk_is_function_of_region": bool((per_region == 1).all()),
        "n_distinct_risk_values": int(df["disaster_risk"].nunique()),
        "regions": [{"region": r, "risk": float(x.risk), "mean_charges": float(x.mean_charges),
                     "cost_relativity": float(x.cost_relativity)} for r, x in d.iterrows()],
        "spearman_risk_vs_relativity": None if np.isnan(rho) else float(rho),
        "note": "n = 4 regions: descriptive only, not statistically meaningful.",
    }


# --------------------------------------------------------------------------
# Final models + estimator used by the website
# --------------------------------------------------------------------------
def fit_final(df, family):
    make = candidate_regressors()[family]
    return {"family": family,
            **{s: build_pipeline(s, make()).fit(df, df["charges"]) for s in ("A", "B", "C")}}


class InsuranceEstimator:
    def __init__(self, models, region_risk):
        self.m, self.region_risk = models, dict(region_risk)

    @staticmethod
    def _row(profile, region, risk):
        return pd.DataFrame([{**profile, "region": region, "disaster_risk": risk}])

    def _predict(self, spec, profile, region, risk):
        return max(0.0, float(self.m[spec].predict(self._row(profile, region, risk))[0]))

    def estimate(self, profile, region, risk=None):
        r = self.region_risk[region] if risk is None else risk
        a = self._predict("A", profile, region, self.region_risk[region])
        b = self._predict("B", profile, region, r)
        return {"baseline": a, "disaster_aware": b, "difference": b - a, "risk_used": r}

    def whatif_curve(self, profile, steps=101):
        """Model C: cost as a function of risk alone (region removed), so the risk
        variable is the only geographic signal. Only 4 risk levels were seen in
        training, so the curve between them is extrapolation - the UI says so."""
        grid = np.linspace(0, 1, steps)
        curve = [self._predict("C", profile, "n/a", float(g)) for g in grid]
        support = [{"region": r, "risk": float(v),
                    "cost": self._predict("C", profile, "n/a", float(v))}
                   for r, v in self.region_risk.items()]
        return {"grid": grid.tolist(), "cost": curve, "support": support}

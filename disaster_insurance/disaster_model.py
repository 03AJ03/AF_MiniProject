"""MODEL 1 - Disaster risk prediction.

Question answered:
    "Given everything that happened in a state up to year t, what is the
     probability that the state gets at least one FEMA major-disaster
     declaration in year t+1?"

Why this target?  The raw file is a list of past events, not a risk label.
We turn it into a state-year panel: one row per (state, as-of year), features
use ONLY information up to the as-of year, and the label is whether an event
happens the following year. That avoids leaking the future into the features.

Evaluation is walk-forward (train on years < Y, test on year Y, repeat), which
mimics real forecasting and is much more honest than a random split.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import xgb_classifier
from config import (DISASTER_CSV, FIRST_TEST_YEAR, HISTORY_START,
                    LAST_RELIABLE_YEAR, SEED, STATE_TO_REGION)

EVENT_TYPES = ["Severe Storm", "Hurricane", "Fire", "Flood", "Severe Ice Storm", "Tornado"]
_slug = lambda s: s.lower().replace(" ", "_")

FEATURES = (["n_events_1y", "n_events_3y", "n_events_5y", "n_events_all",
             "years_since_last", "hist_rate", "years_with_event_5y",
             "avg_areas_5y", "avg_duration_5y"]
            + [f"n_{_slug(t)}_5y" for t in EVENT_TYPES])


# --------------------------------------------------------------------------
# 1. Raw rows -> one row per (state, disaster)
# --------------------------------------------------------------------------
def load_events(path=DISASTER_CSV):
    """The CSV has one row per designated COUNTY. A single disaster hits many
    counties, so we collapse to one event per (state, disasterNumber)."""
    raw = pd.read_csv(path)
    begin = pd.to_datetime(raw["incidentBeginDate"], utc=True, errors="coerce")
    raw["event_year"] = begin.dt.year.fillna(raw["declarationYear"]).astype(int)
    # keep the 50 states + DC (territories cannot be mapped to the 4 regions)
    raw = raw[raw["state"].isin(STATE_TO_REGION) & (raw["event_year"] >= HISTORY_START)]
    return (raw.groupby(["state", "disasterNumber"], as_index=False)
               .agg(event_year=("event_year", "min"),
                    incident_type=("incidentType", "first"),
                    n_areas=("designatedArea", "nunique"),
                    duration=("incidentDuration", "max")))


# --------------------------------------------------------------------------
# 2. Feature engineering (history up to and including year t)
# --------------------------------------------------------------------------
def features_as_of(events, t):
    by_state = {s: g for s, g in events.groupby("state")}
    empty = events.iloc[0:0]
    rows = []
    for state in STATE_TO_REGION:
        h = by_state.get(state, empty)
        h = h[h["event_year"] <= t]
        h5 = h[h["event_year"] > t - 5]
        f = {
            "state": state, "as_of": t,
            "n_events_1y": int((h["event_year"] > t - 1).sum()),
            "n_events_3y": int((h["event_year"] > t - 3).sum()),
            "n_events_5y": len(h5),
            "n_events_all": len(h),
            "years_since_last": min(t - h["event_year"].max(), 10) if len(h) else 10,
            "hist_rate": h["event_year"].nunique() / (t - HISTORY_START + 1),
            "years_with_event_5y": h5["event_year"].nunique(),
            "avg_areas_5y": h5["n_areas"].mean() if len(h5) else 0.0,
            "avg_duration_5y": h5["duration"].mean() if len(h5) else 0.0,
        }
        for typ in EVENT_TYPES:
            f[f"n_{_slug(typ)}_5y"] = int((h5["incident_type"] == typ).sum())
        rows.append(f)
    return pd.DataFrame(rows)


def build_panel(events):
    """Training table: features at year t, label = any event in year t+1."""
    happened = set(zip(events["state"], events["event_year"]))
    parts = []
    for t in range(HISTORY_START + 1, LAST_RELIABLE_YEAR):  # t+1 <= LAST_RELIABLE_YEAR
        p = features_as_of(events, t)
        p["target_year"] = t + 1
        p["y"] = [int((s, t + 1) in happened) for s in p["state"]]
        parts.append(p)
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------
# 3. Candidate models
# --------------------------------------------------------------------------
def candidate_models():
    return {
        "Logistic Regression": lambda: make_pipeline(StandardScaler(),
                                                     LogisticRegression(max_iter=2000)),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1),
        "XGBoost": xgb_classifier,
    }


def metrics(y, p, threshold=0.5):
    yhat = (p >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, yhat),
        "precision": precision_score(y, yhat, zero_division=0),
        "recall": recall_score(y, yhat, zero_division=0),
        "f1": f1_score(y, yhat, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
    }


def walk_forward(panel, make_model):
    """Train on all target years < Y, predict year Y, for Y = FIRST_TEST_YEAR.."""
    out = []
    for yr in range(FIRST_TEST_YEAR, LAST_RELIABLE_YEAR + 1):
        tr, te = panel[panel["target_year"] < yr], panel[panel["target_year"] == yr]
        m = make_model().fit(tr[FEATURES], tr["y"])
        out.append(te.assign(p=m.predict_proba(te[FEATURES])[:, 1]))
    return pd.concat(out, ignore_index=True)


def compare_models(panel):
    """Returns (table, {model name: out-of-sample predictions})."""
    table, preds = [], {}
    for name, make in candidate_models().items():
        pr = walk_forward(panel, make)
        preds[name] = pr
        table.append({"model": name, **metrics(pr["y"], pr["p"])})
    # Reference: "just use the state's historical frequency" - ML must beat this.
    ref = preds["Logistic Regression"]
    table.append({"model": "Reference: historical rate only",
                  **metrics(ref["y"], ref["hist_rate"].clip(0, 1))})
    return table, preds


def importances(model, panel):
    est = model[-1] if hasattr(model, "steps") else model
    if hasattr(est, "feature_importances_"):
        v = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        v = np.abs(est.coef_).ravel()            # features are standardised
    else:                                        # e.g. fallback boosting model
        r = permutation_importance(model, panel[FEATURES], panel["y"],
                                   scoring="roc_auc", n_repeats=10, random_state=SEED)
        v = np.clip(r.importances_mean, 0, None)
    v = v / v.sum() if v.sum() > 0 else v
    return dict(sorted(zip(FEATURES, map(float, v)), key=lambda kv: -kv[1]))


# --------------------------------------------------------------------------
# 4. Full pipeline: compare -> pick best -> score every state -> regions
# --------------------------------------------------------------------------
def risk_category(p, cutoffs):
    return "Low" if p < cutoffs[0] else ("Medium" if p < cutoffs[1] else "High")


def run_disaster_pipeline(path=DISASTER_CSV):
    events = load_events(path)
    panel = build_panel(events)
    table, preds = compare_models(panel)

    candidates = [r for r in table if not r["model"].startswith("Reference")]
    best = sorted(candidates, key=lambda r: -round(r["roc_auc"], 3))[0]["model"]

    final = candidate_models()[best]().fit(panel[FEATURES], panel["y"])
    now = features_as_of(events, LAST_RELIABLE_YEAR)      # predicts year LAST+1
    now["risk"] = final.predict_proba(now[FEATURES])[:, 1]
    now["region"] = now["state"].map(STATE_TO_REGION)

    cutoffs = [float(np.quantile(now["risk"], 1 / 3)), float(np.quantile(now["risk"], 2 / 3))]
    now["category"] = now["risk"].map(lambda p: risk_category(p, cutoffs))
    region_risk = now.groupby("region")["risk"].mean()   # simple average of member states

    info = {
        "n_events": int(len(events)),
        "events_per_year": {int(k): int(v) for k, v in
                            events.groupby("event_year").size().items()},
        "panel_rows": int(len(panel)),
        "positive_rate": float(panel["y"].mean()),
        "test_years": [FIRST_TEST_YEAR, LAST_RELIABLE_YEAR],
        "predicts_year": LAST_RELIABLE_YEAR + 1,
        "selected_model": best,
        "category_cutoffs": cutoffs,
        "features": FEATURES,
    }
    return {"table": table, "info": info, "model": final, "panel": panel,
            "state_risk": now[["state", "region", "risk", "category"]]
                            .sort_values("risk", ascending=False).reset_index(drop=True),
            "region_risk": region_risk, "importances": importances(final, panel)}

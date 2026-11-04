"""Website backend:  python app.py  ->  http://127.0.0.1:5000
(run `python train.py` first so ./artifacts exists)"""
import json

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from config import ARTIFACT_DIR, STATE_NAMES, STATE_TO_REGION
from insurance_model import InsuranceEstimator

app = Flask(__name__)

for f in ("insurance_models.joblib", "state_risk.csv", "region_risk.csv", "results.json"):
    if not (ARTIFACT_DIR / f).exists():
        raise SystemExit(f"Missing artifacts/{f} - run `python train.py` first.")

RESULTS = json.loads((ARTIFACT_DIR / "results.json").read_text())
STATE_RISK = pd.read_csv(ARTIFACT_DIR / "state_risk.csv").set_index("state")
REGION_RISK = pd.read_csv(ARTIFACT_DIR / "region_risk.csv").set_index("region")["risk"].to_dict()
EST = InsuranceEstimator(joblib.load(ARTIFACT_DIR / "insurance_models.joblib"), REGION_RISK)
CUTOFFS = RESULTS["disaster"]["info"]["category_cutoffs"]
YEAR = RESULTS["disaster"]["info"]["predicts_year"]
VERDICTS = {p["family"]: p["verdict"] for p in RESULTS["insurance"]["paired"]}
EVIDENCE = VERDICTS.get(RESULTS["insurance"]["best_family"], "none")


def category(p):
    return "Low" if p < CUTOFFS[0] else ("Medium" if p < CUTOFFS[1] else "High")


def parse_profile(d):
    """Validate user input; returns (profile, warnings) or raises ValueError."""
    try:
        p = {"age": float(d["age"]), "bmi": float(d["bmi"]), "children": int(d["children"]),
             "sex": str(d["sex"]).lower(), "smoker": str(d["smoker"]).lower()}
    except (KeyError, ValueError, TypeError):
        raise ValueError("Please fill in age, sex, BMI, children and smoker.")
    if p["sex"] not in ("male", "female") or p["smoker"] not in ("yes", "no"):
        raise ValueError("sex must be male/female and smoker must be yes/no.")
    if not (18 <= p["age"] <= 100 and 10 <= p["bmi"] <= 60 and 0 <= p["children"] <= 10):
        raise ValueError("Values out of range (age 18-100, BMI 10-60, children 0-10).")
    warn = []
    if p["age"] > 64:
        warn.append("Age is above the training range (18-64); the estimate is an extrapolation.")
    return p, warn


def explain(state, region, s_risk, r_risk, diff):
    name = STATE_NAMES[state]
    txt = (f"{name} belongs to the {region.title()} region. The disaster model gives {name} a "
           f"{s_risk:.0%} chance of at least one FEMA major-disaster declaration in {YEAR} "
           f"({category(s_risk)} compared with other states); the {region.title()} region "
           f"averages {r_risk:.0%} ({category(r_risk)}). ")
    if EVIDENCE == "gain":
        txt += (f"Cross-validation shows disaster risk improves cost prediction, so the "
                f"${abs(diff):,.0f} difference is a learned effect.")
    else:
        txt += (f"The two estimates differ by ${abs(diff):,.0f}. In cross-validation, adding "
                f"disaster risk did NOT measurably improve cost prediction (the insurance data only "
                f"has a 4-level region, so risk carries the same information as region). Treat "
                f"this gap as model noise, not a disaster effect.")
    return txt


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/meta")
def meta():
    states = [{"code": s, "name": STATE_NAMES[s], "region": STATE_TO_REGION[s],
               "risk": float(STATE_RISK.loc[s, "risk"]), "category": STATE_RISK.loc[s, "category"]}
              for s in sorted(STATE_NAMES, key=STATE_NAMES.get)]
    regions = [{"region": r, "risk": float(v), "category": category(v)} for r, v in REGION_RISK.items()]
    return jsonify(states=states, regions=regions, year=YEAR, evidence=EVIDENCE,
                   best_family=RESULTS["insurance"]["best_family"])


@app.get("/api/results")
def results():
    return jsonify(RESULTS)


@app.post("/api/estimate")
def estimate():
    d = request.get_json(force=True, silent=True) or {}
    try:
        profile, warn = parse_profile(d)
        state = str(d.get("state", "")).upper()
        if state not in STATE_TO_REGION:
            raise ValueError("Please choose a valid US state.")
    except ValueError as e:
        return jsonify(error=str(e)), 400
    region = STATE_TO_REGION[state]
    s_risk, r_risk = float(STATE_RISK.loc[state, "risk"]), float(REGION_RISK[region])
    est = EST.estimate(profile, region)
    return jsonify(state=state, state_name=STATE_NAMES[state], region=region,
                   state_risk=s_risk, state_category=category(s_risk),
                   region_risk=r_risk, region_category=category(r_risk),
                   baseline=est["baseline"], disaster_aware=est["disaster_aware"],
                   difference=est["difference"], evidence=EVIDENCE, warnings=warn,
                   explanation=explain(state, region, s_risk, r_risk, est["difference"]))


@app.post("/api/whatif")
def whatif():
    d = request.get_json(force=True, silent=True) or {}
    try:
        profile, _ = parse_profile(d)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    return jsonify(EST.whatif_curve(profile))


if __name__ == "__main__":
    app.run(debug=True)

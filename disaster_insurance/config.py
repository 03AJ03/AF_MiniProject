"""Central configuration for the disaster-aware insurance pipeline.

Everything a teammate might want to change (time windows, the state->region
mapping, random seed) lives here.
"""
from pathlib import Path

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
ARTIFACT_DIR = BASE / "artifacts"
DISASTER_CSV = DATA_DIR / "preprocessed_data.csv"
INSURANCE_CSV = DATA_DIR / "insurance.csv"

SEED = 42

# ---------------------------------------------------------------------------
# Time windows for the disaster model
# ---------------------------------------------------------------------------
# 1998 only contains events from Aug-Dec (partial year) so history starts in 1999.
HISTORY_START = 1999
# Event counts drop to 1-5 events/year after 2012 in this file (looks like the
# file was truncated/filtered, not that disasters stopped), so we do not trust
# later years. Raise this if you obtain a more complete FEMA extract.
LAST_RELIABLE_YEAR = 2011
# Walk-forward validation: first year used as a test year (train on all earlier years).
FIRST_TEST_YEAR = 2006

# ---------------------------------------------------------------------------
# State -> insurance region mapping (RULE-BASED, not ML)
# ---------------------------------------------------------------------------
# The insurance dataset only has 4 regions, so this is our own assumption
# (roughly: east/west of the Plains, north/south of ~37N). Edit freely,
# but keep every state in exactly one region and re-run train.py.
REGION_STATES = {
    "northeast": ["CT", "DE", "DC", "ME", "MD", "MA", "NH", "NJ", "NY", "PA",
                  "RI", "VT", "OH", "MI", "IN", "IL", "WI", "MN", "IA"],
    "southeast": ["VA", "WV", "KY", "TN", "NC", "SC", "GA", "FL", "AL", "MS",
                  "LA", "AR", "MO"],
    "southwest": ["CA", "NV", "UT", "CO", "AZ", "NM", "TX", "OK", "KS", "HI"],
    "northwest": ["WA", "OR", "ID", "MT", "WY", "ND", "SD", "NE", "AK"],
}
STATE_TO_REGION = {s: r for r, states in REGION_STATES.items() for s in states}
assert len(STATE_TO_REGION) == 51, "every state/DC must be in exactly one region"

STATE_NAMES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
    "DC": "District of Columbia", "FL": "Florida", "GA": "Georgia", "HI": "Hawaii",
    "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
    "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
    "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming",
}
assert set(STATE_NAMES) == set(STATE_TO_REGION)

"""Model factories shared by the disaster and insurance modules.

If `xgboost` is not installed we fall back to scikit-learn's
HistGradientBoosting (same family of model: gradient-boosted trees) so the
pipeline still runs. `HAS_XGB` tells you which one you are actually using.
"""
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)

from config import SEED

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:  # ImportError or a broken install
    HAS_XGB = False


def xgb_classifier():
    if HAS_XGB:
        return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric="logloss", random_state=SEED, n_jobs=1)
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                          max_iter=300, min_samples_leaf=10,
                                          l2_regularization=1.0, random_state=SEED)


def xgb_regressor():
    if HAS_XGB:
        return XGBRegressor(n_estimators=400, max_depth=2, learning_rate=0.03,
                            min_child_weight=5, reg_lambda=5,
                            subsample=0.9, colsample_bytree=0.9,
                            random_state=SEED, n_jobs=1)
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                         max_iter=300, min_samples_leaf=10,
                                         l2_regularization=1.0, random_state=SEED)

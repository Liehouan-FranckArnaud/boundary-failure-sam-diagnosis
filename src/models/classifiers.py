"""
src/models/classifiers.py
=========================
Classifier pipeline: median imputation -> classifier.

Median imputation is the standard low-resource clinical AI baseline
(Janssen et al., 2024). It is identifiable under MNAR without
structural assumptions.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

CLASSIFIER_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest":       "Random Forest",
    "gradient_boosting":   "Gradient Boosting",
    "xgboost":             "XGBoost",
}
ALL_CLASSIFIERS = list(CLASSIFIER_LABELS)


def build_classifier(name: str, seed: int = 42) -> Pipeline:
    """
    Build a fresh Pipeline: SimpleImputer(median) -> classifier.
    A new Pipeline is returned on every call.
    """
    _clfs = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=seed,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, random_state=seed,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            eval_metric="mlogloss", random_state=seed, verbosity=0,
        ),
    }
    if name not in _clfs:
        raise ValueError(
            f"Unknown classifier '{name}'. Available: {ALL_CLASSIFIERS}"
        )
    return Pipeline([
        ("imputer",    SimpleImputer(strategy="median")),
        ("classifier", _clfs[name]),
    ])


def cross_validate_classifier(
    pipeline: Pipeline,
    X_train:  pd.DataFrame,
    y_train:  np.ndarray,
    n_splits: int = 5,
    seed:     int = 42,
) -> dict:
    """Stratified k-fold CV. Returns cv_mean, cv_std, cv_scores."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv, scoring="balanced_accuracy",
    )
    return {
        "cv_mean":   float(scores.mean()),
        "cv_std":    float(scores.std()),
        "cv_scores": scores.tolist(),
    }

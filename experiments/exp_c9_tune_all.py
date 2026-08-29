"""
experiments/exp_c9_tune_all.py
==============================
Hyperparameter tuning of ALL models, not only XGBoost.
This extends exp_c4 to Logistic Regression, Random Forest, Gradient Boosting and
XGBoost, so the claim "the boundary failure is structural, not a default-settings
artefact" covers every classifier in the paper.

Protocol (no leakage): fit on TRAIN, select the configuration on the held-out TEST
split (high_burden), report boundary Recall on the untouched crisis VALIDATION
split. Selection criterion: SAM recall on the test split (the paper's primary
metric); the best-by-balanced-accuracy configuration is reported too.

Output: results/tables/exp_c9_tune_all.csv

Run:
  python experiments/exp_c9_tune_all.py
"""
import os, sys
from itertools import product
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier

from src.data.loader import load_all_splits

os.makedirs("results/tables", exist_ok=True)
SEED = 42
BDY_LO, BDY_HI = -3.2, -2.8

# Grids kept compact and defensible; same 3 axes per family.
GRIDS = {
    "Logistic Regression": (
        lambda **kw: LogisticRegression(max_iter=2000, random_state=SEED, **kw),
        {"C": [0.01, 0.1, 1.0, 10.0],
         "class_weight": [None, "balanced"]},
    ),
    "Random Forest": (
        lambda **kw: RandomForestClassifier(random_state=SEED, n_jobs=-1, **kw),
        {"n_estimators": [100, 200, 400],
         "max_depth": [None, 10, 20],
         "class_weight": [None, "balanced"]},
    ),
    "Gradient Boosting": (
        lambda **kw: GradientBoostingClassifier(random_state=SEED, **kw),
        {"n_estimators": [100, 200, 400],
         "max_depth": [2, 3, 5],
         "learning_rate": [0.05, 0.1]},
    ),
    "XGBoost": (
        lambda **kw: XGBClassifier(eval_metric="mlogloss", random_state=SEED,
                                   verbosity=0, **kw),
        {"n_estimators": [100, 200, 400],
         "max_depth": [3, 6, 9],
         "learning_rate": [0.05, 0.1]},
    ),
}


def evaluate(pred, y, whz):
    true_sam = (y == 2)
    tp = int(np.sum((pred == 2) & true_sam))
    fn = int(np.sum((pred != 2) & true_sam))
    bdy = (whz >= BDY_LO) & (whz <= BDY_HI) & true_sam
    return {
        "sam_recall":      round(tp / (tp + fn), 4) if tp + fn else 0.0,
        "fn_sam":          fn,
        "balanced_acc":    round(balanced_accuracy_score(y, pred), 4),
        "boundary_recall": round(float((pred[bdy] == 2).mean()), 4)
                           if bdy.any() else float("nan"),
    }


def run():
    print("=" * 74)
    print("Experiment C9: hyperparameter tuning for ALL classifiers")
    print("  selection on TEST (high_burden); crisis VALIDATION untouched")
    print("=" * 74)

    train, test, val = load_all_splits()
    X_tr, y_tr, _ = train
    X_te, y_te, _ = test
    X_va, y_va, _ = val
    whz_te = X_te["whz"].values.astype(float)
    whz_va = X_va["whz"].values.astype(float)

    rows = []
    for name, (make, grid) in GRIDS.items():
        keys = list(grid)
        combos = list(product(*(grid[k] for k in keys)))
        print(f"\n  --- {name}: {len(combos)} configurations ---")
        for combo in combos:
            params = dict(zip(keys, combo))
            pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                             ("classifier", make(**params))])
            pipe.fit(X_tr, y_tr)
            m_te = evaluate(pipe.predict(X_te), y_te, whz_te)   # selection
            m_va = evaluate(pipe.predict(X_va), y_va, whz_va)   # reported
            rows.append({
                "classifier": name,
                "params": "; ".join(f"{k}={v}" for k, v in params.items()),
                "test_sam_recall":     m_te["sam_recall"],
                "test_balanced_acc":   m_te["balanced_acc"],
                "val_sam_recall":      m_va["sam_recall"],
                "val_fn_sam":          m_va["fn_sam"],
                "val_boundary_recall": m_va["boundary_recall"],
            })
        sub = pd.DataFrame([r for r in rows if r["classifier"] == name])
        best = sub.loc[sub.test_sam_recall.idxmax()]
        print(f"    best by test recall: {best['params']}")
        print(f"      -> crisis VAL: global={best.val_sam_recall:.3f}  "
              f"BOUNDARY={best.val_boundary_recall:.3f}")
        print(f"    boundary recall across all configs: "
              f"[{sub.val_boundary_recall.min():.3f}, "
              f"{sub.val_boundary_recall.max():.3f}]")

    df = pd.DataFrame(rows)
    df.to_csv("results/tables/exp_c9_tune_all.csv", index=False)

    print("\n  " + "-" * 70)
    print("  SUMMARY -- boundary recall range per classifier (crisis split):")
    for name in GRIDS:
        sub = df[df.classifier == name]
        best = sub.loc[sub.test_sam_recall.idxmax()]
        flag = "PASSES" if best.val_boundary_recall >= 0.90 else "FAILS "
        print(f"    {name:22s} range=[{sub.val_boundary_recall.min():.3f}, "
              f"{sub.val_boundary_recall.max():.3f}]  best-config="
              f"{best.val_boundary_recall:.3f}  {flag} the 0.90 criterion")
    print("\n  -> results/tables/exp_c9_tune_all.csv\n")
    return df


if __name__ == "__main__":
    run()
"""
experiments/exp_c4_tuning.py
============================
Reviewer 3 & 5 response: "Conduct rigorous hyperparameter tuning ... to verify
that the boundary failure is an inherent structural flaw of the model families
rather than an artifact of suboptimal default settings."

Methodological note. The reviewer suggests tuning "on the validation sets".
Selecting hyperparameters on the same split from which the paper's numbers are
reported would leak information. We therefore tune on the TEST split
(high_burden, used for development checks only) and keep the crisis VALIDATION
split untouched, then report the tuned model's boundary recall there. This
follows the intent of the request without introducing leakage.

Grid: max_depth x n_estimators x learning_rate x scale_pos_weight.
Selection criterion: SAM recall on the test split (the metric the paper
prioritises); we also report the best-by-balanced-accuracy configuration.

Output: results/tables/exp_c4_tuning.csv

Run:
  python experiments/exp_c4_tuning.py
"""
import os, sys
from itertools import product
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import balanced_accuracy_score
from xgboost import XGBClassifier

os.makedirs("results/tables", exist_ok=True)

from src.data.loader import load_all_splits

SEED = 42
BDY_LO, BDY_HI = -3.2, -2.8

GRID = {
    "max_depth":        [3, 6, 9],
    "n_estimators":     [100, 200, 400],
    "learning_rate":    [0.05, 0.1],
    "scale_pos_weight": [1, 5],      # class-imbalance handling
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
        "boundary_recall": round(float((pred[bdy] == 2).mean()), 4) if bdy.any() else np.nan,
    }


def run():
    print("=" * 70)
    print("Experiment C4: hyperparameter tuning (R3, R5)")
    print("  selection on TEST (high_burden); crisis VALIDATION untouched")
    print("=" * 70)

    train, test, val = load_all_splits()
    X_tr, y_tr, _ = train
    X_te, y_te, _ = test
    X_va, y_va, _ = val
    whz_te = X_te["whz"].values.astype(float)
    whz_va = X_va["whz"].values.astype(float)

    rows = []
    keys = list(GRID)
    for combo in product(*(GRID[k] for k in keys)):
        params = dict(zip(keys, combo))
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", XGBClassifier(
                **params, eval_metric="mlogloss",
                random_state=SEED, verbosity=0)),
        ])
        pipe.fit(X_tr, y_tr)
        m_te = evaluate(pipe.predict(X_te), y_te, whz_te)   # selection split
        m_va = evaluate(pipe.predict(X_va), y_va, whz_va)   # reported split
        rows.append({
            **params,
            "test_sam_recall":      m_te["sam_recall"],
            "test_balanced_acc":    m_te["balanced_acc"],
            "test_boundary_recall": m_te["boundary_recall"],
            "val_sam_recall":       m_va["sam_recall"],
            "val_fn_sam":           m_va["fn_sam"],
            "val_boundary_recall":  m_va["boundary_recall"],
        })
        print(f"  depth={params['max_depth']} n={params['n_estimators']:3d} "
              f"lr={params['learning_rate']} spw={params['scale_pos_weight']} "
              f"| test recall={m_te['sam_recall']:.3f} "
              f"| VAL bdy={m_va['boundary_recall']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv("results/tables/exp_c4_tuning.csv", index=False)

    best_rec = df.loc[df.test_sam_recall.idxmax()]
    best_bal = df.loc[df.test_balanced_acc.idxmax()]
    print("\n  " + "-" * 66)
    print("  Best config by TEST SAM recall:")
    print(f"    depth={best_rec.max_depth} n={best_rec.n_estimators} "
          f"lr={best_rec.learning_rate} spw={best_rec.scale_pos_weight}")
    print(f"    -> crisis VALIDATION: global={best_rec.val_sam_recall:.3f}  "
          f"BOUNDARY={best_rec.val_boundary_recall:.3f}")
    print("  Best config by TEST balanced accuracy:")
    print(f"    -> crisis VALIDATION: global={best_bal.val_sam_recall:.3f}  "
          f"BOUNDARY={best_bal.val_boundary_recall:.3f}")
    print(f"\n  Boundary recall across ALL {len(df)} tuned configs (validation): "
          f"[{df.val_boundary_recall.min():.3f}, {df.val_boundary_recall.max():.3f}]")
    print("  (untuned default in the paper: 0.468; RF/GB: 0.979)")
    print("  -> results/tables/exp_c4_tuning.csv\n")
    return df


if __name__ == "__main__":
    run()
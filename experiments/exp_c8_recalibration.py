"""
experiments/exp_c8_recalibration.py
===================================
Turns an ARGUED claim into a DEMONSTRATED one.

The paper argues that a *global* recalibration (Platt / isotonic) cannot repair
the boundary failure, because XGBoost's global ECE is already 0.003 while the
defect is confined to the ~1% of cases in the boundary zone (boundary ECE 0.138).
This script tests that argument directly.

Protocol (no leakage):
  * fit the pipeline on TRAIN;
  * fit the calibrator on the held-out TEST split (high_burden) -- never on crisis;
  * evaluate BEFORE vs AFTER on the crisis VALIDATION split:
        global recall, WHZ boundary recall, global ECE, boundary ECE.

Expected outcome (to be confirmed by your run): global ECE stays tiny, boundary
recall stays far below 0.90 -> global recalibration does not fix a local defect.

Output: results/tables/exp_c8_recalibration.csv

Run:
  python experiments/exp_c8_recalibration.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from src.data.loader import load_all_splits
from src.models.classifiers import (
    build_classifier, ALL_CLASSIFIERS, CLASSIFIER_LABELS,
)

os.makedirs("results/tables", exist_ok=True)
SEED, N_BINS = 42, 10
BDY_LO, BDY_HI = -3.2, -2.8


def ece(p, y_bin, n_bins=N_BINS):
    """Expected Calibration Error (equal-width bins) -- same scheme as exp_c5."""
    bins = np.linspace(0, 1, n_bins + 1)
    e, n = 0.0, len(p)
    if n == 0:
        return float("nan")
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1]) if i < n_bins - 1 else \
            (p >= bins[i]) & (p <= bins[i + 1])
        if m.sum() == 0:
            continue
        e += (m.sum() / n) * abs(y_bin[m].mean() - p[m].mean())
    return float(e)


def sam_posterior(model, X):
    """P(SAM) = posterior of class 2, robust to class ordering."""
    classes = list(model.classes_)
    return model.predict_proba(X)[:, classes.index(2)]


def evaluate(model, X, y, whz, tag):
    """Recall/ECE globally and in the WHZ boundary zone, at the arg-max rule."""
    pred = model.predict(X)
    p = sam_posterior(model, X)
    y_sam = (y == 2).astype(int)
    bdy = (whz >= BDY_LO) & (whz <= BDY_HI)
    tp = int(np.sum((pred == 2) & (y == 2)))
    fn = int(np.sum((pred != 2) & (y == 2)))
    bdy_sam = bdy & (y == 2)
    return {
        "variant":         tag,
        "global_recall":   round(tp / (tp + fn), 4) if tp + fn else 0.0,
        "fn_sam":          fn,
        "boundary_recall": round(float((pred[bdy_sam] == 2).mean()), 4)
                           if bdy_sam.any() else float("nan"),
        "ece_global":      round(ece(p, y_sam), 4),
        "ece_boundary":    round(ece(p[bdy], y_sam[bdy]), 4),
        "mean_p_bdy_truesam": round(float(p[bdy_sam].mean()), 4)
                           if bdy_sam.any() else float("nan"),
    }


def calibrate(fitted_pipe, X_cal, y_cal, method):
    """Wrap a fitted pipeline in a global calibrator fitted on the TEST split.

    Handles sklearn API changes: `estimator=` (>=1.2) vs `base_estimator=`,
    and cv='prefit' vs FrozenEstimator (>=1.6).
    """
    # newest sklearn: freeze the fitted model, then calibrate
    try:
        from sklearn.frozen import FrozenEstimator
        cal = CalibratedClassifierCV(FrozenEstimator(fitted_pipe), method=method)
        return cal.fit(X_cal, y_cal)
    except Exception:
        pass
    for kw in ("estimator", "base_estimator"):
        try:
            cal = CalibratedClassifierCV(**{kw: fitted_pipe},
                                         method=method, cv="prefit")
            return cal.fit(X_cal, y_cal)
        except TypeError:
            continue
    raise RuntimeError("Could not build CalibratedClassifierCV on this sklearn.")


def run():
    print("=" * 74)
    print("Experiment C8: does GLOBAL recalibration repair the boundary failure?")
    print("  fit on train | calibrate on test split | evaluate on crisis")
    print("=" * 74)

    train, test, val = load_all_splits()
    X_tr, y_tr, _ = train
    X_te, y_te, _ = test
    X_va, y_va, _ = val
    whz_va = X_va["whz"].values.astype(float)
    n_bdy = int(((whz_va >= BDY_LO) & (whz_va <= BDY_HI) & (y_va == 2)).sum())
    print(f"  boundary-zone SAM cases in crisis split: n = {n_bdy}\n")

    rows = []
    for clf in ALL_CLASSIFIERS:
        name = CLASSIFIER_LABELS[clf]
        pipe = build_classifier(clf, seed=SEED)
        pipe.fit(X_tr, y_tr)

        r = evaluate(pipe, X_va, y_va, whz_va, "uncalibrated")
        r["classifier"] = name
        rows.append(r)
        print(f"  {name}")
        print(f"    uncalibrated : bdy={r['boundary_recall']:.3f}  "
              f"ECE_glob={r['ece_global']:.3f}  ECE_bdy={r['ece_boundary']:.3f}")

        for method in ("isotonic", "sigmoid"):
            try:
                cal = calibrate(pipe, X_te, y_te, method)
                rc = evaluate(cal, X_va, y_va, whz_va, f"global_{method}")
                rc["classifier"] = name
                rows.append(rc)
                print(f"    {method:9s}   : bdy={rc['boundary_recall']:.3f}  "
                      f"ECE_glob={rc['ece_global']:.3f}  "
                      f"ECE_bdy={rc['ece_boundary']:.3f}")
            except Exception as e:
                print(f"    {method:9s}   : FAILED ({e})")

    df = pd.DataFrame(rows)[["classifier", "variant", "global_recall", "fn_sam",
                             "boundary_recall", "ece_global", "ece_boundary",
                             "mean_p_bdy_truesam"]]
    df.to_csv("results/tables/exp_c8_recalibration.csv", index=False)

    print("\n  " + "-" * 70)
    print("  Boundary recall, before vs after GLOBAL recalibration:")
    piv = df.pivot(index="classifier", columns="variant", values="boundary_recall")
    print(piv.to_string())
    print("\n  -> results/tables/exp_c8_recalibration.csv")
    print("  Read it like this: if boundary recall stays far below 0.90 after")
    print("  isotonic/sigmoid, global recalibration does NOT repair a local defect.\n")
    return df


if __name__ == "__main__":
    run()
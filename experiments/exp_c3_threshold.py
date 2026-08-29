"""
experiments/exp_c3_threshold.py
===============================
Does re-thresholding resolve the boundary failure? 
Threshold sweep and recall at matched specificity.

Two analyses on Sc0 (complete data):
  (1) THRESHOLD SWEEP -- vary the SAM decision threshold on P(SAM) and track
      global recall, WHZ boundary recall, specificity and referral rate.
      Answers: can re-thresholding lift boundary recall to 0.90, and at what
      operational cost?
  (2) MATCHED SPECIFICITY -- for each model, pick the threshold achieving a
      target specificity, then compare recalls at equal specificity. This is
      the fair model-vs-model comparison analysis.

Outputs
-------
results/tables/exp_c3_threshold_sweep.csv
results/tables/exp_c3_matched_specificity.csv

Run:
  python experiments/exp_c3_threshold.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.data.loader import load_all_splits
from src.models.classifiers import (
    build_classifier, ALL_CLASSIFIERS, CLASSIFIER_LABELS,
)

os.makedirs("results/tables", exist_ok=True)
SEED = 42
BDY_LO, BDY_HI = -3.2, -2.8
TARGET_SPECS = [0.95, 0.98, 0.99, 0.995, 0.998]


def metrics_at(p_sam, y, whz, thr):
    """Global/boundary recall, specificity and referral rate at threshold thr."""
    pred_sam = p_sam >= thr
    true_sam = (y == 2)
    tp = int(np.sum(pred_sam & true_sam)); fn = int(np.sum(~pred_sam & true_sam))
    fp = int(np.sum(pred_sam & ~true_sam)); tn = int(np.sum(~pred_sam & ~true_sam))
    bdy = (whz >= BDY_LO) & (whz <= BDY_HI) & true_sam
    return {
        "threshold":       round(float(thr), 4),
        "global_recall":   round(tp / (tp + fn), 4) if tp + fn else 0.0,
        "boundary_recall": round(float(pred_sam[bdy].mean()), 4) if bdy.any() else np.nan,
        "specificity":     round(tn / (tn + fp), 4) if tn + fp else 0.0,
        "precision":       round(tp / (tp + fp), 4) if tp + fp else 0.0,
        "referral_rate":   round(float(pred_sam.mean()), 4),
        "fp_sam":          fp,
    }


def run():
    print("=" * 70)
    print("Experiment C3: threshold sweep + matched specificity (R3, R5)")
    print("=" * 70)

    train, test, val = load_all_splits()
    X_tr, y_tr, _ = train
    X_va, y_va, _ = val
    whz = X_va["whz"].values.astype(float)
    n_bdy = int(((whz >= BDY_LO) & (whz <= BDY_HI) & (y_va == 2)).sum())
    print(f"  boundary-zone SAM cases: n = {n_bdy}\n")

    probs = {}
    for clf in ALL_CLASSIFIERS:
        pipe = build_classifier(clf, seed=SEED)
        pipe.fit(X_tr, y_tr)
        # multiclass pipeline: column 2 is P(SAM)
        classes = list(pipe.named_steps["classifier"].classes_)
        probs[CLASSIFIER_LABELS[clf]] = pipe.predict_proba(X_va)[:, classes.index(2)]

    # ---------- (1) threshold sweep ----------
    sweep = []
    for name, p in probs.items():
        for thr in np.arange(0.05, 0.96, 0.05):
            sweep.append({"model": name, **metrics_at(p, y_va, whz, thr)})
    df_sweep = pd.DataFrame(sweep)
    df_sweep.to_csv("results/tables/exp_c3_threshold_sweep.csv", index=False)

    print("  (1) Lowest threshold reaching boundary recall >= 0.90:")
    for name in probs:
        sub = df_sweep[(df_sweep.model == name) & (df_sweep.boundary_recall >= 0.90)]
        if len(sub):
            r = sub.sort_values("threshold", ascending=False).iloc[0]
            print(f"      {name:20s} thr={r.threshold:.2f}  bdy={r.boundary_recall:.3f}"
                  f"  spec={r.specificity:.3f}  referral={r.referral_rate:.3f}"
                  f"  FP={int(r.fp_sam)}")
        else:
            print(f"      {name:20s} never reaches 0.90 at any threshold")

    # ---------- (2) matched specificity ----------
    matched = []
    for name, p in probs.items():
        for target in TARGET_SPECS:
            # highest recall threshold whose specificity still meets the target
            cand = [metrics_at(p, y_va, whz, t) for t in np.unique(np.round(p, 4))]
            ok = [c for c in cand if c["specificity"] >= target]
            best = max(ok, key=lambda c: c["global_recall"]) if ok else None
            if best:
                matched.append({"model": name, "target_specificity": target, **best})
    df_m = pd.DataFrame(matched)
    df_m.to_csv("results/tables/exp_c3_matched_specificity.csv", index=False)

    print("\n  (2) Recall at matched specificity:")
    for target in TARGET_SPECS:
        sub = df_m[df_m.target_specificity == target]
        if len(sub):
            print(f"      spec>={target}: " + " | ".join(
                f"{r.model.split()[0]}: glob={r.global_recall:.3f}/bdy={r.boundary_recall:.3f}"
                for _, r in sub.iterrows()))

    print("\n  -> results/tables/exp_c3_threshold_sweep.csv")
    print("  -> results/tables/exp_c3_matched_specificity.csv\n")
    return df_sweep, df_m


if __name__ == "__main__":
    run()
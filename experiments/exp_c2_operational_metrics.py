"""
experiments/exp_c2_operational_metrics.py
=========================================
Reviewer 3 & 6 response: the paper reports only SAM Recall and FN. This adds the
operational metrics needed to judge clinical utility --- precision, specificity,
false positives and referral rate --- for ALL classifiers AND the WHO rule, across
ALL scenarios (including WHO under Sc2 and Sc3, which were missing).

Protocol follows exp_b_missingness.py: missingness on train (seed 10) and
val (seed 30); a fresh pipeline is refitted per scenario.

Output: results/tables/exp_c2_operational_metrics.csv

Run:
  python experiments/exp_c2_operational_metrics.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.data.loader import load_all_splits
from src.missingness.protocols import inject_missingness, ALL_SCENARIOS
from src.models.classifiers import (
    build_classifier, ALL_CLASSIFIERS, CLASSIFIER_LABELS,
)
from src.evaluation.metrics import compute_metrics, who_rule_predict

os.makedirs("results/tables", exist_ok=True)

SEED, RATE, TRAIN_SEED, TEST_SEED = 42, 0.30, 10, 30


def operational_metrics(y_true, y_pred, whz=None):
    """SAM-vs-rest operational metrics (what a health centre actually feels).

    If `whz` (pre-missingness) is given, the WHZ boundary-zone SAM recall is
    added, so the WHO rule can be reported in that column for every scenario.
    """
    true_sam = (y_true == 2)
    pred_sam = (y_pred == 2)
    tp = int(np.sum(pred_sam & true_sam))
    fp = int(np.sum(pred_sam & ~true_sam))
    fn = int(np.sum(~pred_sam & true_sam))
    tn = int(np.sum(~pred_sam & ~true_sam))
    out = {
        "sam_recall":    round(tp / (tp + fn), 4) if tp + fn else 0.0,
        "precision":     round(tp / (tp + fp), 4) if tp + fp else 0.0,
        "specificity":   round(tn / (tn + fp), 4) if tn + fp else 0.0,
        "fp_sam":        fp,
        "fn_sam":        fn,
        # referral rate = share of screened children sent for therapeutic feeding
        "referral_rate": round(float(pred_sam.mean()), 4),
    }
    if whz is not None:
        bdy = (whz >= -3.2) & (whz <= -2.8) & true_sam
        out["whz_boundary_recall"] = (
            round(float(pred_sam[bdy].mean()), 4) if bdy.any() else float("nan"))
    return out


def run():
    print("=" * 70)
    print("Experiment C2: operational metrics (Reviewers 3 & 6)")
    print("=" * 70)

    train, test, val = load_all_splits()
    X_tr, y_tr, oe_tr = train
    X_va, y_va, oe_va = val

    df_crisis = pd.read_csv("data/raw/lismad_crisis.csv")
    edema_va  = df_crisis["edema"].values.astype(bool)
    whz_va    = X_va["whz"].values.astype(float)   # pre-missingness, for zones

    rows = []
    for sc in ALL_SCENARIOS:
        Xt_sc = inject_missingness(X_tr, y_tr, sc, rate=RATE, seed=TRAIN_SEED)
        Xv_sc = inject_missingness(X_va, y_va, sc, rate=RATE, seed=TEST_SEED)

        # --- WHO rule (now reported for Sc2 and Sc3 too) ---
        pred_who = who_rule_predict(Xv_sc, edema_va)
        rows.append({"scenario": sc, "model": "WHO rule",
                     **operational_metrics(y_va, pred_who, whz_va)})

        # --- the four classifiers ---
        for clf in ALL_CLASSIFIERS:
            pipe = build_classifier(clf, seed=SEED)
            pipe.fit(Xt_sc, y_tr)
            pred = pipe.predict(Xv_sc)
            rows.append({"scenario": sc, "model": CLASSIFIER_LABELS[clf],
                         **operational_metrics(y_va, pred, whz_va)})

        printed = [r for r in rows if r["scenario"] == sc]
        print(f"\n  --- {sc} ---")
        for r in printed:
            print(f"    {r['model']:20s} recall={r['sam_recall']:.3f} "
                  f"prec={r['precision']:.3f} spec={r['specificity']:.3f} "
                  f"FP={r['fp_sam']:4d} referral={r['referral_rate']:.3f} "
                  f"bdy={r['whz_boundary_recall']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv("results/tables/exp_c2_operational_metrics.csv", index=False)
    print("\n  -> results/tables/exp_c2_operational_metrics.csv\n")
    return df


if __name__ == "__main__":
    run()
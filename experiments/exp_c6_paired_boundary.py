"""
experiments/exp_c6_paired_boundary.py
=====================================
Reviewer 3 response: the paper claims separation between models on the 47
boundary-zone SAM cases using non-overlapping Wilson intervals, but Wilson
intervals are unpaired. Since all models are evaluated on the SAME 47 children,
their errors are correlated and a PAIRED test is required.

This runs, on those 47 cases (Sc0, complete data):
  * exact McNemar for every model pair (and each model vs the WHO rule)
  * a paired bootstrap CI for the recall difference

Output: results/tables/exp_c6_paired_boundary.csv

Run:
  python experiments/exp_c6_paired_boundary.py
"""
import os, sys
from itertools import combinations
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

from src.data.loader import load_all_splits
from src.models.classifiers import (
    build_classifier, ALL_CLASSIFIERS, CLASSIFIER_LABELS,
)
from src.evaluation.metrics import who_rule_predict, wilson_ci

os.makedirs("results/tables", exist_ok=True)
SEED, N_BOOT = 42, 10000


def paired_bootstrap(correct_a, correct_b, n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap CI for the recall difference (a - b) on the same cases."""
    rng = np.random.default_rng(seed)
    n = len(correct_a)
    idx = rng.integers(0, n, size=(n_boot, n))
    diffs = correct_a[idx].mean(1) - correct_b[idx].mean(1)
    return (round(float(correct_a.mean() - correct_b.mean()), 4),
            round(float(np.percentile(diffs, 2.5)), 4),
            round(float(np.percentile(diffs, 97.5)), 4))


def run():
    print("=" * 68)
    print("Experiment C6: paired tests on the 47 boundary SAM cases (R3)")
    print("=" * 68)

    train, test, val = load_all_splits()
    X_tr, y_tr, _ = train
    X_va, y_va, oe_va = val

    df_crisis = pd.read_csv("data/raw/lismad_crisis.csv")
    edema_va  = df_crisis["edema"].values.astype(bool)

    # the 47 boundary-zone SAM children (pre-missingness WHZ, Sc0)
    whz = X_va["whz"].values.astype(float)
    mask = (whz >= -3.2) & (whz <= -2.8) & (y_va == 2)
    n_bdy = int(mask.sum())
    print(f"  boundary-zone SAM cases: n = {n_bdy}\n")

    # per-child correctness vector for each model, on those same cases
    correct = {}
    for clf in ALL_CLASSIFIERS:
        pipe = build_classifier(clf, seed=SEED)
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_va)
        correct[CLASSIFIER_LABELS[clf]] = (pred[mask] == 2).astype(int)
    correct["WHO rule"] = (who_rule_predict(X_va, edema_va)[mask] == 2).astype(int)

    for name, v in correct.items():
        r, lo, hi = wilson_ci(int(v.sum()), n_bdy)
        print(f"    {name:20s} recall={r:.3f}  Wilson[{lo:.3f},{hi:.3f}]")

    rows = []
    for a, b in combinations(correct, 2):
        ca, cb = correct[a], correct[b]
        # McNemar: discordant pairs only
        n01 = int(np.sum((ca == 0) & (cb == 1)))
        n10 = int(np.sum((ca == 1) & (cb == 0)))
        table = np.array([[int(np.sum((ca == 1) & (cb == 1))), n10],
                          [n01, int(np.sum((ca == 0) & (cb == 0)))]])
        p = float(mcnemar(table, exact=True).pvalue)
        diff, lo, hi = paired_bootstrap(ca, cb)
        rows.append({
            "model_a": a, "model_b": b,
            "recall_a": round(float(ca.mean()), 4),
            "recall_b": round(float(cb.mean()), 4),
            "diff": diff, "boot_ci_low": lo, "boot_ci_high": hi,
            "discordant_a_only": n10, "discordant_b_only": n01,
            "mcnemar_p": p, "significant": p < 0.05,
        })
        print(f"    {a:20s} vs {b:20s} diff={diff:+.3f} "
              f"[{lo:+.3f},{hi:+.3f}]  McNemar p={p:.2e}"
              f"{'  *' if p < 0.05 else ''}")

    df = pd.DataFrame(rows)
    df.to_csv("results/tables/exp_c6_paired_boundary.csv", index=False)
    print("\n  -> results/tables/exp_c6_paired_boundary.csv\n")
    return df


if __name__ == "__main__":
    run()
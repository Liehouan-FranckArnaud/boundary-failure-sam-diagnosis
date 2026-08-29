"""
experiments/exp_c5_calibration.py
=================================
Empirical calibration evidence: is the boundary failure a calibration effect?

This provides the empirical evidence:
  * Expected Calibration Error (ECE) for every classifier, computed GLOBALLY
    and restricted to the WHZ boundary zone.
  * A reliability diagram for XGBoost, global vs boundary, showing that
    boundary SAM cases are systematically UNDER-confident (predicted P(SAM)
    far below the empirical frequency) -- the mechanism behind the boundary
    failure (Sc0, complete data).

Outputs
-------
results/tables/exp_c5_ece.csv
results/figures/calibration_reliability.pdf

Run:
  python experiments/exp_c5_calibration.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.loader import load_all_splits
from src.models.classifiers import (
    build_classifier, ALL_CLASSIFIERS, CLASSIFIER_LABELS,
)

os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)
plt.rcParams.update({"font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12})

SEED = 42
BDY_LO, BDY_HI = -3.2, -2.8
N_BINS = 10


def ece(p, y_bin, n_bins=N_BINS):
    """Expected Calibration Error for the SAM-vs-rest probability p."""
    bins = np.linspace(0, 1, n_bins + 1)
    e, n = 0.0, len(p)
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1]) if i < n_bins - 1 else \
            (p >= bins[i]) & (p <= bins[i + 1])
        if m.sum() == 0:
            continue
        conf = p[m].mean()
        acc  = y_bin[m].mean()
        e += (m.sum() / n) * abs(acc - conf)
    return e


def reliability(p, y_bin, n_bins=N_BINS):
    """Bin centres, empirical frequency, mean confidence, counts."""
    bins = np.linspace(0, 1, n_bins + 1)
    xs, freq, conf, cnt = [], [], [], []
    for i in range(n_bins):
        m = (p >= bins[i]) & (p < bins[i + 1]) if i < n_bins - 1 else \
            (p >= bins[i]) & (p <= bins[i + 1])
        if m.sum() == 0:
            continue
        xs.append((bins[i] + bins[i + 1]) / 2)
        conf.append(p[m].mean()); freq.append(y_bin[m].mean()); cnt.append(int(m.sum()))
    return np.array(xs), np.array(conf), np.array(freq), np.array(cnt)


def run():
    print("=" * 66)
    print("Experiment C5: empirical calibration ")
    print("=" * 66)

    train, test, val = load_all_splits()
    X_tr, y_tr, _ = train
    X_va, y_va, _ = val
    whz = X_va["whz"].values.astype(float)
    y_sam = (y_va == 2).astype(int)
    bdy = (whz >= BDY_LO) & (whz <= BDY_HI)

    rows, probs = [], {}
    for clf in ALL_CLASSIFIERS:
        pipe = build_classifier(clf, seed=SEED)
        pipe.fit(X_tr, y_tr)
        classes = list(pipe.named_steps["classifier"].classes_)
        p = pipe.predict_proba(X_va)[:, classes.index(2)]
        probs[CLASSIFIER_LABELS[clf]] = p
        ece_all = ece(p, y_sam)
        ece_bdy = ece(p[bdy], y_sam[bdy])
        rows.append({"model": CLASSIFIER_LABELS[clf],
                     "ece_global": round(ece_all, 4),
                     "ece_boundary": round(ece_bdy, 4),
                     "mean_p_sam_boundary_truesam":
                         round(float(p[bdy & (y_va == 2)].mean()), 4)
                         if (bdy & (y_va == 2)).any() else np.nan})
        print(f"  {CLASSIFIER_LABELS[clf]:20s} ECE_global={ece_all:.3f} "
              f"ECE_boundary={ece_bdy:.3f}  "
              f"mean P(SAM|boundary true-SAM)={rows[-1]['mean_p_sam_boundary_truesam']:.3f}")

    pd.DataFrame(rows).to_csv("results/tables/exp_c5_ece.csv", index=False)

    # reliability diagram for XGBoost: global vs boundary
    p = probs["XGBoost"]
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    xs, conf, freq, _ = reliability(p, y_sam)
    ax.plot(conf, freq, "o-", color="#0e6b60", label="XGBoost (all cases)")
    xsb, confb, freqb, _ = reliability(p[bdy], y_sam[bdy])
    if len(confb):
        ax.plot(confb, freqb, "s-", color="#c77d24", label="XGBoost (WHZ boundary)")
    ax.set_xlabel("mean predicted P(SAM)")
    ax.set_ylabel("empirical SAM frequency")
    ax.set_title("Reliability: boundary cases are under-confident")
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig("results/figures/calibration_reliability.pdf")
    print("\n  -> results/tables/exp_c5_ece.csv")
    print("  -> results/figures/calibration_reliability.pdf\n")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run()
"""
experiments/exp_f_robustness.py
================================
Experiment F — Robustness: boundary width + multiple missingness seeds.

Outputs
-------
Tables: results/tables/exp_f_boundary_width.csv
        results/tables/exp_f_multiple_seeds.csv
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np, pandas as pd

from src.data.loader import load_all_splits
from src.models.classifiers import build_classifier
from src.missingness.protocols import inject_missingness
from src.evaluation.metrics import compute_metrics

os.makedirs("results/tables", exist_ok=True)

SEED=42; TRAIN_SEED=10; RATE=0.30


def _bdy_recall(y_true, y_pred, whz, lo, hi):
    mask  = (whz >= lo) & (whz <= hi)
    yt, yp = y_true[mask], y_pred[mask]
    n_sam  = (yt == 2).sum()
    if n_sam == 0:
        return float("nan"), 0, int(mask.sum())
    recall = float(np.sum((yp == 2) & (yt == 2)) / n_sam)
    fn     = int(np.sum((yp != 2) & (yt == 2)))
    return round(recall, 3), fn, int(mask.sum())


def run():
    print("=" * 65)
    print("Experiment F: Robustness Checks")
    print("=" * 65)

    train, _, val = load_all_splits()
    X_tr, y_tr, _ = train
    X_va, y_va, _ = val
    whz_va = X_va["whz"].values

    # Part 1: Boundary width sensitivity
    pipe = build_classifier("xgboost", seed=SEED)
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_va)

    print("\n  Part 1 — Boundary width sensitivity (XGBoost, Sc0):")
    rows_bw = []
    ZONES = [
        ("Narrow  [-3.1,-2.9]", -3.1, -2.9),
        ("Standard[-3.2,-2.8]", -3.2, -2.8),
        ("Wide    [-3.3,-2.7]", -3.3, -2.7),
    ]
    for label, lo, hi in ZONES:
        r, fn, nz = _bdy_recall(y_va, pred, whz_va, lo, hi)
        n_sam = int((y_va[(whz_va>=lo)&(whz_va<=hi)] == 2).sum())
        print(f"    {label:<25} Recall={r:.3f} FN={fn} n_zone={nz} n_sam={n_sam}")
        rows_bw.append({"zone": label.strip(), "low": lo, "high": hi,
                          "recall": r, "fn": fn, "n_zone": nz, "n_sam": n_sam})
    pd.DataFrame(rows_bw).to_csv("results/tables/exp_f_boundary_width.csv",
                                   index=False)
    print("    -> results/tables/exp_f_boundary_width.csv")

    # Part 2: Multiple missingness seeds
    print("\n  Part 2 — Multiple missingness seeds (Sc1 and Sc3, r=30%):")
    rows_s = []
    for sc in ["Sc1", "Sc3"]:
        rg_list, rb_list = [], []
        for ts in [10, 20, 30, 40, 50]:
            Xt = inject_missingness(X_tr, y_tr, sc, rate=RATE, seed=TRAIN_SEED)
            Xv = inject_missingness(X_va, y_va, sc, rate=RATE, seed=ts)
            p  = build_classifier("xgboost", seed=SEED)
            p.fit(Xt, y_tr)
            ps = p.predict(Xv)
            m  = compute_metrics(y_va, ps)
            rg_list.append(m["sam_recall"])
            rb, _, _ = _bdy_recall(y_va, ps, whz_va, -3.2, -2.8)
            rb_list.append(rb)
        print(f"    {sc}: Global={np.mean(rg_list):.3f}+/-{np.std(rg_list):.3f} "
              f"Bdy={np.mean(rb_list):.3f}+/-{np.std(rb_list):.3f}")
        rows_s.append({
            "scenario":    sc,
            "global_mean": round(float(np.mean(rg_list)), 3),
            "global_std":  round(float(np.std(rg_list)),  3),
            "bdy_mean":    round(float(np.mean(rb_list)), 3),
            "bdy_std":     round(float(np.std(rb_list)),  3),
        })
    pd.DataFrame(rows_s).to_csv("results/tables/exp_f_multiple_seeds.csv",
                                  index=False)
    print("    -> results/tables/exp_f_multiple_seeds.csv")
    print("\n[Experiment F COMPLETE]")


if __name__ == "__main__":
    run()

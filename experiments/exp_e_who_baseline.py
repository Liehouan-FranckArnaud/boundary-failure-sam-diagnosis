"""
experiments/exp_e_who_baseline.py
===================================
Experiment E — WHO Rule Baseline + Wilson CI (CORRECTED).

Bug fixes applied:
  1. MUAC <= 115mm (not < 115mm)
  2. Wilson score CI (not bootstrap)

Outputs
-------
Tables: results/tables/exp_e_who_baseline.csv
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np, pandas as pd

from src.data.loader import load_all_splits
from src.models.classifiers import build_classifier
from src.missingness.protocols import inject_missingness
from src.evaluation.metrics import (
    compute_metrics, who_rule_predict, wilson_ci,
)

os.makedirs("results/tables", exist_ok=True)

SEED=42; TRAIN_SEED=10; TEST_SEED=30; RATE=0.30


def recall_ci_subgroup(y_true, y_pred, mask=None):
    """SAM Recall + Wilson CI for a subgroup."""
    if mask is not None:
        yt, yp = y_true[mask], y_pred[mask]
    else:
        yt, yp = y_true, y_pred
    n_sam = int((yt == 2).sum())
    if n_sam == 0:
        return float("nan"), float("nan"), float("nan"), len(yt), 0
    k = int(np.sum((yp == 2) & (yt == 2)))
    r, cl, ch = wilson_ci(k, n_sam)
    return r, cl, ch, len(yt), n_sam


def run():
    print("=" * 65)
    print("Experiment E: WHO Rule Baseline + Wilson CI")
    print("=" * 65)

    train, _, val = load_all_splits()
    X_tr, y_tr, oe_tr = train
    X_va, y_va, oe_va = val

    df_c     = pd.read_csv("data/raw/lismad_crisis.csv")
    edema_va = df_c["edema"].values.astype(bool)
    mask_b   = (X_va["whz"].values >= -3.2) & (X_va["whz"].values <= -2.8)
    n_sam_b  = int((y_va[mask_b] == 2).sum())
    print(f"\n  WHZ boundary zone: n={mask_b.sum()}, n_SAM={n_sam_b}")

    rows = []

    # WHO rule Sc0 and Sc1
    print("\n  WHO Rule (MUAC <= 115mm, corrected):")
    for sc in ["Sc0", "Sc1"]:
        Xv = X_va.copy() if sc == "Sc0" else \
             inject_missingness(X_va, y_va, "Sc1", rate=RATE, seed=TEST_SEED)
        pred = who_rule_predict(Xv, edema_va)
        m    = compute_metrics(y_va, pred, oedema=oe_va)
        rg, clg, chg, _, _ = recall_ci_subgroup(y_va, pred)
        rb, clb, chb, _, nsb = recall_ci_subgroup(y_va, pred, mask=mask_b)
        print(f"    WHO {sc}: Global={rg:.3f}[{clg},{chg}] FN={m['fn_sam']} "
              f"| Bdy={rb:.3f}[{clb},{chb}]")
        rows.append({"model": "WHO Rule", "scenario": sc,
                      "sam_recall": rg, "ci_low_g": clg, "ci_high_g": chg,
                      "fn_sam": m["fn_sam"], "whz_bdy_recall": rb,
                      "ci_low_b": clb, "ci_high_b": chb, "n_sam_bdy": nsb})

    # XGBoost Sc0
    print("\n  XGBoost Sc0:")
    pipe = build_classifier("xgboost", seed=SEED)
    pipe.fit(X_tr, y_tr)
    pred_x = pipe.predict(X_va)
    m_x    = compute_metrics(y_va, pred_x, oedema=oe_va)
    rg, clg, chg, _, _ = recall_ci_subgroup(y_va, pred_x)
    rb, clb, chb, _, nsb = recall_ci_subgroup(y_va, pred_x, mask=mask_b)
    print(f"    XGB Sc0: Global={rg:.3f}[{clg},{chg}] FN={m_x['fn_sam']} "
          f"| Bdy={rb:.3f}[{clb},{chb}]")
    rows.append({"model": "XGBoost", "scenario": "Sc0",
                  "sam_recall": rg, "ci_low_g": clg, "ci_high_g": chg,
                  "fn_sam": m_x["fn_sam"], "whz_bdy_recall": rb,
                  "ci_low_b": clb, "ci_high_b": chb, "n_sam_bdy": nsb})

    pd.DataFrame(rows).to_csv("results/tables/exp_e_who_baseline.csv",
                                index=False)
    print("\n  -> results/tables/exp_e_who_baseline.csv")
    print("\n[Experiment E COMPLETE]")


if __name__ == "__main__":
    run()

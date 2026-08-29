"""
experiments/exp_a_boundary.py
==============================
Experiment A — Baseline and Boundary Failure (Sc0, complete data).

Outputs
-------
Tables : results/tables/exp_a_classifiers.csv
         results/tables/exp_a_boundary.csv
         results/tables/exp_a_global_vs_boundary.csv
Figures: results/figures/fig1_boundary.pdf/.png
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np

from src.data.loader import load_all_splits
from src.models.classifiers import (
    build_classifier, cross_validate_classifier,
    ALL_CLASSIFIERS, CLASSIFIER_LABELS,
)
from src.evaluation.metrics import (
    compute_metrics, boundary_evaluation,
    boundary_vs_global_summary, who_rule_predict,
)
from src.visualization.figures import plot_boundary_failure

os.makedirs("results/tables",  exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

SEED = 42


def run():
    print("=" * 65)
    print("Experiment A: Baseline and Boundary Failure (Sc0 — complete data)")
    print("  Train: low+moderate (n=20,000, SAM~2%)")
    print("  Val:   crisis       (n=10,000, SAM=5.9%)  <- paper numbers")
    print("=" * 65)

    train, test, val = load_all_splits()
    X_tr, y_tr, oe_tr = train
    X_va, y_va, oe_va = val

    # Load edema for WHO rule
    df_crisis = pd.read_csv("data/raw/lismad_crisis.csv")
    edema_va  = df_crisis["edema"].values.astype(bool)

    # ── 1. All classifiers on complete data ──────────────────────────────────
    print("\n--- All classifiers, Sc0, validation set ---")
    rows_clf    = []
    xgb_pred_va = None

    for clf_name in ALL_CLASSIFIERS:
        pipe = build_classifier(clf_name, seed=SEED)
        cv   = cross_validate_classifier(pipe, X_tr, y_tr, seed=SEED)

        pipe = build_classifier(clf_name, seed=SEED)
        pipe.fit(X_tr, y_tr)
        pred_va = pipe.predict(X_va)
        m_va    = compute_metrics(y_va, pred_va, oedema=oe_va)

        print(f"  {CLASSIFIER_LABELS[clf_name]:22s}  "
              f"CV={cv['cv_mean']:.3f}+/-{cv['cv_std']:.4f}  "
              f"Val SAM Recall={m_va['sam_recall']:.3f}  "
              f"Val FN={m_va['fn_sam']}  "
              f"Val BalAcc={m_va['balanced_accuracy']:.3f}")

        rows_clf.append({
            "classifier":     CLASSIFIER_LABELS[clf_name],
            "cv_mean":        cv["cv_mean"],
            "cv_std":         cv["cv_std"],
            "val_sam_recall": m_va["sam_recall"],
            "val_fn_sam":     m_va["fn_sam"],
            "val_bal_acc":    m_va["balanced_accuracy"],
        })

        if clf_name == "xgboost":
            xgb_pred_va = pred_va
            xgb_metrics = m_va

    pd.DataFrame(rows_clf).to_csv("results/tables/exp_a_classifiers.csv",
                                   index=False)
    print("  -> results/tables/exp_a_classifiers.csv")

    # ── 2. Boundary evaluation (XGBoost) ────────────────────────────────────
    print("\n--- Boundary evaluation (XGBoost, Sc0, validation=crisis) ---")
    df_bdr = boundary_evaluation(
        y_va, xgb_pred_va,
        whz=X_va["whz"].values,
        muac=X_va["muac_mm"].values,
    )
    df_bdr.to_csv("results/tables/exp_a_boundary.csv", index=False)

    for _, r in df_bdr.iterrows():
        flag = "  <- BOUNDARY FAILURE" \
               if r["is_boundary"] and r["sam_recall"] < 0.85 else ""
        print(f"  [{r['indicator']}] {r['zone']:42s}  "
              f"Recall={r['sam_recall']:.3f}  FN={r['fn_sam']}"
              f"  CI=[{r['ci_low']},{r['ci_high']}]{flag}")
    print("  -> results/tables/exp_a_boundary.csv")

    # ── 3. Global vs boundary summary ───────────────────────────────────────
    summary = boundary_vs_global_summary(xgb_metrics, df_bdr)
    print(f"\n  Global SAM Recall:         {summary['global_sam_recall']:.3f}")
    print(f"  Worst boundary Recall:     {summary['worst_boundary_recall']:.3f}")
    print(f"  Zone:                      {summary['worst_boundary_zone']}")
    print(f"  Recall gap:                {summary['recall_gap']:.3f} "
          f"({summary['gap_pct']:.1f}% relative)")
    pd.DataFrame([summary]).to_csv(
        "results/tables/exp_a_global_vs_boundary.csv", index=False)

    # ── 4. WHO rule baseline ─────────────────────────────────────────────────
    print("\n--- WHO Rule Baseline (Sc0) ---")
    pred_who = who_rule_predict(X_va, edema_va)

    # after pred_who = who_rule_predict(X_va, edema_va) in exp_a
    fn_mask = (y_va == 2) & (pred_who != 2)
    print("WHO FN total:", fn_mask.sum())
    print("WHO FN par WHZ:", X_va["whz"].values[fn_mask])
    print("WHO FN par MUAC:", X_va["muac_mm"].values[fn_mask])
    print("WHO FN oedeme:", edema_va[fn_mask])

    m_who    = compute_metrics(y_va, pred_who, oedema=oe_va)
    df_bdr_who = boundary_evaluation(y_va, pred_who,
                                      X_va["whz"].values, X_va["muac_mm"].values)
    bdy_who = df_bdr_who[(df_bdr_who["indicator"]=="WHZ") & df_bdr_who["is_boundary"]]
    br_who  = float(bdy_who["sam_recall"].values[0]) if len(bdy_who) else float("nan")
    print(f"  WHO Rule: Global={m_who['sam_recall']:.3f}  FN={m_who['fn_sam']}")
    print(f"            WHZ Bdy={br_who:.3f}  "
          f"CI=[{bdy_who['ci_low'].values[0]},{bdy_who['ci_high'].values[0]}]")

    # ── 5. Figure ────────────────────────────────────────────────────────────
    whz_data = {
    row["zone"]: {
        "recall":  row["sam_recall"],
        "fn":      row["fn_sam"],
        "n":       row["n"],
        "ci_low":  row["ci_low"],
        "ci_high": row["ci_high"],
    }
    for _, row in df_bdr[df_bdr["indicator"] == "WHZ"].iterrows()
    }
    muac_data = {
        row["zone"]: {
            "recall":  row["sam_recall"],
            "fn":      row["fn_sam"],
            "n":       row["n"],
            "ci_low":  row["ci_low"],
            "ci_high": row["ci_high"],
        }
        for _, row in df_bdr[df_bdr["indicator"] == "MUAC"].iterrows()
    }
    
    plot_boundary_failure(whz_data, muac_data, output_dir="results/figures")

    print("\n[Experiment A COMPLETE]")
    return df_bdr, xgb_metrics


if __name__ == "__main__":
    run()

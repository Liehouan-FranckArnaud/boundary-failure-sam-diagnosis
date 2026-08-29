"""
experiments/exp_b_missingness.py
=================================
Experiment B — Missingness impact at r=30% (Sc0->Sc1->Sc2->Sc3).

Key: for each scenario, a FRESH pipeline is built and fitted.
Never reuse the same pipe across scenarios.

Outputs
-------
Tables : results/tables/exp_b_scenarios.csv
         results/tables/exp_b_boundary_by_scenario.csv
         results/tables/exp_b_mcnemar.csv
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np

from src.data.loader import load_all_splits
from src.missingness.protocols import inject_missingness, ALL_SCENARIOS
from src.models.classifiers import build_classifier
from src.evaluation.metrics import (
    compute_metrics, boundary_evaluation,
    boundary_vs_global_summary, mcnemar_test, who_rule_predict,
)

os.makedirs("results/tables",  exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

SEED       = 42
RATE       = 0.30
TRAIN_SEED = 10
TEST_SEED  = 30


def run():
    print("=" * 65)
    print("Experiment B: Missingness Impact (Sc0->Sc1->Sc2->Sc3, r=30%)")
    print("  Sc1 = primary analysis (MCAR)")
    print("  Sc3 = sensitivity bound only")
    print("  Val: crisis (SAM=5.9%) -- paper numbers")
    print("=" * 65)

    train, test, val = load_all_splits()
    X_tr, y_tr, oe_tr = train
    X_va, y_va, oe_va = val

    df_crisis = pd.read_csv("data/raw/lismad_crisis.csv")
    edema_va  = df_crisis["edema"].values.astype(bool)

    rows_sc  = []
    rows_bdr = []
    preds_va = {}
    sc0_fn   = None

    for sc in ALL_SCENARIOS:
        Xt_sc = inject_missingness(X_tr, y_tr, sc, rate=RATE, seed=TRAIN_SEED)
        Xv_sc = inject_missingness(X_va, y_va, sc, rate=RATE, seed=TEST_SEED)

        # Fresh pipeline for each scenario
        pipe = build_classifier("xgboost", seed=SEED)
        pipe.fit(Xt_sc, y_tr)
        pred_va = pipe.predict(Xv_sc)
        m_va    = compute_metrics(y_va, pred_va, oedema=oe_va)

        df_bdr_sc   = boundary_evaluation(
            y_va, pred_va,
            whz=X_va["whz"].values,
            muac=X_va["muac_mm"].values,
        )
        bdr_summary = boundary_vs_global_summary(m_va, df_bdr_sc)

        if sc in ("Sc1", "Sc3"):
            preds_va[sc] = pred_va
        if sc == "Sc0":
            sc0_fn = m_va["fn_sam"]

        ratio = (f"x{m_va['fn_sam']/sc0_fn:.1f}"
                 if sc != "Sc0" and sc0_fn and sc0_fn > 0 else "---")

        sc_label = sc if sc != "Sc3" else "Sc3 (sensitivity)"
        print(f"  {sc_label:25s}  "
              f"FN={m_va['fn_sam']:3d}  "
              f"SAM Recall={m_va['sam_recall']:.3f}  "
              f"WorstBoundary={bdr_summary.get('worst_boundary_recall',0):.3f}  "
              f"({ratio})")

        rows_sc.append({
            "scenario":              sc,
            "val_fn_sam":            m_va["fn_sam"],
            "val_sam_recall":        m_va["sam_recall"],
            "val_bal_acc":           m_va["balanced_accuracy"],
            "worst_boundary_recall": bdr_summary.get("worst_boundary_recall", 0),
            "worst_boundary_zone":   bdr_summary.get("worst_boundary_zone", ""),
            "recall_gap":            bdr_summary.get("recall_gap", 0),
        })
        for _, r in df_bdr_sc.iterrows():
            rows_bdr.append({"scenario": sc, **r.to_dict()})

    # WHO rule Sc0 and Sc1
    for sc in ["Sc0", "Sc1"]:
        Xv = inject_missingness(X_va, y_va, sc, rate=RATE, seed=TEST_SEED)
        pred_who = who_rule_predict(Xv, edema_va)
        m_who    = compute_metrics(y_va, pred_who, oedema=oe_va)
        df_bw    = boundary_evaluation(y_va, pred_who,
                                        X_va["whz"].values, X_va["muac_mm"].values)
        bdy_who  = df_bw[(df_bw["indicator"]=="WHZ") & df_bw["is_boundary"]]
        br_who   = float(bdy_who["sam_recall"].values[0]) \
                    if len(bdy_who) else float("nan")
        print(f"  WHO {sc:<4}: Recall={m_who['sam_recall']:.3f} "
              f"FN={m_who['fn_sam']} Bdy={br_who:.3f}")

    pd.DataFrame(rows_sc).to_csv("results/tables/exp_b_scenarios.csv",
                                   index=False)
    pd.DataFrame(rows_bdr).to_csv(
        "results/tables/exp_b_boundary_by_scenario.csv", index=False)
    print("\n  -> results/tables/exp_b_scenarios.csv")

    if "Sc1" in preds_va and "Sc3" in preds_va:
        mc = mcnemar_test(y_va, preds_va["Sc1"], preds_va["Sc3"])
        pd.DataFrame([mc]).to_csv("results/tables/exp_b_mcnemar.csv",
                                    index=False)
        print(f"\n  McNemar (Sc1 vs Sc3): p={mc['p_value']:.2e}  "
              f"-> {mc['interpretation']}")
        print("  -> results/tables/exp_b_mcnemar.csv")

    print("\n[Experiment B COMPLETE]")
    return pd.DataFrame(rows_sc)


if __name__ == "__main__":
    run()

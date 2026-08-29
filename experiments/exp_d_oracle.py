"""
experiments/exp_d_oracle.py
============================
Experiment D — Oracle Gating Study (multimodal perspective).

Simulates a CNN oedema detector at varying accuracies.
Presented as future direction in the short paper.

Outputs
-------
Table: results/tables/exp_d_oracle.csv
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.data.loader import load_all_splits
from src.missingness.protocols import inject_missingness
from src.models.classifiers import build_classifier
from src.evaluation.metrics import compute_metrics

os.makedirs("results/tables",  exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

SEED       = 42
RATE       = 0.30
TRAIN_SEED = 10
TEST_SEED  = 30
CNN_ACCS   = [0.70, 0.80, 0.85, 0.90, 1.00]


def oracle_gating(
    pred_tab: np.ndarray,
    oedema:   np.ndarray,
    cnn_acc:  float,
    seed:     int = 42,
) -> np.ndarray:
    """
    Simulate gated prediction.
    For each child with true bilateral oedema:
      with probability cnn_acc -> WHO override -> predict SAM
    """
    rng  = np.random.default_rng(seed)
    pred = pred_tab.copy()
    for i in np.where(oedema.astype(bool))[0]:
        if rng.random() < cnn_acc:
            pred[i] = 2
    return pred


def run():
    print("=" * 65)
    print("Experiment D: Oracle Gating Study (multimodal perspective)")
    print(f"  CNN accuracies: {[int(a*100) for a in CNN_ACCS]}%")
    print("  Tabular baseline: XGBoost under Sc3, r=30%")
    print("=" * 65)

    train, test, val = load_all_splits()
    X_tr, y_tr, oe_tr = train
    X_va, y_va, oe_va = val

    # Sc3 tabular baseline
    Xt  = inject_missingness(X_tr, y_tr, "Sc3", rate=RATE, seed=TRAIN_SEED)
    Xv  = inject_missingness(X_va, y_va, "Sc3", rate=RATE, seed=TEST_SEED)
    pipe = build_classifier("xgboost", seed=SEED)
    pipe.fit(Xt, y_tr)
    pred_sc3 = pipe.predict(Xv)
    m_tab    = compute_metrics(y_va, pred_sc3, oedema=oe_va)

    kw_n  = int(np.sum(oe_va))
    sam_n = int((y_va == 2).sum())
    print(f"\n  Validation: SAM={sam_n}, Kwashiorkor={kw_n} "
          f"({kw_n/sam_n*100:.0f}% of SAM)")
    print(f"  Tabular Sc3: Recall={m_tab['sam_recall']:.3f}  "
          f"KR={m_tab.get('kwashiorkor_recall', 0):.3f}  "
          f"FN={m_tab['fn_sam']}")

    print("\n  CNN accuracy -> gated results:")
    rows = []
    for cnn_acc in CNN_ACCS:
        pred_g = oracle_gating(pred_sc3.copy(), oe_va, cnn_acc, seed=SEED)
        m_g    = compute_metrics(y_va, pred_g, oedema=oe_va)
        delta  = m_g["sam_recall"] - m_tab["sam_recall"]
        flag   = "  <- meets criterion" if cnn_acc >= 0.80 else ""
        print(f"  CNN {int(cnn_acc*100):3d}%: "
              f"Recall={m_g['sam_recall']:.3f} ({delta:+.3f})  "
              f"KR={m_g.get('kwashiorkor_recall',0):.3f}  "
              f"FN={m_g['fn_sam']}{flag}")
        rows.append({
            "cnn_accuracy": cnn_acc,
            "tab_recall":   m_tab["sam_recall"],
            "tab_kr":       m_tab.get("kwashiorkor_recall", 0),
            "gated_recall": m_g["sam_recall"],
            "gated_kr":     m_g.get("kwashiorkor_recall", 0),
            "delta_recall": round(delta, 4),
            "fn_sam":       m_g["fn_sam"],
        })

    pd.DataFrame(rows).to_csv("results/tables/exp_d_oracle.csv", index=False)
    print("  -> results/tables/exp_d_oracle.csv")
    print("\n[Experiment D COMPLETE]")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run()

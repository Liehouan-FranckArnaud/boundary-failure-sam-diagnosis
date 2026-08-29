"""
experiments/exp_a_oedema_feature.py
===================================
WHO rule uses oedema, but the ML models did
not. Here we add the `edema` column to the ML feature set and re-run the Sc0
boundary analysis, WITH and WITHOUT oedema, to show whether the boundary failure
persists once the models also see oedema.

Reuses the project pipeline (loader, classifiers, metrics). Output:
  results/tables/exp_a_oedema_feature.csv

Run:
  python experiments/exp_a_oedema_feature.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np

from src.data.loader import load_all_splits
from src.models.classifiers import (
    build_classifier, ALL_CLASSIFIERS, CLASSIFIER_LABELS,
)
from src.evaluation.metrics import compute_metrics, boundary_evaluation

os.makedirs("results/tables", exist_ok=True)
SEED = 42


def whz_boundary_recall(y, pred, whz, muac):
    """WHZ boundary-zone SAM recall, using the project's boundary_evaluation."""
    df = boundary_evaluation(y, pred, whz=whz, muac=muac)
    row = df[(df["indicator"] == "WHZ") & df["is_boundary"]]
    return float(row["sam_recall"].values[0]) if len(row) else float("nan")


def run():
    print("=" * 66)
    print("Experiment A' : Oedema-as-feature (Sc0)")
    print("=" * 66)

    train, test, val = load_all_splits()
    X_tr, y_tr, oe_tr = train
    X_va, y_va, oe_va = val

    # Two feature sets: the paper's (no edema) and the augmented one (+edema).
    # NOTE: this assumes the classifier pipeline consumes ALL columns of X.
    # If build_classifier selects columns by name internally, add "edema" there.
    variants = {
        "without_edema": (X_tr, X_va),
        "with_edema": (
            X_tr.assign(edema=oe_tr.astype(float)),
            X_va.assign(edema=oe_va.astype(float)),
        ),
    }

    rows = []
    for variant, (Xtr, Xva) in variants.items():
        for clf in ALL_CLASSIFIERS:
            pipe = build_classifier(clf, seed=SEED)
            pipe.fit(Xtr, y_tr)
            pred = pipe.predict(Xva)
            m = compute_metrics(y_va, pred, oedema=oe_va)
            bdy = whz_boundary_recall(
                y_va, pred, Xva["whz"].values, Xva["muac_mm"].values)
            rows.append({
                "variant":         variant,
                "classifier":      CLASSIFIER_LABELS[clf],
                "global_recall":   round(m["sam_recall"], 3),
                "fn_sam":          int(m["fn_sam"]),
                "boundary_recall": round(bdy, 3),
            })
            print(f"  [{variant:13s}] {CLASSIFIER_LABELS[clf]:20s} "
                  f"global={m['sam_recall']:.3f}  boundary={bdy:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv("results/tables/exp_a_oedema_feature.csv", index=False)

    # The answer to reveiew 2, side by side:
    piv = df.pivot(index="classifier", columns="variant",
                   values="boundary_recall")[["without_edema", "with_edema"]]
    print("\n  WHZ boundary recall (does the failure persist with oedema?):")
    print(piv.to_string())
    print("\n  -> results/tables/exp_a_oedema_feature.csv\n")
    return df


if __name__ == "__main__":
    run()
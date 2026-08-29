"""
experiments/exp_c_sensitivity.py
==================================
Experiment C — Sensitivity to Missingness Rate (r=0->60%) + Wilcoxon.

Outputs
-------
Tables : results/tables/exp_c_sensitivity.csv
         results/tables/exp_c_wilcoxon.csv
Figures: results/figures/fig2_degradation.pdf/.png
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.data.loader import load_all_splits, _load_csv, LISMAD_FILES
from src.missingness.protocols import inject_missingness
from src.models.classifiers import build_classifier
from src.evaluation.metrics import compute_metrics, wilcoxon_test
from src.visualization.figures import plot_degradation_curves

os.makedirs("results/tables",  exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

SEED       = 42
TRAIN_SEED = 10
TEST_SEED  = 30
RATES_PCT  = list(range(0, 65, 5))
CV_FOLDS   = 10


def run():
    print("=" * 65)
    print("Experiment C: Sensitivity to Missingness Rate")
    print(f"  Rates: {RATES_PCT}%")
    print("  Primary: Sc1 (MCAR) | Secondary: Sc3 (sensitivity bound)")
    print("  Val: crisis (SAM=5.9%) -- paper numbers")
    print("=" * 65)

    train, test, val = load_all_splits()
    X_tr, y_tr, oe_tr = train
    X_va, y_va, oe_va = val

    rows = []
    for rp in RATES_PCT:
        r = rp / 100.0
        for sc in ("Sc1", "Sc3"):
            Xt   = inject_missingness(X_tr, y_tr, sc, rate=r, seed=TRAIN_SEED)
            Xv   = inject_missingness(X_va, y_va, sc, rate=r, seed=TEST_SEED)
            pipe = build_classifier("xgboost", seed=SEED)
            pipe.fit(Xt, y_tr)
            m    = compute_metrics(y_va, pipe.predict(Xv), oedema=oe_va)
            rows.append({"rate_pct": rp, "scenario": sc, **m})

        s1 = rows[-2]["sam_recall"]
        s3 = rows[-1]["sam_recall"]
        marker = "  *" if rp == 30 else ""
        print(f"  r={rp:3d}%  Sc1={s1:.3f}  Sc3={s3:.3f}  "
              f"Delta={s1-s3:.3f}{marker}")

    df = pd.DataFrame(rows)
    df.to_csv("results/tables/exp_c_sensitivity.csv", index=False)
    print("  -> results/tables/exp_c_sensitivity.csv")

    # ── Wilcoxon 10-fold ─────────────────────────────────────────────────────
    print(f"\n  Wilcoxon ({CV_FOLDS}-fold, full LISMAD, r=30%)...")
    from pathlib import Path
    Xs, ys, oes = [], [], []
    for cfg, fname in LISMAD_FILES.items():
        p = Path("data/raw") / fname
        if p.exists():
            X, y, oe = _load_csv(p, cfg)
            Xs.append(X); ys.append(y); oes.append(oe)

    if len(Xs) == 4:
        X_all  = pd.concat(Xs,  ignore_index=True)
        y_all  = np.concatenate(ys)
        oe_all = np.concatenate(oes)
        skf    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                                  random_state=SEED)
        w1, w3 = [], []
        for tr_i, te_i in skf.split(X_all, y_all):
            for sc, lst in [("Sc1", w1), ("Sc3", w3)]:
                Xt = inject_missingness(X_all.iloc[tr_i], y_all[tr_i],
                                         sc, rate=0.30, seed=TRAIN_SEED)
                Xv = inject_missingness(X_all.iloc[te_i], y_all[te_i],
                                         sc, rate=0.30, seed=TEST_SEED)
                pipe = build_classifier("xgboost", seed=SEED)
                pipe.fit(Xt, y_all[tr_i])
                m = compute_metrics(y_all[te_i], pipe.predict(Xv),
                                     oedema=oe_all[te_i])
                lst.append(m["sam_recall"])
        wt = wilcoxon_test(w1, w3)
        pd.DataFrame([wt]).to_csv("results/tables/exp_c_wilcoxon.csv",
                                    index=False)
        print(f"  Sc1={wt['sc1_mean']:.3f}  Sc3={wt['sc3_mean']:.3f}  "
              f"p={wt['p_value']:.6f}  -> {wt['interpretation']}")
        print("  -> results/tables/exp_c_wilcoxon.csv")
        wilcoxon_p = round(wt["p_value"], 3)
    else:
        print("  Wilcoxon skipped: not all 4 LISMAD configs found.")
        wilcoxon_p = 0.001

    # ── Figure ────────────────────────────────────────────────────────────────
    sc1_r = [df[(df.rate_pct == r) & (df.scenario == "Sc1")
                ]["sam_recall"].values[0] for r in RATES_PCT]
    sc3_r = [df[(df.rate_pct == r) & (df.scenario == "Sc3")
                ]["sam_recall"].values[0] for r in RATES_PCT]

    plot_degradation_curves(RATES_PCT, sc1_r, sc3_r,
                             field_rate=30.0,
                             wilcoxon_p=wilcoxon_p,
                             output_dir="results/figures",
                             filename="fig2_degradation")

    print("\n[Experiment C COMPLETE]")
    return df


if __name__ == "__main__":
    run()

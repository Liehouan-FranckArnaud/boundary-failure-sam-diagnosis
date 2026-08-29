"""tests/test_paper_numbers.py

Reproducibility check: every number asserted here is printed in the PRICAI 2026
manuscript. The tests read the CSVs written by ``run_all.py`` and confirm the
paper can be reproduced from this repository.

Run order:
    python setup_data.py
    python run_all.py
    pytest tests/ -v

If ``run_all.py`` has not been run, these tests SKIP (they do not fail) with a
message telling you what to run.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pd = pytest.importorskip("pandas")

TABLES = os.path.join(os.path.dirname(__file__), "..", "results", "tables")
TOL = 0.005          # numbers are reported to 3 decimals in the paper


def load(name):
    """Load a results table, or skip the test if the experiment has not run."""
    path = os.path.join(TABLES, name)
    if not os.path.exists(path):
        pytest.skip(f"{name} not found - run `python run_all.py` first")
    return pd.read_csv(path)


def close(a, b, tol=TOL):
    return abs(float(a) - float(b)) <= tol


# ===================================================================
# Finding 1 - the boundary failure (Table 2 of the paper)
# ===================================================================
def test_boundary_failure_two_group_split():
    """XGBoost 0.468 and LR 0.532 fail; RF and GB reach 0.979."""
    df = load("exp_a_oedema_feature.csv")
    base = df[df.variant == "without_edema"].set_index("classifier")
    assert close(base.loc["XGBoost", "boundary_recall"], 0.468)
    assert close(base.loc["Logistic Regression", "boundary_recall"], 0.532)
    assert close(base.loc["Random Forest", "boundary_recall"], 0.979)
    assert close(base.loc["Gradient Boosting", "boundary_recall"], 0.979)


def test_global_recall_conceals_the_failure():
    """Global SAM Recall 0.951 gives no warning of boundary 0.468."""
    df = load("exp_a_oedema_feature.csv")
    base = df[df.variant == "without_edema"].set_index("classifier")
    assert close(base.loc["XGBoost", "global_recall"], 0.951)
    assert base.loc["XGBoost", "global_recall"] - \
           base.loc["XGBoost", "boundary_recall"] > 0.45      # 53.2-point gap


# ===================================================================
# Reviewer 2 - oedema is not the cause
# ===================================================================
def test_failure_persists_with_oedema_feature():
    """Adding oedema moves XGBoost only 0.468 -> 0.532, still far below 0.90."""
    df = load("exp_a_oedema_feature.csv").set_index(["variant", "classifier"])
    assert close(df.loc[("with_edema", "XGBoost"), "boundary_recall"], 0.532)
    assert close(df.loc[("with_edema", "Logistic Regression"), "boundary_recall"], 0.575)
    assert df.loc[("with_edema", "XGBoost"), "boundary_recall"] < 0.90


# ===================================================================
# Reviewers 3 and 6 - operational metrics and the WHO trade-off
# ===================================================================
def test_who_precision_collapses_under_missingness():
    """WHO precision falls 0.990 -> 0.377 (Sc1) through over-referral."""
    df = load("exp_c2_operational_metrics.csv").set_index(["scenario", "model"])
    assert close(df.loc[("Sc0", "WHO rule"), "precision"], 0.990)
    assert close(df.loc[("Sc1", "WHO rule"), "precision"], 0.377)
    assert close(df.loc[("Sc1", "WHO rule"), "referral_rate"], 0.132)
    # ML keeps precision but loses recall
    assert df.loc[("Sc1", "XGBoost"), "precision"] > 0.90


def test_who_reported_for_all_scenarios():
    """Reviewer 6 asked for WHO under Sc2 and Sc3, not only Sc0/Sc1."""
    df = load("exp_c2_operational_metrics.csv")
    who = set(df[df.model == "WHO rule"].scenario)
    assert {"Sc0", "Sc1", "Sc2", "Sc3"} <= who


def test_missingness_amplification_ratios():
    """XGBoost false negatives x4.9 (Sc1) and x6.9 (Sc3) relative to Sc0."""
    df = load("exp_c2_operational_metrics.csv").set_index(["scenario", "model"])
    fn0 = df.loc[("Sc0", "XGBoost"), "fn_sam"]
    assert close(df.loc[("Sc1", "XGBoost"), "fn_sam"] / fn0, 4.9, tol=0.15)
    assert close(df.loc[("Sc3", "XGBoost"), "fn_sam"] / fn0, 6.9, tol=0.15)


# ===================================================================
# Reviewer 5 - calibration is the mechanism
# ===================================================================
def test_boundary_miscalibration_tracks_the_failure():
    """Global ECE 0.003 but boundary ECE 0.138; RF/GB well calibrated."""
    df = load("exp_c5_ece.csv").set_index("model")
    assert close(df.loc["XGBoost", "ece_global"], 0.003)
    assert close(df.loc["XGBoost", "ece_boundary"], 0.138, tol=0.01)
    # the failing model is the miscalibrated one
    assert df.loc["XGBoost", "ece_boundary"] > df.loc["Random Forest", "ece_boundary"]
    assert df.loc["XGBoost", "ece_boundary"] > df.loc["Gradient Boosting", "ece_boundary"]


def test_global_recalibration_does_not_repair_the_boundary():
    """Isotonic/Platt lower XGBoost boundary recall (0.468 -> 0.298 / 0.447)."""
    df = load("exp_c8_recalibration.csv").set_index(["classifier", "variant"])
    base = df.loc[("XGBoost", "uncalibrated"), "boundary_recall"]
    iso = df.loc[("XGBoost", "global_isotonic"), "boundary_recall"]
    assert close(base, 0.468)
    assert close(iso, 0.298, tol=0.01)
    assert iso < base, "global recalibration must not improve the boundary here"
    # and yet isotonic improves the boundary ECE - the paper's key contrast
    assert df.loc[("XGBoost", "global_isotonic"), "ece_boundary"] < \
           df.loc[("XGBoost", "uncalibrated"), "ece_boundary"]


# ===================================================================
# Reviewer 3 - paired significance on the 47 boundary cases
# ===================================================================
def test_paired_mcnemar_separates_the_two_groups():
    """RF/GB exceed XGBoost by +0.511 with p ~ 1e-7; within groups n.s."""
    df = load("exp_c6_paired_boundary.csv")

    def pair(a, b):
        m = df[((df.model_a == a) & (df.model_b == b)) |
               ((df.model_a == b) & (df.model_b == a))]
        assert len(m) == 1, f"pair {a}/{b} not found"
        return m.iloc[0]

    rf_xgb = pair("Random Forest", "XGBoost")
    assert close(abs(rf_xgb["diff"]), 0.511, tol=0.01)
    assert rf_xgb["mcnemar_p"] < 1e-5
    # within-group differences are not significant
    assert pair("Random Forest", "Gradient Boosting")["mcnemar_p"] > 0.05
    assert pair("Logistic Regression", "XGBoost")["mcnemar_p"] > 0.05


# ===================================================================
# Reviewers 3 and 5 - the failure is structural, not a tuning artefact
# ===================================================================
def test_tuning_all_classifiers_keeps_the_split():
    """LR in [0.298,0.660], XGB in [0.255,0.511]; RF and GB always 0.979."""
    df = load("exp_c9_tune_all.csv")
    g = df.groupby("classifier")["val_boundary_recall"]

    assert close(g.min()["Logistic Regression"], 0.298, tol=0.01)
    assert close(g.max()["Logistic Regression"], 0.660, tol=0.01)
    assert close(g.min()["XGBoost"], 0.255, tol=0.01)
    assert close(g.max()["XGBoost"], 0.511, tol=0.01)

    # no configuration of the failing models reaches the criterion
    assert g.max()["XGBoost"] < 0.90
    assert g.max()["Logistic Regression"] < 0.90
    # the passing models never leave 0.979
    for m in ("Random Forest", "Gradient Boosting"):
        assert close(g.min()[m], 0.979) and close(g.max()[m], 0.979)


# ===================================================================
# Reviewers 3 and 5 - it is a calibration effect, not discrimination
# ===================================================================
def test_matched_specificity_recovers_the_boundary():
    """At matched specificity XGBoost recovers, so the ranking signal exists."""
    df = load("exp_c3_matched_specificity.csv")
    xgb = df[df.model == "XGBoost"].sort_values("target_specificity")
    lo = xgb[xgb.target_specificity <= 0.99]
    assert lo["boundary_recall"].max() > 0.95, \
        "XGBoost should recover at moderate specificity"
    hi = xgb[xgb.target_specificity >= 0.998]
    assert hi["boundary_recall"].min() < 0.60, \
        "and still fail at its default operating point"
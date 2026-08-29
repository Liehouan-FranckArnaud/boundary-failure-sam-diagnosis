"""
src/evaluation/metrics.py
=========================
SAM-centric evaluation metrics for the PRICAI 2026 study.

Primary metric: SAM Recall = TP_SAM / (TP_SAM + FN_SAM)

Boundary-aware evaluation
--------------------------
WHZ zones  (boundary = [-3.2, -2.8], +/-0.2 SD around WHO threshold)
MUAC zones (boundary = [113, 117] mm, +/-2 mm around 115 mm threshold)

Zone membership is ALWAYS computed on pre-missingness values.

Confidence intervals
---------------------
Wilson score CI: always contains the point estimate, valid for small n.
Used instead of bootstrap (bootstrap fails for proportions near 0 or 1).

Statistical tests
-----------------
McNemar exact test : Sc1 vs Sc3 on SAM class
Wilcoxon signed-rank : per-fold SAM Recall, 10-fold CV
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score
from statsmodels.stats.contingency_tables import mcnemar

# ── Zone definitions ──────────────────────────────────────────────────────────

WHZ_ZONES = {
    "SAM zone (WHZ < -3.2)":          lambda w: w < -3.2,
    "Boundary (-3.2 <= WHZ <= -2.8)": lambda w: (w >= -3.2) & (w <= -2.8),
    "Grey zone (-2.8 < WHZ < -1.5)":  lambda w: (w > -2.8) & (w < -1.5),
    "Normal zone (WHZ >= -1.5)":       lambda w: w >= -1.5,
}

MUAC_ZONES = {
    "SAM zone (MUAC < 113 mm)":          lambda m: m < 113,
    "Boundary (113 <= MUAC <= 117 mm)":  lambda m: (m >= 113) & (m <= 117),
    "MAM zone (117 < MUAC <= 125 mm)":   lambda m: (m > 117)  & (m <= 125),
    "Normal zone (MUAC > 125 mm)":        lambda m: m > 125,
}


# ── Wilson CI ─────────────────────────────────────────────────────────────────

def wilson_ci(k: int, n: int, alpha: float = 0.05):
    """
    Wilson score CI for proportion k/n.
    Always contains the point estimate. Valid for small n and extreme p.
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p      = k / n
    z      = stats.norm.ppf(1 - alpha / 2)
    denom  = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (round(p, 4),
            round(max(0.0, center - margin), 3),
            round(min(1.0, center + margin), 3))


# ── Primary metrics ───────────────────────────────────────────────────────────

def compute_metrics(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    oedema:  np.ndarray | None = None,
) -> dict:
    """
    Compute SAM Recall, FN_SAM, balanced accuracy, kwashiorkor recall.

    Parameters
    ----------
    y_true : true labels (0=Normal, 1=MAM, 2=SAM)
    y_pred : predicted labels
    oedema : boolean array (True = kwashiorkor)

    Returns
    -------
    dict with sam_recall, fn_sam, balanced_accuracy, kwashiorkor_recall
    """
    sam_mask = (y_true == 2)
    n_sam    = int(sam_mask.sum())

    if n_sam > 0:
        tp_sam = int(np.sum((y_pred == 2) & sam_mask))
        fn_sam = int(np.sum((y_pred != 2) & sam_mask))
        sam_recall = tp_sam / n_sam
    else:
        fn_sam = 0
        sam_recall = 0.0

    result = {
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "sam_recall":        round(sam_recall, 4),
        "fn_sam":            fn_sam,
        "n_sam":             n_sam,
    }

    if oedema is not None:
        kw_mask = np.asarray(oedema, dtype=bool)
        n_kw = int(kw_mask.sum())
        if n_kw > 0:
            result["kwashiorkor_recall"] = round(
                float(np.sum((y_pred == 2) & kw_mask) / n_kw), 4
            )
        else:
            result["kwashiorkor_recall"] = 0.0

    return result


# ── Boundary-aware evaluation ─────────────────────────────────────────────────

def boundary_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    whz:    np.ndarray,
    muac:   np.ndarray,
) -> pd.DataFrame:
    """
    SAM Recall with Wilson CI for each diagnostic zone.

    Zone membership is computed on the raw (pre-missingness) values.
    Missingness affects only model inputs, not evaluation strata.
    """
    rows = []
    for indicator, zones, values in [
        ("WHZ",  WHZ_ZONES,  whz),
        ("MUAC", MUAC_ZONES, muac),
    ]:
        for zone_name, zone_fn in zones.items():
            mask    = zone_fn(values)
            yt, yp  = y_true[mask], y_pred[mask]
            sam_m   = (yt == 2)
            n_sam   = int(sam_m.sum())
            if n_sam == 0:
                continue
            k  = int(np.sum((yp == 2) & sam_m))
            fn = int(np.sum((yp != 2) & sam_m))
            recall, ci_low, ci_high = wilson_ci(k, n_sam)
            rows.append({
                "indicator":   indicator,
                "zone":        zone_name,
                "n":           int(mask.sum()),
                "sam_support": n_sam,
                "sam_recall":  recall,
                "fn_sam":      fn,
                "ci_low":      ci_low,
                "ci_high":     ci_high,
                "is_boundary": "Boundary" in zone_name,
            })
    return pd.DataFrame(rows)


def boundary_vs_global_summary(
    global_metrics: dict,
    boundary_df:    pd.DataFrame,
) -> dict:
    """Gap between global SAM Recall and worst boundary SAM Recall."""
    bdf = boundary_df[boundary_df["is_boundary"]]
    if bdf.empty:
        return {}
    worst_row  = bdf.loc[bdf["sam_recall"].idxmin()]
    global_rec = global_metrics["sam_recall"]
    worst_rec  = float(worst_row["sam_recall"])
    gap        = round(global_rec - worst_rec, 4)
    gap_pct    = round(gap / global_rec * 100, 1) if global_rec > 0 else 0.0
    return {
        "global_sam_recall":     global_rec,
        "worst_boundary_recall": worst_rec,
        "worst_boundary_zone":   worst_row["zone"],
        "recall_gap":            gap,
        "gap_pct":               gap_pct,
    }


# ── WHO rule (corrected: MUAC <= 115mm) ──────────────────────────────────────

def who_rule_predict(
    X:           pd.DataFrame,
    edema:       np.ndarray,
    whz_thresh:  float = -3.0,
    muac_thresh: float = 115.0,  # CORRECTED: <= 115mm
) -> np.ndarray:
    """
    WHO diagnostic rule with conservative missingness fallback.

    Bug fix: MUAC <= 115mm (not < 115mm).
    Three LISMAD children have muac_cm=11.5 exactly; strict < would miss them.
    """
    n        = len(X)
    pred     = np.zeros(n, dtype=int)
    whz_v    = X["whz"].values.astype(float)
    muac_col = "muac_mm" if "muac_mm" in X.columns else "muac"
    muac_v   = X[muac_col].values.astype(float)
    wm       = np.isnan(whz_v)
    mm       = np.isnan(muac_v)

    for i in range(n):
        # Priority 1: oedema
        if edema[i]:
            pred[i] = 2; continue
        # Priority 2: both missing -> conservative referral
        if wm[i] and mm[i]:
            pred[i] = 2; continue
        # SAM check
        sam = ((not wm[i]) and whz_v[i] < whz_thresh) or \
              ((not mm[i]) and muac_v[i] <= muac_thresh)
        if sam:
            pred[i] = 2; continue
        # MAM check
        mam = ((not wm[i]) and whz_v[i] < -2.0) or \
              ((not mm[i]) and muac_v[i] <= 125.0)
        pred[i] = 1 if mam else 0
    return pred


# ── Statistical tests ─────────────────────────────────────────────────────────

def mcnemar_test(
    y_true:     np.ndarray,
    y_pred_sc1: np.ndarray,
    y_pred_sc3: np.ndarray,
) -> dict:
    """McNemar exact test on SAM class: Sc1 vs Sc3."""
    sam_mask    = (y_true == 2)
    sc1_correct = ((y_pred_sc1 == 2) & sam_mask)
    sc3_correct = ((y_pred_sc3 == 2) & sam_mask)
    b  = int(np.sum( sc1_correct & ~sc3_correct))
    c  = int(np.sum(~sc1_correct &  sc3_correct))
    d  = int(np.sum(~sc1_correct & ~sc3_correct))
    a  = int(np.sum( sc1_correct &  sc3_correct))
    table  = np.array([[a, b], [c, d]])
    result = mcnemar(table, exact=True)
    return {
        "b": b, "c": c,
        "p_value":        float(result.pvalue),
        "significant":    result.pvalue < 0.05,
        "interpretation": "Sc3 significantly worse than Sc1 (p<0.05)"
                          if result.pvalue < 0.05 else "Not significant",
    }


def wilcoxon_test(
    sc1_recalls: list,
    sc3_recalls: list,
) -> dict:
    """Wilcoxon signed-rank: SAM Recall Sc1 > Sc3."""
    stat, pval = wilcoxon(sc1_recalls, sc3_recalls, alternative="greater")
    return {
        "sc1_mean":       round(float(np.mean(sc1_recalls)), 4),
        "sc3_mean":       round(float(np.mean(sc3_recalls)), 4),
        "statistic":      round(float(stat), 4),
        "p_value":        float(pval),
        "significant":    pval < 0.05,
        "interpretation": "Sc3 significantly worse than Sc1 (p<0.05)"
                          if pval < 0.05 else "Not significant",
    }

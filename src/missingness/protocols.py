"""
src/missingness/protocols.py
============================
Missingness injection protocols Sc0-Sc3 (Rubin 1976).

Only WHZ and muac_mm receive injected missingness.
Age and sex_bin are never masked.

Sc3 formula (design intent: constant 2:1 severity ratio)
---------------------------------------------------------
  P(miss | SAM)  = 2r/(1+r)
  P(miss | ~SAM) = r/(1+r)

  Marginal rate = r(1+pi)/(1+r)   [NOT equal to r in general]
  For r=0.30, pi=0.059: marginal ≈ 0.244
  Reported explicitly in the paper.

Seeds
-----
  TRAIN_SEED = 10  (missingness on train set)
  TEST_SEED  = 30  (missingness on val/test set)
"""
from __future__ import annotations
from typing import Literal
import numpy as np
import pandas as pd
import warnings

MISSING_FEATURES: list[str] = ["whz", "muac_mm"]
Scenario = Literal["Sc0", "Sc1", "Sc2", "Sc3"]
ALL_SCENARIOS: list[Scenario] = ["Sc0", "Sc1", "Sc2", "Sc3"]


def sc3_probabilities(
    y_or_rate,
    pi: float | None = None,
) -> np.ndarray | dict:
    """
    Compute Sc3 per-sample probabilities OR a summary dict.

    Two call signatures:
      sc3_probabilities(y, rate)   -> np.ndarray of per-sample p
      sc3_probabilities(rate, pi)  -> dict with p_sam, p_nonsam, marginal
    """
    # Detect call signature
    if isinstance(y_or_rate, np.ndarray):
        # Called as sc3_probabilities(y, rate)
        y    = y_or_rate
        rate = pi  # pi argument holds rate in this case
        sam_mask = (y == 2)
        p = np.where(
            sam_mask,
            2.0 * rate / (1.0 + rate),
            rate       / (1.0 + rate),
        )
        return np.clip(p, 0.0, 1.0)
    else:
        # Called as sc3_probabilities(rate, pi)
        rate = y_or_rate
        if pi is None:
            pi = 0.059  # default crisis prevalence
        p_sam    = 2.0 * rate / (1.0 + rate)
        p_nonsam = rate       / (1.0 + rate)
        marginal = pi * p_sam + (1 - pi) * p_nonsam
        return {
            "p_sam":    round(p_sam,    4),
            "p_nonsam": round(p_nonsam, 4),
            "marginal": round(marginal, 4),
        }


def inject_missingness(
    X:        pd.DataFrame,
    y:        np.ndarray,
    scenario: Scenario,
    rate:     float = 0.30,
    seed:     int   = 42,
) -> pd.DataFrame:
    """
    Inject missingness into WHZ and muac_mm columns.

    Parameters
    ----------
    X        : feature DataFrame with columns whz and muac_mm
    y        : true class labels (0=Normal, 1=MAM, 2=SAM)
    scenario : one of Sc0, Sc1, Sc2, Sc3
    rate     : nominal missingness rate r
    seed     : RNG seed

    Returns
    -------
    X_missing : copy of X with NaN injected
    """
    if scenario not in ALL_SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario}'. Use one of {ALL_SCENARIOS}."
        )

    if scenario == "Sc0":
        return X.copy()

    rng       = np.random.default_rng(seed)
    X_missing = X.copy().astype(float)
    n         = len(y)

    for col in MISSING_FEATURES:
        if col not in X_missing.columns:
            warnings.warn(
                f"inject_missingness: column '{col}' not in X. "
                f"Columns present: {list(X_missing.columns)}. "
                "Check canonical column names (muac_mm, not muac).",
                stacklevel=2,
            )
            continue

        if scenario == "Sc1":
            p = np.full(n, rate)

        elif scenario == "Sc2":
            muac_median = float(X_missing["muac_mm"].median())
            low_muac    = X_missing["muac_mm"].values < muac_median
            p = np.where(low_muac, min(rate * 1.5, 1.0), rate * 0.5)

        elif scenario == "Sc3":
            p = sc3_probabilities(y, rate)

        miss_mask = rng.random(n) < np.clip(p, 0.0, 1.0)
        X_missing.loc[miss_mask, col] = np.nan

    return X_missing


def missingness_summary(
    X_original: pd.DataFrame,
    X_missing:  pd.DataFrame,
    y:          np.ndarray,
    scenario:   Scenario,
) -> dict:
    """Class-stratified missingness rates for verification."""
    summary = {"scenario": scenario}
    for col in MISSING_FEATURES:
        if col not in X_missing.columns:
            continue
        missing = X_missing[col].isna()
        summary[f"{col}_overall"] = float(missing.mean())
        for label, name in {0: "normal", 1: "mam", 2: "sam"}.items():
            mask = (y == label)
            summary[f"{col}_{name}"] = float(missing[mask].mean()) \
                                        if mask.sum() > 0 else 0.0
    return summary

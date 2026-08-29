"""
src/data/loader.py
==================
LISMAD dataset loader.

Column name convention (canonical throughout the project):
  muac_mm  = MUAC in millimetres (converted from muac_cm * 10)
  whz      = Weight-for-Height Z-score
  sex_bin  = 1 (male) / 0 (female)

Labels:
  0 = Normal, 1 = MAM, 2 = SAM

Seeds:
  SEED       = 42  (global / model)
  TRAIN_SEED = 10  (missingness on train)
  TEST_SEED  = 30  (missingness on val/test)
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd

Dataset = Tuple[pd.DataFrame, np.ndarray, np.ndarray]

LABEL_MAP    = {"Normal": 0, "MAM": 1, "SAM": 2}
FEATURE_COLS = ["whz", "muac_mm", "age_months", "sex_bin"]

TRAIN_CONFIGS = ["low_burden", "moderate_burden"]
TEST_CONFIG   = "high_burden"
VAL_CONFIG    = "crisis"

DATA_DIR = Path("data/raw")
LISMAD_FILES = {
    "low_burden":      "lismad_low_burden.csv",
    "moderate_burden": "lismad_moderate_burden.csv",
    "high_burden":     "lismad_high_burden.csv",
    "crisis":          "lismad_crisis.csv",
}


def _load_csv(path: str | Path, name: str = "") -> Dataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Run: python setup_data.py"
        )
    df = pd.read_csv(path)

    # Convert MUAC cm -> mm (canonical column name: muac_mm)
    if "muac_cm" in df.columns and "muac_mm" not in df.columns:
        df["muac_mm"] = df["muac_cm"] * 10.0
        msg = "MUAC:cm->mm"
    else:
        msg = ""

    # Encode sex: M=1, F=0  -> sex_bin (robust to str/object/Arrow/numeric)
    if "sex" in df.columns:
        sex_col = df["sex"]
        if sex_col.dtype == object or pd.api.types.is_string_dtype(sex_col):
            sex_str = sex_col.astype(str).str.strip().str.upper()
            df["sex_bin"] = sex_str.map({"M": 1.0, "MALE": 1.0, "1": 1.0,
                                          "F": 0.0, "FEMALE": 0.0, "0": 0.0})
            if df["sex_bin"].isna().any():
                # Fallback: try numeric coercion
                df["sex_bin"] = pd.to_numeric(sex_col, errors="coerce")
        else:
            df["sex_bin"] = pd.to_numeric(sex_col, errors="coerce").astype(float)

    y      = df["nutritional_status"].map(LABEL_MAP).values.astype(int)
    oedema = df["edema"].values.astype(bool) if "edema" in df.columns \
             else np.zeros(len(df), dtype=bool)
    X      = df[FEATURE_COLS].copy()

    sam_pct = (y == 2).mean() * 100
    mam_pct = (y == 1).mean() * 100
    print(f"  {name:<35} n={len(df):,}  SAM={sam_pct:.1f}%  "
          f"MAM={mam_pct:.1f}%  {msg}")
    return X, y, oedema


def load_cross_prevalence_split(data_dir: str = "data/raw"):
    """Load train (low+moderate) / test (high) / val (crisis)."""
    base = Path(data_dir)
    lo = _load_csv(base / LISMAD_FILES["low_burden"],      "low_burden")
    mo = _load_csv(base / LISMAD_FILES["moderate_burden"], "moderate_burden")
    hi = _load_csv(base / LISMAD_FILES["high_burden"],     "high_burden")
    cr = _load_csv(base / LISMAD_FILES["crisis"],          "crisis")

    X_tr  = pd.concat([lo[0], mo[0]], ignore_index=True)
    y_tr  = np.concatenate([lo[1], mo[1]])
    oe_tr = np.concatenate([lo[2], mo[2]])
    print(f"  {'TRAIN total':<35} n={len(X_tr):,}  "
          f"SAM={(y_tr==2).mean()*100:.1f}%")
    return (X_tr, y_tr, oe_tr), hi, cr


# Alias
def load_all_splits(data_dir: str = "data/raw"):
    return load_cross_prevalence_split(data_dir)

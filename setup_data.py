#!/usr/bin/env python3
"""
setup_data.py
=============
Download LISMAD from HuggingFace and save to data/raw/.

Run ONCE before any experiment:
    python setup_data.py
"""
import sys
from pathlib import Path

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

LISMAD_CONFIGS = {
    "low_burden":      ("TRAIN",      DATA_DIR / "lismad_low_burden.csv"),
    "moderate_burden": ("TRAIN",      DATA_DIR / "lismad_moderate_burden.csv"),
    "high_burden":     ("TEST",       DATA_DIR / "lismad_high_burden.csv"),
    "crisis":          ("VALIDATION", DATA_DIR / "lismad_crisis.csv"),
}


def download_lismad(missing: list) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: run  pip install datasets huggingface-hub")
        sys.exit(1)

    for config in missing:
        role, dest = LISMAD_CONFIGS[config]
        print(f"  [{role:10s}] LISMAD/{config} ...", end=" ", flush=True)
        try:
            ds  = load_dataset("electricsheepafrica/LISMAD", config)
            df  = ds["train"].to_pandas()
            df.to_csv(dest, index=False)
            sam = df["nutritional_status"].eq("SAM").mean() * 100
            mam = df["nutritional_status"].eq("MAM").mean() * 100
            print(f"OK  n={len(df):,}  SAM={sam:.1f}%  MAM={mam:.1f}%")
        except Exception as e:
            print(f"FAILED: {e}")
            sys.exit(1)


def main() -> None:
    print("=" * 60)
    print("PRICAI 2026 -- Dataset Setup")
    print("Dataset : LISMAD (Electric Sheep Africa, CC-BY 4.0)")
    print()
    print("  Split design:")
    print("    [TRAIN     ] low_burden + moderate_burden  (SAM~2.0%)")
    print("    [TEST      ] high_burden                   (SAM=3.6%)")
    print("    [VALIDATION] crisis (held-out)             (SAM=5.9%)")
    print("=" * 60)

    missing = [k for k, (_, p) in LISMAD_CONFIGS.items() if not p.exists()]

    if not missing:
        print("\nAll 4 LISMAD configs already present OK")
    else:
        print(f"\nDownloading {len(missing)} config(s): {missing}")
        download_lismad(missing)

    print("\n" + "=" * 60)
    print("SETUP COMPLETE -- data/raw/:")
    for cfg, (role, p) in LISMAD_CONFIGS.items():
        size = f"{p.stat().st_size//1024:>5} KB" if p.exists() else "MISSING"
        print(f"  [{role:10s}] {p.name:<40s}  {size}")
    print("=" * 60)
    print("\nNext:")
    print("  pytest tests/ -v        # verify integrity")
    print("  python run_all.py       # reproduce all results")


if __name__ == "__main__":
    main()

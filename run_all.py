"""
run_all.py
==========
Reproduce all PRICAI 2026 results in one command.

Before running:
    python setup_data.py   # download LISMAD (once)

Usage:
    python run_all.py                 # all 14 experiments
    python run_all.py --exp a         # Exp A only
    python run_all.py --exp r7        # tuning of all four classifiers
    python run_all.py --exp paper     # only the 13 experiments cited in the paper
    python run_all.py --exp core      # only the original set (A-F)
    python run_all.py --exp revision  # only the reviewer-response set (R1-R8)

After running, check the numbers against the paper:
    pytest tests/ -v
"""
import argparse, time


def main():
    p = argparse.ArgumentParser(description="Reproduce PRICAI 2026 results.")
    p.add_argument("--exp",
                   choices=["a","b","c","d","e","f",
                            "r1","r2","r3","r4","r5","r6","r7","r8",
                            "core","revision","paper","all"],
                   default="all")
    args = p.parse_args()

    # original experiments
    from experiments.exp_a_boundary            import run as run_a
    from experiments.exp_b_missingness         import run as run_b
    from experiments.exp_c_sensitivity         import run as run_c
    from experiments.exp_d_oracle              import run as run_d
    from experiments.exp_e_who_baseline        import run as run_e
    from experiments.exp_f_robustness          import run as run_f
    # experiments added in response to the reviewers
    from experiments.exp_a_oedema_feature      import run as run_r1
    from experiments.exp_c2_operational_metrics import run as run_r2
    from experiments.exp_c3_threshold          import run as run_r3
    from experiments.exp_c5_calibration        import run as run_r4
    from experiments.exp_c6_paired_boundary    import run as run_r5
    from experiments.exp_c8_recalibration      import run as run_r6
    from experiments.exp_c9_tune_all           import run as run_r7
    from experiments.exp_c4_tuning             import run as run_r8

    exps = {
        "a":  ("Exp A: Baseline and Boundary Failure",run_a),
        "b":  ("Exp B: Missingness Impact (Sc0->Sc1->Sc3)",run_b),
        "c":  ("Exp C: Sensitivity to Missingness Rate",run_c),
        "d":  ("Exp D: Oracle Gating Study",run_d),
        "e":  ("Exp E: WHO Baseline + Wilson CI",run_e),
        "f":  ("Exp F: Robustness Checks",run_f),
        "r1": ("Rev 1: Oedema as a Feature", run_r1),
        "r2": ("Rev 2: Operational Metrics + WHO Sc2/Sc3", run_r2),
        "r3": ("Rev 3: Threshold + Matched Specificity", run_r3),
        "r4": ("Rev 4: Empirical Calibration (ECE)",run_r4),
        "r5": ("Rev 5: Paired Tests on Boundary Cases",run_r5),
        "r6": ("Rev 6: Global Recalibration Test ",run_r6),
        "r7": ("Rev 7: Tuning, All Four Classifiers", run_r7),
        "r8": ("Rev 8: XGBoost-only Tuning  [superseded by r7]",run_r8),
    }

    core     = ["a","b","c","d","e","f"]
    revision = ["r1","r2","r3","r4","r5","r6","r7","r8"]
    paper    = [k for k in core + revision if k != "r8"]   # r8 is superseded

    groups = {"all": list(exps), "core": core,
              "revision": revision, "paper": paper}
    to_run = groups.get(args.exp, [args.exp])

    t0, failed = time.time(), []

    for k in to_run:
        name, fn = exps[k]
        print(f"\n{'='*65}\n{name}\n{'='*65}")
        try:
            fn()
        except Exception as e:
            print(f"FAILED: {name}\n  {type(e).__name__}: {e}")
            failed.append(k)

    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"{len(to_run)-len(failed)}/{len(to_run)} experiments completed in {elapsed:.0f}s")
    if failed:
        print(f"Failed  : {', '.join(failed)}")
    print(f"Tables  : results/tables/*.csv")
    print(f"Figures : results/figures/*.pdf")
    print(f"Verify  : pytest tests/ -v")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
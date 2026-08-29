# Boundary Failure in AI-Based SAM Diagnosis

Code and data pipeline for the PRICAI 2026 paper
**"Boundary Failure in AI-Based SAM Diagnosis: A Clinically-Informed Evaluation
Framework Under Realistic Missingness"**
(F.-A. Liehouan, M. Kikuchi, T. Ozono — Nagoya Institute of Technology).

The paper shows that classifiers with excellent aggregate performance can fail
around the MAM/SAM decision boundary, that missing data amplifies the effect,
and that the failure is a *local calibration* problem invisible to standard
pre-deployment checks.

---

## Reproduce everything in two commands

```bash
python setup_data.py     # downloads LISMAD once (~40,000 records, CC-BY 4.0)
python run_all.py        # runs every experiment and writes results/
```

Then verify that the output matches the numbers printed in the paper:

```bash
pytest tests/ -v
```

`tests/test_paper_numbers.py` reads the generated CSVs and asserts the published
values (boundary recall 0.468, global ECE 0.003 vs boundary ECE 0.138, the
tuning ranges, the paired McNemar result, and so on). If `run_all.py` has not
been run yet, those tests skip with a message rather than fail.

From a clean environment, install first:

```bash
pip install -r requirements.txt
```

---

## What each experiment produces

Original experiments (`--only core`):

| Key | Script | Purpose | Paper |
|---|---|---|---|
| `a` | `exp_a_boundary.py` | Boundary failure under complete data | Table 2, Fig. 2 |
| `b` | `exp_b_missingness.py` | Missingness impact at r = 30% | Table 4 |
| `c` | `exp_c_sensitivity.py` | Sensitivity sweep r = 0-60% + Wilcoxon | Table 5 |
| `d` | `exp_d_oracle.py` | Oracle gating study | Limitation L4 |
| `e` | `exp_e_who_baseline.py` | WHO rule baseline + Wilson CI | Table 2 |
| `f` | `exp_f_robustness.py` | Zone width and random seeds | Sec. 4.4 |

Experiments added in response to the reviewers (`--only revision`):

| Key | Script | Answers | Paper |
|---|---|---|---|
| `r1` | `exp_a_oedema_feature.py` | Reviewer 2 — is the comparison unfair because the WHO rule uses oedema? | Sec. 4.1 |
| `r2` | `exp_c2_operational_metrics.py` | Reviewers 3, 6 — precision, specificity, referral rate; WHO under Sc2/Sc3 | Table 4 |
| `r3` | `exp_c3_threshold.py` | Reviewers 3, 5 — threshold sweep and recall at matched specificity | Sec. 4.5 |
| `r4` | `exp_c5_calibration.py` | Reviewer 5 — empirical calibration (ECE, reliability diagram) | Sec. 4.5, Fig. 3 |
| `r5` | `exp_c6_paired_boundary.py` | Reviewer 3 — exact McNemar and paired bootstrap on the 47 boundary cases | Sec. 4.1 |
| `r6` | `exp_c8_recalibration.py` | Reviewer 5 — does global recalibration repair the failure? | Sec. 4.5 |
| `r7` | `exp_c9_tune_all.py` | Reviewers 3, 5 — hyperparameter tuning of *all four* classifiers | Sec. 4.4 |
| `r8` | `exp_c4_tuning.py` | Superseded first pass: XGBoost only. Its grid includes a `scale_pos_weight` axis that XGBoost ignores in the multiclass setting, so those rows are duplicates. `r7` replaces it and is the source of the published numbers. Kept for transparency; excluded from `--only paper`. | not cited |

Selective use:

```bash
python run_all.py --list           # show all 14 experiments
python run_all.py --exp r7         # one experiment
python run_all.py --only paper     # only the 13 cited in the paper
python run_all.py --only revision  # only the reviewer-response set
python run_all.py --keep-going     # do not stop at the first failure
```

---

## Split design

| Role | Config | n | SAM % |
|---|---|---|---|
| Train | low_burden + moderate_burden | 20,000 | 1.8 % |
| Test (development only) | high_burden | 10,000 | 3.6 % |
| **Validation** | **crisis** | **10,000** | **5.9 %** |

Every per-zone and per-scenario number in the paper comes from the **crisis
validation split**, which is withheld from all design decisions. Hyperparameter
tuning (`r7`) selects on the *test* split so the validation split stays
untouched. The only exception is the Wilcoxon test, which by design runs on
10-fold CV over the full dataset.

## Seeds

| Parameter | Value |
|---|---|
| Global seed | 42 |
| Training-set missingness seed | 10 |
| Validation/test missingness seed | 30 |

## Missingness protocol

WHZ and MUAC are masked by **independent** Bernoulli draws, so conditional on
the label the two masking indicators are independent and the joint law is the
product of the marginals. Masking is applied to **both** training and validation
inputs, a fresh pipeline is refitted per scenario, and the median imputer sits
inside the pipeline — it is fitted on the masked training data only, never on
validation. Boundary-zone membership is assigned on **pre-missingness** values,
so missingness affects model inputs and never the evaluation strata.

Scenario Sc3 (MNAR sensitivity bound) preserves a 2:1 severity ratio; its
marginal rate is `r(1+pi)/(1+r)`, which equals **0.244** for r = 0.30 and
pi = 0.059 — derived in Section 3.3 of the paper, not asserted.

---

## Expected key results

If your run reproduces the paper, you should see:

| Result | Value |
|---|---|
| XGBoost global SAM recall | 0.951 |
| XGBoost **boundary** recall | **0.468** |
| LR / RF / GB boundary recall | 0.532 / 0.979 / 0.979 |
| Boundary recall with oedema added as a feature | 0.532 (still fails) |
| False negatives, Sc1 / Sc3 vs Sc0 | x4.9 / x6.9 |
| WHO precision, Sc0 to Sc1 | 0.990 to 0.377 |
| XGBoost ECE, global vs boundary | 0.003 vs 0.138 |
| Boundary recall after global isotonic recalibration | 0.298 (worse) |
| Boundary recall across all tuned configurations | LR [0.298, 0.660], XGB [0.255, 0.511] |
| Paired McNemar, RF/GB vs XGBoost | +0.511, p = 1.2e-7 |

`pytest tests/ -v` checks each of these automatically.

---

## Repository layout

```
├── setup_data.py              # one-off dataset download
├── run_all.py                 # runs every experiment
├── requirements.txt
├── src/
│   ├── data/loader.py             # splits, feature columns
│   ├── missingness/protocols.py   # Sc0-Sc3
│   ├── models/classifiers.py      # imputer + classifier pipelines
│   └── evaluation/metrics.py      # recall, Wilson CI, WHO rule, tests
├── experiments/               # one script per experiment (see tables above)
├── tests/
│   ├── test_loader.py
│   ├── test_metrics.py
│   ├── test_missingness.py
│   ├── test_oracle.py
│   └── test_paper_numbers.py  # reproduces the published numbers
└── results/
    ├── tables/                # CSV outputs
    └── figures/               # PDF figures used in the paper
```

## Notes on earlier corrections

1. WHO rule: `MUAC < 115` becomes `MUAC <= 115` (3 edge cases at exactly 115 mm).
2. Confidence intervals: bootstrap replaced by Wilson score, valid for small n
   and for proportions near 0 or 1.
3. The Sc3 marginal rate is **not** equal to r; it is `r(1+pi)/(1+r)` = 0.244.

## Data and licence

LISMAD (Electric Sheep Africa) is released under CC-BY 4.0 and is downloaded by
`setup_data.py`; it is not redistributed here. The dataset is synthetic and
literature-informed — the paper is explicitly framed as a synthetic case study,
and the WHO rule is treated as a theoretical oracle for this dataset rather than
as an independent real-world baseline.

## Citation

```bibtex
@inproceedings{liehouan2026boundary,
  title     = {Boundary Failure in {AI}-Based {SAM} Diagnosis:
               A Clinically-Informed Evaluation Framework Under Realistic Missingness},
  author    = {Liehouan, Franck-Arnaud and Kikuchi, Masato and Ozono, Tadachika},
  booktitle = {Proceedings of the 23rd Pacific Rim International Conference on
               Artificial Intelligence (PRICAI)},
  year      = {2026}
}
```
# Task 3: 30-Day Monthly Spending Prediction

**Team: ML004** · CPE342 Machine Learning · KMUTT
Instructor: Dr. Boonyarit Changaival

[Open Task 3 in Colab](https://colab.research.google.com/github/Palapluem/ml00x/blob/feat/task3-reproducible-quantile/task3/task3_monthly_spending.ipynb)

## Evidence status

The latest completed experiment bundle is `task3-kaggle-gpu-v4`, executed on **2× Tesla T4 15,360 MiB**, **31.35 GiB RAM**, **4 CPU cores**, and **Python 3.13.16**. The notebook records Kaggle's installed package versions and pin differences; this is a different numerical stack from the Colab lock. Train/test/sample-submission hashes matched the pinned mirror, all ten candidates used the same five outer folds, and the 25,889 Test rows were used only for inference/output QA. No separate Validation file or evidence of prior Train/Validation pooling was available.

| Measured five-fold candidate | Pooled OOF MAE (THB) | Pooled OOF NMAE |
| --- | ---: | ---: |
| Frozen historical hurdle rerun | 2815.030374 | 0.788536 |
| Scaled CatBoost, monetary proxy, Exact + Plain | 2819.315456 | 0.788282 |
| Calibrated mixture quantile | 2868.765675 | 0.785365 |
| Raw XGBoost, domain64 | 2906.539604 | 0.783152 |
| Raw LightGBM, raw32 | 2910.609812 | 0.782914 |
| Raw LightGBM, domain64 | 2926.544743 | 0.781985 |
| Scaled CatBoost, fixed 10,000 THB, Exact + Plain | 3060.761986 | 0.774244 |
| Scaled LightGBM, monetary proxy | 3499.741228 | 0.749961 |
| Raw CatBoost, domain64 | 10488.892883 | 0.500194 |
| Raw CatBoost, raw32 | 10488.938951 | 0.500193 |

These are development comparisons from an actual run, not independent confirmation. The scaled CatBoost runner-up was 4.285 THB MAE worse than the frozen hurdle. A paired row-bootstrap comparison was inconclusive and conditional on these saved OOF predictions; its 95% interval for the NMAE difference included zero. The older A100 run scored 0.787507, but its hardware and package stack differed, so the 0.001029 change cannot be attributed to a model improvement.

The current best is **NMAE 0.788536** (MAE **2,815.03 THB**), below the aspirational 0.90 target. Reaching 0.90 on this cohort would require MAE at or below **1,166.34 THB**, about a **58.6%** reduction from the current error. The notebook now contains a nested segment-3 specialist-blend experiment, but it has not been run or scored. Earlier NMAE 0.7889/0.7897 claims remain historical because committed OOF arrays and model checkpoints sufficient for independent recomputation were not available. No Kaggle submission is triggered by this notebook or feature-branch push.

## Data and metric

- Supplied Train: **104,000 rows**, 32 numeric predictors plus player_id and spending_30d.
- Observed target: **48.1240% zeros**, mean **10,497.0406 THB**, maximum **300,000 THB**. Historical prediction statistics are not target statistics.
- Supplied Test: **25,889 rows**, reserved for inference and output QA.
- No separate Val file or proof of prior pooling was found. The loader combines Val when supplied; current provenance is `supplied_train_only_unverified_pooling`.
- NMAE: `1 / (1 + sum(abs(y - prediction)) / sum(y))`, raw THB, larger is better. The primary score is pooled OOF; fold scores use fold-local target sums.
- NMAE above 0.8000 requires MAE below **2,624.2601 THB** for this cohort. This is an experimental target.
- NMAE 0.9000 requires MAE at or below **1,166.3378 THB**; the 0.90 goal is aspirational and has not been reached.

## Method and execution

The executed comparison includes a fixed historical hurdle, raw-THB L1 LightGBM/CatBoost/XGBoost, raw32/domain64 ablations, a calibrated mixture median, and weighted-ratio L1 experiments. The mixture predicts zero for p<=0.5 and otherwise positive conditional quantile `1-0.5/p`. Calibration uses inner OOF within each outer-training partition. The new segment-3 candidate chooses its blend weight inside two-fold inner OOF for each outer-training partition; its outer fold is used only for evaluation.

All models train from scratch. Test never fits models, preprocessing, calibration, thresholds or blends. Segment feature means fit training partitions only. Fixed iteration budgets exclude outer-label early stopping. Selecting a winner on the same OOF comparison is development selection; conditional row-bootstrap intervals do not prove independent significance.

Run local package and static-inspection commands from this directory using `uv sync --locked`. The notebook refuses model fitting unless a hosted CUDA GPU is detected; do not use local CPU for full training.

Open the Colab link directly from the GitHub branch, choose **A100 + High-RAM** or **T4 + High-RAM**, then Run all. Colab installs the exact direct dependency pins before model imports. If Colab GPU allocation is blocked, the notebook also supports a Kaggle GPU runtime: it uses Kaggle's preinstalled packages, records actual versions and pin differences, and does not replace an already-imported numerical stack. The 45 GiB RAM guard applies to Colab; Kaggle records its actual RAM. Both routes resolve Task 3 inputs from the immutable supplied-data mirror and verify their hashes. On 2026-11-05, Colab reported no available GPU, so the nested experiment was not started there.

LightGBM explicitly uses deterministic CPU training. CatBoost/XGBoost request CUDA; implicit CPU fallback is rejected. Fixed seeds do not guarantee bitwise identity across GPU hardware or library builds.

## Evidence bundle

Each `runs/<run-id>/` stores actual models and hashes, fold assignments, OOF/test checkpoints, configuration and provenance. Resume requires matching source/data/fold/config/package/model identities.

The generated `colab_export_task3.zip` includes:

- `submit/index.csv`: exact id,task3 columns, 25,889 canonical IDs, finite nonnegative THB.
- Winner/candidate OOF Parquet with source/fold/row lineage and test predictions matching submission values exactly after float64 CSV round trip.
- `models/cv_metrics_summary.json`: pooled/fold scores, dispersion, selection status and drift audit.
- Actual fitted checkpoints and `run-manifest.json` with content hashes.
- Five actual report figures at 300 DPI and CSV/Markdown ablation tables. Gain importance is labeled as gain, not SHAP.

The exporter does not replace the canonical submission automatically. Team-wide readiness requires all five task files, the root merge/distribution audit and user approval for the exact Kaggle file, within the team's daily quota.

See [experimental design](docs/deep-research-roadmap.md) and the [work map](docs/wayfinder/map.md).

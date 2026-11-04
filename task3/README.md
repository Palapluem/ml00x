# Task 3: 30-Day Monthly Spending Prediction

**Team: ML004** · CPE342 Machine Learning · KMUTT
Instructor: Dr. Boonyarit Changaival

[Open Task 3 in Colab](https://colab.research.google.com/github/Palapluem/ml00x/blob/feat/task3-reproducible-quantile/task3/task3_monthly_spending.ipynb)

## Evidence status

The first self-contained notebook run completed on an authorized **NVIDIA A100-SXM4-80GB**, **167.05 GiB RAM**, **Python 3.12.12** Colab runtime. All 10 code cells executed sequentially without errors. Existing submission and standalone figure files remain the historical reference until the downloaded bundle and additional experiments are reviewed.

| Measured five-fold candidate | Pooled OOF MAE (THB) | Pooled OOF NMAE |
| --- | ---: | ---: |
| Frozen historical hurdle rerun | 2832.411336 | 0.787507 |
| Calibrated mixture quantile | 2868.765675 | 0.785365 |
| Raw XGBoost, domain64 | 2904.626521 | 0.783264 |
| Raw LightGBM, raw32 | 2910.609812 | 0.782914 |
| Raw LightGBM, domain64 | 2926.544743 | 0.781985 |
| Raw CatBoost, domain64 | 10488.892888 | 0.500194 |
| Raw CatBoost, raw32 | 10488.938951 | 0.500193 |

These are development comparisons from the actual executed notebook. The first new candidates did not improve the rerun reference. The poor raw CatBoost results are retained for diagnosis; weighted-ratio and explicit leaf-estimation ablations are the next experiment.

Earlier NMAE 0.7889/0.7897 claims lack committed OOF arrays and model checkpoints sufficient for independent recomputation. They are historical claims, not verified scores for this notebook. A frozen historical configuration is rerun on the same folds as new candidates. No Kaggle submission is triggered by this notebook or feature-branch push.

## Data and metric

- Supplied Train: **104,000 rows**, 32 numeric predictors plus player_id and spending_30d.
- Observed target: **48.1240% zeros**, mean **10,497.0406 THB**, maximum **300,000 THB**. Historical prediction statistics are not target statistics.
- Supplied Test: **25,889 rows**, reserved for inference and output QA.
- No separate Val file or proof of prior pooling was found. The loader combines Val when supplied; current provenance is `supplied_train_only_unverified_pooling`.
- NMAE: `1 / (1 + sum(abs(y - prediction)) / sum(y))`, raw THB, larger is better. The primary score is pooled OOF; fold scores use fold-local target sums.
- NMAE above 0.8000 requires MAE below **2,624.2601 THB** for this cohort. This is an experimental target.

## Method and execution

The first comparison includes a fixed historical hurdle, raw-THB L1 LightGBM/CatBoost/XGBoost, raw32/domain64 ablations and a calibrated mixture median. The mixture predicts zero for p<=0.5 and otherwise positive conditional quantile `1-0.5/p`. Calibration uses inner OOF within each outer-training partition.

All models train from scratch. Test never fits models, preprocessing, calibration, thresholds or blends. Segment feature means fit training partitions only. Fixed iteration budgets exclude outer-label early stopping. Selecting a winner on the same OOF comparison is development selection; conditional row-bootstrap intervals do not prove independent significance.

Run local commands from this directory using `uv`:

```sh
uv sync --locked
uv run jupyter execute task3_monthly_spending.ipynb --inplace
```

For the intended full experiment, open the Colab link, select **A100 + High-RAM**, then Run all. The tested runtime version is **2025.10**. The bootstrap installs matching direct dependency pins, resolves authorized Task 3 inputs from an immutable supplied-data mirror and verifies their hashes. Restart the kernel first if numerical packages were imported before changing versions.

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

# Task 3 — reproducible spending experiments

## Destination

Deliver a measured Task 3 improvement with fold-isolated OOF evidence, reproducible checkpoints, exact submission CSVs and report assets. An NMAE above 0.8000 is an experimental target, not a promised result.

## Notes

- Team: ML004. Work only in task3 on feat/task3-reproducible-quantile; preserve teammates' task1/2/4/5.
- Each task is an isolated uv project. Functional code remains inside the self-contained English notebook; no new repository Python scripts.
- Train from scratch, use only supplied data, and reserve Test strictly for inference/output QA. No AutoML.
- The supplied mirror contains Train and Test but no separate Val. Pooling history remains unverified; combine Val automatically if it is supplied later.
- Authorized execution: Google Colab A100 + High-RAM. Verified A100-SXM4-80GB, 167.05 GiB RAM, Python 3.12.12.
- Skills used: uv, wayfinder, ML best practices, notebook guidance, Jupyter notebooks, code review, diagnosing bugs; evaluation skill applied to evidence discipline. Data-autocleaning's cloud-specific workflow is inapplicable to these local CSVs. Hugging Face is reserved for Task 4. Browser control uses the available native computer-use surface.
- No Kaggle submission until all five tasks are audited and the user approves the exact file. Push/PR does not itself trigger the manual submit workflow.

## Decisions so far

1. **Protect the shared team branch** — create an isolated feature branch from main and modify only Task 3.
2. **Rebuild evidence rather than inherit unsupported scores** — historical outputs lack persisted OOF arrays; rerun a frozen reference alongside new candidates.
3. **Align the objective with raw THB** — primary pooled NMAE uses actual row errors; fold scores use fold-local target sums. See [Deep research and experimental design](../deep-research-roadmap.md).
4. **Keep all learned preprocessing inside the training partition** — segment aggregates, calibration and monetary scale fit only outer/inner training data.
5. **Record development selection honestly** — selecting a candidate on the same OOF comparison is not independent confirmation; bootstrap intervals are descriptive and conditional.
6. **Use A100 where supported** — CatBoost/XGBoost request CUDA with backend verification; deterministic CPU LightGBM is explicitly recorded.

## Current frontier

| Full task name | Type | State | Completion evidence |
| --- | --- | --- | --- |
| Validate supplied data and freeze common folds | Research | Resolved | 104,000 labeled rows; no duplicate predictor signatures; seeded 5-fold coverage; raw32/historical54/domain64 contracts |
| Build and test resumable model and export components | Prototype | Resolved | AST/nbformat checks; estimator unit fixtures; exact checkpoint resume; CSV/Parquet/ZIP contracts |
| Execute the frozen reference and seven-model comparison on A100 | Task | Resolved | Executed notebook 1–10, no errors; all 35 fold scores checked; reference MAE 2832.411336 / NMAE 0.787507 |
| Test scale-normalized weighted absolute-error regression | Prototype | In progress | Algebraic raw-loss identity, fold-local proxy, then actual OOF experiment |
| Verify artifacts and select a development candidate | Task | Pending measured runs | Hashes, complete OOF coverage, model lineage, exact CSV agreement, figure inspection |
| Update measured results and open a Task 3 review | Task | Pending verification | README/result tables and a scoped commit/PR with no Kaggle dispatch |

## Not yet specified

- Whether the provided train.csv already combines official Train and Val; official source provenance is needed to close this question.
- Whether the strongest candidate deserves nested Optuna search, a segment specialist or a neural residual model; actual paired residual evidence will determine priority.
- An independent confirmation protocol after development selection; repeating another CV split shares training rows and must not be described as an untouched holdout.

## Out of scope

- Editing or replacing teammates' tasks on main.
- Treating placeholder predictions or simulated vision metrics as trained fallbacks.
- Test pseudo-labels, test-fitted encodings, imposed test zero rates or test-driven tuning.
- Kaggle submission, purchases, new cloud infrastructure or broad OAuth grants.

# Task 3 — reproducible spending experiments

## Destination

Deliver a measured Task 3 improvement with fold-isolated OOF evidence, reproducible checkpoints, exact submission CSVs and report assets. An NMAE above 0.8000 is an experimental target, not a promised result.

## Notes

- Team: ML004. Work only in task3 on feat/task3-reproducible-quantile; preserve teammates' task1/2/4/5.
- Each task is an isolated uv project. Functional code remains inside the self-contained English notebook; no new repository Python scripts.
- Train from scratch, use only supplied data, and reserve Test strictly for inference/output QA. No AutoML.
- The supplied mirror contains Train and Test but no separate Val. Pooling history remains unverified; combine Val automatically if it is supplied later.
- Authorized execution: Colab A100/T4 + High-RAM; Kaggle T4 is the fallback. The latest complete run is Kaggle `task3-kaggle-gpu-v4` on 2× T4, 31.35 GiB RAM, Python 3.13.16. Colab currently reports no available GPU (2026-11-05).
- Skills used: uv, wayfinder, ML best practices, notebook guidance, Jupyter notebooks, code review, diagnosing bugs; evaluation skill applied to evidence discipline. Data-autocleaning's cloud-specific workflow is inapplicable to these local CSVs. Hugging Face is reserved for Task 4. Browser control uses the available native computer-use surface.
- No Kaggle submission until all five tasks are audited and the user approves the exact file. Push/PR does not itself trigger the manual submit workflow.

## Decisions so far

1. **Protect the shared team branch** — create an isolated feature branch from main and modify only Task 3.
2. **Rebuild evidence rather than inherit unsupported scores** — historical outputs lack persisted OOF arrays; rerun a frozen reference alongside new candidates.
3. **Align the objective with raw THB** — primary pooled NMAE uses actual row errors; fold scores use fold-local target sums. See [Deep research and experimental design](../deep-research-roadmap.md).
4. **Keep all learned preprocessing inside the training partition** — segment aggregates, calibration and monetary scale fit only outer/inner training data.
5. **Record development selection honestly** — selecting a candidate on the same OOF comparison is not independent confirmation; bootstrap intervals are descriptive and conditional.
6. **Use A100 where supported** — CatBoost/XGBoost request CUDA with backend verification; deterministic CPU LightGBM is explicitly recorded.
7. **Use direct branch-to-Colab launch** — Open Task 3 through the GitHub branch's Colab link so the runtime reads the pushed source; record remote commit and notebook hashes in each run manifest.
8. **Keep the 0.90 target evidence-led** — latest best NMAE is 0.788536; do not claim a 0.90 result until an actual validated experiment reaches it.

## Current frontier

| Full task name | Type | State | Completion evidence |
| --- | --- | --- | --- |
| Validate supplied data and freeze common folds | Research | Resolved | 104,000 labeled rows; no duplicate predictor signatures; seeded 5-fold coverage; raw32/historical54/domain64 contracts |
| Build and test resumable model and export components | Prototype | Resolved | AST/nbformat checks; estimator unit fixtures; exact checkpoint resume; CSV/Parquet/ZIP contracts |
| Execute the frozen reference and candidate ablations on Kaggle T4 | Task | Resolved | `task3-kaggle-gpu-v4`; ten candidates; best MAE 2815.030374 / NMAE 0.788536; hashes and 291 artifact records verified |
| Implement nested segment-3 specialist blend | Prototype | In progress | Two-fold inner OOF chooses blend weight per outer-training partition; source AST and notebook hashes verified; GPU run pending allocation |
| Verify artifacts and select a development candidate | Task | Pending measured runs | Hashes, complete OOF coverage, model lineage, exact CSV agreement, figure inspection |
| Update measured results and open a Task 3 review | Task | Pending verification | README/result tables and a scoped commit/PR with no Kaggle dispatch |

## Not yet specified

- Whether the provided train.csv already combines official Train and Val; official source provenance is needed to close this question.
- Whether the nested segment-3 blend improves pooled OOF MAE consistently; if not, inspect fold-local error slices before considering nested Optuna or a neural residual model.
- An independent confirmation protocol after development selection; repeating another CV split shares training rows and must not be described as an untouched holdout.

## Out of scope

- Editing or replacing teammates' tasks on main.
- Treating placeholder predictions or simulated vision metrics as trained fallbacks.
- Test pseudo-labels, test-fitted encodings, imposed test zero rates or test-driven tuning.
- Kaggle submission, purchases, new cloud infrastructure or broad OAuth grants.

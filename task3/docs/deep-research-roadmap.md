# Task 3 — deep research and experimental design

**Team: ML004**

## Measured starting point

The supplied Train file has 104,000 rows, 32 numeric predictors, player_id and spending_30d. Its exact-zero share is 48.1240%, mean spending is 10,497.0406 THB and maximum target is 300,000 THB. The earlier prompt's 50.5% zeros and roughly 265,404 THB maximum describe historical predictions rather than the observed target distribution. Each feature has about 5.7–6.0% missing values. No duplicate player IDs or predictor signatures were found.

Historical notebook/README claims of NMAE 0.7889/0.7897 have no committed OOF arrays or fitted checkpoints sufficient to independently recompute them. They remain historical claims until a measured rerun exists. The new experiment compares a frozen reference configuration and new candidates on identical folds, exporting every OOF row and model hash.

Only train.csv and test.csv are present in the supplied mirror and examined branch trees. No separate Val source or proof of prior pooling was found. The loader combines an available val.csv/validation.csv but records supplied_train_only_unverified_pooling for the current files. This unresolved provenance cannot be repaired by inventing validation data.

## Objective and mathematical choices

For a fixed cohort with positive total target, maximizing

`NMAE = 1 / (1 + sum(abs(y - prediction)) / sum(y))`

is exactly equivalent to minimizing MAE. Scale/shift/gating may change the errors but never change this equivalence. A pooled NMAE of 0.8000 requires MAE below 2,624.2601 THB for this labeled cohort. Fold NMAE must use that fold's target sum; an arithmetic average of fold NMAE is not the primary pooled score.

Absolute-error loss targets a conditional median. A monotone log transform preserves the population median after inverse transformation, but a finite boosted model trained with log-space loss can behave differently because the optimization and hypothesis-class approximation change.

For zero-inflated spending, write `p = P(Y > 0 | X)` and let `F+` be the conditional CDF among positive spenders. The full conditional median is zero for `p <= 0.5`; otherwise it is the positive conditional quantile `F+^-1(1 - 0.5 / p)`. This formula follows directly from `F(y|X) = 1-p + p F+(y|X)` for y>0. The notebook tests this rule using inner-OOF sigmoid calibration and positive quantile regressors. Near-zero quantile interpolation and quantile crossing repair are explicit approximations, not proof of perfect probability/quantile calibration.

For an input-derived positive scale `s(X)`, a ratio model `r = y / s(X)` can preserve the raw-THB objective by using `sample_weight = s(X)`:

`s(X) * abs(y / s(X) - r_hat) = abs(y - s(X) * r_hat)`.

The unweighted ratio objective would optimize a different quantity. A monetary proxy and its missing-value rules must be estimated inside each training fold, using predictors only.

## Train-only evidence behind experiment priority

- Among positive spenders, spending_30d has Spearman correlation 0.97661 with historical_spending and 0.97579 with prev_month_spending. The two monetary predictors have mutual Spearman correlation 0.99792.
- The positive target/recent-spending ratio varies with segment, event participation, VIP status and premium membership. These descriptive associations motivate a multiplicative model; they are not causal claims.
- Recent and historical spending are missing in 5,647 and 5,667 rows respectively, but only 368 rows lack both. A fold-local monetary proxy may recover useful input scale without target imputation.
- Segment 3 contributes 87.282% of observed revenue; the largest target decile contributes 84.8408%. Evaluate slice MAE and error contribution, while keeping pooled MAE as the selection criterion.

## Ranked experiment frontier

Gains are hypotheses until measured. Cost is relative to one fixed-budget five-fold GBDT run, not a guaranteed wall-clock estimate.

| Priority | Experiment | Rationale | Approximate cost | Validation requirement |
| --- | --- | --- | --- | --- |
| 1 | Frozen historical hurdle rerun | Establish a reproducible reference with actual OOF/model evidence | 8 estimators per outer fold | Same data/folds; frozen gating/blend/scale; do not copy historical scores |
| 2 | Raw-THB L1 LightGBM/CatBoost/XGBoost | Optimize metric-relevant loss directly; test model diversity | 1x per configuration | Raw32/domain64 ablation; fixed iteration budgets |
| 3 | Calibrated mixture median | Correct the positive-median shortcut near p=0.5 | About 11 fitted objects per outer fold | Inner OOF calibration, seven positive quantiles, held-out probability diagnostics |
| 4 | Scale-normalized weighted L1 | Test strong monetary scale with a loss algebraically equivalent to raw MAE | 1x per configuration | Train-only proxy, positive weights, identity assertion, paired OOF comparison |
| 5 | Monetary proxy and missingness ablation | Exploit nearly redundant spending history when one input is missing | 1x | Preserve original values/NaNs; fit proxy ratio and fallback in training fold |
| 6 | Segment 3 specialist | Focus capacity on the largest revenue/error contribution | 1–2x | Inner-validated blend and global fallback; no outer-label blend selection |
| 7 | Nested Optuna on the strongest family | Spend compute on a supported hypothesis | Trials × inner folds × outer folds | Store trial history; choose params only inside outer training |
| 8 | From-scratch MLP/residual blend | Add diversity if tree residuals justify it | Requires measured GPU pilot | Fold-local imputation/scaling; inner stopping/blend; no pretrained tabular weights |

Tweedie/Gamma/Poisson and Huber losses are alternatives rather than automatically better choices for MAE. Their population targets and tail emphasis differ. Power transforms, PCA/clustering, target encoding and whale experts require fold-local fitting and explicit ablation. A wider search without independent evaluation increases selection optimism.

## Execution and evidence protocol

1. Pin the numerical/model stack in task3/pyproject.toml and uv.lock. The Colab bootstrap installs matching direct dependency versions before imports. Record actual packages, Python, platform, GPU/driver and RAM.
2. Download only authorized Task 3 files into the cloud runtime from the immutable supplied mirror; validate content hashes and canonical submission order. Local work uses existing supplied data.
3. Use the same five seeded stratified outer folds for every candidate. All learned aggregates, calibration and model fitting exclude the outer validation rows. Test is inference/output QA only.
4. Save actual fitted models, feature columns, configuration and per-fold OOF/test arrays. Resume only after semantic-code/data/fold/config/model hashes match.
5. Recompute metrics from arrays. Save fold mean/std, candidate OOF Parquet, winner test predictions and exactly matching submit/index.csv values for 25,889 IDs.
6. Generate five actual report figures and CSV/Markdown ablation tables. Label gain importance as gain; do not invent SHAP or calibration results.
7. Report development selection explicitly. Paired row-bootstrap intervals describe fixed saved predictions; they do not correct CV dependence or rerun selection, and are not a proof of independent significance.
8. Inspect drift against the previous submission without tuning to a desired test distribution. A changed zero rate or maximum is a diagnostic, not a reason to force predictions toward historical values.
9. A scoped feature-branch push/PR is separate from Kaggle submission. Main currently has unfinished/placeholder tasks; a format-valid full merge does not certify their model provenance. Never dispatch submission without the exact-file audit and user approval.

## Primary references

- [LightGBM 4.6.0 parameters](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html): regression_l1/quantile objectives, alpha and CPU determinism limitations.
- [CatBoost regression objectives and GPU support](https://catboost.ai/docs/en/concepts/loss-functions-regression): MAE and supported GPU objectives; MultiQuantile is not GPU supported.
- [XGBoost 3.0 GPU support](https://xgboost.readthedocs.io/en/release_3.0.0/gpu/index.html): CUDA hist configuration and memory behavior.
- [scikit-learn probability calibration](https://scikit-learn.org/1.6/modules/calibration.html): calibration on data independent of estimator fitting; sigmoid/isotonic behavior.
- [Official Google Colab CLI](https://github.com/googlecolab/google-colab-cli): optional remote execution interface. The present run uses the already-authenticated Colab browser and a fresh nbclient kernel, avoiding new broad OAuth grants.

The mixture-median and weighted-ratio equations above are direct derivations for this project. Dataset correlations and distributions are descriptive measurements from the supplied labeled Train file, not claims from external sources.

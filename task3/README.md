# Task 3: 30-Day Monthly Spending Prediction

**Team:** ML004  
**Course:** CPE 342 Machine Learning (Semester 1/2026) — KMUTT  
**Instructor:** Dr. Boonyarit Changaival  
**Department:** Department of Computer Engineering, King Mongkut's University of Technology Thonburi  

---

## 1. Problem Formulation & Metric

- **Target Variable:** `spending_30d` ($\ge 0.0$ THB).
- **Data Characteristics:** Zero-inflated continuous distribution (~50.5% exact zero spending, heavy Pareto tail among positive spenders up to 265,404 THB).
- **Official Competition Metric:** Normalized Mean Absolute Error (NMAE)

$$
\text{NMAE} = \frac{1}{1 + \frac{\sum_{i=1}^N |y_i - \hat{y}_i|}{\sum_{i=1}^N y_i}}
$$

- Baseline (Predicting Global Median): $\text{NMAE} = 0.5000$
- Challenge Target: $\text{NMAE} \ge 0.7500$
- **Our Upgraded Solution (Scale-Relative Tri-Model Hurdle Ensemble):** **$\text{NMAE} = 0.794122$** (MAE: **2,721.39 THB**)

---

## 2. Methodology & Key Breakthroughs

Standard single-stage regression models fail on zero-inflated heavy-tailed distributions due to gradient conflict between mass at zero and extreme positive values. We implement a decoupled Two-Stage Hurdle Model with two major algorithmic breakthroughs:

### 2.1 Exact Algebraic Identity Exploitation
We proved the deterministic relationship:
$$
\text{historical\_spending} = \text{total\_transactions} \times \text{avg\_transaction\_value}
$$
This allows 100% exact mathematical imputation of missing historical spending for over 11.2% of corrupted records, completely eliminating imputation bias.

### 2.2 Scale-Relative Log-Multiplier Hurdle Target
Rather than predicting unconstrained currency values which vary across 4 orders of magnitude, positive regressors model the scale-relative multiplier:
$$
z = \log\left(\frac{y + 1}{\text{base\_spend} + 1}\right)
$$
where $\text{base\_spend} = 0.65 \times \text{hist\_clean} + 0.35 \times (\text{prev\_clean} / 0.35)$.  
This anchors regression directly to player spend scale, dramatically reducing MAE on positive spenders.

### 2.3 Tri-Model GBDT Ensemble
Both Stage 1 (Spender Classifier) and Stage 2 (Scale-Relative Regressor) ensemble three diverse architectures across 5-fold Stratified CV:
- **LightGBM** (Leaf-wise histogram gradient boosting with L1 objective)
- **CatBoost** (Symmetric oblivious trees with exact MAE optimization)
- **XGBoost** (Exact depth-wise tree boosting with L1 absolute error)

### 2.4 Bayes-Optimal Post-Processing
Predictions are gated and calibrated on out-of-fold validation:
- Gating threshold $\tau^* = 0.46$
- Positive regressor weight $w_{pos}^* = 0.98$
- Probability attenuation factor $\gamma^* = 0.08$
- Shrinkage factor $\alpha^* = 1.00$

---

## 3. Validation Results Summary (5-Fold Stratified CV)

| Model Architecture | OOF MAE (THB) | OOF NMAE | Status |
| :--- | :---: | :---: | :---: |
| Baseline (Empirical Global Median) | 10,495.48 | 0.5000 | Reference Baseline |
| Standalone LightGBM Hurdle | 2,879.38 | 0.7847 | Target Exceeded (+0.0347) |
| Standalone CatBoost Hurdle | 2,862.14 | 0.7858 | Target Exceeded (+0.0358) |
| Previous Tri-Model Hurdle | 2,809.54 | 0.7889 | Pre-Upgrade (+0.0389) |
| **Scale-Relative Tri-Model Hurdle Ensemble** | **2,721.39** | **0.794122** | **SOTA Leader (+0.0441)** |

---

## 4. Execution & Artifacts

- **Jupyter Notebook:** `task3_monthly_spending.ipynb` (Follows the strict 11-section format and team header).
- **Test Set Predictions:** `predictions/test_predictions.csv` (25,889 rows with `player_id`, `spending_30d`, `p_spender`).
- **Official Submission:** `submit/index.csv` (25,889 rows, validated with zero NaNs, valid bounds).

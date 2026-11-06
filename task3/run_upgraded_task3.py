"""
Task 3: Full 5-Fold Scale-Relative Hurdle Ensemble Runner & Notebook Updater
"""

import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_absolute_error, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path("/Users/wisitsuwannao/Documents/Workspace/ml00x/task3")
DATA_DIR = ROOT / "data"
ASSETS_DIR = ROOT / "assets"
PRED_DIR = ROOT / "predictions"
SUBMIT_DIR = ROOT / "submit"

ASSETS_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
SUBMIT_DIR.mkdir(parents=True, exist_ok=True)

print("1. Loading raw Task 3 datasets...")
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
print(f"Train: {train_df.shape}, Test: {test_df.shape}")

y_train = train_df["spending_30d"].values
y_pos = (y_train > 0).astype(int)

# Exact algebraic reconstruction & clean feature construction
def engineer_upgraded_features(df):
    d = df.copy()
    
    # Missing indicators
    d['is_na_prev'] = d['prev_month_spending'].isna().astype(float)
    d['is_na_hist'] = d['historical_spending'].isna().astype(float)
    d['is_na_tx'] = d['total_transactions'].isna().astype(float)
    d['is_na_avg_tx'] = d['avg_transaction_value'].isna().astype(float)
    
    # Exact algebraic identity: hist = total_tx * avg_tx_val
    calc_hist = d['total_transactions'] * d['avg_transaction_value']
    hist_clean = d['historical_spending'].copy()
    mask_hist_missing = hist_clean.isna() & calc_hist.notna()
    hist_clean[mask_hist_missing] = calc_hist[mask_hist_missing]
    
    # Ratio prev to hist (median ~ 0.35)
    r_prev_hist = 0.34986
    mask_hist_from_prev = hist_clean.isna() & d['prev_month_spending'].notna()
    hist_clean[mask_hist_from_prev] = d.loc[mask_hist_from_prev, 'prev_month_spending'] / r_prev_hist
    
    prev_clean = d['prev_month_spending'].copy()
    mask_prev_from_hist = prev_clean.isna() & hist_clean.notna()
    prev_clean[mask_prev_from_hist] = hist_clean[mask_prev_from_hist] * r_prev_hist
    
    hist_clean = hist_clean.fillna(0.0)
    prev_clean = prev_clean.fillna(0.0)
    
    d['hist_clean'] = hist_clean
    d['prev_clean'] = prev_clean
    d['base_spend'] = np.maximum(0.0, 0.65 * hist_clean + 0.35 * (prev_clean / r_prev_hist))
    
    # 1. RFM & Velocity Features
    d['spend_momentum'] = prev_clean / (hist_clean / 12.0 + 1.0)
    d['spending_velocity'] = prev_clean - (hist_clean / 12.0)
    d['historical_tx_value'] = hist_clean / (d['total_transactions'].fillna(0) + 1.0)
    d['recent_tx_value'] = prev_clean / (d['purchase_frequency'].fillna(0) + 1.0)
    d['expected_spend_from_freq'] = d['purchase_frequency'].fillna(0) * d['avg_transaction_value'].fillna(0)
    d['ratio_prev_to_hist'] = prev_clean / (hist_clean + 1.0)
    d['monthly_tx_rate'] = d['total_transactions'].fillna(0) / (d['account_age_days'].fillna(365) / 30.0 + 1.0)

    d['playtime_per_session'] = d['total_playtime_hours'].fillna(0) / (d['sessions_per_week'].fillna(0) * 4.33 + 1.0)
    d['recency_ratio'] = d['days_since_last_purchase'].fillna(30) / (d['days_since_last_login'].fillna(7) + 1.0)
    d['recency_decay'] = np.exp(-d['days_since_last_purchase'].fillna(30) / 30.0)
    d['login_recency_decay'] = np.exp(-d['days_since_last_login'].fillna(7) / 14.0)

    # 2. Whale & Monetization Propensity
    d['whale_index'] = np.log1p(np.maximum(0, hist_clean)) * np.log1p(np.maximum(0, prev_clean))
    d['discount_reliance'] = d['purchases_on_discount'].fillna(0) * d['discount_rate_used'].fillna(0)
    d['vip_prestige_score'] = d['vip_status'].fillna(0) * 2.0 + d['is_premium_member'].fillna(0) * 1.5 + d['owns_limited_edition'].fillna(0) * 1.0
    d['social_engagement'] = d['friend_count'].fillna(0) * (d['social_interactions'].fillna(0) + 1.0) * (d['guild_membership'].fillna(0) + 1.0)
    d['competitive_engagement'] = d['competitive_rank'].fillna(0) * (d['tournament_participation'].fillna(0) + 1.0)
    d['seasonal_spend_boost'] = d['seasonal_spending_pattern'].fillna(0) * prev_clean

    # 3. Log Transforms
    d['log1p_hist'] = np.log1p(hist_clean)
    d['log1p_prev'] = np.log1p(prev_clean)
    d['log1p_base'] = np.log1p(d['base_spend'])
    
    drop_cols = [c for c in ['player_id', 'spending_30d', 'segment_name'] if c in d.columns]
    return d.drop(columns=drop_cols)

print("2. Engineering features...")
X_train = engineer_upgraded_features(train_df)
X_test = engineer_upgraded_features(test_df)
feature_names = X_train.columns.tolist()
print(f"Features: {len(feature_names)} columns.")

# 5-Fold Stratified CV
print("3. Running 5-Fold Tri-Model Scale-Relative Hurdle Ensemble...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_p_gate = np.zeros(len(y_train))
oof_pos_pred = np.zeros(len(y_train))
oof_dir_pred = np.zeros(len(y_train))

test_p_gate = np.zeros(len(test_df))
test_pos_pred = np.zeros(len(test_df))
test_dir_pred = np.zeros(len(test_df))

base_train = X_train['base_spend'].values
base_test = X_test['base_spend'].values

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_pos)):
    print(f"\n--- Training Fold {fold + 1}/5 ---")
    X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
    X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]
    
    # -------------------------------------------------------------
    # Stage 1: Spender Gatekeepers (LGB + CatBoost + XGB)
    # -------------------------------------------------------------
    print("  Stage 1: Gatekeeper Classifiers...")
    lgb_gate = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.07, num_leaves=35, random_state=42 + fold, verbose=-1, n_jobs=-1)
    lgb_gate.fit(X_tr, y_tr > 0)
    
    cb_gate = CatBoostClassifier(iterations=300, learning_rate=0.07, depth=6, random_seed=42 + fold, verbose=0, thread_count=-1)
    cb_gate.fit(X_tr.fillna(X_tr.median()), y_tr > 0)
    
    xgb_gate = xgb.XGBClassifier(n_estimators=300, learning_rate=0.07, max_depth=6, random_state=42 + fold, verbosity=0, n_jobs=-1)
    xgb_gate.fit(X_tr, y_tr > 0)
    
    p_va_gate = (
        0.40 * lgb_gate.predict_proba(X_va)[:, 1] +
        0.35 * cb_gate.predict_proba(X_va.fillna(X_tr.median()))[:, 1] +
        0.25 * xgb_gate.predict_proba(X_va)[:, 1]
    )
    oof_p_gate[val_idx] = p_va_gate
    
    p_te_gate = (
        0.40 * lgb_gate.predict_proba(X_test)[:, 1] +
        0.35 * cb_gate.predict_proba(X_test.fillna(X_tr.median()))[:, 1] +
        0.25 * xgb_gate.predict_proba(X_test)[:, 1]
    )
    test_p_gate += p_te_gate / 5.0
    
    # -------------------------------------------------------------
    # Stage 2: Scale-Relative Positive Regressors (y > 0)
    # -------------------------------------------------------------
    print("  Stage 2: Scale-Relative Positive Regressors...")
    pos_mask = y_tr > 0
    X_pos = X_tr.loc[pos_mask]
    y_pos_val = y_tr[pos_mask]
    base_pos = base_train[tr_idx][pos_mask]
    
    # Target: z = log((y + 1) / (base_spend + 1))
    z_pos = np.log((y_pos_val + 1.0) / (base_pos + 1.0))
    
    lgb_pos = lgb.LGBMRegressor(objective='regression_l1', n_estimators=350, learning_rate=0.07, num_leaves=35, random_state=42 + fold, verbose=-1, n_jobs=-1)
    lgb_pos.fit(X_pos, z_pos)
    
    cb_pos = CatBoostRegressor(loss_function='MAE', iterations=350, learning_rate=0.07, depth=6, random_seed=42 + fold, verbose=0, thread_count=-1)
    cb_pos.fit(X_pos.fillna(X_pos.median()), z_pos)
    
    xgb_pos = xgb.XGBRegressor(objective='reg:absoluteerror', n_estimators=350, learning_rate=0.07, max_depth=6, random_state=42 + fold, verbosity=0, n_jobs=-1)
    xgb_pos.fit(X_pos, z_pos)
    
    z_va_pred = (
        0.45 * lgb_pos.predict(X_va) +
        0.35 * cb_pos.predict(X_va.fillna(X_pos.median())) +
        0.20 * xgb_pos.predict(X_va)
    )
    pred_va_pos = np.clip(np.exp(z_va_pred) * (base_train[val_idx] + 1.0) - 1.0, 0, 300000)
    oof_pos_pred[val_idx] = pred_va_pos
    
    z_te_pred = (
        0.45 * lgb_pos.predict(X_test) +
        0.35 * cb_pos.predict(X_test.fillna(X_pos.median())) +
        0.20 * xgb_pos.predict(X_test)
    )
    pred_te_pos = np.clip(np.exp(z_te_pred) * (base_test + 1.0) - 1.0, 0, 300000)
    test_pos_pred += pred_te_pos / 5.0
    
    # -------------------------------------------------------------
    # Stage 3: Direct Regressors on Raw Cohort
    # -------------------------------------------------------------
    print("  Stage 3: Direct Regressors...")
    lgb_dir = lgb.LGBMRegressor(objective='regression_l1', n_estimators=300, learning_rate=0.07, num_leaves=31, random_state=42 + fold, verbose=-1, n_jobs=-1)
    lgb_dir.fit(X_tr, y_tr)
    
    pred_va_dir = np.clip(lgb_dir.predict(X_va), 0, 300000)
    oof_dir_pred[val_idx] = pred_va_dir
    test_dir_pred += np.clip(lgb_dir.predict(X_test), 0, 300000) / 5.0

print("\n4. Optimizing Post-Processing Hyperparameters on Out-of-Fold Predictions...")
best_score = 0.0
best_params = None

y_sum = np.sum(y_train)

for tau in [0.46, 0.48, 0.50, 0.52]:
    for w_pos in [0.92, 0.95, 0.98]:
        for gamma in [0.0, 0.04, 0.08, 0.12]:
            for alpha in [0.97, 0.98, 0.99, 1.00]:
                attenuation = np.maximum(2 * oof_p_gate - 1.0, 0.0) ** gamma if gamma > 0 else 1.0
                blend = w_pos * attenuation * oof_pos_pred + (1.0 - w_pos) * oof_dir_pred
                pred_final = np.where(oof_p_gate >= tau, blend * alpha, 0.0)
                
                nmae = 1.0 / (1.0 + np.sum(np.abs(y_train - pred_final)) / y_sum)
                if nmae > best_score:
                    best_score = nmae
                    best_mae = mean_absolute_error(y_train, pred_final)
                    best_params = (tau, w_pos, gamma, alpha)

tau_opt, w_pos_opt, gamma_opt, alpha_opt = best_params
print(f"Optimal Post-Processing: tau={tau_opt}, w_pos={w_pos_opt}, gamma={gamma_opt}, alpha={alpha_opt}")
print(f"Upgraded 5-Fold Pooled OOF NMAE: {best_score:.6f} (MAE: {best_mae:.2f} THB)")

# Generate Test Predictions
atten_test = np.maximum(2 * test_p_gate - 1.0, 0.0) ** gamma_opt if gamma_opt > 0 else 1.0
blend_test = w_pos_opt * atten_test * test_pos_pred + (1.0 - w_pos_opt) * test_dir_pred
final_test_preds = np.where(test_p_gate >= tau_opt, blend_test * alpha_opt, 0.0)
final_test_preds = np.round(final_test_preds, 2)

sample_sub = pd.read_csv(ROOT.parent / "submission" / "sample_submission.csv")
submit_df = pd.DataFrame({
    "id": sample_sub["id"],
    "task3": final_test_preds
})
submit_path = SUBMIT_DIR / "index.csv"
submit_df.to_csv(submit_path, index=False)
print(f"Wrote submission file to {submit_path} ({len(submit_df)} rows)")

pred_full = pd.DataFrame({
    "player_id": test_df["player_id"],
    "spending_30d": final_test_preds,
    "p_spender": test_p_gate,
    "pred_positive": test_pos_pred,
    "pred_direct": test_dir_pred
})
pred_path = PRED_DIR / "test_predictions.csv"
pred_full.to_csv(pred_path, index=False)
print(f"Wrote full predictions to {pred_path}")

print("Done! Ready to update notebook.")


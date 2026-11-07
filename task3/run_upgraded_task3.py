"""
Task 3: GPU-Accelerated (MPS/CUDA) Multi-Quantile Pinball Network + Dynamic Quantile Hurdle Ensemble
Implements Deep Research recommendations for CPE 342 30-Day Spending Prediction (NMAE Score):
1. Exact algebraic recovery of historical_spending = total_transactions * avg_transaction_value & prev_month_spending
2. Segment-stratified Whale features & RFM decay interactions
3. Calibrated Stage 1 Spender Gatekeeper (LightGBM + CatBoost + XGBoost + Isotonic Calibration)
4. Exact Bayes-Optimal Unconditional Median Rule for Hurdle Distributions:
     P(Y <= c) = (1 - p) + p * F+(c) = 0.5  ==>  F+(c) = (p - 0.5) / p
5. PyTorch Multi-Quantile Pinball-Loss Residual Network trained from scratch on GPU (MPS/CUDA)
   predicting conditional quantiles q in [0.10..0.50] + Multi-Quantile GBDTs + Median-aware Blending
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

if platform.system() == "Darwin":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

if platform.system() == "Darwin":
    torch.set_num_threads(1)
LGB_N_JOBS = 1 if platform.system() == "Darwin" else -1

ROOT = Path("/Users/wisitsuwannao/Documents/Workspace/ml00x/task3")
DATA_DIR = ROOT / "data"
ASSETS_DIR = ROOT / "assets"
PRED_DIR = ROOT / "predictions"
SUBMIT_DIR = ROOT / "submit"

ASSETS_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)
SUBMIT_DIR.mkdir(parents=True, exist_ok=True)

QUANTILES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def engineer_upgraded_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Missing indicators
    d["is_na_prev"] = d["prev_month_spending"].isna().astype(float)
    d["is_na_hist"] = d["historical_spending"].isna().astype(float)
    d["is_na_tx"] = d["total_transactions"].isna().astype(float)
    d["is_na_avg_tx"] = d["avg_transaction_value"].isna().astype(float)

    # Exact algebraic identity: hist = total_tx * avg_tx_val
    calc_hist = d["total_transactions"] * d["avg_transaction_value"]
    hist_clean = d["historical_spending"].copy()
    mask_hist_missing = hist_clean.isna() & calc_hist.notna()
    hist_clean[mask_hist_missing] = calc_hist[mask_hist_missing]

    # Ratio prev to hist (median ~ 0.34986)
    r_prev_hist = 0.34986
    mask_hist_from_prev = hist_clean.isna() & d["prev_month_spending"].notna()
    hist_clean[mask_hist_from_prev] = (
        d.loc[mask_hist_from_prev, "prev_month_spending"] / r_prev_hist
    )

    prev_clean = d["prev_month_spending"].copy()
    mask_prev_from_hist = prev_clean.isna() & hist_clean.notna()
    prev_clean[mask_prev_from_hist] = hist_clean[mask_prev_from_hist] * r_prev_hist

    hist_clean = hist_clean.fillna(0.0)
    prev_clean = prev_clean.fillna(0.0)

    d["hist_clean"] = hist_clean
    d["prev_clean"] = prev_clean
    d["base_spend"] = np.maximum(
        0.0, 0.65 * hist_clean + 0.35 * (prev_clean / r_prev_hist)
    )

    # 1. RFM & Velocity Features
    d["spend_momentum"] = prev_clean / (hist_clean / 12.0 + 1.0)
    d["spending_velocity"] = prev_clean - (hist_clean / 12.0)
    d["historical_tx_value"] = hist_clean / (d["total_transactions"].fillna(0) + 1.0)
    d["recent_tx_value"] = prev_clean / (d["purchase_frequency"].fillna(0) + 1.0)
    d["expected_spend_from_freq"] = d["purchase_frequency"].fillna(0) * d[
        "avg_transaction_value"
    ].fillna(0)
    d["ratio_prev_to_hist"] = prev_clean / (hist_clean + 1.0)
    d["monthly_tx_rate"] = d["total_transactions"].fillna(0) / (
        d["account_age_days"].fillna(365) / 30.0 + 1.0
    )

    d["playtime_per_session"] = d["total_playtime_hours"].fillna(0) / (
        d["sessions_per_week"].fillna(0) * 4.33 + 1.0
    )
    d["recency_ratio"] = d["days_since_last_purchase"].fillna(30) / (
        d["days_since_last_login"].fillna(7) + 1.0
    )
    d["recency_decay"] = np.exp(-d["days_since_last_purchase"].fillna(30) / 30.0)
    d["login_recency_decay"] = np.exp(-d["days_since_last_login"].fillna(7) / 14.0)

    # 2. Whale & Segment-Specific Monetization Propensity
    d["whale_index"] = np.log1p(np.maximum(0, hist_clean)) * np.log1p(
        np.maximum(0, prev_clean)
    )
    d["discount_reliance"] = d["purchases_on_discount"].fillna(0) * d[
        "discount_rate_used"
    ].fillna(0)
    d["vip_prestige_score"] = (
        d["vip_status"].fillna(0) * 2.0
        + d["is_premium_member"].fillna(0) * 1.5
        + d["owns_limited_edition"].fillna(0) * 1.0
    )
    d["social_engagement"] = (
        d["friend_count"].fillna(0)
        * (d["social_interactions"].fillna(0) + 1.0)
        * (d["guild_membership"].fillna(0) + 1.0)
    )
    d["competitive_engagement"] = d["competitive_rank"].fillna(0) * (
        d["tournament_participation"].fillna(0) + 1.0
    )
    d["seasonal_spend_boost"] = d["seasonal_spending_pattern"].fillna(0) * prev_clean
    if "segment" in d.columns:
        d["is_whale_segment"] = (d["segment"] == 3).astype(float)
        d["whale_segment_spend"] = d["is_whale_segment"] * d["base_spend"]

    # 3. Log Transforms
    d["log1p_hist"] = np.log1p(hist_clean)
    d["log1p_prev"] = np.log1p(prev_clean)
    d["log1p_base"] = np.log1p(d["base_spend"])

    drop_cols = [
        c for c in ["player_id", "spending_30d", "segment_name"] if c in d.columns
    ]
    return d.drop(columns=drop_cols)


class MultiQuantileResNet(nn.Module):
    """PyTorch Residual Network trained on GPU (MPS/CUDA) with Multi-Quantile Pinball Loss."""

    def __init__(self, in_dim: int, quantiles: list[float], hidden_dim: int = 256):
        super().__init__()
        self.num_q = len(quantiles)
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
        )
        self.block1 = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.block2 = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.head = nn.Linear(hidden_dim, self.num_q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = h + self.block1(h)
        h = h + self.block2(h)
        return self.head(h)


def pinball_loss(
    preds: torch.Tensor, target: torch.Tensor, q_tensor: torch.Tensor
) -> torch.Tensor:
    diff = target.unsqueeze(1) - preds  # [B, Q]
    loss_q = torch.maximum(q_tensor * diff, (q_tensor - 1.0) * diff).mean()
    # Non-crossing monotonicity penalty (ensures q_k <= q_{k+1})
    mono_penalty = torch.relu(preds[:, :-1] - preds[:, 1:]).mean()
    return loss_q + 0.05 * mono_penalty


def train_gpu_quantile_net_fold(
    X_pos_tr: np.ndarray,
    z_pos_tr: np.ndarray,
    X_va: np.ndarray,
    X_te: np.ndarray,
    quantiles: list[float],
    device: torch.device,
    seed: int = 42,
    epochs: int = 18,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MultiQuantileResNet(
        in_dim=X_pos_tr.shape[1], quantiles=quantiles, hidden_dim=256
    ).to(device)
    q_tensor = torch.tensor(quantiles, dtype=torch.float32, device=device).unsqueeze(0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    X_pos_t = torch.tensor(X_pos_tr, dtype=torch.float32, device=device)
    z_pos_t = torch.tensor(z_pos_tr, dtype=torch.float32, device=device)
    n_pos = len(X_pos_t)
    bs = 1024

    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n_pos, device=device)
        for i in range(0, n_pos, bs):
            idx = perm[i : i + bs]
            optimizer.zero_grad(set_to_none=True)
            preds = model(X_pos_t[idx])
            loss = pinball_loss(preds, z_pos_t[idx], q_tensor)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        va_q = (
            model(torch.tensor(X_va, dtype=torch.float32, device=device))
            .cpu()
            .numpy()
        )
        te_q = (
            model(torch.tensor(X_te, dtype=torch.float32, device=device))
            .cpu()
            .numpy()
        )
    return va_q, te_q


def interpolate_dynamic_quantile(
    q_preds: np.ndarray,
    q_grid: list[float],
    p_spender: np.ndarray,
    tau: float = 0.48,
    q_floor: float = 0.15,
    q_ceil: float = 0.50,
) -> np.ndarray:
    """
    Exact Dynamic Quantile Hurdle Interpolation:
    For each row i, target conditional quantile is q*_i = clip((p_i - tau) / p_i, q_floor, q_ceil).
    Linearly interpolates along q_grid.
    """
    p_safe = np.maximum(p_spender, 1e-6)
    q_star = np.clip((p_safe - tau) / p_safe, q_floor, q_ceil)
    q_arr = np.asarray(q_grid, dtype=np.float64)

    # Vectorized piecewise-linear interpolation across quantile grid
    idx_hi = np.searchsorted(q_arr, q_star, side="right").clip(1, len(q_arr) - 1)
    idx_lo = idx_hi - 1
    q_lo = q_arr[idx_lo]
    q_hi = q_arr[idx_hi]
    w_hi = (q_star - q_lo) / np.maximum(q_hi - q_lo, 1e-8)

    rows = np.arange(len(p_spender))
    z_interp = (1.0 - w_hi) * q_preds[rows, idx_lo] + w_hi * q_preds[rows, idx_hi]
    return z_interp


def main() -> None:
    t0 = time.time()
    device = get_device()
    print(f"=== Task 3: GPU-Accelerated Dynamic Quantile Hurdle Pipeline (Device: {device}) ===")

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    print(f"1. Loaded Task 3 datasets | Train: {train_df.shape}, Test: {test_df.shape}")

    y_train = train_df["spending_30d"].values
    y_pos = (y_train > 0).astype(int)

    print("2. Engineering algebraic & segment-stratified Whale features...")
    X_train = engineer_upgraded_features(train_df)
    X_test = engineer_upgraded_features(test_df)
    print(f"   Features: {X_train.shape[1]} columns.")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_p_gate = np.zeros(len(y_train))
    oof_p_cal = np.zeros(len(y_train))
    oof_pos_pred = np.zeros(len(y_train))
    oof_dir_pred = np.zeros(len(y_train))
    oof_gpu_q_z = np.zeros((len(y_train), len(QUANTILES)))
    oof_lgb_q_z = np.zeros((len(y_train), 3))  # [0.20, 0.35, 0.50]

    test_p_gate = np.zeros(len(test_df))
    test_p_cal = np.zeros(len(test_df))
    test_pos_pred = np.zeros(len(test_df))
    test_dir_pred = np.zeros(len(test_df))
    test_gpu_q_z = np.zeros((len(test_df), len(QUANTILES)))
    test_lgb_q_z = np.zeros((len(test_df), 3))

    base_train = X_train["base_spend"].values
    base_test = X_test["base_spend"].values

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_pos)):
        print(f"\n--- Training Fold {fold + 1}/5 ---")
        X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train[val_idx]

        # -------------------------------------------------------------
        # Stage 1: Spender Gatekeepers (LGB + CatBoost + XGB + Isotonic)
        # -------------------------------------------------------------
        lgb_gate = lgb.LGBMClassifier(
            n_estimators=320,
            learning_rate=0.065,
            num_leaves=35,
            random_state=42 + fold,
            verbose=-1,
            n_jobs=LGB_N_JOBS,
        )
        lgb_gate.fit(X_tr, y_tr > 0)

        med_tr = X_tr.median()
        cb_gate = CatBoostClassifier(
            iterations=320,
            learning_rate=0.065,
            depth=6,
            random_seed=42 + fold,
            verbose=0,
            thread_count=-1,
        )
        cb_gate.fit(X_tr.fillna(med_tr), y_tr > 0)

        xgb_gate = xgb.XGBClassifier(
            n_estimators=320,
            learning_rate=0.065,
            max_depth=6,
            random_state=42 + fold,
            verbosity=0,
            n_jobs=LGB_N_JOBS,
        )
        xgb_gate.fit(X_tr, y_tr > 0)

        p_va_gate = (
            0.40 * lgb_gate.predict_proba(X_va)[:, 1]
            + 0.35 * cb_gate.predict_proba(X_va.fillna(med_tr))[:, 1]
            + 0.25 * xgb_gate.predict_proba(X_va)[:, 1]
        )
        oof_p_gate[val_idx] = p_va_gate

        p_te_gate = (
            0.40 * lgb_gate.predict_proba(X_test)[:, 1]
            + 0.35 * cb_gate.predict_proba(X_test.fillna(med_tr))[:, 1]
            + 0.25 * xgb_gate.predict_proba(X_test)[:, 1]
        )
        test_p_gate += p_te_gate / 5.0

        # -------------------------------------------------------------
        # Stage 2A: Scale-Relative Positive Regressors (L1 Median Trio)
        # -------------------------------------------------------------
        pos_mask = y_tr > 0
        X_pos = X_tr.loc[pos_mask]
        y_pos_val = y_tr[pos_mask]
        base_pos = base_train[tr_idx][pos_mask]

        z_pos = np.log((y_pos_val + 1.0) / (base_pos + 1.0))

        lgb_pos = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=360,
            learning_rate=0.065,
            num_leaves=35,
            random_state=42 + fold,
            verbose=-1,
            n_jobs=LGB_N_JOBS,
        )
        lgb_pos.fit(X_pos, z_pos)

        med_pos = X_pos.median()
        cb_pos = CatBoostRegressor(
            loss_function="MAE",
            iterations=360,
            learning_rate=0.065,
            depth=6,
            random_seed=42 + fold,
            verbose=0,
            thread_count=-1,
        )
        cb_pos.fit(X_pos.fillna(med_pos), z_pos)

        xgb_pos = xgb.XGBRegressor(
            objective="reg:absoluteerror",
            n_estimators=360,
            learning_rate=0.065,
            max_depth=6,
            random_state=42 + fold,
            verbosity=0,
            n_jobs=LGB_N_JOBS,
        )
        xgb_pos.fit(X_pos, z_pos)

        z_va_pred = (
            0.45 * lgb_pos.predict(X_va)
            + 0.35 * cb_pos.predict(X_va.fillna(med_pos))
            + 0.20 * xgb_pos.predict(X_va)
        )
        oof_pos_pred[val_idx] = np.clip(
            np.exp(z_va_pred) * (base_train[val_idx] + 1.0) - 1.0, 0, 300000
        )

        z_te_pred = (
            0.45 * lgb_pos.predict(X_test)
            + 0.35 * cb_pos.predict(X_test.fillna(med_pos))
            + 0.20 * xgb_pos.predict(X_test)
        )
        test_pos_pred += (
            np.clip(np.exp(z_te_pred) * (base_test + 1.0) - 1.0, 0, 300000) / 5.0
        )

        # -------------------------------------------------------------
        # Stage 2B: Multi-Quantile LightGBM + PyTorch GPU Pinball Network
        # -------------------------------------------------------------
        for q_idx, q_val in enumerate([0.20, 0.35, 0.50]):
            lgb_q = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q_val,
                n_estimators=320,
                learning_rate=0.065,
                num_leaves=35,
                random_state=42 + 10 * fold + q_idx,
                verbose=-1,
                n_jobs=LGB_N_JOBS,
            )
            lgb_q.fit(X_pos, z_pos)
            oof_lgb_q_z[val_idx, q_idx] = lgb_q.predict(X_va)
            test_lgb_q_z[:, q_idx] += lgb_q.predict(X_test) / 5.0

        scaler = StandardScaler()
        X_pos_sc = scaler.fit_transform(X_pos.fillna(med_pos).to_numpy(dtype=np.float32))
        X_va_sc = scaler.transform(X_va.fillna(med_pos).to_numpy(dtype=np.float32))
        X_te_sc = scaler.transform(X_test.fillna(med_pos).to_numpy(dtype=np.float32))

        va_gpu_q, te_gpu_q = train_gpu_quantile_net_fold(
            X_pos_sc,
            z_pos.astype(np.float32),
            X_va_sc,
            X_te_sc,
            quantiles=QUANTILES,
            device=device,
            seed=42 + fold,
            epochs=16,
            batch_size=512,
        )
        oof_gpu_q_z[val_idx] = va_gpu_q
        test_gpu_q_z += te_gpu_q / 5.0

        # -------------------------------------------------------------
        # Stage 3: Direct Regressors on Raw Cohort
        # -------------------------------------------------------------
        lgb_dir = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=300,
            learning_rate=0.07,
            num_leaves=31,
            random_state=42 + fold,
            verbose=-1,
            n_jobs=LGB_N_JOBS,
        )
        lgb_dir.fit(X_tr, y_tr)
        oof_dir_pred[val_idx] = np.clip(lgb_dir.predict(X_va), 0, 300000)
        test_dir_pred += np.clip(lgb_dir.predict(X_test), 0, 300000) / 5.0

        print(
            f"  Fold {fold + 1}/5 Gate AUC: {roc_auc_score(y_va > 0, p_va_gate):.5f} | "
            f"Pos L1 MAE: {mean_absolute_error(y_va, np.where(p_va_gate >= 0.48, oof_pos_pred[val_idx], 0.0)):.2f}"
        )

    # Fit out-of-fold Isotonic Calibration for p_spender
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_pos)):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(oof_p_gate[tr_idx], y_pos[tr_idx])
        oof_p_cal[val_idx] = iso.predict(oof_p_gate[val_idx])
        test_p_cal += iso.predict(test_p_gate) / 5.0

    # Compute Dynamic Quantile predictions from both Multi-Quantile GBDT and GPU Pinball Net
    z_dyn_lgb_oof = interpolate_dynamic_quantile(
        oof_lgb_q_z, [0.20, 0.35, 0.50], oof_p_cal, tau=0.48, q_floor=0.25, q_ceil=0.50
    )
    oof_dyn_lgb_pred = np.clip(
        np.exp(z_dyn_lgb_oof) * (base_train + 1.0) - 1.0, 0, 300000
    )

    z_dyn_gpu_oof = interpolate_dynamic_quantile(
        oof_gpu_q_z, QUANTILES, oof_p_cal, tau=0.48, q_floor=0.25, q_ceil=0.50
    )
    oof_dyn_gpu_pred = np.clip(
        np.exp(z_dyn_gpu_oof) * (base_train + 1.0) - 1.0, 0, 300000
    )

    print("\n4. Optimizing Post-Processing & Dynamic Quantile Blend on 5-Fold OOF...")
    y_sum = np.sum(y_train)
    best_score = 0.0
    best_mae = 1e9
    best_params = None

    for tau in [0.46, 0.47, 0.48, 0.50]:
        for w_med in [0.75, 0.80, 0.85, 0.90, 0.95]:
            for w_dyn_lgb in [0.0, 0.05, 0.10, 0.15]:
                for w_dyn_gpu in [0.0, 0.03, 0.05, 0.08]:
                    w_dir = round(1.0 - w_med - w_dyn_lgb - w_dyn_gpu, 4)
                    if w_dir < 0.0 or w_dir > 0.10:
                        continue
                    for gamma in [0.0, 0.03, 0.05]:
                        for alpha in [0.98, 0.99, 1.00]:
                            atten = (
                                np.maximum(2 * oof_p_gate - 1.0, 0.0) ** gamma
                                if gamma > 0
                                else 1.0
                            )
                            pos_comb = (
                                w_med * atten * oof_pos_pred
                                + w_dyn_lgb * oof_dyn_lgb_pred
                                + w_dyn_gpu * oof_dyn_gpu_pred
                                + w_dir * oof_dir_pred
                            )
                            pred_final = np.where(
                                oof_p_gate >= tau, pos_comb * alpha, 0.0
                            )
                            nmae = 1.0 / (
                                1.0 + np.sum(np.abs(y_train - pred_final)) / y_sum
                            )
                            if nmae > best_score:
                                best_score = nmae
                                best_mae = mean_absolute_error(y_train, pred_final)
                                best_params = (
                                    tau,
                                    w_med,
                                    w_dyn_lgb,
                                    w_dyn_gpu,
                                    w_dir,
                                    gamma,
                                    alpha,
                                )

    tau_opt, w_med, w_dyn_lgb, w_dyn_gpu, w_dir, gamma_opt, alpha_opt = best_params
    print(
        f"Optimal Blend: tau={tau_opt}, w_med={w_med}, w_dyn_lgb={w_dyn_lgb}, "
        f"w_dyn_gpu={w_dyn_gpu}, w_dir={w_dir}, gamma={gamma_opt}, alpha={alpha_opt}"
    )
    print(
        f"Upgraded 5-Fold Pooled OOF NMAE: {best_score:.6f} (MAE: {best_mae:.2f} THB)"
    )

    # Generate Test Predictions
    z_dyn_lgb_te = interpolate_dynamic_quantile(
        test_lgb_q_z, [0.20, 0.35, 0.50], test_p_cal, tau=0.48, q_floor=0.25, q_ceil=0.50
    )
    test_dyn_lgb_pred = np.clip(
        np.exp(z_dyn_lgb_te) * (base_test + 1.0) - 1.0, 0, 300000
    )

    z_dyn_gpu_te = interpolate_dynamic_quantile(
        test_gpu_q_z, QUANTILES, test_p_cal, tau=0.48, q_floor=0.25, q_ceil=0.50
    )
    test_dyn_gpu_pred = np.clip(
        np.exp(z_dyn_gpu_te) * (base_test + 1.0) - 1.0, 0, 300000
    )

    atten_test = (
        np.maximum(2 * test_p_gate - 1.0, 0.0) ** gamma_opt if gamma_opt > 0 else 1.0
    )
    pos_comb_test = (
        w_med * atten_test * test_pos_pred
        + w_dyn_lgb * test_dyn_lgb_pred
        + w_dyn_gpu * test_dyn_gpu_pred
        + w_dir * test_dir_pred
    )
    final_test_preds = np.where(test_p_gate >= tau_opt, pos_comb_test * alpha_opt, 0.0)
    final_test_preds = np.round(final_test_preds, 2)

    sample_sub = pd.read_csv(ROOT.parent / "submission" / "sample_submission.csv")
    submit_df = pd.DataFrame({"id": sample_sub["id"], "task3": final_test_preds})
    submit_path = SUBMIT_DIR / "index.csv"
    submit_df.to_csv(submit_path, index=False)
    print(f"Wrote submission file to {submit_path} ({len(submit_df)} rows)")

    pred_full = pd.DataFrame(
        {
            "player_id": test_df["player_id"],
            "spending_30d": final_test_preds,
            "p_spender": test_p_gate,
            "p_calibrated": test_p_cal,
            "pred_positive_median": test_pos_pred,
            "pred_dynamic_lgb": test_dyn_lgb_pred,
            "pred_dynamic_gpu": test_dyn_gpu_pred,
            "pred_direct": test_dir_pred,
        }
    )
    pred_path = PRED_DIR / "test_predictions.csv"
    pred_full.to_csv(pred_path, index=False)
    print(f"Wrote full predictions to {pred_path} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

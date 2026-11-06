"""
Task 5: Unsupervised Account Security Anomaly Detection
Course: CPE 342 Machine Learning — KMUTT
Metric: F3 Score (10 * P * R / (9 * P + R), beta=3: Recall weighted 9x over Precision)

Multi-Detector Ensemble:
1. Exact Synthetic Domain Archetype Detectors:
   - Archetype 1: Account Takeover (ATO) & Asset Liquidation (password_changed, mass_item_sale, vpn_usage)
   - Archetype 2: Period-4 Directional Jump Violations (Aimbot injection, currency drain, skill surge, geo teleport)
   - Archetype 3: Behavior Stability Collapse (< 0.60) & Silent Bot Network (< 0.175)
   - Archetype 4: Scripted Farming Bot Clusters (variance collapse across IP entropy, duration, KDA)
2. Assignment 9 Coursework Latent & Residual Subspace Anomaly Scoring:
   - Latent Subspace Hotelling's T^2 (Mahalanobis distance on PCA scores)
   - Residual Subspace Reconstruction Error (SPE / MAE)
3. Density-Based Outlier Ranking:
   - Isolation Forest (500 trees)
   - Local Outlier Factor (LOF negative outlier factor)
4. F3-Optimal Threshold Calibration:
   - Flag union of deterministic archetypes + top borderline accounts to reach optimal Recall (k = 3,500, ~13.5%)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from scipy.stats import rankdata

def run_sota_task5():
    task_dir = Path(__file__).resolve().parent
    data_path = task_dir / "data" / "test.csv"
    submit_dir = task_dir / "submit"
    pred_dir = task_dir / "predictions"
    assets_dir = task_dir / "assets"

    submit_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    n_samples, n_cols = df.shape
    print(f"Loaded {n_samples:,} rows, {n_cols} columns.")

    player_ids = df["player_id"].values
    feature_cols = [c for c in df.columns if c != "player_id"]
    X_raw = df[feature_cols].copy()

    # Robust imputation for continuous subspace models
    medians = X_raw.median()
    X_filled = X_raw.fillna(medians)

    # -------------------------------------------------------------------------
    # 1. Deterministic Domain Anomaly Archetypes
    # -------------------------------------------------------------------------
    print("Computing deterministic anomaly archetypes...")

    # Archetype 1: Account Takeover (ATO) & Liquidation
    ato_risk = (
        (df["password_changed"] == 1) & (
            (df["vpn_usage"] == 1) | 
            (df["suspicious_login_time"] == 1) | 
            (df["mass_item_sale"] == 1)
        )
    ).fillna(False)

    mass_drain = (
        (df["mass_item_sale"] == 1) & (df["currency_transfers_4"] > 2.0)
    ).fillna(False)

    arch1_mask = ato_risk | mass_drain

    # Archetype 2: Period-4 Directional Jump Violations
    def get_max_prior(prefix):
        cols = [f"{prefix}_{i}" for i in range(1, 4)]
        return df[cols].max(axis=1)

    def get_min_prior(prefix):
        cols = [f"{prefix}_{i}" for i in range(1, 4)]
        return df[cols].min(axis=1)

    diff_curr = df["currency_transfers_4"] - get_max_prior("currency_transfers")
    diff_skill_traj = df["skill_trajectory_4"] - get_max_prior("skill_trajectory")
    diff_skill_rate = df["skill_rating_4"] - get_max_prior("skill_rating")
    diff_loc = df["location_changes_4"] - get_max_prior("location_changes")
    diff_purch_f = df["purchase_frequency_4"] - get_max_prior("purchase_frequency")
    diff_purch_a = df["purchase_amount_mean_4"] - get_max_prior("purchase_amount_mean")
    diff_acc = df["accuracy_mean_4"] - get_max_prior("accuracy_mean")
    diff_kda = df["kda_mean_4"] - get_max_prior("kda_mean")
    diff_react = df["reaction_time_mean_4"] - get_min_prior("reaction_time_mean")
    diff_pat = df["play_pattern_consistency_4"] - get_min_prior("play_pattern_consistency")
    diff_rem = df["friend_removes_4"] - get_max_prior("friend_removes")

    lat_j = np.abs(df["login_lat_4"] - df["login_lat_3"])
    lon_j = np.abs(df["login_lon_4"] - df["login_lon_3"])
    geo_j = np.sqrt(lat_j**2 + lon_j**2)

    arch2_mask = (
        (diff_curr > 4.5) |
        (diff_skill_traj > 9.5) |
        (diff_skill_rate > 220.0) |
        (diff_loc > 1.6) |
        (diff_purch_f > 2.8) |
        (diff_purch_a > 180.0) |
        (diff_acc > 7.5) |
        (diff_kda > 0.95) |
        (diff_react < -70.0) |
        (diff_pat < -0.28) |
        (diff_rem > 4.0) |
        (geo_j > 30.0)
    ).fillna(False)

    # Archetype 3: Low Stability & Silent Bots
    g_stab = (df["behavior_stability"] < 0.60).fillna(False)
    comm_cols = [f"communication_frequency_{i}" for i in range(1, 5)]
    g_silent = (df[comm_cols].mean(axis=1) < 0.175).fillna(False)
    arch3_mask = g_stab | g_silent

    # Archetype 4: Automated Farming Bots (Variance Collapse)
    arch4_mask = (
        (df[[f"ip_hash_entropy_{i}" for i in range(1, 5)]].mean(axis=1) < 0.25) &
        (df[[f"login_time_variance_{i}" for i in range(1, 5)]].mean(axis=1) < 0.60) &
        (df[[f"session_duration_std_{i}" for i in range(1, 5)]].mean(axis=1) < 5.50) &
        (df[[f"kda_std_{i}" for i in range(1, 5)]].mean(axis=1) < 0.12)
    ).fillna(False)

    core_anomalies = arch1_mask | arch2_mask | arch3_mask | arch4_mask
    print(f"Deterministic Core Anomalies: {core_anomalies.sum():,} ({core_anomalies.mean()*100:.2f}%)")

    # -------------------------------------------------------------------------
    # 2. Assignment 9 Coursework Subspace & Outlier Models
    # -------------------------------------------------------------------------
    print("Fitting Continuous Unsupervised Detectors...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_filled)

    # 2.1 Isolation Forest
    print("  -> Running Isolation Forest (500 trees)...")
    iso = IsolationForest(n_estimators=500, max_samples=0.8, contamination="auto", random_state=42, n_jobs=-1)
    iso.fit(X_scaled)
    iso_scores = -iso.score_samples(X_scaled)
    iso_rank = rankdata(iso_scores) / n_samples

    # 2.2 Local Outlier Factor (LOF)
    print("  -> Running Local Outlier Factor...")
    lof = LocalOutlierFactor(n_neighbors=35, novelty=False, n_jobs=-1)
    lof.fit_predict(X_scaled)
    lof_scores = -lof.negative_outlier_factor_
    lof_rank = rankdata(lof_scores) / n_samples

    # 2.3 Assignment 9 Latent Subspace Hotelling's T^2 & Residual SPE
    print("  -> Computing PCA Latent Subspace T^2 & Residual SPE...")
    pca = PCA(n_components=14, random_state=42)
    Z = pca.fit_transform(X_scaled)
    X_recon = pca.inverse_transform(Z)
    spe = np.mean(np.abs(X_scaled - X_recon), axis=1)
    spe_rank = rankdata(spe) / n_samples

    var = np.maximum(pca.explained_variance_, 1e-6)
    t2 = np.sum((Z ** 2) / var, axis=1)
    t2_rank = rankdata(t2) / n_samples

    # Composite Continuous Anomaly Rank
    cont_score = (
        0.30 * iso_rank +
        0.25 * lof_rank +
        0.25 * t2_rank +
        0.20 * spe_rank
    )

    # -------------------------------------------------------------------------
    # 3. F3-Optimal Threshold Calibration (Recall Priority)
    # -------------------------------------------------------------------------
    # In F3 metric (10*P*R/(9*P+R)), Recall is 9x more important than Precision.
    # Marginal precision of only 10% improves F3 score.
    # Target k = 3,500 accounts (~13.52% contamination rate).
    target_positive_count = 3500
    final_preds = core_anomalies.values.astype(int)

    remaining_budget = target_positive_count - final_preds.sum()
    if remaining_budget > 0:
        candidate_indices = np.where(final_preds == 0)[0]
        sorted_candidates = candidate_indices[np.argsort(-cont_score[candidate_indices])]
        top_candidates = sorted_candidates[:remaining_budget]
        final_preds[top_candidates] = 1

    total_flagged = final_preds.sum()
    flagged_pct = (total_flagged / n_samples) * 100
    print(f"\nFinal Task 5 Predictions: {total_flagged:,} anomalies ({flagged_pct:.2f}%)")

    # -------------------------------------------------------------------------
    # 4. Save Submission & Predictions
    # -------------------------------------------------------------------------
    sample_sub_path = task_dir.parent / "submission" / "sample_submission.csv"
    if sample_sub_path.exists():
        sample_ids = pd.read_csv(sample_sub_path)["id"].values
    else:
        sample_ids = [f"ANS{i+1:05d}" for i in range(n_samples)]

    submit_df = pd.DataFrame({
        "id": sample_ids,
        "task5": final_preds
    })
    submit_path = submit_dir / "index.csv"
    submit_df.to_csv(submit_path, index=False)
    print(f"Wrote submission file to {submit_path} ({len(submit_df):,} rows)")

    pred_full_df = pd.DataFrame({
        "player_id": player_ids,
        "is_anomaly": final_preds,
        "is_core_anomaly": core_anomalies.astype(int),
        "continuous_anomaly_score": cont_score,
        "iso_rank": iso_rank,
        "lof_rank": lof_rank,
        "spe_rank": spe_rank,
        "t2_rank": t2_rank
    })
    pred_path = pred_dir / "test_predictions.csv"
    pred_full_df.to_csv(pred_path, index=False)
    print(f"Wrote test predictions with scores to {pred_path}")

    # -------------------------------------------------------------------------
    # 5. Diagnostic Figures
    # -------------------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Continuous Anomaly Score Distribution
    sns.histplot(cont_score[final_preds == 0], bins=50, color="steelblue", alpha=0.6, label="Normal", ax=axes[0], stat="density")
    sns.histplot(cont_score[final_preds == 1], bins=50, color="crimson", alpha=0.7, label=f"Anomaly ({flagged_pct:.1f}%)", ax=axes[0], stat="density")
    axes[0].set_title(f"Task 5 Multi-Detector Score Distribution (N={n_samples:,})", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Composite Anomaly Percentile Rank")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # Subplot 2: Anomaly Archetype Breakdown
    counts = [arch1_mask.sum(), arch2_mask.sum(), arch3_mask.sum(), arch4_mask.sum(), (final_preds & ~core_anomalies).sum()]
    labels = ["ATO / Liquidation", "Period-4 Jumps", "Stability/Silence", "Farming Bots", "Borderline Tail"]
    axes[1].barh(labels, counts, color=["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#9b59b6"])
    axes[1].set_title(f"F3-Optimized Anomaly Composition (k = {total_flagged:,})", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Flagged Accounts")
    for i, v in enumerate(counts):
        axes[1].text(v + 20, i, f"{v:,}", va="center", fontweight="bold")

    plt.tight_layout()
    fig_path = assets_dir / "task5_anomaly_detection_diagnostics.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved diagnostic figure to {fig_path}")

if __name__ == "__main__":
    run_sota_task5()


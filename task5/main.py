"""
Task 5: Unsupervised Account Security Anomaly Detection
Course: CPE 342 Machine Learning — KMUTT
Metric: F3 Score (10 * P * R / (9 * P + R), beta=3: Recall weighted 9x over Precision)

Multi-Detector Ensemble:
1. Exact Synthetic Anomaly Archetype Detectors:
   - Group 1: Period-4 Directional Jump Anomalies (Takeover/Drain, Boosting/Aimbot, Payment/VPN Fraud)
   - Group 2: Low Behavior Stability Gap Anomalies (< 0.60)
   - Group 3: Silent Bot Network Gap Anomalies (< 0.175)
   - Group 4: Automated Scripted Farming Bot Clusters
2. Continuous Unsupervised Density & Subspace Outlier Detectors:
   - Isolation Forest (500 trees)
   - Local Outlier Factor (LOF negative_outlier_factor_)
   - PCA Residual Reconstruction Error (SPE) + Hotelling's T^2 (Mahalanobis Distance)
   - Static Account Risk Score (password_changed, mass_item_sale, suspicious_login_time, vpn_usage)
3. F3-Optimal Threshold Calibration:
   - Evaluates union of deterministic archetypes + top ranked borderline tail
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

def run_task5_pipeline():
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

    # Fill NaNs with robust median for continuous detectors
    medians = X_raw.median()
    X_filled = X_raw.fillna(medians)

    # -------------------------------------------------------------------------
    # 1. Deterministic Synthetic Anomaly Detectors
    # -------------------------------------------------------------------------
    print("Computing deterministic anomaly archetypes...")

    # Group 2: Low Behavior Stability
    g2_mask = (df["behavior_stability"] < 0.60).fillna(False)

    # Group 3: Silent Bots (Communication < 0.175)
    comm_cols = [f"communication_frequency_{i}" for i in range(1, 5)]
    g3_mask = (df[comm_cols].mean(axis=1) < 0.175).fillna(False)

    # Group 4: Automated Farming Bots
    g4_mask = (
        (df[[f"ip_hash_entropy_{i}" for i in range(1, 5)]].mean(axis=1) < 0.25) &
        (df[[f"login_time_variance_{i}" for i in range(1, 5)]].mean(axis=1) < 0.60) &
        (df[[f"session_duration_std_{i}" for i in range(1, 5)]].mean(axis=1) < 5.50) &
        (df[[f"kda_std_{i}" for i in range(1, 5)]].mean(axis=1) < 0.12)
    ).fillna(False)

    # Group 1: Period-4 Directional Jump Violations
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
    diff_adds = df["friend_adds_4"] - get_max_prior("friend_adds")

    lat_j = np.abs(df["login_lat_4"] - df["login_lat_3"])
    lon_j = np.abs(df["login_lon_4"] - df["login_lon_3"])
    geo_j = np.sqrt(lat_j**2 + lon_j**2)

    g1_mask = (
        (diff_curr > 5.7) |
        (diff_skill_traj > 11.86) |
        (diff_skill_rate > 289.3) |
        (diff_loc > 2.01) |
        (diff_purch_f > 3.24) |
        (diff_purch_a > 213.1) |
        (diff_acc > 9.75) |
        (diff_kda > 1.21) |
        (diff_react < -60.7) |
        (diff_pat < -0.152) |
        (diff_rem > 2.73) |
        (diff_adds > 12.88) |
        (geo_j > 3.0)
    ).fillna(False)

    core_anomalies = g1_mask | g2_mask | g3_mask | g4_mask
    print(f"  Group 1 (Period-4 jumps): {g1_mask.sum():,} accounts")
    print(f"  Group 2 (Low stability): {g2_mask.sum():,} accounts")
    print(f"  Group 3 (Silent bots): {g3_mask.sum():,} accounts")
    print(f"  Group 4 (Farming bots): {g4_mask.sum():,} accounts")
    print(f"  Total Core Deterministic Anomalies: {core_anomalies.sum():,} accounts ({core_anomalies.mean()*100:.2f}%)")

    # -------------------------------------------------------------------------
    # 2. Continuous Multi-Detector Unsupervised Ensemble
    # -------------------------------------------------------------------------
    print("Fitting continuous density and subspace detectors...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_filled)

    # 2.1 Isolation Forest
    print("  -> Training Isolation Forest (500 estimators)...")
    iso = IsolationForest(
        n_estimators=500,
        max_samples=0.8,
        contamination=0.08,
        random_state=42,
        n_jobs=-1
    )
    iso.fit(X_scaled)
    iso_scores = -iso.score_samples(X_scaled)
    iso_rank = rankdata(iso_scores) / n_samples

    # 2.2 Local Outlier Factor
    print("  -> Computing Local Outlier Factor (LOF)...")
    lof = LocalOutlierFactor(n_neighbors=35, novelty=False, n_jobs=-1)
    lof.fit_predict(X_scaled)
    lof_scores = -lof.negative_outlier_factor_
    lof_rank = rankdata(lof_scores) / n_samples

    # 2.3 PCA Reconstruction Error (SPE) + Mahalanobis Latent T^2
    print("  -> Computing PCA Reconstruction Error & Mahalanobis Distance...")
    pca = PCA(n_components=12, random_state=42)
    Z = pca.fit_transform(X_scaled)
    X_recon = pca.inverse_transform(Z)
    spe = np.mean(np.abs(X_scaled - X_recon), axis=1)
    spe_rank = rankdata(spe) / n_samples

    var = np.maximum(pca.explained_variance_, 1e-6)
    t2 = np.sum((Z ** 2) / var, axis=1)
    t2_rank = rankdata(t2) / n_samples

    # 2.4 Static Security Risk Score
    risk_flags = df[["password_changed", "mass_item_sale", "suspicious_login_time", "vpn_usage"]].fillna(0).sum(axis=1)
    risk_rank = rankdata(risk_flags) / n_samples

    # Composite Continuous Anomaly Score
    cont_score = (
        0.30 * iso_rank +
        0.25 * lof_rank +
        0.20 * spe_rank +
        0.15 * t2_rank +
        0.10 * risk_rank
    )

    # -------------------------------------------------------------------------
    # 3. F3-Optimal Thresholding
    # -------------------------------------------------------------------------
    # F3 weights Recall 9x over Precision. Include all deterministic anomalies
    # plus the top borderline accounts from the continuous score.
    target_positive_count = 2150  # ~8.3% contamination
    final_preds = core_anomalies.values.astype(int)

    remaining_budget = target_positive_count - final_preds.sum()
    if remaining_budget > 0:
        candidate_indices = np.where(final_preds == 0)[0]
        sorted_candidates = candidate_indices[np.argsort(-cont_score[candidate_indices])]
        top_candidates = sorted_candidates[:remaining_budget]
        final_preds[top_candidates] = 1

    total_flagged = final_preds.sum()
    flagged_pct = (total_flagged / n_samples) * 100
    print(f"Final Task 5 Predictions: {total_flagged:,} anomalies ({flagged_pct:.2f}%)")

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
        "spe_rank": spe_rank
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
    sns.histplot(cont_score[final_preds == 0], bins=50, color="steelblue", alpha=0.6, label="Normal (91.7%)", ax=axes[0], stat="density")
    sns.histplot(cont_score[final_preds == 1], bins=50, color="crimson", alpha=0.7, label=f"Anomaly ({flagged_pct:.1f}%)", ax=axes[0], stat="density")
    axes[0].set_title("Multi-Detector Anomaly Score Distribution", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Composite Anomaly Percentile Rank")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # Subplot 2: Anomaly Archetype Breakdown
    counts = [g1_mask.sum(), g2_mask.sum(), g3_mask.sum(), g4_mask.sum(), (final_preds & ~core_anomalies).sum()]
    labels = ["Period-4 Jumps", "Low Stability", "Silent Bots", "Farming Bots", "Borderline Tail"]
    axes[1].barh(labels, counts, color=["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#9b59b6"])
    axes[1].set_title("Anomaly Detection Composition (N = 2,150)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Account Count")
    for i, v in enumerate(counts):
        axes[1].text(v + 20, i, f"{v:,}", va="center", fontweight="bold")

    plt.tight_layout()
    fig_path = assets_dir / "task5_anomaly_detection_diagnostics.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved diagnostic figure to {fig_path}")

if __name__ == "__main__":
    run_task5_pipeline()

"""
Task 4: GPU-Accelerated (MPS/CUDA) DINOv2 + Cosine k-NN + Mixup Residual Head + Coordinate Ascent Macro-F1
Implements Deep Research recommendations for CPE 342 Game Title Detection (5-Class Macro F1):
1. Multi-modal representation: Meta DINOv2 ViT-S/14 (384-d, L2-normalized) + Spatial/Color/Edge descriptors (687-d)
2. GPU-accelerated Cosine Similarity k-NN visual stream retriever (capturing near-duplicate frames from the same stream)
3. PyTorch Residual MLP + Cosine Classifier trained on GPU (MPS/CUDA) with Feature Mixup (alpha=0.2) & Label Smoothing (0.08)
4. Multi-family ensemble (GPU Mixup-ResNet + LightGBM + Logistic Regression)
5. Post-hoc Coordinate Ascent on Class Logit Offsets directly maximizing 5-Fold OOF Macro F1
"""

from __future__ import annotations

import os
import platform
import time
import warnings
from pathlib import Path

if platform.system() == "Darwin":
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

if platform.system() == "Darwin":
    torch.set_num_threads(1)
LGB_N_JOBS = 1 if platform.system() == "Darwin" else -1

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PRED_DIR = ROOT / "predictions"
SUBMIT_DIR = ROOT / "submit"
ASSETS_DIR = ROOT / "assets"

SEED = 42
CLASSES = ["7-11K", "fiveN", "freefried", "pube", "roblock"]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_gpu_cosine_knn_features(
    X_dino_tr: np.ndarray,
    y_tr: np.ndarray,
    X_dino_query: np.ndarray,
    device: torch.device,
    k: int = 15,
    exclude_self: bool = False,
    batch_size: int = 2048,
) -> np.ndarray:
    """
    Uses GPU matrix multiplication to compute temperature-weighted Cosine k-NN class probabilities
    and max similarity (stream near-duplicate detector) against the training DINOv2 bank.
    """
    tr_norm = F.normalize(
        torch.tensor(X_dino_tr, dtype=torch.float32, device=device), p=2, dim=1
    )
    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=device)
    y_onehot = F.one_hot(y_tr_t, num_classes=5).float()  # [N_tr, 5]

    n_q = len(X_dino_query)
    knn_probs = np.zeros((n_q, 6), dtype=np.float32)  # 5 class probs + max_sim

    with torch.no_grad():
        for start in range(0, n_q, batch_size):
            end = min(start + batch_size, n_q)
            q_norm = F.normalize(
                torch.tensor(
                    X_dino_query[start:end], dtype=torch.float32, device=device
                ),
                p=2,
                dim=1,
            )
            sim = q_norm @ tr_norm.T  # [B, N_tr]
            if exclude_self:
                # Mask diagonal for in-sample queries
                row_idx = torch.arange(end - start, device=device)
                col_idx = torch.arange(start, end, device=device)
                sim[row_idx, col_idx] = -1.0

            top_sim, top_idx = torch.topk(sim, k=k, dim=1)  # [B, k]
            weights = F.softmax(top_sim * 15.0, dim=1).unsqueeze(-1)  # [B, k, 1]
            neighbor_labels = y_onehot[top_idx]  # [B, k, 5]
            weighted_probs = (weights * neighbor_labels).sum(dim=1)  # [B, 5]
            max_sim = top_sim[:, 0:1]  # [B, 1]
            out_batch = torch.cat([weighted_probs, max_sim], dim=1)
            knn_probs[start:end] = out_batch.cpu().numpy()

    return knn_probs


class VisionResidualMLP(nn.Module):
    """GPU Residual MLP with Cosine-Normalized Classifier Head for DINOv2 + Multi-Modal Embeddings."""

    def __init__(self, in_dim: int, hidden_dim: int = 512, num_classes: int = 5):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.30),
        )
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.30),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.block2 = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(0.20),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(0.15),
        )
        self.head = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = F.silu(h + self.block1(h))
        h = self.block2(h)
        return self.head(h)


def train_gpu_vision_mlp_fold(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    X_te: np.ndarray,
    device: torch.device,
    seed: int = 42,
    epochs: int = 24,
    batch_size: int = 256,
    mixup_alpha: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    model = VisionResidualMLP(
        in_dim=X_tr.shape[1], hidden_dim=512, num_classes=5
    ).to(device)

    # Mild inverse-frequency class weights + label smoothing 0.08
    counts = np.bincount(y_tr, minlength=5).astype(np.float32)
    inv_w = (counts.sum() / (5.0 * counts)) ** 0.30
    class_weights = torch.tensor(inv_w, dtype=torch.float32, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=device)
    X_va_t = torch.tensor(X_va, dtype=torch.float32, device=device)
    X_te_t = torch.tensor(X_te, dtype=torch.float32, device=device)
    n_tr = len(X_tr_t)
    bs = 512

    best_f1 = -1.0
    best_va_prob = None
    best_te_prob = None

    for epoch in range(epochs):
        model.train()
        perm_ep = torch.randperm(n_tr, device=device)
        for i in range(0, n_tr, bs):
            idx = perm_ep[i : i + bs]
            bx = X_tr_t[idx]
            by = y_tr_t[idx]
            optimizer.zero_grad(set_to_none=True)

            if mixup_alpha > 0 and rng.random() < 0.65:
                lam = float(rng.beta(mixup_alpha, mixup_alpha))
                perm = torch.randperm(bx.size(0), device=device)
                mixed_x = lam * bx + (1.0 - lam) * bx[perm]
                logits = model(mixed_x)
                loss = lam * F.cross_entropy(
                    logits, by, weight=class_weights, label_smoothing=0.08
                ) + (1.0 - lam) * F.cross_entropy(
                    logits, by[perm], weight=class_weights, label_smoothing=0.08
                )
            else:
                logits = model(bx)
                loss = F.cross_entropy(
                    logits, by, weight=class_weights, label_smoothing=0.08
                )

            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        best_va_prob = torch.softmax(model(X_va_t), dim=1).cpu().numpy()
        best_te_prob = torch.softmax(model(X_te_t), dim=1).cpu().numpy()

    return best_va_prob, best_te_prob


def optimize_macro_f1_offsets(
    probs: np.ndarray, y_true: np.ndarray, n_rounds: int = 4
) -> np.ndarray:
    """Coordinate Ascent on log-probabilities to directly maximize 5-Class Macro F1."""
    log_p = np.log(np.clip(probs, 1e-7, 1.0))
    offsets = np.zeros(5, dtype=np.float64)
    best_score = f1_score(y_true, log_p.argmax(axis=1), average="macro")

    for _ in range(n_rounds):
        for c in range(1, 5):
            best_c = offsets[c]
            for delta in np.linspace(-0.45, 0.45, 181):
                cand = offsets.copy()
                cand[c] = delta
                preds = (log_p + cand).argmax(axis=1)
                score = f1_score(y_true, preds, average="macro")
                if score > best_score + 1e-6:
                    best_score = score
                    best_c = delta
            offsets[c] = best_c
    return offsets


def main() -> None:
    t0 = time.time()
    device = get_device()
    print(f"=== Task 4: GPU-Accelerated DINOv2 + Mixup-ResNet + Cosine-kNN Pipeline (Device: {device}) ===")

    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    raw_labels = train_df["class"].astype(str).str.strip().values
    raw_labels = np.where(raw_labels == "7-11k", "7-11K", raw_labels)
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    idx_to_class = {i: c for i, c in enumerate(CLASSES)}
    y_train = np.array([class_to_idx[c] for c in raw_labels], dtype=np.int64)

    print("1. Loading DINOv2 ViT-S/14 & Spatial/Color/Edge features...")
    X_dino_train = np.load(PRED_DIR / "dinov2_train_features.npz")["X"].astype(np.float32)
    X_dino_test = np.load(PRED_DIR / "dinov2_test_features.npz")["X"].astype(np.float32)

    X_base_train = np.load(PRED_DIR / "train-features.npz")["X"].astype(np.float32)
    X_base_test = np.load(PRED_DIR / "test-features.npz")["X"].astype(np.float32)

    # L2 normalize DINOv2 representations before concatenation
    dino_tr_norm = X_dino_train / (
        np.linalg.norm(X_dino_train, axis=1, keepdims=True) + 1e-8
    )
    dino_te_norm = X_dino_test / (
        np.linalg.norm(X_dino_test, axis=1, keepdims=True) + 1e-8
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    oof_knn = np.zeros((len(y_train), 6), dtype=np.float32)
    test_knn = np.zeros((len(test_df), 6), dtype=np.float32)

    print("2. Computing GPU Cosine-Similarity k-NN stream retrieval features...")
    for fold, (tr_idx, val_idx) in enumerate(skf.split(dino_tr_norm, y_train), 1):
        oof_knn[val_idx] = compute_gpu_cosine_knn_features(
            dino_tr_norm[tr_idx],
            y_train[tr_idx],
            dino_tr_norm[val_idx],
            device=device,
            k=15,
        )
        test_knn += (
            compute_gpu_cosine_knn_features(
                dino_tr_norm[tr_idx],
                y_train[tr_idx],
                dino_te_norm,
                device=device,
                k=15,
            )
            / 5.0
        )

    knn_oof_f1 = f1_score(y_train, oof_knn[:, :5].argmax(axis=1), average="macro")
    print(f"   Standalone GPU Cosine k-NN (k=15) OOF Macro F1: {knn_oof_f1:.5f}")

    # Concatenate DINOv2 (384) + Spatial/Color (687) + Cosine-kNN (6) -> 1,077 features
    X_all = np.hstack([dino_tr_norm, X_base_train, oof_knn])
    X_test_all = np.hstack([dino_te_norm, X_base_test, test_knn])

    scaler = StandardScaler()
    X_all_sc = scaler.fit_transform(X_all).astype(np.float32)
    X_test_sc = scaler.transform(X_test_all).astype(np.float32)

    # Compact 70-d representation (64-d strided DINOv2/spatial descriptors + 6-d GPU Cosine k-NN) for fast GBDT splitting
    stride_idx = np.linspace(0, X_all_sc.shape[1] - 7, 64, dtype=int)
    X_gbdt_all = np.hstack([X_all_sc[:, stride_idx], oof_knn]).astype(np.float32)
    X_gbdt_test = np.hstack([X_test_sc[:, stride_idx], test_knn]).astype(np.float32)

    oof_gpu_mlp = np.zeros((len(y_train), 5), dtype=np.float64)
    test_gpu_mlp = np.zeros((len(test_df), 5), dtype=np.float64)

    oof_lr = np.zeros((len(y_train), 5), dtype=np.float64)
    test_lr = np.zeros((len(test_df), 5), dtype=np.float64)

    oof_lgb = np.zeros((len(y_train), 5), dtype=np.float64)
    test_lgb = np.zeros((len(test_df), 5), dtype=np.float64)

    print("3. Training 5-Fold GPU Mixup-ResNet + GPU Cosine Linear Probe + LightGBM...")
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all_sc, y_train), 1):
        X_tr, y_tr = X_all_sc[tr_idx], y_train[tr_idx]
        X_va, y_va = X_all_sc[val_idx], y_train[val_idx]

        # Model 1: GPU VisionResidualMLP with Mixup & Label Smoothing
        va_mlp, te_mlp = train_gpu_vision_mlp_fold(
            X_tr,
            y_tr,
            X_va,
            y_va,
            X_test_sc,
            device=device,
            seed=SEED + fold,
            epochs=22,
            batch_size=512,
            mixup_alpha=0.20,
        )
        oof_gpu_mlp[val_idx] = va_mlp
        test_gpu_mlp += te_mlp / 5.0

        # Model 2: GPU L2-Regularized Linear Probe (trained on MPS/CUDA)
        torch.manual_seed(SEED + 100 + fold)
        lin_probe = nn.Linear(X_tr.shape[1], 5).to(device)
        opt_lp = torch.optim.AdamW(lin_probe.parameters(), lr=3e-3, weight_decay=0.15)
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
        y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=device)
        X_va_t = torch.tensor(X_va, dtype=torch.float32, device=device)
        X_te_t = torch.tensor(X_test_sc, dtype=torch.float32, device=device)
        for _ in range(16):
            perm_lp = torch.randperm(len(X_tr_t), device=device)
            for i in range(0, len(X_tr_t), 1024):
                idx = perm_lp[i : i + 1024]
                opt_lp.zero_grad(set_to_none=True)
                loss_lp = F.cross_entropy(lin_probe(X_tr_t[idx]), y_tr_t[idx], label_smoothing=0.04)
                loss_lp.backward()
                opt_lp.step()
        with torch.no_grad():
            oof_lr[val_idx] = torch.softmax(lin_probe(X_va_t), dim=1).cpu().numpy()
            test_lr += torch.softmax(lin_probe(X_te_t), dim=1).cpu().numpy() / 5.0

        # Model 3: LightGBM Classifier on 64-d PCA + 6-d GPU Cosine k-NN
        lgb_params = {
            "objective": "multiclass",
            "num_class": 5,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.07,
            "num_leaves": 31,
            "max_bin": 127,
            "feature_fraction": 0.80,
            "bagging_fraction": 0.80,
            "bagging_freq": 1,
            "verbose": -1,
            "random_state": SEED + fold,
            "n_jobs": LGB_N_JOBS,
        }
        dtrain = lgb.Dataset(X_gbdt_all[tr_idx], label=y_tr)
        dval = lgb.Dataset(X_gbdt_all[val_idx], label=y_va, reference=dtrain)
        gbm = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=320,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(35, verbose=False), lgb.log_evaluation(0)],
        )
        oof_lgb[val_idx] = gbm.predict(X_gbdt_all[val_idx], num_iteration=gbm.best_iteration)
        test_lgb += gbm.predict(X_gbdt_test, num_iteration=gbm.best_iteration) / 5.0

        print(
            f"   Fold {fold}/5 | GPU_MixupMLP F1: {f1_score(y_va, va_mlp.argmax(1), average='macro'):.5f} | "
            f"LGBM F1: {f1_score(y_va, oof_lgb[val_idx].argmax(1), average='macro'):.5f} | "
            f"LR F1: {f1_score(y_va, oof_lr[val_idx].argmax(1), average='macro'):.5f}"
        )

    print("\n================ 5-FOLD OOF SUMMARY ================")
    print(f"GPU Mixup-ResNet OOF Macro F1:    {f1_score(y_train, oof_gpu_mlp.argmax(1), average='macro'):.5f}")
    print(f"LightGBM OOF Macro F1:            {f1_score(y_train, oof_lgb.argmax(1), average='macro'):.5f}")
    print(f"Logistic Regression OOF Macro F1: {f1_score(y_train, oof_lr.argmax(1), average='macro'):.5f}")

    # Search optimal 3-model ensemble weights
    best_f1 = -1.0
    best_w = (0.4, 0.4, 0.2)
    best_oof_blend = None

    for w_mlp in np.linspace(0.40, 0.95, 12):
        for w_lgb in np.linspace(0.00, 0.30, 7):
            w_lr = round(1.0 - w_mlp - w_lgb, 4)
            if w_lr < -1e-5 or w_lr > 0.40:
                continue
            w_lr = max(0.0, w_lr)
            blend = w_mlp * oof_gpu_mlp + w_lgb * oof_lgb + w_lr * oof_lr
            score = f1_score(y_train, blend.argmax(axis=1), average="macro")
            if score > best_f1:
                best_f1 = score
                best_w = (w_mlp, w_lgb, w_lr)
                best_oof_blend = blend.copy()

    w_mlp, w_lgb, w_lr = best_w
    print(
        f"Optimal Blend ({w_mlp:.2f} GPU_MixupMLP + {w_lgb:.2f} LGBM + {w_lr:.2f} LR) "
        f"OOF Macro F1 (argmax): {best_f1:.5f}"
    )

    # Coordinate Ascent on Class Logit Offsets (Nested OOF check)
    nested_preds = np.zeros(len(y_train), dtype=int)
    fold_offsets = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_all_sc, y_train), 1):
        off_k = optimize_macro_f1_offsets(best_oof_blend[tr_idx], y_train[tr_idx], n_rounds=3)
        fold_offsets.append(off_k)
        log_p_va = np.log(np.clip(best_oof_blend[val_idx], 1e-7, 1.0))
        nested_preds[val_idx] = (log_p_va + off_k).argmax(axis=1)

    nested_f1 = f1_score(y_train, nested_preds, average="macro")
    final_offsets = 0.5 * np.mean(fold_offsets, axis=0) + 0.5 * optimize_macro_f1_offsets(
        best_oof_blend, y_train, n_rounds=4
    )
    oof_final_preds = (
        np.log(np.clip(best_oof_blend, 1e-7, 1.0)) + final_offsets
    ).argmax(axis=1)
    final_oof_f1 = f1_score(y_train, oof_final_preds, average="macro")

    print(
        f"Coordinate Ascent Logit Offsets: {np.round(final_offsets, 4).tolist()} | "
        f"Nested OOF Macro F1: {nested_f1:.5f} | Final OOF Macro F1: {final_oof_f1:.5f}"
    )
    print("\nClassification Report (Final OOF):")
    print(classification_report(y_train, oof_final_preds, target_names=CLASSES, digits=4))

    # Predict on Test Set
    final_test_prob = w_mlp * test_gpu_mlp + w_lgb * test_lgb + w_lr * test_lr
    if nested_f1 >= best_f1:
        test_pred_indices = (
            np.log(np.clip(final_test_prob, 1e-7, 1.0)) + final_offsets
        ).argmax(axis=1)
    else:
        test_pred_indices = final_test_prob.argmax(axis=1)

    test_pred_labels = [idx_to_class[i] for i in test_pred_indices]

    sample_sub = pd.read_csv(ROOT.parent / "submission" / "sample_submission.csv")
    submit_df = pd.DataFrame({"id": sample_sub["id"], "task4": test_pred_labels})
    submit_df.to_csv(SUBMIT_DIR / "index.csv", index=False)

    # Also save detailed test predictions + alternative encodings for diagnostic analysis
    pred_df = pd.DataFrame(
        {
            "id": test_df["id"].values,
            "predicted_class": test_pred_labels,
            "predicted_int": test_pred_indices,
        }
    )
    for i, c in enumerate(CLASSES):
        pred_df[f"prob_{c}"] = final_test_prob[:, i]
    pred_df.to_csv(PRED_DIR / "test_predictions.csv", index=False)

    print(
        f"Saved Task 4 predictions to {SUBMIT_DIR / 'index.csv'}: "
        f"{pd.Series(test_pred_labels).value_counts().to_dict()} in {time.time() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()


"""
Task 4: Multi-Head Deep Residual & Cosine Bottleneck Candidate (fold-local class weights)
Implements:
1. Joint 1152-d Multi-Scale Representation:
   - Meta DINOv2 ViT-Base/14 (768-d, L2-normalized)
   - Meta DINOv2 ViT-Small/14 (384-d, L2-normalized)
   - Strictly fold-local z-score standardization inside 5-Fold Stratified CV
2. 3-Head Diverse Neural Ensemble:
   - M1: DeepResidualMLP (768-d, 2 residual blocks, GELU, BatchNorm, Dropout 0.25, Mixup alpha=0.15)
   - M2: CosineBottleneckMLP (1024 -> 384 normalized cosine hypersphere classifier with learnable temperature)
   - M3: Class-Balanced DeepResidualMLP (inverse square-root class weighting for Macro F1, Mixup alpha=0.20)
3. Official Numeric Target Formatting per task4/data/datacard.txt & TA Specification:
   - 0: 7-11 Knights ('7-11k')
   - 1: FiveN ('fiveN')
   - 2: Free Fried ('freefried')
   - 3: PUBE ('pube')
   - 4: Roblock ('roblock')
   - Strictly preserves test.csv row order (Row 0 -> ANS00001 .. Row 25888 -> ANS25889)
Previously saved OOF array: Macro F1 0.9470246 / accuracy 95.39%; not promoted; requires fold-local-weight rerun
"""

from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PRED_DIR = ROOT / "predictions"
SUBMIT_DIR = ROOT / "submit"
SUBMIT_DIR.mkdir(parents=True, exist_ok=True)

# Official datacard.txt mapping:
# 0: 7-11 Knights, 1: FiveN, 2: Free Fried, 3: PUBE, 4: Roblock
CLASSES = ["7-11K", "fiveN", "freefried", "pube", "roblock"]
CLASS_TITLES = ["7-11 Knights", "FiveN", "Free Fried", "PUBE", "Roblock"]
c2i = {c: i for i, c in enumerate(CLASSES)}


class DeepResidualMLP(nn.Module):
    def __init__(self, in_dim: int = 1152, hidden_dim: int = 768, num_classes: int = 5, dropout: float = 0.25):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )
        self.act = nn.GELU()
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.act(h + self.block1(h))
        h = self.act(h + self.block2(h))
        return self.head(h)


class CosineBottleneckMLP(nn.Module):
    def __init__(self, in_dim: int = 1152, hidden_dim: int = 1024, neck_dim: int = 384, num_classes: int = 5, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, neck_dim),
            nn.BatchNorm1d(neck_dim),
        )
        self.weight = nn.Parameter(torch.randn(num_classes, neck_dim) * 0.02)
        self.scale = nn.Parameter(torch.tensor(16.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.normalize(self.net(x), dim=1)
        w = F.normalize(self.weight, dim=1)
        return self.scale * (z @ w.T)


def train_head(
    model: nn.Module,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    X_te: np.ndarray,
    device: torch.device,
    epochs: int = 25,
    lr: float = 1e-3,
    class_weights: torch.Tensor | None = None,
    mixup_alpha: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.5e-2)
    ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    loader = DataLoader(ds, batch_size=256, shuffle=True, drop_last=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(loader), eta_min=1e-5)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            if mixup_alpha > 0.0 and np.random.rand() < 0.5:
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                perm = torch.randperm(bx.size(0), device=device)
                mx = lam * bx + (1.0 - lam) * bx[perm]
                logits = model(mx)
                loss = lam * criterion(logits, by) + (1.0 - lam) * criterion(logits, by[perm])
            else:
                logits = model(bx)
                loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            scheduler.step()

    model.eval()
    with torch.no_grad():
        va_logits = model(torch.tensor(X_va, dtype=torch.float32, device=device))
        va_probs = F.softmax(va_logits, dim=1).cpu().numpy()
        te_logits = model(torch.tensor(X_te, dtype=torch.float32, device=device))
        te_probs = F.softmax(te_logits, dim=1).cpu().numpy()
    return va_probs, te_probs


def main():
    t0 = time.time()
    print("=== Generating Task 4 multi-head candidate with fold-local class weights (Numeric 0-4) ===")

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    raw_labels = train_df["class"].astype(str).str.strip().values
    raw_labels = np.where(raw_labels == "7-11k", "7-11K", raw_labels)
    y = np.array([c2i[c] for c in raw_labels], dtype=np.int64)

    X_b_tr = np.load(PRED_DIR / "dinov2_vitb14_train_features.npz")["X"].astype(np.float32)
    X_b_te = np.load(PRED_DIR / "dinov2_vitb14_test_features.npz")["X"].astype(np.float32)
    X_s_tr = np.load(PRED_DIR / "dinov2_train_features.npz")["X"].astype(np.float32)
    X_s_te = np.load(PRED_DIR / "dinov2_test_features.npz")["X"].astype(np.float32)

    b_tr_norm = X_b_tr / (np.linalg.norm(X_b_tr, axis=1, keepdims=True) + 1e-8)
    b_te_norm = X_b_te / (np.linalg.norm(X_b_te, axis=1, keepdims=True) + 1e-8)
    s_tr_norm = X_s_tr / (np.linalg.norm(X_s_tr, axis=1, keepdims=True) + 1e-8)
    s_te_norm = X_s_te / (np.linalg.norm(X_s_te, axis=1, keepdims=True) + 1e-8)

    j_tr = np.hstack([b_tr_norm, s_tr_norm])
    j_te = np.hstack([b_te_norm, s_te_norm])

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n = len(y)
    n_te = len(test_df)

    oof_m1 = np.zeros((n, 5), dtype=np.float32)
    oof_m2 = np.zeros((n, 5), dtype=np.float32)
    oof_m3 = np.zeros((n, 5), dtype=np.float32)

    test_m1 = np.zeros((n_te, 5), dtype=np.float32)
    test_m2 = np.zeros((n_te, 5), dtype=np.float32)
    test_m3 = np.zeros((n_te, 5), dtype=np.float32)

    for fold, (tr, va) in enumerate(skf.split(j_tr, y), 1):
        ft0 = time.time()
        mu = j_tr[tr].mean(axis=0, keepdims=True)
        sd = j_tr[tr].std(axis=0, keepdims=True) + 1e-6
        X_tr_std = (j_tr[tr] - mu) / sd
        X_va_std = (j_tr[va] - mu) / sd
        X_te_std = (j_te - mu) / sd

        fold_counts = np.bincount(y[tr], minlength=5).astype(np.float32)
        cw_balanced = torch.as_tensor(
            (len(tr) / (5.0 * fold_counts)) ** 0.5, dtype=torch.float32, device=device
        )

        torch.manual_seed(42 + fold)
        np.random.seed(42 + fold)
        m1 = DeepResidualMLP(in_dim=1152, hidden_dim=768, num_classes=5, dropout=0.25)
        va1, te1 = train_head(m1, X_tr_std, y[tr], X_va_std, X_te_std, device, epochs=25, lr=1e-3, class_weights=None, mixup_alpha=0.15)
        oof_m1[va] = va1
        test_m1 += te1 / 5.0

        torch.manual_seed(142 + fold)
        np.random.seed(142 + fold)
        m2 = CosineBottleneckMLP(in_dim=1152, hidden_dim=1024, neck_dim=384, num_classes=5, dropout=0.25)
        va2, te2 = train_head(m2, X_tr_std, y[tr], X_va_std, X_te_std, device, epochs=25, lr=1e-3, class_weights=None, mixup_alpha=0.15)
        oof_m2[va] = va2
        test_m2 += te2 / 5.0

        torch.manual_seed(242 + fold)
        np.random.seed(242 + fold)
        m3 = DeepResidualMLP(in_dim=1152, hidden_dim=768, num_classes=5, dropout=0.28)
        va3, te3 = train_head(m3, X_tr_std, y[tr], X_va_std, X_te_std, device, epochs=25, lr=1e-3, class_weights=cw_balanced, mixup_alpha=0.20)
        oof_m3[va] = va3
        test_m3 += te3 / 5.0

        blend_va = 0.40 * va1 + 0.30 * va2 + 0.30 * va3
        print(f"  Fold {fold}/5 ({time.time() - ft0:.1f}s) Super-Blend Macro F1: {f1_score(y[va], blend_va.argmax(1), average='macro'):.5f}")

    oof_super = 0.40 * oof_m1 + 0.30 * oof_m2 + 0.30 * oof_m3
    test_super = 0.40 * test_m1 + 0.30 * test_m2 + 0.30 * test_m3

    oof_preds = oof_super.argmax(axis=1)
    oof_f1 = f1_score(y, oof_preds, average="macro")
    oof_acc = (y == oof_preds).mean()

    print("\n================ FINAL 5-FOLD SUPER-BLEND OOF ================")
    print(f"Overall OOF Macro F1: {oof_f1:.6f} (Overall Accuracy: {oof_acc * 100:.2f}%)")
    print(classification_report(y, oof_preds, target_names=[f"{i}:{t}" for i, t in enumerate(CLASS_TITLES)], digits=4))

    np.save(PRED_DIR / "test_super_probabilities.npy", test_super)

    # Strictly follow test.csv row order: Row 0 -> ANS00001 .. Row 25888 -> ANS25889
    ans_ids = [f"ANS{i:05d}" for i in range(1, len(test_df) + 1)]
    test_int_labels = test_super.argmax(axis=1).astype(int)

    submit_df = pd.DataFrame({"id": ans_ids, "task4": test_int_labels})
    submit_path = SUBMIT_DIR / "index.csv"
    submit_df.to_csv(submit_path, index=False)

    pred_df = pd.DataFrame({
        "id": ans_ids,
        "image_id": test_df["id"].values,
        "task4": test_int_labels,
        "game_title": [CLASSES[idx] for idx in test_int_labels],
        "prob_0_711K": test_super[:, 0],
        "prob_1_fiveN": test_super[:, 1],
        "prob_2_freefried": test_super[:, 2],
        "prob_3_pube": test_super[:, 3],
        "prob_4_roblock": test_super[:, 4],
    })
    pred_df.to_csv(PRED_DIR / "test_predictions.csv", index=False)
    print(f"Saved Task 4 numeric predictions (0..4) to {submit_path} ({len(submit_df)} rows) in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

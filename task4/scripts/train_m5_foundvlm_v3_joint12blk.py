#!/usr/bin/env python3
"""Train True Joint 12-Block (13,568d -> 768 -> 5) Dual-Condition FoundVLM-v3 on Apple M5 MPS.

Memory-safe per-fold fp16 assembly (<1.8 GB CPU RAM) with mini-batch-to-MPS transfer (<80 MB VRAM).
Verified ~1.85s/epoch on MPS (task-29069).
Saves strictly to scratch/t4_m5_outputs/task4-foundvlm-v3-joint12blk.npz (does NOT touch repos).
"""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import gc
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
from sklearn.metrics import f1_score, log_loss

SEED = 42
SCRATCH_DIR = Path(__file__).resolve().parent
M5_OUT_DIR = SCRATCH_DIR / "t4_m5_outputs"
CPE_T4_DIR = Path("/Users/wisitsuwannao/Documents/Workspace/cpe342-machine-learning-2026/project/task4")
ML_T4_DIR = Path("/Users/wisitsuwannao/Documents/Workspace/ml00x/task4")

CLASS_MAP: Dict[str, int] = {
    "7-11k": 0,
    "7-11K": 0,
    "fiveN": 1,
    "freefried": 2,
    "pube": 3,
    "roblock": 4,
}

BACKBONE_FILES: List[Tuple[str, Path]] = [
    ("siglip2_so400m", CPE_T4_DIR / "emb_v4_siglip2_so400m.npz"),
    ("dinov3_vitl", CPE_T4_DIR / "emb_v4_dinov3_vitl.npz"),
    ("dinov2_vitg14", CPE_T4_DIR / "emb_v4_dinov2_vitg14.npz"),
    ("clip_cnxl", M5_OUT_DIR / "emb_v4_clip_cnxl.npz"),
    ("clip_vith14", M5_OUT_DIR / "emb_v4_clip_vith14.npz"),
    ("dinov3_cnxb", M5_OUT_DIR / "emb_v4_dinov3_cnxb.npz"),
]


def _l2_norm_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-8, None)


def _power_cl2n_f16(x: np.ndarray, mu: np.ndarray, scale: float, beta: float = 0.75) -> np.ndarray:
    z = _l2_norm_np(np.asarray(x, dtype=np.float32) - mu)
    if beta != 1.0:
        z = _l2_norm_np(np.sign(z) * (np.abs(z) ** beta))
    return (z * np.float32(scale)).astype(np.float16)


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def unif_to_train_prior(p_unif: np.ndarray, pi_vec: np.ndarray) -> np.ndarray:
    p = np.clip(p_unif, 1e-7, 1.0) * pi_vec[None, :]
    return p / p.sum(axis=1, keepdims=True)


def train_to_unif_prior(p_train: np.ndarray, pi_vec: np.ndarray) -> np.ndarray:
    p = np.clip(p_train, 1e-7, 1.0) / np.clip(pi_vec[None, :], 1e-7, 1.0)
    return p / p.sum(axis=1, keepdims=True)


def geo_pool(ps: Sequence[np.ndarray], ws: Sequence[float]) -> np.ndarray:
    lp = sum(float(wi) * np.log(np.clip(pi, 1e-7, 1.0)) for wi, pi in zip(ws, ps) if float(wi) > 0.0)
    return _softmax_np(lp)


def fit_slsqp_simplex(p_list: Sequence[np.ndarray], y_lbl: np.ndarray, max_w: float = 0.85) -> np.ndarray:
    m = len(p_list)
    lps = [np.log(np.clip(p, 1e-7, 1.0)) for p in p_list]
    idx = np.arange(len(y_lbl))

    def obj(w: np.ndarray) -> float:
        w = np.maximum(w, 0.0)
        w /= w.sum() + 1e-12
        pr = _softmax_np(sum(wi * lp_i for wi, lp_i in zip(w, lps)))
        return float(-np.mean(np.log(np.clip(pr[idx, y_lbl], 1e-12, 1.0))))

    res = minimize(
        obj,
        np.ones(m) / m,
        bounds=[(0.0, float(max_w))] * m,
        constraints={"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0},
        method="SLSQP",
    )
    w = np.maximum(res.x, 0.0)
    return w / w.sum()


def fit_slsqp_hybrid(
    p_clean_list: Sequence[np.ndarray],
    p_ext_list: Sequence[np.ndarray],
    y_lbl: np.ndarray,
    w_clean_loss: float = 0.75,
    max_w: float = 0.85,
) -> np.ndarray:
    m = len(p_clean_list)
    lps_c = [np.log(np.clip(p, 1e-7, 1.0)) for p in p_clean_list]
    lps_e = [np.log(np.clip(p, 1e-7, 1.0)) for p in p_ext_list]
    idx = np.arange(len(y_lbl))

    def _nll(lps: Sequence[np.ndarray], w: np.ndarray) -> float:
        pr = _softmax_np(sum(wi * lp_i for wi, lp_i in zip(w, lps)))
        return float(-np.mean(np.log(np.clip(pr[idx, y_lbl], 1e-12, 1.0))))

    def obj(w: np.ndarray) -> float:
        w = np.maximum(w, 0.0)
        w /= w.sum() + 1e-12
        return w_clean_loss * _nll(lps_c, w) + (1.0 - w_clean_loss) * _nll(lps_e, w)

    res = minimize(
        obj,
        np.ones(m) / m,
        bounds=[(0.0, float(max_w))] * m,
        constraints={"type": "eq", "fun": lambda w: float(np.sum(w)) - 1.0},
        method="SLSQP",
    )
    w = np.maximum(res.x, 0.0)
    return w / w.sum()


def _apply_block_drop(x: torch.Tensor, training: bool, block_slices: Tuple[Tuple[int, int], ...], block_drop: float) -> torch.Tensor:
    if not training or len(block_slices) <= 1 or block_drop <= 0.0:
        return x
    bsz = x.shape[0]
    n_blk = len(block_slices)
    keep = (torch.rand((bsz, n_blk), device=x.device) >= block_drop).float()
    none_kept = keep.sum(dim=1, keepdim=True) == 0
    if none_kept.any():
        rand_blk = F.one_hot(torch.randint(0, n_blk, (bsz,), device=x.device), num_classes=n_blk).float()
        keep = torch.where(none_kept, rand_blk, keep)
    scale = float(n_blk) / keep.sum(dim=1, keepdim=True)
    masks = [keep[:, bi : bi + 1].expand(-1, e - s) for bi, (s, e) in enumerate(block_slices)]
    return x * (torch.cat(masks, dim=1) * scale)


class CosineResMLP(nn.Module):
    def __init__(self, in_dim: int, block_slices: Sequence[Tuple[int, int]], hidden: int = 768, drop: float = 0.50, block_drop: float = 0.22):
        super().__init__()
        self.block_slices = tuple(block_slices)
        self.block_drop = float(block_drop)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(drop)
        self.res_fc = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.w = nn.Parameter(torch.randn(5, hidden) * 0.02)
        self.scale = nn.Parameter(torch.tensor(16.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _apply_block_drop(x, self.training, self.block_slices, self.block_drop)
        h = F.gelu(self.ln1(self.fc1(x)))
        h = h + self.drop(F.gelu(self.ln2(self.res_fc(self.drop(h)))))
        z = F.normalize(h, p=2, dim=-1)
        w = F.normalize(self.w, p=2, dim=-1)
        return self.scale.clamp(4.0, 40.0) * (z @ w.T)


class CosineLinear(nn.Module):
    def __init__(self, in_dim: int, block_slices: Sequence[Tuple[int, int]], block_drop: float = 0.15):
        super().__init__()
        self.block_slices = tuple(block_slices)
        self.block_drop = float(block_drop)
        self.w = nn.Parameter(torch.randn(5, in_dim) * 0.02)
        self.scale = nn.Parameter(torch.tensor(16.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _apply_block_drop(x, self.training, self.block_slices, self.block_drop)
        z = F.normalize(x, p=2, dim=-1)
        w = F.normalize(self.w, p=2, dim=-1)
        return self.scale.clamp(4.0, 40.0) * (z @ w.T)


def train_minibatch_mps(
    model_fn,
    X_tr_c: np.ndarray,
    X_tr_e: np.ndarray,
    y_tr: np.ndarray,
    eval_list: Sequence[np.ndarray],
    seeds: Sequence[int],
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream 512-row mini-batches from CPU float16 to MPS float32 (<80 MB VRAM)."""
    pi_k = np.bincount(y_tr, minlength=5).astype(np.float64) / len(y_tr)
    log_pi = torch.tensor(np.log(np.clip(pi_k, 1e-4, 1.0)), device=device, dtype=torch.float32)
    n = len(y_tr)
    steps_per_ep = math.ceil(n / batch_size)

    p_va_c = np.zeros((len(eval_list[0]), 5), dtype=np.float64)
    p_va_e = np.zeros((len(eval_list[1]), 5), dtype=np.float64)
    p_te_c = np.zeros((len(eval_list[2]), 5), dtype=np.float64)
    t_kl = 1.5

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        model = model_fn().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs * steps_per_ep), eta_min=1e-5)

        for _ep in range(epochs):
            model.train()
            perm = rng.permutation(n)
            for s in range(0, n, batch_size):
                idx = perm[s : s + batch_size]
                xc = torch.from_numpy(X_tr_c[idx]).to(device, dtype=torch.float32)
                xe = torch.from_numpy(X_tr_e[idx]).to(device, dtype=torch.float32)
                yb = torch.from_numpy(y_tr[idx].astype(np.int64)).to(device)

                xc = xc + torch.randn_like(xc) * 0.020
                xe = xe + torch.randn_like(xe) * 0.020
                yb_oh = F.one_hot(yb, num_classes=5).float() * 0.98 + (0.02 / 5.0)
                if rng.random() < 0.50 and len(xc) > 1:
                    lam = float(rng.beta(0.40, 0.40))
                    mp = torch.randperm(len(xc), device=device)
                    xc = lam * xc + (1.0 - lam) * xc[mp]
                    xe = lam * xe + (1.0 - lam) * xe[mp]
                    yb_oh = lam * yb_oh + (1.0 - lam) * yb_oh[mp]

                opt.zero_grad(set_to_none=True)
                post_c = model(xc) + log_pi.unsqueeze(0)
                post_e = model(xe) + log_pi.unsqueeze(0)
                ce_c = -(yb_oh * F.log_softmax(post_c, dim=1)).sum(dim=1).mean()
                ce_e = -(yb_oh * F.log_softmax(post_e, dim=1)).sum(dim=1).mean()

                pc = F.softmax(post_c / t_kl, dim=1)
                pe = F.softmax(post_e / t_kl, dim=1)
                lpc = F.log_softmax(post_c / t_kl, dim=1)
                lpe = F.log_softmax(post_e / t_kl, dim=1)
                sym_kl = 0.5 * ((pc * (lpc - lpe)).sum(dim=1).mean() + (pe * (lpe - lpc)).sum(dim=1).mean())

                loss = ce_c + 0.45 * ce_e + 0.25 * sym_kl
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                sched.step()

        model.eval()
        outs = []
        with torch.no_grad():
            for Xe_np in eval_list:
                chunks = []
                for s in range(0, len(Xe_np), 1024):
                    xt = torch.from_numpy(Xe_np[s : s + 1024]).to(device, dtype=torch.float32)
                    chunks.append(F.softmax(model(xt), dim=1).cpu().numpy())
                outs.append(np.concatenate(chunks, axis=0))
        p_va_c += outs[0] / len(seeds)
        p_va_e += outs[1] / len(seeds)
        p_te_c += outs[2] / len(seeds)
        del model, opt, sched

    if device.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    return unif_to_train_prior(p_va_c, pi_k), unif_to_train_prior(p_va_e, pi_k), unif_to_train_prior(p_te_c, pi_k)


def atomic_savez_compressed(path: Path, **arrays) -> None:
    tmp_path = path.with_suffix(f".tmp.{os.getpid()}.npz")
    np.savez_compressed(tmp_path, **arrays)
    os.replace(tmp_path, path)


def main() -> None:
    t0 = time.time()
    M5_OUT_DIR.mkdir(parents=True, exist_ok=True)
    v2_path = M5_OUT_DIR / "task4-foundvlm-v2-backup.npz"
    v3_fast_path = M5_OUT_DIR / "task4-foundvlm-v3-backup.npz"
    assert v2_path.is_file() and v3_fast_path.is_file()

    tr_df = pd.read_csv(ML_T4_DIR / "data/train.csv")
    y = tr_df["class"].map(CLASS_MAP).astype(int).to_numpy()
    folds = np.load(ML_T4_DIR / "predictions/model-level/task4-sgkf-e2e-bundle.npz")["folds"].astype(int)
    n_tr, n_te = len(y), 25889
    pi_all = np.bincount(y, minlength=5) / n_tr
    sw_all = 1.0 / pi_all[y]

    with np.load(v2_path) as z2:
        v2_o2, v2_w6, v2_ex = z2["oof_orig2"].astype(np.float64), z2["oof_w6"].astype(np.float64), z2["ext_w6"].astype(np.float64)
        v2_to2, v2_tw6 = z2["te_orig2"].astype(np.float64), z2["te_w6"].astype(np.float64)
    with np.load(v3_fast_path) as z3:
        v3f_o2, v3f_w6, v3f_ex = z3["oof_orig2"].astype(np.float64), z3["oof_w6"].astype(np.float64), z3["ext_w6"].astype(np.float64)
        v3f_to2, v3f_tw6 = z3["te_orig2"].astype(np.float64), z3["te_w6"].astype(np.float64)

    slices_12: List[Tuple[int, int]] = []
    bb_spans: List[Tuple[int, int, int, int]] = []
    cur = 0
    for key, fpath in BACKBONE_FILES:
        with np.load(fpath) as z:
            d = int(z["tr_all4"].shape[1])
            assert z["tr_all4"].shape[0] == n_tr and z["te_all4"].shape[0] == n_te
            s0 = cur
            slices_12.append((s0, s0 + d))
            s1 = s0 + d
            slices_12.append((s1, s1 + d))
            bb_spans.append((s0, s0 + d, s1, s1 + d))
            cur = s1 + d
    d_total = cur

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"[Joint12Blk] Total dim={d_total} (12 blocks) | Device={device}", flush=True)

    inv_s12 = 1.0 / math.sqrt(12.0)
    oof_j1, ext_j1, te_j1 = np.zeros((n_tr, 5)), np.zeros((n_tr, 5)), np.zeros((n_te, 5))
    oof_j2, ext_j2, te_j2 = np.zeros((n_tr, 5)), np.zeros((n_tr, 5)), np.zeros((n_te, 5))

    for k in range(5):
        tf0 = time.time()
        tr_m, va_m = (folds != k), (folds == k)
        n_trk, n_vak = int(tr_m.sum()), int(va_m.sum())
        y_trk = y[tr_m]

        X_tr_c = np.empty((n_trk, d_total), dtype=np.float16)
        X_tr_e = np.empty((n_trk, d_total), dtype=np.float16)
        X_va_c = np.empty((n_vak, d_total), dtype=np.float16)
        X_va_e = np.empty((n_vak, d_total), dtype=np.float16)
        X_te_c = np.empty((n_te, d_total), dtype=np.float16)

        for bi, (_, fpath) in enumerate(BACKBONE_FILES):
            s0, e0, s1, e1 = bb_spans[bi]
            with np.load(fpath) as z:
                r_tr = _l2_norm_np(z["tr_raw2"])
                a_tr = _l2_norm_np(z["tr_all4"])
                r_ex = _l2_norm_np(z["ext_raw2"])
                a_ex = _l2_norm_np(z["ext_all4"])
                r_te = _l2_norm_np(z["te_raw2"])
                a_te = _l2_norm_np(z["te_all4"])

            mu_r = r_tr[tr_m].mean(axis=0, keepdims=True)
            mu_a = a_tr[tr_m].mean(axis=0, keepdims=True)
            X_tr_c[:, s0:e0] = _power_cl2n_f16(r_tr[tr_m], mu_r, inv_s12)
            X_tr_c[:, s1:e1] = _power_cl2n_f16(a_tr[tr_m], mu_a, inv_s12)
            X_tr_e[:, s0:e0] = _power_cl2n_f16(r_ex[tr_m], mu_r, inv_s12)
            X_tr_e[:, s1:e1] = _power_cl2n_f16(a_ex[tr_m], mu_a, inv_s12)
            X_va_c[:, s0:e0] = _power_cl2n_f16(r_tr[va_m], mu_r, inv_s12)
            X_va_c[:, s1:e1] = _power_cl2n_f16(a_tr[va_m], mu_a, inv_s12)
            X_va_e[:, s0:e0] = _power_cl2n_f16(r_ex[va_m], mu_r, inv_s12)
            X_va_e[:, s1:e1] = _power_cl2n_f16(a_ex[va_m], mu_a, inv_s12)
            X_te_c[:, s0:e0] = _power_cl2n_f16(r_te, mu_r, inv_s12)
            X_te_c[:, s1:e1] = _power_cl2n_f16(a_te, mu_a, inv_s12)
            del r_tr, a_tr, r_ex, a_ex, r_te, a_te
        gc.collect()
        t_prep = time.time() - tf0

        t_mlp = time.time()
        ov1, oe1, ot1 = train_minibatch_mps(
            lambda: CosineResMLP(d_total, slices_12, hidden=768, drop=0.50, block_drop=0.22),
            X_tr_c, X_tr_e, y_trk, [X_va_c, X_va_e, X_te_c],
            seeds=(42 + k * 37, 2025 + k * 37), epochs=14, lr=1.2e-3, weight_decay=2.5e-2, device=device, batch_size=512,
        )
        oof_j1[va_m], ext_j1[va_m] = ov1, oe1
        te_j1 += ot1 / 5.0

        ov2, oe2, ot2 = train_minibatch_mps(
            lambda: CosineLinear(d_total, slices_12, block_drop=0.15),
            X_tr_c, X_tr_e, y_trk, [X_va_c, X_va_e, X_te_c],
            seeds=(11 + k * 19, 22 + k * 19), epochs=12, lr=3.0e-3, weight_decay=1.0e-2, device=device, batch_size=512,
        )
        oof_j2[va_m], ext_j2[va_m] = ov2, oe2
        te_j2 += ot2 / 5.0

        del X_tr_c, X_tr_e, X_va_c, X_va_e, X_te_c
        gc.collect()
        print(
            f"  [Fold {k}] total={time.time()-tf0:.1f}s (prep={t_prep:.1f}s, mps={time.time()-t_mlp:.1f}s) "
            f"| J1_ClnF1={f1_score(y[va_m], ov1.argmax(1), average='macro'):.4f} "
            f"J1_ExtF1={f1_score(y[va_m], oe1.argmax(1), average='macro'):.4f} "
            f"J2_LinF1={f1_score(y[va_m], ov2.argmax(1), average='macro'):.4f}",
            flush=True,
        )

    for hname, po, pe in [
        ("v2_anchor_w6", v2_w6, v2_ex),
        ("v3_fast_backup", v3f_w6, v3f_ex),
        ("J1_Joint12Blk_ResMLP", oof_j1, ext_j1),
        ("J2_Joint12Blk_CosLin", oof_j2, ext_j2),
    ]:
        print(
            f"  [Head] {hname:22s} | ClnF1={f1_score(y, po.argmax(1), average='macro'):.5f} "
            f"| ExtF1={f1_score(y, pe.argmax(1), average='macro'):.5f} "
            f"| ClnLL={log_loss(y, np.clip(po, 1e-7, 1.0), labels=[0,1,2,3,4]):.5f} "
            f"| ExtLL={log_loss(y, np.clip(pe, 1e-7, 1.0), labels=[0,1,2,3,4]):.5f}",
            flush=True,
        )

    # Fold-Internal Outer SLSQP combining [v2_anchor, J1_Joint12Blk, J2_Joint12Blk, v3_fast]
    oof_o2_j = np.zeros((n_tr, 5), dtype=np.float64)
    oof_w6_j = np.zeros((n_tr, 5), dtype=np.float64)
    ext_w6_j = np.zeros((n_tr, 5), dtype=np.float64)
    ws_o2, ws_w6 = [], []

    for k in range(5):
        tr_m, va_m = (folds != k), (folds == k)
        wo = fit_slsqp_simplex([v2_o2[tr_m], oof_j1[tr_m], oof_j2[tr_m], v3f_o2[tr_m]], y[tr_m], max_w=0.85)
        wh = fit_slsqp_hybrid(
            [v2_w6[tr_m], oof_j1[tr_m], oof_j2[tr_m], v3f_w6[tr_m]],
            [v2_ex[tr_m], ext_j1[tr_m], ext_j2[tr_m], v3f_ex[tr_m]],
            y[tr_m], w_clean_loss=0.75, max_w=0.85,
        )
        ws_o2.append(wo); ws_w6.append(wh)
        oof_o2_j[va_m] = geo_pool([v2_o2[va_m], oof_j1[va_m], oof_j2[va_m], v3f_o2[va_m]], wo)
        oof_w6_j[va_m] = geo_pool([v2_w6[va_m], oof_j1[va_m], oof_j2[va_m], v3f_w6[va_m]], wh)
        ext_w6_j[va_m] = geo_pool([v2_ex[va_m], ext_j1[va_m], ext_j2[va_m], v3f_ex[va_m]], wh)

    wm_o2, wm_w6 = np.mean(ws_o2, axis=0), np.mean(ws_w6, axis=0)
    te_o2_j = geo_pool([v2_to2, te_j1, te_j2, v3f_to2], wm_o2)
    te_w6_j = geo_pool([v2_tw6, te_j1, te_j2, v3f_tw6], wm_w6)

    oof_uj, ext_uj = np.zeros_like(oof_w6_j), np.zeros_like(ext_w6_j)
    oof_u2, ext_u2 = np.zeros_like(v2_w6), np.zeros_like(v2_ex)
    for k in range(5):
        tr_m, va_m = (folds != k), (folds == k)
        pi_k = np.bincount(y[tr_m], minlength=5).astype(np.float64) / tr_m.sum()
        oof_uj[va_m] = train_to_unif_prior(oof_w6_j[va_m], pi_k)
        ext_uj[va_m] = train_to_unif_prior(ext_w6_j[va_m], pi_k)
        oof_u2[va_m] = train_to_unif_prior(v2_w6[va_m], pi_k)
        ext_u2[va_m] = train_to_unif_prior(v2_ex[va_m], pi_k)

    rng = np.random.default_rng(SEED)
    pjc, p2c = oof_uj.argmax(1), oof_u2.argmax(1)
    pje, p2e = ext_uj.argmax(1), ext_u2.argmax(1)
    win_c, win_e = 0, 0
    for _ in range(500):
        idx = rng.integers(0, n_tr, size=n_tr)
        if f1_score(y[idx], pjc[idx], average="macro", sample_weight=sw_all[idx]) > f1_score(
            y[idx], p2c[idx], average="macro", sample_weight=sw_all[idx]
        ):
            win_c += 1
        if f1_score(y[idx], pje[idx], average="macro", sample_weight=sw_all[idx]) > f1_score(
            y[idx], p2e[idx], average="macro", sample_weight=sw_all[idx]
        ):
            win_e += 1

    print(f"\n=== [Joint12Blk FoundVLM-v3] COMPLETE in {time.time()-t0:.1f}s ===", flush=True)
    print(f"  Outer orig2 w [v2, J1_13568d, J2_CosLin, v3_fast]: {np.round(wm_o2, 4).tolist()}", flush=True)
    print(f"  Outer hard  w [v2, J1_13568d, J2_CosLin, v3_fast]: {np.round(wm_w6, 4).tolist()}", flush=True)
    print(
        f"  v2 Baseline  : UnifF1={f1_score(y, p2c, average='macro', sample_weight=sw_all):.5f} "
        f"| OOF_LL={log_loss(y, np.clip(v2_w6, 1e-7, 1.0)):.5f} "
        f"| ExtUnifF1={f1_score(y, p2e, average='macro', sample_weight=sw_all):.5f} "
        f"| ExtLL={log_loss(y, np.clip(v2_ex, 1e-7, 1.0)):.5f}",
        flush=True,
    )
    print(
        f"  Joint12Blk v3: UnifF1={f1_score(y, pjc, average='macro', sample_weight=sw_all):.5f} "
        f"| OOF_LL={log_loss(y, np.clip(oof_w6_j, 1e-7, 1.0)):.5f} "
        f"| ExtUnifF1={f1_score(y, pje, average='macro', sample_weight=sw_all):.5f} "
        f"| ExtLL={log_loss(y, np.clip(ext_w6_j, 1e-7, 1.0)):.5f}",
        flush=True,
    )
    print(f"  500-Boot vs v2: CleanUnifWin={win_c/500:.3f} | ExtHardUnifWin={win_e/500:.3f}", flush=True)

    out_npz = M5_OUT_DIR / "task4-foundvlm-v3-joint12blk.npz"
    atomic_savez_compressed(
        out_npz,
        folds=folds,
        oof_orig2=oof_o2_j.astype(np.float32),
        oof_w6=oof_w6_j.astype(np.float32),
        ext_w6=ext_w6_j.astype(np.float32),
        te_orig2=te_o2_j.astype(np.float32),
        te_w6=te_w6_j.astype(np.float32),
    )
    print(f"Saved -> {out_npz} (repos untouched)", flush=True)


if __name__ == "__main__":
    main()

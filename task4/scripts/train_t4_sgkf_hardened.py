"""5-Fold SGKF Corruption-Hardened Trainer for Task 4 (2:1 Aspect Ratio, Pure-Train).

Targets the four visual corruption regimes observed in Task 4:
1. Rotation [-35, +35] deg with black corner borders
2. Extreme overexposure (white-blown highlights) & ultra-dark underexposure + fog
3. Neon RGB channel gain tints (without destructive inversion) & contrast compression
4. Shot/Gaussian noise, Gaussian blur, and low-quality JPEG artifacts

Includes Turing (Tesla T4, sm_75) safety, disconnect resilience, & speed upgrades:
- Forces float16 + GradScaler on sm_75 GPUs even when torch.cuda.is_bf16_supported()
  returns True via emulated software support (avoids cuDNN FIND failures on depthwise 7x7).
- Uses contiguous NCHW on T4 + 1-step forward/backward/eval preflight probe.
- Vectorized GPU 6-view TTA (interior_gw_stretch_torch + uint8 raw loader) avoids 2-vCPU stalls.
- Epoch-level atomic checkpoints (fold{k}_ckpt.pt) resume cleanly after Colab disconnects.
- Smart ViT partial freeze keeps patch_embed/cls/reg_token trainable for RGB adaptation & grad_ckpt.

Usage:
    python train_t4_sgkf_hardened.py --data-dir /content/t4_data --out-dir /content/drive/MyDrive/t4_hard_outputs --models dinov3l eva02l
"""
from __future__ import annotations

import argparse
import gc
import io
import math
import os
import random
import shutil
import time
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFilter
from sklearn.metrics import f1_score, log_loss
from torch.utils.data import DataLoader, Dataset


CLASS_MAP: Dict[str, int] = {
    "7-11k": 0,
    "7-11K": 0,
    "fiveN": 1,
    "freefried": 2,
    "pube": 3,
    "roblock": 4,
}


@dataclass(frozen=True)
class ModelSpec:
    """Configuration for a corruption-hardened Task 4 vision backbone."""

    key: str
    timm_name: str
    img_hw: Tuple[int, int]
    epochs: int
    base_lr: float
    backbone_lr_mult: float
    batch_a100: int
    batch_l4: int
    batch_t4: int
    eval_batch_t4: int
    grad_ckpt: bool
    use_img_size_arg: bool
    channels_last_ampere: bool
    default_freeze_vit_blocks: int


MODEL_SPECS: Dict[str, ModelSpec] = {
    "cnxv3": ModelSpec(
        key="cnxv3",
        timm_name="convnext_base.dinov3_lvd1689m",
        img_hw=(192, 384),
        epochs=6,
        base_lr=3.0e-4,
        backbone_lr_mult=0.25,
        batch_a100=96,
        batch_l4=64,
        batch_t4=48,
        eval_batch_t4=96,
        grad_ckpt=False,
        use_img_size_arg=False,
        channels_last_ampere=True,
        default_freeze_vit_blocks=0,
    ),
    "cnxv2b": ModelSpec(
        key="cnxv2b",
        timm_name="convnextv2_base.fcmae_ft_in22k_in1k",
        img_hw=(192, 384),
        epochs=6,
        base_lr=3.0e-4,
        backbone_lr_mult=0.25,
        batch_a100=96,
        batch_l4=64,
        batch_t4=48,
        eval_batch_t4=96,
        grad_ckpt=False,
        use_img_size_arg=False,
        channels_last_ampere=True,
        default_freeze_vit_blocks=0,
    ),
    "eva02l": ModelSpec(
        key="eva02l",
        timm_name="eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        img_hw=(196, 392),
        epochs=4,
        base_lr=2.0e-4,
        backbone_lr_mult=0.22,
        batch_a100=48,
        batch_l4=32,
        batch_t4=16,
        eval_batch_t4=48,
        grad_ckpt=True,
        use_img_size_arg=True,
        channels_last_ampere=False,
        default_freeze_vit_blocks=6,
    ),
    "dinov3l": ModelSpec(
        key="dinov3l",
        timm_name="vit_large_patch16_dinov3.lvd1689m",
        img_hw=(192, 384),
        epochs=5,
        base_lr=2.0e-4,
        backbone_lr_mult=0.22,
        batch_a100=64,
        batch_l4=32,
        batch_t4=16,
        eval_batch_t4=48,
        grad_ckpt=True,
        use_img_size_arg=True,
        channels_last_ampere=False,
        default_freeze_vit_blocks=6,
    ),
}


def seed_everything(seed: int) -> None:
    """Set deterministic RNG seeds across stdlib, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_amp_dtype(device: torch.device) -> Tuple[Optional[torch.dtype], bool]:
    """Select native AMP precision for the device; only use bfloat16 on Ampere+ (cc >= 8.0)."""
    if device.type != "cuda":
        return None, False
    cc_major, _ = torch.cuda.get_device_capability(0)
    if cc_major >= 8 and torch.cuda.is_bf16_supported():
        return torch.bfloat16, False
    return torch.float16, True


def load_jpeg_bytes(folder: str, filenames: List[str]) -> List[bytes]:
    """Preload compressed JPEG byte buffers into host RAM to avoid disk I/O stalls."""
    buffers: List[bytes] = []
    for fn in filenames:
        with open(os.path.join(folder, fn), "rb") as f:
            buffers.append(f.read())
    return buffers


def apply_nightmare_aug(pil_img: Image.Image, rng: np.random.Generator) -> np.ndarray:
    """Apply realistic hard corruptions matching Task 4's four visual shift regimes."""
    if rng.random() < 0.50:
        pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.55:
        ang = float(rng.uniform(-35.0, 35.0))
        pil_img = pil_img.rotate(ang, resample=Image.BILINEAR, fillcolor=(0, 0, 0))
    if rng.random() < 0.16:
        pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.4, 1.3))))

    arr = np.asarray(pil_img, dtype=np.float32) * (1.0 / 255.0)

    # Neon channel gain / color tint (without destructive 1 - x color inversion)
    if rng.random() < 0.55:
        gains = rng.uniform(0.30, 2.20, size=(1, 1, 3)).astype(np.float32)
        gains /= float(gains.mean())
        arr = np.clip(arr * gains, 0.0, 1.0)

    # Exposure & contrast corruption
    r_exp = rng.random()
    if r_exp < 0.20:
        arr = np.power(np.clip(arr, 1e-4, 1.0), float(rng.uniform(1.8, 3.0))) * float(rng.uniform(0.32, 0.70))
    elif r_exp < 0.40:
        arr = np.clip(arr * float(rng.uniform(1.25, 2.10)) + float(rng.uniform(0.24, 0.48)), 0.0, 1.0)
    elif r_exp < 0.54:
        arr = np.clip(0.5 + (arr - 0.5) * float(rng.uniform(0.30, 0.65)) + float(rng.uniform(-0.15, 0.15)), 0.0, 1.0)

    if rng.random() < 0.40:
        arr = arr + rng.normal(0.0, float(rng.uniform(0.015, 0.075)), size=arr.shape).astype(np.float32)
    arr_u8 = np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8)

    if rng.random() < 0.20:
        buf = io.BytesIO()
        Image.fromarray(arr_u8).save(buf, format="JPEG", quality=int(rng.integers(28, 75)))
        buf.seek(0)
        arr_u8 = np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)

    return arr_u8


def interior_gw_stretch_np(arr_u8: np.ndarray) -> np.ndarray:
    """Normalize extreme color tints and exposures using central 60% interior statistics."""
    h, w, _ = arr_u8.shape
    h0, h1 = int(h * 0.2), int(h * 0.8)
    w0, w1 = int(w * 0.2), int(w * 0.8)
    b = arr_u8.astype(np.float32) * (1.0 / 255.0)
    patch = b[h0:h1:2, w0:w1:2, :]
    ch_m = np.clip(patch.mean(axis=(0, 1), keepdims=True), 0.025, 1.0)
    b_gw = np.clip(b * (ch_m.mean(axis=2, keepdims=True) / ch_m), 0.0, 1.0)
    p_sub = b_gw[h0:h1:3, w0:w1:3, :].ravel()
    lo, hi = np.quantile(p_sub, (0.02, 0.98))
    b_st = np.clip((b_gw - float(lo)) / max(float(hi - lo), 0.10), 0.0, 1.0)
    m_int = float(np.clip(b_st[h0:h1:3, w0:w1:3, :].mean(), 0.05, 0.95))
    gamma = float(np.clip(math.log(0.45) / math.log(m_int), 0.50, 1.80))
    return np.clip(np.power(b_st, gamma), 0.0, 1.0).astype(np.float32)


def interior_gw_stretch_torch(b: torch.Tensor) -> torch.Tensor:
    """Vectorized GPU Gray-World + Percentile Contrast Stretch matching interior_gw_stretch_np."""
    _, _, h, w = b.shape
    h0, h1 = int(h * 0.2), int(h * 0.8)
    w0, w1 = int(w * 0.2), int(w * 0.8)
    patch = b[:, :, h0:h1:2, w0:w1:2]
    ch_m = patch.mean(dim=(2, 3), keepdim=True).clamp(0.025, 1.0)
    b_gw = (b * (ch_m.mean(dim=1, keepdim=True) / ch_m)).clamp(0.0, 1.0)
    p_sub = b_gw[:, :, h0:h1:3, w0:w1:3].reshape(b.shape[0], -1)
    q = torch.tensor([0.02, 0.98], device=b.device, dtype=b.dtype)
    qs = torch.quantile(p_sub, q, dim=1)
    lo = qs[0].view(-1, 1, 1, 1)
    hi = qs[1].view(-1, 1, 1, 1)
    b_st = ((b_gw - lo) / (hi - lo).clamp(min=0.10)).clamp(0.0, 1.0)
    m_int = b_st[:, :, h0:h1:3, w0:w1:3].mean(dim=(1, 2, 3)).clamp(0.05, 0.95)
    gamma = (math.log(0.45) / torch.log(m_int)).clamp(0.50, 1.80).view(-1, 1, 1, 1)
    return b_st.pow(gamma).clamp(0.0, 1.0)


def make_deterministic_exthard(jpeg_bytes_list: List[bytes], indices: np.ndarray, seed: int = 123) -> List[np.ndarray]:
    """Generate deterministic ExtremeHard validation images for clean fold-level stress testing."""
    rng = np.random.default_rng(seed)
    out: List[np.ndarray] = []
    for k, idx in enumerate(indices):
        im = Image.open(io.BytesIO(jpeg_bytes_list[int(idx)])).convert("RGB")
        im = im.rotate(float(rng.uniform(-35.0, 35.0)), resample=Image.BILINEAR, fillcolor=(0, 0, 0))
        arr = np.asarray(im, dtype=np.float32) * (1.0 / 255.0)
        mode = k % 4
        if mode == 0:
            arr = np.power(np.clip(arr, 1e-4, 1.0), float(rng.uniform(1.8, 2.8))) * float(rng.uniform(0.35, 0.65))
        elif mode == 1:
            arr = np.clip(arr * float(rng.uniform(1.3, 2.0)) + float(rng.uniform(0.30, 0.50)), 0.0, 1.0)
        elif mode == 2:
            g = rng.uniform(0.35, 2.0, size=(1, 1, 3)).astype(np.float32)
            g /= float(g.max())
            arr = np.clip(np.power(arr, 0.85) * g * float(rng.uniform(0.8, 1.4)), 0.0, 1.0)
        else:
            arr = np.clip(0.5 + (arr - 0.5) * float(rng.uniform(0.4, 0.7)) + rng.uniform(-0.15, 0.15, (1, 1, 3)), 0.0, 1.0)
        arr = arr + rng.normal(0.0, float(rng.uniform(0.03, 0.07)), size=arr.shape).astype(np.float32)
        out.append(np.clip(arr * 255.0, 0.0, 255.0).astype(np.uint8))
    return out


class Task4TrainDataset(Dataset):
    """Training dataset with cached worker RNG and balanced clean/nightmare + optional Interior-GW views."""

    def __init__(
        self,
        jpeg_bytes: List[bytes],
        indices: np.ndarray,
        labels: np.ndarray,
        img_hw: Tuple[int, int],
        p_nightmare: float = 0.50,
        p_norm_view: float = 0.25,
        seed: int = 42,
    ) -> None:
        self.jpeg_bytes = jpeg_bytes
        self.indices = indices
        self.labels = labels
        self.img_hw = img_hw
        self.p_nightmare = p_nightmare
        self.p_norm_view = p_norm_view
        self.seed = seed
        self._rng: Optional[np.random.Generator] = None

    def _get_rng(self) -> np.random.Generator:
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            wid = info.id if info is not None else 0
            self._rng = np.random.default_rng(self.seed + wid * 7919)
        return self._rng

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        rng = self._get_rng()
        idx = int(self.indices[pos])
        pil = Image.open(io.BytesIO(self.jpeg_bytes[idx])).convert("RGB")
        target_wh = (self.img_hw[1], self.img_hw[0])
        if pil.size != target_wh:
            pil = pil.resize(target_wh, resample=Image.BILINEAR)

        if rng.random() < self.p_nightmare:
            arr_u8 = apply_nightmare_aug(pil, rng)
        else:
            if rng.random() < 0.50:
                pil = pil.transpose(Image.FLIP_LEFT_RIGHT)
            arr_u8 = np.asarray(pil, dtype=np.uint8)

        arr_f = interior_gw_stretch_np(arr_u8) if rng.random() < self.p_norm_view else (arr_u8.astype(np.float32) * (1.0 / 255.0))
        return torch.from_numpy(arr_f).permute(2, 0, 1).contiguous(), torch.tensor(int(self.labels[idx]), dtype=torch.long)


class Task4RawViewsDataset(Dataset):
    """Lightweight single-decode dataset yielding raw uint8 spatial images for GPU-side TTA/GW."""

    def __init__(
        self,
        source_items: List[bytes] | List[np.ndarray],
        indices: Optional[np.ndarray],
        img_hw: Tuple[int, int],
        need_rot: bool = True,
        is_bytes: bool = True,
    ) -> None:
        self.source_items = source_items
        self.indices = indices if indices is not None else np.arange(len(source_items))
        self.img_hw = img_hw
        self.need_rot = need_rot
        self.is_bytes = is_bytes

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, pos: int) -> torch.Tensor:
        item = self.source_items[int(self.indices[pos])]
        pil = Image.open(io.BytesIO(item)).convert("RGB") if self.is_bytes else Image.fromarray(item)
        target_wh = (self.img_hw[1], self.img_hw[0])
        if pil.size != target_wh:
            pil = pil.resize(target_wh, resample=Image.BILINEAR)

        frames = [np.asarray(pil, dtype=np.uint8)]
        if self.need_rot:
            frames.append(np.asarray(pil.rotate(18.0, resample=Image.BILINEAR, fillcolor=(0, 0, 0)), dtype=np.uint8))
            frames.append(np.asarray(pil.rotate(-18.0, resample=Image.BILINEAR, fillcolor=(0, 0, 0)), dtype=np.uint8))

        stacked = np.stack(frames, axis=0)  # (R, H, W, 3) uint8
        return torch.from_numpy(stacked).permute(0, 3, 1, 2).contiguous()


def apply_t4_freeze(model: nn.Module, spec: ModelSpec, freeze_vit_blocks: int = 6) -> Tuple[int, int]:
    """Freeze only lowest ViT blocks (never patch_embed/tokens or ConvNeXt stem) for T4 speedup + RGB adaptation."""
    total_params = sum(p.numel() for p in model.parameters())
    if spec.default_freeze_vit_blocks > 0 and freeze_vit_blocks > 0:
        for name, param in model.named_parameters():
            if name.startswith("blocks."):
                parts = name.split(".")
                if len(parts) > 1 and parts[1].isdigit() and int(parts[1]) < freeze_vit_blocks:
                    param.requires_grad_(False)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params, total_params


def build_model(
    spec: ModelSpec,
    pretrained: bool = True,
    freeze_vit_blocks: int = 6,
) -> Tuple[nn.Module, Tuple[float, ...], Tuple[float, ...], Tuple[int, int]]:
    """Instantiate timm model with exact aspect-ratio img_size, smart freeze, and resolved RGB stats."""
    kwargs: Dict[str, object] = {"pretrained": pretrained, "num_classes": 5}
    if spec.use_img_size_arg:
        kwargs["img_size"] = spec.img_hw
    model = timm.create_model(spec.timm_name, **kwargs)
    if spec.grad_ckpt and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
    param_counts = apply_t4_freeze(model, spec, freeze_vit_blocks=freeze_vit_blocks)
    cfg = timm.data.resolve_model_data_config(model)
    return model, tuple(cfg["mean"]), tuple(cfg["std"]), param_counts


def pick_batch_and_accum(spec: ModelSpec, device: torch.device, target_eff_bs: int = 48) -> Tuple[int, int, int]:
    """Select train batch size, grad accumulation, and eval batch size for the detected GPU VRAM."""
    if device.type != "cuda":
        return 8, 1, 16
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if vram_gb >= 36.0:
        bs, eval_bs = spec.batch_a100, 96
    elif vram_gb >= 21.0:
        bs, eval_bs = spec.batch_l4, 64
    else:
        bs, eval_bs = spec.batch_t4, spec.eval_batch_t4
    return bs, max(1, math.ceil(target_eff_bs / bs)), eval_bs


def make_param_groups(model: nn.Module, base_lr: float, bb_mult: float, wd: float = 2e-2) -> List[Dict]:
    """Separate classifier head vs backbone weights and exclude 1D norm/bias tensors from weight decay."""
    hd_d, hd_n, bb_d, bb_n = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_h = name.startswith("head") or name.startswith("fc") or ".head." in name
        no_w = p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower()
        target = (hd_n if no_w else hd_d) if is_h else (bb_n if no_w else bb_d)
        target.append(p)
    groups: List[Dict] = []
    for lst, lr_v, wd_v in [
        (hd_d, base_lr, wd),
        (hd_n, base_lr, 0.0),
        (bb_d, base_lr * bb_mult, wd),
        (bb_n, base_lr * bb_mult, 0.0),
    ]:
        if lst:
            groups.append({"params": lst, "lr": lr_v, "weight_decay": wd_v})
    return groups


def update_ema(ema_state: Dict[str, torch.Tensor], model: nn.Module, decay: float = 0.996) -> None:
    """Update exponential moving average shadow weights on device."""
    with torch.no_grad():
        for k, v in model.state_dict().items():
            if k in ema_state and v.dtype.is_floating_point:
                ema_state[k].mul_(decay).add_(v.detach(), alpha=1.0 - decay)
            elif k not in ema_state:
                ema_state[k] = v.detach().clone()


def probe_runtime_engine(
    model: nn.Module,
    spec: ModelSpec,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    use_scaler: bool,
) -> Tuple[bool, str]:
    """Run a 1-step dummy forward+backward+eval probe to choose a verified non-crashing CUDA/cuDNN mode."""
    h, w = spec.img_hw
    if device.type != "cuda":
        xb = torch.randn(2, 3, h, w, device=device)
        yb = torch.tensor([0, 1], device=device, dtype=torch.long)
        model.train()
        loss = F.cross_entropy(model(xb).float(), yb)
        loss.backward()
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad), "No trainable grads after backward"
        model.zero_grad(set_to_none=True)
        model.eval()
        with torch.no_grad():
            _ = model(xb)
        return False, f"{device.type}-fp32"

    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    candidates: List[Tuple[bool, bool, bool, str]] = []
    if cc_major >= 8:
        if spec.channels_last_ampere:
            candidates.append((True, True, True, f"sm{cc_major}{cc_minor}-nhwc-cudnn"))
        candidates.append((False, True, True, f"sm{cc_major}{cc_minor}-nchw-cudnn"))
        candidates.append((False, True, False, f"sm{cc_major}{cc_minor}-nchw-cudnn-safe"))
        candidates.append((False, False, False, f"sm{cc_major}{cc_minor}-nchw-nocudnn"))
    else:
        candidates.append((False, True, False, f"sm{cc_major}{cc_minor}-nchw-cudnn-safe"))
        candidates.append((False, False, False, f"sm{cc_major}{cc_minor}-nchw-nocudnn"))

    last_exc: Optional[Exception] = None
    for use_cl, use_cudnn, use_bench, tag in candidates:
        try:
            torch.backends.cudnn.enabled = use_cudnn
            torch.backends.cudnn.benchmark = use_bench
            model.to(memory_format=(torch.channels_last if use_cl else torch.contiguous_format))
            model.zero_grad(set_to_none=True)

            probe_scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, enabled=use_scaler)
            xb = torch.randn(2, 3, h, w, device=device)
            if use_cl:
                xb = xb.to(memory_format=torch.channels_last)
            yb = torch.tensor([0, 1], device=device, dtype=torch.long)

            model.train()
            with torch.amp.autocast(device.type, dtype=amp_dtype) if amp_dtype is not None else torch.enable_grad():
                logits = model(xb)
                assert torch.isfinite(logits.float()).all(), "Non-finite logits in preflight probe"
                loss = F.cross_entropy(logits.float(), yb)
            if use_scaler:
                probe_scaler.scale(loss).backward()
            else:
                loss.backward()
            has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
            assert has_grad, "Gradients missing in preflight probe"
            model.zero_grad(set_to_none=True)

            model.eval()
            xb_odd = torch.randn(3, 3, h, w, device=device)
            if use_cl:
                xb_odd = xb_odd.to(memory_format=torch.channels_last)
            with torch.no_grad(), (torch.amp.autocast(device.type, dtype=amp_dtype) if amp_dtype is not None else torch.no_grad()):
                _ = model(xb_odd)
            torch.cuda.synchronize()
            return use_cl, tag
        except Exception as exc:
            last_exc = exc
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

    raise RuntimeError(f"All runtime engine candidates failed for {spec.key}: {last_exc}") from last_exc


def _build_gpu_tta_batch(raw_f: torch.Tensor, views: List[str]) -> torch.Tensor:
    """Expand raw (B, R, 3, H, W) float32 [0,1] frames into (B, V, 3, H, W) on GPU."""
    x_orig = raw_f[:, 0]
    x_flip = torch.flip(x_orig, dims=[-1]) if any(v in ("flip", "gw_flip") for v in views) else None
    built: List[torch.Tensor] = []
    for v in views:
        if v == "orig":
            built.append(x_orig)
        elif v == "flip":
            assert x_flip is not None
            built.append(x_flip)
        elif v == "gw":
            built.append(interior_gw_stretch_torch(x_orig))
        elif v == "gw_flip":
            assert x_flip is not None
            built.append(interior_gw_stretch_torch(x_flip))
        elif v == "rot_p18_gw":
            built.append(interior_gw_stretch_torch(raw_f[:, 1]))
        elif v == "rot_m18_gw":
            built.append(interior_gw_stretch_torch(raw_f[:, 2]))
        else:
            raise ValueError(f"Unknown TTA view: {v}")
    return torch.stack(built, dim=1)


@torch.no_grad()
def predict_views(
    model: nn.Module,
    source_items: List[bytes] | List[np.ndarray],
    indices: Optional[np.ndarray],
    spec: ModelSpec,
    mean_t: torch.Tensor,
    std_t: torch.Tensor,
    views: List[str],
    eval_bs: int,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    use_channels_last: bool = False,
    is_bytes: bool = True,
    num_workers: int = 2,
) -> Dict[str, np.ndarray]:
    """Run single-decode uint8 transfer + vectorized GPU TTA inference."""
    model.eval()
    n = len(indices) if indices is not None else len(source_items)
    v_cnt = len(views)
    need_rot = any(v in ("rot_p18_gw", "rot_m18_gw") for v in views)
    dl_bs = max(16, eval_bs) if v_cnt == 1 else max(8, min(32, eval_bs))
    ds = Task4RawViewsDataset(source_items, indices, spec.img_hw, need_rot=need_rot, is_bytes=is_bytes)
    dl = DataLoader(
        ds,
        batch_size=dl_bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    all_probs: List[np.ndarray] = []
    for batch_u8 in dl:
        bsz = batch_u8.shape[0]
        raw_f = batch_u8.to(device, non_blocking=True).float() * (1.0 / 255.0)
        views_t = _build_gpu_tta_batch(raw_f, views)
        flat = views_t.view(bsz * v_cnt, 3, views_t.shape[3], views_t.shape[4])
        out_chunks: List[np.ndarray] = []
        for s in range(0, len(flat), eval_bs):
            xb = flat[s : s + eval_bs]
            if use_channels_last and device.type == "cuda":
                xb = xb.to(memory_format=torch.channels_last)
            xb = (xb - mean_t) / std_t
            with torch.amp.autocast(device.type, dtype=amp_dtype) if amp_dtype is not None else torch.no_grad():
                logits = model(xb)
            out_chunks.append(F.softmax(logits.float(), dim=1).cpu().numpy())
        all_probs.append(np.concatenate(out_chunks, axis=0).reshape(bsz, v_cnt, 5))

    merged = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, v_cnt, 5), dtype=np.float32)
    assert len(merged) == n
    return {v: merged[:, i, :].copy() for i, v in enumerate(views)}


def combine_tta_views(vdict: Dict[str, np.ndarray], mode: str) -> np.ndarray:
    """Combine view probabilities in geometric log-space."""
    if mode == "orig2":
        keys, ws = ["orig", "flip"], [0.55, 0.45]
    elif mode == "weighted6":
        keys = ["orig", "flip", "gw", "gw_flip", "rot_p18_gw", "rot_m18_gw"]
        ws = [0.28, 0.20, 0.20, 0.12, 0.10, 0.10]
    else:
        raise ValueError(f"Unknown combine mode: {mode}")
    lp = sum(w * np.log(np.clip(vdict[k], 1e-7, 1.0)) for k, w in zip(keys, ws))
    lp -= lp.max(axis=1, keepdims=True)
    ep = np.exp(lp)
    return (ep / ep.sum(axis=1, keepdims=True)).astype(np.float32)


def save_csv_preds(
    out_path: str,
    df_meta: pd.DataFrame,
    probs: np.ndarray,
    folds_col: Optional[np.ndarray] = None,
) -> None:
    """Write predictions matching project/task4/predictions/model-level CSV schema."""
    out = pd.DataFrame({"id": df_meta["id"].values, "file_name": df_meta["file_name"].values})
    if "label" in df_meta.columns and df_meta["label"].notna().all():
        out["label"] = df_meta["label"].astype(int).values
    if folds_col is not None:
        out["fold"] = folds_col.astype(int)
    for c in range(5):
        out[f"prob_{c}"] = np.round(probs[:, c].astype(np.float64), 7)
    out.to_csv(out_path, index=False)


def atomic_savez_compressed(path: str, **arrays: np.ndarray) -> None:
    """Atomically write compressed .npz file so disconnects never leave partial corrupt checkpoints."""
    tmp_path = f"{path}.tmp.{os.getpid()}.npz"
    np.savez_compressed(tmp_path, **arrays)
    os.replace(tmp_path, path)


def load_fold_npz(path: str) -> Optional[Dict[str, np.ndarray]]:
    """Safely load completed fold checkpoint if present and intact."""
    req = ("va_orig2", "va_w6", "ext_w6", "te_orig2", "te_w6")
    if not os.path.isfile(path):
        return None
    try:
        with np.load(path) as z:
            if not all(k in z for k in req):
                return None
            return {k: z[k].copy() for k in req}
    except Exception:
        return None


def resolve_epoch_ckpt_paths(cache_dir: str, model_key: str, k: int) -> Tuple[str, str]:
    """Return primary (Drive/out_dir) and local fallback checkpoint paths for mid-fold resume."""
    drive_ckpt = os.path.join(cache_dir, f"fold{k}_ckpt.pt")
    if os.path.isdir("/content"):
        local_dir = os.path.join("/content/t4_hard_ckpt_tmp", f"folds_{model_key}")
        os.makedirs(local_dir, exist_ok=True)
        local_ckpt = os.path.join(local_dir, f"fold{k}_ckpt.pt")
    else:
        local_ckpt = drive_ckpt
    return drive_ckpt, local_ckpt


def save_epoch_ckpt(
    cache_dir: str,
    spec: ModelSpec,
    k: int,
    ep: int,
    epochs: int,
    freeze_vit_blocks: int,
    model: nn.Module,
    ema_state: Dict[str, torch.Tensor],
    opt: torch.optim.Optimizer,
    sched: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
) -> str:
    """Save only trainable/buffer tensors + optimizer/scheduler state atomically for zero-loss resume."""
    drive_ckpt, local_ckpt = resolve_epoch_ckpt_paths(cache_dir, spec.key, k)
    trainable_keys = {n for n, p in model.named_parameters() if p.requires_grad}
    buf_keys = {n for n, _ in model.named_buffers()}
    keep_keys = trainable_keys | buf_keys

    state_dict_cpu = {sk: sv.detach().cpu() for sk, sv in model.state_dict().items() if sk in keep_keys}
    ema_cpu = {ek: ev.detach().cpu() for ek, ev in ema_state.items() if ek in keep_keys}
    payload = {
        "spec_key": spec.key,
        "img_hw": tuple(spec.img_hw),
        "epochs": int(epochs),
        "freeze_vit_blocks": int(freeze_vit_blocks),
        "epoch": int(ep),
        "trainable_model": state_dict_cpu,
        "ema_state": ema_cpu,
        "opt": opt.state_dict(),
        "sched": sched.state_dict(),
        "scaler": scaler.state_dict(),
    }

    target = drive_ckpt
    try:
        if drive_ckpt != local_ckpt and shutil.disk_usage(cache_dir).free < 4_500_000_000:
            target = local_ckpt
    except OSError:
        target = local_ckpt

    tmp_path = f"{target}.tmp.{os.getpid()}"
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, target)
    except OSError:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        if target != local_ckpt:
            tmp_fb = f"{local_ckpt}.tmp.{os.getpid()}"
            torch.save(payload, tmp_fb)
            os.replace(tmp_fb, local_ckpt)
            target = local_ckpt
        else:
            raise
    return target


def load_epoch_ckpt(
    cache_dir: str,
    spec: ModelSpec,
    k: int,
    epochs: int,
    freeze_vit_blocks: int,
) -> Optional[Dict]:
    """Load the newest valid mid-fold epoch checkpoint matching current architecture config."""
    drive_ckpt, local_ckpt = resolve_epoch_ckpt_paths(cache_dir, spec.key, k)
    cands = [drive_ckpt] if drive_ckpt == local_ckpt else [drive_ckpt, local_ckpt]
    best_payload = None
    for cp in cands:
        if not os.path.isfile(cp):
            continue
        try:
            pl = torch.load(cp, map_location="cpu", weights_only=False)
            if (
                pl.get("spec_key") == spec.key
                and tuple(pl.get("img_hw", ())) == tuple(spec.img_hw)
                and int(pl.get("epochs", -1)) == int(epochs)
                and int(pl.get("freeze_vit_blocks", -1)) == int(freeze_vit_blocks)
            ):
                if best_payload is None or int(pl.get("epoch", -1)) > int(best_payload.get("epoch", -1)):
                    best_payload = pl
        except Exception:
            continue
    return best_payload


def remove_epoch_ckpt(cache_dir: str, model_key: str, k: int) -> None:
    """Delete temporary mid-fold epoch checkpoints once final fold{k}.npz is saved."""
    drive_ckpt, local_ckpt = resolve_epoch_ckpt_paths(cache_dir, model_key, k)
    for cp in {drive_ckpt, local_ckpt}:
        if os.path.isfile(cp):
            os.remove(cp)


def train_single_fold(
    k: int,
    spec: ModelSpec,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    tr_bytes: List[bytes],
    te_bytes: List[bytes],
    y: np.ndarray,
    bs: int,
    accum: int,
    eval_bs: int,
    epochs: int,
    eff_workers: int,
    views: List[str],
    cache_dir: str,
    resume: bool,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    use_scaler: bool,
    pretrained: bool,
    freeze_vit_blocks: int,
    p_nightmare: float,
    p_norm_view: float,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Train one SGKF fold with epoch checkpoints, EMA tracking, and GPU 6-view OOF/ExtHard/Test TTA."""
    t0 = time.time()
    seed_everything(seed + k * 101)
    model, mean_tup, std_tup, (n_train, n_tot) = build_model(
        spec,
        pretrained=pretrained,
        freeze_vit_blocks=freeze_vit_blocks,
    )
    model = model.to(device)
    use_cl, runtime_tag = probe_runtime_engine(model, spec, device, amp_dtype, use_scaler)
    seed_everything(seed + k * 101)

    mean_t = torch.tensor(mean_tup, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std_tup, device=device, dtype=torch.float32).view(1, 3, 1, 1)

    pi_tr = np.bincount(y[tr_idx], minlength=5) / max(1, len(tr_idx))
    logit_adj = torch.tensor(0.20 * np.log(np.clip(pi_tr, 1e-4, 1.0)), device=device, dtype=torch.float32)

    tr_ds = Task4TrainDataset(
        tr_bytes,
        tr_idx,
        y,
        spec.img_hw,
        p_nightmare=p_nightmare,
        p_norm_view=p_norm_view,
        seed=seed + k * 1000,
    )
    tr_dl = DataLoader(
        tr_ds,
        batch_size=bs,
        shuffle=True,
        drop_last=len(tr_ds) >= bs,
        num_workers=eff_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(eff_workers > 0),
    )

    opt = torch.optim.AdamW(make_param_groups(model, spec.base_lr, spec.backbone_lr_mult))
    steps_per_ep = max(1, math.ceil(len(tr_dl) / accum))
    total_steps = steps_per_ep * epochs
    warmup_steps = max(1, int(0.08 * total_steps))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    ema_state: Dict[str, torch.Tensor] = {}

    start_ep = 0
    ckpt = load_epoch_ckpt(cache_dir, spec, k, epochs, freeze_vit_blocks) if resume else None
    if ckpt is not None:
        model.load_state_dict(ckpt["trainable_model"], strict=False)
        for ek, ev in ckpt["ema_state"].items():
            ema_state[ek] = ev.to(device=device)
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        if ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        start_ep = int(ckpt["epoch"]) + 1
        print(
            f"[{spec.key}] Fold {k} resumed mid-fold at Ep {start_ep}/{epochs} | "
            f"engine={runtime_tag} | amp={amp_dtype} | trainable={n_train/1e6:.1f}M/{n_tot/1e6:.1f}M",
            flush=True,
        )
    else:
        print(
            f"[{spec.key}] Fold {k} started | engine={runtime_tag} | amp={amp_dtype} | "
            f"trainable={n_train/1e6:.1f}M/{n_tot/1e6:.1f}M",
            flush=True,
        )

    for ep in range(start_ep, epochs):
        seed_everything(seed + k * 101 + ep * 17)
        model.train()
        opt.zero_grad(set_to_none=True)
        ep_loss, n_seen = 0.0, 0
        for step_i, (xb, yb) in enumerate(tr_dl):
            xb = xb.to(device, non_blocking=True)
            if use_cl and device.type == "cuda":
                xb = xb.to(memory_format=torch.channels_last)
            yb = yb.to(device, non_blocking=True)
            xb = (xb - mean_t) / std_t

            with torch.amp.autocast(device.type, dtype=amp_dtype) if amp_dtype is not None else torch.enable_grad():
                logits = model(xb)
                loss = F.cross_entropy(logits.float() + logit_adj, yb, label_smoothing=0.03)
                loss_scaled = loss / accum

            if use_scaler:
                scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            if ((step_i + 1) % accum == 0) or ((step_i + 1) == len(tr_dl)):
                if use_scaler:
                    scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                if use_scaler:
                    prev_scale = scaler.get_scale()
                    scaler.step(opt)
                    scaler.update()
                    if scaler.get_scale() >= prev_scale:
                        sched.step()
                else:
                    opt.step()
                    sched.step()
                opt.zero_grad(set_to_none=True)
                update_ema(ema_state, model, decay=0.996 if ep >= 1 else 0.98)

            ep_loss += float(loss.item()) * len(yb)
            n_seen += len(yb)

        val_q = predict_views(
            model, tr_bytes, va_idx, spec, mean_t, std_t, ["orig"], eval_bs, device, amp_dtype, use_cl, True, eff_workers
        )["orig"]
        vf1 = f1_score(y[va_idx], val_q.argmax(1), average="macro")
        ckpt_saved = save_epoch_ckpt(cache_dir, spec, k, ep, epochs, freeze_vit_blocks, model, ema_state, opt, sched, scaler)
        print(
            f"[{spec.key}] Fold {k} Ep {ep+1}/{epochs} loss={ep_loss/max(1, n_seen):.4f} "
            f"ValCleanF1={vf1:.5f} | saved ckpt ({os.path.basename(ckpt_saved)}) ({time.time()-t0:.1f}s)",
            flush=True,
        )

    model.load_state_dict(ema_state, strict=False)
    t_ev = time.time()
    va_views = predict_views(
        model, tr_bytes, va_idx, spec, mean_t, std_t, views, eval_bs, device, amp_dtype, use_cl, True, eff_workers
    )
    ext_imgs = make_deterministic_exthard(tr_bytes, va_idx, seed=123 + k)
    ext_views = predict_views(
        model, ext_imgs, None, spec, mean_t, std_t, views, eval_bs, device, amp_dtype, use_cl, False, eff_workers
    )
    te_views = predict_views(
        model, te_bytes, None, spec, mean_t, std_t, views, eval_bs, device, amp_dtype, use_cl, True, eff_workers
    )

    fold_out = {
        "va_orig2": combine_tta_views(va_views, "orig2"),
        "va_w6": combine_tta_views(va_views, "weighted6"),
        "ext_w6": combine_tta_views(ext_views, "weighted6"),
        "te_orig2": combine_tta_views(te_views, "orig2"),
        "te_w6": combine_tta_views(te_views, "weighted6"),
    }
    f_val = f1_score(y[va_idx], fold_out["va_w6"].argmax(1), average="macro")
    f_ext = f1_score(y[va_idx], fold_out["ext_w6"].argmax(1), average="macro")
    print(
        f"[{spec.key}] DONE Fold {k} in {time.time()-t0:.1f}s (eval={time.time()-t_ev:.1f}s) | "
        f"Val6TTA F1={f_val:.5f} | ExtHard F1={f_ext:.5f}",
        flush=True,
    )

    del model, opt, sched, scaler, ema_state, tr_dl, tr_ds, ext_imgs, va_views, ext_views, te_views
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return fold_out


def package_model_zip(out_dir: str, model_key: str) -> str:
    """Create per-model downloadable ZIP archive in out_dir and /content if present."""
    model_files = [
        f"task4-{model_key}-hard6-oof.csv",
        f"task4-{model_key}-hard6-test-predictions.csv",
        f"task4-{model_key}-orig2-oof.csv",
        f"task4-{model_key}-orig2-test-predictions.csv",
        f"task4-{model_key}-sgkf-hard-bundle.npz",
    ]
    mzip_out = os.path.join(out_dir, f"t4_hard_{model_key}.zip")
    with zipfile.ZipFile(mzip_out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fn in model_files:
            fp = os.path.join(out_dir, fn)
            if os.path.isfile(fp):
                zf.write(fp, fn)
    if os.path.isdir("/content") and os.path.abspath(mzip_out) != f"/content/t4_hard_{model_key}.zip":
        shutil.copy2(mzip_out, f"/content/t4_hard_{model_key}.zip")
    return mzip_out


def train_one_model(
    spec: ModelSpec,
    tr_bytes: List[bytes],
    te_bytes: List[bytes],
    y: np.ndarray,
    folds: np.ndarray,
    tr_df: pd.DataFrame,
    te_df: pd.DataFrame,
    out_dir: str,
    device: torch.device,
    pretrained: bool = True,
    epochs_override: Optional[int] = None,
    max_folds: int = 5,
    num_workers: int = 2,
    resume: bool = True,
    freeze_vit_blocks: Optional[int] = None,
    p_nightmare: float = 0.50,
    p_norm_view: float = 0.25,
    seed: int = 42,
) -> None:
    """Train all SGKF folds for one backbone with automatic fold & epoch checkpoint resume."""
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = os.path.join(out_dir, f"folds_{spec.key}")
    os.makedirs(cache_dir, exist_ok=True)

    amp_dtype, use_scaler = select_amp_dtype(device)
    bs, accum, eval_bs = pick_batch_and_accum(spec, device)
    epochs = epochs_override if epochs_override is not None else spec.epochs
    eff_freeze = spec.default_freeze_vit_blocks if freeze_vit_blocks is None else freeze_vit_blocks
    eff_workers = min(num_workers, max(1, os.cpu_count() or num_workers)) if num_workers > 0 else 0
    views = ["orig", "flip", "gw", "gw_flip", "rot_p18_gw", "rot_m18_gw"]
    print(
        f"[{spec.key}] Config: img_hw={spec.img_hw} | bs={bs} (accum={accum}, eval_bs={eval_bs}) | "
        f"epochs={epochs} | amp={amp_dtype} | freeze_vit_blocks={eff_freeze} | workers={eff_workers}"
    )

    n_tr, n_te = len(tr_df), len(te_df)
    oof_orig2 = np.zeros((n_tr, 5), dtype=np.float32)
    oof_w6 = np.zeros((n_tr, 5), dtype=np.float32)
    ext_w6 = np.zeros((n_tr, 5), dtype=np.float32)
    te_orig2 = np.zeros((n_te, 5), dtype=np.float32)
    te_w6 = np.zeros((n_te, 5), dtype=np.float32)
    folds_done = 0

    for k in range(min(5, max_folds)):
        fold_npz = os.path.join(cache_dir, f"fold{k}.npz")
        va_idx = np.where(folds == k)[0]
        tr_idx = np.where(folds != k)[0]

        cached = load_fold_npz(fold_npz) if resume else None
        if cached is not None:
            remove_epoch_ckpt(cache_dir, spec.key, k)
            oof_orig2[va_idx] = cached["va_orig2"]
            oof_w6[va_idx] = cached["va_w6"]
            ext_w6[va_idx] = cached["ext_w6"]
            te_orig2 += cached["te_orig2"]
            te_w6 += cached["te_w6"]
            folds_done += 1
            f_val = f1_score(y[va_idx], oof_w6[va_idx].argmax(1), average="macro")
            f_ext = f1_score(y[va_idx], ext_w6[va_idx].argmax(1), average="macro")
            print(f"[{spec.key}] Fold {k} resumed -> ValF1={f_val:.5f} | ExtHardF1={f_ext:.5f}")
            continue

        fold_res = train_single_fold(
            k=k,
            spec=spec,
            tr_idx=tr_idx,
            va_idx=va_idx,
            tr_bytes=tr_bytes,
            te_bytes=te_bytes,
            y=y,
            bs=bs,
            accum=accum,
            eval_bs=eval_bs,
            epochs=epochs,
            eff_workers=eff_workers,
            views=views,
            cache_dir=cache_dir,
            resume=resume,
            device=device,
            amp_dtype=amp_dtype,
            use_scaler=use_scaler,
            pretrained=pretrained,
            freeze_vit_blocks=eff_freeze,
            p_nightmare=p_nightmare,
            p_norm_view=p_norm_view,
            seed=seed,
        )
        atomic_savez_compressed(fold_npz, **fold_res)
        remove_epoch_ckpt(cache_dir, spec.key, k)
        oof_orig2[va_idx] = fold_res["va_orig2"]
        oof_w6[va_idx] = fold_res["va_w6"]
        ext_w6[va_idx] = fold_res["ext_w6"]
        te_orig2 += fold_res["te_orig2"]
        te_w6 += fold_res["te_w6"]
        folds_done += 1

    if folds_done == 5:
        te_orig2 /= 5.0
        te_w6 /= 5.0
        f_all = f1_score(y, oof_w6.argmax(1), average="macro")
        f_o2 = f1_score(y, oof_orig2.argmax(1), average="macro")
        f_ext = f1_score(y, ext_w6.argmax(1), average="macro")
        ll_all = log_loss(y, np.clip(oof_w6, 1e-7, 1.0), labels=[0, 1, 2, 3, 4])
        print(
            f"\n=== [{spec.key}] 5-FOLD COMPLETE | OOF_W6 F1={f_all:.5f} LL={ll_all:.5f} | "
            f"Orig2 F1={f_o2:.5f} | ExtHard F1={f_ext:.5f} | Combo={0.5*(f_all+f_ext):.5f} ===\n",
            flush=True,
        )
        save_csv_preds(os.path.join(out_dir, f"task4-{spec.key}-hard6-oof.csv"), tr_df, oof_w6, folds)
        save_csv_preds(os.path.join(out_dir, f"task4-{spec.key}-hard6-test-predictions.csv"), te_df, te_w6)
        save_csv_preds(os.path.join(out_dir, f"task4-{spec.key}-orig2-oof.csv"), tr_df, oof_orig2, folds)
        save_csv_preds(os.path.join(out_dir, f"task4-{spec.key}-orig2-test-predictions.csv"), te_df, te_orig2)
        atomic_savez_compressed(
            os.path.join(out_dir, f"task4-{spec.key}-sgkf-hard-bundle.npz"),
            folds=folds,
            oof_orig2=oof_orig2,
            oof_w6=oof_w6,
            ext_w6=ext_w6,
            te_orig2=te_orig2,
            te_w6=te_w6,
        )
        mzip = package_model_zip(out_dir, spec.key)
        print(f"[READY] Packaged 5-fold {spec.key} bundle -> {mzip}", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="5-Fold SGKF Hardened Task 4 Trainer")
    ap.add_argument("--data-dir", required=True, help="Directory with train.csv, test.csv, folds_sgkf.npy, train/, test/")
    ap.add_argument("--out-dir", required=True, help="Output directory for fold checkpoints and CSV bundles")
    ap.add_argument("--models", nargs="+", default=["dinov3l", "eva02l"], choices=list(MODEL_SPECS.keys()))
    ap.add_argument("--epochs", type=int, default=None, help="Override default epoch count")
    ap.add_argument("--max-folds", type=int, default=5, help="Train up to N folds (5 for full run)")
    ap.add_argument("--freeze-vit-blocks", type=int, default=None, help="Override frozen ViT lower block count")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-pretrained", action="store_true", help="Used for offline smoke testing")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="Force CPU device (e.g., smoke tests)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Device: {device} | Models: {args.models}")

    tr_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    te_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    y = tr_df["label"].astype(int).to_numpy() if "label" in tr_df.columns else tr_df["class"].map(CLASS_MAP).astype(int).to_numpy()
    folds = np.load(os.path.join(args.data_dir, "folds_sgkf.npy")).astype(int)
    assert len(tr_df) == len(folds) == len(y), "Manifest and folds_sgkf length mismatch"

    print("Preloading compressed JPEG bytes into RAM...")
    t_load = time.time()
    tr_bytes = load_jpeg_bytes(os.path.join(args.data_dir, "train"), tr_df["file_name"].tolist())
    te_bytes = load_jpeg_bytes(os.path.join(args.data_dir, "test"), te_df["file_name"].tolist())
    print(f"Preloaded train={len(tr_bytes):,}, test={len(te_bytes):,} images in {time.time()-t_load:.1f}s")

    for mkey in args.models:
        train_one_model(
            spec=MODEL_SPECS[mkey],
            tr_bytes=tr_bytes,
            te_bytes=te_bytes,
            y=y,
            folds=folds,
            tr_df=tr_df,
            te_df=te_df,
            out_dir=args.out_dir,
            device=device,
            pretrained=not args.no_pretrained,
            epochs_override=args.epochs,
            max_folds=args.max_folds,
            num_workers=args.num_workers,
            resume=not args.no_resume,
            freeze_vit_blocks=args.freeze_vit_blocks,
        )


if __name__ == "__main__":
    main()

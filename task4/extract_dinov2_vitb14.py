"""Extract DINOv2 ViT-Base/14 (768-d) features for Task 4 train and test images using Apple MPS GPU."""
from __future__ import annotations
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PRED_DIR = ROOT / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 64
NUM_WORKERS = 0

transform = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ImageFolderDataset(Dataset):
    def __init__(self, ids: list[str], folder: Path):
        self.ids = ids
        self.folder = folder

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        img_path = self.folder / f"{img_id}.jpg"
        with Image.open(img_path) as img:
            t = transform(img.convert("RGB"))
        return t, img_id

def extract_features(model: torch.nn.Module, ids: list[str], folder: Path, out_path: Path):
    if out_path.exists():
        print(f"File {out_path} already exists, skipping extraction.")
        return
    print(f"Extracting features for {len(ids)} images from {folder} to {out_path}...")
    dataset = ImageFolderDataset(ids, folder)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )
    
    feats = []
    ret_ids = []
    t0 = time.time()
    with torch.no_grad():
        for i, (imgs, batch_ids) in enumerate(loader):
            imgs = imgs.to(DEVICE)
            out = model(imgs) # [B, 768]
            feats.append(out.cpu().numpy().astype(np.float32))
            ret_ids.extend(batch_ids)
            if (i + 1) % 25 == 0 or (i + 1) == len(loader):
                elapsed = time.time() - t0
                speed = len(ret_ids) / elapsed
                remaining = (len(ids) - len(ret_ids)) / speed
                print(f"  Processed {len(ret_ids)}/{len(ids)} images ({speed:.1f} img/s, ETA: {remaining/60:.1f}m)...", flush=True)

    X = np.vstack(feats)
    np.savez_compressed(out_path, X=X, ids=np.array(ret_ids))
    print(f"Saved {out_path} with shape {X.shape} in {time.time() - t0:.1f}s")

def main():
    print(f"Using device: {DEVICE}")
    print("Loading dinov2_vitb14 model...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(DEVICE)
    model.eval()

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    train_ids = train_df["id"].tolist()
    test_ids = test_df["id"].tolist()

    # Extract Train features
    train_out = PRED_DIR / "dinov2_vitb14_train_features.npz"
    extract_features(model, train_ids, DATA_DIR / "train", train_out)

    # Extract Test features
    test_out = PRED_DIR / "dinov2_vitb14_test_features.npz"
    extract_features(model, test_ids, DATA_DIR / "test", test_out)

    print("DINOv2 ViT-B/14 feature extraction complete!")

if __name__ == "__main__":
    main()

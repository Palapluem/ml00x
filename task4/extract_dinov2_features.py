"""
High-Speed DINOv2 ViT-S/14 Feature Extractor for Task 4
Leverages Apple Silicon GPU (MPS) and PyTorch DataLoader to extract 384-dim
foundation representations for 31,546 train images and 25,889 test images.
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROOT = Path("/Users/wisitsuwannao/Documents/Workspace/ml00x/task4")
DATA_DIR = ROOT / "data"
PRED_DIR = ROOT / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Extraction hardware accelerator: {device}")

# ViT-S/14 standard input transform (224x224, ImageNet normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class Task4Dataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.ids = df["id"].values if "id" in df.columns else df["image_id"].values
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = self.img_dir / f"{img_id}.jpg"
        if not img_path.exists():
            # Fallback to zero image if missing
            return torch.zeros(3, 224, 224), img_id
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), img_id

def extract_set(model, dataloader, total_count, desc):
    features = []
    ids_list = []
    t0 = time.time()
    processed = 0
    with torch.no_grad():
        for i, (batch, ids) in enumerate(dataloader):
            batch = batch.to(device, non_blocking=True)
            feats = model(batch) # [B, 384]
            features.append(feats.cpu().numpy())
            ids_list.extend(ids)
            processed += len(batch)
            if (i + 1) % 25 == 0 or processed == total_count:
                elapsed = time.time() - t0
                speed = processed / elapsed if elapsed > 0 else 0
                eta = (total_count - processed) / speed if speed > 0 else 0
                print(f"[{desc}] {processed}/{total_count} ({processed/total_count*100:5.1f}%) | Speed: {speed:5.1f} img/s | ETA: {eta:4.1f}s", flush=True)
    return np.vstack(features), ids_list

def main():
    print("1. Loading Meta DINOv2 ViT-S/14 Foundation Model...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
    model.eval()
    print("Model initialized.")

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    # Check if train features already exist
    train_feat_path = PRED_DIR / "dinov2_train_features.npz"
    test_feat_path = PRED_DIR / "dinov2_test_features.npz"

    train_ds = Task4Dataset(train_df, DATA_DIR / "train", transform)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=4, prefetch_factor=2)

    test_ds = Task4Dataset(test_df, DATA_DIR / "test", transform)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4, prefetch_factor=2)

    print("\n2. Extracting DINOv2 features for Training Set...")
    X_train_dino, train_ids = extract_set(model, train_loader, len(train_df), "Train")
    np.savez_compressed(train_feat_path, X=X_train_dino.astype(np.float32), ids=train_ids)
    print(f"Saved Train features: {X_train_dino.shape} to {train_feat_path}")

    print("\n3. Extracting DINOv2 features for Test Set...")
    X_test_dino, test_ids = extract_set(model, test_loader, len(test_df), "Test")
    np.savez_compressed(test_feat_path, X=X_test_dino.astype(np.float32), ids=test_ids)
    print(f"Saved Test features: {X_test_dino.shape} to {test_feat_path}")

    print("\nDINOv2 Feature Extraction Complete! Ready for Model Training.")

if __name__ == "__main__":
    main()


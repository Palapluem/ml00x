import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

def main():
    task_dir = Path(__file__).resolve().parent
    pred_dir = task_dir / "predictions"
    submit_dir = task_dir / "submit"
    assets_dir = task_dir / "assets"
    data_dir = task_dir / "data"

    submit_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)

    print("=== Training SOTA DINOv2 + Multi-Modal Model for Task 4 ===")

    # 1. Check feature files
    dino_train_path = pred_dir / "dinov2_train_features.npz"
    dino_test_path = pred_dir / "dinov2_test_features.npz"
    resnet_train_path = pred_dir / "train-features.npz"
    resnet_test_path = pred_dir / "test-features.npz"

    if not dino_train_path.exists() or not dino_test_path.exists():
        print(f"Waiting for DINOv2 features... train exists: {dino_train_path.exists()}, test exists: {dino_test_path.exists()}")
        sys.exit(1)

    print("Loading feature representations...")
    dino_train_file = np.load(dino_train_path)
    dino_test_file = np.load(dino_test_path)
    dino_train = dino_train_file["X"] if "X" in dino_train_file else dino_train_file["features"]
    dino_test = dino_test_file["X"] if "X" in dino_test_file else dino_test_file["features"]
    print(f"DINOv2 features shape: Train {dino_train.shape}, Test {dino_test.shape}")

    # Optionally load ResNet+Color features if available
    has_resnet = resnet_train_path.exists() and resnet_test_path.exists()
    if has_resnet:
        resnet_train = np.load(resnet_train_path)["X"]
        resnet_test = np.load(resnet_test_path)["X"]
        print(f"Teammate ResNet features shape: Train {resnet_train.shape}, Test {resnet_test.shape}")
        
        # Combine DINOv2 + ResNet features
        scaler_res = StandardScaler()
        resnet_train_scaled = scaler_res.fit_transform(resnet_train)
        resnet_test_scaled = scaler_res.transform(resnet_test)
        
        X_train = np.hstack([dino_train, resnet_train_scaled])
        X_test = np.hstack([dino_test, resnet_test_scaled])
        print(f"Combined Multi-Modal feature shape: Train {X_train.shape}, Test {X_test.shape}")
    else:
        X_train = dino_train
        X_test = dino_test

    # 2. Load ground truth labels
    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    # Normalize labels: 7-11k -> 7-11K
    label_map_fix = {"7-11k": "7-11K"}
    raw_labels = train_df["class"].replace(label_map_fix).values

    classes = np.sort(np.unique(raw_labels))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for i, c in enumerate(classes)}
    y_train = np.array([class_to_idx[c] for c in raw_labels])

    print(f"Classes ({len(classes)}): {classes}")
    for c in classes:
        print(f"  {c}: {np.sum(raw_labels == c):,} ({(np.sum(raw_labels == c)/len(raw_labels))*100:.1f}%)")

    # 3. Stratified 5-Fold Cross Validation
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_preds_lr = np.zeros((len(X_train), len(classes)), dtype=np.float32)
    test_preds_lr = np.zeros((len(X_test), len(classes)), dtype=np.float32)

    oof_preds_lgb = np.zeros((len(X_train), len(classes)), dtype=np.float32)
    test_preds_lgb = np.zeros((len(X_test), len(classes)), dtype=np.float32)

    print("\nStarting 5-Fold Stratified Model Training...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"\n--- Fold {fold}/{n_splits} ---")
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_va, y_va = X_train[val_idx], y_train[val_idx]

        # Model 1: Logistic Regression with Scaled DINO Features
        lr = LogisticRegression(C=1.5, max_iter=500, solver="lbfgs", random_state=42 + fold)
        lr.fit(X_tr, y_tr)
        val_prob_lr = lr.predict_proba(X_va)
        oof_preds_lr[val_idx] = val_prob_lr
        test_preds_lr += lr.predict_proba(X_test) / n_splits
        f1_lr = f1_score(y_va, np.argmax(val_prob_lr, axis=1), average="macro")
        print(f"  Fold {fold} Logistic Regression Macro F1: {f1_lr:.4f}")

        # Model 2: LightGBM Classifier
        lgb_params = {
            "objective": "multiclass",
            "num_class": len(classes),
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "verbose": -1,
            "random_state": 42 + fold,
            "n_jobs": -1
        }
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_va, label=y_va, reference=train_data)

        gbm = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=800,
            valid_sets=[train_data, val_data],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)]
        )
        val_prob_lgb = gbm.predict(X_va, num_iteration=gbm.best_iteration)
        oof_preds_lgb[val_idx] = val_prob_lgb
        test_preds_lgb += gbm.predict(X_test, num_iteration=gbm.best_iteration) / n_splits
        f1_lgb = f1_score(y_va, np.argmax(val_prob_lgb, axis=1), average="macro")
        print(f"  Fold {fold} LightGBM Macro F1: {f1_lgb:.4f}")

    # Evaluation on full Out-of-Fold
    f1_lr_total = f1_score(y_train, np.argmax(oof_preds_lr, axis=1), average="macro")
    f1_lgb_total = f1_score(y_train, np.argmax(oof_preds_lgb, axis=1), average="macro")
    print("\n================ OOF RESULTS ================")
    print(f"Logistic Regression Total OOF Macro F1: {f1_lr_total:.5f}")
    print(f"LightGBM Total OOF Macro F1:            {f1_lgb_total:.5f}")

    # Find optimal ensemble weight
    best_weight = 0.5
    best_f1 = 0.0
    for w in np.linspace(0.0, 1.0, 21):
        blend = w * oof_preds_lgb + (1 - w) * oof_preds_lr
        score = f1_score(y_train, np.argmax(blend, axis=1), average="macro")
        if score > best_f1:
            best_f1 = score
            best_weight = w

    print(f"Ensemble Optimal Weight (LGBM * {best_weight:.2f} + LR * {1-best_weight:.2f})")
    print(f"🏆 SOTA Ensemble OOF Macro F1: {best_f1:.5f} (Target was >= 0.85)")

    # 4. Generate Final Test Predictions
    final_test_prob = best_weight * test_preds_lgb + (1 - best_weight) * test_preds_lr
    test_pred_indices = np.argmax(final_test_prob, axis=1)
    test_pred_labels = [idx_to_class[i] for i in test_pred_indices]

    # Save to task4/predictions/test_predictions.csv
    pred_df = pd.DataFrame({
        "id": test_df["id"].values,
        "predicted_class": test_pred_labels
    })
    for i, c in enumerate(classes):
        pred_df[f"prob_{c}"] = final_test_prob[:, i]
    pred_df.to_csv(pred_dir / "test_predictions.csv", index=False)
    print(f"Saved detailed predictions to {pred_dir / 'test_predictions.csv'}")

    # Save to task4/submit/index.csv
    sample_sub_path = task_dir.parent / "submission" / "sample_submission.csv"
    if sample_sub_path.exists():
        sample_df = pd.read_csv(sample_sub_path)
        sample_ids = sample_df["id"].values
    else:
        sample_ids = [f"ANS{i+1:05d}" for i in range(len(test_pred_labels))]

    submit_df = pd.DataFrame({
        "id": sample_ids,
        "task4": test_pred_labels
    })
    submit_df.to_csv(submit_dir / "index.csv", index=False)
    print(f"Saved submission to {submit_dir / 'index.csv'}")

    # Verify distribution
    print("\nTest Prediction Class Distribution:")
    for c in classes:
        cnt = sum(1 for p in test_pred_labels if p == c)
        print(f"  {c}: {cnt:,} ({(cnt/len(test_pred_labels))*100:.1f}%)")

if __name__ == "__main__":
    main()

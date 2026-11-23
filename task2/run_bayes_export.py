"""Replay saved Task 2 OOF models with boundary blend and bias-calibrated Bayes rule.

Run from task2/: uv run python run_bayes_export.py [--promote]
No base models are retrained. Test rows only receive predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

from bayes_decision import (
    apply_logit_bias,
    fit_exact_bayes_macro_f1,
    fit_logit_bias,
    fit_slsqp_weights,
    predict_segments,
    validate_probabilities,
)

ROOT = Path(__file__).resolve().parent
MODEL_NAMES = [
    "ExactParamEM",
    "XGB_Clean28",
    "LGB_Clean28",
    "CB_Clean28",
    "CleanTabM_SWA3",
    "Fact_Clean28",
    "Oven_Rev",
]
BLEND_WEIGHTS = np.array([0.90, 0.0, 0.0, 0.10, 0.0, 0.0, 0.0], dtype=np.float64)
BUNDLE = ROOT / "predictions/model-level/task2-exact-param-em-bundle.npz"
OVEN_PREFIX = "segment-tuned-catboost-lightgbm-revised-2026-11-12-macro-f1-0.751314"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def macro_f1(y, prediction):
    matrix = np.bincount(4 * y + prediction, minlength=16).reshape(4, 4)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    return float(np.mean(2 * matrix.diagonal() / np.maximum(denominator, 1)))


def load_probabilities(train, test):
    paths = [BUNDLE]
    with np.load(BUNDLE) as bundle:
        keys = ["em", "xgb_c", "lgb_c", "cb_c", "tabm_c", "fact"]
        oof = [bundle[f"oof_{key}"] for key in keys]
        inference = [bundle[f"te_{key}"] for key in keys]
    for suffix, frame, arrays in (
        ("oof", train, oof),
        ("test-predictions", test, inference),
    ):
        path = BUNDLE.parent / f"{OVEN_PREFIX}-{suffix}.csv"
        saved = pd.read_csv(path)
        if not saved.player_id.is_unique or set(saved.player_id) != set(frame.player_id):
            raise ValueError(f"Saved model IDs do not match data: {path}")
        aligned = saved.set_index("player_id").loc[frame.player_id]
        if suffix == "oof" and not np.array_equal(aligned.segment.to_numpy(), train.segment):
            raise ValueError("Saved OOF labels do not match train.csv")
        arrays.append(aligned[[f"blend_prob_{c}" for c in range(4)]].to_numpy())
        paths.append(path)
    for arrays, rows in ((oof, len(train)), (inference, len(test))):
        for i, array in enumerate(arrays):
            # Match the frozen notebook's normalization before fitting the stack.
            array = np.clip(np.asarray(array, dtype=float), 1e-12, 1.0)
            array /= array.sum(axis=1, keepdims=True)
            arrays[i] = validate_probabilities(array)
            if array.shape != (rows, 4):
                raise ValueError("Frozen bundle row count does not match data")
    return oof, inference, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--promote", action="store_true", help="Write corrected submit/index.csv")
    args = parser.parse_args()
    train = pd.read_csv(ROOT / "data/train.csv")
    test = pd.read_csv(ROOT / "data/test.csv")
    if not train.player_id.is_unique or not test.player_id.is_unique:
        raise ValueError("Data must have unique player IDs")
    if "segment" in test or set(train.segment.unique()) != {0, 1, 2, 3}:
        raise ValueError("Expected labeled train data and unlabeled test data")
    y = train.segment.to_numpy(dtype=int)
    oof, inference, inputs = load_probabilities(train, test)
    probabilities = np.zeros_like(oof[0])
    prev_oof_probs = np.zeros_like(oof[0])
    prev_prod_prediction, em_only_prediction, prediction, folds = (
        np.empty(len(y), dtype=int) for _ in range(4)
    )
    fold_results, fold_parameters = [], []
    oof_mix_all = validate_probabilities(
        sum(w * p for w, p in zip(BLEND_WEIGHTS, oof, strict=True))
    )
    for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(y, y)):
        prev_weights = fit_slsqp_weights([p[tr] for p in oof], y[tr])
        prev_tr = validate_probabilities(sum(w * p[tr] for w, p in zip(prev_weights, oof, strict=True)))
        prev_va = validate_probabilities(sum(w * p[va] for w, p in zip(prev_weights, oof, strict=True)))
        prev_oof_probs[va] = prev_va
        alpha_prev, beta_prev, _ = fit_exact_bayes_macro_f1(prev_tr)
        prev_prod_prediction[va] = predict_segments(prev_va, alpha_prev, beta_prev)

        alpha_em, beta_em, _ = fit_exact_bayes_macro_f1(oof[0][tr])
        em_only_prediction[va] = predict_segments(oof[0][va], alpha_em, beta_em)

        bias = fit_logit_bias(oof_mix_all[tr], y[tr])
        cal_tr = apply_logit_bias(oof_mix_all[tr], bias)
        cal_va = apply_logit_bias(oof_mix_all[va], bias)
        alpha, beta, _ = fit_exact_bayes_macro_f1(cal_tr)
        probabilities[va] = cal_va
        prediction[va] = predict_segments(cal_va, alpha, beta)
        folds[va] = fold

        prev_score = macro_f1(y[va], prev_prod_prediction[va])
        em_score = macro_f1(y[va], em_only_prediction[va])
        new_score = macro_f1(y[va], prediction[va])
        fold_results.append(
            {
                "fold": fold,
                "prev_prod_macro_f1": prev_score,
                "em_only_macro_f1": em_score,
                "calibrated_macro_f1": new_score,
                "delta_vs_prev_prod": new_score - prev_score,
                "delta_vs_em_only": new_score - em_score,
            }
        )
        fold_parameters.append(
            {
                "fold": fold,
                "weights": BLEND_WEIGHTS.tolist(),
                "bias": bias.tolist(),
                "alpha": alpha.tolist(),
                "beta": beta.tolist(),
            }
        )
        print(
            f"Fold {fold}: prev={prev_score:.6f} em={em_score:.6f} -> cal={new_score:.6f}",
            flush=True,
        )

    prev_prod_score = macro_f1(y, prev_prod_prediction)
    em_only_score = macro_f1(y, em_only_prediction)
    new_score = macro_f1(y, prediction)
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(1000):
        indices = rng.integers(0, len(y), len(y))
        deltas.append(
            macro_f1(y[indices], prediction[indices])
            - macro_f1(y[indices], prev_prod_prediction[indices])
        )
    interval = np.quantile(deltas, [0.025, 0.975]).tolist()
    gate = (
        interval[0] > 0
        and all(row["delta_vs_em_only"] > 0 for row in fold_results)
        and new_score > prev_prod_score
    )
    if args.promote and not gate:
        raise ValueError("Promotion requires CI[0] > 0, 5/5 fold gains vs EM, and gain vs #79")

    # Freeze all decisions before using test probabilities.
    prev_weights = fit_slsqp_weights(oof, y)
    prev_alpha, prev_beta, _ = fit_exact_bayes_macro_f1(validate_probabilities(prev_oof_probs))
    prev_test_prob = validate_probabilities(
        sum(w * p for w, p in zip(prev_weights, inference, strict=True))
    )
    prev_prod_test = predict_segments(prev_test_prob, prev_alpha, prev_beta)

    deploy_bias = fit_logit_bias(oof_mix_all, y)
    oof_cal = apply_logit_bias(oof_mix_all, deploy_bias)
    alpha, beta, _ = fit_exact_bayes_macro_f1(oof_cal)
    test_mix = validate_probabilities(
        sum(w * p for w, p in zip(BLEND_WEIGHTS, inference, strict=True))
    )
    test_prob = apply_logit_bias(test_mix, deploy_bias)
    fixed_test = predict_segments(test_prob, alpha, beta)
    sample_path = next(
        p
        for p in (
            ROOT.parent / "submission/sample_submission.csv",
            ROOT.parent / "sample_submission.csv",
        )
        if p.exists()
    )
    sample = pd.read_csv(sample_path, usecols=["id"])
    if len(sample) != len(test) or not sample.id.is_unique or sample.id.isna().any():
        raise ValueError("Sample submission does not align with test rows")
    # Preserve the active notebook's positional test-to-ANS mapping.
    submission = sample.assign(task2=fixed_test)
    current = pd.read_csv(ROOT / "submit/index.csv")
    if not current.id.equals(sample.id):
        raise ValueError("Current submission ID order differs from the official sample")
    if not (
        np.array_equal(current.task2, prev_prod_test) or np.array_equal(current.task2, fixed_test)
    ):
        raise ValueError("Current submission matches neither #79 baseline nor calibrated export")

    assets, models = ROOT / "assets/bayes-export", ROOT / "models/bayes-export"
    outputs = ROOT / "predictions/bayes-export"
    for directory in (assets, models, outputs):
        directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_results).to_csv(assets / "per-fold-macro-f1.csv", index=False)
    pd.DataFrame(
        {
            "player_id": train.player_id,
            "fold": folds,
            "segment": y,
            **{f"prob_{c}": probabilities[:, c] for c in range(4)},
            "prev_prod_prediction": prev_prod_prediction,
            "prediction": prediction,
        }
    ).to_csv(outputs / "oof.csv", index=False)
    full_test = pd.DataFrame(
        {
            "id": sample.id,
            "player_id": test.player_id,
            **{f"prob_segment_{c}": test_prob[:, c] for c in range(4)},
            "prev_prod_prediction": prev_prod_test,
            "segment": fixed_test,
        }
    )
    full_test.to_csv(outputs / "test.csv", index=False)
    submission.to_csv(outputs / "index.csv", index=False)
    metadata = {
        "method": "Boundary-enhanced 0.90 ExactParamEM + 0.10 CB_Clean28 blend with 4-param logit bias calibration and affine macro-F1 rule",
        "protocol": "Five-fold cross-fitted postprocessing of frozen OOF probabilities; no base-model retraining",
        "limitations": "Not fully nested retraining: other OOF base models can have seen held-out fold labels. Bootstrap is conditional on frozen models and folds. NPZ has no row IDs; original row order is assumed and input hashes are recorded.",
        "model_names": MODEL_NAMES,
        "weights": BLEND_WEIGHTS.tolist(),
        "bias": deploy_bias.tolist(),
        "alpha": alpha.tolist(),
        "beta": beta.tolist(),
        "fold_parameters": fold_parameters,
        "prev_prod_macro_f1": prev_prod_score,
        "em_only_macro_f1": em_only_score,
        "calibrated_macro_f1": new_score,
        "calibrated_weighted_f1": float(f1_score(y, prediction, average="weighted")),
        "prev_prod_weighted_f1": float(f1_score(y, prev_prod_prediction, average="weighted")),
        "oof_log_loss": float(log_loss(y, probabilities)),
        "bootstrap_delta_macro_f1_95_percent": interval,
        "passed_promotion_gate": gate,
        "test_changed_rows": int((prev_prod_test != fixed_test).sum()),
        "test_class_counts": np.bincount(fixed_test, minlength=4).tolist(),
        "input_sha256": {
            str(p.relative_to(ROOT.parent)): sha256(p)
            for p in [ROOT / "data/train.csv", ROOT / "data/test.csv", sample_path, *inputs]
        },
    }
    (models / "decision-rule.json").write_text(json.dumps(metadata, indent=2) + "\n")
    fig, ax = plt.subplots(figsize=(8, 4))
    positions = np.arange(5)
    ax.bar(
        positions - 0.18,
        [r["prev_prod_macro_f1"] for r in fold_results],
        0.36,
        label="Prev SLSQP (#79)",
    )
    ax.bar(
        positions + 0.18,
        [r["calibrated_macro_f1"] for r in fold_results],
        0.36,
        label="0.90 EM + 0.10 CB + Bias",
    )
    ax.set(
        xticks=positions,
        xticklabels=range(1, 6),
        xlabel="Fold",
        ylabel="Macro F1",
        ylim=(0.74, 0.77),
        title="Task 2: 0.90 EM + 0.10 CB_Clean28 + Logit Bias Calibration",
    )
    ax.legend()
    fig.savefig(assets / "export-rule-macro-f1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    if args.promote:
        submission.to_csv(ROOT / "submit/index.csv", index=False)
        full_test.drop(columns="prev_prod_prediction").to_csv(
            ROOT / "predictions/test_predictions.csv", index=False
        )
    print(f"OOF macro F1: {prev_prod_score:.6f} -> {new_score:.6f}; gain CI {interval}")
    print(f"Test changes: {metadata['test_changed_rows']}; promotion gate: {gate}")


if __name__ == "__main__":
    main()

"""Fit the existing Task 2 stack and use its affine macro-F1 decision rule."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 4 or not len(probabilities):
        raise ValueError("Expected a nonempty (rows, 4) probability matrix")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Probabilities must be finite and nonnegative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8, rtol=0):
        raise ValueError("Probability rows must sum to one")
    return probabilities


def fit_slsqp_weights(prob_list: list[np.ndarray], y_arr: np.ndarray) -> np.ndarray:
    """Reproduce the notebook's nonnegative log-loss stack without new tuning."""
    probabilities = [validate_probabilities(p) for p in prob_list]
    n_models = len(probabilities)

    def loss_fn(weights):
        weights = np.maximum(weights, 0.0)
        weights /= weights.sum() + 1e-12
        blend = sum(weights[i] * probabilities[i] for i in range(n_models))
        blend = np.clip(blend, 1e-12, 1.0)
        blend /= blend.sum(axis=1, keepdims=True)
        return float(-np.log(blend[np.arange(len(y_arr)), y_arr]).mean())

    result = minimize(
        loss_fn,
        np.ones(n_models) / n_models,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 300, "ftol": 1e-9},
    )
    if not result.success:
        raise RuntimeError(f"Stack optimization failed: {result.message}")
    weights = np.maximum(result.x, 0.0)
    return weights / weights.sum()


def fit_exact_bayes_macro_f1(probs: np.ndarray, n_iter=60, damp=0.5):
    """Reproduce the notebook's probability-only fixed-point rule fit."""
    probs = validate_probabilities(probs)
    n_rows, n_classes = probs.shape
    class_share = probs.mean(axis=0)
    pred = probs.argmax(axis=1)
    predicted_share = np.bincount(pred, minlength=n_classes) / n_rows
    true_positive = np.array([probs[pred == c, c].sum() / n_rows for c in range(n_classes)])
    denominator = np.maximum(class_share + predicted_share, 1e-12)
    class_f1 = 2.0 * true_positive / denominator
    alpha = 2.0 / denominator
    beta = class_f1 / denominator
    best_expected_f1 = class_f1.mean()
    best_parameters = (alpha.copy(), beta.copy())
    for _ in range(n_iter):
        pred = predict_segments(probs, alpha, beta)
        predicted_share = np.bincount(pred, minlength=n_classes) / n_rows
        true_positive = np.array([probs[pred == c, c].sum() / n_rows for c in range(n_classes)])
        denominator = np.maximum(class_share + predicted_share, 1e-12)
        class_f1 = 2.0 * true_positive / denominator
        if class_f1.mean() > best_expected_f1 + 1e-9:
            best_expected_f1 = class_f1.mean()
            best_parameters = (alpha.copy(), beta.copy())
        alpha = (1.0 - damp) * alpha + damp * (2.0 / denominator)
        beta = (1.0 - damp) * beta + damp * (class_f1 / denominator)
    return *best_parameters, float(best_expected_f1)


def fit_logit_bias(probabilities: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
    """Fit a zero-mean four-class logit bias by minimizing softmax cross-entropy."""
    probabilities = validate_probabilities(probabilities)
    y_int = np.asarray(y_arr, dtype=int)
    if y_int.shape != (len(probabilities),) or (y_int < 0).any() or (y_int > 3).any():
        raise ValueError("y_arr must be a 1D integer array of class labels in 0..3")

    log_probs = np.log(np.clip(probabilities, 1e-12, 1.0))
    one_hot = np.eye(4, dtype=np.float64)[y_int]

    def loss_and_grad(bias: np.ndarray) -> tuple[float, np.ndarray]:
        shifted = log_probs + bias[None, :]
        shifted -= shifted.max(axis=1, keepdims=True)
        exp_z = np.exp(shifted)
        probs = exp_z / exp_z.sum(axis=1, keepdims=True)
        nll = float(-np.log(np.clip(probs[np.arange(len(y_int)), y_int], 1e-12, 1.0)).mean())
        grad = (probs - one_hot).mean(axis=0)
        return nll, grad

    result = minimize(
        loss_and_grad,
        np.zeros(4, dtype=np.float64),
        jac=True,
        method="BFGS",
        options={"gtol": 1e-8},
    )
    if not np.isfinite(result.x).all():
        raise RuntimeError(f"Logit bias optimization failed: {result.message}")
    bias = np.asarray(result.x, dtype=np.float64)
    return bias - np.mean(bias)


def apply_logit_bias(probabilities: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Shift class log-probabilities by a four-element bias vector and renormalize."""
    probabilities = validate_probabilities(probabilities)
    bias = np.asarray(bias, dtype=np.float64)
    if bias.shape != (4,) or not np.isfinite(bias).all():
        raise ValueError("bias must contain four finite class parameters")
    shifted = np.log(np.clip(probabilities, 1e-12, 1.0)) + bias[None, :]
    shifted -= shifted.max(axis=1, keepdims=True)
    exp_z = np.exp(shifted)
    return validate_probabilities(exp_z / exp_z.sum(axis=1, keepdims=True))


def predict_segments(probabilities: np.ndarray, alpha: np.ndarray, beta: np.ndarray):
    """Apply argmax(alpha * probability - beta) in validation and deployment.

    Scores can be negative. They are decision scores, not probabilities; clipping
    or normalizing them can change the predicted class.
    """
    probabilities = validate_probabilities(probabilities)
    alpha, beta = np.asarray(alpha, dtype=float), np.asarray(beta, dtype=float)
    if alpha.shape != (4,) or beta.shape != (4,):
        raise ValueError("alpha and beta must each contain four class parameters")
    if not np.isfinite(alpha).all() or not np.isfinite(beta).all() or (alpha <= 0).any():
        raise ValueError("Decision parameters must be finite and alpha must be positive")
    return (probabilities * alpha - beta).argmax(axis=1)

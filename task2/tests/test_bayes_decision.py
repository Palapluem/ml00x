"""Regression checks for the Task 2 affine export rule."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bayes_decision import (
    apply_logit_bias,
    fit_exact_bayes_macro_f1,
    fit_logit_bias,
    predict_segments,
)


def test_export_keeps_beta_and_negative_scores():
    probabilities = np.array([[0.4, 0.3, 0.2, 0.1]])
    alpha = np.ones(4)
    beta = np.array([0.9, 0.7, 0.8, 0.9])
    assert (probabilities * alpha).argmax(1).tolist() == [0]
    # All scores are negative. Clipping scores to zero would incorrectly choose 0.
    assert predict_segments(probabilities, alpha, beta).tolist() == [1]


def test_rule_parameters_round_trip_through_json():
    probabilities = np.random.default_rng(42).dirichlet([4, 3, 2, 1], size=500)
    alpha, beta, _ = fit_exact_bayes_macro_f1(probabilities)
    saved = json.loads(json.dumps({"alpha": alpha.tolist(), "beta": beta.tolist()}))
    np.testing.assert_array_equal(
        predict_segments(probabilities, alpha, beta),
        predict_segments(probabilities, saved["alpha"], saved["beta"]),
    )


@pytest.mark.parametrize(
    "probabilities",
    [
        [[0.4, 0.3, 0.2, float("nan")]],
        [[0.4, 0.3, 0.2, -0.1]],
        [[0.4, 0.3, 0.2, 0.2]],
        [[0.5, 0.5]],
    ],
)
def test_invalid_probabilities_fail_before_export(probabilities):
    with pytest.raises(ValueError):
        predict_segments(probabilities, np.ones(4), np.zeros(4))


def test_notebook_executes_one_affine_export(tmp_path):
    notebook = json.loads(
        (Path(__file__).resolve().parents[1] / "task2_player_segment.ipynb").read_text()
    )
    exports = [
        cell
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and "sub_index.to_csv" in "".join(cell["source"])
    ]
    assert len(exports) == 1
    rows = 25889
    probabilities = np.tile([0.4, 0.3, 0.2, 0.1], (rows, 1))
    namespace = {
        "np": np,
        "pd": pd,
        "Path": Path,
        "predict_segments": predict_segments,
        "ens_test_preds": probabilities,
        "DEPLOY_ALPHA": np.ones(4),
        "DEPLOY_BETA": np.array([0.9, 0.7, 0.8, 0.9]),
        "test_df": pd.DataFrame({"player_id": [f"P{i}" for i in range(rows)]}),
        "PRED_DIR": str(tmp_path / "predictions"),
        "SUBMIT_DIR": str(tmp_path / "submit"),
        "y": np.arange(4),
        "opt_oof_preds": np.arange(4),
    }
    exec(compile("".join(exports[0]["source"]), "notebook-export", "exec"), namespace)
    saved = pd.read_csv(tmp_path / "submit/index.csv")
    details = pd.read_csv(tmp_path / "predictions/test_predictions.csv")
    assert saved.task2.tolist() == [1] * rows
    assert saved.id.tolist() == [f"ANS{i + 1:05d}" for i in range(rows)]
    np.testing.assert_array_equal(saved.task2, details.segment)
    np.testing.assert_allclose(details[[f"prob_segment_{c}" for c in range(4)]], probabilities)


def test_fit_and_apply_logit_bias_round_trip():
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(800, 4))
    shifted_logits = logits + np.array([0.3, -0.2, 0.1, -0.2])
    probs_true = np.exp(shifted_logits - shifted_logits.max(axis=1, keepdims=True))
    probs_true /= probs_true.sum(axis=1, keepdims=True)
    y_arr = np.array([rng.choice(4, p=p) for p in probs_true])

    uncal = np.exp(logits - logits.max(axis=1, keepdims=True))
    uncal /= uncal.sum(axis=1, keepdims=True)
    bias = fit_logit_bias(uncal, y_arr)
    assert bias.shape == (4,)
    assert abs(bias.mean()) < 1e-12

    cal = apply_logit_bias(uncal, bias)
    assert cal.shape == uncal.shape
    np.testing.assert_allclose(cal.sum(axis=1), 1.0, atol=1e-8)


@pytest.mark.parametrize("bad_bias", [[1.0, 2.0, 3.0], [0.0, float("nan"), 0.0, 0.0]])
def test_apply_logit_bias_rejects_invalid_bias(bad_bias):
    probs = np.array([[0.4, 0.3, 0.2, 0.1]])
    with pytest.raises(ValueError):
        apply_logit_bias(probs, np.array(bad_bias))

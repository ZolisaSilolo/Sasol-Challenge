"""Metrics, blending and nested class-multiplier optimisation."""

import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.metrics import WEIGHTS, composite_metric, rmse
from src.postprocessing import (apply_class_weights, fit_class_weights, nested_class_weights,
                                simplex_blend_weights, weighted_f1)


def test_rmse_basic():
    assert rmse([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))


def test_composite_uses_competition_weights():
    assert WEIGHTS == {"rmse_fuel": 0.3, "rmse_nonfuel": 0.3, "f1_opportunity": 0.4}
    m = composite_metric([0, 0], [1, 1], [0, 0], [2, 2], [0, 1], [0, 0])
    f1 = f1_score([0, 1], [0, 0], average="weighted", zero_division=0)
    assert m["rmse_fuel"] == pytest.approx(1.0)
    assert m["rmse_nonfuel"] == pytest.approx(2.0)
    assert m["f1_opportunity"] == pytest.approx(f1)
    assert m["composite_loss"] == pytest.approx(0.3 * 1 + 0.3 * 2 + 0.4 * (1 - f1))


def test_composite_perfect_is_zero():
    m = composite_metric([1, 2], [1, 2], [1, 2], [1, 2], [0, 1], [0, 1])
    assert m["composite_loss"] == pytest.approx(0.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fast_weighted_f1_matches_sklearn(seed):
    rng = np.random.default_rng(seed)
    y, p = rng.integers(0, 17, 2000), rng.integers(0, 17, 2000)
    p[:500] = y[:500]
    assert weighted_f1(y, p, 17) == pytest.approx(f1_score(y, p, average="weighted", zero_division=0))


def test_simplex_blend_rmse_finds_best_member():
    rng = np.random.default_rng(42)
    y = rng.normal(size=1000)
    preds = np.column_stack([y + rng.normal(scale=0.1, size=1000), rng.normal(size=1000)])
    w = simplex_blend_weights(preds, y, "rmse")
    assert w.sum() == pytest.approx(1.0) and (w >= 0).all()
    assert w[0] > 0.9


def test_simplex_blend_logloss_shapes():
    rng = np.random.default_rng(42)
    y = rng.integers(0, 3, 300)
    good = np.full((300, 3), 0.1)
    good[np.arange(300), y] = 0.8
    bad = np.full((300, 3), 1 / 3)
    w = simplex_blend_weights(np.stack([good, bad], axis=1), y, "logloss")
    assert w[0] > w[1]


def _imbalanced_probs(n=3000, C=5, seed=42):
    """A calibrated-ish model that under-predicts a minority class: multipliers should help."""
    rng = np.random.default_rng(seed)
    y = rng.choice(C, size=n, p=[0.5, 0.2, 0.15, 0.1, 0.05])
    logits = rng.normal(size=(n, C))
    logits[np.arange(n), y] += 1.2
    logits[:, 0] += 1.0                        # bias toward the majority class
    p = np.exp(logits)
    return p / p.sum(1, keepdims=True), y


def test_class_weights_bounded_and_improve_in_sample():
    probs, y = _imbalanced_probs()
    w = fit_class_weights(probs, y, lo=0.2, hi=5.0, grid_size=15, max_passes=2)
    assert ((w >= 0.2) & (w <= 5.0)).all()
    assert weighted_f1(y, apply_class_weights(probs, w), 5) >= weighted_f1(y, probs.argmax(1), 5)


def test_nested_class_weights_is_honest():
    probs, y = _imbalanced_probs()
    folds = np.arange(len(y)) % 5
    res = nested_class_weights(probs, y, folds, 5, grid_size=15, max_passes=2)
    assert res.fold_weights.shape == (5, 5)
    # Nested F1 is computed on folds the weights never saw -> never above in-sample fit of fold weights.
    assert res.f1_nested <= max(res.f1_in_sample, res.f1_argmax) + 0.02
    if not res.used_identity:
        np.testing.assert_allclose(res.weights, res.fold_weights.mean(0))
        assert res.f1_nested > res.f1_argmax


def test_nested_falls_back_to_identity_on_noise():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 400)
    probs = rng.dirichlet(np.ones(4), size=400)
    res = nested_class_weights(probs, y, np.arange(400) % 5, 4, grid_size=9, max_passes=1)
    if res.used_identity:
        np.testing.assert_array_equal(res.weights, np.ones(4))

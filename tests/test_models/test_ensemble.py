"""Tests for ensemble meta-learner combiner."""

from __future__ import annotations

import numpy as np
import pytest

from bist_predict.models.ensemble import EnsembleCombiner


@pytest.fixture
def model_predictions() -> dict:
    """Simulated predictions from 4 models."""
    rng = np.random.default_rng(42)
    n = 100
    return {
        "xgboost": (rng.uniform(0, 1, n), rng.normal(0, 0.01, n)),
        "lightgbm": (rng.uniform(0, 1, n), rng.normal(0, 0.01, n)),
        "lstm": (rng.uniform(0, 1, n), rng.normal(0, 0.01, n)),
        "transformer": (rng.uniform(0, 1, n), rng.normal(0, 0.01, n)),
    }


class TestEnsembleCombiner:
    def test_train_meta_learner(self, model_predictions: dict) -> None:
        rng = np.random.default_rng(42)
        y_dir = rng.integers(0, 2, 100).astype(np.int64)
        y_pct = rng.normal(0, 0.01, 100)

        combiner = EnsembleCombiner()
        combiner.train(model_predictions, y_dir, y_pct)
        assert combiner.is_trained

    def test_combine_predictions(self, model_predictions: dict) -> None:
        rng = np.random.default_rng(42)
        y_dir = rng.integers(0, 2, 100).astype(np.int64)
        y_pct = rng.normal(0, 0.01, 100)

        combiner = EnsembleCombiner()
        combiner.train(model_predictions, y_dir, y_pct)

        probs, pct = combiner.predict(model_predictions)
        assert probs.shape == (100,)
        assert pct.shape == (100,)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_with_regime_weights(self, model_predictions: dict) -> None:
        rng = np.random.default_rng(42)
        y_dir = rng.integers(0, 2, 100).astype(np.int64)
        y_pct = rng.normal(0, 0.01, 100)

        combiner = EnsembleCombiner()
        combiner.train(model_predictions, y_dir, y_pct)

        regime_weights = {"momentum_weight": 0.6, "mean_reversion_weight": 0.3, "pairs_weight": 0.1}
        probs, pct = combiner.predict(model_predictions, regime_weights=regime_weights)
        assert probs.shape == (100,)

    def test_simple_average_fallback(self) -> None:
        """Without training, falls back to simple average."""
        preds = {
            "m1": (np.array([0.8, 0.2]), np.array([0.01, -0.01])),
            "m2": (np.array([0.6, 0.4]), np.array([0.02, -0.02])),
        }
        combiner = EnsembleCombiner()
        probs, pct = combiner.predict(preds)
        np.testing.assert_array_almost_equal(probs, [0.7, 0.3])
        np.testing.assert_array_almost_equal(pct, [0.015, -0.015])

    def test_save_and_load_trained_combiner(self, model_predictions: dict, tmp_path) -> None:
        rng = np.random.default_rng(42)
        y_dir = rng.integers(0, 2, 100).astype(np.int64)
        y_pct = rng.normal(0, 0.01, 100)

        combiner = EnsembleCombiner()
        combiner.train(model_predictions, y_dir, y_pct)
        expected_probs, expected_pct = combiner.predict(model_predictions)

        combiner.save(str(tmp_path))
        loaded = EnsembleCombiner()
        loaded.load(str(tmp_path))
        actual_probs, actual_pct = loaded.predict(model_predictions)

        assert loaded.is_trained
        np.testing.assert_allclose(actual_probs, expected_probs)
        np.testing.assert_allclose(actual_pct, expected_pct)

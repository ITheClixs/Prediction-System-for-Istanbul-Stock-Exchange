"""Ensemble meta-learner -- combines predictions from multiple models."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression, Ridge


class EnsembleCombiner:
    """Meta-learner that combines predictions from individual models.

    If trained, uses logistic regression on model probabilities for direction
    and ridge regression for percentage move. Falls back to simple averaging
    if not trained.
    """

    def __init__(self) -> None:
        self._dir_meta: LogisticRegression | None = None
        self._pct_meta: Ridge | None = None
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def save(self, path: str) -> None:
        """Persist trained ensemble meta-learners."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "ensemble.pkl", "wb") as f:
            pickle.dump(
                {
                    "dir_meta": self._dir_meta,
                    "pct_meta": self._pct_meta,
                    "is_trained": self._is_trained,
                },
                f,
            )

    def load(self, path: str) -> None:
        """Load trained ensemble meta-learners."""
        p = Path(path)
        with open(p / "ensemble.pkl", "rb") as f:
            payload = pickle.load(f)
        self._dir_meta = payload["dir_meta"]
        self._pct_meta = payload["pct_meta"]
        self._is_trained = bool(payload["is_trained"])

    def train(
        self,
        model_predictions: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
        y_dir: NDArray[np.int64],
        y_pct: NDArray[np.float64],
    ) -> None:
        """Train meta-learner on stacked model predictions."""
        X_dir, X_pct = self._stack_predictions(model_predictions)

        self._dir_meta = LogisticRegression(random_state=42, max_iter=1000)
        self._dir_meta.fit(X_dir, y_dir)

        self._pct_meta = Ridge(alpha=1.0)
        self._pct_meta.fit(X_pct, y_pct)

        self._is_trained = True

    def predict(
        self,
        model_predictions: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
        regime_weights: dict[str, float] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Combine model predictions into ensemble output."""
        if not self._is_trained:
            return self._simple_average(model_predictions)

        X_dir, X_pct = self._stack_predictions(model_predictions)
        dir_probs = self._dir_meta.predict_proba(X_dir)[:, 1]
        pct_pred = self._pct_meta.predict(X_pct)

        return dir_probs.astype(np.float64), pct_pred.astype(np.float64)

    def _stack_predictions(
        self,
        model_predictions: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Stack individual model predictions into feature matrices."""
        dir_cols = []
        pct_cols = []
        for name in sorted(model_predictions.keys()):
            probs, pct = model_predictions[name]
            dir_cols.append(probs)
            pct_cols.append(pct)

        X_dir = np.column_stack(dir_cols)
        X_pct = np.column_stack(pct_cols)
        return X_dir, X_pct

    def _simple_average(
        self,
        model_predictions: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Fallback: simple average of all model predictions."""
        all_probs = []
        all_pct = []
        for probs, pct in model_predictions.values():
            all_probs.append(probs)
            all_pct.append(pct)

        avg_probs = np.mean(all_probs, axis=0)
        avg_pct = np.mean(all_pct, axis=0)
        return avg_probs.astype(np.float64), avg_pct.astype(np.float64)

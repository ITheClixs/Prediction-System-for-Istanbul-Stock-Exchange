"""Platt scaling for confidence calibration."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    """Platt scaling -- fits sigmoid to map raw scores to calibrated probabilities.

    When the model outputs "78% UP", we want ~78% of such predictions to actually
    be correct.
    """

    def __init__(self, min_confidence: float = 0.60) -> None:
        self._min_confidence = min_confidence
        self._model: LogisticRegression | None = None
        self._fitted = False
        self._status = "unfitted"

    @property
    def min_confidence(self) -> float:
        return self._min_confidence

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def status(self) -> str:
        return self._status

    def fit(self, raw_scores: NDArray[np.float64], true_labels: NDArray[np.int64]) -> None:
        """Fit Platt scaling sigmoid on validation set."""
        if len(np.unique(true_labels)) < 2:
            self._model = None
            self._fitted = False
            self._status = "skipped_single_class"
            return
        self._model = LogisticRegression(random_state=42, max_iter=1000)
        self._model.fit(raw_scores.reshape(-1, 1), true_labels)
        self._fitted = True
        self._status = "fitted"

    def transform(self, raw_scores: NDArray[np.float64]) -> NDArray[np.float64]:
        """Transform raw scores to calibrated probabilities."""
        if not self._fitted or self._model is None:
            raise RuntimeError("Calibrator not fitted -- call fit() first")

        calibrated = self._model.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
        return calibrated.astype(np.float64)

    def save(self, path: str) -> None:
        """Persist calibration state."""
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "calibrator.pkl", "wb") as f:
            pickle.dump(
                {
                    "min_confidence": self._min_confidence,
                    "model": self._model,
                    "fitted": self._fitted,
                    "status": self._status,
                },
                f,
            )

    def load(self, path: str) -> None:
        """Load calibration state."""
        p = Path(path)
        with open(p / "calibrator.pkl", "rb") as f:
            payload = pickle.load(f)
        self._min_confidence = float(payload["min_confidence"])
        self._model = payload["model"]
        self._fitted = bool(payload["fitted"])
        self._status = str(payload["status"])

"""Tests for calibrated ensemble research orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bist_predict.features.store import FeatureStore
from bist_predict.research.ensemble_pipeline import (
    EnsembleBacktestConfig,
    EnsembleTrainConfig,
    run_walk_forward_backtest,
    train_calibrated_ensemble,
)
from bist_predict.storage.database import Database


class FakeModel:
    """Small deterministic model for orchestration tests."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._bias = 0.0
        self._n_features: int | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def n_features(self) -> int | None:
        return self._n_features

    def train(self, X_train, y_dir_train, y_pct_train, X_val=None, y_dir_val=None, y_pct_val=None):
        self._n_features = X_train.shape[-1]
        self._bias = float(np.mean(y_dir_train) - 0.5)
        return {"val_accuracy": 0.5, "val_mae": 0.01}

    def predict(self, X):
        flat = X.reshape(X.shape[0], -1)
        raw = np.tanh(flat[:, 0] * 0.01 + self._bias)
        if self._name.endswith("2"):
            raw = -raw
        probs = np.clip(0.5 + raw * 0.2, 0.01, 0.99)
        pct = (probs - 0.5) / 10.0
        return probs.astype(np.float64), pct.astype(np.float64)

    def save(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "fake.txt").write_text(self._name)

    def load(self, path: str) -> None:
        self._name = (Path(path) / "fake.txt").read_text()


def fake_factory(name: str, input_size: int | None):
    return FakeModel(name)


@pytest.fixture
def db(tmp_db_path: Path) -> Database:
    db = Database(tmp_db_path)
    db.initialize()
    store = FeatureStore(db)
    with db.connect() as conn:
        for i in range(60):
            date_str = f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}"
            close = 100.0 + (i % 2)
            conn.execute(
                """INSERT INTO raw_prices
                   (ticker, date, open, high, low, close, adj_close, volume, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("THYAO", date_str, close, close + 1, close - 1, close, close, 1000, "test"),
            )
        conn.commit()

    for i in range(60):
        date_str = f"2026-{1 + i // 28:02d}-{1 + i % 28:02d}"
        store.save(
            "THYAO",
            date_str,
            {"f1": float(i), "f2": float(i % 3), "f3": float(i % 5)},
        )
    return db


def test_train_calibrated_ensemble_registers_artifacts(db: Database) -> None:
    report = train_calibrated_ensemble(
        db,
        EnsembleTrainConfig(
            model_names=["fake1", "fake2"],
            validation_fraction=0.25,
            min_common_validation=3,
            version="test-version",
        ),
        tickers=["THYAO"],
        model_factory=fake_factory,
    )

    assert report.version == "test-version"
    assert report.base_models == ["fake1", "fake2"]
    assert report.validation_samples > 0
    assert Path(report.ensemble_path, "ensemble.pkl").exists()
    assert Path(report.ensemble_path, "calibrator.pkl").exists()
    assert Path(report.ensemble_path, "ensemble_metadata.json").exists()


def test_train_requires_two_common_models(db: Database) -> None:
    with pytest.raises(ValueError, match="At least two base models"):
        train_calibrated_ensemble(
            db,
            EnsembleTrainConfig(
                model_names=["fake1"],
                validation_fraction=0.25,
                min_common_validation=3,
                version="one-model",
            ),
            tickers=["THYAO"],
            model_factory=fake_factory,
        )


def test_walk_forward_backtest_returns_metrics(db: Database) -> None:
    report = run_walk_forward_backtest(
        db,
        EnsembleBacktestConfig(
            model_names=["fake1", "fake2"],
            train_window=20,
            val_window=8,
            step_size=8,
            validation_fraction=0.25,
            min_confidence=0.55,
        ),
        tickers=["THYAO"],
        model_factory=fake_factory,
    )

    assert report.folds > 0
    assert report.trade_count > 0
    assert "accuracy" in report.prediction_metrics
    assert "sharpe_ratio" in report.trading_metrics

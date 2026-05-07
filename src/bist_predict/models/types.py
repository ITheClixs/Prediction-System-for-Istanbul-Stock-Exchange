"""Model types — Prediction dataclass, PredictionModel protocol, dataset builders."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from bist_predict.features.store import FeatureStore
from bist_predict.storage.database import Database


@dataclass(frozen=True)
class Prediction:
    """A single stock prediction."""

    ticker: str
    direction: str  # "UP" or "DOWN"
    confidence: float  # 0.0 to 1.0
    predicted_pct_move: float
    model_name: str

    @property
    def is_buy(self) -> bool:
        return self.direction == "UP"

    @property
    def is_sell(self) -> bool:
        return self.direction == "DOWN"

    @property
    def signal_tier(self) -> str:
        if self.direction == "UP" and self.confidence >= 0.80:
            return "STRONG BUY"
        elif self.direction == "UP" and self.confidence >= 0.70:
            return "BUY"
        elif self.direction == "DOWN" and self.confidence >= 0.80:
            return "STRONG SELL"
        elif self.direction == "DOWN" and self.confidence >= 0.70:
            return "SELL"
        return "HOLD"


@dataclass(frozen=True, order=True)
class DatasetKey:
    """Stable identity for a supervised sample."""

    date: str
    ticker: str


class PredictionModel(Protocol):
    """Protocol for all prediction models."""

    @property
    def name(self) -> str: ...

    def train(
        self,
        X_train: NDArray[np.float64],
        y_dir_train: NDArray[np.int64],
        y_pct_train: NDArray[np.float64],
        X_val: NDArray[np.float64] | None = None,
        y_dir_val: NDArray[np.int64] | None = None,
        y_pct_val: NDArray[np.float64] | None = None,
    ) -> dict[str, float]: ...

    def predict(
        self, X: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Returns (direction_probabilities, predicted_pct_moves)."""
        ...

    def save(self, path: str) -> None: ...

    def load(self, path: str) -> None: ...


# Type aliases for dataset tuples
TrainDataset = tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64], list[str]]
KeyedTrainDataset = tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
    list[DatasetKey],
    list[str],
]


def _coerce_feature_value(value: object) -> float:
    """Convert persisted feature values to model-friendly floats."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        value = float(value)
        return 0.0 if math.isnan(value) else value
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(value) else value


def _get_training_feature_names(
    db: Database,
    ticker: str,
    min_features: int = 3,
) -> tuple[list[str], list[str], dict[str, float]]:
    """Return canonical feature names and latest feature snapshot for inference."""
    store = FeatureStore(db)

    with db.connect() as conn:
        date_rows = conn.execute(
            """SELECT DISTINCT date FROM features
               WHERE ticker = ? ORDER BY date""",
            (ticker,),
        ).fetchall()
        price_rows = conn.execute(
            """SELECT date, close FROM raw_prices
               WHERE ticker = ? ORDER BY date""",
            (ticker,),
        ).fetchall()

    if not date_rows:
        return [], [], {}

    all_dates = [r[0] for r in date_rows]
    price_map = {r[0]: r[1] for r in price_rows}
    feature_names: list[str] | None = None

    for i, d in enumerate(all_dates[:-1]):
        next_date = all_dates[i + 1]
        if d not in price_map or next_date not in price_map:
            continue
        features = store.load(ticker, d)
        if len(features) < min_features:
            continue
        feature_names = sorted(features.keys())
        break

    latest_date = all_dates[-1]
    latest_features = store.load(ticker, latest_date)
    return feature_names or sorted(latest_features.keys()), all_dates, latest_features


def build_tabular_dataset(
    db: Database,
    ticker: str,
    min_features: int = 3,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64], list[str]]:
    """Build feature matrix and labels from stored features and prices.

    Labels: next-day direction (1=UP, 0=DOWN) and next-day percentage move.
    """
    X, y_dir, y_pct, keys, _ = build_tabular_dataset_with_keys(
        db, ticker, min_features=min_features,
    )
    return X, y_dir, y_pct, [key.date for key in keys]


def build_tabular_dataset_with_keys(
    db: Database,
    ticker: str,
    min_features: int = 3,
) -> KeyedTrainDataset:
    """Build feature matrix, labels, and sample keys from stored data."""
    store = FeatureStore(db)

    with db.connect() as conn:
        date_rows = conn.execute(
            """SELECT DISTINCT date FROM features
               WHERE ticker = ? ORDER BY date""",
            (ticker,),
        ).fetchall()

    if not date_rows:
        return np.empty((0, 0)), np.empty(0, dtype=np.int64), np.empty(0), [], []

    all_dates = [r[0] for r in date_rows]

    with db.connect() as conn:
        price_rows = conn.execute(
            """SELECT date, close FROM raw_prices
               WHERE ticker = ? ORDER BY date""",
            (ticker,),
        ).fetchall()

    price_map = {r[0]: r[1] for r in price_rows}

    feature_rows = []
    labels_dir = []
    labels_pct = []
    keys: list[DatasetKey] = []
    feature_names = None

    for i, d in enumerate(all_dates[:-1]):
        next_date = all_dates[i + 1]
        if d not in price_map or next_date not in price_map:
            continue

        features = store.load(ticker, d)
        if len(features) < min_features:
            continue

        if feature_names is None:
            feature_names = sorted(features.keys())

        row = [_coerce_feature_value(features.get(f)) for f in feature_names]
        feature_rows.append(row)

        current_price = price_map[d]
        next_price = price_map[next_date]
        pct_move = (next_price - current_price) / current_price if current_price > 0 else 0.0
        direction = 1 if pct_move > 0 else 0

        labels_dir.append(direction)
        labels_pct.append(pct_move)
        keys.append(DatasetKey(date=d, ticker=ticker))

    if not feature_rows:
        return np.empty((0, 0)), np.empty(0, dtype=np.int64), np.empty(0), [], []

    X = np.array(feature_rows, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0)
    y_dir = np.array(labels_dir, dtype=np.int64)
    y_pct = np.array(labels_pct, dtype=np.float64)

    return X, y_dir, y_pct, keys, feature_names or []


def build_sequence_dataset(
    db: Database,
    ticker: str,
    seq_len: int = 30,
    min_features: int = 3,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.float64], list[str]]:
    """Build sequential dataset for LSTM/Transformer models."""
    X_seq, y_dir, y_pct, keys, _ = build_sequence_dataset_with_keys(
        db, ticker, min_features=min_features,
        seq_len=seq_len,
    )
    return X_seq, y_dir, y_pct, [key.date for key in keys]


def build_sequence_dataset_with_keys(
    db: Database,
    ticker: str,
    seq_len: int = 30,
    min_features: int = 3,
) -> KeyedTrainDataset:
    """Build sequential dataset with stable sample keys."""
    X_flat, y_dir_flat, y_pct_flat, keys_flat, feature_names = build_tabular_dataset_with_keys(
        db, ticker, min_features=min_features,
    )

    if X_flat.shape[0] < seq_len + 1:
        return (
            np.empty((0, seq_len, 0)),
            np.empty(0, dtype=np.int64),
            np.empty(0),
            [],
            feature_names,
        )

    sequences = []
    labels_dir = []
    labels_pct = []
    keys: list[DatasetKey] = []

    for i in range(seq_len, len(X_flat)):
        sequences.append(X_flat[i - seq_len : i])
        labels_dir.append(y_dir_flat[i])
        labels_pct.append(y_pct_flat[i])
        keys.append(keys_flat[i])

    X_seq = np.array(sequences, dtype=np.float64)
    y_dir = np.array(labels_dir, dtype=np.int64)
    y_pct = np.array(labels_pct, dtype=np.float64)

    return X_seq, y_dir, y_pct, keys, feature_names


def build_sequence_inference_row(
    db: Database,
    ticker: str,
    seq_len: int = 30,
    min_features: int = 3,
) -> tuple[NDArray[np.float64], str] | None:
    """Build a single latest sequence row from stored feature snapshots."""
    feature_names, all_dates, latest_features = _get_training_feature_names(
        db, ticker, min_features=min_features,
    )
    if not all_dates or len(latest_features) < min_features or len(all_dates) < seq_len:
        return None

    store = FeatureStore(db)
    rows = []
    for feature_date in all_dates[-seq_len:]:
        features = store.load(ticker, feature_date)
        if len(features) < min_features:
            return None
        rows.append([_coerce_feature_value(features.get(name)) for name in feature_names])

    X = np.array([rows], dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0)
    return X, all_dates[-1]


def build_inference_row(
    db: Database,
    ticker: str,
    min_features: int = 3,
) -> tuple[NDArray[np.float64], str] | None:
    """Build a single inference row from the latest stored feature snapshot."""
    feature_names, all_dates, latest_features = _get_training_feature_names(
        db, ticker, min_features=min_features,
    )
    if not all_dates or len(latest_features) < min_features:
        return None

    latest_date = all_dates[-1]
    row = [_coerce_feature_value(latest_features.get(name)) for name in feature_names]
    X = np.array([row], dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0)
    return X, latest_date

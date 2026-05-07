"""Calibrated ensemble training and walk-forward backtesting."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from bist_predict.evaluation.backtest import WalkForwardBacktest
from bist_predict.evaluation.metrics import compute_prediction_metrics, compute_trading_metrics
from bist_predict.models.calibration import PlattCalibrator
from bist_predict.models.ensemble import EnsembleCombiner
from bist_predict.models.factory import create_model, parse_model_names
from bist_predict.models.registry import ModelRegistry
from bist_predict.models.types import (
    DatasetKey,
    build_sequence_dataset_with_keys,
    build_tabular_dataset_with_keys,
)
from bist_predict.storage.database import Database

ModelFactory = Callable[[str, int | None], object]

SEQUENCE_MODELS = {"lstm", "transformer"}


@dataclass(frozen=True)
class EnsembleTrainConfig:
    """Configuration for calibrated ensemble training."""

    model_names: list[str] = field(default_factory=lambda: ["xgboost", "lightgbm"])
    include_neural: bool = False
    seq_len: int = 30
    validation_fraction: float = 0.2
    min_features: int = 3
    min_common_validation: int = 5
    min_confidence: float = 0.70
    version: str | None = None


@dataclass(frozen=True)
class EnsembleBacktestConfig:
    """Configuration for walk-forward calibrated ensemble evaluation."""

    model_names: list[str] = field(default_factory=lambda: ["xgboost", "lightgbm"])
    include_neural: bool = False
    seq_len: int = 30
    validation_fraction: float = 0.2
    min_features: int = 3
    min_confidence: float = 0.70
    train_window: int = 252
    val_window: int = 63
    step_size: int = 21
    commission: float = 0.001
    slippage: float = 0.0005


@dataclass(frozen=True)
class TrainingReport:
    """Summary of a fitted calibrated ensemble."""

    version: str
    ensemble_path: str
    base_models: list[str]
    validation_samples: int
    metrics: dict[str, float]
    calibration_status: str


@dataclass(frozen=True)
class BacktestReport:
    """Summary of a walk-forward ensemble backtest."""

    folds: int
    trade_count: int
    prediction_metrics: dict[str, float]
    trading_metrics: dict[str, float]


@dataclass(frozen=True)
class _ModelDataset:
    name: str
    X: NDArray[np.float64]
    y_dir: NDArray[np.int64]
    y_pct: NDArray[np.float64]
    keys: list[DatasetKey]
    feature_count: int


def _default_model_factory(name: str, input_size: int | None):
    return create_model(name, input_size=input_size)


def _selected_model_names(
    names: list[str],
    include_neural: bool,
) -> list[str]:
    selected = parse_model_names(names)
    if include_neural:
        for name in ["lstm", "transformer"]:
            if name not in selected:
                selected.append(name)
    return selected


def _build_pooled_dataset(
    db: Database,
    tickers: list[str],
    model_name: str,
    seq_len: int,
    min_features: int,
) -> _ModelDataset:
    per_ticker = []
    for ticker in tickers:
        if model_name in SEQUENCE_MODELS:
            X, y_dir, y_pct, keys, feature_names = build_sequence_dataset_with_keys(
                db, ticker, seq_len=seq_len, min_features=min_features,
            )
        else:
            X, y_dir, y_pct, keys, feature_names = build_tabular_dataset_with_keys(
                db, ticker, min_features=min_features,
            )
        if X.shape[0] > 0:
            per_ticker.append((X, y_dir, y_pct, keys, len(feature_names)))

    if not per_ticker:
        empty_shape = (0, seq_len, 0) if model_name in SEQUENCE_MODELS else (0, 0)
        return _ModelDataset(
            model_name,
            np.empty(empty_shape),
            np.empty(0, dtype=np.int64),
            np.empty(0),
            [],
            0,
        )

    counts = [item[4] for item in per_ticker]
    target_count = max(set(counts), key=counts.count)
    filtered = [item for item in per_ticker if item[4] == target_count]

    X = np.concatenate([item[0] for item in filtered], axis=0)
    y_dir = np.concatenate([item[1] for item in filtered], axis=0)
    y_pct = np.concatenate([item[2] for item in filtered], axis=0)
    keys = [key for item in filtered for key in item[3]]
    order = np.array(sorted(range(len(keys)), key=lambda i: keys[i]))

    return _ModelDataset(
        model_name,
        X[order],
        y_dir[order],
        y_pct[order],
        [keys[i] for i in order],
        target_count,
    )


def _split_train_validation(n_rows: int, validation_fraction: float) -> tuple[slice, slice]:
    if n_rows < 2:
        raise ValueError("Need at least two rows to split train/validation data")
    split = int(n_rows * (1.0 - validation_fraction))
    split = min(max(split, 1), n_rows - 1)
    return slice(0, split), slice(split, n_rows)


def _predictions_by_key(
    keys: list[DatasetKey],
    probs: NDArray[np.float64],
    pct: NDArray[np.float64],
) -> dict[DatasetKey, tuple[float, float]]:
    return {key: (float(prob), float(move)) for key, prob, move in zip(keys, probs, pct)}


def _labels_by_key(
    keys: list[DatasetKey],
    y_dir: NDArray[np.int64],
    y_pct: NDArray[np.float64],
) -> dict[DatasetKey, tuple[int, float]]:
    return {key: (int(direction), float(move)) for key, direction, move in zip(keys, y_dir, y_pct)}


def _align_predictions(
    prediction_maps: dict[str, dict[DatasetKey, tuple[float, float]]],
    label_map: dict[DatasetKey, tuple[int, float]],
    min_common: int,
) -> tuple[
    dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]],
    NDArray[np.int64],
    NDArray[np.float64],
    list[DatasetKey],
]:
    if len(prediction_maps) < 2:
        raise ValueError("At least two base models are required for ensemble training")

    common = set(label_map)
    for model_predictions in prediction_maps.values():
        common &= set(model_predictions)
    common_keys = sorted(common)
    if len(common_keys) < min_common:
        raise ValueError(
            f"Need at least {min_common} common validation rows, got {len(common_keys)}"
        )

    stacked: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    for model_name, model_predictions in prediction_maps.items():
        probs = np.array([model_predictions[key][0] for key in common_keys], dtype=np.float64)
        pct = np.array([model_predictions[key][1] for key in common_keys], dtype=np.float64)
        stacked[model_name] = (probs, pct)

    y_dir = np.array([label_map[key][0] for key in common_keys], dtype=np.int64)
    y_pct = np.array([label_map[key][1] for key in common_keys], dtype=np.float64)
    return stacked, y_dir, y_pct, common_keys


def _calibrated_probabilities(
    calibrator: PlattCalibrator,
    raw_probs: NDArray[np.float64],
) -> NDArray[np.float64]:
    if calibrator.is_fitted:
        return calibrator.transform(raw_probs)
    return raw_probs


def train_calibrated_ensemble(
    db: Database,
    config: EnsembleTrainConfig,
    *,
    tickers: list[str] | None = None,
    model_factory: ModelFactory = _default_model_factory,
) -> TrainingReport:
    """Train base models, a stacking combiner, and Platt calibration."""
    registry = ModelRegistry(db)
    selected = _selected_model_names(config.model_names, config.include_neural)
    tickers = tickers or db.list_tracked_stocks()
    version = config.version or date.today().isoformat()
    models_root = db.path.parent / "models"

    prediction_maps: dict[str, dict[DatasetKey, tuple[float, float]]] = {}
    label_map: dict[DatasetKey, tuple[int, float]] = {}
    trained_models: list[str] = []
    base_metrics: dict[str, dict[str, float]] = {}

    for model_name in selected:
        dataset = _build_pooled_dataset(
            db, tickers, model_name, config.seq_len, config.min_features,
        )
        if dataset.X.shape[0] < 2:
            continue

        train_slice, val_slice = _split_train_validation(
            dataset.X.shape[0], config.validation_fraction,
        )
        model = model_factory(model_name, dataset.feature_count)
        metrics = model.train(
            dataset.X[train_slice],
            dataset.y_dir[train_slice],
            dataset.y_pct[train_slice],
            dataset.X[val_slice],
            dataset.y_dir[val_slice],
            dataset.y_pct[val_slice],
        )
        probs, pct = model.predict(dataset.X[val_slice])
        val_keys = dataset.keys[val_slice]
        prediction_maps[model_name] = _predictions_by_key(val_keys, probs, pct)
        label_map.update(_labels_by_key(val_keys, dataset.y_dir[val_slice], dataset.y_pct[val_slice]))

        model_path = models_root / model_name / version
        model.save(str(model_path))
        registry.register(model_name, version, str(model_path), metrics)
        registry.activate(model_name, version)
        trained_models.append(model_name)
        base_metrics[model_name] = metrics

    stacked, y_dir, y_pct, common_keys = _align_predictions(
        prediction_maps, label_map, config.min_common_validation,
    )

    combiner = EnsembleCombiner()
    combiner.train(stacked, y_dir, y_pct)
    raw_probs, pct_pred = combiner.predict(stacked)

    calibrator = PlattCalibrator(min_confidence=config.min_confidence)
    calibrator.fit(raw_probs, y_dir)
    probs = _calibrated_probabilities(calibrator, raw_probs)
    metrics = compute_prediction_metrics(y_dir, probs, y_pct, pct_pred)
    metrics["validation_samples"] = float(len(common_keys))

    ensemble_path = models_root / "ensemble" / version
    combiner.save(str(ensemble_path))
    calibrator.save(str(ensemble_path))
    metadata = {
        "version": version,
        "base_models": trained_models,
        "seq_len": config.seq_len,
        "validation_samples": len(common_keys),
        "calibration_status": calibrator.status,
        "base_metrics": base_metrics,
        "metrics": metrics,
    }
    with open(ensemble_path / "ensemble_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    registry.register("ensemble", version, str(ensemble_path), metrics | metadata)
    registry.activate("ensemble", version)

    return TrainingReport(
        version=version,
        ensemble_path=str(ensemble_path),
        base_models=trained_models,
        validation_samples=len(common_keys),
        metrics=metrics,
        calibration_status=calibrator.status,
    )


def _fit_fold_ensemble(
    datasets: dict[str, _ModelDataset],
    common_keys: list[DatasetKey],
    train_range: tuple[int, int],
    val_range: tuple[int, int],
    config: EnsembleBacktestConfig,
    model_factory: ModelFactory,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], list[DatasetKey]]:
    train_start, train_end = train_range
    val_start, val_end = val_range
    if train_end - train_start < 3:
        raise ValueError("Fold train window is too small")

    meta_split = train_start + int((train_end - train_start) * (1.0 - config.validation_fraction))
    meta_split = min(max(meta_split, train_start + 1), train_end - 1)
    meta_keys = common_keys[meta_split:train_end]
    val_keys = common_keys[val_start:val_end]

    meta_prediction_maps: dict[str, dict[DatasetKey, tuple[float, float]]] = {}
    val_prediction_maps: dict[str, dict[DatasetKey, tuple[float, float]]] = {}
    label_map: dict[DatasetKey, tuple[int, float]] = {}

    for model_name, dataset in datasets.items():
        index_by_key = {key: i for i, key in enumerate(dataset.keys)}
        train_idx = [index_by_key[key] for key in common_keys[train_start:meta_split]]
        meta_idx = [index_by_key[key] for key in meta_keys]
        val_idx = [index_by_key[key] for key in val_keys]

        model = model_factory(model_name, dataset.feature_count)
        model.train(
            dataset.X[train_idx],
            dataset.y_dir[train_idx],
            dataset.y_pct[train_idx],
        )
        meta_probs, meta_pct = model.predict(dataset.X[meta_idx])
        val_probs, val_pct = model.predict(dataset.X[val_idx])
        meta_prediction_maps[model_name] = _predictions_by_key(meta_keys, meta_probs, meta_pct)
        val_prediction_maps[model_name] = _predictions_by_key(val_keys, val_probs, val_pct)
        label_map.update(_labels_by_key(
            meta_keys + val_keys,
            np.concatenate([dataset.y_dir[meta_idx], dataset.y_dir[val_idx]]),
            np.concatenate([dataset.y_pct[meta_idx], dataset.y_pct[val_idx]]),
        ))

    meta_stacked, meta_y_dir, meta_y_pct, _ = _align_predictions(
        meta_prediction_maps, label_map, min_common=1,
    )
    combiner = EnsembleCombiner()
    combiner.train(meta_stacked, meta_y_dir, meta_y_pct)
    meta_raw_probs, _ = combiner.predict(meta_stacked)
    calibrator = PlattCalibrator(min_confidence=config.min_confidence)
    calibrator.fit(meta_raw_probs, meta_y_dir)

    val_stacked, val_y_dir, val_y_pct, aligned_val_keys = _align_predictions(
        val_prediction_maps, label_map, min_common=1,
    )
    raw_probs, pct_pred = combiner.predict(val_stacked)
    probs = _calibrated_probabilities(calibrator, raw_probs)
    return val_y_dir, val_y_pct, probs, pct_pred, aligned_val_keys


def _portfolio_returns(
    keys: list[DatasetKey],
    probs: NDArray[np.float64],
    actual_pct: NDArray[np.float64],
    backtest: WalkForwardBacktest,
    min_confidence: float,
) -> tuple[NDArray[np.float64], int]:
    by_date: dict[str, list[float]] = {}
    trade_count = 0
    for key, prob, move in zip(keys, probs, actual_pct):
        if prob >= min_confidence:
            gross = move
            trade_count += 1
        elif prob <= 1.0 - min_confidence:
            gross = -move
            trade_count += 1
        else:
            gross = 0.0
        net = backtest.apply_costs(float(gross)) if gross != 0.0 else 0.0
        by_date.setdefault(key.date, []).append(net)

    daily = np.array([np.mean(values) for _, values in sorted(by_date.items())], dtype=np.float64)
    return daily, trade_count


def run_walk_forward_backtest(
    db: Database,
    config: EnsembleBacktestConfig,
    *,
    tickers: list[str] | None = None,
    model_factory: ModelFactory = _default_model_factory,
) -> BacktestReport:
    """Run leakage-aware walk-forward evaluation for the calibrated ensemble."""
    selected = _selected_model_names(config.model_names, config.include_neural)
    tickers = tickers or db.list_tracked_stocks()
    datasets = {
        name: _build_pooled_dataset(db, tickers, name, config.seq_len, config.min_features)
        for name in selected
    }
    datasets = {name: dataset for name, dataset in datasets.items() if dataset.X.shape[0] > 0}
    if len(datasets) < 2:
        raise ValueError("At least two datasets are required for ensemble backtesting")

    common = set(next(iter(datasets.values())).keys)
    for dataset in datasets.values():
        common &= set(dataset.keys)
    common_keys = sorted(common)

    backtest = WalkForwardBacktest(
        train_window=config.train_window,
        val_window=config.val_window,
        step_size=config.step_size,
        commission=config.commission,
        slippage=config.slippage,
    )
    folds = backtest.generate_folds(len(common_keys))
    if not folds:
        return BacktestReport(
            folds=0,
            trade_count=0,
            prediction_metrics={},
            trading_metrics=compute_trading_metrics(np.empty(0)),
        )

    all_y_dir = []
    all_y_pct = []
    all_probs = []
    all_pct_pred = []
    all_daily_returns = []
    trade_count = 0

    for train_start, train_end, val_start, val_end in folds:
        y_dir, y_pct, probs, pct_pred, val_keys = _fit_fold_ensemble(
            datasets,
            common_keys,
            (train_start, train_end),
            (val_start, val_end),
            config,
            model_factory,
        )
        daily_returns, fold_trades = _portfolio_returns(
            val_keys, probs, y_pct, backtest, config.min_confidence,
        )
        all_y_dir.append(y_dir)
        all_y_pct.append(y_pct)
        all_probs.append(probs)
        all_pct_pred.append(pct_pred)
        all_daily_returns.append(daily_returns)
        trade_count += fold_trades

    y_dir_all = np.concatenate(all_y_dir)
    y_pct_all = np.concatenate(all_y_pct)
    probs_all = np.concatenate(all_probs)
    pct_pred_all = np.concatenate(all_pct_pred)
    daily_returns_all = np.concatenate(all_daily_returns)

    return BacktestReport(
        folds=len(folds),
        trade_count=trade_count,
        prediction_metrics=compute_prediction_metrics(
            y_dir_all, probs_all, y_pct_all, pct_pred_all,
        ),
        trading_metrics=compute_trading_metrics(daily_returns_all),
    )

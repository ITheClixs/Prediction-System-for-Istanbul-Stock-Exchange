"""Configuration management — loads and validates config.toml."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.toml"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "bist.db"


@dataclass(frozen=True)
class DataConfig:
    tcmb_api_key: str = ""
    fetch_retries: int = 3
    rate_limit_delay: float = 1.0


@dataclass(frozen=True)
class SignalsConfig:
    min_confidence: float = 0.70
    lookback_days: int = 30


@dataclass(frozen=True)
class ModelsConfig:
    retrain_interval: str = "monthly"
    ensemble_weights: str = "learned"
    active_models: str = "xgboost,lightgbm"
    include_neural: bool = False
    seq_len: int = 30
    validation_fraction: float = 0.2


@dataclass(frozen=True)
class QuantConfig:
    hmm_states: int = 3
    kelly_fraction: float = 0.25
    hurst_window: int = 252


@dataclass(frozen=True)
class BacktestConfig:
    commission: float = 0.001
    slippage: float = 0.0005


@dataclass(frozen=True)
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    signals: SignalsConfig = field(default_factory=SignalsConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    quant: QuantConfig = field(default_factory=QuantConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    db_path: Path = DEFAULT_DB_PATH


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load configuration from a TOML file. Returns defaults if file missing."""
    if not path.exists():
        return Config()

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    return Config(
        data=DataConfig(**raw.get("data", {})),
        signals=SignalsConfig(**raw.get("signals", {})),
        models=ModelsConfig(**raw.get("models", {})),
        quant=QuantConfig(**raw.get("quant", {})),
        backtest=BacktestConfig(**raw.get("backtest", {})),
    )

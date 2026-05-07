"""Factory helpers for prediction model construction."""

from __future__ import annotations

from bist_predict.models.lightgbm_model import LightGBMModel
from bist_predict.models.lstm_model import LSTMModel
from bist_predict.models.transformer_model import TransformerModel
from bist_predict.models.xgboost_model import XGBoostModel


def create_model(name: str, *, input_size: int | None = None):
    """Create a prediction model by registry name."""
    normalized = name.strip().lower()
    if normalized == "xgboost":
        return XGBoostModel()
    if normalized == "lightgbm":
        return LightGBMModel()
    if normalized == "lstm":
        return LSTMModel(input_size=input_size or 80)
    if normalized == "transformer":
        return TransformerModel(input_size=input_size or 80)
    raise ValueError(f"Unknown model name: {name}")


def parse_model_names(raw: str | list[str] | tuple[str, ...]) -> list[str]:
    """Normalize comma-delimited model names while preserving order."""
    if isinstance(raw, str):
        values = raw.split(",")
    else:
        values = list(raw)
    result: list[str] = []
    for value in values:
        name = value.strip().lower()
        if name and name not in result:
            result.append(name)
    return result

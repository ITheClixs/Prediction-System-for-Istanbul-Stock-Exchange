"""Tests for model factory helpers."""

from __future__ import annotations

import pytest

from bist_predict.models.factory import create_model, parse_model_names


def test_create_model_by_name() -> None:
    assert create_model("xgboost").name == "xgboost"
    assert create_model("lightgbm").name == "lightgbm"
    assert create_model("lstm", input_size=7).n_features == 7
    assert create_model("transformer", input_size=9).n_features == 9


def test_create_model_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown model name"):
        create_model("catboost")


def test_parse_model_names_deduplicates() -> None:
    assert parse_model_names("XGBoost, lightgbm, xgboost") == ["xgboost", "lightgbm"]

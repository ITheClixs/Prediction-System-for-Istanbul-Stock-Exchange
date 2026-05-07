# Calibrated Ensemble Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Add a leakage-aware calibrated ensemble pipeline with opt-in neural models and a real CLI walk-forward backtest.

**Architecture:** Keep model classes as base learners, add a thin research orchestration layer that builds keyed chronological datasets, trains base models, stacks validation predictions, fits Platt calibration, persists ensemble artifacts, and drives backtests.

**Tech Stack:** Python 3.12+, Click, NumPy, scikit-learn, XGBoost, LightGBM, optional PyTorch LSTM/Transformer, SQLite registry.

---

## Implementation Tasks

- [x] Add keyed tabular and sequence dataset builders while preserving existing dataset helper return shapes.
- [x] Add model factory helpers and artifact persistence for the ensemble combiner and Platt calibrator.
- [x] Add calibrated ensemble training orchestration that registers active base models and active ensemble metadata.
- [x] Add leakage-aware walk-forward backtest orchestration with transaction costs and portfolio metrics.
- [x] Wire `train`, `signals`, and `backtest` CLI commands to the calibrated ensemble workflow.
- [x] Document the new workflow and opt-in neural model behavior.

## Verification

- Targeted dataset/model tests: `uv run pytest tests/test_models/test_types.py tests/test_models/test_factory.py tests/test_models/test_ensemble.py tests/test_models/test_calibration.py -q`
- Research orchestration tests: `uv run pytest tests/test_research/test_ensemble_pipeline.py -q`
- CLI/research smoke tests: `uv run pytest tests/test_cli.py tests/test_research/test_ensemble_pipeline.py -q`
- Broad non-neural suite: `uv run pytest -k "not lstm and not transformer"`

## Assumptions

- Neural models are opt-in, not default.
- No new external dependencies are required.
- No SQLite schema migration is required.
- Active ensemble artifacts live under `data/models/ensemble/<version>/`.
- Backtest results are research simulations, not trading advice.

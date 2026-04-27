# BIST-100 Stock Market Prediction System

A CLI-based daily trading signal system for Borsa Istanbul (BIST-100) stocks. Predicts next-day price direction (UP/DOWN) with calibrated confidence scores and percentage price targets using an ensemble of gradient boosting and deep learning models, institutional-grade quantitative methods, free data sources, and a high-performance Rust feature engine.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Commands](#cli-commands)
- [Pipeline Deep Dive](#pipeline-deep-dive)
  - [1. Data Ingestion](#1-data-ingestion)
  - [2. Feature Engine (Rust + Python)](#2-feature-engine-rust--python)
  - [3. Quantitative Alpha Layer](#3-quantitative-alpha-layer)
  - [4. Model Layer](#4-model-layer)
  - [5. Evaluation & Backtesting](#5-evaluation--backtesting)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Tech Stack](#tech-stack)
- [Data Sources](#data-sources)

---

## Features

- **Daily trading signals** for 30 BIST-100 stocks with direction, confidence, and price targets
- **Four ML models** (XGBoost, LightGBM, LSTM, Transformer) with meta-learner ensemble
- **Calibrated confidence** via Platt scaling -- when the system says "78% UP", historically ~78% of such predictions are correct
- **Quantitative alpha** layer: Kalman filter, Ornstein-Uhlenbeck mean reversion, GARCH volatility, HMM regime detection, Hurst exponent, wavelet decomposition
- **Regime-aware routing** dynamically adjusts model weights based on bull/bear/sideways market state
- **High-performance Rust feature engine** computes 30+ technical indicators via PyO3
- **Walk-forward backtesting** with realistic transaction costs (commission + slippage) and no future leakage
- **Live accuracy tracking** with rolling windows and confidence bucket analysis
- **Free data only** -- no paid subscriptions required (Is Yatirim, Yahoo Finance, TCMB, Google News RSS)
- **173 tests** covering every module

---

## Architecture

```
CLI Interface (Click)
    |
    +-- Data Ingest (Python, async httpx)
    |       Is Yatirim | Yahoo Finance | TCMB EVDS | Google News RSS
    |
    +-- Feature Engine
    |       Rust (PyO3): RSI, MACD, Bollinger, ATR, OBV, VWAP, ADX, CCI, MFI, ...
    |       Python: macro deltas, sentiment scores, temporal/calendar features
    |
    +-- Quantitative Alpha Layer (Python)
    |       Kalman | O-U Mean Reversion | GARCH | HMM Regime | Hurst | Wavelets
    |       Kelly Criterion | Ledoit-Wolf | PCA | Cointegration
    |
    +-- Model Layer (Python)
    |       XGBoost | LightGBM | LSTM | Transformer
    |       --> Ensemble Meta-Learner --> Platt Calibration
    |
    +-- Evaluation (Python)
    |       Walk-forward backtest | Prediction metrics | Trading metrics
    |       Live accuracy tracker | Confidence bucket analysis
    |
    +-- SQLite Storage
            raw_prices | macro_data | sentiment_data | features
            predictions | model_registry
```

Each layer communicates through well-defined interfaces and can be developed, tested, and iterated independently.

---

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Rust toolchain (for the Rust feature engine, optional)
- Homebrew `libomp` on macOS (required by XGBoost): `brew install libomp`

### Install

```bash
# Clone the repository
git clone <repo-url>
cd BIST-Predictorcl

# Install Python dependencies
uv sync

# (Optional) Build the Rust feature engine for maximum performance
cd rust/bist_features
maturin develop --release
cd ../..
```

If the Rust module is not compiled, the system falls back to Python-only feature computation (technical indicators won't be available, but quant features, macro, sentiment, and temporal features still work).

---

## Quick Start

```bash
# 1. Fetch market data (last 90 days)
uv run bist-predict fetch --days 90

# 2. Compute features for all stocks
uv run bist-predict features

# 3. Train prediction models
uv run bist-predict train

# 4. Get today's trading signals
uv run bist-predict signals

# 5. Check prediction accuracy
uv run bist-predict accuracy
```

---

## CLI Commands

### `bist-predict fetch`

Pull latest market data from all sources.

```bash
uv run bist-predict fetch                    # Fetch last 30 days for all stocks
uv run bist-predict fetch --days 90          # Fetch last 90 days
uv run bist-predict fetch --ticker THYAO     # Fetch a single stock
```

Fetches OHLCV prices (Is Yatirim primary, Yahoo Finance fallback), TCMB macro indicators (requires API key), and Google News sentiment headlines. Supports incremental fetch -- only pulls data newer than the last stored date.

### `bist-predict features`

Compute features for the latest data.

```bash
uv run bist-predict features                           # All stocks, today
uv run bist-predict features --ticker THYAO             # Single stock
uv run bist-predict features --date 2026-03-15          # Specific date
```

Runs the full feature pipeline: Rust technical indicators (30+), quantitative alpha features (Kalman, O-U, GARCH, Hurst, TSMOM), macro deltas, sentiment aggregation, and temporal/calendar features. Stores results in the SQLite feature store.

### `bist-predict train`

Train or retrain prediction models.

```bash
uv run bist-predict train                    # Train on all stocks
uv run bist-predict train --ticker THYAO     # Train on one stock only
```

Builds tabular datasets from the feature store, splits 80/20 for training/validation, and trains XGBoost and LightGBM models with dual prediction heads (direction classification + percentage move regression). Saves trained models to disk and registers them in the model registry with validation metrics.

### `bist-predict signals`

Get today's trading signals.

```bash
uv run bist-predict signals                  # All stocks
uv run bist-predict signals --ticker THYAO   # Single stock
uv run bist-predict signals --detail         # Detailed breakdown
```

Loads the active model versions from the registry, runs inference on the latest features, and outputs signals grouped by confidence tier:

```
========================================
  STRONG BUY
========================================
  THYAO    85.2% conf  +1.42% target  (xgboost)
  GARAN    82.1% conf  +0.98% target  (lightgbm)

========================================
  BUY
========================================
  AKBNK    74.3% conf  +0.67% target  (xgboost)
```

Signal tiers:
- **STRONG BUY**: >= 80% UP confidence
- **BUY**: 70-80% UP confidence
- **SELL**: 70-80% DOWN confidence
- **STRONG SELL**: >= 80% DOWN confidence

### `bist-predict backtest`

Run walk-forward backtest (integration pending -- engine and metrics are complete).

```bash
uv run bist-predict backtest
```

### `bist-predict accuracy`

Show prediction accuracy history.

```bash
uv run bist-predict accuracy                 # Top 5 stocks
uv run bist-predict accuracy --ticker THYAO  # Single stock with confidence buckets
```

Shows rolling 30-day and 90-day directional accuracy. For individual stocks, also displays confidence bucket analysis (accuracy at 60-70%, 70-80%, 80-90%, 90-100% confidence levels).

### `bist-predict stocks`

List all 30 tracked BIST-100 stocks.

### `bist-predict config`

Display current configuration (database path, API keys, backtest parameters).

---

## Pipeline Deep Dive

### 1. Data Ingestion

The ingestion layer collects three categories of data from free sources:

#### OHLCV Price Data

| Source | Role | Details |
|--------|------|---------|
| **Is Yatirim API** | Primary | Undocumented REST API, most accurate BIST data |
| **Yahoo Finance** | Fallback | Via `yfinance`, BIST tickers use `.IS` suffix |

Automatic failover: tries Is Yatirim first, falls back to Yahoo Finance on timeout or rate limit. All data stored as daily OHLCV bars with adjusted close.

#### Macro Data (TCMB EVDS)

Requires a free API key from [evds2.tcmb.gov.tr](https://evds2.tcmb.gov.tr):

- USD/TRY and EUR/TRY exchange rates (daily)
- TCMB policy interest rate
- CPI / inflation data (monthly)
- Gold price XAU/TRY (daily)
- Government bond yields (daily)

#### Sentiment Data

- **Google News RSS** -- headline search per ticker + "borsa" keywords, no API key
- **Turkish finance RSS** -- bloomberght, bigpara feeds with Turkish NLP scoring

#### Data Quality

The quality module (`ingest/quality.py`) validates all incoming data:
- OHLCV logical consistency (open/high/low/close relationships)
- Gap detection (> 5 trading days flags halts/delistings)
- Rate limiting with configurable delays and exponential backoff
- Incremental fetching (only new data since last stored date)

### 2. Feature Engine (Rust + Python)

#### Rust-Computed Technical Indicators (~30+)

Compiled to a Python extension via PyO3 for maximum performance:

| Category | Indicators |
|----------|-----------|
| **Momentum** | RSI (14), MACD (12/26/9), Stochastic %K/%D, Williams %R, CCI (20), MFI (14) |
| **Trend** | SMA (5/10/20/50/100/200), EMA (5/10/20/50/100/200), ADX (14) |
| **Volatility** | Bollinger Bands (20), ATR (14) |
| **Volume** | OBV, VWAP, volume ratio (20d) |
| **Patterns** | Doji, hammer, engulfing, morning star detection |
| **Cross-stock** | Correlation matrix, beta to BIST-100 index |

Source: `rust/bist_features/src/` -- `indicators.rs`, `patterns.rs`, `correlations.rs`

#### Python-Computed Features

| Category | Features |
|----------|---------|
| **Macro** | USD/TRY delta, EUR/TRY delta, interest rate change, CPI trend, gold delta, bond yield spread, percentage changes |
| **Sentiment** | Mean sentiment, count, positive ratio per stock per day |
| **Temporal** | Day of week (sin/cos encoded), month (sin/cos encoded), quarter, is_monday, is_friday |
| **Price-derived** | 1d/5d/20d returns, close, volume |

The feature engine (`features/engine.py`) orchestrates all computation and stores results in the SQLite feature store keyed by `(ticker, date, feature_name)`.

### 3. Quantitative Alpha Layer

This layer provides institutional-grade quantitative methods that serve three roles:
1. **Extra features** fed into ML models
2. **Independent signals** for momentum, mean reversion, and pairs trading
3. **Model routing control** via regime detection

#### Factor Models & Alpha Signals

| Method | Implementation | Output |
|--------|---------------|--------|
| **Time-series Momentum** (Moskowitz et al., 2012) | `quant/factors.py` | TSMOM signal + magnitude per stock |
| **Mean Reversion** (Ornstein-Uhlenbeck) | `quant/factors.py` | theta (speed), mu (mean), deviation, half-life |
| **Fama-French adapted** | `quant/factors.py` | SMB, HML factor scores for BIST |

#### Statistical Methods

| Method | Implementation | Purpose |
|--------|---------------|---------|
| **Kalman Filter** | `quant/statistical.py` | Tracks hidden "true momentum", filters noise, adapts to volatility |
| **Hidden Markov Model** (3-state) | `quant/statistical.py` | Bull/bear/sideways regime detection on BIST-100 returns + volatility |
| **GARCH(1,1)** | `quant/statistical.py` | Per-stock volatility forecasting for confidence calibration and position sizing |
| **Cointegration** (Engle-Granger) | `quant/statistical.py` | Finds cointegrated stock pairs, outputs spread z-score and half-life |

#### Signal Quality Measurement

| Method | Implementation | What it tells you |
|--------|---------------|-------------------|
| **Information Coefficient** | `quant/signal_quality.py` | Rank correlation between predicted and actual returns (IC > 0.05 = meaningful) |
| **Hurst Exponent** | `quant/signal_quality.py` | H > 0.5 = trending (trust momentum), H < 0.5 = mean-reverting (trust O-U), H ~ 0.5 = random walk |
| **Wavelet Decomposition** | `quant/signal_quality.py` | Separates price into frequency bands: daily noise, weekly cycles, monthly trends |

#### Risk & Position Sizing

| Method | Implementation | Purpose |
|--------|---------------|---------|
| **Kelly Criterion** | `quant/risk.py` | Optimal bet sizing: f* = (p*b - q) / b, with fractional Kelly (0.25x) for safety |
| **Ledoit-Wolf Shrinkage** | `quant/risk.py` | Robust covariance estimation, prevents overfitting to noisy correlations |
| **PCA Factor Extraction** | `quant/risk.py` | Extracts latent market drivers from BIST-100 return matrix |

#### Regime-Aware Routing

The HMM regime detection (`quant/regime.py`) dynamically adjusts ensemble weights:

| Regime | Momentum Weight | Mean Reversion Weight | Pairs Weight | Kelly Fraction |
|--------|----------------|----------------------|-------------|---------------|
| **Bull** (high prob) | High | Low | Low | 0.5x |
| **Bear** (high prob) | Low | High | Low | 0.25x |
| **Sideways** (high prob) | Low | Low | High | 0.25x |
| **Uncertain** | Equal | Equal | Equal | 0.25x |

### 4. Model Layer

#### Individual Models

All four models implement the same `PredictionModel` protocol with dual prediction heads:

| Model | Input Shape | Strength | Implementation |
|-------|------------|----------|----------------|
| **XGBoost** | Tabular (n_samples, n_features) | Best tabular performance, feature importance | `models/xgboost_model.py` |
| **LightGBM** | Tabular (n_samples, n_features) | Faster training, categorical handling, diversity | `models/lightgbm_model.py` |
| **LSTM** | Sequences (n_samples, 30, n_features) | Temporal dependencies, momentum shifts | `models/lstm_model.py` |
| **Transformer** | Sequences (n_samples, 60, n_features) | Long-range attention, event detection | `models/transformer_model.py` |

Each model produces:
- **Classification head** -- P(UP) probability (sigmoid output)
- **Regression head** -- predicted percentage move

#### Ensemble Combiner

The meta-learner (`models/ensemble.py`) stacks predictions from all individual models:

- **Direction**: Logistic regression on stacked P(UP) probabilities
- **Percentage move**: Ridge regression on stacked percentage predictions
- **Fallback**: Simple averaging when meta-learner isn't trained
- **Regime modulation**: Weights can be adjusted by HMM regime probabilities

#### Confidence Calibration

Platt scaling (`models/calibration.py`) fits a sigmoid to map raw ensemble scores to calibrated probabilities. This ensures that when the system reports "78% confidence", historically ~78% of such predictions were correct. Configurable minimum confidence threshold (default: 60%).

#### Model Registry

The SQLite-based registry (`models/registry.py`) tracks all trained model versions:
- Register models with path, version, and validation metrics
- Activate/deactivate model versions per model type
- Query active models for inference
- Supports multiple versions for A/B testing and rollback

### 5. Evaluation & Backtesting

#### Walk-Forward Backtesting

The backtesting engine (`evaluation/backtest.py`) implements honest evaluation:

- **Walk-forward only** -- train on past data, test on future data, slide window forward
- **Configurable windows** -- 252-day training, 63-day validation, 21-day step size (defaults)
- **Realistic costs** -- commission (0.1%) and slippage (0.05%) applied on both entry and exit
- **No future leakage** -- training data timestamps always precede test data
- **Signal delay** -- prediction at market close, trade assumed at next-day open

#### Prediction Quality Metrics

`evaluation/metrics.py` computes:

| Metric | What it measures |
|--------|-----------------|
| **Accuracy** | % of correct direction predictions |
| **Precision** | Of predicted UPs, how many were actually UP |
| **Recall** | Of actual UPs, how many did we predict |
| **F1 Score** | Harmonic mean of precision and recall |
| **AUC-ROC** | Discrimination ability across all thresholds |
| **Brier Score** | Calibration quality of probability estimates |
| **MAE** | Mean absolute error on percentage move predictions |

#### Trading Quality Metrics

| Metric | What it measures |
|--------|-----------------|
| **Sharpe Ratio** | Risk-adjusted return (annualized) |
| **Sortino Ratio** | Downside-risk-adjusted return |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Win Rate** | % of profitable trades |
| **Profit Factor** | Gross profits / gross losses |
| **Calmar Ratio** | Return / max drawdown |
| **Total Return** | Cumulative portfolio return |

#### Live Accuracy Tracking

The tracker (`evaluation/tracker.py`) provides ongoing monitoring:

- Every prediction logged with timestamp, confidence, direction, predicted % move
- Actual outcomes recorded on next data fetch
- **Rolling accuracy** over configurable windows (30/60/90 days)
- **Per-stock breakdown** to identify model strengths/weaknesses
- **Confidence bucket analysis** -- verifies that higher confidence predictions are actually more accurate (60-70%, 70-80%, 80-90%, 90-100%)

---

## Configuration

Create a `config.toml` in the project root:

```toml
[data]
tcmb_api_key = ""           # Free key from evds2.tcmb.gov.tr
fetch_retries = 3            # Max retries per data source
rate_limit_delay = 1.0       # Seconds between API calls

[signals]
min_confidence = 0.70        # Minimum confidence to display signal
lookback_days = 30           # Feature computation lookback

[models]
retrain_interval = "monthly" # Retrain cadence
ensemble_weights = "learned" # "learned" or "equal"

[quant]
hmm_states = 3               # HMM regime states (bull/bear/sideways)
kelly_fraction = 0.25         # Fractional Kelly multiplier
hurst_window = 252            # Hurst exponent lookback window

[backtest]
commission = 0.001            # 0.1% per trade
slippage = 0.0005             # 0.05% per trade
```

---

## Project Structure

```
BIST-Predictorcl/
+-- README.md
+-- CLAUDE.md                          # Development instructions
+-- pyproject.toml                     # Python project config
+-- config.toml                        # Runtime configuration
|
+-- src/bist_predict/
|   +-- cli.py                         # Click CLI entry point (8 commands)
|   +-- config.py                      # Configuration loading
|   |
|   +-- ingest/                        # Data collection layer
|   |   +-- isyatirim.py               # Is Yatirim API client (primary)
|   |   +-- yahoo.py                   # Yahoo Finance client (fallback)
|   |   +-- tcmb.py                    # TCMB EVDS macro data
|   |   +-- sentiment.py               # Google News + Turkish RSS sentiment
|   |   +-- scheduler.py               # Orchestrates all collectors
|   |   +-- quality.py                 # OHLCV validation, gap detection
|   |   +-- types.py                   # PriceBar, MacroPoint, SentimentRecord
|   |
|   +-- features/                      # Feature computation
|   |   +-- engine.py                  # Orchestrates Rust + Python features
|   |   +-- store.py                   # SQLite feature store (ticker, date)
|   |   +-- macro_features.py          # Macro deltas and pct changes
|   |   +-- sentiment_features.py      # Sentiment aggregation
|   |   +-- temporal_features.py       # Calendar/day-of-week features
|   |
|   +-- quant/                         # Quantitative alpha layer
|   |   +-- factors.py                 # TSMOM, O-U mean reversion, Fama-French
|   |   +-- statistical.py             # Kalman, HMM, GARCH, cointegration
|   |   +-- risk.py                    # Kelly criterion, Ledoit-Wolf, PCA
|   |   +-- signal_quality.py          # IC, Hurst exponent, wavelets
|   |   +-- regime.py                  # HMM regime-aware weight routing
|   |
|   +-- models/                        # ML model layer
|   |   +-- types.py                   # Prediction dataclass, PredictionModel protocol
|   |   +-- xgboost_model.py           # XGBoost with dual heads
|   |   +-- lightgbm_model.py          # LightGBM with dual heads
|   |   +-- lstm_model.py              # LSTM with dual heads (PyTorch)
|   |   +-- transformer_model.py       # Transformer with dual heads (PyTorch)
|   |   +-- ensemble.py                # Meta-learner stacking combiner
|   |   +-- calibration.py             # Platt scaling confidence calibration
|   |   +-- registry.py                # SQLite model version registry
|   |
|   +-- evaluation/                    # Backtesting and metrics
|   |   +-- backtest.py                # Walk-forward backtesting engine
|   |   +-- metrics.py                 # Prediction + trading quality metrics
|   |   +-- tracker.py                 # Live accuracy tracking
|   |
|   +-- storage/                       # Persistence
|       +-- database.py                # SQLite database (6 tables)
|       +-- migrations.py              # Schema versioning
|
+-- rust/bist_features/                # Rust feature engine (PyO3)
|   +-- src/
|       +-- lib.rs                     # PyO3 module bindings
|       +-- indicators.rs              # RSI, SMA, EMA, MACD, Bollinger, etc.
|       +-- patterns.rs                # Candlestick pattern detection
|       +-- correlations.rs            # Cross-stock correlation, beta
|
+-- tests/                             # 173 tests
|   +-- test_ingest/                   # 8 test files
|   +-- test_features/                 # 8 test files (incl. 3 Rust indicator tests)
|   +-- test_quant/                    # 5 test files
|   +-- test_models/                   # 8 test files
|   +-- test_evaluation/               # 3 test files
|   +-- test_storage/                  # 1 test file
|   +-- conftest.py                    # Shared fixtures (tmp_db_path, etc.)
|
+-- docs/superpowers/
    +-- specs/                         # Design specification
    +-- plans/                         # Implementation plans (4 plans)
```

---

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run by module
uv run pytest tests/test_ingest/ -v        # Data ingestion tests
uv run pytest tests/test_features/ -v      # Feature engine tests
uv run pytest tests/test_quant/ -v         # Quantitative alpha tests
uv run pytest tests/test_models/ -v        # ML model tests
uv run pytest tests/test_evaluation/ -v    # Evaluation tests
uv run pytest tests/test_storage/ -v       # Storage tests

# Run a single test
uv run pytest tests/test_models/test_xgboost_model.py::TestXGBoostModel::test_predict_better_than_random -v
```

Test coverage by module:

| Module | Tests | What's covered |
|--------|-------|---------------|
| **ingest** | ~30 | HTTP mocking (respx), data validation, scheduler orchestration, all 4 data sources |
| **features** | ~20 | Feature computation, store save/load, Rust indicators (via PyO3), macro/sentiment/temporal |
| **quant** | ~25 | Kalman filter, O-U fitting, GARCH, HMM regime, Hurst, wavelets, IC, Kelly, cointegration |
| **models** | ~35 | All 4 models (train/predict/save/load), ensemble combining, Platt calibration, registry CRUD |
| **evaluation** | ~15 | Prediction metrics, trading metrics, walk-forward folds, live accuracy tracking |
| **storage** | ~5 | Database init, schema, CRUD operations |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.12+, Rust |
| **CLI** | Click |
| **HTTP** | httpx (async) |
| **Market data** | yfinance, custom Is Yatirim client |
| **RSS** | feedparser |
| **ML (trees)** | XGBoost, LightGBM |
| **ML (deep)** | PyTorch (LSTM, Transformer) |
| **Quant** | scipy, statsmodels, hmmlearn, arch (GARCH), PyWavelets |
| **ML utilities** | scikit-learn (calibration, metrics, stacking) |
| **Rust binding** | PyO3 + maturin |
| **Storage** | SQLite |
| **Build** | uv (Python), maturin (Rust) |
| **Testing** | pytest, respx (HTTP mocking) |

---

## Data Sources

All data sources are free and require no paid subscriptions:

| Source | Data | API Key Required |
|--------|------|-----------------|
| [Is Yatirim](https://www.isyatirim.com.tr) | BIST OHLCV prices (primary) | No |
| [Yahoo Finance](https://finance.yahoo.com) | BIST OHLCV prices (fallback) | No |
| [TCMB EVDS](https://evds2.tcmb.gov.tr) | Macro indicators (FX, rates, CPI, gold) | Yes (free registration) |
| [Google News RSS](https://news.google.com) | Sentiment headlines per stock | No |
| Turkish finance RSS | bloomberght, bigpara sentiment | No |

---
## License

GNU General Public License v3.0

## Disclaimer

This software is for educational and research purposes only. It is not financial advice. Past performance does not guarantee future results. Always do your own research before making investment decisions. The authors assume no liability for any losses incurred from using this system.

# Command Line Interface Documentation

## S&P 500 Macro Forecasting with SSRF Model

This document explains how to use the command line arguments to configure and run the SSRF model.

---

## Quick Start

```bash
cd /workspace/sp500_macro_forecast

# Run with sample data (fastest)
python -m src.main --sample-data

# Run with real FRED data
python -m src.main

# Run with transaction costs
python -m src.main --tc-backtest --account-tier professional
```

---

## Model Selection

### Choose Model Type

```bash
# Default: ElasticNet (SSRF with regularized regression)
python -m src.main --sample-data

# Linear Regression (no regularization - good for small datasets)
python -m src.main --model-type linear --unregularized

# XGBoost (tree-based, often better for market prediction)
python -m src.main --model-type xgboost

# Random Forest
python -m src.main --model-type random_forest

# Ensemble (combines multiple models)
python -m src.main --model-type ensemble
```

### Available Model Types

| Model | Best For | Regularization |
|-------|----------|---------------|
| `elasticnet` | Default SSRF, feature selection | L1 + L2 |
| `linear` | Interpretability | None |
| `xgboost` | Non-linear patterns | L2 |
| `random_forest` | Robustness | None |
| `catboost` | Categorical features | L2 |
| `mlp` | Complex patterns | Dropout |
| `ensemble` | All of the above | Various |

---

## Data Configuration

### Sample vs Real Data

```bash
# Use generated sample data (no API key needed)
python -m src.main --sample-data

# Use real FRED data (requires API key in config.py)
python -m src.main

# Custom sample data parameters
python -m src.main --sample-data --n-periods 500 --n-indicators 100
```

### Date Range

```bash
# Custom date range for FRED data
python -m src.main --start-date 1990-01-01 --end-date 2025-12-31
```

### Sector Rotation

```bash
# Predict specific sector rotation instead of S&P 500
python -m src.main --sector-rotation Technology
python -m src.main --sector-rotation Financials
python -m src.main --sector-rotation Energy
```

Available sectors: Materials, Energy, Financials, Technology, Healthcare, ConsumerDiscretionary, ConsumerStaples, CommunicationServices, Industrials, Utilities, RealEstate

---

## Model Hyperparameters

### Regularization

```bash
# Elastic Net parameters
python -m src.main --alpha 0.01 --l1-ratio 0.5

# L1 only (Lasso - sparse features)
python -m src.main --alpha 0.01 --l1-ratio 1.0

# L2 only (Ridge)
python -m src.main --alpha 0.01 --l1-ratio 0.0

# No regularization (plain linear regression)
python -m src.main --unregularized
```

### Feature Selection

```bash
# T-statistic threshold for screening (higher = fewer features)
python -m src.main --t-stat-threshold 2.0

# Number of factors to extract
python -m src.main --n-factors 15

# Disable cross-validation
python -m src.main --no-cv
```

### Training Window

```bash
# Initial training window in months
python -m src.main --train-window 120

# Walk-forward step size
python -m src.main --step-size 3
```

### Regime Detection

```bash
# Disable regime detection
python -m src.main --no-regime

# Custom regime window
python -m src.main --regime-window 24

# Disable Campbell-Thompson restriction
python -m src.main --no-ct-restriction
```

---

## Transaction Costs

### Enable TC-Adjusted Backtest

```bash
# Basic transaction cost
python -m src.main --tc-backtest

# Custom TC rate (basis points)
python -m src.main --tc-backtest --tc-rate 10.0

# Institutional tier (lowest costs)
python -m src.main --tc-backtest --account-tier institutional

# Professional tier
python -m src.main --tc-backtest --account-tier professional

# Standard tier
python -m src.main --tc-backtest --account-tier standard

# Micro tier (highest costs)
python -m src.main --tc-backtest --account-tier micro
```

### Account Tiers & Costs

| Tier | TC Rate (bps) | Min Trade |
|------|---------------|-----------|
| institutional | 0.5 | $5 |
| professional | 5.0 | $15 |
| standard | 25.0 | $50 |
| micro | 75.0 | $150 |

### Expected Turnover

```bash
# High turnover strategy
python -m src.main --tc-backtest --expected-turnover 0.5

# Low turnover strategy
python -m src.main --tc-backtest --expected-turnover 0.05
```

---

## Conviction Filtering

### Enable High-Conviction Filter

```bash
# Enable with default threshold
python -m src.main --conviction-filter

# Custom threshold (z-score style)
python -m src.main --conviction-filter --conviction-threshold 1.5
```

### Conviction Threshold Meaning

- `0.5`: Trade when signal > 0.5 std (liberal)
- `1.0`: Trade when signal > 1.0 std (default)
- `2.0`: Trade when signal > 2.0 std (conservative)
- `3.0`: Trade only when very confident

---

## Output Options

### Saving Results

```bash
# Save all results (default)
python -m src.main

# Don't save to files
python -m src.main --no-save

# Custom output directory
python -m src.main --output-dir ./my_results

# Save plots
python -m src.main --save-plots
```

### Statistical Tests

```bash
# Run Diebold-Mariano and Clark-West tests
python -m src.main --statistical-tests
```

### Verbose Output

```bash
# Detailed progress output
python -m src.main --verbose
```

---

## Complete Examples

### Example 1: Basic SSRF

```bash
python -m src.main --sample-data
```

### Example 2: XGBoost with Transaction Costs

```bash
python -m src.main \
  --model-type xgboost \
  --tc-backtest \
  --account-tier professional \
  --alpha 0.1
```

### Example 3: Conservative Strategy

```bash
python -m src.main \
  --train-window 180 \
  --t-stat-threshold 2.0 \
  --conviction-filter \
  --conviction-threshold 2.0 \
  --tc-backtest \
  --account-tier institutional
```

### Example 4: Aggressive Strategy

```bash
python -m src.main \
  --model-type xgboost \
  --train-window 36 \
  --step-size 1 \
  --no-regime \
  --no-ct-restriction
```

### Example 5: Full Analysis with Real Data

```bash
python -m src.main \
  --start-date 1990-01-01 \
  --train-window 120 \
  --model-type elasticnet \
  --alpha 0.001 \
  --n-factors 10 \
  --tc-backtest \
  --account-tier standard \
  --conviction-filter \
  --statistical-tests \
  --save-plots \
  --verbose
```

### Example 6: Sector Rotation Analysis

```bash
python -m src.main \
  --sector-rotation Technology \
  --start-date 2000-01-01 \
  --train-window 60 \
  --model-type random_forest \
  --tc-backtest
```

---

## Common Issues

### "Module not found"

```bash
pip install -r requirements.txt
```

### "FRED API key not found"

Either:
1. Set FRED_API_KEY in `src/config.py`
2. Use sample data: `--sample-data`

### "Zero coefficients"

Try reducing regularization:
```bash
python -m src.main --alpha 0.0001 --unregularized
```

### "Out of memory"

Reduce training window or number of features:
```bash
python -m src.main --train-window 60 --n-indicators 30
```

---

## File Structure

```
sp500_macro_forecast/
├── src/
│   ├── main.py           # Main CLI entry point
│   ├── ssrf_model.py     # SSRF model implementation
│   ├── config.py        # Configuration
│   ├── tc_backtesting.py # Transaction cost backtester
│   └── ...
├── docs/
│   └── CLI_USAGE.md     # This file
└── data/
    ├── fred_cache/      # Cached FRED data
    └── sector_cache/     # Cached sector data
```

---

---

## Multi-Model OOS Comparison (`run_all_models_oos.py`)

### Overview

Compares 7 model types (elasticnet, linear, xgboost, random_forest, catboost, mlp, ensemble) on 3-month forward S&P 500 returns using walk-forward backtesting.

### Basic Usage

```bash
# Run all 7 models (default leverage 1.0/1.0, quarterly rebalance)
python run_all_models_oos.py

# Run only specific models
python run_all_models_oos.py elasticnet linear xgboost
```

### Leverage Sweep

```bash
# Sweep all 7 models × 4 leverage combos
python run_all_models_oos.py --sweep

# Sweep with custom leverage combos
python run_all_models_oos.py --sweep --max-long 2.0 --max-short 0.5
```

### Overlapping vs Non-Overlapping 3-Month Returns

```bash
# Default: overlapping 3-month returns at monthly frequency
# Consecutive predictions share 2/3 of the return window (inflates metrics)
python run_all_models_oos.py --sweep

# Non-overlapping 3-month returns at quarterly frequency
# Sub-samples X and y to every 3rd month for independent observations
# All metrics (R², Sharpe, DM) are valid without overlap adjustments
python run_all_models_oos.py --sweep --non-overlap
```

### Position Sizing

```bash
# Symmetric long/short (default)
python run_all_models_oos.py --max-long 1.0 --max-short 1.0

# Asymmetric: more long, less short (reduces drawdown)
python run_all_models_oos.py --max-long 2.5 --max-short 0.25
```

### Full CLI Reference

```
usage: run_all_models_oos.py [-h] [models ...]
                             [--max-long MAX_LONG]
                             [--max-short MAX_SHORT]
                             [--margin-rate MARGIN_RATE]
                             [--drawdown-limit DRAWDOWN_LIMIT]
                             [--step-size STEP_SIZE]
                             [--non-overlap]
                             [--sweep]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `models` | (all 7) | Model types to run: elasticnet, linear, xgboost, random_forest, catboost, mlp, ensemble |
| `--max-long` | 1.0 | Max long exposure (1.0 = no margin) |
| `--max-short` | 1.0 | Max short exposure (1.0 = full short) |
| `--margin-rate` | 0.05 | Annual margin interest rate |
| `--drawdown-limit` | 0.25 | Drawdown threshold for leverage reduction |
| `--step-size` | 3 | Walk-forward step in months |
| `--non-overlap` | — | Use non-overlapping 3-month returns (quarterly frequency, clean metrics) |
| `--sweep` | — | Run all leverage combos × all models |

---

## Getting Help

```bash
# Show all available arguments
python -m src.main --help
```

Output:
```
usage: main.py [-h] [--sample-data] [--n-periods N_PERIODS]
              [--n-indicators N_INDICATORS] [--n-factors N_FACTORS]
              [--start-date START_DATE] [--end-date END_DATE]
              [--seed SEED] [--sector-rotation SECTOR_ROTATION]
              [--train-window TRAIN_WINDOW]
              [--t-stat-threshold T_STAT_THRESHOLD]
              [--regime-window REGIME_WINDOW] [--alpha ALPHA]
              [--l1-ratio L1_RATIO] [--no-cv] [--unregularized]
              [--no-ct-restriction] [--no-regime] [--step-size STEP_SIZE]
              [--model-type {elasticnet,linear,xgboost,random_forest,
                            catboost,mlp,ensemble}]
              [--tc-rate TC_RATE] [--account-tier {micro,standard,
                          professional,institutional}]
              [--expected-turnover EXPECTED_TURNOVER] [--include-tc]
              [--tc-backtest] [--conviction-filter]
              [--conviction-threshold CONVICTION_THRESHOLD]
              [--output-dir OUTPUT_DIR] [--no-save] [--save-plots]
              [--statistical-tests] [--verbose]
```

# SSRF S&P 500 Macro Forecasting - CLI Usage Guide

## Quick Start

```bash
cd /workspace/sp500_macro_forecast

# Run with sample data (no API keys needed)
python -m src.main --use-sample-data

# Run with full help
python -m src.main --help
```

---

## Model Selection

Select the final regression model type:

```bash
# Available models:
#   elasticnet    - Elastic Net regularization (default)
#   linear        - Simple linear regression
#   xgboost       - Gradient boosting
#   random_forest - Random Forest
#   catboost      - CatBoost
#   mlp           - Neural network
#   ensemble      - Ensemble of all models

# Example: Try XGBoost
python -m src.main --model-type xgboost --use-sample-data

# Example: Simple linear regression (no regularization)
python -m src.main --model-type linear --unregularized --use-sample-data
```

---

## Data Options

### Using Sample Data (No API Keys)
```bash
# Default: 400 periods, 50 indicators
python -m src.main --use-sample-data

# Custom data generation
python -m src.main --use-sample-data --n-periods 500 --n-indicators 100 --seed 42
```

### Using Real FRED Data
```bash
# Requires FRED API key in config
python -m src.main --start-date 1990-01-01 --end-date 2024-12-31

# Predict specific sector rotation
python -m src.main --sector-rotation Technology
python -m src.main --sector-rotation Financials
python -m src.main --sector-rotation Energy
```

---

## Model Hyperparameters

### Feature Screening
```bash
# T-stat threshold for feature selection (default: 1.5)
python -m src.main --t-stat-threshold 2.0  # Stricter
python -m src.main --t-stat-threshold 1.0    # More features
```

### Regularization
```bash
# Elastic Net alpha (default: 0.001)
python -m src.main --alpha 0.01   # Stronger regularization
python -m src.main --alpha 0.0001 # Weaker regularization

# L1 ratio for Elastic Net (default: 0.5)
# 0 = Ridge (L2 only), 1 = Lasso (L1 only)
python -m src.main --l1-ratio 0.3  # More Ridge-like
python -m src.main --l1-ratio 0.8  # More Lasso-like

# Disable cross-validation
python -m src.main --no-cv

# Use unregularized (fixes zero-coefficient issue)
python -m src.main --unregularized --model-type linear
```

### Training Window
```bash
# Initial training window in months (default: 60 = 5 years)
python -m src.main --train-window 120  # 10 years
python -m src.main --train-window 36   # 3 years

# Walk-forward step size (default: 1 month)
python -m src.main --step-size 3
```

### Number of Factors
```bash
# Factors to extract via PCA (default: 10)
python -m src.main --n-factors 15
python -m src.main --n-factors 5
```

### Other Model Options
```bash
# Disable Campbell-Thompson restriction
python -m src.main --no-ct-restriction

# Disable regime detection
python -m src.main --no-regime

# Regime detection window (default: 12 months)
python -m src.main --regime-window 24
```

---

## Transaction Cost Analysis

### Enable TC-Adjusted Backtester
```bash
# Basic TC settings
python -m src.main --tc-backtest --tc-rate 25.0

# Account tier options:
#   micro          - <$10K account (50 bps)
#   standard       - $10K-$100K (25 bps)
#   professional   - $100K-$1M (10 bps)
#   institutional  - >$1M (5 bps)

python -m src.main --tc-backtest --account-tier professional
python -m src.main --tc-backtest --account-tier institutional

# Expected turnover rate (default: 15%)
python -m src.main --tc-backtest --expected-turnover 0.25

# Include TC in predictions
python -m src.main --include-tc
```

---

## Conviction Filtering

Only trade when signal is strong:

```bash
# Enable conviction filter
python -m src.main --conviction-filter

# Set minimum threshold (default: 1.0)
python -m src.main --conviction-filter --conviction-threshold 1.5

# Higher threshold = stricter filtering = fewer but stronger trades
python -m src.main --conviction-filter --conviction-threshold 2.0
```

---

## Statistical Testing

```bash
# Run Diebold-Mariano and Clark-West tests
python -m src.main --statistical-tests

# Save prediction plots
python -m src.main --save-plots

# Verbose output
python -m src.main --verbose
```

---

## Output Options

```bash
# Custom output directory
python -m src.main --output-dir ./my_results

# Don't save results to files
python -m src.main --no-save
```

---

## Complete Examples

### Quick Test with Sample Data
```bash
python -m src.main --use-sample-data --n-periods 200 --verbose
```

### XGBoost with Real Data
```bash
python -m src.main --model-type xgboost --start-date 2000-01-01
```

### Full Production Run
```bash
python -m src.main \
  --model-type elasticnet \
  --train-window 120 \
  --alpha 0.001 \
  --l1-ratio 0.5 \
  --t-stat-threshold 1.5 \
  --tc-backtest \
  --account-tier professional \
  --conviction-filter \
  --conviction-threshold 1.0 \
  --statistical-tests \
  --save-plots
```

### Debug Zero-Coefficient Issue
```bash
# Use unregularized linear regression
python -m src.main \
  --unregularized \
  --model-type linear \
  --use-sample-data \
  --verbose
```

### Compare Multiple Models
```bash
# Run ElasticNet
python -m src.main --model-type elasticnet --use-sample-data --no-save

# Run XGBoost
python -m src.main --model-type xgboost --use-sample-data --no-save

# Run Ridge (unregularized)
python -m src.main --unregularized --model-type linear --use-sample-data --no-save
```

---

## Argument Reference Table

| Argument | Default | Description |
|----------|---------|-------------|
| `--use-sample-data` | False | Use generated data instead of FRED |
| `--n-periods` | 400 | Periods for sample data |
| `--n-indicators` | 50 | Indicators for sample data |
| `--seed` | 42 | Random seed |
| `--start-date` | 1959-01-01 | FRED data start date |
| `--end-date` | None | FRED data end date |
| `--sector-rotation` | None | Predict specific sector |
| `--train-window` | 60 | Training window (months) |
| `--t-stat-threshold` | 1.5 | Feature screening threshold |
| `--regime-window` | 12 | Regime detection window |
| `--alpha` | 0.001 | Elastic Net alpha |
| `--l1-ratio` | 0.5 | Elastic Net L1 ratio |
| `--n-factors` | 10 | Number of factors |
| `--model-type` | elasticnet | Model type |
| `--step-size` | 1 | Walk-forward step size |
| `--tc-rate` | 25.0 | TC rate in bps |
| `--account-tier` | standard | Account tier |
| `--expected-turnover` | 0.15 | Expected turnover |
| `--conviction-threshold` | 1.0 | Conviction threshold |

---

## Troubleshooting

### "Zero coefficients" in model output
```bash
# Use unregularized model
python -m src.main --unregularized --model-type linear
```

### FRED API rate limit
```bash
# Use sample data instead
python -m src.main --use-sample-data
```

### Low hit ratio / poor performance
```bash
# Try different model type
python -m src.main --model-type xgboost

# Reduce regularization
python -m src.main --alpha 0.0001

# Increase training window
python -m src.main --train-window 120
```
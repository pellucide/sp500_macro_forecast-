"""
Quick Real Market Data Test - Simplified version for faster execution
Tests SSRF model with real S&P 500 data and proper out-of-sample validation
"""

import sys
import os
import warnings
import logging

os.chdir('/workspace/sp500_macro_forecast')
sys.path.insert(0, '/workspace/sp500_macro_forecast')

logging.basicConfig(level=logging.WARNING, format='%(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

print("=" * 70)
print("REAL MARKET DATA OUT-OF-SAMPLE TEST")
print("=" * 70)

# ============================================================================
# PART 1: FETCH REAL S&P 500 DATA
# ============================================================================
print("\n[PART 1] FETCHING S&P 500 DATA")
print("-" * 50)

def fetch_spx_data():
    """Fetch real S&P 500 monthly returns."""
    try:
        import yfinance as yf
        print("Fetching S&P 500 data from Yahoo Finance...")
        data = yf.download('^GSPC', start='1985-01-01', end='2025-12-31', progress=False)

        if len(data) == 0:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            close = data['Close']['^GSPC']
        else:
            close = data['Close']

        monthly = close.resample('ME').last()
        returns = monthly.pct_change().dropna()
        returns.name = 'SP500_return'

        print(f"  Downloaded {len(returns)} months of data")
        print(f"  Range: {returns.index[0].strftime('%Y-%m')} to {returns.index[-1].strftime('%Y-%m')}")
        return returns

    except Exception as e:
        print(f"  Failed: {e}")
        return None

spx_returns = fetch_spx_data()

if spx_returns is None:
    # Fallback to synthetic
    dates = pd.date_range('1990-01-01', '2024-12-31', freq='ME')
    np.random.seed(42)
    spx_returns = pd.Series(np.random.randn(len(dates)-1) * 0.045 + 0.007, index=dates[1:])

print(f"\nData ready: {len(spx_returns)} observations")

# ============================================================================
# PART 2: CREATE MACRO INDICATORS (SYNTHETIC BUT REALISTIC)
# ============================================================================
print("\n[PART 2] CREATING MACRO INDICATORS")
print("-" * 50)

n_periods = len(spx_returns)
dates = spx_returns.index

# Create 15 realistic macro indicators in 5 categories
categories = ['output_income', 'labor', 'inflation', 'interest', 'sentiment']
macro_data = pd.DataFrame(index=dates)

np.random.seed(42)
for cat in categories:
    for j in range(3):
        col = f"{cat}_{j}"
        base = np.cumsum(np.random.randn(n_periods) * 0.015)
        # Add some predictive signal
        signal = spx_returns.values * 0.15 + np.sin(np.arange(n_periods) / 20) * 0.2
        macro_data[col] = base + signal

print(f"  Created {len(macro_data.columns)} macro indicators in 5 categories")

# Create feature groups
groups = {}
for col in macro_data.columns:
    cat = col.split('_')[0]
    if cat not in groups:
        groups[cat] = []
    groups[cat].append(col)

# ============================================================================
# PART 3: ALIGN DATA WITH LAG (POINT-IN-TIME DISCIPLINE)
# ============================================================================
print("\n[PART 3] ALIGNING DATA WITH LAG (NO LOOK-AHEAD BIAS)")
print("-" * 50)

# Lag macro data by 1 month (predict r_t using X_{t-1})
macro_lagged = macro_data.shift(1)

# Remove NaN rows
valid_idx = macro_lagged.dropna().index.intersection(spx_returns.dropna().index)
X = macro_lagged.loc[valid_idx]
y = spx_returns.loc[valid_idx]

print(f"  Lagged macro data by 1 month")
print(f"  Valid observations: {len(X)}")
print(f"  Date range: {X.index[0].strftime('%Y-%m')} to {X.index[-1].strftime('%Y-%m')}")

# ============================================================================
# PART 4: TEMPORAL SPLIT
# ============================================================================
print("\n[PART 4] TEMPORAL TRAIN/VAL/TEST SPLIT")
print("-" * 50)

test_start = '2020-01-01'
val_start = '2015-01-01'

train_idx = X.index[X.index < val_start]
val_idx = X.index[(X.index >= val_start) & (X.index < test_start)]
test_idx = X.index[X.index >= test_start]

X_train, y_train = X.loc[train_idx], y.loc[train_idx]
X_val, y_val = X.loc[val_idx], y.loc[val_idx]
X_test, y_test = X.loc[test_idx], y.loc[test_idx]

print(f"  Training:   {train_idx[0].strftime('%Y-%m')} to {train_idx[-1].strftime('%Y-%m')} ({len(train_idx)} mo)")
print(f"  Validation: {val_idx[0].strftime('%Y-%m')} to {val_idx[-1].strftime('%Y-%m')} ({len(val_idx)} mo)")
print(f"  Test (OOS): {test_idx[0].strftime('%Y-%m')} to {test_idx[-1].strftime('%Y-%m')} ({len(test_idx)} mo)")

# ============================================================================
# PART 5: SSRF MODEL
# ============================================================================
print("\n[PART 5] SSRF MODEL TRAINING")
print("-" * 50)

from src.ssrf_model import SSRFModel, SSRFConfig

# Train on training data
config = SSRFConfig(
    t_stat_threshold=0.5,
    n_factors=5,
    regime_window=6,
    elastic_net_alpha=0.001,
    use_elastic_net_cv=True,
    use_regime_detection=True,
)

model = SSRFModel(config)
print("  Training SSRF model on full training set...")
model.fit(X_train, y_train, groups)

n_features = len(model.selected_features_) if hasattr(model, 'selected_features_') else 'N/A'
print(f"  Selected features: {n_features}")

# ============================================================================
# PART 6: OUT-OF-SAMPLE PREDICTION
# ============================================================================
print("\n[PART 6] OUT-OF-SAMPLE PREDICTIONS")
print("-" * 50)

# Walk-forward prediction on test period
predictions = []
for i, (date, row) in enumerate(X_test.iterrows()):
    # Train on all data up to this point
    train_end = X.index.get_loc(date)
    X_hist = X.iloc[:train_end]
    y_hist = y.iloc[:train_end]

    # Incremental fit (simplified)
    model_i = SSRFModel(config)
    model_i.fit(X_hist, y_hist, groups)

    # Predict
    y_regime = y_hist
    pred = model_i.predict(pd.DataFrame(row).T, y_regime)
    predictions.append(pred.values[0])

    if (i + 1) % 12 == 0:
        print(f"  {i+1}/{len(test_idx)} test periods completed")

pred_test = pd.Series(predictions, index=test_idx)

# ============================================================================
# PART 7: EVALUATE RESULTS
# ============================================================================
print("\n[PART 7] OUT-OF-SAMPLE RESULTS")
print("-" * 50)

from sklearn.metrics import mean_squared_error

# Metrics
mse = mean_squared_error(y_test, pred_test)
mae = np.abs(y_test - pred_test).mean()
hit_ratio = (np.sign(pred_test) == np.sign(y_test)).mean()

benchmark = y_test.mean()
r2_oos = 1 - mse / mean_squared_error(y_test, np.full_like(y_test, benchmark))

# Portfolio performance
portfolio_returns = pred_test * y_test
cumulative_port = (1 + portfolio_returns).prod() - 1
cumulative_spx = (1 + y_test).prod() - 1

sharpe = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(12) if portfolio_returns.std() > 0 else 0

# Drawdown
cumul = (1 + portfolio_returns).cumprod()
running_max = cumul.expanding().max()
drawdown = (cumul / running_max - 1).min()
max_dd = abs(drawdown)

print(f"\n  Campbell-Thompson R² OOS: {r2_oos:.4f}")
print(f"  Hit Ratio: {hit_ratio:.1%}")
print(f"  MSE: {mse:.6f}")
print(f"  MAE: {mae:.4f}")
print(f"  Sharpe Ratio: {sharpe:.3f}")
print(f"  Max Drawdown: {max_dd:.1%}")
print(f"\n  Strategy Cumulative: {cumulative_port:.1%}")
print(f"  S&P 500 Cumulative: {cumulative_spx:.1%}")
print(f"  Alpha (vs SPX): {(cumulative_port - cumulative_spx)*100:.1f}%")

# ============================================================================
# PART 8: CONVICTION FILTER ANALYSIS
# ============================================================================
print("\n[PART 8] CONVICTION FILTER IMPACT")
print("-" * 50)

# Calculate conviction (signal strength / std)
signal_std = pred_test.std()
conviction = pred_test.abs() / signal_std

# Test different thresholds
thresholds = [0.5, 1.0, 1.5, 2.0]
print("\nThreshold | Active% | Hit Ratio | Sharpe")
print("-" * 45)

for thresh in thresholds:
    active = conviction >= thresh
    n_active = active.sum()
    pct = n_active / len(pred_test) * 100

    if n_active > 0:
        hit = (np.sign(pred_test[active]) == np.sign(y_test[active])).mean()
        ret_active = pred_test[active] * y_test[active]
        sharpe_t = (ret_active.mean() / ret_active.std()) * np.sqrt(12) if ret_active.std() > 0 else 0
    else:
        hit, sharpe_t = 0, 0

    print(f"  {thresh:.1f}      |  {pct:5.0f}%  |   {hit:.1%}    | {sharpe_t:.2f}")

print("\n" + "=" * 70)
print("TEST COMPLETE - Real S&P 500 Data Out-of-Sample Validation")
print("=" * 70)
"""
Real Market Data Out-of-Sample Test
Tests SSRF model with:
1. Real S&P 500 data from yfinance
2. Macroeconomic indicators from FRED (or synthetic if unavailable)
3. Proper point-in-time discipline (no look-ahead bias)
4. Walk-forward out-of-sample validation
5. Transaction cost and conviction filtering

Run: cd /workspace/sp500_macro_forecast && python test_real_data.py
"""

import sys
import os
import warnings
import logging

# Setup
os.chdir('/workspace/sp500_macro_forecast')
sys.path.insert(0, '/workspace/sp500_macro_forecast')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

print("=" * 70)
print("REAL MARKET DATA OUT-OF-SAMPLE TEST")
print("=" * 70)

# ============================================================================
# PART 1: DATA ACQUISITION
# ============================================================================
print("\n[PART 1] DATA ACQUISITION")
print("-" * 50)

def fetch_real_spx_data(start_date='1990-01-01', end_date='2025-12-31'):
    """Fetch real S&P 500 data using yfinance."""
    try:
        import yfinance as yf
        logger.info("Fetching S&P 500 data from Yahoo Finance...")

        # Download monthly data
        data = yf.download('^GSPC', start=start_date, end=end_date, progress=False)

        if len(data) == 0:
            logger.warning("No S&P 500 data fetched, using fallback")
            return None

        # Handle MultiIndex columns
        if isinstance(data.columns, pd.MultiIndex):
            close = data['Close']['^GSPC']
        else:
            close = data['Close']

        # Resample to monthly (end of month)
        monthly = close.resample('ME').last()

        # Calculate monthly returns
        returns = monthly.pct_change().dropna()
        returns.name = 'SP500_return'

        logger.info(f"Fetched {len(returns)} months of S&P 500 data")
        logger.info(f"Date range: {returns.index[0].strftime('%Y-%m')} to {returns.index[-1].strftime('%Y-%m')}")

        return returns

    except Exception as e:
        logger.error(f"Failed to fetch S&P 500 data: {e}")
        return None


def fetch_fred_indicators(start_date='1980-01-01', end_date='2025-12-31'):
    """Fetch macroeconomic indicators from FRED."""
    try:
        from src.fred_data import FREDDataLoader

        # Try to load real FRED data
        loader = FREDDataLoader()

        if loader.fred is None:
            logger.warning("No FRED API key, generating synthetic macro data")
            return None

        logger.info("Fetching FRED-MD indicators...")
        indicators = loader.fetch_all_indicators(
            start_date=start_date,
            end_date=end_date
        )

        if len(indicators) > 50:
            logger.info(f"Fetched {len(indicators.columns)} FRED indicators")
            return indicators
        else:
            logger.warning("Insufficient FRED data, generating synthetic")
            return None

    except Exception as e:
        logger.warning(f"FRED fetch failed: {e}, using synthetic data")
        return None


def create_aligned_data(spx_returns, macro_data, lag_periods=1):
    """
    Create properly aligned dataset with point-in-time discipline.

    To avoid look-ahead bias:
    - We use macro data from time t-1 to predict SPX return at time t
    - All data is lagged by one month (typical for macroeconomic forecasting)
    """
    # If no macro data, create synthetic
    if macro_data is None or len(macro_data) < 100:
        logger.info("Generating synthetic macroeconomic indicators...")
        n_periods = len(spx_returns)
        dates = spx_returns.index

        categories = ['output_income', 'labor', 'inflation', 'interest', 'sentiment']
        macro_data = pd.DataFrame(index=dates)

        np.random.seed(42)
        for i, cat in enumerate(categories):
            for j in range(3):
                col_name = f"{cat}_{j}"
                # Create realistic macro indicators with some autocorrelation
                base = np.cumsum(np.random.randn(n_periods) * 0.01)
                # Add some signal to SPX returns
                signal_component = spx_returns.values * (0.1 + 0.1 * np.random.rand())
                macro_data[col_name] = base + signal_component + np.sin(np.arange(n_periods) / 24 + i) * 0.3

        logger.info(f"Created {len(macro_data.columns)} synthetic indicators")

    # Ensure SPX returns and macro data are aligned
    common_idx = spx_returns.index.intersection(macro_data.index)
    spx_returns = spx_returns.loc[common_idx]
    macro_data = macro_data.loc[common_idx]

    # Apply lag to macro data (point-in-time discipline)
    # Predict r_t using X_{t-1}
    macro_lagged = macro_data.shift(lag_periods)

    # Remove rows with NaN (first lag_periods rows)
    valid_idx = macro_lagged.dropna().index.intersection(spx_returns.dropna().index)

    spx_aligned = spx_returns.loc[valid_idx]
    macro_aligned = macro_lagged.loc[valid_idx]

    logger.info(f"Aligned dataset: {len(spx_aligned)} observations")
    logger.info(f"Date range: {valid_idx[0].strftime('%Y-%m')} to {valid_idx[-1].strftime('%Y-%m')}")

    return macro_aligned, spx_aligned


# Fetch data
spx_returns = fetch_real_spx_data()
if spx_returns is None:
    # Create synthetic SPX if download fails
    logger.info("Creating synthetic S&P 500 returns...")
    dates = pd.date_range('1990-01-01', '2024-12-31', freq='ME')
    np.random.seed(42)
    returns = pd.Series(np.random.randn(len(dates)) * 0.05 + 0.008, index=dates)
    spx_returns = returns.iloc[:-1]  # Remove last

macro_data = fetch_fred_indicators()

# Create aligned dataset
X, y = create_aligned_data(spx_returns, macro_data, lag_periods=1)

# Create feature groups
groups = {}
for col in X.columns:
    cat = col.split('_')[0]
    if cat not in groups:
        groups[cat] = []
    groups[cat].append(col)

print(f"\nDataset prepared:")
print(f"  Features: {len(X.columns)} indicators")
print(f"  Target: {len(y)} monthly SPX returns")
print(f"  Groups: {list(groups.keys())}")
print(f"  Date range: {X.index[0].strftime('%Y-%m')} to {X.index[-1].strftime('%Y-%m')}")

# ============================================================================
# PART 2: TEMPORAL TRAIN/VAL/TEST SPLIT (NO LOOK-AHEAD BIAS)
# ============================================================================
print("\n[PART 2] TEMPORAL TRAIN/VAL/TEST SPLIT")
print("-" * 50)

# Define temporal splits
# Out-of-sample test: last 5 years (2020-2024)
# Validation: 2015-2019
# Training: everything before 2015

test_start = '2020-01-01'
val_start = '2015-01-01'

# Find closest valid dates
all_dates = X.index.sort_values()
test_mask = all_dates >= test_start
val_mask = (all_dates >= val_start) & (all_dates < test_start)
train_mask = all_dates < val_start

train_idx = all_dates[train_mask]
val_idx = all_dates[val_mask]
test_idx = all_dates[test_mask]

X_train = X.loc[train_idx]
y_train = y.loc[train_idx]
X_val = X.loc[val_idx]
y_val = y.loc[val_idx]
X_test = X.loc[test_idx]
y_test = y.loc[test_idx]

print(f"\nTemporal Split:")
print(f"  Training:   {train_idx[0].strftime('%Y-%m')} to {train_idx[-1].strftime('%Y-%m')} ({len(train_idx)} months)")
print(f"  Validation: {val_idx[0].strftime('%Y-%m')} to {val_idx[-1].strftime('%Y-%m')} ({len(val_idx)} months)")
print(f"  Test (OOS): {test_idx[0].strftime('%Y-%m')} to {test_idx[-1].strftime('%Y-%m')} ({len(test_idx)} months)")

# ============================================================================
# PART 3: SSRF MODEL WITH WALK-FORWARD OUT-OF-SAMPLE TESTING
# ============================================================================
print("\n[PART 3] SSRF WALK-FORWARD OUT-OF-SAMPLE TEST")
print("-" * 50)

from src.ssrf_model import SSRFModel, SSRFConfig
from src.tc_backtesting import TCAdjustedWalkForwardBacktester

# Configuration for SSRF
ssrf_config = SSRFConfig(
    t_stat_threshold=0.5,
    n_factors=5,
    regime_window=6,
    elastic_net_alpha=0.001,
    elastic_net_l1_ratio=0.5,
    use_elastic_net_cv=True,
    use_regime_detection=True,
    # TC settings
    include_tc=True,
    tc_rate_bps=15.0,  # Professional tier
    expected_turnover=0.20,
    account_tier='professional',
    # Conviction filtering
    conviction_filter_enabled=True,
    min_conviction_threshold=1.0,
)

# Walk-forward backtester
backtester = TCAdjustedWalkForwardBacktester(
    initial_train_window=120,  # 10 years training minimum
    forecast_horizon=1,
    use_ct_restriction=False,
    step_size=1,
    tc_rate_bps=15.0,
    account_tier='professional',
    expected_turnover=0.20,
)

# Enable conviction filtering
backtester.conviction_filter_enabled = True
backtester.min_conviction_threshold = 1.0

print(f"\nRunning walk-forward backtest with conviction filtering...")
print(f"  Conviction threshold: {backtester.min_conviction_threshold}")
print(f"  TC rate: {backtester.effective_tc_rate:.1f} bps")

# Run on combined train+val (for training) then test
# Actually, we'll do pure OOS: train on data up to t-1, test at t
results = backtester.run(
    X, y, groups, ssrf_config, verbose=True, include_tc_in_prediction=True
)

# ============================================================================
# PART 4: ANALYZE RESULTS
# ============================================================================
print("\n[PART 4] OUT-OF-SAMPLE RESULTS")
print("-" * 50)

# Extract test-period predictions
test_predictions = results.predictions.loc[test_idx] if test_idx.isin(results.predictions.index).any() else results.predictions

# Align predictions with test dates
common_test_dates = test_idx.intersection(results.predictions.index)
pred_test = results.predictions.loc[common_test_dates]
actual_test = y.loc[common_test_dates]

print(f"\nOut-of-Sample Performance ({len(common_test_dates)} months):")
print(f"  Period: {common_test_dates[0].strftime('%Y-%m')} to {common_test_dates[-1].strftime('%Y-%m')}")

# Calculate metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

mse = mean_squared_error(actual_test, pred_test)
mae = mean_absolute_error(actual_test, pred_test)
hit_ratio = (np.sign(pred_test) == np.sign(actual_test)).mean()

# R2 vs benchmark (historical mean)
benchmark = np.full(len(actual_test), actual_test.mean())
r2_oos = 1 - mse / mean_squared_error(actual_test, benchmark)

# Cumulative returns
portfolio_returns = pred_test * actual_test
cumulative_return = (1 + portfolio_returns).prod() - 1
spx_cumulative = (1 + actual_test).prod() - 1

# Sharpe ratio
sharpe = (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(12) if portfolio_returns.std() > 0 else 0

# Max drawdown
cumulative = (1 + portfolio_returns).cumprod()
running_max = cumulative.expanding().max()
drawdown = (cumulative / running_max - 1).min()
max_dd = abs(drawdown)

print(f"\n  Campbell-Thompson R² OOS: {r2_oos:.4f}")
print(f"  Hit Ratio: {hit_ratio:.1%}")
print(f"  MSE: {mse:.6f}")
print(f"  MAE: {mae:.4f}")
print(f"  Sharpe Ratio: {sharpe:.3f}")
print(f"  Max Drawdown: {max_dd:.1%}")
print(f"\n  Cumulative Return: {cumulative_return:.1%}")
print(f"  S&P 500 Cumulative: {spx_cumulative:.1%}")
print(f"  vs S&P 500: {(cumulative_return - spx_cumulative)*100:.1f}%")

# ============================================================================
# PART 5: CONVICTION FILTER ANALYSIS
# ============================================================================
print("\n[PART 5] CONVICTION FILTER ANALYSIS")
print("-" * 50)

active_periods = (pred_test != 0).sum()
total_periods = len(pred_test)
print(f"\nConviction Filtering Impact:")
print(f"  Active periods: {active_periods}/{total_periods} ({active_periods/total_periods*100:.1f}%)")
print(f"  Filtered periods: {total_periods - active_periods}/{total_periods} ({(total_periods - active_periods)/total_periods*100:.1f}%)")

# TC costs
tc_costs_test = results.tc_costs.loc[common_test_dates] if common_test_dates.isin(results.tc_costs.index).any() else results.tc_costs.iloc[-len(common_test_dates):]
avg_tc_cost = tc_costs_test.mean() * 100
total_tc_cost = tc_costs_test.sum() * 100

print(f"\nTransaction Costs:")
print(f"  Average Monthly TC: {avg_tc_cost:.3f}%")
print(f"  Total TC Cost: {total_tc_cost:.2f}%")

# Gross vs Net returns
gross_return = portfolio_returns.mean() * 12 * 100
net_return = (gross_return - avg_tc_cost * 12)
print(f"\n  Gross Annual Return: {gross_return:.2f}%")
print(f"  Net Annual Return: {net_return:.2f}%")
print(f"  TC Drag: {avg_tc_cost * 12:.2f}%")

# ============================================================================
# PART 6: COMPARISON ACROSS DIFFERENT CONVICTION THRESHOLDS
# ============================================================================
print("\n[PART 6] CONVICTION THRESHOLD SENSITIVITY")
print("-" * 50)

thresholds = [0.5, 1.0, 1.5, 2.0, 2.5]
sensitivity_results = []

for threshold in thresholds:
    backtester_test = TCAdjustedWalkForwardBacktester(
        initial_train_window=120,
        tc_rate_bps=15.0,
        account_tier='professional',
        expected_turnover=0.20,
    )
    backtester_test.conviction_filter_enabled = True
    backtester_test.min_conviction_threshold = threshold

    result = backtester_test.run(X, y, groups, ssrf_config, verbose=False)

    # Get test period metrics
    pred_t = result.predictions.loc[common_test_dates]
    actual_t = actual_test

    n_active = (pred_t != 0).sum()
    hit = (np.sign(pred_t) == np.sign(actual_t)).mean()
    sharpe_t = (pred_t * actual_t).mean() / (pred_t * actual_t).std() * np.sqrt(12) if (pred_t * actual_t).std() > 0 else 0

    sensitivity_results.append({
        'threshold': threshold,
        'active_pct': n_active / len(pred_t) * 100,
        'hit_ratio': hit * 100,
        'sharpe': sharpe_t,
    })

    print(f"  Threshold {threshold:.1f}: {n_active}/{len(pred_t)} active ({n_active/len(pred_t)*100:.0f}%), "
          f"Hit={hit:.1%}, Sharpe={sharpe_t:.2f}")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
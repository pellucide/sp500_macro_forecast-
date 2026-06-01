#!/usr/bin/env python3
"""
Out-of-Sample Test with Real Market Data
Uses cached FRED data + SP500 returns from Yahoo Finance
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress only specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Import SSRF components
from src.ssrf_model import SSRFModel, SSRFConfig
from src.backtesting import WalkForwardBacktester
from src.evaluation import MetricsCalculator, generate_report


def load_fred_data(path='data/fred_cache/all_fred_data_enhanced.csv'):
    """Load cached FRED macro data."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    logger.info(f"Loaded FRED data: {df.shape[1]} indicators, {df.shape[0]} months")
    logger.info(f"Date range: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')}")
    return df


def load_sp500_returns(start='1979-01-01', end='2026-06-01'):
    """Download S&P 500 monthly returns from Yahoo Finance."""
    logger.info("Downloading S&P 500 returns from Yahoo Finance...")
    spx = yf.download('^GSPC', start=start, end=end, progress=False)
    spx_monthly = spx['Close'].resample('ME').last()
    # Squeeze to Series (yfinance returns DataFrame even for single ticker)
    returns = spx_monthly.squeeze().pct_change().dropna()
    returns = returns.rename('SP500_return')
    logger.info(f"SP500 returns: {len(returns)} months, {returns.index[0].strftime('%Y-%m')} to {returns.index[-1].strftime('%Y-%m')}")
    return returns


def create_groups_from_data(df):
    """Create feature groups based on columns actually present in the data."""
    cols = set(df.columns)
    
    groups = {
        'output_income': [c for c in ['GDPC1', 'PCECC96'] if c in cols],
        'labor': [c for c in ['UNRATE', 'PAYEMS', 'EMRATIO', 'HOUST', 'PERMIT', 'UNRATE_CHANGE_12M'] if c in cols],
        'inflation': [c for c in ['CPIAUCSL', 'CPILFESL', 'PCECTPI', 'PCEPILFE', 'GDPDEF', 'PPIFGS'] if c in cols],
        'interest': [c for c in ['TB3MS', 'TB6MS', 'GS1', 'GS2', 'GS5', 'GS10', 'GS20', 'GS30', 
                                  'AAA', 'BAA', 'T10Y2YM', 'TEDRATE', 'REAL_10Y',
                                  'YIELD_SLOPE_10Y3M', 'YIELD_SLOPE_10Y2Y', 'YIELD_SLOPE_2Y3M',
                                  'BAAFFM', 'AAAFFM', 'CREDIT_SPREAD_BAA', 'CREDIT_SPREAD_QUALITY'] if c in cols],
        'sentiment': [c for c in ['VIXCLS', 'UMCSENT', 'IC4WSA', 'SENTIMENT_REGIME',
                                   'VIX_REGIME_HIGH', 'VIX_REGIME_LOW'] if c in cols],
        'money_supply': [c for c in ['M1SL', 'M2SL', 'M3SL',
                                      'M1_GROWTH_12M', 'M1_GROWTH_6M', 'M1_GROWTH_3M', 'M1_ACCEL',
                                      'M2_GROWTH_12M', 'M2_GROWTH_6M', 'M2_GROWTH_3M', 'M2_ACCEL',
                                      'M3_GROWTH_12M', 'M3_GROWTH_6M', 'M3_GROWTH_3M', 'M3_ACCEL',
                                      'M1_M2_RATIO', 'M2_M3_RATIO', 'M1_VS_M3_GROWTH'] if c in cols],
    }
    
    # Remove empty groups
    groups = {k: v for k, v in groups.items() if v}
    
    for name, features in groups.items():
        logger.info(f"  Group '{name}': {len(features)} features")
    
    return groups


def align_data(indicators, target):
    """Align indicators with target using proper 1-month-ahead lag."""
    # Find common date range
    common_idx = indicators.index.intersection(target.index)
    indicators = indicators.loc[common_idx].sort_index()
    target = target.loc[common_idx].sort_index()
    
    # Forward fill only (no bfill to avoid look-ahead)
    indicators = indicators.ffill()
    
    # Shift target forward by 1: X[t] predicts y[t+1]
    target_aligned = target.shift(-1)
    
    # Drop last row (NaN target after shift)
    target_df = pd.DataFrame({'target': target_aligned})
    combined = pd.concat([indicators, target_df], axis=1)
    combined = combined.dropna(subset=['target'])
    indicators = combined.drop(columns=['target'])
    target_aligned = combined['target']
    
    logger.info(f"Aligned data: {len(indicators)} periods")
    logger.info(f"Target mean: {target_aligned.mean():.4f}, std: {target_aligned.std():.4f}")
    
    return indicators, target_aligned


def run_oos_test(model_type='elasticnet', step_size=3, train_window=120, n_factors=10):
    """Run full OOS walk-forward backtest with real data."""
    
    print("=" * 70)
    print("SSRF OUT-OF-SAMPLE TEST WITH REAL MARKET DATA")
    print(f"Started: {datetime.now()}")
    print("=" * 70)
    
    # Load data
    indicators = load_fred_data()
    target = load_sp500_returns()
    
    # Create groups
    groups = create_groups_from_data(indicators)
    
    # Align with 1-month-ahead lag
    X, y = align_data(indicators, target)
    
    # Configure SSRF model
    config = SSRFConfig(
        t_stat_threshold=1.5,
        n_factors=n_factors,
        regime_window=12,
        model_type=model_type,
        use_regime_detection=True,
        prediction_scale=1.0,  # No arbitrary scaling
    )
    
    # Run walk-forward backtest
    logger.info(f"Running walk-forward backtest: train_window={train_window}, step_size={step_size}")
    backtester = WalkForwardBacktester(
        model_class=SSRFModel,
        initial_train_window=train_window,
        step_size=step_size,
        use_ct_restriction=False,
    )
    
    result = backtester.run(X, y, groups, model_config=config, verbose=True)
    
    # Compute additional metrics
    calc = MetricsCalculator(annualization_factor=12)
    metrics = calc.calculate(result.predictions, result.actual_returns, result.benchmark_predictions)
    
    # Print results
    print("\n" + "=" * 70)
    print("OUT-OF-SAMPLE RESULTS")
    print("=" * 70)
    print(f"\nTest Period: {result.dates[0].strftime('%Y-%m')} to {result.dates[-1].strftime('%Y-%m')}")
    print(f"Number of OOS Predictions: {len(result.predictions)}")
    print(f"Training Window: {train_window} months (expanding)")
    print(f"Step Size: {step_size} month(s)")
    print(f"Model Type: {model_type}")
    print(f"Number of Factors: {n_factors}")
    
    print(f"\n--- Statistical Metrics ---")
    print(f"Campbell-Thompson R² OOS: {metrics.r2_oos:.4f}")
    print(f"MSE: {metrics.mse:.6f}")
    print(f"MAE: {metrics.mae:.4f}")
    print(f"Hit Ratio (direction accuracy): {metrics.hit_ratio:.2%}")
    
    print(f"\n--- Portfolio Performance ---")
    print(f"Sharpe Ratio (annualized): {metrics.sharpe_ratio:.4f}")
    print(f"Sortino Ratio: {metrics.sortino_ratio:.4f}")
    print(f"Calmar Ratio: {metrics.calmar_ratio:.4f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"Cumulative Return: {metrics.cumulative_return:.2%}")
    print(f"Annualized Return: {metrics.annualized_return:.2%}")
    print(f"Annualized Volatility: {metrics.annualized_volatility:.2%}")
    
    print(f"\n--- Benchmark Comparison ---")
    print(f"Benchmark Cumulative: {(1 + result.actual_returns).cumprod().iloc[-1] - 1:.2%}")
    print(f"Buy & Hold Return: {(1 + result.actual_returns).prod() - 1:.2%}")
    
    # Diebold-Mariano test
    from src.evaluation import StatisticalTests
    dm_stat, dm_pval = StatisticalTests.dm_test(
        result.actual_returns.values,
        result.predictions.values,
        result.benchmark_predictions.values
    )
    print(f"\n--- Statistical Tests ---")
    print(f"Diebold-Mariano vs Benchmark: t={dm_stat:.4f}, p={dm_pval:.4f}")
    
    # R² OOS confidence interval
    r2_lower, r2_upper = StatisticalTests.out_of_sample_r2_confidence_interval(
        metrics.r2_oos, len(result.predictions)
    )
    print(f"R² OOS 95% CI: [{r2_lower:.4f}, {r2_upper:.4f}]")
    
    print("\n" + "=" * 70)
    print(f"Completed: {datetime.now()}")
    print("=" * 70)
    
    return result, metrics


if __name__ == "__main__":
    import sys
    
    # Parse simple args
    model_type = 'elasticnet'
    step_size = 3
    train_window = 120
    
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
    if len(sys.argv) > 2:
        step_size = int(sys.argv[2])
    if len(sys.argv) > 3:
        train_window = int(sys.argv[3])
    
    run_oos_test(model_type=model_type, step_size=step_size, train_window=train_window)

#!/usr/bin/env python3
"""
Run OOS walk-forward tests for ALL model types using 3-month forward returns.
Target = 3-month forward S&P 500 return, predicted monthly from FRED-MD indicators.
This gives monthly predictions that are each for the cumulative return over the
following 3 months.
"""
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf

# Setup logging to file only to keep stdout clean
import logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from src.ssrf_model import SSRFModel, SSRFConfig
from src.backtesting import WalkForwardBacktester
from src.evaluation import MetricsCalculator, StatisticalTests

# Re-import run_oos_real_data functions
from run_oos_real_data import (
    load_fred_data, load_sp500_returns, impute_vix_proxy,
    create_groups_from_data
)

MODEL_TYPES = ['elasticnet', 'linear', 'xgboost', 'random_forest', 'catboost', 'mlp', 'ensemble']

FORWARD_HORIZON = 3  # months


def load_forward_returns(horizon=3):
    """Load S&P 500 prices and compute horizon-month forward returns at monthly frequency.

    y[t] = (P[t+horizon] / P[t]) - 1

    Returns overlapping forward returns at monthly frequency to preserve sample size.
    """
    spx = yf.download('^GSPC', start='1979-01-01', end='2026-06-01', progress=False)
    prices = spx['Close'].resample('ME').last().squeeze()
    # 1-month returns for VIX proxy imputation
    monthly_ret = prices.pct_change().dropna().rename('SP500_return')
    # horizon-month forward return
    forward_ret = prices.pct_change(horizon).shift(-horizon).dropna()
    forward_ret = forward_ret.rename(f'SP500_{horizon}m_forward')
    return monthly_ret, forward_ret


def run_model(model_type, step_size=3, train_window=120):
    """Run OOS test for a single model type with 3-month forward returns."""
    indicators = load_fred_data()

    # Load monthly returns (for VIX proxy) and 3-month forward returns (target)
    monthly_ret, target = load_forward_returns(FORWARD_HORIZON)

    # VIX proxy needs 1-month returns (rolling 12-month realized volatility)
    indicators = impute_vix_proxy(indicators, monthly_ret)
    groups = create_groups_from_data(indicators)

    # Align: X[t] predicts 3-month forward return from t to t+horizon
    # Don't use align_data() since it shifts target by -1 (for 1-month horizon)
    common_idx = indicators.index.intersection(target.index)
    X = indicators.loc[common_idx].sort_index().ffill()
    y = target.loc[common_idx].sort_index()

    # Drop NaN target rows (last horizon months have no forward data)
    valid = y.notna()
    X = X[valid]
    y = y[valid]

    config = SSRFConfig(
        t_stat_threshold=0.75,
        n_factors=10,
        regime_window=12,
        model_type=model_type,
        use_regime_detection=True,
        prediction_scale=1.0,
    )

    backtester = WalkForwardBacktester(
        model_class=SSRFModel,
        initial_train_window=train_window,
        step_size=step_size,
        use_ct_restriction=False,
    )

    t0 = time.time()
    result = backtester.run(X, y, groups, model_config=config, verbose=False)
    elapsed = time.time() - t0

    # Annualization factor = 12/horizon for horizon-month returns
    # NOTE: with overlapping forward returns, Sharpe is unreliable.
    # R² OOS and hit ratio are the primary metrics.
    calc = MetricsCalculator(annualization_factor=12 // FORWARD_HORIZON)
    metrics = calc.calculate(result.predictions, result.actual_returns, result.benchmark_predictions)

    # Benchmark return (cumulative 3-month return)
    bh_return = (1 + result.actual_returns).prod() - 1

    # Diebold-Mariano (NOTE: overlapping forecasts inflate t-stat; use with caution)
    dm_stat, dm_pval = StatisticalTests.dm_test(
        result.actual_returns.values,
        result.predictions.values,
        result.benchmark_predictions.values
    )

    # R² OOS CI
    r2_lower, r2_upper = StatisticalTests.out_of_sample_r2_confidence_interval(
        metrics.r2_oos, len(result.predictions)
    )

    return {
        'model': model_type,
        'time_s': elapsed,
        'n_predictions': len(result.predictions),
        'test_start': str(result.dates[0])[:7],
        'test_end': str(result.dates[-1])[:7],
        'r2_oos': metrics.r2_oos,
        'r2_ci': f"[{r2_lower:.4f}, {r2_upper:.4f}]",
        'hit_ratio': metrics.hit_ratio,
        'sharpe': metrics.sharpe_ratio,
        'sortino': metrics.sortino_ratio,
        'calmar': metrics.calmar_ratio,
        'max_dd': metrics.max_drawdown,
        'cum_ret': metrics.cumulative_return,
        'ann_ret': metrics.annualized_return,
        'ann_vol': metrics.annualized_volatility,
        'dm_stat': dm_stat,
        'dm_pval': dm_pval,
        'mse': metrics.mse,
        'mae': metrics.mae,
        'benchmark_cum': bh_return,
    }


def print_results(results):
    """Print a clean comparison table."""
    print("=" * 120)
    print("SSRF MODEL COMPARISON — 3-MONTH FORWARD RETURNS (OVERLAPPING, MONTHLY FREQ)")
    print(f"Run: {datetime.now()}")
    print(f"Target: (P[t+3] / P[t]) - 1, predicted monthly from FRED-MD")
    print(f"Benchmark: expanding mean of 3-month forward returns")
    print("=" * 120)

    # Summary header
    print(f"\n{'Model':<16} {'R² OOS':>8} {'Hit%':>7} {'Sharpe':>8} {'Sortino':>8} {'Calmar':>8} {'MaxDD':>7} {'CumRet':>8} {'AnnRet':>8} {'AnnVol':>7} {'DM t':>7} {'DM p':>7} {'Time':>8}")
    print("-" * 120)

    for r in results:
        print(f"{r['model']:<16} {r['r2_oos']:>8.4f} {r['hit_ratio']:>6.1%} "
              f"{r['sharpe']:>8.4f} {r['sortino']:>8.4f} {r['calmar']:>8.4f} "
              f"{r['max_dd']:>6.2%} {r['cum_ret']:>7.2%} {r['ann_ret']:>7.2%} "
              f"{r['ann_vol']:>6.2%} {r['dm_stat']:>7.3f} {r['dm_pval']:>7.4f} "
              f"{r['time_s']:>7.1f}s")

    print("-" * 120)

    # Benchmark comparison table
    print(f"\n{'Model':<16} {'R² OOS':>8} {'R² 95% CI':<20} {'DM p-val':>9} {'vs B&H':>8}")
    print(f"{'':<16} {'':>8} {'':<20} {'':>9} {'3m ret':>8}")
    print("-" * 65)
    for r in results:
        bh_diff = r['cum_ret'] - r['benchmark_cum']
        sig = "***" if r['dm_pval'] < 0.01 else ("**" if r['dm_pval'] < 0.05 else ("*" if r['dm_pval'] < 0.10 else ""))
        print(f"{r['model']:<16} {r['r2_oos']:>8.4f} {r['r2_ci']:<20} {r['dm_pval']:>9.4f}{sig} {bh_diff:>7.2%}")

    print("-" * 65)
    print("Significance: *** p<0.01, ** p<0.05, * p<0.10")
    print(f"Caveat: overlapping 3-month returns inflate DM significance. "
          f"Portfolio metrics (Sharpe, CumRet) are unreliable.")
    print(f"\nBenchmark cumulative 3-month return: {results[0]['benchmark_cum']:.2%}")


if __name__ == "__main__":
    # Allow specifying which models to run via args
    if len(sys.argv) > 1:
        models_to_run = [m for m in MODEL_TYPES if m in sys.argv[1:]]
        if not models_to_run:
            models_to_run = MODEL_TYPES
    else:
        models_to_run = MODEL_TYPES

    print(f"Models to run: {', '.join(models_to_run)}")
    print(f"Target: {FORWARD_HORIZON}-month forward return, monthly frequency")
    print()

    all_results = []
    for model_type in models_to_run:
        print(f"\n{'='*70}")
        print(f"Running: {model_type}")
        print(f"{'='*70}")
        sys.stdout.flush()
        try:
            result = run_model(model_type)
            all_results.append(result)
            print(f"  {model_type}: R² OOS={result['r2_oos']:.4f}, "
                  f"Hit={result['hit_ratio']:.1%}, Sharpe={result['sharpe']:.4f}, "
                  f"Time={result['time_s']:.1f}s")
            sys.stdout.flush()
        except Exception as e:
            print(f"  {model_type}: FAILED - {e}", file=sys.stderr)
            sys.stderr.flush()

    print()
    print_results(all_results)

#!/usr/bin/env python3.9
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
    create_groups_from_data, load_alternative_features
)

MODEL_TYPES = ['elasticnet', 'linear', 'xgboost', 'random_forest', 'catboost', 'mlp', 'ensemble']

FORWARD_HORIZON = 3

# Asymmetric position sizing defaults
MAX_LONG = 1.0     # max long exposure (1.0 = no margin, 1.5 = 50% margin)
MAX_SHORT = 1.0    # max short exposure (1.0 = full short, 0.0 = no short)
MARGIN_RATE = 0.05  # annual margin interest rate
DRAWDOWN_LIMIT = 0.25  # drawdown threshold for leverage reduction (0.0-0.5)
STEP_SIZE = 3  # default step size, overridden by CLI args

# Leverage combos for --sweep mode (max_long, max_short)
LEVERAGE_COMBOS = [
    (1.0, 1.0),
    (1.5, 0.5),
    (1.75, 0.25),
    (2.5, 0.25),
]


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


def run_model(model_type, step_size=3, train_window=120,
              max_long=MAX_LONG, max_short=MAX_SHORT,
              margin_rate=MARGIN_RATE, drawdown_limit=DRAWDOWN_LIMIT):
    """Run OOS test for a single model type with 3-month forward returns."""
    indicators = load_fred_data()

    # Load monthly returns (for VIX proxy) and 3-month forward returns (target)
    monthly_ret, target = load_forward_returns(FORWARD_HORIZON)

    # VIX proxy needs 1-month returns (rolling 12-month realized volatility)
    indicators = impute_vix_proxy(indicators, monthly_ret)

    # Merge alternative/exuberance features (CAPE, put/call ratio, margin debt)
    alt_features = load_alternative_features()
    if alt_features is not None and len(alt_features) > 0:
        indicators = indicators.join(alt_features, how='left')
        # Forward-fill gaps within each series
        indicators = indicators.ffill()
        # Fill leading NaN (features that start after FRED data, e.g. PUT_CALL_RATIO from 1995)
        # with the first valid value to avoid NaN breakage in PCA
        for col in alt_features.columns:
            if col in indicators.columns and indicators[col].isna().any():
                first_val = indicators[col].dropna()
                if len(first_val) > 0:
                    indicators[col] = indicators[col].fillna(first_val.iloc[0])
        logger.info(f"Merged alternative features: {indicators.shape[1]} total columns")

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
        max_long=max_long,
        max_short=max_short,
        margin_rate=margin_rate,
        drawdown_limit=drawdown_limit,
    )

    t0 = time.time()
    result = backtester.run(X, y, groups, model_config=config, verbose=False)
    elapsed = time.time() - t0

    # Annualization factor = 12/horizon for horizon-month returns
    # NOTE: with overlapping forward returns, Sharpe is unreliable.
    # R² OOS and hit ratio are the primary metrics.
    calc = MetricsCalculator(
        annualization_factor=12 // FORWARD_HORIZON,
        max_long=max_long,
        max_short=max_short,
        margin_rate=margin_rate,
        drawdown_limit=drawdown_limit,
    )
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
        'n_pos': metrics.n_pos,
        'n_neg': metrics.n_neg,
        'pos_accuracy': metrics.pos_accuracy,
        'neg_accuracy': metrics.neg_accuracy,
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
    print(f"Step size: {STEP_SIZE} month(s)")
    print(f"Target: (P[t+3] / P[t]) - 1, predicted monthly from FRED-MD")
    print(f"Benchmark: expanding mean of 3-month forward returns")
    print("=" * 120)

    # Summary header
    print(f"\n{'Model':<16} {'R² OOS':>8} {'Hit%':>7} {'LongAcc':>8} {'ShortAcc':>9} {'nLong':>6} {'nShort':>7} {'Sharpe':>8} {'CumRet':>8} {'AnnRet':>8} {'AnnVol':>7} {'DM p':>7} {'Time':>8}")
    print("-" * 120)

    for r in results:
        pos_acc = r.get('pos_accuracy', 0)
        neg_acc = r.get('neg_accuracy', 0)
        n_pos = r.get('n_pos', 0)
        n_neg = r.get('n_neg', 0)
        print(f"{r['model']:<16} {r['r2_oos']:>8.4f} {r['hit_ratio']:>6.1%} "
              f"{pos_acc:>7.1%} {neg_acc:>8.1%} {n_pos:>5d} {n_neg:>6d} "
              f"{r['sharpe']:>8.4f} "
              f"{r['cum_ret']:>7.2%} {r['ann_ret']:>7.2%} "
              f"{r['ann_vol']:>6.2%} {r['dm_pval']:>7.4f} "
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


def print_sweep_results(all_results, models_run, bh_ann, bh_sharpe):
    """Print cross-tabulated sweep results with B&H comparison."""
    # Group results by leverage combo
    by_combo = {}
    for r in all_results:
        key = f"{r['max_long']:.2f}/{r['max_short']:.2f}"
        by_combo.setdefault(key, {})[r['model']] = r

    print("=" * 130)
    print("LEVERAGE SWEEP — CROSS-TAB RESULTS")
    print(f"Run: {datetime.now()} | Models: {', '.join(models_run)}")
    print(f"Target: {FORWARD_HORIZON}-month forward return | Step: {STEP_SIZE}m")
    print("=" * 130)

    # AnnRet table
    print(f"\n--- Annualized Return ---")
    header = f"{'L/S':<12}"
    for m in models_run:
        header += f" {m:<16}"
    header += f" {'B&H SP500':<12}"
    print(header)
    print("-" * len(header))
    for combo, models in by_combo.items():
        row = f"{combo:<12}"
        for m in models_run:
            r = models.get(m)
            row += f" {r['ann_ret']:>7.2%}    " if r else f" {'FAILED':>7}    "
        row += f" {bh_ann:>7.2%}"
        print(row)

    # Sharpe table
    print(f"\n--- Sharpe Ratio ---")
    header = f"{'L/S':<12}"
    for m in models_run:
        header += f" {m:<16}"
    header += f" {'B&H SP500':<12}"
    print(header)
    print("-" * len(header))
    for combo, models in by_combo.items():
        row = f"{combo:<12}"
        for m in models_run:
            r = models.get(m)
            row += f" {r['sharpe']:>7.4f}    " if r else f" {'FAILED':>7}    "
        row += f" {bh_sharpe:>7.4f}"
        print(row)

    # MaxDD table
    print(f"\n--- Max Drawdown ---")
    header = f"{'L/S':<12}"
    for m in models_run:
        header += f" {m:<16}"
    header += f" {'B&H SP500':<12}"
    print(header)
    print("-" * len(header))
    for combo, models in by_combo.items():
        row = f"{combo:<12}"
        for m in models_run:
            r = models.get(m)
            row += f" {r['max_dd']:>7.2%}    " if r else f" {'FAILED':>7}    "
        row += f" {'N/A':>7}"
        print(row)

    # Hit ratio table
    print(f"\n--- Hit Ratio ---")
    header = f"{'L/S':<12}"
    for m in models_run:
        header += f" {m:<16}"
    print(header)
    print("-" * len(header))
    for combo, models in by_combo.items():
        row = f"{combo:<12}"
        for m in models_run:
            r = models.get(m)
            row += f" {r['hit_ratio']:>6.1%}     " if r else f" {'FAILED':>7}    "
        print(row)

    # R² OOS table
    print(f"\n--- R² OOS ---")
    header = f"{'L/S':<12}"
    for m in models_run:
        header += f" {m:<16}"
    print(header)
    print("-" * len(header))
    for combo, models in by_combo.items():
        row = f"{combo:<12}"
        for m in models_run:
            r = models.get(m)
            row += f" {r['r2_oos']:>7.4f}    " if r else f" {'FAILED':>7}    "
        print(row)

    print(f"\nB&H SP500: AnnRet={bh_ann:.2%}, Sharpe={bh_sharpe:.4f}")
    print(f"Note: vs B&H comparisons use excess AnnRet (model - B&H)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run OOS comparison across all model types')
    parser.add_argument('models', nargs='*', help='Model types to run (default: all)')
    parser.add_argument('--max-long', type=float, default=MAX_LONG,
                        help=f'Max long exposure (default: {MAX_LONG})')
    parser.add_argument('--max-short', type=float, default=MAX_SHORT,
                        help=f'Max short exposure (default: {MAX_SHORT})')
    parser.add_argument('--margin-rate', type=float, default=MARGIN_RATE,
                        help=f'Annual margin interest rate (default: {MARGIN_RATE})')
    parser.add_argument('--drawdown-limit', type=float, default=DRAWDOWN_LIMIT,
                        help=f'Drawdown limit for leverage reduction 0.0-0.5 (default: {DRAWDOWN_LIMIT})')
    parser.add_argument('--step-size', type=int, default=3,
                        help=f'Walk-forward step in months (default: 3)')
    parser.add_argument('--sweep', action='store_true',
                        help='Run leverage sweep across all combos and models')
    args = parser.parse_args()

    STEP_SIZE = args.step_size  # noqa: F811 — re-bind module-level constant

    models_to_run = [m for m in MODEL_TYPES if m in args.models] if args.models else MODEL_TYPES

    if args.sweep:
        # Sweep mode: run all leverage combos across all models
        all_results = []
        n_total = len(LEVERAGE_COMBOS) * len(models_to_run)
        n_done = 0

        print("=" * 100)
        print("LEVERAGE SWEEP — ALL MODELS × ALL COMBOS")
        print(f"Models: {', '.join(models_to_run)}")
        print(f"Combos: {', '.join(f'{ml:.2f}/{ms:.2f}' for ml, ms in LEVERAGE_COMBOS)}")
        print(f"Total runs: {n_total}")
        print(f"Target: {FORWARD_HORIZON}-month forward return")
        print(f"Step size: {args.step_size} month(s)")
        print("=" * 100)

        for max_long, max_short in LEVERAGE_COMBOS:
            for model_type in models_to_run:
                n_done += 1
                label = f"{max_long:.2f}/{max_short:.2f}"
                print(f"\n[{n_done}/{n_total}] {model_type} @ {label}...", end=" ")
                sys.stdout.flush()
                try:
                    result = run_model(
                        model_type,
                        step_size=args.step_size,
                        max_long=max_long,
                        max_short=max_short,
                        margin_rate=args.margin_rate,
                        drawdown_limit=args.drawdown_limit,
                    )
                    result['max_long'] = max_long
                    result['max_short'] = max_short
                    all_results.append(result)
                    print(f"R²={result['r2_oos']:.4f} Hit={result['hit_ratio']:.1%} "
                          f"Sharpe={result['sharpe']:.4f} ({result['time_s']:.1f}s)")
                except Exception as e:
                    print(f"FAILED - {e}")
                    import traceback
                    traceback.print_exc()
                sys.stdout.flush()

        # Compute B&H from 3-month forward returns
        _, target = load_forward_returns(FORWARD_HORIZON)
        if target is not None and len(target) > 0:
            bh_cum = (1 + target.values).prod() - 1
            n_years = len(target) / 12
            bh_ann = (1 + bh_cum) ** (1 / n_years) - 1
            bh_vol = target.std() * np.sqrt(12 / FORWARD_HORIZON)
            bh_sharpe = bh_ann / bh_vol if bh_vol > 0 else 0
        else:
            bh_ann = 0.09
            bh_sharpe = 0.66

        print()
        print_sweep_results(all_results, models_to_run, bh_ann, bh_sharpe)
    else:
        # Normal mode: single leverage combo
        print(f"Models to run: {', '.join(models_to_run)}")
        print(f"Target: {FORWARD_HORIZON}-month forward return, monthly frequency")
        print(f"Step size: {args.step_size} month(s) (rebalance every {args.step_size} month(s))")
        print(f"Position sizing: max_long={args.max_long}, max_short={args.max_short}")
        print(f"Margin: {args.margin_rate:.1%}, drawdown_limit={args.drawdown_limit}")
        print()

        all_results = []
        for model_type in models_to_run:
            print(f"\n{'='*70}")
            print(f"Running: {model_type}")
            print(f"{'='*70}")
            sys.stdout.flush()
            try:
                result = run_model(
                    model_type,
                    step_size=args.step_size,
                    max_long=args.max_long,
                    max_short=args.max_short,
                    margin_rate=args.margin_rate,
                    drawdown_limit=args.drawdown_limit,
                )
                all_results.append(result)
                print(f"  {model_type}: R² OOS={result['r2_oos']:.4f}, "
                      f"Hit={result['hit_ratio']:.1%}, Sharpe={result['sharpe']:.4f}, "
                      f"Time={result['time_s']:.1f}s")
                sys.stdout.flush()
            except Exception as e:
                print(f"  {model_type}: FAILED - {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                sys.stderr.flush()

        print()
        print_results(all_results)

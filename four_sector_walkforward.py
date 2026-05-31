"""
4-Sector Combination Analysis with PROPER Walk-Forward Validation
CRITICAL FIX: True out-of-sample testing with selection at each decision point

Key improvements over naive approach:
1. At EACH decision point, evaluate ALL 126 combinations using training data
2. SELECT the best-performing combination based on training performance
3. Apply ONLY the selected combination to the test period (out-of-sample)
4. Repeat for each walk-forward window
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from itertools import combinations
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_data(start_date='1999-01-01', end_date='2026-05-29'):
    """Load sector ETF data."""

    sector_etfs = {
        'Materials': 'XLB',
        'Energy': 'XLE',
        'Financials': 'XLF',
        'Industrials': 'XLI',
        'Technology': 'XLK',
        'Consumer_Staples': 'XLP',
        'Health_Care': 'XLV',
        'Utilities': 'XLU',
        'Consumer_Discretionary': 'XLY',
    }

    all_prices = {}

    for sector, ticker in sector_etfs.items():
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(data) > 0:
                if isinstance(data.columns, pd.MultiIndex):
                    all_prices[sector] = data['Close'][ticker]
                else:
                    all_prices[sector] = data['Close']
        except Exception as e:
            logger.warning(f"Failed to load {sector}: {e}")

    if not all_prices:
        raise ValueError("No data loaded")

    prices_df = pd.DataFrame(all_prices)
    monthly = prices_df.resample('ME').last()
    returns = monthly.pct_change()

    # Load S&P 500 benchmark
    spx = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
    if isinstance(spx.columns, pd.MultiIndex):
        spx_returns = spx['Close']['^GSPC'].resample('ME').last().pct_change()
    else:
        spx_returns = spx['Close'].resample('ME').last().pct_change()

    return prices_df, monthly, returns, spx_returns


def walkforward_selection_test(returns, spx_returns, train_window=24, test_window=12):
    """
    Walk-forward test with TRUE selection at each decision point.

    At each decision point:
    1. Evaluate ALL 126 combinations using training data (in-sample)
    2. SELECT the best combination based on training performance
    3. Apply the SELECTED combination to test period (out-of-sample)
    4. Roll forward

    This is how you would actually implement a real strategy.
    """

    sector_list = list(returns.columns)
    all_combinations = list(combinations(sector_list, 4))

    logger.info(f"Testing {len(all_combinations)} combinations with walk-forward selection")
    logger.info(f"Train window: {train_window} months, Test window: {test_window} months")

    # Walk-forward: find decision points
    n_total = len(returns)
    decision_points = list(range(train_window, n_total - test_window, test_window))

    logger.info(f"Number of walk-forward decisions: {len(decision_points)}")

    # Store which combination was selected at each decision point
    selection_history = []
    portfolio_values = [1.0]
    oos_returns_list = []
    oos_dates = []

    # Track in-sample vs out-of-sample performance
    is_performance_by_period = []
    oos_performance_by_period = []

    for period_idx, decision_idx in enumerate(decision_points):
        # Training period: indices [decision_idx - train_window, decision_idx - 1]
        train_start = decision_idx - train_window
        train_end = decision_idx
        train_data = returns.iloc[train_start:train_end]

        # Test period: indices [decision_idx, decision_idx + test_window - 1]
        test_start = decision_idx
        test_end = decision_idx + test_window

        if test_end > n_total:
            continue

        test_data = returns.iloc[test_start:test_end]

        # Step 1: Evaluate ALL combinations using training data (IN-SAMPLE)
        in_sample_scores = []
        for sectors in all_combinations:
            sector_list_test = list(sectors)
            train_rets = train_data[sector_list_test].mean(axis=1)
            train_total = (1 + train_rets).prod() - 1
            in_sample_scores.append((sectors, train_total))

        # Step 2: SELECT best combination based on training performance
        in_sample_scores.sort(key=lambda x: x[1], reverse=True)
        best_combo, best_in_sample = in_sample_scores[0]
        best_sectors = list(best_combo)

        selection_history.append({
            'period': period_idx + 1,
            'decision_date': returns.index[decision_idx],
            'test_start': returns.index[test_start],
            'test_end': returns.index[test_end - 1],
            'selected_sectors': best_sectors,
            'in_sample_return': best_in_sample
        })

        # Track in-sample performance
        is_performance_by_period.append(best_in_sample)

        # Step 3: Apply SELECTED combination to test period (OUT-OF-SAMPLE)
        test_rets = test_data[best_sectors].mean(axis=1)

        for month_idx, month_ret in enumerate(test_rets):
            new_value = portfolio_values[-1] * (1 + month_ret)
            portfolio_values.append(new_value)
            oos_returns_list.append(month_ret)
            oos_dates.append(test_data.index[month_idx])

        # Track OOS performance for this period
        period_oos = (1 + test_rets).prod() - 1
        oos_performance_by_period.append(period_oos)

        logger.info(f"  Period {period_idx + 1}: Selected {best_sectors} "
                   f"(IS: {best_in_sample:.2%}, OOS: {period_oos:.2%})")

    # Calculate metrics for the actual selected strategy
    oos_returns = pd.Series(oos_returns_list, index=oos_dates)
    metrics = calculate_metrics(oos_returns, spx_returns, portfolio_values)

    # Also track what would happen if we picked RANDOM combination
    # This shows the value of selection
    random_results = []
    np.random.seed(42)
    for _ in range(1000):
        random_oos = []
        random_pv = [1.0]
        for period_idx, decision_idx in enumerate(decision_points):
            test_start = decision_idx
            test_end = decision_idx + test_window
            if test_end > n_total:
                continue

            test_data = returns.iloc[test_start:test_end]
            random_combo = all_combinations[np.random.randint(len(all_combinations))]
            random_sectors = list(random_combo)
            test_rets = test_data[random_sectors].mean(axis=1)

            for month_ret in test_rets:
                new_value = random_pv[-1] * (1 + month_ret)
                random_pv.append(new_value)
                random_oos.append(month_ret)

        if len(random_oos) > 0:
            random_final = random_pv[-1]
            random_return = random_final - 1
            random_results.append(random_return)

    random_mean = np.mean(random_results)
    random_std = np.std(random_results)

    return metrics, selection_history, is_performance_by_period, oos_performance_by_period, random_mean, random_std


def calculate_metrics(returns, spx_returns, portfolio_values):
    """Calculate performance metrics."""

    total_return = portfolio_values[-1] - 1
    n_months = len(returns)
    n_years = n_months / 12 if n_months > 0 else 0

    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 and total_return > -1 else 0

    annualized_vol = returns.std() * np.sqrt(12)
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

    # Max drawdown
    rolling_max = pd.Series(portfolio_values).cummax()
    drawdown = (pd.Series(portfolio_values) - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Hit ratio
    hit_ratio = (returns > 0).mean()

    # Benchmark alignment
    if len(returns) > 0 and returns.index.isin(spx_returns.index).all():
        aligned_spx = spx_returns.loc[returns.index]
    else:
        aligned_spx = spx_returns.iloc[:len(returns)]

    benchmark_total = (1 + aligned_spx).prod() - 1
    benchmark_ann = (1 + benchmark_total) ** (1 / n_years) - 1 if n_years > 0 else 0

    excess_return = total_return - benchmark_total
    info_ratio = excess_return / annualized_vol if annualized_vol > 0 else 0

    # Calmar
    calmar = annualized_return / abs(max_dd) if max_dd != 0 else 0

    return {
        'Total Return': total_return,
        'Ann. Return': annualized_return,
        'Ann. Vol': annualized_vol,
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd,
        'Hit Ratio': hit_ratio,
        'Benchmark Ann': benchmark_ann,
        'Excess Return': excess_return,
        'Info Ratio': info_ratio,
        'Calmar Ratio': calmar,
        'N Months': n_months
    }


def main():
    """Run walk-forward selection analysis."""

    print("\n" + "=" * 80)
    print("4-SECTOR COMBINATION WALK-FORWARD ANALYSIS")
    print("PROPER OUT-OF-SAMPLE TESTING WITH SELECTION AT EACH DECISION POINT")
    print("=" * 80)

    # Load data
    logger.info("\nLoading data...")
    prices, monthly, returns, spx_returns = load_data()

    logger.info(f"Data period: {returns.index[0].strftime('%Y-%m')} to {returns.index[-1].strftime('%Y-%m')}")
    logger.info(f"Total months: {len(returns)}")

    # Run the proper walk-forward test
    print("\nRunning walk-forward test with TRUE selection at each decision point...")
    print("  - At EACH decision point, evaluate all 126 combinations (in-sample)")
    print("  - SELECT the best combination based on training performance")
    print("  - Apply the SELECTED combination to test period (out-of-sample)")
    print("=" * 80)

    metrics, selection_history, is_perf, oos_perf, random_mean, random_std = walkforward_selection_test(
        returns, spx_returns,
        train_window=24,
        test_window=12
    )

    # Results
    print("\n" + "=" * 80)
    print("RESULTS: WALK-FORWARD SELECTION STRATEGY")
    print("=" * 80)

    print(f"\n  Total Return:      {metrics['Total Return']:.2%}")
    print(f"  Ann. Return:       {metrics['Ann. Return']:.2%}")
    print(f"  Ann. Vol:          {metrics['Ann. Vol']:.2%}")
    print(f"  Sharpe Ratio:     {metrics['Sharpe Ratio']:.3f}")
    print(f"  Max Drawdown:      {metrics['Max Drawdown']:.2%}")
    print(f"  Hit Ratio:         {metrics['Hit Ratio']:.1%}")
    print(f"  Benchmark Ann:      {metrics['Benchmark Ann']:.2%}")
    print(f"  Excess Return:    {metrics['Excess Return']:.2%}")

    # Selection history
    print("\n" + "=" * 80)
    print("SELECTION HISTORY (What was selected at each decision point)")
    print("=" * 80)

    print(f"\n{'Period':<8} {'Decision':<12} {'Test Period':<24} {'Selected Sectors':<40} {'IS Ret':>10} {'OOS Ret':>10}")
    print("-" * 130)

    for sel in selection_history:
        sectors_short = ', '.join([s[:6] for s in sel['selected_sectors']])
        print(f"{sel['period']:<8} {sel['decision_date'].strftime('%Y-%m'):<12} "
              f"{sel['test_start'].strftime('%Y-%m')} to {sel['test_end'].strftime('%Y-%m'):<12} "
              f"{sectors_short:<40} {sel['in_sample_return']:>9.1%} {oos_perf[sel['period']-1]:>9.1%}")

    # In-sample vs Out-of-sample analysis
    print("\n" + "=" * 80)
    print("IN-SAMPLE VS OUT-OF-SAMPLE PERFORMANCE")
    print("=" * 80)

    avg_is = np.mean(is_perf) * 100
    avg_oos = np.mean(oos_perf) * 100
    selection_decay = avg_oos - avg_is

    print(f"\n  Average In-Sample Return:  {avg_is:.2f}%")
    print(f"  Average Out-of-Sample Return: {avg_oos:.2f}%")
    print(f"  Selection Decay (OOS - IS): {selection_decay:.2f}%")
    print(f"  This shows how much performance degrades when using OOS data")

    # Comparison with baselines
    print("\n" + "=" * 80)
    print("COMPARISON WITH BASELINES")
    print("=" * 80)

    # Simple equal weight all 9 sectors
    portfolio_values_eq = [1.0]
    for i in range(24, len(returns)):
        month_return = returns.iloc[i].mean()
        new_value = portfolio_values_eq[-1] * (1 + month_return)
        portfolio_values_eq.append(new_value)

    final_eq = portfolio_values_eq[-1]
    total_eq = final_eq - 1
    n_months_eq = len(portfolio_values_eq) - 1
    n_years_eq = n_months_eq / 12
    ann_eq = (1 + total_eq) ** (1 / n_years_eq) - 1 if n_years_eq > 0 else 0

    print(f"\nWalk-Forward Selection Strategy:")
    print(f"  Ann. Return: {metrics['Ann. Return']:.2%}")
    print(f"  Sharpe:      {metrics['Sharpe Ratio']:.3f}")
    print(f"  Max DD:      {metrics['Max Drawdown']:.2%}")

    print(f"\nEqual Weight All 9 Sectors (no selection):")
    print(f"  Ann. Return: {ann_eq:.2%}")

    # S&P 500 benchmark
    aligned_spx = spx_returns.loc[returns.index[24:]]
    spx_total = (1 + aligned_spx).prod() - 1
    spx_ann = (1 + spx_total) ** (12 / len(aligned_spx)) - 1
    print(f"\nS&P 500 Benchmark:")
    print(f"  Ann. Return: {spx_ann:.2%}")

    # Random selection baseline
    print(f"\nRandom 4-Sector Selection (1000 simulations):")
    print(f"  Mean Return: {random_mean:.2%}")
    print(f"  Std Dev:     {random_std:.2%}")

    # Sector frequency analysis
    print("\n" + "=" * 80)
    print("SECTOR SELECTION FREQUENCY")
    print("=" * 80)

    from collections import Counter
    all_selected = []
    for sel in selection_history:
        all_selected.extend(sel['selected_sectors'])

    sector_counts = Counter(all_selected)
    for sector, count in sector_counts.most_common():
        pct = count / len(selection_history) * 100
        print(f"  {sector}: {count}/{len(selection_history)} periods ({pct:.1f}%)")

    # Key insight
    print("\n" + "=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print(f"""
The walk-forward test with TRUE selection shows:
- Selection Strategy Sharpe: {metrics['Sharpe Ratio']:.3f}
- Selection Strategy Ann Return: {metrics['Ann. Return']:.2%}
- Average In-Sample Performance: {avg_is:.2f}%
- Average Out-of-Sample Performance: {avg_oos:.2f}%
- Selection Decay: {selection_decay:.2f}%

This demonstrates the true value of combination selection.
The difference between in-sample and out-of-sample performance
shows the impact of overfitting when selecting based on past data.
""")

    return metrics, selection_history


if __name__ == "__main__":
    metrics, history = main()

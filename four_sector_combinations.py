"""
4-Sector Combination Analysis
Tests all C(9,4) = 126 combinations of 4 sectors from 9 available sectors
Equal weight across each combination
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

    # Resample to monthly (end of month)
    monthly = prices_df.resample('ME').last()

    # Calculate simple returns
    returns = monthly.pct_change()

    # Load S&P 500 benchmark
    spx = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
    if isinstance(spx.columns, pd.MultiIndex):
        spx_returns = spx['Close']['^GSPC'].resample('ME').last().pct_change()
    else:
        spx_returns = spx['Close'].resample('ME').last().pct_change()

    return prices_df, monthly, returns, spx_returns


def calculate_metrics(returns, spx_returns, portfolio_values):
    """Calculate performance metrics."""

    total_return = portfolio_values[-1] - 1
    n_months = len(returns)
    n_years = n_months / 12
    annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    annualized_vol = returns.std() * np.sqrt(12)
    sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

    # Max drawdown
    rolling_max = pd.Series(portfolio_values).cummax()
    drawdown = (pd.Series(portfolio_values) - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Hit ratio
    hit_ratio = (returns > 0).mean()

    # Benchmark
    aligned_spx = spx_returns.loc[returns.index]
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


def test_combination(returns, spx_returns, sectors, start_idx=24):
    """Test equal-weight portfolio for a specific sector combination."""

    # Get returns for these sectors (convert tuple to list if needed)
    sector_list = list(sectors)
    sector_returns = returns[sector_list]

    # Start from train_window month
    portfolio_values = [1.0]

    for i in range(start_idx, len(sector_returns)):
        # Equal weight
        month_return = sector_returns.iloc[i].mean()
        new_value = portfolio_values[-1] * (1 + month_return)
        portfolio_values.append(new_value)

    # Calculate metrics
    strategy_returns = pd.Series([portfolio_values[i] / portfolio_values[i-1] - 1
                                  for i in range(1, len(portfolio_values))],
                                 index=sector_returns.index[start_idx:start_idx + len(portfolio_values) - 1])

    metrics = calculate_metrics(strategy_returns, spx_returns, portfolio_values)
    metrics['Sectors'] = ', '.join(sectors)
    metrics['Sector List'] = sectors

    return metrics, portfolio_values


def main():
    """Run all 4-sector combination analysis."""

    print("\n" + "=" * 80)
    print("4-SECTOR COMBINATION ANALYSIS")
    print("Testing all C(9,4) = 126 combinations with equal weights")
    print("=" * 80)

    # Load data
    logger.info("\nLoading data...")
    prices, monthly, returns, spx_returns = load_data()

    logger.info(f"Data period: {returns.index[0].strftime('%Y-%m')} to {returns.index[-1].strftime('%Y-%m')}")
    logger.info(f"Total months: {len(returns)}")
    logger.info(f"Sectors: {list(returns.columns)}")

    # Get all combinations of 4 sectors from 9
    sector_list = list(returns.columns)
    all_combinations = list(combinations(sector_list, 4))

    logger.info(f"\nTesting {len(all_combinations)} combinations of 4 sectors...")

    results = []

    for i, sectors in enumerate(all_combinations):
        if (i + 1) % 20 == 0:
            logger.info(f"  Progress: {i+1}/{len(all_combinations)} combinations tested...")

        metrics, _ = test_combination(returns, spx_returns, sectors)
        results.append(metrics)

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Sort by Sharpe ratio
    results_df = results_df.sort_values('Sharpe Ratio', ascending=False)

    # Print top 20 combinations
    print("\n" + "=" * 80)
    print("TOP 20 BEST 4-SECTOR COMBINATIONS (by Sharpe Ratio)")
    print("=" * 80)

    print(f"\n{'Rank':<5} {'Sectors':<55} {'Ann.Ret':>10} {'Sharpe':>8} {'Max DD':>10} {'Hit':>8}")
    print("-" * 100)

    for rank, (_, row) in enumerate(results_df.head(20).iterrows(), 1):
        sectors_str = row['Sectors'][:52] + "..." if len(row['Sectors']) > 55 else row['Sectors']
        print(f"{rank:<5} {sectors_str:<55} {row['Ann. Return']:>9.1%} {row['Sharpe Ratio']:>8.3f} "
              f"{row['Max Drawdown']:>9.1%} {row['Hit Ratio']:>7.1%}")

    # Print bottom 10
    print("\n" + "-" * 100)
    print("BOTTOM 10 WORST 4-SECTOR COMBINATIONS")
    print("-" * 100)

    for rank, (_, row) in enumerate(results_df.tail(10).iterrows(), 1):
        sectors_str = row['Sectors'][:52] + "..." if len(row['Sectors']) > 55 else row['Sectors']
        print(f"{rank:<5} {sectors_str:<55} {row['Ann. Return']:>9.1%} {row['Sharpe Ratio']:>8.3f} "
              f"{row['Max Drawdown']:>9.1%} {row['Hit Ratio']:>7.1%}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS ACROSS ALL 126 COMBINATIONS")
    print("=" * 80)

    print(f"\n{'Metric':<20} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("-" * 70)

    for metric in ['Ann. Return', 'Sharpe Ratio', 'Max Drawdown', 'Hit Ratio', 'Ann. Vol']:
        mean_val = results_df[metric].mean()
        std_val = results_df[metric].std()
        min_val = results_df[metric].min()
        max_val = results_df[metric].max()
        print(f"{metric:<20} {mean_val:>11.2%} {std_val:>11.2%} {min_val:>11.2%} {max_val:>11.2%}")

    # Best sector frequency analysis
    print("\n" + "=" * 80)
    print("SECTOR FREQUENCY IN TOP 20 COMBINATIONS")
    print("=" * 80)

    top_20_sectors = []
    for sectors in results_df.head(20)['Sector List']:
        top_20_sectors.extend(sectors)

    sector_counts = pd.Series(top_20_sectors).value_counts()
    for sector, count in sector_counts.items():
        print(f"  {sector}: {count}/20 appearances")

    # Compare to benchmark and equal weight all 9
    print("\n" + "=" * 80)
    print("COMPARISON TO BENCHMARKS")
    print("=" * 80)

    # Equal weight all 9 sectors
    all_metrics, _ = test_combination(returns, spx_returns, sector_list)
    print(f"\nEqual Weight All 9 Sectors:")
    print(f"  Ann. Return: {all_metrics['Ann. Return']:.2%}")
    print(f"  Sharpe:      {all_metrics['Sharpe Ratio']:.3f}")
    print(f"  Max DD:      {all_metrics['Max Drawdown']:.2%}")
    print(f"  Hit Ratio:   {all_metrics['Hit Ratio']:.1%}")

    # S&P 500 benchmark
    aligned_spx = spx_returns.loc[returns.index[24:]]
    spx_total = (1 + aligned_spx).prod() - 1
    spx_ann = (1 + spx_total) ** (12 / len(aligned_spx)) - 1
    print(f"\nS&P 500 Benchmark:")
    print(f"  Ann. Return: {spx_ann:.2%}")

    # Best 4-sector combo
    best = results_df.iloc[0]
    print(f"\nBest 4-Sector Combo ({best['Sectors']}):")
    print(f"  Ann. Return: {best['Ann. Return']:.2%}")
    print(f"  Sharpe:      {best['Sharpe Ratio']:.3f}")
    print(f"  Max DD:      {best['Max Drawdown']:.2%}")
    print(f"  Hit Ratio:   {best['Hit Ratio']:.1%}")

    # Save results
    save_df = results_df.drop(columns=['Sector List'])
    save_df.to_csv('four_sector_combinations.csv', index=False)
    logger.info(f"\nResults saved to: four_sector_combinations.csv")

    return results_df


if __name__ == "__main__":
    results = main()

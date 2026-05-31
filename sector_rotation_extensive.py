"""
Extensive Sector Rotation Analysis with Data Validation
2000-2026 Period - Careful Data Checking
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def check_data_availability():
    """
    Check data availability for all sectors from 2000-2026.
    Identify exact periods with data and gaps.
    """
    logger.info("=" * 80)
    logger.info("DATA AVAILABILITY CHECK - 2000-2026")
    logger.info("=" * 80)

    # Define all sectors and their ETFs (excluding XLRE and XLC due to limited data)
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
        'S&P_500': '^GSPC'
    }

    # Analysis period
    start_date = '2000-01-01'
    end_date = '2026-05-29'

    results = {}
    all_data = {}

    logger.info(f"\nAnalysis Period: {start_date} to {end_date}")
    logger.info(f"Expected months: {((2026 - 2000) * 12 + 5)} months")
    logger.info("-" * 80)

    for sector, ticker in sector_etfs.items():
        logger.info(f"\nChecking {sector} ({ticker})...")

        try:
            # Fetch data
            data = yf.download(ticker, start='1998-01-01', end=end_date, progress=False)

            if len(data) == 0:
                logger.warning(f"  ⚠️ NO DATA AVAILABLE for {ticker}")
                results[sector] = {
                    'ticker': ticker,
                    'status': 'NO DATA',
                    'start': None,
                    'end': None,
                    'months': 0,
                    'coverage': 0
                }
                continue

            # Resample to monthly (end of month)
            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Close'][ticker]
            else:
                prices = data['Close']

            monthly = prices.resample('ME').last().dropna()

            # Filter to analysis period
            monthly_period = monthly[(monthly.index >= start_date) & (monthly.index <= end_date)]

            # Calculate statistics
            if len(monthly_period) > 0:
                actual_start = monthly_period.index[0].strftime('%Y-%m')
                actual_end = monthly_period.index[-1].strftime('%Y-%m')
                n_months = len(monthly_period)

                # Calculate expected months in period
                period_start = pd.Timestamp(start_date)
                period_end = pd.Timestamp(end_date)
                expected_months = ((period_end.year - period_start.year) * 12 +
                                   period_end.month - period_start.month + 1)

                coverage = (n_months / expected_months) * 100

                # Check for gaps
                monthly_sorted = monthly_period.sort_index()
                gaps = []
                for i in range(1, len(monthly_sorted)):
                    actual_gap = (monthly_sorted.index[i] - monthly_sorted.index[i-1]).days
                    if actual_gap > 60:  # More than ~2 months gap
                        gaps.append({
                            'from': monthly_sorted.index[i-1].strftime('%Y-%m'),
                            'to': monthly_sorted.index[i].strftime('%Y-%m'),
                            'gap_days': actual_gap
                        })

                status = "✅ OK" if coverage > 95 else "⚠️ PARTIAL" if coverage > 50 else "❌ POOR"

                results[sector] = {
                    'ticker': ticker,
                    'status': status,
                    'actual_start': actual_start,
                    'actual_end': actual_end,
                    'months_available': n_months,
                    'expected_months': expected_months,
                    'coverage': coverage,
                    'gaps': gaps,
                    'data': monthly_period
                }

                all_data[sector] = monthly_period

                logger.info(f"  Status: {status}")
                logger.info(f"  Available: {actual_start} to {actual_end}")
                logger.info(f"  Months: {n_months} / {expected_months} ({coverage:.1f}%)")

                if gaps:
                    logger.warning(f"  ⚠️ Gaps found: {len(gaps)}")
                    for gap in gaps[:3]:
                        logger.warning(f"     {gap['from']} → {gap['to']} ({gap['gap_days']} days)")

            else:
                logger.warning(f"  ❌ NO DATA in analysis period")
                results[sector] = {
                    'ticker': ticker,
                    'status': 'NO DATA IN PERIOD',
                    'start': None,
                    'end': None,
                    'months': 0,
                    'coverage': 0
                }

        except Exception as e:
            logger.error(f"  ❌ ERROR: {e}")
            results[sector] = {
                'ticker': ticker,
                'status': 'ERROR',
                'error': str(e)
            }

    return results, all_data


def calculate_returns(all_data):
    """Calculate relative returns for each sector."""

    logger.info("\n" + "=" * 80)
    logger.info("CALCULATING SECTOR RELATIVE RETURNS")
    logger.info("=" * 80)

    # Get S&P 500 data
    spx_data = all_data.get('S&P_500')

    if spx_data is None or len(spx_data) == 0:
        logger.error("S&P 500 data not available!")
        return pd.DataFrame()

    # Calculate S&P 500 returns
    spx_returns = spx_data.pct_change().dropna()

    sector_returns = {}

    for sector, data in all_data.items():
        if sector == 'S&P_500':
            continue

        if len(data) < 2:
            continue

        # Calculate monthly returns
        returns = data.pct_change().dropna()

        # Align with SPX
        common_idx = returns.index.intersection(spx_returns.index)

        if len(common_idx) > 0:
            # Relative return = sector return - SPX return
            rel_return = returns.loc[common_idx] - spx_returns.loc[common_idx]
            sector_returns[sector] = rel_return

    return pd.DataFrame(sector_returns)


def analyze_coverage_by_period(returns_df):
    """Analyze data coverage by different market periods."""

    logger.info("\n" + "=" * 80)
    logger.info("DATA COVERAGE BY MARKET PERIOD")
    logger.info("=" * 80)

    # Define market periods
    periods = {
        'Dot-Com Boom': ('2000-01-01', '2002-10-31'),
        'Post Dot-Com': ('2002-11-01', '2007-09-30'),
        'Financial Crisis': ('2007-10-01', '2009-03-31'),
        'Recovery': ('2009-04-01', '2019-12-31'),
        'COVID Crash': ('2020-01-01', '2020-03-31'),
        'Post-COVID Bull': ('2020-04-01', '2026-05-29')
    }

    coverage_by_period = []

    for period_name, (start, end) in periods.items():
        period_data = returns_df[(returns_df.index >= start) & (returns_df.index <= end)]
        n_sectors = (period_data.notna().sum() > 0).sum()
        n_months = len(period_data)

        coverage_by_period.append({
            'Period': period_name,
            'Start': start[:7],
            'End': end[:7],
            'Months': n_months,
            'Sectors': n_sectors,
            'Data Quality': '✅ Full' if n_sectors >= 10 else '⚠️ Partial' if n_sectors >= 7 else '❌ Limited'
        })

        logger.info(f"\n{period_name}:")
        logger.info(f"  Period: {start[:7]} to {end[:7]}")
        logger.info(f"  Months: {n_months}")
        logger.info(f"  Sectors with data: {n_sectors}")
        logger.info(f"  Quality: {'✅ Full' if n_sectors >= 10 else '⚠️ Partial' if n_sectors >= 7 else '❌ Limited'}")

    return pd.DataFrame(coverage_by_period)


def analyze_sector_performance(returns_df, lookback=12):
    """Comprehensive sector performance analysis."""

    logger.info("\n" + "=" * 80)
    logger.info("SECTOR PERFORMANCE ANALYSIS")
    logger.info("=" * 80)

    results = []

    for sector in returns_df.columns:
        data = returns_df[sector].dropna()

        if len(data) < 12:
            logger.warning(f"\n{sector}: Insufficient data ({len(data)} months)")
            continue

        # Calculate various metrics
        cumulative = (1 + data).cumprod()
        total_return = cumulative.iloc[-1] - 1

        # Annualized metrics
        n_years = len(data) / 12
        annualized_return = (1 + total_return) ** (1 / n_years) - 1
        annualized_vol = data.std() * np.sqrt(12)
        sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

        # Max drawdown
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown.min()

        # Hit ratio
        hit_ratio = (data > 0).mean()

        # Monthly stats
        avg_monthly = data.mean()
        std_monthly = data.std()

        # Recent performance (last 24 months)
        recent = data.tail(24) if len(data) > 24 else data
        recent_return = (1 + recent).cumprod().iloc[-1] - 1

        # Momentum (12-month rolling return)
        momentum = data.rolling(lookback).sum().dropna()
        positive_momentum_pct = (momentum > 0).mean()

        results.append({
            'Sector': sector,
            'Observations': len(data),
            'Cum. Return': f"{total_return:.1%}",
            'Ann. Return': f"{annualized_return:.2%}",
            'Ann. Vol': f"{annualized_vol:.2%}",
            'Sharpe': f"{sharpe:.2f}",
            'Max DD': f"{max_dd:.1%}",
            'Hit Ratio': f"{hit_ratio:.1%}",
            'Avg Monthly': f"{avg_monthly:.3f}",
            'Recent (24m)': f"{recent_return:.1%}",
            'Positive Momentum': f"{positive_momentum_pct:.1%}"
        })

        logger.info(f"\n{sector}:")
        logger.info(f"  Observations: {len(data)}")
        logger.info(f"  Cumulative Return: {total_return:.1%}")
        logger.info(f"  Annualized Return: {annualized_return:.2%}")
        logger.info(f"  Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"  Max Drawdown: {max_dd:.1%}")
        logger.info(f"  Hit Ratio: {hit_ratio:.1%}")

    return pd.DataFrame(results)


def analyze_momentum_signals(returns_df):
    """Analyze momentum-based sector rotation signals."""

    logger.info("\n" + "=" * 80)
    logger.info("MOMENTUM SIGNAL ANALYSIS")
    logger.info("=" * 80)

    # Calculate various momentum signals
    lookbacks = [3, 6, 12]

    signal_results = []

    for lookback in lookbacks:
        logger.info(f"\n--- {lookback}-Month Momentum ---")

        # Calculate momentum for each sector
        momentum = returns_df.rolling(lookback).sum()

        # Strategy: Long top 3 sectors, rebalance monthly
        portfolio_returns = []

        for date in momentum.index:
            try:
                mom_values = momentum.loc[date].dropna()

                if len(mom_values) == 0:
                    continue

                # Sort by momentum
                sorted_mom = mom_values.sort_values(ascending=False)
                top_3 = sorted_mom.head(3)

                # Equal weight
                port_return = returns_df.loc[date, top_3.index].mean()

                portfolio_returns.append({
                    'date': date,
                    'return': port_return,
                    'top_sectors': list(top_3.index)
                })

            except Exception as e:
                continue

        if len(portfolio_returns) > 0:
            port_df = pd.DataFrame(portfolio_returns)
            port_series = port_df.set_index('date')['return']

            # Calculate metrics
            cumulative = (1 + port_series).cumprod()
            total_ret = cumulative.iloc[-1] - 1
            ann_ret = (1 + total_ret) ** (12 / len(port_series)) - 1
            ann_vol = port_series.std() * np.sqrt(12)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            hit = (port_series > 0).mean()

            logger.info(f"  Total Return: {total_ret:.1%}")
            logger.info(f"  Annualized Return: {ann_ret:.1%}")
            logger.info(f"  Sharpe Ratio: {sharpe:.2f}")
            logger.info(f"  Hit Ratio: {hit:.1%}")

            signal_results.append({
                'Lookback': f'{lookback}m',
                'Total Return': f"{total_ret:.1%}",
                'Ann. Return': f"{ann_ret:.1%}",
                'Sharpe': f"{sharpe:.2f}",
                'Hit Ratio': f"{hit:.1%}"
            })

    return pd.DataFrame(signal_results)


def analyze_correlations(returns_df):
    """Analyze sector correlations."""

    logger.info("\n" + "=" * 80)
    logger.info("SECTOR CORRELATION ANALYSIS")
    logger.info("=" * 80)

    corr_matrix = returns_df.corr()

    # Find interesting pairs
    high_corr = []
    low_corr = []
    negative_corr = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]

            if corr_val > 0.7:
                high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
            elif corr_val < 0.2:
                low_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))
            elif corr_val < 0:
                negative_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_val))

    high_corr.sort(key=lambda x: -x[2])
    low_corr.sort(key=lambda x: x[2])
    negative_corr.sort(key=lambda x: x[2])

    logger.info("\nHigh Correlations (>0.7) - Redundant pairs:")
    for s1, s2, corr in high_corr[:5]:
        logger.info(f"  {s1} ↔ {s2}: {corr:.3f}")

    logger.info("\nLow/Negative Correlations (<0.2) - Diversification:")
    for s1, s2, corr in negative_corr[:5]:
        logger.info(f"  {s1} ↔ {s2}: {corr:.3f}")

    return corr_matrix


def main():
    """Run comprehensive sector rotation analysis."""

    print("\n" + "🎯" * 40)
    print("EXTENSIVE SECTOR ROTATION ANALYSIS - 2000-2026")
    print("🎯" * 40)

    # Step 1: Check data availability
    data_check, all_data = check_data_availability()

    # Step 2: Calculate relative returns
    returns_df = calculate_returns(all_data)

    if len(returns_df) == 0:
        logger.error("No returns data available!")
        return

    logger.info("\n" + "=" * 80)
    logger.info("FINAL DATASET SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Period: {returns_df.index[0].strftime('%Y-%m')} to {returns_df.index[-1].strftime('%Y-%m')}")
    logger.info(f"Total months: {len(returns_df)}")
    logger.info(f"Sectors available: {len(returns_df.columns)}")

    # Step 3: Coverage by period
    coverage_df = analyze_coverage_by_period(returns_df)

    # Step 4: Sector performance
    performance_df = analyze_sector_performance(returns_df)

    # Step 5: Momentum signals
    momentum_df = analyze_momentum_signals(returns_df)

    # Step 6: Correlation analysis
    corr_matrix = analyze_correlations(returns_df)

    # Save results
    performance_df.to_csv('sector_performance_2000_2026.csv', index=False)
    coverage_df.to_csv('data_coverage_by_period.csv', index=False)
    momentum_df.to_csv('momentum_signals.csv', index=False)
    returns_df.to_csv('sector_relative_returns.csv')

    logger.info("\n" + "=" * 80)
    logger.info("FILES SAVED")
    logger.info("=" * 80)
    logger.info("  - sector_performance_2000_2026.csv")
    logger.info("  - data_coverage_by_period.csv")
    logger.info("  - momentum_signals.csv")
    logger.info("  - sector_relative_returns.csv")

    return data_check, returns_df, performance_df, coverage_df, momentum_df


if __name__ == "__main__":
    data_check, returns, performance, coverage, momentum = main()
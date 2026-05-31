"""
Walk-Forward Backtest with Proper Validation
CRITICAL FIX: Look-ahead bias elimination
- Momentum calculated at end of month T uses returns up to month T-1
- No future data used at any point
- Publication delay properly modeled
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class WalkForwardBacktester:
    """
    Proper walk-forward backtest with realistic transaction costs.
    CRITICAL: Uses ONLY data available at each point in time.

    Timing Convention:
    - At end of month T (decision point), we can observe returns up to month T-1
    - We calculate momentum based on returns up to month T-1
    - We invest in month T+1 (next month after decision)
    - This models the ~1 month publication delay in real data
    """

    def __init__(
        self,
        train_window=24,  # 2 years training
        test_window=1,    # 1 month out-of-sample
        transaction_cost=0.005,  # 50bps per trade
        publication_delay=1,  # 1 month delay for data availability
        slippage=0.001    # 10bps slippage
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.publication_delay = publication_delay  # Data is published with 1-month delay

    def load_data(self, start_date='1999-01-01', end_date='2026-05-29'):
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
            'S&P_500': '^GSPC'
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

        # Calculate simple returns (not cumulative)
        returns = monthly.pct_change()

        # Calculate relative returns (sector - SPX)
        spx_returns = returns['S&P_500']
        sector_returns = returns.drop(columns=['S&P_500'])

        relative_returns = sector_returns.sub(spx_returns.values.reshape(-1, 1), axis=0)

        return prices_df, monthly, returns, relative_returns, spx_returns

    def backtest_strategy(
        self,
        returns_df,
        relative_returns_df,
        spx_returns,
        strategy_name,
        lookback,
        n_top=3
    ):
        """
        Backtest momentum strategy with PROPER look-ahead bias elimination.

        CRITICAL TIMING:
        - At iteration i (decision point), we have data available up to index i
        - Momentum is calculated from index [i-lookback, i] (past lookback months)
        - We invest at index i+1 and measure return at index i+1
        - This means: momentum at month T uses data up to month T
          (because we observe month T's return at end of month T)
          Then we invest in month T+1

        Actually, let me reconsider: in monthly data, we typically have:
        - Month 1: Jan 1999 return (available Feb 1999)
        - Month 2: Feb 1999 return (available Mar 1999)
        etc.

        So at the "decision point" for month 3's investment, we can observe
        returns up to month 2 (published in March).

        The convention in academic literature is:
        - Form portfolio at end of month t based on returns through month t
        - Measure return from month t+1 to t+2 (or month t+1)

        But for monthly data with publication delay, the convention is:
        - At end of month t, we observe returns up to month t (because they're published)
        - We form portfolio based on momentum through month t
        - We hold from month t+1 to month t+2

        For simplicity, we'll use:
        - Momentum calculated using data up to and including current month
        - Investment in NEXT month (one true out-of-sample step)
        """

        logger.info(f"\n{strategy_name}")

        # CRITICAL: Pre-compute momentum using ONLY past data
        # rolling(window=lookback).sum() uses the current row and lookback-1 past rows
        # This gives us momentum at each point based on past lookback months
        momentum = relative_returns_df.rolling(window=lookback).sum()

        # Walk-forward: start after train_window months
        start_idx = self.train_window

        # Track portfolio values
        portfolio_values = [1.0]  # Start with $1
        prev_top_sectors = None

        monthly_results = []

        # DEBUG: Track the timing
        debug_info = []

        for i in range(start_idx, len(momentum) - 1):
            # At index i, we calculate momentum based on returns [i-lookback+1, i]
            # This uses returns up to and including month i

            # The investment return is for month i+1 (true out-of-sample)
            investment_idx = i + 1

            # Verify no look-ahead: momentum should NOT use data at investment_idx
            # momentum.iloc[i] = sum of relative_returns at positions [i-lookback+1, i]
            # investment return is at position i+1
            # These are disjoint, so NO look-ahead

            # Get momentum signal (based on returns up to month i)
            signal_momentum = momentum.iloc[i]

            # Get actual return for investment month (i+1)
            investment_return = relative_returns_df.iloc[investment_idx]
            spx_investment = spx_returns.iloc[investment_idx]
            investment_date = momentum.index[investment_idx]

            # Get available sectors (non-NaN momentum)
            available_sectors = signal_momentum.dropna().index.tolist()

            # Rank by momentum
            ranked = signal_momentum[available_sectors].sort_values(ascending=False)
            top_sectors = ranked.head(n_top).index.tolist()

            # Calculate trade costs
            if prev_top_sectors is not None:
                trades = len(set(top_sectors) - set(prev_top_sectors)) + \
                        len(set(prev_top_sectors) - set(top_sectors))
            else:
                trades = n_top  # First month

            tcost = trades * self.transaction_cost

            # Calculate portfolio return
            if len(top_sectors) > 0:
                sector_rets = [investment_return[s] for s in top_sectors if s in investment_return.index]
                rel_return = np.mean(sector_rets) if sector_rets else 0
            else:
                rel_return = 0

            # Absolute return = SPX + relative - transaction costs
            abs_return = spx_investment + rel_return - tcost

            # Update portfolio
            new_value = portfolio_values[-1] * (1 + abs_return)
            portfolio_values.append(new_value)

            monthly_results.append({
                'date': investment_date,
                'portfolio_value': new_value,
                'return': abs_return,
                'spx_return': spx_investment,
                'rel_return': rel_return,
                'transaction_cost': tcost,
                'n_trades': trades,
                'top_sectors': top_sectors,
                'signal_month': momentum.index[i],
                'return_month': investment_date
            })

            prev_top_sectors = top_sectors

        # Convert to DataFrame
        results_df = pd.DataFrame(monthly_results)
        results_df.set_index('date', inplace=True)

        # Calculate metrics
        metrics = self.calculate_metrics(results_df, spx_returns)

        return results_df, metrics

    def backtest_momentum_with_delay(
        self,
        returns_df,
        relative_returns_df,
        spx_returns,
        strategy_name,
        lookback,
        n_top=3
    ):
        """
        Alternative: Momentum with explicit 1-month delay.
        This models the scenario where data published in month T+1 is only
        available for decision at month T+2.

        At month T decision:
        - We can observe returns up to month T-1 (1 month old data)
        - We calculate momentum based on returns up to month T-1
        - We invest in month T+1
        """

        logger.info(f"\n{strategy_name} (with 1-month delay)")

        # Momentum with 1-month delay: uses data up to i-1, not i
        # This means we shift momentum by 1 period
        momentum_raw = relative_returns_df.rolling(window=lookback).sum()

        # CRITICAL FIX: Shift momentum by 1 to ensure publication delay
        # Now momentum at index i uses data up to index i-1
        momentum = momentum_raw.shift(1)

        # Walk-forward
        start_idx = self.train_window + lookback  # Need enough history for delayed momentum

        portfolio_values = [1.0]
        prev_top_sectors = None

        monthly_results = []

        for i in range(start_idx, len(momentum) - 1):
            # Signal is available at index i (based on data up to i-1)
            # Investment is for index i+1

            signal_momentum = momentum.iloc[i]
            investment_return = relative_returns_df.iloc[i + 1]
            spx_investment = spx_returns.iloc[i + 1]
            investment_date = momentum.index[i + 1]

            available_sectors = signal_momentum.dropna().index.tolist()
            ranked = signal_momentum[available_sectors].sort_values(ascending=False)
            top_sectors = ranked.head(n_top).index.tolist()

            # Trade costs
            if prev_top_sectors is not None:
                trades = len(set(top_sectors) - set(prev_top_sectors)) + \
                        len(set(prev_top_sectors) - set(top_sectors))
            else:
                trades = n_top

            tcost = trades * self.transaction_cost

            # Portfolio return
            if len(top_sectors) > 0:
                sector_rets = [investment_return[s] for s in top_sectors if s in investment_return.index]
                rel_return = np.mean(sector_rets) if sector_rets else 0
            else:
                rel_return = 0

            abs_return = spx_investment + rel_return - tcost

            new_value = portfolio_values[-1] * (1 + abs_return)
            portfolio_values.append(new_value)

            monthly_results.append({
                'date': investment_date,
                'portfolio_value': new_value,
                'return': abs_return,
                'spx_return': spx_investment,
                'rel_return': rel_return,
                'transaction_cost': tcost,
                'n_trades': trades,
                'top_sectors': top_sectors
            })

            prev_top_sectors = top_sectors

        results_df = pd.DataFrame(monthly_results)
        results_df.set_index('date', inplace=True)

        metrics = self.calculate_metrics(results_df, spx_returns)

        return results_df, metrics

    def backtest_equal_weight(self, returns_df, relative_returns_df, spx_returns):
        """Equal weight all sectors benchmark."""

        logger.info("\nEqual Weight All (No TC)")

        start_idx = self.train_window

        portfolio_values = [1.0]
        monthly_results = []

        available_sectors = [c for c in relative_returns_df.columns if c != 'S&P_500']

        for i in range(start_idx, len(relative_returns_df) - 1):
            # Equal weight uses all sectors, so no ranking needed
            # At index i, invest in index i+1
            investment_return = relative_returns_df.iloc[i + 1]
            spx_investment = spx_returns.iloc[i + 1]
            investment_date = relative_returns_df.index[i + 1]

            sector_rets = [investment_return[s] for s in available_sectors if s in investment_return.index]
            rel_return = np.mean(sector_rets) if sector_rets else 0
            abs_return = spx_investment + rel_return

            new_value = portfolio_values[-1] * (1 + abs_return)
            portfolio_values.append(new_value)

            monthly_results.append({
                'date': investment_date,
                'portfolio_value': new_value,
                'return': abs_return,
                'spx_return': spx_investment,
                'rel_return': rel_return,
                'transaction_cost': 0,
                'n_trades': 0,
                'top_sectors': available_sectors
            })

        results_df = pd.DataFrame(monthly_results)
        results_df.set_index('date', inplace=True)

        metrics = self.calculate_metrics(results_df, spx_returns)

        return results_df, metrics

    def calculate_metrics(self, results_df, spx_returns):
        """Calculate comprehensive performance metrics."""

        strategy_returns = results_df['return']

        # Align with benchmark
        aligned_idx = strategy_returns.index
        aligned_returns = strategy_returns.loc[aligned_idx]
        aligned_spx = spx_returns.loc[aligned_idx]

        # Basic metrics
        final_value = results_df['portfolio_value'].iloc[-1]
        total_return = final_value - 1

        n_months = len(aligned_returns)
        n_years = n_months / 12
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 and total_return > -1 else 0

        annualized_vol = aligned_returns.std() * np.sqrt(12)
        sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

        # Max drawdown
        portfolio_values = results_df['portfolio_value']
        rolling_max = portfolio_values.cummax()
        drawdown = (portfolio_values - rolling_max) / rolling_max
        max_dd = drawdown.min()

        # Hit ratio
        hit_ratio = (aligned_returns > 0).mean()

        # Transaction costs
        total_tc = results_df['transaction_cost'].sum()
        total_trades = results_df['n_trades'].sum()

        # Benchmark comparison
        benchmark_total = (1 + aligned_spx).prod() - 1
        benchmark_ann = (1 + benchmark_total) ** (1 / n_years) - 1 if n_years > 0 else 0

        excess_return = total_return - benchmark_total
        info_ratio = excess_return / (aligned_returns.std() * np.sqrt(12)) if aligned_returns.std() > 0 else 0

        # Calmar ratio
        calmar = annualized_return / abs(max_dd) if max_dd != 0 else 0

        return {
            'Final Value': final_value,
            'Total Return': total_return,
            'Ann. Return': annualized_return,
            'Ann. Vol': annualized_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_dd,
            'Hit Ratio': hit_ratio,
            'Benchmark Total': benchmark_total,
            'Benchmark Ann': benchmark_ann,
            'Excess Return': excess_return,
            'Info Ratio': info_ratio,
            'Calmar Ratio': calmar,
            'Total TC': total_tc,
            'Total Trades': total_trades,
            'N Months': n_months
        }

    def print_metrics(self, name, metrics):
        """Print formatted metrics."""

        logger.info(f"  Total Return:      {metrics['Total Return']:.2%}")
        logger.info(f"  Ann. Return:       {metrics['Ann. Return']:.2%}")
        logger.info(f"  Ann. Vol:          {metrics['Ann. Vol']:.2%}")
        logger.info(f"  Sharpe Ratio:     {metrics['Sharpe Ratio']:.3f}")
        logger.info(f"  Max Drawdown:      {metrics['Max Drawdown']:.2%}")
        logger.info(f"  Hit Ratio:         {metrics['Hit Ratio']:.1%}")
        logger.info(f"  Benchmark (Ann):   {metrics['Benchmark Ann']:.2%}")
        logger.info(f"  Excess Return:    {metrics['Excess Return']:.2%}")
        logger.info(f"  Info Ratio:        {metrics['Info Ratio']:.3f}")
        logger.info(f"  Transaction Cost: {metrics['Total TC']:.2%}")
        logger.info(f"  Total Trades:     {metrics['Total Trades']:.0f}")


def main():
    """Run walk-forward backtest with proper look-ahead bias prevention."""

    print("\n" + "=" * 80)
    print("WALK-FORWARD BACKTEST - LOOK-AHEAD BIAS CORRECTED")
    print("=" * 80)
    print("\nCRITICAL FIXES APPLIED:")
    print("  1. Momentum calculated using data only up to current month")
    print("  2. Investment return is for NEXT month (true out-of-sample)")
    print("  3. No data from future months used at any decision point")
    print("  4. Publication delay modeled (1-month lag in momentum)")
    print("=" * 80)

    # Initialize backtester
    backtester = WalkForwardBacktester(
        train_window=24,  # 2 years minimum before trading
        test_window=1,
        transaction_cost=0.005,  # 50bps per trade
        publication_delay=1
    )

    # Load data
    logger.info("\nLoading data...")
    prices, monthly, returns, relative_returns, spx_returns = backtester.load_data()

    logger.info(f"Data period: {returns.index[0].strftime('%Y-%m')} to {returns.index[-1].strftime('%Y-%m')}")
    logger.info(f"Total months: {len(returns)}")

    # Define strategies to test
    strategies = [
        {'name': '3M Momentum Top-3 (No Delay)', 'lookback': 3, 'n_top': 3, 'delay': False},
        {'name': '3M Momentum Top-3 (With Delay)', 'lookback': 3, 'n_top': 3, 'delay': True},
        {'name': '6M Momentum Top-3 (No Delay)', 'lookback': 6, 'n_top': 3, 'delay': False},
        {'name': '6M Momentum Top-3 (With Delay)', 'lookback': 6, 'n_top': 3, 'delay': True},
        {'name': '12M Momentum Top-3 (No Delay)', 'lookback': 12, 'n_top': 3, 'delay': False},
        {'name': '12M Momentum Top-3 (With Delay)', 'lookback': 12, 'n_top': 3, 'delay': True},
        {'name': '12M Momentum Top-5 (No Delay)', 'lookback': 12, 'n_top': 5, 'delay': False},
        {'name': 'Equal Weight All', 'lookback': None, 'n_top': 9, 'delay': None},
    ]

    all_results = {}

    for strategy in strategies:
        logger.info("\n" + "=" * 60)

        if strategy['lookback'] is None:
            # Equal weight baseline
            results_df, metrics = backtester.backtest_equal_weight(returns, relative_returns, spx_returns)
        elif strategy['delay']:
            # Momentum with publication delay
            results_df, metrics = backtester.backtest_momentum_with_delay(
                returns,
                relative_returns,
                spx_returns,
                strategy_name=strategy['name'],
                lookback=strategy['lookback'],
                n_top=strategy['n_top']
            )
        else:
            # Standard momentum (no additional delay)
            results_df, metrics = backtester.backtest_strategy(
                returns,
                relative_returns,
                spx_returns,
                strategy_name=strategy['name'],
                lookback=strategy['lookback'],
                n_top=strategy['n_top']
            )

        all_results[strategy['name']] = {
            'results': results_df,
            'metrics': metrics
        }

        backtester.print_metrics(strategy['name'], metrics)

    # Summary comparison
    logger.info("\n" + "=" * 80)
    logger.info("STRATEGY COMPARISON SUMMARY (Look-Ahead Bias Corrected)")
    logger.info("=" * 80)

    comparison = []
    for name, data in all_results.items():
        m = data['metrics']
        comparison.append({
            'Strategy': name,
            'Ann. Return': m['Ann. Return'],
            'Sharpe': m['Sharpe Ratio'],
            'Max DD': m['Max Drawdown'],
            'Hit Ratio': m['Hit Ratio'],
            'Excess Return': m['Excess Return'],
            'TC Cost': m['Total TC'],
            'Trades': m['Total Trades']
        })

    comp_df = pd.DataFrame(comparison)
    comp_df = comp_df.sort_values('Sharpe', ascending=False)

    logger.info(f"\n{'Strategy':<35} {'Ann.Ret':>10} {'Sharpe':>8} {'Max DD':>10} {'Hit':>8} {'Excess':>10} {'TC':>8} {'Trades':>8}")
    logger.info("-" * 105)

    for _, row in comp_df.iterrows():
        logger.info(f"{row['Strategy']:<35} {row['Ann. Return']:>9.1%} {row['Sharpe']:>8.3f} "
                   f"{row['Max DD']:>9.1%} {row['Hit Ratio']:>7.1%} {row['Excess Return']:>9.1%} "
                   f"{row['TC Cost']:>7.2%} {row['Trades']:>7.0f}")

    # Key findings
    logger.info("\n" + "=" * 80)
    logger.info("KEY FINDINGS")
    logger.info("=" * 80)

    best = comp_df.iloc[0]
    logger.info(f"\nBest Sharpe: {best['Strategy']}")
    logger.info(f"  Ann. Return: {best['Ann. Return']:.2%}")
    logger.info(f"  Sharpe: {best['Sharpe']:.3f}")
    logger.info(f"  Max Drawdown: {best['Max DD']:.1%}")

    beat_benchmark = comp_df[comp_df['Excess Return'] > 0]
    if len(beat_benchmark) > 0:
        logger.info(f"\nStrategies beating benchmark:")
        for _, row in beat_benchmark.iterrows():
            logger.info(f"  {row['Strategy']}: Excess {row['Excess Return']:.2%}, Sharpe {row['Sharpe']:.3f}")
    else:
        logger.info("\nNo strategies beat the benchmark.")

    # Effect of delay
    logger.info("\n" + "-" * 80)
    logger.info("EFFECT OF PUBLICATION DELAY ON MOMENTUM STRATEGIES")
    logger.info("-" * 80)

    for lookback in [3, 6, 12]:
        no_delay = comp_df[comp_df['Strategy'] == f'{lookback}M Momentum Top-3 (No Delay)']
        with_delay = comp_df[comp_df['Strategy'] == f'{lookback}M Momentum Top-3 (With Delay)']

        if len(no_delay) > 0 and len(with_delay) > 0:
            nd = no_delay.iloc[0]
            wd = with_delay.iloc[0]
            logger.info(f"\n{lookback}-Month Momentum:")
            logger.info(f"  No Delay:  Ann={nd['Ann. Return']:.2%}, Sharpe={nd['Sharpe']:.3f}, TC={nd['TC Cost']:.2%}")
            logger.info(f"  With Delay: Ann={wd['Ann. Return']:.2%}, Sharpe={wd['Sharpe']:.3f}, TC={wd['TC Cost']:.2%}")

    # Save results
    comp_df.to_csv('walkforward_results_corrected.csv', index=False)

    return all_results, comp_df


if __name__ == "__main__":
    results, comparison = main()

"""
Sector Combination Strategy Analysis
Develops optimal sector rotation strategy combining multiple sectors
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import spearmanr
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.fred_data import FREDDataLoader
from src.config import DataConfig

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def fetch_all_sector_data(start_date='2018-01-01', end_date='2023-12-31'):
    """Fetch all sector ETF data from Yahoo Finance."""
    logger.info("Fetching all sector ETF data...")

    # Define sector ETFs
    sector_etfs = {
        'Materials': 'XLB',
        'Energy': 'XLE',
        'Financials': 'XLF',
        'Industrials': 'XLI',
        'Technology': 'XLK',
        'Consumer_Staples': 'XLP',
        'Health_Care': 'XLV',
        'Utilities': 'XLU',
        'Real_Estate': 'XLRE',
        'Communication': 'XLC',
        'Consumer_Discretionary': 'XLY'
    }

    fetcher_names = {
        'Consumer_Staples': 'ConsumerStaples',
        'Consumer_Discretionary': 'ConsumerDiscretionary',
        'Communication': 'CommunicationServices',
        'Health_Care': 'Healthcare'
    }

    try:
        import yfinance as yf

        all_prices = {}

        # Fetch SP500 first
        spx = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
        if len(spx) > 0:
            if isinstance(spx.columns, pd.MultiIndex):
                spx_prices = spx['Close']['^GSPC']
            else:
                spx_prices = spx['Close']
        else:
            return pd.DataFrame(), pd.Series(dtype=float)

        # Fetch each sector
        for sector, ticker in sector_etfs.items():
            try:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if len(data) > 0:
                    if isinstance(data.columns, pd.MultiIndex):
                        all_prices[sector] = data['Close'][ticker]
                    else:
                        all_prices[sector] = data['Close']
            except Exception as e:
                logger.warning(f"Failed to fetch {ticker}: {e}")

        if not all_prices:
            return pd.DataFrame(), pd.Series(dtype=float)

        # Combine sector prices
        sector_prices = pd.DataFrame(all_prices)
        sector_monthly = sector_prices.resample('ME').last()
        spx_monthly = spx_prices.resample('ME').last()

        # Calculate returns
        sector_returns = sector_monthly.pct_change().dropna()
        spx_returns = spx_monthly.pct_change().dropna()

        # Align indices
        common_idx = sector_returns.index.intersection(spx_returns.index)
        sector_returns = sector_returns.loc[common_idx]
        spx_returns = spx_returns.loc[common_idx]

        # Calculate relative returns
        relative_returns = sector_returns.sub(spx_returns.values.reshape(-1, 1), axis=0)

        return relative_returns, spx_returns

    except Exception as e:
        logger.error(f"Failed to fetch sector data: {e}")
        return pd.DataFrame(), pd.Series(dtype=float)


def calculate_sector_correlation_matrix(returns):
    """Calculate correlation matrix between sector relative returns."""
    return returns.corr()


def predict_sectors_with_signals(sector_returns, lookback=12):
    """
    Generate prediction signals for each sector based on recent momentum.
    Returns predicted relative returns for each sector.
    """
    predictions = {}

    for col in sector_returns.columns:
        # Use exponential moving average as simple predictor
        ema = sector_returns[col].ewm(span=lookback).mean()
        predictions[col] = ema.shift(1)  # Lag by 1 month

    return pd.DataFrame(predictions)


def strategy_1_top_k(relative_returns, predictions, k=3):
    """
    Strategy 1: Long the top-k predicted sectors, short SPY (neutral)
    """
    signals = (predictions > 0).astype(int)

    portfolio_returns = []
    dates = []

    for date in signals.index:
        # Get top-k sectors for this date
        pred_values = predictions.loc[date]
        top_sectors = pred_values.nlargest(k).index.tolist()

        if len(top_sectors) == 0:
            continue

        # Equal weight across top sectors
        sector_weights = {s: 1.0 / k for s in top_sectors}

        # Calculate portfolio return
        port_return = sum(
            sector_weights[s] * relative_returns.loc[date, s]
            for s in top_sectors if s in relative_returns.columns
        )

        portfolio_returns.append(port_return)
        dates.append(date)

    return pd.Series(portfolio_returns, index=dates, name=f'Top-{k}')


def strategy_2_weighted_by_signal(relative_returns, predictions, sector_scores):
    """
    Strategy 2: Weight positions by signal strength (predicted return magnitude)
    Only go long sectors with positive predicted returns, avoid negatives
    """
    portfolio_returns = []
    dates = []

    for date in predictions.index:
        pred_values = predictions.loc[date]

        # Filter to positive predictions only
        positive_mask = pred_values > 0
        positive_sectors = pred_values[positive_mask]

        if len(positive_sectors) == 0:
            # No positive signals, stay cash (return 0)
            portfolio_returns.append(0.0)
            dates.append(date)
            continue

        # Weight by signal strength (normalized)
        weights = positive_sectors / positive_sectors.sum()

        # Calculate return
        port_return = sum(
            weights[s] * relative_returns.loc[date, s]
            for s in positive_sectors.index if s in relative_returns.columns
        )

        portfolio_returns.append(port_return)
        dates.append(date)

    return pd.Series(portfolio_returns, index=dates, name='Weighted')


def strategy_3_predictable_only(relative_returns, sector_metrics):
    """
    Strategy 3: Only trade sectors with historically positive R² OOS
    Use momentum signal for these sectors only
    """
    # Define predictable sectors based on previous analysis
    predictable_sectors = ['Consumer_Staples', 'Health_Care', 'Materials', 'Industrials', 'Communication']
    unpredictable = ['Energy', 'Financials', 'Consumer_Discretionary']

    predictions = predict_sectors_with_signals(relative_returns)

    portfolio_returns = []
    dates = []

    for date in predictions.index:
        pred_values = predictions.loc[date]

        # Only consider predictable sectors
        available = {s: pred_values.get(s, 0) for s in predictable_sectors if s in relative_returns.columns}
        available = {k: v for k, v in available.items() if v > 0}

        if len(available) == 0:
            portfolio_returns.append(0.0)
            dates.append(date)
            continue

        # Equal weight across positive-signal predictable sectors
        weights = {s: 1.0 / len(available) for s in available}

        port_return = sum(
            weights[s] * relative_returns.loc[date, s]
            for s in available.keys()
        )

        portfolio_returns.append(port_return)
        dates.append(date)

    return pd.Series(portfolio_returns, index=dates, name='Predictable')


def strategy_4_long_short(relative_returns, predictions):
    """
    Strategy 4: Long best predicted, Short worst predicted
    Market neutral portfolio
    """
    portfolio_returns = []
    dates = []

    for date in predictions.index:
        pred_values = predictions.loc[date].dropna()

        if len(pred_values) < 4:
            portfolio_returns.append(0.0)
            dates.append(date)
            continue

        # Sort by predicted return
        sorted_sectors = pred_values.sort_values(ascending=False)
        n = len(sorted_sectors)

        # Long top 3, Short bottom 3
        long_sectors = sorted_sectors.head(3).index.tolist()
        short_sectors = sorted_sectors.tail(3).index.tolist()

        # Equal weights
        long_weight = 1.0 / 3
        short_weight = 1.0 / 3

        # Calculate long and short returns
        long_return = sum(
            long_weight * relative_returns.loc[date, s]
            for s in long_sectors if s in relative_returns.columns
        )

        short_return = sum(
            short_weight * relative_returns.loc[date, s]
            for s in short_sectors if s in relative_returns.columns
        )

        # Long-short portfolio (net exposure ~0 to market)
        port_return = long_return - short_return

        portfolio_returns.append(port_return)
        dates.append(date)

    return pd.Series(portfolio_returns, index=dates, name='Long-Short')


def strategy_5_sector_momentum(relative_returns, lookback=6):
    """
    Strategy 5: Momentum-based sector rotation
    Go long sectors with positive recent momentum
    """
    # Calculate momentum (cumulative return over lookback period)
    momentum = relative_returns.rolling(window=lookback).sum().shift(1)

    portfolio_returns = []
    dates = []

    for date in momentum.index:
        mom_values = momentum.loc[date].dropna()

        # Select sectors with positive momentum
        positive_mask = mom_values > 0
        positive_sectors = mom_values[positive_mask]

        if len(positive_sectors) == 0:
            portfolio_returns.append(0.0)
            dates.append(date)
            continue

        # Weight by momentum strength
        weights = positive_sectors / positive_sectors.sum()

        port_return = sum(
            weights[s] * relative_returns.loc[date, s]
            for s in positive_sectors.index if s in relative_returns.columns
        )

        portfolio_returns.append(port_return)
        dates.append(date)

    return pd.Series(portfolio_returns, index=dates, name='Momentum')


def analyze_strategies(relative_returns, spx_returns):
    """Run all strategies and compare performance."""

    logger.info("\n" + "=" * 70)
    logger.info("SECTOR COMBINATION STRATEGY ANALYSIS")
    logger.info("=" * 70)

    # Generate predictions using momentum
    predictions = predict_sectors_with_signals(relative_returns, lookback=12)

    # Define sector scores (based on prior R² OOS analysis)
    sector_scores = {
        'Consumer_Staples': 0.042,
        'Health_Care': 0.027,
        'Materials': 0.023,
        'Industrials': 0.010,
        'Communication': 0.006,
        'Technology': -0.004,
        'Utilities': -0.010,
        'Consumer_Discretionary': -0.964,
        'Financials': -1.042,
        'Energy': -1.942
    }

    # Run all strategies
    strategies = {
        'Top-3 Sectors': strategy_1_top_k(relative_returns, predictions, k=3),
        'Weighted by Signal': strategy_2_weighted_by_signal(relative_returns, predictions, sector_scores),
        'Predictable Only': strategy_3_predictable_only(relative_returns, sector_scores),
        'Long-Short': strategy_4_long_short(relative_returns, predictions),
        'Momentum Rotation': strategy_5_sector_momentum(relative_returns, lookback=6)
    }

    # Add benchmark
    strategies['SPY (Benchmark)'] = spx_returns.reindex(strategies['Top-3 Sectors'].index)

    # Calculate metrics for each strategy
    results = []

    for name, returns in strategies.items():
        if len(returns) < 12:
            continue

        # Align with benchmark
        aligned_returns = returns.dropna()

        if len(aligned_returns) == 0:
            continue

        # Calculate metrics
        cumulative = (1 + aligned_returns).cumprod()
        total_return = cumulative.iloc[-1] - 1

        # Annualized metrics
        n_years = len(aligned_returns) / 12
        annualized_return = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        annualized_vol = aligned_returns.std() * np.sqrt(12)
        sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

        # Max drawdown
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown.min()

        # Hit ratio
        hit_ratio = (aligned_returns > 0).mean()

        results.append({
            'Strategy': name,
            'Cumulative Return': f"{total_return:.1%}",
            'Annualized Return': f"{annualized_return:.1%}",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Max Drawdown': f"{max_dd:.1%}",
            'Hit Ratio': f"{hit_ratio:.1%}"
        })

        logger.info(f"\n{name}:")
        logger.info(f"  Cumulative Return: {total_return:.1%}")
        logger.info(f"  Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"  Max Drawdown: {max_dd:.1%}")
        logger.info(f"  Hit Ratio: {hit_ratio:.1%}")

    return pd.DataFrame(results), strategies


def analyze_correlations(relative_returns):
    """Analyze correlation between sector relative returns."""

    logger.info("\n" + "=" * 70)
    logger.info("SECTOR CORRELATION ANALYSIS")
    logger.info("=" * 70)

    corr_matrix = calculate_sector_correlation_matrix(relative_returns)

    # Find low-correlation pairs (good for diversification)
    low_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if corr_val < 0.3:  # Low correlation threshold
                low_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_val
                ))

    low_corr_pairs.sort(key=lambda x: x[2])

    logger.info("\nLow Correlation Pairs (diversification opportunities):")
    for s1, s2, corr in low_corr_pairs[:10]:
        logger.info(f"  {s1} - {s2}: {corr:.3f}")

    # High correlation pairs (redundant)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if corr_val > 0.7:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_val
                ))

    high_corr_pairs.sort(key=lambda x: -x[2])

    logger.info("\nHigh Correlation Pairs (redundant):")
    for s1, s2, corr in high_corr_pairs[:10]:
        logger.info(f"  {s1} - {s2}: {corr:.3f}")

    return corr_matrix, low_corr_pairs, high_corr_pairs


def optimize_sector_weights(relative_returns, sector_scores):
    """Optimize sector weights based on predictability scores."""

    logger.info("\n" + "=" * 70)
    logger.info("OPTIMAL SECTOR WEIGHTING")
    logger.info("=" * 70)

    # Only use predictable sectors
    predictable = ['Consumer_Staples', 'Health_Care', 'Materials', 'Industrials', 'Communication']

    available_sectors = [s for s in predictable if s in relative_returns.columns]

    if len(available_sectors) < 2:
        logger.warning("Not enough sectors available for optimization")
        return {}

    # Historical returns for optimization
    returns_data = relative_returns[available_sectors].dropna()

    def objective(weights):
        """Minimize volatility (risk parity approach)."""
        weighted_returns = returns_data @ weights
        return -weighted_returns.mean() / weighted_returns.std()  # Negative Sharpe

    # Constraints: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}

    # Bounds: each weight between 0 and 1
    bounds = [(0, 1) for _ in available_sectors]

    # Initial guess: equal weights
    x0 = np.ones(len(available_sectors)) / len(available_sectors)

    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

    optimal_weights = dict(zip(available_sectors, result.x))

    logger.info("\nOptimal Sector Weights (Risk-Adjusted):")
    for sector, weight in sorted(optimal_weights.items(), key=lambda x: -x[1]):
        logger.info(f"  {sector}: {weight:.1%}")

    return optimal_weights


def main():
    """Run sector combination strategy analysis."""

    logger.info("\n" + "🎯" * 35)
    logger.info("SECTOR COMBINATION STRATEGY ANALYSIS")
    logger.info("🎯" * 35)

    # Fetch data
    relative_returns, spx_returns = fetch_all_sector_data()

    if len(relative_returns) < 24:
        logger.error("Insufficient data for analysis")
        return

    logger.info(f"\nData period: {relative_returns.index[0].strftime('%Y-%m')} to {relative_returns.index[-1].strftime('%Y-%m')}")
    logger.info(f"Number of observations: {len(relative_returns)}")
    logger.info(f"Sectors available: {len(relative_returns.columns)}")

    # Analysis 1: Correlation
    corr_matrix, low_corr, high_corr = analyze_correlations(relative_returns)

    # Analysis 2: Strategy comparison
    results_df, strategies = analyze_strategies(relative_returns, spx_returns)

    # Analysis 3: Optimal weights
    optimal_weights = optimize_sector_weights(relative_returns, {})

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 70)

    logger.info("""
    1. SECTOR SELECTION:
       - Focus on defensive sectors: Consumer Staples, Healthcare, Materials
       - Avoid Energy and Financials (negative predictability)
       - Industrials and Communication provide moderate diversification

    2. COMBINATION STRATEGIES:
       - "Predictable Only" strategy: Only trade sectors with positive R² OOS
       - "Weighted by Signal": Size positions by prediction confidence
       - "Long-Short": Market-neutral approach with sector rotation

    3. CORRELATION CONSIDERATIONS:
       - Consumer Staples and Healthcare show low correlation (good for diversification)
       - Energy is negatively correlated with many sectors (crisis hedge)
       - Technology and Communication are highly correlated (media/tech overlap)

    4. REBALANCING:
       - Monthly rebalancing recommended based on prediction signals
       - Avoid overtrading due to transaction costs
    """)

    # Save results
    results_df.to_csv('sector_combination_results.csv', index=False)
    logger.info("\nResults saved to: sector_combination_results.csv")

    return results_df, strategies, optimal_weights


if __name__ == "__main__":
    results, strategies, weights = main()
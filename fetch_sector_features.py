"""
Sector Features for SSRF Model
Combines:
1. Sector ETFs (11 S&P sectors)
2. Relative Momentum (sector vs SPX)
3. Technical Indicators (RSI, Breadth)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.config import SECTOR_ETFS_BY_TICKER

# S&P 500 Sector ETFs — shared with other scripts via src.config
SECTOR_ETFS = SECTOR_ETFS_BY_TICKER

def fetch_sector_data(start_date='1990-01-01', end_date=None):
    """Fetch sector ETF data from yfinance."""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    print(f"Fetching sector data from {start_date} to {end_date}...")

    sector_data = {}
    for ticker, name in SECTOR_ETFS.items():
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if len(data) > 0:
                # Handle multi-level columns from yfinance
                if isinstance(data.columns, pd.MultiIndex):
                    adj_close = data['Adj Close'].iloc[:, 0] if 'Adj Close' in data.columns.get_level_values(0) else data.iloc[:, 0]
                else:
                    adj_close = data['Adj Close']
                sector_data[ticker] = adj_close
                print(f"  {ticker} ({name}): {len(data)} rows")
            else:
                print(f"  X {ticker} ({name}): No data")
        except Exception as e:
            print(f"  X {ticker} ({name}): Error - {e}")

    df = pd.DataFrame(sector_data)
    df.index = pd.to_datetime(df.index)
    return df

def compute_relative_momentum(sector_prices, spx_prices):
    """Compute relative momentum (sector vs SPX)."""
    print("\nComputing relative momentum...")

    # Calculate 1M, 3M, 6M, 12M returns for sectors
    rel_momentum = pd.DataFrame(index=sector_prices.index)

    for ticker in sector_prices.columns:
        sector_returns = sector_prices[ticker].pct_change()
        spx_returns = spx_prices.pct_change()

        # Relative returns (sector - SPX)
        rel_returns = sector_returns - spx_returns

        # Rolling relative momentum
        rel_momentum[f'{ticker}_REL_1M'] = rel_returns.rolling(21).mean() * 21
        rel_momentum[f'{ticker}_REL_3M'] = rel_returns.rolling(63).mean() * 63
        rel_momentum[f'{ticker}_REL_6M'] = rel_returns.rolling(126).mean() * 126

        # Relative strength vs SPX
        sector_total = (sector_prices[ticker] / sector_prices[ticker].iloc[0]) - 1
        spx_total = (spx_prices / spx_prices.iloc[0]) - 1
        rel_momentum[f'{ticker}_REL_12M'] = sector_total - spx_total

    return rel_momentum

def compute_technical_indicators(sector_prices):
    """Compute RSI and breadth indicators for sectors."""
    print("Computing technical indicators...")

    rsi_data = pd.DataFrame(index=sector_prices.index)
    breadth_data = pd.DataFrame(index=sector_prices.index)

    for ticker in sector_prices.columns:
        prices = sector_prices[ticker]

        # RSI (14-day)
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi_data[f'{ticker}_RSI'] = rsi

        # Simple Moving Averages
        sma_20 = prices.rolling(20).mean()
        sma_50 = prices.rolling(50).mean()

        # % above/below SMA
        breadth_data[f'{ticker}_ABOVE_SMA20'] = (prices > sma_20).astype(float) * 100
        breadth_data[f'{ticker}_ABOVE_SMA50'] = (prices > sma_50).astype(float) * 100

        # Rate of change
        breadth_data[f'{ticker}_ROC_20'] = prices.pct_change(20) * 100

    return rsi_data, breadth_data

def compute_sector_inflow_indicators(sector_prices):
    """Compute sector inflow/flow of money indicators."""
    print("Computing sector inflow indicators...")

    inflow = pd.DataFrame(index=sector_prices.index)

    # Calculate money flow proxies using price and volume relationships
    # Since we don't have volume data for ETFs, we'll use relative strength
    # as a proxy for money flow

    # 1. Acceleration: Is the sector gaining relative momentum?
    returns_1M = sector_prices.pct_change(21)
    returns_3M = sector_prices.pct_change(63)

    # 2. Momentum acceleration (increasing momentum = inflow)
    for ticker in sector_prices.columns:
        inflow[f'{ticker}_ACCEL'] = returns_1M[ticker] - returns_3M[ticker]

    # 3. Sector strength ranking (relative to all sectors)
    # Z-score of 3M momentum across all sectors
    momentums = returns_3M.rolling(21).mean()
    for ticker in sector_prices.columns:
        mean_mom = momentums.mean(axis=1)
        std_mom = momentums.std(axis=1)
        inflow[f'{ticker}_ZSCORE'] = (momentums[ticker] - mean_mom) / (std_mom + 1e-6)

    # 4. Sector momentum rank (1-11)
    for date in returns_3M.dropna().index:
        vals = returns_3M.loc[date].values
        ranks = pd.Series(vals).rank(ascending=False)
        for i, ticker in enumerate(returns_3M.columns):
            inflow.loc[date, f'{ticker}_RANK'] = ranks.iloc[i]

    # 5. Bull market leadership score
    # Sectors leading during bull markets typically have:
    # - High relative strength
    # - Low volatility
    # - Strong momentum
    for ticker in sector_prices.columns:
        rel_mom = returns_3M[ticker] - returns_3M.mean(axis=1)
        vol = sector_prices[ticker].pct_change().rolling(21).std()
        inflow[f'{ticker}_BULL_SCORE'] = rel_mom / (vol + 1e-6)

    # 6. Sector correlation with SPX (stable sectors get more weight)
    spx_returns = sector_prices.pct_change()
    for ticker in sector_prices.columns:
        corr = spx_returns[ticker].rolling(63).corr(spx_returns.mean(axis=1))
        inflow[f'{ticker}_SPX_CORR'] = corr

    # 7. Sector dispersion (high dispersion = rotation opportunity)
    inflow['SECTOR_DISPERSION'] = returns_3M.std(axis=1)

    return inflow

def compute_sector_rotation_signals(sector_prices):
    """Compute sector rotation signals (leadership changes)."""
    print("Computing rotation signals...")

    rotation = pd.DataFrame(index=sector_prices.index)

    # Calculate momentum rankings
    returns_3M = sector_prices.pct_change(63)
    returns_6M = sector_prices.pct_change(126)

    # Leadership score (current momentum / historical volatility)
    for ticker in sector_prices.columns:
        mom = returns_3M[ticker]
        vol = sector_prices[ticker].pct_change().rolling(63).std() * np.sqrt(252)
        rotation[f'{ticker}_LEAD'] = mom / (vol + 1e-6)

        # Trend change (6M momentum - 3M momentum)
        rotation[f'{ticker}_TREND'] = returns_6M[ticker] - returns_3M[ticker]

    # Market-wide sector rotation
    rotation['SECTOR_SPREAD'] = returns_3M.max(axis=1) - returns_3M.min(axis=1)

    return rotation

def get_spx_data(start_date, end_date):
    """Fetch SPX data for relative calculations."""
    spx = yf.download('^SPX', start=start_date, end=end_date, progress=False)
    # Handle multi-level columns from yfinance
    if isinstance(spx.columns, pd.MultiIndex):
        return spx['Adj Close'].iloc[:, 0] if 'Adj Close' in spx.columns.get_level_values(0) else spx.iloc[:, 0]
    return spx['Adj Close']

def main():
    print("=" * 60)
    print("SECTOR FEATURES FOR SSRF MODEL")
    print("=" * 60)

    # Date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = '1990-01-01'

    # Fetch SPX data first
    print("\n[1] Fetching SPX benchmark...")
    spx = get_spx_data(start_date, end_date)
    print(f"  SPX data: {len(spx)} rows, {spx.index[0].strftime('%Y-%m')} to {spx.index[-1].strftime('%Y-%m')}")

    # Fetch sector data
    print("\n[2] Fetching sector ETF data...")
    sector_prices = fetch_sector_data(start_date, end_date)
    print(f"\n  Total sector data: {len(sector_prices)} rows")

    # Align SPX to sector dates
    common_idx = sector_prices.index
    spx_aligned = spx.reindex(common_idx).ffill()

    # Compute features
    print("\n[3] Computing relative momentum...")
    rel_momentum = compute_relative_momentum(sector_prices, spx_aligned)

    print("\n[4] Computing technical indicators...")
    rsi_data, breadth_data = compute_technical_indicators(sector_prices)

    print("\n[5] Computing rotation signals...")
    rotation = compute_sector_rotation_signals(sector_prices)

    print("\n[6] Computing sector inflow indicators...")
    inflow = compute_sector_inflow_indicators(sector_prices)

    # Combine all sector features
    print("\n[7] Combining features...")
    sector_features = pd.concat([rel_momentum, rsi_data, breadth_data, rotation, inflow], axis=1)

    # Save to cache
    output_path = 'data/sector_cache/sector_features.csv'
    sector_features.to_csv(output_path)

    # Summary
    print("\n" + "=" * 60)
    print("SECTOR FEATURES SUMMARY")
    print("=" * 60)
    print(f"Date Range: {sector_features.index[0].strftime('%Y-%m')} to {sector_features.index[-1].strftime('%Y-%m')}")
    print(f"Total Features: {len(sector_features.columns)}")
    print(f"\nBreakdown:")
    print(f"  - Relative Momentum: {len(rel_momentum.columns)} features")
    print(f"  - RSI Indicators: {len(rsi_data.columns)} features")
    print(f"  - Breadth Indicators: {len(breadth_data.columns)} features")
    print(f"  - Rotation Signals: {len(rotation.columns)} features")
    print(f"  - Inflow Indicators: {len(inflow.columns)} features")
    print(f"\nSectors: {', '.join(SECTOR_ETFS.keys())}")
    print(f"\nSaved to: {output_path}")

    # Feature details
    print("\nFeature Names:")
    print(sector_features.columns.tolist()[:20], "..." if len(sector_features.columns) > 20 else "")

    return sector_features

if __name__ == "__main__":
    features = main()
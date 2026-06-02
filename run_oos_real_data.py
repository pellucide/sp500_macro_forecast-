#!/usr/bin/env python3.9
"""
Out-of-Sample Test with Real Market Data
Uses cached FRED data + SP500 returns from Yahoo Finance
"""

import os
import time
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


# =============================================================================
# Alternative/Exuberance Features
# =============================================================================

ALT_CACHE_FILE = 'data/fred_cache/alternative_features.csv'
ALT_CACHE_MAX_DAYS = 30  # Re-fetch after 30 days


def fetch_shiller_cape():
    """Download Shiller CAPE (Cyclically Adjusted P/E) from Yale.

    Source: http://www.econ.yale.edu/~shiller/data/ie_data.xls
    Returns monthly DataFrame with 'CAPE' column.
    """
    url = 'http://www.econ.yale.edu/~shiller/data/ie_data.xls'
    logger.info("Downloading Shiller CAPE data...")
    try:
        df = pd.read_excel(url, sheet_name='Data', skiprows=7)
        # Column 0 = Date (YYYY.MM decimal), Column 12 = CAPE
        cape = df.iloc[:, [0, 12]].copy()
        cape.columns = ['date_frac', 'CAPE']
        cape = cape.dropna(subset=['CAPE'])

        # Convert YYYY.MM decimal to datetime
        def date_frac_to_ts(x):
            year = int(x)
            # Month from decimal: 1871.01 = Jan, 1871.02 = Feb, etc.
            # Need round() to handle floating point imprecision
            month = int(round((x - year) * 100))
            if month < 1:
                month = 1
            if month > 12:
                month = 12
            return pd.Timestamp(year=year, month=month, day=1)

        cape['date'] = cape['date_frac'].apply(date_frac_to_ts)
        cape = cape.set_index('date')[['CAPE']].sort_index()
        cape = cape[~cape.index.duplicated(keep='last')]
        # Convert to month-end to align with FRED data (which uses month-end)
        cape.index = cape.index + pd.offsets.MonthEnd(0)
        # Remove any future data beyond what we need (before 2026-06)
        cape = cape[cape.index <= '2026-06-30']
        logger.info(f"  CAPE: {len(cape)} months, {cape.index[0].strftime('%Y-%m')} to {cape.index[-1].strftime('%Y-%m')}")
        return cape
    except Exception as e:
        logger.error(f"Failed to fetch Shiller CAPE: {e}")
        return None


def fetch_put_call_ratio():
    """Download CBOE put/call ratio and aggregate to monthly.

    Combines archive (1995-2003) with daily data (2006-2019+).
    Returns monthly DataFrame with 'PUT_CALL_RATIO' column.
    """
    logger.info("Downloading CBOE put/call ratio...")
    try:
        # Archive: 1995-2003, has Total, Index, Equity columns
        archive_url = 'https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/pcratioarchive.csv'
        archive = pd.read_csv(archive_url, skiprows=2, encoding='latin1')
        arch = archive.iloc[:, [0, 1]].copy()
        arch.columns = ['Date', 'PUT_CALL_RATIO']
        arch['Date'] = pd.to_datetime(arch['Date'])
        arch = arch.dropna()
        logger.info(f"  Archive: {len(arch)} obs ({arch['Date'].min():%Y-%m} to {arch['Date'].max():%Y-%m})")

        # Daily: 2006-2019, has Date, Calls, Puts, Total, Ratio
        daily_url = 'https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/totalpc.csv'
        daily = pd.read_csv(
            daily_url, skiprows=3, header=None,
            names=['Date', 'Calls', 'Puts', 'Total', 'PUT_CALL_RATIO']
        )
        daily['Date'] = pd.to_datetime(daily['Date'], errors='coerce')
        daily = daily.dropna(subset=['Date', 'PUT_CALL_RATIO'])
        logger.info(f"  Daily: {len(daily)} obs ({daily['Date'].min():%Y-%m} to {daily['Date'].max():%Y-%m})")

        # Combine both sources
        combined = pd.concat([
            arch[['Date', 'PUT_CALL_RATIO']],
            daily[['Date', 'PUT_CALL_RATIO']]
        ]).dropna().sort_values('Date').drop_duplicates('Date')

        # Aggregate daily to monthly (mean)
        combined = combined.set_index('Date')
        monthly = combined.resample('ME').mean()
        monthly.index.freq = 'ME'
        logger.info(f"  Combined monthly: {len(monthly)} obs ({monthly.index[0].strftime('%Y-%m')} to {monthly.index[-1].strftime('%Y-%m')})")
        return monthly
    except Exception as e:
        logger.error(f"Failed to fetch put/call ratio: {e}")
        return None


def fetch_margin_debt():
    """Fetch NYSE margin debt from FRED Flow of Funds (quarterly → monthly).

    Uses FRED series BOGZ1FL663067003Q (Security Brokers and Dealers;
    Margin Loans) and forward-fills to monthly frequency.
    Returns monthly DataFrame with 'MARGIN_DEBT' column.
    """
    logger.info("Fetching margin debt from FRED Flow of Funds...")
    try:
        from fredapi import Fred
        fred = Fred('48f0923658be7d90ba311c4a55138377')
        # Quarterly margin loans (brokers/dealers receivables from customers)
        qtr = fred.get_series('BOGZ1FL663067003Q')
        qtr = qtr.dropna().sort_index()

        # Convert to DataFrame
        md = pd.DataFrame({'MARGIN_DEBT': qtr})
        md.index = pd.to_datetime(md.index)

        # Upsample to monthly (forward fill within quarter)
        monthly_idx = pd.date_range(start=md.index[0], end=md.index[-1], freq='ME')
        md_monthly = md.reindex(monthly_idx, method='ffill')
        md_monthly.index.freq = 'ME'

        logger.info(f"  Margin debt: {len(md_monthly)} months ({md_monthly.index[0].strftime('%Y-%m')} to {md_monthly.index[-1].strftime('%Y-%m')})")
        return md_monthly
    except Exception as e:
        logger.error(f"Failed to fetch margin debt: {e}")
        return None


def fetch_aaii_sentiment():
    """Fetch AAII Bull/Bear sentiment spread.

    Attempts to download from AAII website or alternative sources.
    Returns monthly DataFrame with 'AAII_BULL_BEAR_SPREAD' column.
    Falls back gracefully if source is unavailable.
    """
    logger.info("Attempting to fetch AAII sentiment data...")
    try:
        # AAII publishes weekly bull/bear survey results
        # Try a known CSV endpoint from a data aggregator
        # Fallback: compute from put/call and VIX (last resort)
        import requests
        from io import StringIO

        # Try multiple potential sources
        sources = [
            # Nasdaq Data Link CSV URL (if available)
            None,
        ]

        # Direct web scrape from AAII public page
        url = 'https://www.aaii.com/sentiment/survey-results'
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=10)

        if r.status_code == 200:
            # Try to parse the page for recent survey data
            import re
            # AAII typically publishes bull/bear/neutral percentages
            # Try to extract from HTML tables
            tables = pd.read_html(StringIO(r.text))
            for t in tables:
                if 'Bullish' in str(t.columns) or 'Bullish' in str(t.values):
                    logger.info(f"  AAII data found in HTML table")
                    return t

        logger.warning("AAII sentiment source not available, skipping")
        return None

    except Exception as e:
        logger.warning(f"AAII sentiment fetch failed: {e}")
        return None


def load_alternative_features():
    """Load all alternative/exuberance features from cache or fetch fresh.

    Returns a DataFrame with monthly index and columns:
        CAPE, PUT_CALL_RATIO, MARGIN_DEBT, AAII_BULL_BEAR_SPREAD
    NaN values indicate the feature is not available for that period.
    """
    # Check cache
    if os.path.exists(ALT_CACHE_FILE):
        file_age = (time.time() - os.path.getmtime(ALT_CACHE_FILE)) / (24 * 3600)
        if file_age < ALT_CACHE_MAX_DAYS:
            logger.info(f"Loading cached alternative features ({file_age:.1f} days old)")
            df = pd.read_csv(ALT_CACHE_FILE, index_col=0, parse_dates=True)
            logger.info(f"  Loaded {df.shape[1]} features, {df.shape[0]} months")
            return df
        else:
            logger.info(f"Cache stale ({file_age:.1f} days), refreshing...")

    # Fetch all features
    logger.info("Fetching alternative features...")

    features = []
    fetchers = [
        ('CAPE', fetch_shiller_cape),
        ('PUT_CALL_RATIO', fetch_put_call_ratio),
        ('MARGIN_DEBT', fetch_margin_debt),
        ('AAII_BULL_BEAR_SPREAD', fetch_aaii_sentiment),
    ]

    for name, fetcher in fetchers:
        try:
            df = fetcher()
            if df is not None and len(df) > 0:
                features.append(df)
                logger.info(f"  ✓ {name}: {len(df)} obs")
            else:
                logger.warning(f"  ✗ {name}: no data")
        except Exception as e:
            logger.warning(f"  ✗ {name}: {e}")

    if not features:
        logger.warning("No alternative features could be fetched")
        return pd.DataFrame()

    # Merge all on index (outer join to preserve all dates)
    merged = features[0]
    for df in features[1:]:
        merged = merged.join(df, how='outer')

    merged = merged.sort_index()
    logger.info(f"Merged alternative features: {merged.shape[1]} cols, {merged.shape[0]} rows")

    # Cache
    os.makedirs(os.path.dirname(ALT_CACHE_FILE), exist_ok=True)
    merged.to_csv(ALT_CACHE_FILE)
    logger.info(f"Cached to {ALT_CACHE_FILE}")

    return merged


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


def impute_vix_proxy(indicators, returns):
    """
    Impute missing VIX values before 1990 using realized volatility proxy.
    
    VIX (implied volatility) is only available from 1990-01. For earlier periods,
    we compute a point-in-time proxy from rolling 12-month realized volatility
    of S&P 500 returns. This is a standard approach in academic finance.
    
    Proxy = rolling_12m_std(returns) * sqrt(12)
    
    Args:
        indicators: DataFrame with macro indicators (must contain 'VIXCLS')
        returns: Series of S&P 500 monthly returns
    """
    if 'VIXCLS' not in indicators.columns:
        logger.warning("VIXCLS not found in indicators, skipping proxy imputation")
        return indicators
    
    vix = indicators['VIXCLS']
    n_missing = vix.isna().sum()
    
    if n_missing == 0:
        logger.info("VIX has no missing values, no proxy needed")
        return indicators
    
    # Compute realized volatility proxy: 12-month rolling std, annualized
    # Align to indicators index (returns may have different start date)
    aligned_returns = returns.reindex(indicators.index)
    rolling_vol = aligned_returns.rolling(window=12, min_periods=6).std()
    vix_proxy = rolling_vol * np.sqrt(12) * 100  # Convert to percentage (VIX units)
    
    # Fill only NaN VIX values with proxy (don't overwrite actual VIX)
    indicators = indicators.copy()
    mask = indicators['VIXCLS'].isna()
    indicators.loc[mask, 'VIXCLS'] = vix_proxy[mask]
    
    n_filled = mask.sum()
    logger.info(f"VIX proxy imputation: filled {n_filled} missing values "
                f"({indicators.index[mask][0].strftime('%Y-%m')} to "
                f"{indicators.index[mask][-1].strftime('%Y-%m')})")
    
    # Also update VIX regime flags for consistency with proxy
    # Use proxy-derived thresholds (no look-ahead: use expanding median up to each point)
    if 'VIX_REGIME_HIGH' in indicators.columns or 'VIX_REGIME_LOW' in indicators.columns:
        vix_filled = indicators['VIXCLS']
        # Expanding 75th percentile for high regime (point-in-time)
        high_threshold = vix_filled.expanding(min_periods=24).quantile(0.75)
        low_threshold = vix_filled.expanding(min_periods=24).quantile(0.25)
        
        if 'VIX_REGIME_HIGH' in indicators.columns:
            indicators['VIX_REGIME_HIGH'] = (vix_filled > high_threshold).astype(float)
        if 'VIX_REGIME_LOW' in indicators.columns:
            indicators['VIX_REGIME_LOW'] = (vix_filled < low_threshold).astype(float)
        
        logger.info("Updated VIX_REGIME_HIGH/LOW flags to match proxy-imputed VIX")
    
    return indicators


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
        'exuberance': [c for c in ['CAPE', 'PUT_CALL_RATIO', 'MARGIN_DEBT',
                                    'AAII_BULL_BEAR_SPREAD'] if c in cols],
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
    
    # FIX: Impute missing pre-1990 VIX with realized volatility proxy
    indicators = impute_vix_proxy(indicators, target)
    
    # Create groups
    groups = create_groups_from_data(indicators)
    
    # Align with 1-month-ahead lag
    X, y = align_data(indicators, target)
    
    # Configure SSRF model
    config = SSRFConfig(
        t_stat_threshold=0.75,
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

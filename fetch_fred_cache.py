"""
FRED Data Cache - Downloads and caches all indicators once
Run this script to populate the cache, then all subsequent tests will use cached data.
"""

import sys
import os
import time
import logging

# Dynamic path resolution - works from any directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

CACHE_DIR = './data/fred_cache'
CACHE_FILE = './data/fred_cache/all_fred_data.csv'
API_KEY = "48f0923658be7d90ba311c4a55138377"

# Complete indicator list
INDICATORS = {
    # Output & Income
    'GDPC1': 'Real GDP',
    'PCECC96': 'Real Consumption',
    'FYFSD': 'Federal Surplus/Deficit',

    # Labor Market
    'UNRATE': 'Unemployment Rate',
    'PAYEMS': 'Payrolls',
    'EMRATIO': 'Employment Ratio',
    'HOUST': 'Housing Starts',
    'PERMIT': 'Building Permits',

    # Inflation
    'CPIAUCSL': 'CPI',
    'CPILFESL': 'Core CPI',
    'PCECTPI': 'PCE Price Index',
    'PCEPILFE': 'Core PCE',
    'GDPDEF': 'GDP Deflator',
    'PPIFGS': 'PPI',

    # Interest Rates
    'TB3MS': '3M Treasury',
    'TB6MS': '6M Treasury',
    'GS1': '1Y Treasury',
    'GS2': '2Y Treasury',
    'GS5': '5Y Treasury',
    'GS10': '10Y Treasury',
    'GS20': '20Y Treasury',
    'GS30': '30Y Treasury',
    'AAA': 'AAA Corporate',
    'BAA': 'BAA Corporate',

    # Yield Curve
    'T10Y2YM': '10Y-2Y Spread',
    'TEDRATE': 'TED Spread',

    # Risk & Volatility
    'VIXCLS': 'VIX',
    'MOVE': 'MOVE Index',

    # Sentiment
    'UMCSENT': 'Consumer Sentiment',
    'IC4WSA': 'Capacity Utilization',

    # Money Supply - M1, M2, M3
    'M1SL': 'M1 Money Supply',
    'M2SL': 'M2 Money Supply',
    'M3SL': 'M3 Money Supply',

    # Credit
    'BAAFFM': 'BAA-10Y Spread',
    'AAAFFM': 'AAA-10Y Spread',
}

# ============================================================================
# CACHE FUNCTIONS
# ============================================================================

def ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    logger.info(f"Cache directory: {CACHE_DIR}")

def fetch_fred_with_cache(api_key: str) -> pd.DataFrame:
    """
    Fetch all FRED indicators with rate limiting.
    Downloads once, caches to CSV.
    """
    cache_file = CACHE_FILE

    # Check if cache exists and is fresh (less than 7 days old)
    if os.path.exists(cache_file):
        file_age_days = (time.time() - os.path.getmtime(cache_file)) / (24 * 3600)
        if file_age_days < 7:
            logger.info(f"Using cached data ({file_age_days:.1f} days old)")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    logger.info(f"Fetching {len(INDICATORS)} FRED indicators...")
    logger.info("This will take ~3-5 minutes due to rate limiting")
    logger.info("Subsequent runs will use cached data.")

    try:
        import fredapi
        fred = fredapi.Fred(api_key)
    except Exception as e:
        logger.error(f"Failed to connect to FRED: {e}")
        # Try to load from cache anyway
        if os.path.exists(cache_file):
            logger.info("Falling back to existing cache")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        raise

    all_data = {}
    failed = []
    delay = 1.5  # 1.5 second delay between requests

    for i, (series_id, name) in enumerate(INDICATORS.items()):
        try:
            logger.info(f"[{i+1}/{len(INDICATORS)}] Fetching {series_id} ({name})...")

            series = fred.get_series(series_id, observation_start='1980-01-01')

            if len(series) > 100:
                df = pd.DataFrame(series)
                df.index = pd.to_datetime(df.index)
                monthly = df.resample('ME').last().iloc[:, 0]
                monthly.name = series_id
                all_data[series_id] = monthly
                logger.info(f"  OK: {len(monthly)} observations")
            else:
                logger.warning(f"  SKIP: Insufficient data ({len(series)} obs)")
                failed.append(series_id)

            # Rate limiting
            time.sleep(delay)

        except Exception as e:
            logger.error(f"  FAIL: {e}")
            failed.append(series_id)
            time.sleep(0.5)

    # Combine into DataFrame
    if all_data:
        fred_df = pd.DataFrame(all_data)
        fred_df = fred_df.ffill().dropna(how='all')

        # Save to cache
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        fred_df.to_csv(cache_file)
        logger.info(f"\nCached {len(fred_df.columns)} indicators to {cache_file}")
    else:
        logger.error("No data fetched!")
        fred_df = None

    # Report
    logger.info(f"\n{'='*60}")
    logger.info(f"FETCH COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"  Successful: {len(all_data)} indicators")
    logger.info(f"  Failed: {len(failed)} indicators")

    if failed:
        logger.info(f"  Failed indicators: {', '.join(failed)}")

    return fred_df


def load_cached_fred_data() -> pd.DataFrame:
    """Load FRED data from cache."""
    cache_file = CACHE_FILE

    if os.path.exists(cache_file):
        logger.info(f"Loading cached FRED data from {cache_file}")
        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
        logger.info(f"  Loaded {len(df.columns)} indicators, {len(df)} observations")
        return df
    else:
        logger.warning("No cache found. Run with --refresh to download data.")
        return None


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features from raw FRED data.
    """
    derived = df.copy()

    # Yield Curve Features
    if 'GS10' in df.columns and 'TB3MS' in df.columns:
        derived['YIELD_SLOPE_10Y3M'] = df['GS10'] - df['TB3MS']

    if 'GS10' in df.columns and 'GS2' in df.columns:
        derived['YIELD_SLOPE_10Y2Y'] = df['GS10'] - df['GS2']

    if 'GS2' in df.columns and 'TB3MS' in df.columns:
        derived['YIELD_SLOPE_2Y3M'] = df['GS2'] - df['TB3MS']

    # Credit Spreads
    if 'BAA' in df.columns and 'GS10' in df.columns:
        derived['CREDIT_SPREAD_BAA'] = df['BAA'] - df['GS10']

    if 'BAA' in df.columns and 'AAA' in df.columns:
        derived['CREDIT_SPREAD_QUALITY'] = df['BAA'] - df['AAA']

    # Real Yields
    if 'GS10' in df.columns and 'CPIAUCSL' in df.columns:
        derived['REAL_10Y'] = df['GS10'] - df['CPIAUCSL'].pct_change(12) * 100

    # Sentiment composite
    if 'UMCSENT' in df.columns:
        derived['SENTIMENT_REGIME'] = (df['UMCSENT'] > 85).astype(int)  # Above 85 = optimistic

    # VIX regime
    if 'VIXCLS' in df.columns:
        derived['VIX_REGIME_HIGH'] = (df['VIXCLS'] > 25).astype(int)
        derived['VIX_REGIME_LOW'] = (df['VIXCLS'] < 15).astype(int)

    # Momentum of M1, M2, M3
    if 'M1SL' in df.columns:
        derived['M1_GROWTH_12M'] = df['M1SL'].pct_change(12) * 100
        derived['M1_GROWTH_6M'] = df['M1SL'].pct_change(6) * 100
        derived['M1_GROWTH_3M'] = df['M1SL'].pct_change(3) * 100
        # M1 Velocity proxy
        derived['M1_ACCEL'] = derived['M1_GROWTH_3M'] - derived['M1_GROWTH_12M']

    if 'M2SL' in df.columns:
        derived['M2_GROWTH_12M'] = df['M2SL'].pct_change(12) * 100
        derived['M2_GROWTH_6M'] = df['M2SL'].pct_change(6) * 100
        derived['M2_GROWTH_3M'] = df['M2SL'].pct_change(3) * 100
        # M2 Velocity proxy
        derived['M2_ACCEL'] = derived['M2_GROWTH_3M'] - derived['M2_GROWTH_12M']

    if 'M3SL' in df.columns:
        derived['M3_GROWTH_12M'] = df['M3SL'].pct_change(12) * 100
        derived['M3_GROWTH_6M'] = df['M3SL'].pct_change(6) * 100
        derived['M3_GROWTH_3M'] = df['M3SL'].pct_change(3) * 100
        # M3 Velocity proxy
        derived['M3_ACCEL'] = derived['M3_GROWTH_3M'] - derived['M3_GROWTH_12M']

    # Money Supply Ratios
    if 'M1SL' in df.columns and 'M2SL' in df.columns:
        derived['M1_M2_RATIO'] = df['M1SL'] / df['M2SL']

    if 'M2SL' in df.columns and 'M3SL' in df.columns:
        derived['M2_M3_RATIO'] = df['M2SL'] / df['M3SL']  # M2 as % of M3

    # Money Supply Divergence (M1 vs M3 growth difference)
    if 'M1_GROWTH_12M' in derived.columns and 'M3_GROWTH_12M' in derived.columns:
        derived['M1_VS_M3_GROWTH'] = derived['M1_GROWTH_12M'] - derived['M3_GROWTH_12M']

    # Unemployment trend
    if 'UNRATE' in df.columns:
        derived['UNRATE_CHANGE_12M'] = df['UNRATE'].diff(12)

    logger.info(f"Computed {len(derived.columns) - len(df.columns)} derived features")
    return derived


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='FRED Data Cache Manager')
    parser.add_argument('--refresh', action='store_true', help='Force refresh from FRED')
    parser.add_argument('--check', action='store_true', help='Check cache status')
    args = parser.parse_args()

    if args.check:
        if os.path.exists(CACHE_FILE):
            df = pd.read_csv(CACHE_FILE, index_col=0)
            print(f"\nCache Status:")
            print(f"  File: {CACHE_FILE}")
            print(f"  Indicators: {len(df.columns)}")
            print(f"  Date range: {df.index[0]} to {df.index[-1]}")
            print(f"  Columns: {', '.join(df.columns[:10])}...")
        else:
            print("No cache found. Run without --check to download data.")
        sys.exit(0)

    # Fetch and cache
    fred_df = fetch_fred_with_cache(API_KEY)

    if fred_df is not None:
        # Compute derived features
        fred_enhanced = compute_derived_features(fred_df)

        # Save enhanced version
        enhanced_cache = CACHE_FILE.replace('.csv', '_enhanced.csv')
        fred_enhanced.to_csv(enhanced_cache)

        print(f"\n{'='*60}")
        print(f"CACHE POPULATED")
        print(f"{'='*60}")
        print(f"  Raw data: {CACHE_FILE}")
        print(f"  Enhanced: {enhanced_cache}")
        print(f"  Total features: {len(fred_enhanced.columns)}")
        print(f"\n  Available features:")
        for col in fred_enhanced.columns:
            print(f"    - {col}")

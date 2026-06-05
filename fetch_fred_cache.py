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

# ============================================================================
# Full FRED-MD indicator list (McCracken & Ng, 134-series database)
# Sourced from: https://fred.stlouisfed.org/docs/fred-md/fred-md-current.csv
# For series ending in 'x', the 'x' suffix denotes a FRED-MD transformation;
# the underlying FRED series is fetched without the suffix.
#
# Categories match the FRED-MD taxonomy:
#   1) Output & Income
#   2) Labor Market
#   3) Housing
#   4) Consumption, Orders & Inventories
#   5) Money & Credit
#   6) Interest & Exchange Rates
#   7) Prices
#   8) Stock Market & Sentiment
# ============================================================================

# Map: FRED-MD column name -> FRED API series ID
# Most map 1:1; some need the 'x' suffix stripped or a different ticker.
FRED_MD_SERIES = {
    # ---- OUTPUT & INCOME (18 series) ----
    'RPI': 'RPI',                                      # Real Personal Income
    'W875RX1': 'W875RX1',                              # Real Personal Income ex Transfer Receipts
    'DPCERA3M086SBEA': 'DPCERA3M086SBEA',              # Real Personal Consumption Expenditures
    'CMRMTSPLx': 'CMRMTSPL',                           # Real Mfg & Trade Industries Sales
    'RETAILx': 'RETAIL',                                # Retail & Food Services Sales
    'INDPRO': 'INDPRO',                                 # Industrial Production Index
    'IPFPNSS': 'IPFPNSS',                               # IP: Final Products & Nonindustrial Supplies
    'IPFINAL': 'IPFINAL',                               # IP: Final Products (Market Group)
    'IPCONGD': 'IPCONGD',                               # IP: Consumer Goods
    'IPDCONGD': 'IPDCONGD',                             # IP: Durable Consumer Goods
    'IPNCONGD': 'IPNCONGD',                             # IP: Nondurable Consumer Goods
    'IPBUSEQ': 'IPBUSEQ',                               # IP: Business Equipment
    'IPMAT': 'IPMAT',                                   # IP: Materials
    'IPDMAT': 'IPDMAT',                                 # IP: Durable Materials
    'IPNMAT': 'IPNMAT',                                 # IP: Nondurable Materials
    'IPMANSICS': 'IPMANSICS',                           # IP: Manufacturing (SIC)
    'IPB51222S': 'IPB51222S',                           # IP: Residential Utilities
    'IPFUELS': 'IPFUELS',                               # IP: Fuels

    # ---- LABOR MARKET (24 series) ----
    'CUMFNS': 'CUMFNS',                                 # Capacity Utilization: Manufacturing
    'HWI': 'HWI',                                       # Help-Wanted Index (discontinued)
    'HWIURATIO': 'HWIURATIO',                           # Help-Wanted/Unemployment Ratio
    'CLF16OV': 'CLF16OV',                               # Civilian Labor Force
    'CE16OV': 'CE16OV',                                 # Civilian Employment
    'UNRATE': 'UNRATE',                                 # Unemployment Rate
    'UEMPMEAN': 'UEMPMEAN',                             # Avg. Weeks Unemployed
    'UEMPLT5': 'UEMPLT5',                               # <5 Weeks Unemployed
    'UEMP5TO14': 'UEMP5TO14',                           # 5-14 Weeks Unemployed
    'UEMP15OV': 'UEMP15OV',                             # 15+ Weeks Unemployed
    'UEMP15T26': 'UEMP15T26',                           # 15-26 Weeks Unemployed
    'UEMP27OV': 'UEMP27OV',                             # 27+ Weeks Unemployed
    'CLAIMSx': 'CLAIMS',                                 # Initial Jobless Claims
    'PAYEMS': 'PAYEMS',                                 # All Employees: Total Nonfarm
    'USGOOD': 'USGOOD',                                 # Goods-Producing Employment
    'CES1021000001': 'CES1021000001',                   # Mining & Logging Employment
    'USCONS': 'USCONS',                                 # Construction Employment
    'MANEMP': 'MANEMP',                                 # Manufacturing Employment
    'DMANEMP': 'DMANEMP',                               # Durable Goods Employment
    'NDMANEMP': 'NDMANEMP',                             # Nondurable Goods Employment
    'SRVPRD': 'SRVPRD',                                 # Service-Providing Employment
    'USTPU': 'USTPU',                                   # Trade, Transportation & Utilities Employment
    'USWTRADE': 'USWTRADE',                             # Wholesale Trade Employment
    'USTRADE': 'USTRADE',                               # Retail Trade Employment
    'USFIRE': 'USFIRE',                                 # Financial Activities Employment
    'USGOVT': 'USGOVT',                                 # Government Employment
    'CES0600000007': 'CES0600000007',                   # Goods-Producing Employment
    'AWOTMAN': 'AWOTMAN',                               # Avg Weekly Overtime Hours: Mfg
    'AWHMAN': 'AWHMAN',                                 # Avg Weekly Hours: Manufacturing

    # ---- HOUSING (10 series) ----
    'HOUST': 'HOUST',                                   # Housing Starts: Total
    'HOUSTNE': 'HOUSTNE',                               # Housing Starts: Northeast
    'HOUSTMW': 'HOUSTMW',                               # Housing Starts: Midwest
    'HOUSTS': 'HOUSTS',                                 # Housing Starts: South
    'HOUSTW': 'HOUSTW',                                 # Housing Starts: West
    'PERMIT': 'PERMIT',                                 # Building Permits: Total
    'PERMITNE': 'PERMITNE',                             # Building Permits: Northeast
    'PERMITMW': 'PERMITMW',                             # Building Permits: Midwest
    'PERMITS': 'PERMITS',                               # Building Permits: South
    'PERMITW': 'PERMITW',                               # Building Permits: West

    # ---- CONSUMPTION, ORDERS & INVENTORIES (6 series) ----
    'ACOGNO': 'ACOGNO',                                 # New Orders for Consumer Goods
    'AMDMNOx': 'AMDMNO',                                # New Orders for Durable Goods
    'ANDENOx': 'ANDENO',                                # New Orders for Nondefense Capital Goods
    'AMDMUOx': 'AMDMUO',                                # Unfilled Orders for Durable Goods
    'BUSINVx': 'BUSINV',                                # Total Business Inventories
    'ISRATIOx': 'ISRATIO',                              # Inventory/Sales Ratio

    # ---- MONEY & CREDIT (12 series) ----
    'M1SL': 'M1SL',                                     # M1 Money Supply
    'M2SL': 'M2SL',                                     # M2 Money Supply
    'M2REAL': 'M2REAL',                                 # Real M2 Money Supply
    'BOGMBASE': 'BOGMBASE',                             # Monetary Base
    'TOTRESNS': 'TOTRESNS',                             # Total Reserves of Depository Institutions
    'NONBORRES': 'NONBORRES',                           # Nonborrowed Reserves
    'BUSLOANS': 'BUSLOANS',                             # Commercial & Industrial Loans
    'REALLN': 'REALLN',                                 # Real Estate Loans at Banks
    'NONREVSL': 'NONREVSL',                             # Consumer Credit Outstanding
    'CONSPI': 'CONSPI',                                 # Consumer Price Index (concluded series)
    'DTCOLNVHFNM': 'DTCOLNVHFNM',                       # Consumer Motor Vehicle Loans Outstanding
    'DTCTHFNM': 'DTCTHFNM',                             # Total Consumer Loans and Leases Outstanding

    # ---- INTEREST & EXCHANGE RATES (22 series) ----
    'FEDFUNDS': 'FEDFUNDS',                             # Effective Federal Funds Rate
    'CP3Mx': 'CP3M',                                    # 3-Month AA Commercial Paper Rate
    'TB3MS': 'TB3MS',                                   # 3-Month Treasury Bill
    'TB6MS': 'TB6MS',                                   # 6-Month Treasury Bill
    'GS1': 'GS1',                                       # 1-Year Treasury Rate
    'GS5': 'GS5',                                       # 5-Year Treasury Rate
    'GS10': 'GS10',                                     # 10-Year Treasury Rate
    'AAA': 'AAA',                                       # AAA Corporate Bond Yield
    'BAA': 'BAA',                                       # BAA Corporate Bond Yield
    'COMPAPFFx': 'COMPAPFF',                            # CP - Fed Funds Spread
    'TB3SMFFM': 'TB3SMFFM',                             # 3M T-Bill - Fed Funds
    'TB6SMFFM': 'TB6SMFFM',                             # 6M T-Bill - Fed Funds
    'T1YFFM': 'T1YFFM',                                 # 1Y T-Bill - Fed Funds
    'T5YFFM': 'T5YFFM',                                 # 5Y T-Bill - Fed Funds
    'T10YFFM': 'T10YFFM',                               # 10Y T-Bill - Fed Funds
    'AAAFFM': 'AAAFFM',                                 # AAA - Fed Funds
    'BAAFFM': 'BAAFFM',                                 # BAA - Fed Funds
    'TWEXAFEGSMTHx': 'TWEXAFEGSMTH',                    # Trade Weighted U.S. Dollar Index
    'EXSZUSx': 'EXSZUS',                                # Switzerland / U.S. FX Rate
    'EXJPUSx': 'EXJPUS',                                # Japan / U.S. FX Rate
    'EXUSUKx': 'EXUSUK',                                # U.K. / U.S. FX Rate
    'EXCAUSx': 'EXCAUS',                                # Canada / U.S. FX Rate

    # ---- PRICES (20 series) ----
    'WPSFD49207': 'WPSFD49207',                         # PPI: Finished Goods
    'WPSFD49502': 'WPSFD49502',                         # PPI: Finished Consumer Goods
    'WPSID61': 'WPSID61',                               # PPI: Intermediate Materials
    'WPSID62': 'WPSID62',                               # PPI: Crude Materials
    'OILPRICEx': 'OILPRICE',                            # Crude Oil Price
    'PPICMM': 'PPICMM',                                 # PPI: Metals & Metal Products
    'CPIAUCSL': 'CPIAUCSL',                             # CPI: All Items
    'CPIAPPSL': 'CPIAPPSL',                             # CPI: Apparel
    'CPITRNSL': 'CPITRNSL',                             # CPI: Transportation
    'CPIMEDSL': 'CPIMEDSL',                             # CPI: Medical Care
    'CUSR0000SAC': 'CUSR0000SAC',                       # CPI: Commodities
    'CUSR0000SAD': 'CUSR0000SAD',                       # CPI: Durables
    'CUSR0000SAS': 'CUSR0000SAS',                       # CPI: Services
    'CPIULFSL': 'CPIULFSL',                             # CPI: All Items Less Food
    'CUSR0000SA0L2': 'CUSR0000SA0L2',                   # CPI: All Items Less Shelter
    'CUSR0000SA0L5': 'CUSR0000SA0L5',                   # CPI: All Items Less Medical Care
    'PCEPI': 'PCEPI',                                   # Personal Consumption Expenditures: Chain-type Price Index
    'DDURRG3M086SBEA': 'DDURRG3M086SBEA',               # PCE: Durable Goods
    'DNDGRG3M086SBEA': 'DNDGRG3M086SBEA',               # PCE: Nondurable Goods
    'DSERRG3M086SBEA': 'DSERRG3M086SBEA',               # PCE: Services

    # ---- STOCK MARKET (3 series) ----
    'S&P 500': 'SP500',                                 # S&P 500 Index
    'S&P div yield': None,                               # Not a direct FRED API series
    'S&P PE ratio': None,                                # Not a direct FRED API series

    # ---- SENTIMENT (3 series) ----
    'UMCSENTx': 'UMCSENT',                              # Consumer Sentiment Index
    'VIXCLSx': 'VIXCLS',                                # CBOE Volatility Index (VIX)
    'INVEST': 'INVEST',                                 # Investment in Securities (from FRB)

    # ---- ADDITIONAL LABOR (4 series) ----
    'CES0600000008': 'CES0600000008',                   # Avg Hourly Earnings: Goods-Producing
    'CES2000000008': 'CES2000000008',                   # Avg Hourly Earnings: Construction
    'CES3000000008': 'CES3000000008',                   # Avg Hourly Earnings: Manufacturing

    # ---- EXTRA: useful series not in official FRED-MD ----
    'GS2': 'GS2',                                       # 2-Year Treasury Rate
    'GS20': 'GS20',                                     # 20-Year Treasury Rate
    'GS30': 'GS30',                                     # 30-Year Treasury Rate
    'T10Y2YM': 'T10Y2YM',                               # 10Y-2Y Treasury Spread
    'TEDRATE': 'TEDRATE',                               # TED Spread
    'M3SL': 'M3SL',                                     # M3 Money Supply
    'MOVE': 'MOVE',                                     # MOVE Bond Volatility Index
    'IC4WSA': 'IC4WSA',                                 # Capacity Utilization: Total Index
    'EMRATIO': 'EMRATIO',                               # Employment-Population Ratio
}

# Build indicator dict: FRED API series ID -> FRED-MD column name
# This drives the fetch loop with ~120+ series from the full FRED-MD database.
INDICATORS = {v: k for k, v in FRED_MD_SERIES.items() if v is not None}

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

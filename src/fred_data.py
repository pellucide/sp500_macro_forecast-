"""
Data Acquisition Pipeline using FRED API with ALFRED Real-Time Vintages
Implements point-in-time discipline to prevent look-ahead bias
"""

import warnings
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import logging
import os
import hashlib

try:
    import fredapi
    FREDAPI_AVAILABLE = True
except ImportError:
    FREDAPI_AVAILABLE = False
    warnings.warn("fredapi not installed. Install with: pip install fredapi")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    warnings.warn("yfinance not installed. Install with: pip install yfinance")

from .config import DataConfig, DATA_DIR

logger = logging.getLogger(__name__)

# Cache directory for downloaded data
CACHE_DIR = os.path.join(DATA_DIR, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)


def _get_cache_key(*args) -> str:
    """Generate a cache key from arguments."""
    key_str = '_'.join(str(arg) for arg in args)
    return hashlib.md5(key_str.encode()).hexdigest()


def _get_cache_path(prefix: str, cache_key: str) -> str:
    """Get the full cache file path."""
    return os.path.join(CACHE_DIR, f"{prefix}_{cache_key}.csv")


def load_from_cache(prefix: str, cache_key: str) -> Optional[pd.DataFrame]:
    """Load data from cache if it exists."""
    cache_path = _get_cache_path(prefix, cache_key)
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            logger.info(f"Loaded data from cache: {cache_path}")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache {cache_path}: {e}")
    return None


def save_to_cache(df: pd.DataFrame, prefix: str, cache_key: str) -> None:
    """Save data to cache."""
    cache_path = _get_cache_path(prefix, cache_key)
    try:
        df.to_csv(cache_path)
        logger.info(f"Saved data to cache: {cache_path}")
    except Exception as e:
        logger.warning(f"Failed to save cache {cache_path}: {e}")


def clear_cache(prefix: Optional[str] = None) -> None:
    """Clear cached data files."""
    if prefix:
        # Clear specific prefix
        pattern = os.path.join(CACHE_DIR, f"{prefix}_*.csv")
        import glob
        for f in glob.glob(pattern):
            os.remove(f)
            logger.info(f"Removed cache: {f}")
    else:
        # Clear all cache
        for f in os.listdir(CACHE_DIR):
            if f.endswith('.csv'):
                os.remove(os.path.join(CACHE_DIR, f))
        logger.info("Cleared all cache files")


class FREDDataLoader:
    """
    Loads macroeconomic indicators from FRED with real-time vintage support.
    Implements point-in-time discipline using ALFRED for look-ahead bias prevention.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED data loader.

        Args:
            api_key: FRED API key. If None, reads from config or environment.
        """
        if not FREDAPI_AVAILABLE:
            raise ImportError("fredapi package required. Install with: pip install fredapi")

        self.api_key = api_key or DataConfig.FRED_API_KEY
        if not self.api_key:
            warnings.warn(
                "No FRED API key provided. Set FRED_API_KEY environment variable "
                "or pass api_key parameter."
            )
            self.fred = None
        else:
            self.fred = fredapi.Fred(self.api_key)

        self.categories = DataConfig.INDICATOR_CATEGORIES
        self.all_indicators = DataConfig.ALL_INDICATORS
        self._fetch_count = 0
        self._last_fetch_time = None

    def _rate_limit(self, min_interval: float = 0.2):
        """Rate limit: sleep between API requests to avoid hitting limits."""
        if self._fetch_count > 0 and self._last_fetch_time is not None:
            elapsed = time.time() - self._last_fetch_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

    def fetch_indicator(
        self,
        indicator: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        vintage_date: Optional[str] = None
    ) -> pd.Series:
        """
        Fetch a single indicator from FRED.

        Args:
            indicator: FRED series ID
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            vintage_date: Vintage date for ALFRED (YYYY-MM-DD)

        Returns:
            pd.Series with the indicator data
        """
        if self.fred is None:
            raise ValueError("FRED API key required to fetch data")

        self._rate_limit()

        self._fetch_count += 1
        self._last_fetch_time = time.time()

        try:
            if vintage_date and DataConfig.USE_REALTIME_VINTAGES:
                # Fetch vintage data from ALFRED
                # Note: fredapi's get_series() supports vintage_date parameter
                data = self.fred.get_series(
                    indicator,
                    observation_start=start_date,
                    observation_end=end_date,
                    vintage_date=vintage_date
                )
            else:
                # Fetch latest data
                data = self.fred.get_series(indicator, start_date, end_date)

            if isinstance(data, pd.DataFrame):
                series = data.iloc[:, 0]
            else:
                series = data

            series.name = indicator
            return series

        except Exception as e:
            logger.warning(f"Failed to fetch {indicator}: {e}")
            return pd.Series(dtype=float, name=indicator)

    def fetch_all_indicators(
        self,
        start_date: str = "1959-01-01",
        end_date: Optional[str] = None,
        vintage_date: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch all macroeconomic indicators from FRED-MD categories.

        Args:
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            vintage_date: Vintage date for real-time discipline
            use_cache: Whether to use cached data if available

        Returns:
            pd.DataFrame with all indicators as columns
        """
        # Check cache first
        if use_cache:
            cache_key = _get_cache_key(start_date, end_date, vintage_date)
            cached = load_from_cache("fred_indicators", cache_key)
            if cached is not None:
                return cached

        all_data = []

        for indicator in self.all_indicators:
            series = self.fetch_indicator(
                indicator,
                start_date=start_date,
                end_date=end_date,
                vintage_date=vintage_date
            )
            if not series.empty:
                all_data.append(series)

        if not all_data:
            return pd.DataFrame()

        # Combine into DataFrame, ensuring monthly frequency
        df = pd.concat(all_data, axis=1)
        df.index = pd.to_datetime(df.index)
        df = df.resample('ME').last()  # Convert to end-of-month

        # Save to cache
        if use_cache:
            cache_key = _get_cache_key(start_date, end_date, vintage_date)
            save_to_cache(df, "fred_indicators", cache_key)

        return df

    def fetch_spx_returns(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.Series:
        """
        Fetch S&P 500 monthly returns.

        Args:
            start_date: Start date
            end_date: End date
            use_cache: Whether to use cached data if available

        Returns:
            pd.Series with monthly returns
        """
        if self.fred is None:
            raise ValueError("FRED API key required")

        # Check cache first
        if use_cache:
            cache_key = _get_cache_key(start_date, end_date, "spx")
            cached = load_from_cache("spx_returns", cache_key)
            if cached is not None:
                return cached['SP500_return']

        try:
            # S&P 500 price index (daily) - try FRED first
            spx = self.fred.get_series(
                "SP500",
                start_date=start_date,
                end_date=end_date
            )

            # Convert to DataFrame and resample to monthly (end of month)
            spx_df = pd.DataFrame(spx)
            spx_df.index = pd.to_datetime(spx_df.index)
            spx_monthly = spx_df.resample('ME').last().iloc[:, 0]

            # Calculate monthly returns
            returns = spx_monthly.pct_change().dropna()
            returns.name = "SP500_return"

            # If we got very few observations, try yfinance as fallback
            if len(returns) < 100:
                logger.warning("FRED SP500 data is sparse, trying yfinance fallback...")
                raise ValueError("Insufficient data from FRED")

            # Save to cache
            if use_cache:
                cache_key = _get_cache_key(start_date, end_date, "spx")
                save_to_cache(returns.to_frame(), "spx_returns", cache_key)

            return returns

        except Exception as e:
            logger.warning(f"FRED SP500 fetch failed ({e}), trying yfinance fallback...")
            try:
                import yfinance as yf
                # Use yfinance to get longer history
                sp500 = yf.download('^GSPC', start='1959-01-01', progress=False)
                close = sp500['Close']
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                monthly = close.resample('ME').last()
                returns = monthly.pct_change().dropna()
                returns.name = "SP500_return"

                # Save to cache
                if use_cache:
                    cache_key = _get_cache_key(start_date, end_date, "spx")
                    save_to_cache(returns.to_frame(), "spx_returns", cache_key)

                logger.info(f"yfinance fallback: {len(returns)} monthly returns")
                return returns
            except Exception as yf_err:
                logger.error(f"yfinance fallback also failed: {yf_err}")

            return pd.Series(dtype=float, name="SP500_return")

    def fetch_sector_returns(self, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch sector ETF returns from FRED for sector rotation prediction.

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            pd.DataFrame with sector relative returns (sector - SP500)
        """
        if self.fred is None:
            raise ValueError("FRED API key required")

        # Sector ETF tickers available on FRED
        sector_etfs = {
            'XLB': 'Materials',      # Materials
            'XLE': 'Energy',         # Energy
            'XLF': 'Financials',      # Financials
            'XLI': 'Industrials',     # Industrials
            'XLK': 'Technology',      # Technology
            'XLU': 'Utilities',       # Utilities
            'XLV': 'Healthcare',      # Healthcare
            'XLP': 'ConsumerStaples', # Consumer Staples
            'XLY': 'ConsumerDiscretionary',  # Consumer Discretionary
            'XLRE': 'RealEstate',     # Real Estate
            'XLC': 'CommunicationServices'  # Communication Services
        }

        all_returns = {}

        for ticker, name in sector_etfs.items():
            try:
                self._rate_limit()

                # Fetch sector ETF price
                sector_data = self.fred.get_series(
                    ticker,
                    start_date=start_date,
                    end_date=end_date
                )

                # Convert to monthly returns
                sector_df = pd.DataFrame(sector_data)
                sector_df.index = pd.to_datetime(sector_df.index)
                sector_monthly = sector_df.resample('ME').last().iloc[:, 0]
                sector_returns = sector_monthly.pct_change().dropna()
                sector_returns.name = name

                all_returns[name] = sector_returns
                self._fetch_count += 1
                self._last_fetch_time = time.time()

            except Exception as e:
                logger.warning(f"Failed to fetch {ticker}: {e}")
                continue

        if not all_returns:
            logger.error("No sector data fetched")
            return pd.DataFrame()

        # Combine all sector returns
        sector_df = pd.DataFrame(all_returns)

        # Fetch SP500 for relative returns
        try:
            spx = self.fred.get_series("SP500", start_date=start_date, end_date=end_date)
            spx_df = pd.DataFrame(spx)
            spx_df.index = pd.to_datetime(spx_df.index)
            spx_monthly = spx_df.resample('ME').last().iloc[:, 0]
            spx_returns = spx_monthly.pct_change().dropna()
            spx_returns.name = 'SP500'

            # Align with sector data
            common_idx = sector_df.index.intersection(spx_returns.index)
            sector_df = sector_df.loc[common_idx]
            spx_returns = spx_returns.loc[common_idx]

            # Calculate relative returns (sector - SP500)
            relative_returns = sector_df.sub(spx_returns, axis=0)

            logger.info(f"Fetched {len(relative_returns.columns)} sector relative returns")
            return relative_returns

        except Exception as e:
            logger.error(f"Failed to fetch SP500 for relative returns: {e}")
            return sector_df

    def fetch_sector_returns_yfinance(
        self,
        start_date: str,
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch sector ETF returns from Yahoo Finance for sector rotation prediction.

        Args:
            start_date: Start date
            end_date: End date (defaults to today)
            use_cache: Whether to use cached data if available

        Returns:
            DataFrame with relative sector returns (sector - SP500)
        """
        if not YFINANCE_AVAILABLE:
            logger.error("yfinance not installed. Cannot fetch sector data.")
            return pd.DataFrame()

        # Check cache first
        if use_cache:
            cache_key = _get_cache_key(start_date, end_date, "sectors")
            cached = load_from_cache("sector_returns", cache_key)
            if cached is not None:
                return cached

        # Sector ETF tickers
        sector_etfs = {
            'XLB': 'Materials',
            'XLE': 'Energy',
            'XLF': 'Financials',
            'XLI': 'Industrials',
            'XLK': 'Technology',
            'XLU': 'Utilities',
            'XLV': 'Healthcare',
            'XLP': 'ConsumerStaples',
            'XLY': 'ConsumerDiscretionary',
            'XLRE': 'RealEstate',
            'XLC': 'CommunicationServices'
        }

        all_prices = {}
        spx_prices = None

        # Fetch SP500 first
        try:
            logger.info("Fetching SP500 data from Yahoo Finance...")
            spx = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
            if len(spx) > 0:
                # Handle different yfinance output formats
                if isinstance(spx.columns, pd.MultiIndex):
                    spx_prices = spx['Close']['^GSPC']
                elif 'Close' in spx.columns:
                    spx_prices = spx['Close']
                elif 'Adj Close' in spx.columns:
                    spx_prices = spx['Adj Close']
                else:
                    spx_prices = spx.iloc[:, 3]  # Usually Adj Close is 4th column
                logger.info(f"SP500: {len(spx_prices)} observations")
        except Exception as e:
            logger.error(f"Failed to fetch SP500: {e}")
            return pd.DataFrame()

        # Fetch each sector ETF
        for ticker, name in sector_etfs.items():
            try:
                logger.info(f"Fetching {ticker} ({name})...")
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if len(data) > 0:
                    # Handle different yfinance output formats
                    if isinstance(data.columns, pd.MultiIndex):
                        all_prices[name] = data['Close'][ticker]
                    elif 'Close' in data.columns:
                        all_prices[name] = data['Close']
                    elif 'Adj Close' in data.columns:
                        all_prices[name] = data['Adj Close']
                    else:
                        all_prices[name] = data.iloc[:, 3]
                    logger.info(f"  {ticker}: {len(data)} observations")
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                logger.warning(f"Failed to fetch {ticker}: {e}")
                continue

        if not all_prices:
            logger.error("No sector data fetched")
            return pd.DataFrame()

        # Combine sector prices
        sector_prices = pd.DataFrame(all_prices)

        # Resample to monthly
        sector_monthly = sector_prices.resample('ME').last()
        spx_monthly = spx_prices.resample('ME').last()

        # Calculate returns
        sector_returns = sector_monthly.pct_change().dropna()
        spx_returns = spx_monthly.pct_change().dropna()

        # Align indices
        common_idx = sector_returns.index.intersection(spx_returns.index)
        sector_returns = sector_returns.loc[common_idx]
        spx_returns = spx_returns.loc[common_idx]

        # Calculate relative returns (sector - SP500)
        relative_returns = sector_returns.sub(spx_returns.values.reshape(-1, 1), axis=0)

        # Save to cache
        if use_cache:
            cache_key = _get_cache_key(start_date, end_date, "sectors")
            save_to_cache(relative_returns, "sector_returns", cache_key)

        logger.info(f"Fetched {len(relative_returns.columns)} sector relative returns")
        return relative_returns

    def get_available_vintage_dates(self, indicator: str, year: int) -> List[str]:
        """
        Get available vintage dates for a given indicator and year.
        Used for ALFRED real-time data acquisition.

        Args:
            indicator: FRED series ID
            year: Year to search

        Returns:
            List of available vintage dates
        """
        if self.fred is None:
            return []

        try:
            # Get vintage dates for the indicator (available in fredapi)
            vintage_dates = self.fred.get_series_vintage_dates(indicator)
            # Filter by year
            return [d for d in vintage_dates if str(d).startswith(str(year))]
        except Exception as e:
            logger.warning(f"Failed to get vintage dates for {indicator}: {e}")
            return []


class DataProcessor:
    """
    Processes and transforms macroeconomic data for modeling.
    Handles missing values, outliers, and transformations.
    """

    def __init__(self):
        self.category_mapping = DataConfig.INDICATOR_CATEGORIES

    def create_category_groups(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Group indicators by economic category.

        Args:
            df: DataFrame with all indicators

        Returns:
            Dictionary mapping category names to DataFrames
        """
        groups = {}

        for category, indicators in self.category_mapping.items():
            # Filter to indicators that exist in the data
            available = [ind for ind in indicators if ind in df.columns]
            if available:
                groups[category] = df[available].copy()

        return groups

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        method: str = "ffill"
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            df: Input DataFrame
            method: Method for handling missing values ("ffill", "bfill", "interpolate")

        Returns:
            DataFrame with missing values handled
        """
        df_clean = df.copy()

        if method == "ffill":
            df_clean = df_clean.ffill()
        elif method == "bfill":
            df_clean = df_clean.bfill()
        elif method == "interpolate":
            df_clean = df_clean.interpolate(method='linear')

        # Drop rows with remaining NaN values (usually at the beginning)
        df_clean = df_clean.dropna(how='all')

        return df_clean

    def winsorize_outliers(
        self,
        df: pd.DataFrame,
        lower: float = 0.01,
        upper: float = 0.99
    ) -> pd.DataFrame:
        """
        Winsorize outliers to reduce the impact of extreme values.

        Args:
            df: Input DataFrame
            lower: Lower percentile for winsorization
            upper: Upper percentile for winsorization

        Returns:
            Winsorized DataFrame
        """
        return df.clip(lower=df.quantile(lower), upper=df.quantile(upper), axis=1)

    def compute_returns(self, df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
        """
        Compute percentage returns for price/volume series.

        Args:
            df: Input DataFrame
            periods: Number of periods for return calculation

        Returns:
            DataFrame with returns
        """
        return df.pct_change(periods=periods).dropna()

    def align_data(
        self,
        indicators: pd.DataFrame,
        target: pd.Series,
        max_lag: int = 1
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Align indicators with target variable using appropriate lag structure.
        Uses point-in-time discipline: predict r_{t+1} using data available at t.

        FIXED: Now properly implements 1-month-ahead forecasting by shifting
        the target forward. X[t] predicts y[t+1], not y[t].

        Args:
            indicators: DataFrame of macroeconomic indicators
            target: Target variable (S&P 500 returns)
            max_lag: Maximum lag for alignment

        Returns:
            Tuple of (aligned indicators, aligned target)
        """
        # Create a combined DataFrame with both indicators and target
        combined = indicators.copy()
        combined['target'] = target

        # Remove rows with NaN in target
        combined = combined.dropna(subset=['target'])

        # FIXED: Use only ffill() - bfill() causes look-ahead bias
        # Forward fill uses only past data (OK), backward fill uses future data (NOT OK)
        combined = combined.ffill()

        # FIXED: Shift target forward by 1 for proper 1-month-ahead forecasting
        # This ensures X[t] predicts the return for month t+1, not month t
        # After shifting: target[t] = original_return[t+1]
        # Result: indicators and target are now aligned for forecasting
        combined['target'] = combined['target'].shift(-1)

        # Remove the last row (which now has NaN target due to forward shift)
        combined = combined.dropna(subset=['target'])

        # Split back into indicators and target
        aligned_indicators = combined.drop(columns=['target'])
        aligned_target = combined['target']

        # Verify alignment
        assert len(aligned_indicators) == len(aligned_target), \
            "Indicators and target must have same length after alignment"

        return aligned_indicators, aligned_target

    def split_train_val_test(
        self,
        df: pd.DataFrame,
        train_end: str,
        val_end: Optional[str] = None,
        test_end: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Split data into train, validation, and test sets respecting temporal order.

        Args:
            df: Combined DataFrame
            train_end: End date for training set
            val_end: End date for validation set
            test_end: End date for test set

        Returns:
            Dictionary with train/val/test DataFrames
        """
        splits = {}

        # Training set
        train_mask = df.index <= train_end
        splits['train'] = df[train_mask].copy()

        # Validation set (if provided)
        if val_end:
            val_mask = (df.index > train_end) & (df.index <= val_end)
            splits['val'] = df[val_mask].copy()

        # Test set (if provided)
        if test_end:
            test_mask = df.index > val_end if val_end else df.index > train_end
            test_mask = test_mask & (df.index <= test_end)
            splits['test'] = df[test_mask].copy()

        return splits


def download_fred_data(
    api_key: str,
    start_date: str = "1959-01-01",
    end_date: Optional[str] = None,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function to download all FRED-MD data.

    Args:
        api_key: FRED API key
        start_date: Start date for data
        end_date: End date for data
        save_path: Optional path to save the downloaded data

    Returns:
        DataFrame with all macroeconomic indicators
    """
    loader = FREDDataLoader(api_key=api_key)

    df = loader.fetch_all_indicators(
        start_date=start_date,
        end_date=end_date
    )

    if save_path:
        df.to_csv(save_path)
        logger.info(f"Saved data to {save_path}")

    return df


def download_with_vintage(
    api_key: str,
    vintage_date: str,
    indicators: Optional[List[str]] = None,
    start_date: str = "1959-01-01"
) -> pd.DataFrame:
    """
    Download data as it would have been available on a specific vintage date.
    Implements strict point-in-time discipline for backtesting.

    Args:
        api_key: FRED API key
        vintage_date: The date for which data would have been available
        indicators: List of indicator codes (defaults to all FRED-MD indicators)
        start_date: Start date for data retrieval

    Returns:
        DataFrame with vintage-appropriate data
    """
    loader = FREDDataLoader(api_key=api_key)

    if indicators is None:
        indicators = loader.all_indicators

    all_data = []
    for indicator in indicators:
        series = loader.fetch_indicator(
            indicator,
            start_date=start_date,
            vintage_date=vintage_date
        )
        if not series.empty:
            all_data.append(series)

    if not all_data:
        return pd.DataFrame()

    df = pd.concat(all_data, axis=1)
    df.index = pd.to_datetime(df.index)
    df = df.resample('ME').last()

    return df


# =============================================================================
# Sample Data Generator (for testing without API key)
# =============================================================================
def generate_sample_data(
    n_periods: int = 500,
    n_indicators: int = 50,
    start_date: str = "1959-01-01",
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """
    Generate sample macroeconomic data and S&P 500 returns for testing.
    Should only be used when FRED API is unavailable.

    Args:
        n_periods: Number of time periods
        n_indicators: Number of indicators to generate
        start_date: Start date
        seed: Random seed for reproducibility

    Returns:
        Tuple of (indicators DataFrame, target Series, groups dict)
    """
    np.random.seed(seed)

    dates = pd.date_range(start=start_date, periods=n_periods, freq='ME')

    # Generate correlated indicators (simulating macroeconomic variables)
    # Use exactly 5 categories to match INDICATOR_CATEGORIES
    categories = ['output_income', 'labor', 'inflation', 'interest', 'sentiment']
    n_cats = len(categories)
    indicators_per_cat = n_indicators // n_cats

    all_series = {}
    groups = {}
    for i, cat in enumerate(categories):
        cat_indicators = []
        for j in range(indicators_per_cat):
            # Random walk base
            base = np.cumsum(np.random.randn(n_periods) * 0.02)
            # Add category-specific patterns
            indicator_name = f"{cat}_{j}"
            all_series[indicator_name] = base + np.sin(np.arange(n_periods) / 20 + i) * 0.5
            cat_indicators.append(indicator_name)
        groups[cat] = cat_indicators

    indicators = pd.DataFrame(all_series, index=dates)

    # Generate target (S&P 500 returns) as function of indicators
    weights = np.random.randn(n_indicators) * 0.01
    signal = (indicators.values @ weights)
    target = pd.Series(
        signal + np.random.randn(n_periods) * 0.05,
        index=dates,
        name='SP500_return'
    )

    # Remove first row (no return for first observation)
    indicators = indicators.iloc[1:]
    target = target.iloc[1:]

    return indicators, target, groups


if __name__ == "__main__":
    # Example usage
    print("FRED Data Loader - Sample Usage")
    print("=" * 50)

    # Check if API key is available
    api_key = DataConfig.FRED_API_KEY

    if api_key:
        # Download real data
        print("Downloading FRED-MD data...")
        loader = FREDDataLoader(api_key=api_key)
        df = loader.fetch_all_indicators(start_date="1959-01-01")
        print(f"Downloaded {len(df.columns)} indicators from {df.index[0]} to {df.index[-1]}")
    else:
        # Generate sample data
        print("No API key found. Generating sample data for testing...")
        indicators, target, _ = generate_sample_data()
        print(f"Generated {len(indicators.columns)} indicators and {len(target)} target values")
        print(f"Date range: {indicators.index[0]} to {indicators.index[-1]}")
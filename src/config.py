"""
Configuration settings for S&P 500 Macroeconomic Forecasting Project
State-Dependent Supervised Screening & Regularized Factor (SSRF) Architecture
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
BACKTEST_DIR = PROJECT_ROOT / "backtest"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, BACKTEST_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Data Configuration
# =============================================================================
class DataConfig:
    """Configuration for data acquisition and processing"""

    # FRED API settings - loaded from environment variable
    FRED_API_KEY = os.getenv("FRED_API_KEY", "")

    # If FRED_API_KEY is not set, you can set it directly here for development
    # IMPORTANT: For production, use environment variable
    if not FRED_API_KEY:
        # Fallback for testing - remove or set via environment in production
        FRED_API_KEY = "48f0923658be7d90ba311c4a55138377"

    # FRED-MD indicator categories (updated with verified working codes)
    INDICATOR_CATEGORIES = {
        "output_income": [
            "GDPPOT", "GDPC1", "GNP", "FYFSD"
        ],
        "labor": [
            "UNRATE", "EMRATIO", "PAYEMS", "USGOVT", "HOUST", "HOUSTNE", "HOUSTMW",
            "HOUSTS", "HOUSTW", "HOUST1F", "PERMIT", "PERMITNE", "PERMITW", "PERMITS"
        ],
        "inflation": [
            "CPIAUCSL", "CPILFESL", "PPIFGS", "PCECTPI", "PCEPILFE", "GDPDEF"
        ],
        "interest": [
            "TB3MS", "TB6MS", "TB1YR", "GS1", "GS2", "GS3", "GS5", "GS7", "GS10",
            "GS20", "AAA", "BAA", "TEDRATE", "T10Y2YM", "T10YFFM", "T5YFFM",
            "AAAFFM", "BAAFFM", "MORTGAGE30US", "FEDFUNDS", "DFII10"
        ],
        "sentiment": [
            "PCCE", "TCD", "IC4WSA"
        ]
    }

    # All indicators flattened
    ALL_INDICATORS = [ind for cats in INDICATOR_CATEGORIES.values() for ind in cats]

    # Data vintage settings
    USE_REALTIME_VINTAGES = True  # Use ALFRED for look-ahead bias prevention
    VINTAGE_OFFSET_MONTHS = 1  # Use data from t-1 for forecasts at time t

    # Sector ETF tickers for sector rotation strategy
    # These can be used with yfinance for real market data
    SECTOR_ETFS = {
        "Materials": "XLB",        # Materials Select Sector SPDR Fund
        "Energy": "XLE",            # Energy Select Sector SPDR Fund
        "Financials": "XLF",        # Financial Select Sector SPDR Fund
        "Industrials": "XLI",       # Industrial Select Sector SPDR Fund
        "Technology": "XLK",        # Technology Select Sector SPDR Fund
        "Consumer_Staples": "XLP",   # Consumer Staples Select Sector SPDR Fund
        "Health_Care": "XLV",       # Health Care Select Sector SPDR Fund
        "Utilities": "XLU",         # Utilities Select Sector SPDR Fund
        "Real_Estate": "XLRE",      # Real Estate Select Sector SPDR Fund
        "Communication": "XLC",    # Communication Services Select Sector SPDR Fund
        "Consumer_Discretionary": "XLY",  # Consumer Discretionary Select Sector SPDR Fund
        "S&P_500": "SPY"            # Benchmark S&P 500 ETF
    }

    # Default sector for analysis
    DEFAULT_SECTOR = "S&P_500"

# =============================================================================
# Model Configuration
# =============================================================================
class ModelConfig:
    """Configuration for SSRF model architecture"""

    # Stage 1: Group-wise supervised screening
    SCREENING_T_STAT_THRESHOLD = 0.5  # Balance for feature selection

    # Stage 2: Predictive scaling
    USE_PREDICTIVE_SCALING = True

    # Stage 3: Supervised factor extraction
    N_FACTORS = 10  # K = 10 latent factors

    # Stage 4: Regime proxy
    REGIME_WINDOW = 6  # Shorter window for more regime sensitivity
    VOLATILITY_PERCENTILE_REF = "training"  # Compute relative to training distribution

    # Regularization - reduced for more signal
    ELASTIC_NET_ALPHA = 0.0001  # Lower regularization
    ELASTIC_NET_L1_RATIO = 0.5  # Balance between L1 and L2
    USE_ELASTIC_NET_CV = True  # Use cross-validation for alpha selection

    # Optimization
    N_INNER_CV_FOLDS = 5  # For nested time series CV

# =============================================================================
# Backtesting Configuration
# =============================================================================
class BacktestConfig:
    """Configuration for walk-forward backtesting"""

    # Initial training window
    INITIAL_TRAIN_WINDOW = 60  # 60 months (5 years) - reduced for more test periods

    # Forecast horizon
    FORECAST_HORIZON = 1  # 1-month ahead

    # Target variable
    TARGET = "SP500"  # S&P 500 excess return
    BENCHMARK = "historical_mean"  # Campbell-Thompson benchmark

    # Campbell-Thompson restriction - disabled to allow signal
    USE_CT_RESTRICTION = False  # Allow model to predict negative returns

    # Walk-forward step
    EXPANDING_WINDOW = True  # Use expanding window (not rolling)
    WALK_FORWARD_STEP = 1  # Monthly rebalancing

    # Performance metrics
    METRICS = [
        "r2_oos",           # Campbell-Thompson R² OOS
        "mse",              # Mean squared error
        "mae",              # Mean absolute error
        "calmar_ratio",     # Calmar ratio
        "max_drawdown",     # Maximum drawdown
        "sharpe_ratio",     # Sharpe ratio
        "hit_ratio"         # Percentage of correct direction predictions
    ]

# =============================================================================
# Output Configuration
# =============================================================================
class OutputConfig:
    """Configuration for output and logging"""

    # Logging
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Results
    SAVE_PREDICTIONS = True
    SAVE_MODEL_STATES = True
    SAVE_FIGURES = True

    # Figure settings
    FIGURE_DPI = 150
    FIGURE_FORMAT = "png"

    # Report
    GENERATE_REPORT = True

# =============================================================================
# Citation Information
# =============================================================================
CITATIONS = {
    "fred_md": "McCracken, M. W. & Ng, S. (2016). FRED-MD: A Monthly Database "
               "for Macroeconomic Research. Journal of Business & Economic Statistics.",

    "scaled_pca": "Huang, D. et al. (2022). Scaled PCA: A New Approach to Dimension "
                  "Reduction. Management Science.",

    "campbell_thompson": "Campbell, J. Y. & Thompson, S. B. (2008). Predicting Excess "
                         "Stock Returns Out of Sample. Review of Financial Studies.",

    "elastic_net": "Zou, H. & Hastie, T. (2005). Regularization and variable selection "
                   "via the Elastic Net. Journal of the Royal Statistical Society.",

    "goyal_welch": "Goyal, A. & Welch, I. (2008). A Comprehensive Look at the Empirical "
                   "Performance of Equity Premium Prediction. Review of Financial Studies."
}

# =============================================================================
# Export all configurations
# =============================================================================
__all__ = [
    "DataConfig",
    "ModelConfig",
    "BacktestConfig",
    "OutputConfig",
    "CITATIONS",
    "PROJECT_ROOT",
    "DATA_DIR",
    "MODELS_DIR",
    "BACKTEST_DIR",
    "NOTEBOOKS_DIR"
]
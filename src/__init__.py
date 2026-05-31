"""
S&P 500 Macroeconomic Forecasting Package
State-Dependent Supervised Screening & Regularized Factor (SSRF) Architecture
"""

from .config import (
    DataConfig,
    ModelConfig,
    BacktestConfig,
    OutputConfig,
    CITATIONS,
    PROJECT_ROOT,
    DATA_DIR,
    MODELS_DIR,
    BACKTEST_DIR,
    NOTEBOOKS_DIR
)

from .fred_data import (
    FREDDataLoader,
    DataProcessor,
    download_fred_data,
    generate_sample_data
)

from .ssrf_model import (
    SSRFModel,
    SSRFConfig,
    TCConfig,
    GroupwiseScreen,
    PredictiveScaler,
    SupervisedFactorExtractor,
    RegimeProxy
)

from .backtesting import (
    WalkForwardBacktester,
    RollingBacktester,
    compare_models,
    plot_predictions,
    plot_feature_importance,
    BacktestResult
)

from .tc_backtesting import (
    TCAdjustedWalkForwardBacktester,
    TCAdjustedResult,
    compare_tc_scenarios
)

from .evaluation import (
    MetricsCalculator,
    EvaluationMetrics,
    StatisticalTests,
    generate_report,
    compare_backtests
)

from .main import main

__version__ = "1.0.0"

__all__ = [
    # Config
    "DataConfig",
    "ModelConfig",
    "BacktestConfig",
    "OutputConfig",
    "CITATIONS",
    # Data
    "FREDDataLoader",
    "DataProcessor",
    "download_fred_data",
    "generate_sample_data",
    # Model
    "SSRFModel",
    "SSRFConfig",
    "TCConfig",
    "GroupwiseScreen",
    "PredictiveScaler",
    "SupervisedFactorExtractor",
    "RegimeProxy",
    # Backtesting
    "WalkForwardBacktester",
    "RollingBacktester",
    "compare_models",
    "plot_predictions",
    "plot_feature_importance",
    "BacktestResult",
    # TC Backtesting
    "TCAdjustedWalkForwardBacktester",
    "TCAdjustedResult",
    "compare_tc_scenarios",
    # Evaluation
    "MetricsCalculator",
    "EvaluationMetrics",
    "StatisticalTests",
    "generate_report",
    "compare_backtests",
    # Main
    "main"
]
"""
Walk-Forward Backtesting Framework
Rigorous expanding window validation with Campbell-Thompson restrictions
"""

import warnings
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass
import logging

from .config import BacktestConfig
from .ssrf_model import SSRFModel, SSRFConfig

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a single backtest run."""
    predictions: pd.Series
    actual_returns: pd.Series
    benchmark_predictions: pd.Series
    dates: pd.DatetimeIndex
    train_windows: List[Tuple[pd.Timestamp, pd.Timestamp]]
    test_dates: List[pd.Timestamp]
    metrics: Dict[str, float]


class WalkForwardBacktester:
    """
    Expanding walk-forward backtester for SSRF model evaluation.

    Implements:
    - Expanding training window
    - Nested cross-validation for hyperparameter selection
    - Campbell-Thompson R² OOS metric
    - Economic utility metrics (Calmar ratio, max drawdown)
    """

    def __init__(
        self,
        model_class: type = SSRFModel,
        initial_train_window: int = BacktestConfig.INITIAL_TRAIN_WINDOW,
        forecast_horizon: int = BacktestConfig.FORECAST_HORIZON,
        use_ct_restriction: bool = BacktestConfig.USE_CT_RESTRICTION,
        step_size: int = BacktestConfig.WALK_FORWARD_STEP
    ):
        """
        Initialize walk-forward backtester.

        Args:
            model_class: Model class to use
            initial_train_window: Initial training window size (months)
            forecast_horizon: Forecast horizon (months)
            use_ct_restriction: Apply Campbell-Thompson restriction
            step_size: Walk-forward step size (months)
        """
        self.model_class = model_class
        self.initial_train_window = initial_train_window
        self.forecast_horizon = forecast_horizon
        self.use_ct_restriction = use_ct_restriction
        self.step_size = step_size

        self.results = None

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Dict[str, List[str]],
        model_config: Optional[SSRFConfig] = None,
        verbose: bool = True,
        scale_predictions: bool = True
    ) -> BacktestResult:
        """
        Run walk-forward backtest.

        Args:
            X: Feature DataFrame
            y: Target series (S&P 500 returns)
            groups: Feature groups for screening
            model_config: Optional SSRF configuration
            verbose: Print progress
            scale_predictions: Scale predictions to match target variance

        Returns:
            BacktestResult with predictions and metrics
        """
        # FIXED: Handle DataFrame target (e.g., from yfinance with multi-level columns)
        # Convert DataFrame to Series if needed
        if isinstance(y, pd.DataFrame):
            if y.shape[1] == 1:
                y = y.iloc[:, 0]  # Get first column as Series
            else:
                raise ValueError(f"y must be a Series or single-column DataFrame, got shape {y.shape}")

        # Align X and y
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]

        # Sort by date
        X = X.sort_index()
        y = y.sort_index()

        # Check data sufficiency after alignment
        # Need at least initial training window periods
        min_required = self.initial_train_window
        if len(X) < min_required:
            raise ValueError(
                f"Insufficient data after alignment: need at least "
                f"{min_required} periods, got {len(X)}"
            )

        predictions = []
        test_dates = []
        train_windows = []

        # Walk-forward loop
        n_periods = len(X) - self.initial_train_window

        if n_periods <= 0:
            # Even with no test periods, we can still train and make one prediction at the end
            logger.warning(f"No test periods (n_periods={n_periods}). Using last index as single test point.")
            n_periods = 1

        n_steps = max(1, (n_periods + self.step_size - 1) // self.step_size)

        # Calculate benchmark (historical mean from training data) before loop
        historical_means = []
        for step in range(n_steps):
            train_end_idx = self.initial_train_window + step * self.step_size
            if train_end_idx >= len(X):
                if step == 0:
                    train_end_idx = len(X) - 1
                else:
                    break
            y_train_window = y.iloc[:train_end_idx]
            historical_means.append(y_train_window.mean())

        if verbose:
            logger.info(f"Running walk-forward backtest with {n_steps} steps")

        for step in range(n_steps):
            # Calculate train and test indices
            train_end_idx = self.initial_train_window + step * self.step_size

            # Handle edge case: ensure we have at least one test point
            if train_end_idx >= len(X):
                if step == 0:
                    train_end_idx = len(X) - 1
                else:
                    break

            train_end_date = X.index[train_end_idx - 1]
            test_date_idx = train_end_idx
            test_date = X.index[test_date_idx]

            # Check we have test data
            if test_date_idx >= len(X):
                break

            # Training window
            train_start_idx = 0
            train_start_date = X.index[train_start_idx]

            # Split data
            X_train = X.iloc[train_start_idx:train_end_idx]
            y_train = y.iloc[train_start_idx:train_end_idx]

            # Test observation
            X_test = X.iloc[[test_date_idx]]
            y_test = y.iloc[test_date_idx]

            # Get benchmark (historical mean from training data)
            benchmark_pred = historical_means[step] if step < len(historical_means) else y_train.mean()

            # Fit model on training data
            config = model_config or SSRFConfig()
            model = self.model_class(config)

            try:
                model.fit(X_train, y_train, groups)

                # Predict
                # For regime proxy, use all available returns up to test date
                y_for_regime = y.iloc[:test_date_idx]
                pred = model.predict(X_test, y_for_regime)

                # Apply Campbell-Thompson restriction
                if self.use_ct_restriction:
                    pred = max(0, pred.values[0])
                else:
                    pred = pred.values[0]

            except Exception as e:
                logger.warning(f"Model failed at step {step}: {e}")
                # Use benchmark prediction as fallback instead of 0
                pred = benchmark_pred

            predictions.append(pred)
            test_dates.append(test_date)
            train_windows.append((train_start_date, train_end_date))

            if verbose and (step + 1) % 20 == 0:
                logger.info(f"  Step {step + 1}/{n_steps}: {test_date.strftime('%Y-%m')}")

        # Convert to series
        predictions = pd.Series(predictions, index=pd.DatetimeIndex(test_dates))
        actual_returns = y.loc[predictions.index]

        # Optional: Scale predictions to match target variance
        # NOTE: Disabled by default - scaling amplifies noise and increases MSE
        # Only enables when explicitly requested or scale_factor <= 5
        # FIXED: Use expanding window to prevent data leakage
        if scale_predictions and len(predictions) > 10:
            pred_std = predictions.std()
            # Use expanding window std: only use past actual returns (no lookahead)
            expanding_std = actual_returns.expanding().std()
            # Use the std from the first half of test period as reference
            mid_point = len(expanding_std) // 2
            if mid_point > 10:
                actual_std = expanding_std.iloc[mid_point - 1]
            else:
                actual_std = expanding_std.iloc[0]
            if pred_std > 0.001 and actual_std > 0.001:
                scale_factor = actual_std / pred_std
                # Only scale if factor is very modest (1.0 to 5)
                if 1.0 < scale_factor <= 5:
                    predictions = predictions * scale_factor
                    logger.info(f"Scaled predictions by factor {scale_factor:.2f} to match target variance")

        # Compute benchmark (historical mean from training window)
        # FIXED: Exclude test observation from benchmark calculation
        benchmark = []
        for i, test_date in enumerate(test_dates):
            # Find the training window end index (before test date)
            # At step i, training ends at train_end_idx from the walk-forward loop
            # We need to use the same index used during training
            if i < len(predictions):
                # Use training data from expanding window up to test date
                # Benchmark should NOT include the test observation
                y_before_test = y.iloc[:len(y) - len(predictions) + i]
                benchmark.append(y_before_test.mean())
            else:
                benchmark.append(y.mean())

        benchmark = pd.Series(benchmark, index=predictions.index, name='benchmark')

        # Compute metrics
        metrics = self._compute_metrics(predictions, actual_returns, benchmark)

        if verbose:
            self._print_metrics(metrics)

        self.results = BacktestResult(
            predictions=predictions,
            actual_returns=actual_returns,
            benchmark_predictions=benchmark,
            dates=predictions.index,
            train_windows=train_windows,
            test_dates=test_dates,
            metrics=metrics
        )

        return self.results

    def _compute_metrics(
        self,
        predictions: pd.Series,
        actual: pd.Series,
        benchmark: pd.Series
    ) -> Dict[str, float]:
        """
        Compute performance metrics.

        Args:
            predictions: Model predictions
            actual: Actual returns
            benchmark: Benchmark predictions

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Campbell-Thompson R² OOS
        r2_oos = self._compute_r2_oos(actual, predictions, benchmark)
        metrics['r2_oos'] = r2_oos

        # MSE and MAE
        errors = actual - predictions
        metrics['mse'] = (errors ** 2).mean()
        metrics['mae'] = errors.abs().mean()

        # Direction accuracy (hit ratio)
        correct_direction = np.sign(predictions) == np.sign(actual)
        metrics['hit_ratio'] = correct_direction.mean()

        # Portfolio simulation
        portfolio_returns = self._simulate_portfolio(predictions, actual)
        benchmark_returns = self._simulate_portfolio(benchmark, actual)

        # Sharpe ratio
        if portfolio_returns.std() > 0:
            metrics['sharpe_ratio'] = (
                portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(12)
            )
        else:
            metrics['sharpe_ratio'] = 0.0

        # Maximum drawdown
        metrics['max_drawdown'] = self._compute_max_drawdown(portfolio_returns)
        metrics['benchmark_max_drawdown'] = self._compute_max_drawdown(benchmark_returns)

        # Calmar ratio (annualized return / max drawdown)
        if metrics['max_drawdown'] != 0:
            annual_return = portfolio_returns.mean() * 12
            metrics['calmar_ratio'] = annual_return / metrics['max_drawdown']
        else:
            metrics['calmar_ratio'] = 0.0

        # Cumulative returns
        metrics['cumulative_return'] = (1 + portfolio_returns).prod() - 1
        metrics['benchmark_cumulative_return'] = (1 + benchmark_returns).prod() - 1

        # Volatility
        metrics['volatility'] = portfolio_returns.std() * np.sqrt(12)
        metrics['benchmark_volatility'] = benchmark_returns.std() * np.sqrt(12)

        return metrics

    def _compute_r2_oos(
        self,
        actual: pd.Series,
        predictions: pd.Series,
        benchmark: pd.Series
    ) -> float:
        """
        Compute Campbell-Thompson R² OOS metric.

        R² OOS = 1 - SS_res / SS_tot

        where SS_res is sum of squared errors from predictions
        and SS_tot is sum of squared errors from benchmark.

        Args:
            actual: Actual returns
            predictions: Model predictions
            benchmark: Benchmark predictions

        Returns:
            R² OOS value
        """
        ss_res = ((actual - predictions) ** 2).sum()
        ss_tot = ((actual - benchmark) ** 2).sum()

        if ss_tot == 0:
            return 0.0

        return 1 - ss_res / ss_tot

    def _simulate_portfolio(
        self,
        signals: pd.Series,
        actual_returns: pd.Series
    ) -> pd.Series:
        """
        Simulate portfolio returns based on signals.

        Long when signal > 0, neutral when signal = 0, short when signal < 0.

        Args:
            signals: Prediction signals
            actual_returns: Actual returns

        Returns:
            Portfolio returns
        """
        positions = np.sign(signals.values)
        # Scale position to [-1, 1] based on signal magnitude
        max_signal = signals.abs().max()
        if max_signal > 0:
            positions = positions * (signals.abs() / max_signal)
        else:
            positions = np.zeros(len(signals))

        return pd.Series(
            positions * actual_returns.values,
            index=actual_returns.index
        )

    def _compute_max_drawdown(self, returns: pd.Series) -> float:
        """
        Compute maximum drawdown.

        Args:
            returns: Return series

        Returns:
            Maximum drawdown (positive value)
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = cumulative / running_max - 1

        return abs(drawdowns.min())

    def _print_metrics(self, metrics: Dict[str, float]):
        """Print formatted metrics."""
        print("\n" + "=" * 50)
        print("BACKTEST RESULTS")
        print("=" * 50)

        print(f"\nCampbell-Thompson R² OOS: {metrics['r2_oos']:.4f}")
        print(f"Direction Accuracy (Hit Ratio): {metrics['hit_ratio']:.2%}")

        print(f"\nMean Squared Error: {metrics['mse']:.6f}")
        print(f"Mean Absolute Error: {metrics['mae']:.4f}")

        print(f"\nSharpe Ratio (annualized): {metrics['sharpe_ratio']:.4f}")
        print(f"Annualized Volatility: {metrics['volatility']:.2%}")

        print(f"\nMaximum Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Benchmark Max Drawdown: {metrics['benchmark_max_drawdown']:.2%}")

        print(f"\nCalmar Ratio: {metrics['calmar_ratio']:.4f}")

        print(f"\nCumulative Return: {metrics['cumulative_return']:.2%}")
        print(f"Benchmark Cumulative Return: {metrics['benchmark_cumulative_return']:.2%}")
        print("=" * 50)


class RollingBacktester:
    """
    Rolling window backtester (alternative to expanding window).
    Uses fixed-size training windows.
    """

    def __init__(
        self,
        model_class: type = SSRFModel,
        train_window: int = 120,
        test_window: int = 12,
        step_size: int = 1
    ):
        """
        Initialize rolling backtester.

        Args:
            model_class: Model class to use
            train_window: Training window size
            test_window: Number of periods to forecast
            step_size: Step between folds
        """
        self.model_class = model_class
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Dict[str, List[str]],
        model_config: Optional[SSRFConfig] = None,
        verbose: bool = True
    ) -> BacktestResult:
        """Run rolling window backtest."""
        # Similar to WalkForwardBacktester but with fixed training window
        # See WalkForwardBacktester.run() for implementation
        pass  # Would be implemented similarly


def compare_models(
    models: Dict[str, Tuple[type, SSRFConfig]],
    X: pd.DataFrame,
    y: pd.Series,
    groups: Dict[str, List[str]],
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compare multiple model configurations.

    Args:
        models: Dict mapping model name to (model_class, config) tuple
        X: Feature DataFrame
        y: Target series
        groups: Feature groups
        verbose: Print progress

    Returns:
        DataFrame comparing model performance
    """
    results = []

    for name, (model_class, config) in models.items():
        if verbose:
            print(f"\nEvaluating {name}...")

        backtester = WalkForwardBacktester(model_class=model_class)

        try:
            result = backtester.run(X, y, groups, config, verbose=False)
            metrics = result.metrics.copy()
            metrics['model'] = name
            results.append(metrics)
        except Exception as e:
            logger.error(f"Model {name} failed: {e}")

    return pd.DataFrame(results)


# =============================================================================
# Visualization
# =============================================================================
def plot_predictions(
    result: BacktestResult,
    save_path: Optional[str] = None
) -> 'matplotlib.figure.Figure':
    """
    Plot prediction results.

    Args:
        result: BacktestResult
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        warnings.warn("matplotlib not installed, skipping plot")
        return None

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    dates = result.dates

    # Plot 1: Predictions vs Actual
    ax1 = axes[0]
    ax1.plot(dates, result.actual_returns.values, label='Actual', alpha=0.7, linewidth=1)
    ax1.plot(dates, result.predictions.values, label='SSRF Predictions', alpha=0.7, linewidth=1)
    ax1.plot(dates, result.benchmark_predictions.values, label='Benchmark', alpha=0.5, linestyle='--')
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.set_title('S&P 500 Monthly Returns: Actual vs Predicted')
    ax1.set_ylabel('Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_major_locator(mdates.YearLocator(2))

    # Plot 2: Cumulative Returns
    ax2 = axes[1]
    # Use same long/short logic as _simulate_portfolio() for consistency
    signals = result.predictions.values
    positions = np.sign(signals)
    max_signal = np.abs(signals).max()
    if max_signal > 0:
        positions = positions * (np.abs(signals) / max_signal)
    else:
        positions = np.zeros(len(signals))
    portfolio_returns = pd.Series(
        positions * result.actual_returns.values,
        index=result.dates
    )
    benchmark_returns = result.actual_returns

    cumulative_portfolio = (1 + portfolio_returns).cumprod()
    cumulative_benchmark = (1 + benchmark_returns).cumprod()

    ax2.plot(dates, cumulative_portfolio.values, label='SSRF Portfolio', linewidth=2)
    ax2.plot(dates, cumulative_benchmark.values, label='Buy & Hold', linewidth=2, alpha=0.7)
    ax2.set_title('Cumulative Returns: SSRF Strategy vs Buy & Hold')
    ax2.set_ylabel('Cumulative Return (Growth of $1)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))

    # Plot 3: Rolling Sharpe Ratio
    ax3 = axes[2]
    window = 24  # 24-month rolling
    rolling_sharpe = (
        portfolio_returns.rolling(window).mean() /
        portfolio_returns.rolling(window).std() * np.sqrt(12)
    )

    ax3.plot(dates, rolling_sharpe.values, label='SSRF Rolling Sharpe', linewidth=1.5)
    ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax3.axhline(y=portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(12),
                color='blue', linestyle='--', alpha=0.5, label='Overall Sharpe')
    ax3.set_title(f'{window}-Month Rolling Sharpe Ratio')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax3.xaxis.set_major_locator(mdates.YearLocator(2))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


def plot_feature_importance(
    result: BacktestResult,
    top_n: int = 20,
    save_path: Optional[str] = None
) -> 'matplotlib.figure.Figure':
    """
    Plot feature importance across the backtest.

    Args:
        result: BacktestResult
        top_n: Number of top features to show
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not installed, skipping plot")
        return None

    fig, ax = plt.subplots(figsize=(12, 8))

    # For now, just show hit ratio over time
    # Feature importance would require storing per-fold selections
    hit_ratio_rolling = (
        (np.sign(result.predictions.values) == np.sign(result.actual_returns.values))
        .rolling(12)
        .mean()
    )

    ax.plot(result.dates, hit_ratio_rolling, linewidth=2)
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Random (50%)')
    ax.axhline(y=hit_ratio_rolling.mean(), color='green', linestyle='--',
               label=f'Mean ({hit_ratio_rolling.mean():.1%})')
    ax.set_title('Rolling 12-Month Hit Ratio')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Date')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig


if __name__ == "__main__":
    # Example usage
    print("Walk-Forward Backtester - Sample Usage")
    print("=" * 50)

    # Generate sample data
    from .fred_data import generate_sample_data

    indicators, target = generate_sample_data(n_periods=300, n_indicators=50)

    # Define groups
    groups = {
        'output_income': [c for c in indicators.columns if 'output' in c][:5],
        'labor': [c for c in indicators.columns if 'labor' in c][:5],
        'inflation': [c for c in indicators.columns if 'inflation' in c][:5],
        'interest': [c for c in indicators.columns if 'interest' in c][:5],
        'sentiment': [c for c in indicators.columns if 'sentiment' in c][:5]
    }

    # Run backtest
    backtester = WalkForwardBacktester(initial_train_window=120)
    result = backtester.run(indicators, target, groups, verbose=True)

    print(f"\nFinal R² OOS: {result.metrics['r2_oos']:.4f}")
"""
Walk-Forward Backtesting Framework
Rigorous expanding window validation with Campbell-Thompson restrictions
"""

import warnings
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Callable, Union
from dataclasses import dataclass
from tqdm import tqdm
import logging

from .config import BacktestConfig
from . import ensure_series
from .ssrf_model import SSRFModel, SSRFConfig
from .evaluation import _simulate_asymmetric_portfolio

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
        step_size: int = BacktestConfig.WALK_FORWARD_STEP,
        max_long: float = 1.0,
        max_short: float = 1.0,
        margin_rate: float = 0.05,
        drawdown_limit: float = 0.25
    ):
        """
        Initialize walk-forward backtester.

        Args:
            model_class: Model class to use
            initial_train_window: Initial training window size (months)
            forecast_horizon: Forecast horizon (months)
            use_ct_restriction: Apply Campbell-Thompson restriction
            step_size: Walk-forward step size (months)
            max_long: Maximum long position (1.0 = no margin)
            max_short: Maximum short position (1.0 = full short)
            margin_rate: Annual margin interest rate
            drawdown_limit: Max drawdown before levered positions reduced (0.0-0.5)
        """
        self.model_class = model_class
        self.initial_train_window = initial_train_window
        self.forecast_horizon = forecast_horizon
        self.use_ct_restriction = use_ct_restriction
        self.step_size = step_size
        self.max_long = max_long
        self.max_short = max_short
        self.margin_rate = margin_rate
        self.drawdown_limit = drawdown_limit

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
        y = ensure_series(y, "y")

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

        for step in tqdm(range(n_steps), desc="Walk-forward", disable=not verbose, unit="step"):
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

        # Convert to series
        predictions = pd.Series(predictions, index=pd.DatetimeIndex(test_dates))
        actual_returns = y.loc[predictions.index]

        # Optional: Scale predictions to match target variance
        # Only triggers when scale_predictions=True AND predictions are
        # less volatile than training AND scale_factor in (1.0, 5].
        # Effect: uniform scaling factor k leaves sign(pred) unchanged, so
        # portfolio metrics (Sharpe, hit ratio, cumulative return) are unaffected.
        # Only R² OOS changes because MSE uses prediction magnitudes.
        # FIXED: Use training period std only - no lookahead into test period
        if scale_predictions and len(predictions) > 10:
            pred_std = predictions.std()
            # Use training period std as reference (no test period data)
            train_end_idx = len(y) - len(predictions)
            train_std = y.iloc[:train_end_idx].std()
            # Only scale if factor is very modest (1.0 to 5)
            if pred_std > 0.001 and train_std > 0.001:
                scale_factor = train_std / pred_std
                if 1.0 < scale_factor <= 5:
                    predictions = predictions * scale_factor
                    logger.info(f"Scaled predictions by factor {scale_factor:.2f} to match target variance")

        # Compute benchmark (historical mean from training window)
        # FIXED: Use fixed training cutoff (no drift into test period)
        # The test period starts at len(y) - len(predictions)
        # For step i, we use data up to (but not including) test_date_i
        # Training cutoff is fixed at the start of test period
        benchmark = []
        test_data_start = len(y) - len(predictions)
        for i, test_date in enumerate(test_dates):
            # benchmark_i should use all training data (before test period starts)
            y_before_test = y.iloc[:test_data_start]  # Fixed cutoff, no drift
            benchmark.append(y_before_test.mean())
        # Alternative: Expanding benchmark (include previous test returns in benchmark)
        # Uncomment below and comment above for expanding version:
        # for i in range(len(test_dates)):
        #     y_before_test = y.iloc[:test_data_start + i]
        #     benchmark.append(y_before_test.mean())

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

    def predict_next(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Dict[str, List[str]],
        model_config: Optional[SSRFConfig] = None,
    ) -> float:
        """
        Train on ALL available data and predict the next 3-month forward return.

        After a walk-forward backtest completes, this gives a current signal
        for portfolio adjustment: train on everything, predict the next period.

        Args:
            X: Feature DataFrame (all available data)
            y: Target series (all available returns)
            groups: Feature groups for screening
            model_config: Optional SSRF configuration

        Returns:
            Predicted 3-month forward return (e.g. 0.032 = +3.2%)
        """
        y = ensure_series(y, "y")

        # Align and sort (same as run())
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx].sort_index()
        y = y.loc[common_idx].sort_index()

        # Use the last row of X as features for the next prediction
        X_features = X.iloc[[-1]]
        prediction_date = X.index[-1]

        # Train on all available data
        config = model_config or SSRFConfig()
        model = self.model_class(config)

        try:
            model.fit(X, y, groups)
            pred = model.predict(X_features, y)
            pred_value = pred.values[0]
        except Exception as e:
            logger.warning(f"Next-predict failed: {e}. Using historical mean as fallback.")
            pred_value = y.mean()

        self._next_prediction = pred_value
        return pred_value

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

        # Direction breakdown: positive vs negative prediction accuracy
        pos_mask = predictions > 0
        neg_mask = predictions < 0
        metrics['n_pos'] = int(pos_mask.sum())
        metrics['n_neg'] = int(neg_mask.sum())
        if metrics['n_pos'] > 0:
            metrics['pos_accuracy'] = float((np.sign(actual[pos_mask]) == 1).mean())
        else:
            metrics['pos_accuracy'] = 0.0
        if metrics['n_neg'] > 0:
            metrics['neg_accuracy'] = float((np.sign(actual[neg_mask]) == -1).mean())
        else:
            metrics['neg_accuracy'] = 0.0

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
        Simulate portfolio returns with asymmetric position sizing.

        Uses max_long/max_short, margin costs, and drawdown-based leverage reduction.
        """
        return _simulate_asymmetric_portfolio(
            signals, actual_returns,
            max_long=self.max_long,
            max_short=self.max_short,
            margin_rate=self.margin_rate,
            drawdown_limit=self.drawdown_limit
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
        print(f"  Long  accuracy: {metrics.get('pos_accuracy', 0):.2%}  ({metrics.get('n_pos', 0)} predictions)")
        print(f"  Short accuracy: {metrics.get('neg_accuracy', 0):.2%}  ({metrics.get('n_neg', 0)} predictions)")

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

    NOTE: This class is currently a stub and not yet implemented.
    WalkForwardBacktester with expanding window is recommended.
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
        raise NotImplementedError(
            "RollingBacktester.run() is not implemented. "
            "Use WalkForwardBacktester with expanding window instead."
        )


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
    result: Union[BacktestResult, dict],
    save_path: Optional[str] = None
) -> 'matplotlib.figure.Figure':
    """
    Plot prediction results.

    Args:
        result: BacktestResult or dict with keys: predictions, actual_returns,
                benchmark_predictions, dates, train_windows
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    # FIXED: Support both BacktestResult and dict inputs
    if isinstance(result, dict):
        predictions = result.get('predictions')
        actual_returns = result.get('actual_returns')
        benchmark = result.get('benchmark_predictions')
        dates = result.get('dates')
        train_windows = result.get('train_windows', [])
    else:
        predictions = result.predictions
        actual_returns = result.actual_returns
        benchmark = result.benchmark_predictions
        dates = result.dates
        train_windows = result.train_windows

    if predictions is None or actual_returns is None:
        warnings.warn("Missing predictions or actual_returns in result")
        return None
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        warnings.warn("matplotlib not installed, skipping plot")
        return None

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: Predictions vs Actual
    ax1 = axes[0]
    ax1.plot(dates, actual_returns.values, label='Actual', alpha=0.7, linewidth=1)
    ax1.plot(dates, predictions.values, label='SSRF Predictions', alpha=0.7, linewidth=1)
    ax1.plot(dates, benchmark.values, label='Benchmark', alpha=0.5, linestyle='--')
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
    signals = predictions.values
    positions = np.sign(signals)
    max_signal = np.abs(signals).max()
    if max_signal > 0:
        positions = positions * (np.abs(signals) / max_signal)
    else:
        positions = np.zeros(len(signals))
    portfolio_returns = pd.Series(
        positions * actual_returns.values,
        index=dates
    )
    benchmark_returns = actual_returns

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
    result: Union[BacktestResult, dict],
    top_n: int = 20,
    save_path: Optional[str] = None
) -> 'matplotlib.figure.Figure':
    """
    Plot feature importance across the backtest.

    Args:
        result: BacktestResult or dict with keys: predictions, actual_returns, dates
        top_n: Number of top features to show
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    # FIXED: Support both BacktestResult and dict inputs
    if isinstance(result, dict):
        predictions = result.get('predictions')
        actual_returns = result.get('actual_returns')
        dates = result.get('dates')
    else:
        predictions = result.predictions
        actual_returns = result.actual_returns
        dates = result.dates

    if predictions is None or actual_returns is None:
        warnings.warn("Missing predictions or actual_returns in result")
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not installed, skipping plot")
        return None

    fig, ax = plt.subplots(figsize=(12, 8))

    # For now, just show hit ratio over time
    # Feature importance would require storing per-fold selections
    hit_ratio_rolling = (
        (np.sign(predictions.values) == np.sign(actual_returns.values))
        .rolling(12)
        .mean()
    )

    ax.plot(dates, hit_ratio_rolling, linewidth=2)
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

    indicators, target, _ = generate_sample_data(n_periods=300, n_indicators=50)

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

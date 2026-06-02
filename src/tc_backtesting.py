"""
TC-Adjusted Walk-Forward Backtester
Wraps the standard WalkForwardBacktester with transaction cost modeling
"""

import warnings
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Callable
from dataclasses import dataclass, field
import logging

from .backtesting import WalkForwardBacktester, BacktestResult
from . import ensure_series
from .ssrf_model import SSRFModel, SSRFConfig, TCConfig
from .evaluation import _simulate_asymmetric_portfolio

logger = logging.getLogger(__name__)


@dataclass
class TCAdjustedResult:
    """Extended backtest result with transaction cost metrics."""
    predictions: pd.Series
    tc_adjusted_predictions: pd.Series
    actual_returns: pd.Series
    benchmark_predictions: pd.Series
    turnover: pd.Series
    tc_costs: pd.Series
    net_returns: pd.Series
    gross_returns: pd.Series
    dates: pd.DatetimeIndex
    train_windows: List[Tuple[pd.Timestamp, pd.Timestamp]]
    test_dates: List[pd.Timestamp]
    metrics: Dict[str, float]
    tc_metrics: Dict[str, float]


class TCAdjustedWalkForwardBacktester:
    """
    Walk-Forward Backtester with integrated Transaction Cost Modeling.

    Extends the standard WalkForwardBacktester with:
    - TC-aware signal adjustment
    - Per-period TC cost tracking
    - Net returns calculation
    - TC-adjusted performance metrics
    """

    def __init__(
        self,
        model_class: type = SSRFModel,
        initial_train_window: int = 60,
        forecast_horizon: int = 1,
        use_ct_restriction: bool = False,
        step_size: int = 1,
        # TC parameters
        tc_rate_bps: float = 25.0,
        account_tier: str = "standard",
        expected_turnover: float = 0.15,
        position_threshold: float = 0.02,
        # Margin / asymmetric position parameters
        max_long: float = 1.0,
        max_short: float = 1.0,
        margin_rate: float = 0.05,
        drawdown_limit: float = 0.25,
    ):
        """
        Initialize TC-adjusted backtester.

        Args:
            model_class: Model class to use
            initial_train_window: Initial training window size (months)
            forecast_horizon: Forecast horizon (months)
            use_ct_restriction: Apply Campbell-Thompson restriction
            step_size: Walk-forward step size (months)
            tc_rate_bps: Transaction cost rate in basis points
            account_tier: Account tier (micro, standard, professional, institutional)
            expected_turnover: Expected portfolio turnover rate
            position_threshold: Minimum signal to trigger trade
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

        # TC parameters
        self.tc_rate_bps = tc_rate_bps
        self.account_tier = account_tier
        self.expected_turnover = expected_turnover
        self.position_threshold = position_threshold

        # Margin / asymmetric position parameters
        self.max_long = max_long
        self.max_short = max_short
        self.margin_rate = margin_rate
        self.drawdown_limit = drawdown_limit

        # Conviction filtering parameters
        self.min_conviction_threshold: float = 0.0
        self.conviction_filter_enabled: bool = False

        # Get effective TC rate from tier
        self.effective_tc_rate = TCConfig.get_tc_rate(account_tier)

        self.results = None

    def run(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Dict[str, List[str]],
        model_config: Optional[SSRFConfig] = None,
        verbose: bool = True,
        include_tc_in_prediction: bool = True,
    ) -> TCAdjustedResult:
        """
        Run TC-adjusted walk-forward backtest.

        Args:
            X: Feature DataFrame
            y: Target series (returns)
            groups: Feature groups for screening
            model_config: Optional SSRF configuration
            verbose: Print progress
            include_tc_in_prediction: Adjust predictions for TC

        Returns:
            TCAdjustedResult with full TC metrics
        """
        y = ensure_series(y, "y")

        # Create base SSRF config with TC settings
        if model_config is None:
            model_config = SSRFConfig()

        # Override TC settings in config
        model_config.include_tc = include_tc_in_prediction
        model_config.tc_rate_bps = self.effective_tc_rate
        model_config.expected_turnover = self.expected_turnover
        model_config.account_tier = self.account_tier

        if verbose:
            logger.info(f"Running TC-adjusted backtest with {self.effective_tc_rate:.1f} bps TC rate")

        # Create standard backtester (pass margin/asymmetric params through)
        base_backtester = WalkForwardBacktester(
            model_class=self.model_class,
            initial_train_window=self.initial_train_window,
            forecast_horizon=self.forecast_horizon,
            use_ct_restriction=self.use_ct_restriction,
            step_size=self.step_size,
            max_long=self.max_long,
            max_short=self.max_short,
            margin_rate=self.margin_rate,
            drawdown_limit=self.drawdown_limit
        )

        # Run base backtest
        base_result = base_backtester.run(
            X, y, groups, model_config, verbose=verbose
        )

        # Get raw predictions
        raw_predictions = base_result.predictions

        # Apply conviction filtering if enabled
        if self.conviction_filter_enabled and self.min_conviction_threshold > 0:
            filtered_predictions = self._apply_conviction_filter(
                raw_predictions, base_result.actual_returns
            )
            if verbose:
                n_filtered = (raw_predictions != filtered_predictions).sum()
                logger.info(f"Conviction filter: {n_filtered} periods filtered out")
            raw_predictions = filtered_predictions

        # Apply TC adjustment to predictions
        tc_adjusted_predictions = self._adjust_predictions_for_tc(raw_predictions)

        # Calculate turnover
        turnover = self._calculate_turnover(raw_predictions)

        # Calculate TC costs per period
        tc_costs = self._calculate_tc_costs(turnover)

        # Calculate gross and net returns
        gross_returns = self._calculate_gross_returns(
            raw_predictions, base_result.actual_returns
        )
        net_returns = self._calculate_net_returns(gross_returns, tc_costs)

        # Compute metrics
        metrics = self._compute_metrics(
            raw_predictions, tc_adjusted_predictions, net_returns, gross_returns,
            base_result.actual_returns, base_result.benchmark_predictions, turnover, tc_costs
        )

        # Create TC metrics summary
        tc_metrics = self._compute_tc_metrics(turnover, tc_costs, net_returns)

        if verbose:
            self._print_tc_results(metrics, tc_metrics)

        return TCAdjustedResult(
            predictions=raw_predictions,
            tc_adjusted_predictions=tc_adjusted_predictions,
            actual_returns=base_result.actual_returns,
            benchmark_predictions=base_result.benchmark_predictions,
            turnover=turnover,
            tc_costs=tc_costs,
            net_returns=net_returns,
            gross_returns=gross_returns,
            dates=base_result.dates,
            train_windows=base_result.train_windows,
            test_dates=base_result.test_dates,
            metrics=metrics,
            tc_metrics=tc_metrics
        )

    def _adjust_predictions_for_tc(self, predictions: pd.Series) -> pd.Series:
        """
        Adjust predictions for expected transaction costs.

        Transaction costs reduce net returns. All predictions are scaled down
        by the expected TC cost, proportional to turnover.

        Net prediction = signal * (1 - expected_tc_cost)
        where expected_tc_cost = turnover * tc_rate / 10000
        """
        # FIXED: Simple, correct TC adjustment - always reduces, never amplifies
        # TC cost = turnover * tc_rate (in decimal form)
        tc_cost = self.expected_turnover * self.effective_tc_rate / 10000

        # Apply fixed reduction to all predictions
        adjustment_factor = 1.0 - tc_cost

        adjusted = predictions * adjustment_factor

        logger.info(f"TC adjustment: predictions scaled by {adjustment_factor:.4f} (TC cost: {tc_cost*100:.4f}%)")

        return adjusted

    def _apply_conviction_filter(
        self,
        predictions: pd.Series,
        actual_returns: pd.Series = None
    ) -> pd.Series:
        """
        Apply high-conviction filtering to predictions.

        Only trades when signal exceeds minimum conviction threshold.
        Conviction is computed as signal strength relative to historical volatility.

        Args:
            predictions: Raw predictions
            actual_returns: Actual returns for conviction calibration (optional)

        Returns:
            Filtered predictions (zeroed out where conviction is low)
        """
        # Compute conviction as signal strength relative to past signals only
        # FIXED: Use expanding std to prevent look-ahead bias.
        # Original used predictions.std() over the full test period,
        # leaking future signal magnitude info into early conviction scores.
        # At time t, only signals[0..t-1] are available for computing std.
        expanding_std = predictions.abs().expanding().std().shift(1)
        # For the first prediction, use its absolute value as the std estimate
        expanding_std.iloc[0] = max(predictions.abs().iloc[0], 1e-8)
        # Fill any NaN from insufficient samples expanding forward
        expanding_std = expanding_std.fillna(method='ffill')

        conviction = predictions.abs() / expanding_std.clip(lower=1e-8)

        # Create filtered predictions
        filtered = predictions.copy()

        # Zero out predictions below conviction threshold
        below_threshold = conviction < self.min_conviction_threshold
        filtered[below_threshold] = 0

        if len(predictions) > 0:
            n_active = (~below_threshold).sum()
            pct_active = n_active / len(predictions) * 100
            logger.info(
                f"Conviction filter: {n_active}/{len(predictions)} ({pct_active:.1f}%) "
                f"periods active with threshold={self.min_conviction_threshold:.2f}"
            )

        return filtered

    def _calculate_turnover(self, predictions: pd.Series) -> pd.Series:
        """
        Calculate portfolio turnover from prediction signals.

        Turnover = |new_position - old_position| / (|old_position| + |new_position|)
        """
        # Convert signals to positions (long/short/neutral)
        positions = np.sign(predictions.values)
        positions[abs(predictions) < self.position_threshold] = 0

        # Calculate position changes
        position_changes = np.abs(np.diff(positions, prepend=positions[0]))

        # Turnover = average of old and new position size
        turnover = pd.Series(position_changes / 2, index=predictions.index)

        # Alternative: simpler turnover calculation
        # turnover = pd.Series(np.abs(np.diff(positions, prepend=0)), index=predictions.index)

        return turnover.fillna(0)

    def _calculate_tc_costs(self, turnover: pd.Series) -> pd.Series:
        """
        Calculate transaction cost per period.

        TC cost = turnover * tc_rate
        """
        tc_cost = turnover * (self.effective_tc_rate / 10000)
        return pd.Series(tc_cost, index=turnover.index)

    def _calculate_gross_returns(
        self,
        predictions: pd.Series,
        actual_returns: pd.Series
    ) -> pd.Series:
        """
        Calculate gross returns (before TC).
        Uses same asymmetric position sizing as _simulate_portfolio().
        """
        return _simulate_asymmetric_portfolio(
            predictions, actual_returns,
            max_long=self.max_long,
            max_short=self.max_short,
            margin_rate=self.margin_rate,
            drawdown_limit=self.drawdown_limit
        )

    def _calculate_net_returns(
        self,
        gross_returns: pd.Series,
        tc_costs: pd.Series
    ) -> pd.Series:
        """
        Calculate net returns (after TC).
        """
        return gross_returns - tc_costs

    def _compute_metrics(
        self,
        predictions: pd.Series,
        tc_adjusted: pd.Series,
        net_returns: pd.Series,
        gross_returns: pd.Series,
        actual_returns: pd.Series,
        benchmark: pd.Series,
        turnover: pd.Series,
        tc_costs: pd.Series
    ) -> Dict[str, float]:
        """
        Compute comprehensive metrics including TC-adjusted metrics.
        """
        metrics = {}

        # === RAW PREDICTION METRICS ===
        errors = actual_returns - predictions
        metrics['mse_raw'] = (errors ** 2).mean()
        metrics['mae_raw'] = errors.abs().mean()

        # Direction accuracy
        correct_direction = np.sign(predictions) == np.sign(actual_returns)
        metrics['hit_ratio_raw'] = correct_direction.mean()

        # Sharpe (gross)
        if gross_returns.std() > 0:
            metrics['sharpe_gross'] = gross_returns.mean() / gross_returns.std() * np.sqrt(12)
        else:
            metrics['sharpe_gross'] = 0.0

        # === NET RETURN METRICS ===
        if net_returns.std() > 0:
            metrics['sharpe_net'] = net_returns.mean() / net_returns.std() * np.sqrt(12)
        else:
            metrics['sharpe_net'] = 0.0

        # Cumulative returns
        metrics['cumulative_gross'] = (1 + gross_returns).prod() - 1
        metrics['cumulative_net'] = (1 + net_returns).prod() - 1

        # Net vs gross (TC drag)
        metrics['tc_drag'] = metrics['cumulative_gross'] - metrics['cumulative_net']

        # === TC-ADJUSTED METRICS ===
        if tc_adjusted.std() > 0:
            # Compare TC-adjusted to benchmark
            tc_errors = actual_returns - tc_adjusted
            ss_tot = ((actual_returns - benchmark) ** 2).sum()
            ss_res = ((actual_returns - tc_adjusted) ** 2).sum()
            metrics['r2_oos_tc_adjusted'] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            # Direction accuracy
            correct_tc = np.sign(tc_adjusted) == np.sign(actual_returns)
            metrics['hit_ratio_tc_adjusted'] = correct_tc.mean()
        else:
            metrics['r2_oos_tc_adjusted'] = 0.0
            metrics['hit_ratio_tc_adjusted'] = 0.0

        # === BENCHMARK COMPARISON ===
        spx_cumulative = (1 + actual_returns).prod() - 1
        metrics['spx_cumulative'] = spx_cumulative
        metrics['vs_spx_gross'] = metrics['cumulative_gross'] - spx_cumulative
        metrics['vs_spx_net'] = metrics['cumulative_net'] - spx_cumulative

        # Max drawdown (net)
        cumulative = (1 + net_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = cumulative / running_max - 1
        metrics['max_drawdown_net'] = abs(drawdowns.min())

        # Volatility
        metrics['volatility_gross'] = gross_returns.std() * np.sqrt(12)
        metrics['volatility_net'] = net_returns.std() * np.sqrt(12)

        # Annual returns
        n_years = len(net_returns) / 12
        metrics['ann_return_gross'] = (1 + metrics['cumulative_gross']) ** (12 / len(net_returns)) - 1
        metrics['ann_return_net'] = (1 + metrics['cumulative_net']) ** (12 / len(net_returns)) - 1

        return metrics

    def _compute_tc_metrics(
        self,
        turnover: pd.Series,
        tc_costs: pd.Series,
        net_returns: pd.Series
    ) -> Dict[str, float]:
        """
        Compute TC-specific metrics.
        """
        return {
            'avg_turnover': turnover.mean(),
            'total_turnover': turnover.sum(),
            'avg_tc_cost_per_period': tc_costs.mean(),
            'total_tc_cost': tc_costs.sum(),
            'total_tc_cost_pct': tc_costs.sum() * 100,
            'effective_tc_rate_bps': self.effective_tc_rate,
            'account_tier': self.account_tier,
            'n_trades': int((turnover > 0).sum()),
            'tc_cost_per_trade_bps': tc_costs[turnover > 0].mean() * 10000 if (turnover > 0).any() else 0,
        }

    def _print_tc_results(self, metrics: Dict[str, float], tc_metrics: Dict[str, float]):
        """Print TC-adjusted results."""
        print("\n" + "=" * 60)
        print("TRANSACTION COST-ADJUSTED BACKTEST RESULTS")
        print("=" * 60)

        print(f"\nTC Configuration:")
        print(f"  Account Tier: {tc_metrics['account_tier']}")
        print(f"  Effective TC Rate: {tc_metrics['effective_tc_rate_bps']:.1f} bps")
        print(f"  Average Turnover: {tc_metrics['avg_turnover']*100:.1f}%")

        print(f"\nTrading Activity:")
        print(f"  Number of Trades: {tc_metrics['n_trades']}")
        print(f"  Total Turnover: {tc_metrics['total_turnover']*100:.1f}%")
        print(f"  Avg TC per Trade: {tc_metrics['tc_cost_per_trade_bps']:.1f} bps")

        print(f"\nPerformance:")
        print(f"  Gross Annual Return: {metrics['ann_return_gross']*100:.2f}%")
        print(f"  Net Annual Return: {metrics['ann_return_net']*100:.2f}%")
        print(f"  TC Drag: {metrics['tc_drag']*100:.2f}%")
        print(f"  Gross Sharpe: {metrics['sharpe_gross']:.3f}")
        print(f"  Net Sharpe: {metrics['sharpe_net']:.3f}")

        print(f"\nBenchmark Comparison:")
        print(f"  S&P 500 Cumulative: {metrics['spx_cumulative']*100:.2f}%")
        print(f"  vs S&P 500 (Gross): {metrics['vs_spx_gross']*100:.2f}%")
        print(f"  vs S&P 500 (Net): {metrics['vs_spx_net']*100:.2f}%")

        print("=" * 60)


def compare_tc_scenarios(
    X: pd.DataFrame,
    y: pd.Series,
    groups: Dict[str, List[str]],
    model_config: Optional[SSRFConfig] = None,
    scenarios: list = None
) -> pd.DataFrame:
    """
    Compare multiple TC scenarios.

    Args:
        X: Feature DataFrame
        y: Target series
        groups: Feature groups
        model_config: Optional SSRF config
        scenarios: List of (name, tc_rate, account_tier) tuples

    Returns:
        DataFrame comparing scenarios
    """
    if scenarios is None:
        scenarios = [
            ("No TC", 0, "standard"),
            ("Standard (25bps)", 25, "standard"),
            ("Professional (15bps)", 15, "professional"),
            ("Institutional (5bps)", 5, "institutional"),
        ]

    results = []

    for name, tc_rate, tier in scenarios:
        print(f"\nRunning scenario: {name}")

        backtester = TCAdjustedWalkForwardBacktester(
            tc_rate_bps=tc_rate,
            account_tier=tier
        )

        result = backtester.run(
            X, y, groups, model_config, verbose=False
        )

        results.append({
            'scenario': name,
            'tc_rate_bps': result.tc_metrics['effective_tc_rate_bps'],
            'n_trades': result.tc_metrics['n_trades'],
            'avg_turnover': result.tc_metrics['avg_turnover'] * 100,
            'total_tc_cost': result.tc_metrics['total_tc_cost_pct'],
            'gross_return': result.metrics['ann_return_gross'] * 100,
            'net_return': result.metrics['ann_return_net'] * 100,
            'tc_drag': result.metrics['tc_drag'] * 100,
            'sharpe_gross': result.metrics['sharpe_gross'],
            'sharpe_net': result.metrics['sharpe_net'],
            'vs_spx_net': result.metrics['vs_spx_net'] * 100,
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Example usage
    print("TC-Adjusted Backtester - Sample Usage")
    print("=" * 50)

    from .fred_data import generate_sample_data

    # Generate sample data
    indicators, target = generate_sample_data(n_periods=300, n_indicators=50)

    groups = {
        'output_income': [c for c in indicators.columns if 'output' in c][:5],
        'labor': [c for c in indicators.columns if 'labor' in c][:5],
        'inflation': [c for c in indicators.columns if 'inflation' in c][:5],
        'interest': [c for c in indicators.columns if 'interest' in c][:5],
        'sentiment': [c for c in indicators.columns if 'sentiment' in c][:5]
    }

    # Run TC-adjusted backtest
    backtester = TCAdjustedWalkForwardBacktester(
        initial_train_window=120,
        tc_rate_bps=25.0,
        account_tier="standard"
    )

    result = backtester.run(indicators, target, groups, verbose=True)

    print(f"\nTC Metrics:")
    for k, v in result.tc_metrics.items():
        print(f"  {k}: {v}")
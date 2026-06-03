"""
Evaluation Metrics and Reporting Module
Campbell-Thompson R² OOS and economic utility metrics
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    r2_oos: float
    r2_oos_adjusted: float
    mse: float
    mae: float
    mape: float
    hit_ratio: float
    n_pos: int          # number of positive (long) predictions
    n_neg: int          # number of negative (short) predictions
    pos_accuracy: float # accuracy on positive predictions
    neg_accuracy: float # accuracy on negative predictions
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    tail_ratio: float
    skewness: float
    kurtosis: float


class MetricsCalculator:
    """
    Calculate comprehensive evaluation metrics for forecasting models.

    Includes:
    - Statistical accuracy metrics (R² OOS, MSE, MAE)
    - Direction accuracy (hit ratio)
    - Risk-adjusted returns (Sharpe, Sortino, Calmar)
    - Drawdown analysis
    - Distribution statistics
    """

    def __init__(
        self,
        annualization_factor: int = 12,
        risk_free_rate: float = 0.0,
        target_return: float = 0.0,
        max_long: float = 1.0,
        max_short: float = 1.0,
        margin_rate: float = 0.05,
        drawdown_limit: float = 0.25
    ):
        """
        Initialize metrics calculator.

        Args:
            annualization_factor: Factor to annualize returns (12 for monthly)
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            target_return: Target return for Sortino ratio
            max_long: Maximum long position (1.0 = no margin)
            max_short: Maximum short position (1.0 = full short)
            margin_rate: Annual margin interest rate
            drawdown_limit: Max drawdown before levered positions reduced (0.0-0.5)
        """
        self.annualization = annualization_factor
        self.risk_free_rate = risk_free_rate
        self.target_return = target_return
        self.max_long = max_long
        self.max_short = max_short
        self.margin_rate = margin_rate
        self.drawdown_limit = drawdown_limit

    def calculate(
        self,
        predictions: pd.Series,
        actual: pd.Series,
        benchmark: Optional[pd.Series] = None
    ) -> EvaluationMetrics:
        """
        Calculate all metrics.

        Args:
            predictions: Model predictions
            actual: Actual returns
            benchmark: Optional benchmark for relative metrics

        Returns:
            EvaluationMetrics object
        """
        # Align series
        common_idx = predictions.index.intersection(actual.index)
        if benchmark is not None:
            common_idx = common_idx.intersection(benchmark.index)

        pred = predictions.loc[common_idx]
        act = actual.loc[common_idx]
        bench = benchmark.loc[common_idx] if benchmark is not None else None

        # Statistical metrics
        r2_oos = self._r2_oos(act, pred, bench)
        r2_oos_adj = self._r2_oos_adjusted(act, pred, len(common_idx))

        mse = self._mse(act, pred)
        mae = self._mae(act, pred)
        mape = self._mape(act, pred)

        # Direction accuracy
        hit_ratio = self._hit_ratio(act, pred)

        # Direction breakdown: positive vs negative prediction accuracy
        pos_mask = pred > 0
        neg_mask = pred < 0
        n_pos = int(pos_mask.sum())
        n_neg = int(neg_mask.sum())
        if n_pos > 0:
            pos_accuracy = float((np.sign(act[pos_mask]) == 1).mean())
        else:
            pos_accuracy = 0.0
        if n_neg > 0:
            neg_accuracy = float((np.sign(act[neg_mask]) == -1).mean())
        else:
            neg_accuracy = 0.0

        # Return series for portfolio metrics
        portfolio_returns = self._create_portfolio_returns(pred, act)
        bench_returns = self._create_portfolio_returns(bench, act) if bench is not None else None

        # Risk-adjusted metrics
        sharpe = self._sharpe_ratio(portfolio_returns)
        sortino = self._sortino_ratio(portfolio_returns)
        calmar = self._calmar_ratio(portfolio_returns)

        # Drawdown metrics
        max_dd, max_dd_duration = self._drawdown_metrics(portfolio_returns)

        # Return metrics
        cum_ret = self._cumulative_return(portfolio_returns)
        ann_ret = self._annualized_return(portfolio_returns)
        ann_vol = self._annualized_volatility(portfolio_returns)

        # Distribution metrics
        tail = self._tail_ratio(portfolio_returns)
        skew = self._skewness(portfolio_returns)
        kurt = self._kurtosis(portfolio_returns)

        return EvaluationMetrics(
            r2_oos=r2_oos,
            r2_oos_adjusted=r2_oos_adj,
            mse=mse,
            mae=mae,
            mape=mape,
            hit_ratio=hit_ratio,
            n_pos=n_pos,
            n_neg=n_neg,
            pos_accuracy=pos_accuracy,
            neg_accuracy=neg_accuracy,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            cumulative_return=cum_ret,
            annualized_return=ann_ret,
            annualized_volatility=ann_vol,
            tail_ratio=tail,
            skewness=skew,
            kurtosis=kurt
        )

    def _r2_oos(
        self,
        actual: pd.Series,
        predictions: pd.Series,
        benchmark: Optional[pd.Series] = None
    ) -> float:
        """
        Compute Campbell-Thompson R² OOS metric.

        R² OOS = 1 - SS_res / SS_tot

        where SS_res is sum of squared errors from predictions
        and SS_tot is sum of squared errors from benchmark (or mean).
        """
        if benchmark is None:
            benchmark = pd.Series(actual.mean(), index=actual.index)

        ss_res = ((actual - predictions) ** 2).sum()
        ss_tot = ((actual - benchmark) ** 2).sum()

        if ss_tot == 0:
            return 0.0

        return 1 - ss_res / ss_tot

    def _r2_oos_adjusted(
        self,
        actual: pd.Series,
        predictions: pd.Series,
        n: int
    ) -> float:
        """Compute adjusted R² OOS."""
        r2 = self._r2_oos(actual, predictions)
        p = 1  # Number of predictors (simplified)
        return 1 - (1 - r2) * (n - 1) / (n - p - 1)

    def _mse(self, actual: pd.Series, predictions: pd.Series) -> float:
        """Mean Squared Error."""
        return ((actual - predictions) ** 2).mean()

    def _mae(self, actual: pd.Series, predictions: pd.Series) -> float:
        """Mean Absolute Error."""
        return (actual - predictions).abs().mean()

    def _mape(self, actual: pd.Series, predictions: pd.Series) -> float:
        """Mean Absolute Percentage Error."""
        mask = actual != 0
        if mask.sum() == 0:
            return np.nan
        return (actual - predictions).abs().div(actual.abs()).loc[mask].mean()

    def _hit_ratio(self, actual: pd.Series, predictions: pd.Series) -> float:
        """
        Hit ratio: percentage of correct directional predictions.
        """
        return (np.sign(actual) == np.sign(predictions)).mean()

    def _create_portfolio_returns(
        self,
        signals: pd.Series,
        actual: pd.Series
    ) -> pd.Series:
        """
        Create portfolio returns from signals using asymmetric position sizing.

        Uses max_long/max_short for asymmetric sizing, margin_rate for borrowing
        costs, and drawdown_limit for leverage reduction.
        """
        return _simulate_asymmetric_portfolio(
            signals, actual,
            max_long=self.max_long,
            max_short=self.max_short,
            margin_rate=self.margin_rate,
            drawdown_limit=self.drawdown_limit
        )

    def _sharpe_ratio(self, returns: pd.Series) -> float:
        """Annualized Sharpe ratio."""
        excess = returns - self.risk_free_rate / self.annualization

        if returns.std() == 0:
            return 0.0

        return excess.mean() / returns.std() * np.sqrt(self.annualization)

    def _sortino_ratio(self, returns: pd.Series) -> float:
        """Annualized Sortino ratio (downside deviation)."""
        excess = returns - self.target_return / self.annualization

        # Downside returns only
        downside = excess.loc[excess < 0]

        if len(downside) == 0 or downside.std() == 0:
            return np.inf if excess.mean() > 0 else 0.0

        return excess.mean() / downside.std() * np.sqrt(self.annualization)

    def _calmar_ratio(self, returns: pd.Series) -> float:
        """Calmar ratio: annualized return / max drawdown."""
        max_dd, _ = self._drawdown_metrics(returns)

        if max_dd == 0:
            return np.inf if returns.mean() > 0 else 0.0

        ann_ret = self._annualized_return(returns)
        return ann_ret / max_dd

    def _drawdown_metrics(
        self,
        returns: pd.Series
    ) -> Tuple[float, int]:
        """
        Calculate maximum drawdown and duration.

        Returns:
            Tuple of (max_drawdown, max_drawdown_duration_in_periods)
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = cumulative / running_max - 1

        max_dd = abs(drawdowns.min())

        # Find duration of max drawdown
        max_dd_idx = drawdowns.idxmin()
        if max_dd_idx is pd.NaT:
            return max_dd, 0

        # Find when running max was last achieved before max dd
        max_dd_date = returns.index.get_loc(max_dd_idx)
        running_max_before = running_max.iloc[:max_dd_date + 1]

        if len(running_max_before) == 0:
            return max_dd, 0

        last_reset = running_max_before.idxmax()
        if last_reset is pd.NaT:
            last_reset_idx = 0
        else:
            last_reset_idx = returns.index.get_loc(last_reset)

        duration = max_dd_date - last_reset_idx + 1

        return max_dd, duration

    def _cumulative_return(self, returns: pd.Series) -> float:
        """Total cumulative return."""
        return (1 + returns).prod() - 1

    def _annualized_return(self, returns: pd.Series) -> float:
        """Annualized return."""
        cum_ret = self._cumulative_return(returns)
        n_periods = len(returns)
        n_years = n_periods / self.annualization

        if n_years == 0:
            return 0.0

        return (1 + cum_ret) ** (1 / n_years) - 1

    def _annualized_volatility(self, returns: pd.Series) -> float:
        """Annualized volatility."""
        return returns.std() * np.sqrt(self.annualization)

    def _tail_ratio(self, returns: pd.Series) -> float:
        """Tail ratio: 95th percentile / 5th percentile."""
        if len(returns) < 20:
            return np.nan

        percentile_95 = returns.quantile(0.95)
        percentile_5 = returns.quantile(0.05)

        if abs(percentile_5) < 1e-10:
            return np.inf if percentile_95 > 0 else 0.0

        return abs(percentile_95 / percentile_5)

    def _skewness(self, returns: pd.Series) -> float:
        """Skewness of returns."""
        return returns.skew()

    def _kurtosis(self, returns: pd.Series) -> float:
        """Excess kurtosis of returns."""
        return returns.kurtosis()


def _simulate_asymmetric_portfolio(
    signals: pd.Series,
    actual_returns: pd.Series,
    max_long: float = 1.0,
    max_short: float = 1.0,
    margin_rate: float = 0.05,
    drawdown_limit: float = 0.25
) -> pd.Series:
    """
    Simulate portfolio returns with asymmetric position sizing, margin, and drawdown limit.

    Asymmetric sizing: positions range from [-max_short, +max_long] instead of [-1, 1].
    Margin cost: leverage > 1.0 incurs annual margin_rate interest.
    Drawdown limit: levered long positions (>1.0x) are reduced proportionally
    when drawdown exceeds the threshold.

    Args:
        signals: Prediction signals
        actual_returns: Actual period returns
        max_long: Maximum long position (1.0 = no margin, 1.5 = 50% margin)
        max_short: Maximum short position (1.0 = full short, 0.5 = half short, 0.0 = no short)
        margin_rate: Annual margin interest rate (e.g., 0.05 = 5%)
        drawdown_limit: Max drawdown before levered positions are reduced (0.0 to 0.5)

    Returns:
        Portfolio return series (adjusted for margin costs and drawdown limits)
    """
    # Signal strength relative to expanding max
    max_signal = signals.abs().expanding().max().clip(lower=1e-8)
    signal_strength = signals.abs() / max_signal.values

    # Asymmetric positions based on signal direction
    raw_positions = np.where(
        signals.values > 0,
        signal_strength * max_long,
        -signal_strength * max_short
    )

    # Apply drawdown-based leverage reduction on long side
    if drawdown_limit > 0 and max_long > 1.0:
        adjusted_positions = np.zeros(len(signals))
        cumulative = 1.0
        peak = 1.0
        for i in range(len(signals)):
            pos = raw_positions[i]
            # Check drawdown for levered long positions
            if pos > 1.0:
                excess = max(0.0, (peak - cumulative) / peak)
                excess_ratio = excess / drawdown_limit
                if excess_ratio > 0:
                    # Linear reduction: at 1x limit = full levered exposure
                    # at 2x limit = back to 1.0x (no leverage)
                    reduction = max(0.0, 1.0 - min(excess_ratio, 1.0) * 0.5)
                    pos = 1.0 + (pos - 1.0) * reduction
            adjusted_positions[i] = pos

            # Update drawdown tracking
            ret = adjusted_positions[i] * actual_returns.values[i]
            cumulative *= (1 + ret)
            peak = max(peak, cumulative)
    else:
        adjusted_positions = raw_positions

    # Apply margin cost
    if margin_rate > 0:
        leverage = np.maximum(0, np.abs(adjusted_positions) - 1.0)
        margin_cost = leverage * (margin_rate / 12)
        portfolio_returns = adjusted_positions * actual_returns.values - margin_cost
    else:
        portfolio_returns = adjusted_positions * actual_returns.values

    return pd.Series(portfolio_returns, index=actual_returns.index)


class StatisticalTests:
    """
    Statistical tests for forecasting model evaluation.
    """

    @staticmethod
    def dm_test(
        actual: np.ndarray,
        pred1: np.ndarray,
        pred2: np.ndarray,
        h: int = 1
    ) -> Tuple[float, float]:
        """
        Diebold-Mariano test for comparing forecast accuracy.

        Args:
            actual: Actual values
            pred1: Predictions from model 1
            pred2: Predictions from model 2
            h: Forecast horizon (unused; maintained for API compatibility)

        Returns:
            Tuple of (DM statistic, p-value)
        """
        from scipy import stats

        # Loss differential
        e1 = (actual - pred1) ** 2
        e2 = (actual - pred2) ** 2
        d = e1 - e2

        # Mean and variance of loss differential
        n = len(d)
        mean_d = np.mean(d)
        var_d = np.var(d, ddof=1)

        if var_d == 0:
            return np.nan, np.nan

        # DM statistic
        dm_stat = mean_d / np.sqrt(var_d / n)

        # Approximate p-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

        return dm_stat, p_value

    @staticmethod
    def cw_test(
        actual: np.ndarray,
        predictions: np.ndarray,
        benchmark: np.ndarray
    ) -> Tuple[float, float]:
        """
        Clark-West test for nested model comparison.

        Tests if the more complex model adds significant predictive power.
        The standard CW adjustment is: f_t = (y - mean)² - (y - pred)²
        This tests whether the complex model outperforms the benchmark.

        Args:
            actual: Actual values
            predictions: Predictions from complex model
            benchmark: Predictions from benchmark model

        Returns:
            Tuple of (CW statistic, p-value)
        """
        from scipy import stats

        # Errors
        e_complex = actual - predictions
        e_benchmark = actual - benchmark

        # Clark-West adjustment (standard formula)
        # f_t = (y - ȳ)² - (y - ŷ)² where ȳ is the mean of actual
        # adj = f_t + (ŷ - ȳ)² (Clark-West 2007 adjustment for nested models)
        y_mean = np.mean(actual)

        # Standard adjustment term
        adj = (actual - y_mean) ** 2 - e_complex ** 2

        # Add the CW adjustment term for nested models
        adj = adj + (predictions - y_mean) ** 2

        # T-statistic
        t_stat = np.mean(adj) / (np.std(adj) / np.sqrt(len(adj)))

        # P-value (one-sided test - we want to know if complex > benchmark)
        p_value = 1 - stats.t.cdf(t_stat, df=len(adj) - 1)

        return t_stat, p_value

    @staticmethod
    def out_of_sample_r2_confidence_interval(
        r2_oos: float,
        n: int,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compute confidence interval for R² OOS.

        Uses asymptotic approximation. The standard error of R² OOS is approximately:
        SE(R²) ≈ 2 * (1 - R²) / sqrt(n) for large n

        Note: Fisher z-transform is NOT appropriate for R² OOS because R² OOS
        can be negative with no lower bound (-∞), while Fisher z is only valid
        for correlation coefficients r ∈ [-1, 1].

        Args:
            r2_oos: Observed R² OOS
            n: Number of out-of-sample observations
            confidence: Confidence level

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        from scipy import stats

        # Critical value from normal distribution
        z_crit = stats.norm.ppf((1 + confidence) / 2)

        # Asymptotic standard error of R² OOS
        # Based on the variance of sample R², approximately:
        # Var(R²) ≈ 4 * (1 - R²)² / n  (asymptotically normal)
        # So SE(R²) ≈ 2 * |1 - R²| / sqrt(n)
        if r2_oos >= 1:
            se_r2 = 0.0
        elif r2_oos <= -10:  # Very negative R² - use wider CI
            se_r2 = 2 * abs(r2_oos) / np.sqrt(n)
        else:
            se_r2 = 2 * abs(1 - r2_oos) / np.sqrt(n)

        # Confidence interval (asymptotically normal)
        r2_lower = r2_oos - z_crit * se_r2
        r2_upper = r2_oos + z_crit * se_r2

        # Cap upper bound at 1
        r2_upper = min(r2_upper, 1.0)

        return r2_lower, r2_upper


def generate_report(
    metrics: Union[EvaluationMetrics, Dict],
    model_name: str = "SSRF",
    additional_info: Optional[Dict] = None
) -> str:
    """
    Generate formatted evaluation report.

    Args:
        metrics: EvaluationMetrics object or dict with metric values
        model_name: Name of the model
        additional_info: Additional information to include

    Returns:
        Formatted report string
    """
    # Handle both EvaluationMetrics object and dict
    if isinstance(metrics, dict):
        m = metrics
    else:
        m = {
            'r2_oos': metrics.r2_oos,
            'r2_oos_adjusted': metrics.r2_oos_adjusted,
            'mse': metrics.mse,
            'mae': metrics.mae,
            'hit_ratio': metrics.hit_ratio,
            'n_pos': metrics.n_pos,
            'n_neg': metrics.n_neg,
            'pos_accuracy': metrics.pos_accuracy,
            'neg_accuracy': metrics.neg_accuracy,
            'sharpe_ratio': metrics.sharpe_ratio,
            'sortino_ratio': metrics.sortino_ratio,
            'calmar_ratio': metrics.calmar_ratio,
            'cumulative_return': metrics.cumulative_return,
            'annualized_return': metrics.annualized_return,
            'annualized_volatility': metrics.annualized_volatility,
            'max_drawdown': metrics.max_drawdown,
            'max_drawdown_duration': metrics.max_drawdown_duration,
            'skewness': metrics.skewness,
            'kurtosis': metrics.kurtosis,
            'tail_ratio': metrics.tail_ratio
        }

    # Helper function to safely get metric values
    def safe_get(key: str, default: Union[str, float] = 'N/A') -> str:
        """Get metric value safely with fallback for missing keys."""
        val = m.get(key, default)
        if val == 'N/A' or val is None:
            return 'N/A'
        # Format based on key type
        if key in ['hit_ratio', 'cumulative_return', 'annualized_return',
                   'annualized_volatility', 'max_drawdown']:
            return f"{val:.2%}"
        elif key == 'mse':
            return f"{val:.6f}"
        elif key == 'max_drawdown_duration':
            return f"{int(val)} months"
        else:
            return f"{val:.4f}"

    report = []
    report.append("=" * 60)
    report.append(f"EVALUATION REPORT: {model_name}")
    report.append("=" * 60)

    report.append("\n## Statistical Accuracy Metrics")
    report.append("-" * 40)
    report.append(f"Campbell-Thompson R² OOS:  {m.get('r2_oos', 0):.4f}")

    r2_adj = m.get('r2_oos_adjusted')
    if r2_adj is not None:
        report.append(f"Adjusted R² OOS:            {r2_adj:.4f}")

    mse = m.get('mse')
    if mse is not None:
        report.append(f"Mean Squared Error:         {mse:.6f}")

    mae = m.get('mae')
    if mae is not None:
        report.append(f"Mean Absolute Error:        {mae:.4f}")

    report.append("\n## Direction Accuracy")
    report.append("-" * 40)
    hit_ratio = m.get('hit_ratio')
    if hit_ratio is not None:
        report.append(f"Hit Ratio (overall):        {hit_ratio:.2%}")
    n_pos = m.get('n_pos', 0)
    n_neg = m.get('n_neg', 0)
    if n_pos > 0:
        pos_acc = m.get('pos_accuracy', 0)
        report.append(f"Long  accuracy:  {pos_acc:.2%}  ({n_pos} predictions)")
    if n_neg > 0:
        neg_acc = m.get('neg_accuracy', 0)
        report.append(f"Short accuracy:  {neg_acc:.2%}  ({n_neg} predictions)")

    report.append("\n## Risk-Adjusted Performance")
    report.append("-" * 40)

    sharpe = m.get('sharpe_ratio')
    if sharpe is not None:
        report.append(f"Sharpe Ratio:               {sharpe:.4f}")

    sortino = m.get('sortino_ratio')
    if sortino is not None:
        report.append(f"Sortino Ratio:              {sortino:.4f}")

    calmar = m.get('calmar_ratio')
    if calmar is not None:
        report.append(f"Calmar Ratio:               {calmar:.4f}")

    report.append("\n## Return Metrics")
    report.append("-" * 40)

    cum_ret = m.get('cumulative_return')
    if cum_ret is not None:
        report.append(f"Cumulative Return:          {cum_ret:.2%}")

    ann_ret = m.get('annualized_return')
    if ann_ret is not None:
        report.append(f"Annualized Return:           {ann_ret:.2%}")

    ann_vol = m.get('annualized_volatility')
    if ann_vol is not None:
        report.append(f"Annualized Volatility:       {ann_vol:.2%}")

    report.append("\n## Drawdown Analysis")
    report.append("-" * 40)

    max_dd = m.get('max_drawdown')
    if max_dd is not None:
        report.append(f"Maximum Drawdown:           {max_dd:.2%}")

    max_dd_dur = m.get('max_drawdown_duration')
    if max_dd_dur is not None:
        report.append(f"Max Drawdown Duration:      {int(max_dd_dur)} months")

    report.append("\n## Return Distribution")
    report.append("-" * 40)

    skew = m.get('skewness')
    if skew is not None:
        report.append(f"Skewness:                   {skew:.4f}")

    kurt = m.get('kurtosis')
    if kurt is not None:
        report.append(f"Excess Kurtosis:            {kurt:.4f}")

    tail = m.get('tail_ratio')
    if tail is not None:
        report.append(f"Tail Ratio:                 {tail:.4f}")

    if additional_info:
        report.append("\n## Additional Information")
        report.append("-" * 40)
        for key, value in additional_info.items():
            report.append(f"{key}: {value}")

    report.append("\n" + "=" * 60)

    return "\n".join(report)


# =============================================================================
# Comparison Utilities
# =============================================================================
def compare_backtests(
    results: List[Dict],
    metrics_to_compare: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare multiple backtest results.

    Args:
        results: List of result dictionaries with 'name' and 'metrics'
        metrics_to_compare: List of metric names to compare

    Returns:
        DataFrame with comparison
    """
    if metrics_to_compare is None:
        metrics_to_compare = [
            'r2_oos', 'hit_ratio', 'sharpe_ratio', 'calmar_ratio',
            'max_drawdown', 'cumulative_return'
        ]

    comparison = []

    for result in results:
        row = {'model': result['name']}
        for metric in metrics_to_compare:
            if metric in result['metrics']:
                row[metric] = result['metrics'][metric]
        comparison.append(row)

    df = pd.DataFrame(comparison)
    df = df.set_index('model')

    return df


def format_metrics_table(metrics: Dict[str, float]) -> str:
    """Format metrics as a markdown table."""
    lines = []
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")

    for key, value in metrics.items():
        # Format based on value magnitude
        if abs(value) > 1:
            formatted = f"{value:.4f}"
        elif abs(value) > 0.01:
            formatted = f"{value:.2%}"
        else:
            formatted = f"{value:.4f}"

        # Pretty print metric name
        name = key.replace('_', ' ').title()
        lines.append(f"| {name} | {formatted} |")

    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    print("Metrics Calculator - Sample Usage")
    print("=" * 50)

    # Sample data
    np.random.seed(42)
    n = 120

    actual = pd.Series(np.random.randn(n) * 0.05)
    predictions = pd.Series(actual.values + np.random.randn(n) * 0.02)
    benchmark = pd.Series(np.zeros(n) + actual.mean())

    # Calculate metrics
    calc = MetricsCalculator(annualization_factor=12)
    metrics = calc.calculate(predictions, actual, benchmark)

    # Generate report
    report = generate_report(metrics, "Sample Model")
    print(report)
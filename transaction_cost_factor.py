"""
Transaction Cost Model with Discounted Retail Rates

This module models transaction costs as a factor in the SSRF framework.
Implements tiered cost structures based on portfolio size, trade frequency, and broker type.

Key Features:
- Discounted retail rates with volume/tier-based discounts
- Market impact estimation
- Turnover-aware cost modeling
- Break-even alpha calculation
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BrokerType(Enum):
    """Broker tier enumeration for cost modeling."""
    RETAIL = "retail"              # Interactive Brokers, TD Ameritrade retail
    DISCOUNT = "discount"          # Charles Schwab, Fidelity
    PREMIUM = "premium"            # Full-service broker
    INSTITUTIONAL = "institutional"  # Major institutions


class AccountTier(Enum):
    """Account size tier for discount application."""
    MICRO = "micro"       # < $10k
    SMALL = "small"       # $10k - $100k
    STANDARD = "standard" # $100k - $1M
    PROFESSIONAL = "professional"  # $1M - $10M
    INSTITUTIONAL = "institutional" # > $10M


@dataclass
class TransactionCostConfig:
    """
    Configuration for transaction cost modeling.

    Models discounted retail rates with tier-based discounts:
    - Base spread: 0.5-2 bps depending on asset class
    - Commission: $0-$1 per trade (already declining)
    - Market impact: Proportional to trade size and volatility
    """

    # Base transaction cost rates (in basis points)
    base_rate_bps: float = 25.0  # Retail base rate

    # Broker/commission structure
    broker_type: BrokerType = BrokerType.RETAIL

    # Account tier for discount calculation
    account_tier: AccountTier = AccountTier.STANDARD

    # Volume-based discount parameters
    monthly_trades_threshold: int = 20  # Trades needed for volume discount
    volume_discount_pct: float = 0.15   # 15% discount for high volume

    # Market impact parameters
    estimated_market_impact_bps: float = 5.0  # Base market impact

    # Spread component (for ETF trading)
    spread_cost_bps: float = 1.0  # Half-spread cost

    # Opportunity cost (slippage simulation)
    slippage_bps: float = 2.0  # Estimated slippage

    def __post_init__(self):
        """Calculate discounted rate based on tier."""
        self.effective_rate = self.calculate_discounted_rate()

    def calculate_discounted_rate(self) -> float:
        """
        Calculate effective transaction cost rate with all discounts.

        Discount factors:
        1. Broker type discount (institutional vs retail)
        2. Account tier discount (larger accounts get better rates)
        3. Volume discount (frequent traders)
        """
        rate = self.base_rate_bps

        # Broker type discount
        broker_discounts = {
            BrokerType.INSTITUTIONAL: 0.30,  # 70% off for institutions
            BrokerType.DISCOUNT: 0.60,        # 40% off
            BrokerType.PREMIUM: 0.75,          # 25% off
            BrokerType.RETAIL: 1.00,           # No discount
        }
        rate *= broker_discounts.get(self.broker_type, 1.0)

        # Account tier discount
        tier_discounts = {
            AccountTier.INSTITUTIONAL: 0.70,
            AccountTier.PROFESSIONAL: 0.80,
            AccountTier.STANDARD: 0.90,
            AccountTier.SMALL: 0.95,
            AccountTier.MICRO: 1.00,
        }
        rate *= tier_discounts.get(self.account_tier, 1.0)

        return rate

    def get_cost_for_trade(self, trade_value: float, turnover_pct: float = 0.10) -> Tuple[float, float]:
        """
        Calculate total cost for a trade.

        Args:
            trade_value: Dollar value of the trade
            turnover_pct: Expected portfolio turnover percentage

        Returns:
            Tuple of (total_cost_bps, total_dollar_cost)
        """
        # Base cost is already in bps
        base_cost = self.effective_rate

        # Market impact scales with trade size
        # Larger trades = higher market impact
        # Assume 10% of portfolio = $100k for standard account
        impact_multiplier = 1.0 + (turnover_pct * 0.5)

        total_bps = base_cost + (self.estimated_market_impact_bps * impact_multiplier) + self.spread_cost_bps

        # Calculate dollar cost
        dollar_cost = trade_value * (total_bps / 10000)

        return total_bps, dollar_cost


@dataclass
class TCModelState:
    """State for transaction cost model."""
    accumulated_costs: float = 0.0
    n_trades: int = 0
    turnover_history: list = field(default_factory=list)
    cost_history: list = field(default_factory=list)


class TransactionCostFactor:
    """
    Transaction Cost Factor for the SSRF framework.

    Models transaction costs as a multiplicative factor applied to signals:
    - Net Signal = Gross Signal * (1 - TC_factor)
    - TC_factor = turnover * effective_cost_rate

    The discount factor represents the cost of trading as a fraction of signal.
    """

    def __init__(self, config: Optional[TransactionCostConfig] = None):
        """
        Initialize Transaction Cost Factor.

        Args:
            config: TC configuration object
        """
        self.config = config or TransactionCostConfig()
        self.state = TCModelState()
        self._turnover_threshold = 0.15  # 15% turnover triggers TC impact

    def compute_tc_factor(
        self,
        position_change: float,
        current_position: float,
        signal: float,
        confidence: float = 1.0
    ) -> float:
        """
        Compute transaction cost factor for signal adjustment.

        Args:
            position_change: Change in position (0-1 scale)
            current_position: Current position size
            signal: Raw prediction signal
            confidence: Model confidence (0-1)

        Returns:
            TC-adjusted signal
        """
        # Calculate turnover for this trade
        if current_position > 0:
            turnover = abs(position_change) / abs(current_position)
        else:
            turnover = abs(position_change)  # New position

        # Effective cost rate with confidence discount
        # Higher confidence = larger trade = more market impact
        effective_rate = self.config.effective_rate * (1.0 + (1 - confidence) * 0.5)

        # TC factor as a fraction of signal
        tc_factor = min(turnover * effective_rate / 10000, 0.5)  # Cap at 50% cost

        # Accumulate for tracking
        self.state.n_trades += 1
        self.state.accumulated_costs += tc_factor
        self.state.turnover_history.append(turnover)
        self.state.cost_history.append(tc_factor)

        return 1.0 - tc_factor

    def adjust_signal(self, signal: float, turnover_estimate: float = 0.10) -> float:
        """
        Adjust signal for transaction costs.

        Args:
            signal: Raw prediction signal
            turnover_estimate: Estimated turnover rate

        Returns:
            TC-adjusted signal
        """
        # Simple adjustment using turnover estimate
        tc_factor = turnover_estimate * (self.config.effective_rate / 10000)
        return signal * (1.0 - tc_factor)

    def get_cost_summary(self) -> Dict:
        """Get summary of transaction costs."""
        if len(self.state.cost_history) == 0:
            return {
                'n_trades': 0,
                'total_cost_bps': 0.0,
                'avg_turnover': 0.0,
                'avg_cost_per_trade_bps': 0.0,
            }

        return {
            'n_trades': self.state.n_trades,
            'total_cost_bps': self.state.accumulated_costs * 10000,
            'avg_turnover': np.mean(self.state.turnover_history),
            'avg_cost_per_trade_bps': np.mean(self.state.cost_history) * 10000,
            'effective_rate': self.config.effective_rate,
        }


class TurnoverEstimator:
    """
    Estimates portfolio turnover based on signal changes.

    Turnover = |new_position - old_position| / (|old_position| + |new_position|)
    """

    def __init__(self, window: int = 12):
        self.window = window
        self.position_history = []

    def estimate_turnover(
        self,
        predictions: pd.Series,
        position_threshold: float = 0.02
    ) -> pd.Series:
        """
        Estimate portfolio turnover from prediction signals.

        Args:
            predictions: Prediction signals
            position_threshold: Minimum signal to trigger trade

        Returns:
            Series of estimated turnover rates
        """
        # Convert signals to positions
        positions = self._signals_to_positions(predictions, position_threshold)

        # Calculate turnover
        turnover = pd.Series(index=predictions.index, dtype=float)

        for i in range(len(positions)):
            if i == 0:
                turnover.iloc[i] = 0.0  # No turnover for first position
            else:
                old_pos = positions.iloc[i - 1]
                new_pos = positions.iloc[i]

                # Turnover formula
                pos_sum = abs(old_pos) + abs(new_pos)
                if pos_sum > 0:
                    turnover.iloc[i] = abs(new_pos - old_pos) / pos_sum
                else:
                    turnover.iloc[i] = 0.0

        self.position_history = positions.tolist()
        return turnover

    def _signals_to_positions(
        self,
        signals: pd.Series,
        threshold: float
    ) -> pd.Series:
        """Convert signals to position sizes."""
        positions = signals.copy()

        # Apply threshold - don't trade for small signals
        positions[abs(positions) < threshold] = 0

        # Scale to [-1, 1]
        max_abs = positions.abs().max()
        if max_abs > 0:
            positions = positions / max_abs

        return positions


def compute_break_even_alpha(
    tc_rate_bps: float,
    avg_turnover: float,
    n_periods: int = 252,
    target_return_pct: float = 0.0
) -> float:
    """
    Compute break-even alpha needed to cover transaction costs.

    Args:
        tc_rate_bps: Transaction cost rate in basis points
        avg_turnover: Average turnover rate per period
        n_periods: Number of periods per year
        target_return_pct: Target net return (0 = break even with zero)

    Returns:
        Required gross alpha in percentage
    """
    # Cost per period
    cost_per_period = tc_rate_bps / 10000 * avg_turnover

    # Annual cost
    annual_cost = cost_per_period * n_periods

    # Break-even gross alpha
    # Net = Gross - Cost => Gross = Net + Cost
    required_gross = target_return_pct + annual_cost

    return required_gross


def run_tc_sensitivity_analysis(
    returns: pd.Series,
    predictions: pd.Series,
    base_tc_bps: float = 25.0,
    turnover_rates: list = None
) -> pd.DataFrame:
    """
    Run sensitivity analysis on transaction costs.

    Args:
        returns: Actual returns
        predictions: Prediction signals
        base_tc_bps: Base transaction cost in bps
        turnover_rates: List of turnover rates to test

    Returns:
        DataFrame with results for each TC/turnover combination
    """
    if turnover_rates is None:
        turnover_rates = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

    results = []

    for turnover in turnover_rates:
        for tc_bps in [10, 15, 20, 25, 30, 50, 75, 100]:
            # Calculate cost impact
            period_cost = tc_bps / 10000 * turnover
            annual_cost = period_cost * 12  # Monthly rebalancing

            # Apply costs to returns
            # Gross return is whatever the strategy produces
            gross_returns = returns.copy()

            # Net return after TC
            net_returns = returns - period_cost

            # Sharpe calculation
            if net_returns.std() > 0:
                sharpe_gross = gross_returns.mean() / gross_returns.std() * np.sqrt(12)
                sharpe_net = net_returns.mean() / net_returns.std() * np.sqrt(12)
            else:
                sharpe_gross = 0.0
                sharpe_net = 0.0

            results.append({
                'turnover': turnover,
                'tc_bps': tc_bps,
                'annual_cost_pct': annual_cost * 100,
                'net_return': net_returns.mean() * 12 * 100,
                'gross_return': gross_returns.mean() * 12 * 100,
                'sharpe_gross': sharpe_gross,
                'sharpe_net': sharpe_net,
            })

    return pd.DataFrame(results)


# =============================================================================
# TC-Adjusted Backtesting
# =============================================================================

class TCAdjustedBacktester:
    """
    Backtester with integrated transaction cost modeling.

    Extends standard walk-forward backtesting with:
    - Per-trade TC calculation
    - Cumulative TC tracking
    - Net return computation
    """

    def __init__(self, tc_config: Optional[TransactionCostConfig] = None):
        """
        Initialize TC-adjusted backtester.

        Args:
            tc_config: Transaction cost configuration
        """
        self.tc_config = tc_config or TransactionCostConfig()
        self.tc_factor = TransactionCostFactor(self.tc_config)

    def simulate_with_tc(
        self,
        predictions: pd.Series,
        actual_returns: pd.Series,
        position_threshold: float = 0.02
    ) -> dict:
        """
        Simulate portfolio returns with transaction costs.

        Args:
            predictions: Prediction signals
            actual_returns: Actual returns
            position_threshold: Minimum signal to trigger trade

        Returns:
            Dictionary with gross/net returns and TC summary
        """
        # Estimate turnover
        turnover_estimator = TurnoverEstimator()
        turnover = turnover_estimator.estimate_turnover(predictions, position_threshold)

        # Calculate TC per period
        tc_per_period = turnover * (self.tc_config.effective_rate / 10000)

        # Gross and net returns
        gross_returns = actual_returns * np.sign(predictions.values).clip(0, 1)
        net_returns = gross_returns - tc_per_period

        # Cumulative returns
        gross_cumulative = (1 + gross_returns).cumprod()
        net_cumulative = (1 + net_returns).cumprod()

        # Summary
        gross_total = gross_cumulative.iloc[-1] - 1
        net_total = net_cumulative.iloc[-1] - 1
        n_years = len(gross_returns) / 12

        gross_ann = (1 + gross_total) ** (1 / n_years) - 1 if n_years > 0 else 0
        net_ann = (1 + net_total) ** (1 / n_years) - 1 if n_years > 0 else 0

        tc_summary = self.tc_factor.get_cost_summary()
        total_tc_cost = tc_per_period.sum() * 100

        return {
            'gross_return_ann': gross_ann,
            'net_return_ann': net_ann,
            'gross_total': gross_total,
            'net_total': net_total,
            'total_tc_cost_pct': total_tc_cost,
            'avg_turnover': turnover.mean(),
            'gross_cumulative': gross_cumulative,
            'net_cumulative': net_cumulative,
            'turnover': turnover,
            'tc_per_period': tc_per_period,
            'tc_config': self.tc_config,
            'n_trades': int(turnover[turnover > 0].count()),
        }


# =============================================================================
# Main Analysis Function
# =============================================================================

def run_tc_analysis_pipeline(
    predictions: pd.Series,
    actual_returns: pd.Series,
    benchmark_returns: pd.Series = None,
    output_dir: str = None
) -> dict:
    """
    Run complete transaction cost analysis pipeline.

    Args:
        predictions: Model predictions
        actual_returns: Actual returns
        benchmark_returns: Benchmark returns (S&P 500)
        output_dir: Optional output directory

    Returns:
        Dictionary with all analysis results
    """
    print("\n" + "=" * 80)
    print("TRANSACTION COST ANALYSIS PIPELINE")
    print("=" * 80)

    # Define TC scenarios
    scenarios = [
        ("Retail (No Discount)", BrokerType.RETAIL, AccountTier.MICRO, 50.0),
        ("Retail (Standard)", BrokerType.RETAIL, AccountTier.STANDARD, 25.0),
        ("Discount Broker", BrokerType.DISCOUNT, AccountTier.SMALL, 15.0),
        ("Professional", BrokerType.DISCOUNT, AccountTier.PROFESSIONAL, 10.0),
        ("Institutional", BrokerType.INSTITUTIONAL, AccountTier.INSTITUTIONAL, 5.0),
    ]

    results = []

    for name, broker, tier, base_rate in scenarios:
        config = TransactionCostConfig(
            base_rate_bps=base_rate,
            broker_type=broker,
            account_tier=tier
        )

        backtester = TCAdjustedBacktester(config)
        result = backtester.simulate_with_tc(predictions, actual_returns)

        # Compare to benchmark
        if benchmark_returns is not None:
            spx_ann = benchmark_returns.mean() * 12
            vs_sp500 = result['net_return_ann'] - spx_ann
        else:
            spx_ann = 0.0
            vs_sp500 = 0.0

        print(f"\n{name} (Rate: {config.effective_rate:.1f} bps):")
        print(f"  Gross Annual Return:  {result['gross_return_ann']*100:.2f}%")
        print(f"  Net Annual Return:   {result['net_return_ann']*100:.2f}%")
        print(f"  Total TC Cost:        {result['total_tc_cost_pct']:.1f}%")
        print(f"  Avg Turnover:         {result['avg_turnover']*100:.1f}%")
        print(f"  vs S&P 500:           {vs_sp500*100:.2f}%")

        results.append({
            'scenario': name,
            'effective_rate_bps': config.effective_rate,
            'gross_return': result['gross_return_ann'] * 100,
            'net_return': result['net_return_ann'] * 100,
            'tc_cost_pct': result['total_tc_cost_pct'],
            'avg_turnover': result['avg_turnover'] * 100,
            'vs_sp500': vs_sp500 * 100,
        })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Sensitivity analysis
    print("\n" + "-" * 80)
    print("SENSITIVITY ANALYSIS: TC Rate vs Turnover")
    print("-" * 80)

    turnover_rates = [0.05, 0.10, 0.15, 0.20, 0.30]
    tc_rates = [5, 10, 15, 20, 25, 50]

    # Compute break-even matrix
    print(f"\n{'Turnover':<12}", end="")
    for tc in tc_rates:
        print(f"  {tc:>5}bps", end="")
    print()

    for turnover in turnover_rates:
        print(f"{turnover*100:>6.0f}%   ", end="")
        for tc in tc_rates:
            annual_cost = tc / 10000 * turnover * 12 * 100
            print(f"  {annual_cost:>5.1f}%", end="")
        print()

    # Find break-even: what alpha do you need?
    print("\n" + "-" * 80)
    print("BREAK-EVEN ALPHA ANALYSIS")
    print("-" * 80)
    print("\nWhat gross return is needed to cover TC and beat S&P 500?")
    print("S&P 500 historical: ~10% nominal, ~7% real")
    print()

    for turnover in [0.10, 0.15, 0.20]:
        print(f"With {turnover*100:.0f}% turnover:")
        for tc in [10, 25, 50]:
            annual_cost = tc / 10000 * turnover * 12
            break_even = 0.10 + annual_cost  # Need to beat 10% + costs
            print(f"  {tc:>3} bps TC: Break-even gross return = {break_even*100:.1f}%")

    return {
        'scenarios': results_df,
        'tc_config': self.tc_config if 'self' in dir() else None,
    }


if __name__ == "__main__":
    # Quick test
    print("Transaction Cost Model Test")
    print("=" * 50)

    # Create sample config
    config = TransactionCostConfig(
        base_rate_bps=25.0,
        broker_type=BrokerType.RETAIL,
        account_tier=AccountTier.STANDARD
    )

    print(f"Base rate: {config.base_rate_bps} bps")
    print(f"Effective rate: {config.effective_rate:.1f} bps")
    print(f"Break-even alpha (10% turnover): {compute_break_even_alpha(25, 0.10)*100:.2f}%")

    # Sensitivity test
    print("\nBreak-even analysis:")
    for turnover in [0.05, 0.10, 0.15, 0.20]:
        break_even = compute_break_even_alpha(25, turnover)
        print(f"  Turnover {turnover*100:.0f}%: Need {break_even*100:.2f}% gross return")
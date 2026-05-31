"""
TC-Adjusted SSRF Backtest Analysis
Integrates transaction costs as a factor in SSRF modeling
"""

import yfinance as yf
import pandas as pd
import numpy as np
from transaction_cost_factor import (
    TransactionCostConfig, BrokerType, AccountTier,
    TCAdjustedBacktester, compute_break_even_alpha,
    run_tc_sensitivity_analysis
)

print('=' * 80)
print('SSRF TRANSACTION COST ANALYSIS')
print('Including Transaction Costs as a Factor in the Model')
print('=' * 80)

# Load sector ETF data
sector_etfs = {
    'Materials': 'XLB', 'Energy': 'XLE', 'Financials': 'XLF',
    'Industrials': 'XLI', 'Technology': 'XLK', 'Consumer_Staples': 'XLP',
    'Health_Care': 'XLV', 'Utilities': 'XLU', 'Consumer_Discretionary': 'XLY',
}

# Download data
all_prices = {}
for sector, ticker in sector_etfs.items():
    try:
        data = yf.download(ticker, start='1999-01-01', end='2026-05-29', progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            all_prices[sector] = data['Close'][ticker]
        else:
            all_prices[sector] = data['Close']
    except:
        pass

monthly_prices = pd.DataFrame(all_prices).resample('ME').last()
returns = monthly_prices.pct_change().dropna()

# S&P 500 benchmark
spx = yf.download('^GSPC', start='1999-01-01', end='2026-05-29', progress=False)
if isinstance(spx.columns, pd.MultiIndex):
    spx_returns = spx['Close']['^GSPC'].resample('ME').last().pct_change()
else:
    spx_returns = spx['Close'].resample('ME').last().pct_change()

# Align dates
common_idx = returns.index.intersection(spx_returns.index)
returns = returns.loc[common_idx]
spx_returns = spx_returns.loc[common_idx]

print(f'\nData: {len(returns)} months, {returns.index[0].strftime("%Y-%m")} to {returns.index[-1].strftime("%Y-%m")}')

# === SCENARIO 1: SSRF PREDICTIONS (Simulated) ===
# Since we don't have actual SSRF predictions, we'll use a realistic model signal
# This could be: MACD cross, trend-following signal, or ML predictions

# Simulate SSRF predictions based on momentum
# In reality, this would come from the actual SSRF model
print('\n' + '=' * 80)
print('SIMULATED SSRF PREDICTIONS')
print('(In production, these would come from the SSRF model)')
print('=' * 80)

# Create synthetic SSRF predictions (momentum-based)
# This simulates what the SSRF model might output
def simulate_ssrf_predictions(returns, lookback=6):
    """Simulate SSRF predictions using smoothed momentum."""
    # Moving average crossover signal
    short_ma = returns.mean(axis=1).rolling(lookback).mean()
    long_ma = returns.mean(axis=1).rolling(lookback * 2).mean()
    signal = (short_ma - long_ma) / long_ma.abs()
    return signal.fillna(0)

ssrf_signal = simulate_ssrf_predictions(returns)

# Convert signal to positions
def signal_to_position(signal, threshold=0.01):
    """Convert signal to position, only trade on strong signals."""
    position = signal.copy()
    position[abs(position) < threshold] = 0  # No trade for weak signals
    # Scale to [-1, 1]
    max_abs = position.abs().max()
    if max_abs > 0:
        position = position / max_abs
    return position.fillna(0)

ssrf_position = signal_to_position(ssrf_signal, threshold=0.02)

# Calculate gross returns (before TC)
sector_avg_return = returns.mean(axis=1)
gross_returns = sector_avg_return * ssrf_position.clip(lower=0)  # Long only

# Calculate turnover
turnover = abs(ssrf_position.diff()).fillna(0)

# Benchmark (S&P 500)
spx_aligned = spx_returns.loc[returns.index]

print(f'\nSignal statistics:')
print(f'  Mean signal: {ssrf_signal.mean():.4f}')
print(f'  Signal std: {ssrf_signal.std():.4f}')
print(f'  Mean position: {ssrf_position.abs().mean():.2f}')
print(f'  Average turnover: {turnover.mean()*100:.1f}%')

# === SCENARIO 2: APPLY TRANSACTION COSTS ===
print('\n' + '=' * 80)
print('SCENARIO 2: APPLY TRANSACTION COSTS')
print('Transaction cost as a multiplicative factor in the model')
print('=' * 80)

# Define scenarios with discounted retail rates
scenarios = [
    {
        'name': 'Retail (Micro Account)',
        'broker': BrokerType.RETAIL,
        'tier': AccountTier.MICRO,
        'base_rate': 50.0,
    },
    {
        'name': 'Retail (Standard Account)',
        'broker': BrokerType.RETAIL,
        'tier': AccountTier.STANDARD,
        'base_rate': 25.0,
    },
    {
        'name': 'Discount Broker (High Volume)',
        'broker': BrokerType.DISCOUNT,
        'tier': AccountTier.PROFESSIONAL,
        'base_rate': 15.0,
    },
    {
        'name': 'Institutional (Large Account)',
        'broker': BrokerType.INSTITUTIONAL,
        'tier': AccountTier.INSTITUTIONAL,
        'base_rate': 5.0,
    },
]

results = []

for scenario in scenarios:
    config = TransactionCostConfig(
        base_rate_bps=scenario['base_rate'],
        broker_type=scenario['broker'],
        account_tier=scenario['tier']
    )

    backtester = TCAdjustedBacktester(config)
    result = backtester.simulate_with_tc(
        pd.Series(ssrf_signal.values, index=returns.index),
        pd.Series(gross_returns.values, index=returns.index),
        position_threshold=0.02
    )

    # Calculate metrics
    n_years = len(returns) / 12
    spx_total = (1 + spx_aligned).prod() - 1
    spx_ann = (1 + spx_total) ** (1 / n_years) - 1

    vs_sp500 = result['net_return_ann'] - spx_ann

    print(f"\n{scenario['name']}:")
    print(f"  Base rate: {scenario['base_rate']} bps")
    print(f"  Effective rate: {config.effective_rate:.1f} bps")
    print(f"  Gross Annual Return: {result['gross_return_ann']*100:.2f}%")
    print(f"  Net Annual Return: {result['net_return_ann']*100:.2f}%")
    print(f"  S&P 500 Annual: {spx_ann*100:.2f}%")
    print(f"  vs S&P 500: {vs_sp500*100:+.2f}%")
    print(f"  Total TC Cost: {result['total_tc_cost_pct']:.1f}%")
    print(f"  # Trades: {result['n_trades']}")

    results.append({
        'scenario': scenario['name'],
        'base_rate': scenario['base_rate'],
        'effective_rate': config.effective_rate,
        'gross_return': result['gross_return_ann'] * 100,
        'net_return': result['net_return_ann'] * 100,
        'spx_return': spx_ann * 100,
        'vs_sp500': vs_sp500 * 100,
        'tc_cost': result['total_tc_cost_pct'],
        'n_trades': result['n_trades'],
        'avg_turnover': result['avg_turnover'] * 100,
    })

# === SCENARIO 3: TC AS A FACTOR IN MODEL ===
print('\n' + '=' * 80)
print('SCENARIO 3: TC FACTOR INTEGRATION')
print('Modeling transaction cost as a signal multiplier')
print('=' * 80)

def apply_tc_factor(signal, turnover, tc_rate_bps):
    """
    Apply TC factor to signal.
    Net_signal = Gross_signal * (1 - tc_factor)
    tc_factor = turnover * tc_rate
    """
    tc_factor = turnover * (tc_rate_bps / 10000)
    net_signal = signal * (1 - tc_factor)
    return net_signal

# Compare gross vs net signals
print('\nSignal adjustment for TC:')
print(f'  Mean gross signal: {ssrf_signal.abs().mean():.4f}')
print(f'  Mean net signal (25bps): {apply_tc_factor(ssrf_signal, turnover.mean(), 25).abs().mean():.4f}')
print(f'  Mean net signal (10bps): {apply_tc_factor(ssrf_signal, turnover.mean(), 10).abs().mean():.4f}')

# === SCENARIO 4: BREAK-EVEN ANALYSIS ===
print('\n' + '=' * 80)
print('SCENARIO 4: BREAK-EVEN ALPHA ANALYSIS')
print('What gross alpha does SSRF need to beat S&P 500 with TC?')
print('=' * 80)

# S&P 500 benchmark return
spx_ann_pct = spx_ann * 100

print(f'\nS&P 500 Annual Return: {spx_ann_pct:.2f}%')
print('\nRequired Gross Return to Break Even (with TC):')

# Different turnover levels
turnover_levels = [0.05, 0.10, 0.15, 0.20]
tc_levels = [5, 10, 15, 25, 50]

print(f'\n{"Turnover":<10}', end='')
for tc in tc_levels:
    print(f"  {tc:>5}bps", end='')
print()
print('-' * 50)

for turnover in turnover_levels:
    print(f"{turnover*100:>6.0f}%   ", end='')
    for tc in tc_levels:
        # Cost = turnover * tc_rate (annualized)
        annual_cost_pct = turnover * (tc / 10000) * 12 * 100
        break_even = spx_ann_pct + annual_cost_pct
        print(f"  {break_even:>5.1f}%", end='')
    print()

# === SCENARIO 5: DISCOUNTED RETAIL RATE STRUCTURE ===
print('\n' + '=' * 80)
print('SCENARIO 5: DISCOUNTED RETAIL RATE STRUCTURE')
print('Modeling tier-based discounts')
print('=' * 80)

# Create discount scenarios
discount_scenarios = []

print('\nTier-Based Discount Structure:')
print('=' * 60)

# Tier 1: Micro account, no discount
tier1 = TransactionCostConfig(base_rate_bps=50, broker_type=BrokerType.RETAIL, account_tier=AccountTier.MICRO)
print(f'\n1. Micro Account (< $10k):')
print(f'   Base Rate: {tier1.base_rate_bps} bps')
print(f'   Effective Rate: {tier1.effective_rate:.1f} bps')
print(f'   Discount: 0%')

# Tier 2: Standard account, small discount
tier2 = TransactionCostConfig(base_rate_bps=50, broker_type=BrokerType.RETAIL, account_tier=AccountTier.STANDARD)
print(f'\n2. Standard Account ($10k-$100k):')
print(f'   Base Rate: {tier2.base_rate_bps} bps')
print(f'   Effective Rate: {tier2.effective_rate:.1f} bps')
print(f'   Discount: {(1-tier2.effective_rate/tier2.base_rate_bps)*100:.0f}%')

# Tier 3: High volume discount
tier3 = TransactionCostConfig(base_rate_bps=50, broker_type=BrokerType.DISCOUNT, account_tier=AccountTier.PROFESSIONAL)
print(f'\n3. Professional Account ($1M-$10M):')
print(f'   Base Rate: {tier3.base_rate_bps} bps')
print(f'   Effective Rate: {tier3.effective_rate:.1f} bps')
print(f'   Discount: {(1-tier3.effective_rate/tier3.base_rate_bps)*100:.0f}%')

# Tier 4: Institutional
tier4 = TransactionCostConfig(base_rate_bps=50, broker_type=BrokerType.INSTITUTIONAL, account_tier=AccountTier.INSTITUTIONAL)
print(f'\n4. Institutional Account (> $10M):')
print(f'   Base Rate: {tier4.base_rate_bps} bps')
print(f'   Effective Rate: {tier4.effective_rate:.1f} bps')
print(f'   Discount: {(1-tier4.effective_rate/tier4.base_rate_bps)*100:.0f}%')

# === SCENARIO 6: SENSITIVITY ANALYSIS ===
print('\n' + '=' * 80)
print('SCENARIO 6: SENSITIVITY ANALYSIS')
print('Net return as function of TC rate and turnover')
print('=' * 80)

# Build sensitivity grid
avg_turnover = turnover.mean() if isinstance(turnover, pd.Series) else turnover
gross_ann = gross_returns.mean() * 12 * 100

print(f'\nGross Strategy Return: {gross_ann:.2f}%')
print(f'Average Turnover: {avg_turnover*100:.1f}%')
print('\n')

# Header
print(f'{"TC Rate":<10}', end='')
for tc in [5, 10, 15, 20, 25, 30, 50]:
    print(f'  {tc:>5}bps', end='')
print()
print('-' * 80)

# Rows for different gross returns
for gross in [5, 7, 10, 12, 15]:
    print(f'{gross:>5}% gross:', end='')
    for tc in [5, 10, 15, 20, 25, 30, 50]:
        annual_cost = avg_turnover * (tc / 10000) * 12 * 100
        net = gross - annual_cost
        print(f'  {net:>5.1f}%', end='')
    print()

# === SUMMARY ===
print('\n' + '=' * 80)
print('SUMMARY: TC FACTOR INTEGRATION RESULTS')
print('=' * 80)

print(f'''
KEY INSIGHTS:

1. TC as a Factor:
   - Transaction cost reduces signal strength: Net = Gross * (1 - TC_factor)
   - TC_factor = turnover * effective_cost_rate
   - Higher turnover = greater signal decay

2. Discount Structure:
   - Micro accounts: Full retail rates (~50 bps)
   - Standard accounts: ~10% discount (~45 bps)
   - Professional: ~40% discount (~30 bps)
   - Institutional: ~70% discount (~15 bps)

3. Break-Even Requirements:
   - With 15% turnover and 25 bps TC: Need ~{0.07*100 + 0.15*(25/10000)*12*100:.1f}% gross return
   - With 10% turnover and 10 bps TC: Need ~{0.07*100 + 0.10*(10/10000)*12*100:.1f}% gross return

4. Practical Implications:
   - Retail investors need SSRF to generate >{(avg_turnover * 25 / 10000 * 12 + 0.07)*100:.1f}% gross alpha
   - Institutional investors need SSRF to generate >{(avg_turnover * 15 / 10000 * 12 + 0.07)*100:.1f}% gross alpha

5. Model Integration:
   - TC factor can be added as a multiplicative adjustment to predictions
   - Signal adjustment: adjusted_signal = signal * exp(-TC_factor)
   - This naturally reduces turnover in low-signal periods
''')

# Save results
results_df = pd.DataFrame(results)
print('\n' + '=' * 80)
print('FINAL RESULTS TABLE')
print('=' * 80)
print(results_df.to_string(index=False))
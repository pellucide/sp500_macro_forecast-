"""
Quick Real FRED Data Test
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '/workspace/sp500_macro_forecast')

from src.ssrf_model import SSRFModel, SSRFConfig
from src.backtesting import WalkForwardBacktester

print("Loading cached FRED data...")
df = pd.read_csv('/workspace/sp500_macro_forecast/data/fred_cache/all_fred_data_enhanced.csv', index_col=0, parse_dates=True)
df = df.dropna(thresh=df.shape[1] * 0.5)

# Create target: use yield curve slope changes as proxy
target = (df['GS10'] - df['TB3MS']).shift(-1)
df['target'] = target
df = df.dropna(subset=['target'])

# Features
feature_cols = [c for c in df.columns if c != 'target' and not c.endswith('_REGIME')]
X = df[feature_cols].ffill().bfill().fillna(0)
y = df['target']

# Create groups
groups = {f'g{i}': list(X.columns)[i::5] for i in range(5)}

print(f"Data: {len(X)} periods, {len(X.columns)} features, {X.index.min()} to {X.index.max()}")

# Quick test with step_size=3
print("\n=== Quick Test: ElasticNet (step=3) ===")

config = SSRFConfig(
    t_stat_threshold=1.5,
    n_factors=10,
    regime_window=12,
    elastic_net_alpha=0.001,
    elastic_net_l1_ratio=0.5,
    use_elastic_net_cv=False,
    model_type='elasticnet',
    prediction_scale=10.0
)

backtester = WalkForwardBacktester(
    model_class=SSRFModel,
    initial_train_window=60,
    forecast_horizon=1,
    step_size=3  # Faster
)

result = backtester.run(X, y, groups, config, verbose=False)
m = result.metrics

print(f"  Hit Ratio:     {m.get('hit_ratio', 0):.1%}")
print(f"  Sharpe:        {m.get('sharpe_ratio', 0):.4f}")
print(f"  Cumulative:    {m.get('cumulative_return', 0):.1%}")
print(f"  Max Drawdown:  {m.get('max_drawdown', 0):.1%}")
print(f"  R² OOS:        {m.get('r2_oos', 0):.4f}")

print("\n=== Quick Test: Linear (step=3) ===")

config2 = SSRFConfig(
    t_stat_threshold=1.5,
    n_factors=10,
    regime_window=12,
    model_type='linear',
    prediction_scale=10.0
)

backtester2 = WalkForwardBacktester(
    model_class=SSRFModel,
    initial_train_window=60,
    forecast_horizon=1,
    step_size=3
)

result2 = backtester2.run(X, y, groups, config2, verbose=False)
m2 = result2.metrics

print(f"  Hit Ratio:     {m2.get('hit_ratio', 0):.1%}")
print(f"  Sharpe:        {m2.get('sharpe_ratio', 0):.4f}")
print(f"  Cumulative:    {m2.get('cumulative_return', 0):.1%}")
print(f"  Max Drawdown:  {m2.get('max_drawdown', 0):.1%}")
print(f"  R² OOS:        {m2.get('r2_oos', 0):.4f}")

print("\nDone!")
"""
Test SSRF Model with Real Cached FRED Data
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, '/workspace/sp500_macro_forecast')

from src.ssrf_model import SSRFModel, SSRFConfig
from src.backtesting import WalkForwardBacktester

def load_cached_fred_data():
    """Load cached FRED data and prepare for testing."""
    # Load cached data
    df = pd.read_csv('/workspace/sp500_macro_forecast/data/fred_cache/all_fred_data_enhanced.csv', index_col=0, parse_dates=True)

    # Get S&P 500 returns (we'll need to compute from price or use proxy)
    # For now, use a proxy: combine multiple indicators as a macro factor

    # Features: all columns except target-related
    # We'll create a target as next period return

    # Drop rows with too many NaN
    df = df.dropna(thresh=df.shape[1] * 0.5)

    # Create target: use interest rate changes as proxy for market direction
    # (or we can use a combination of leading indicators)
    if 'GS10' in df.columns and 'TB3MS' in df.columns:
        # Yield curve slope as target proxy (inverted, leads market)
        target = -(df['GS10'] - df['TB3MS']).shift(-1)
    else:
        # Use inflation change as target
        target = df['CPIAUCSL'].pct_change(12).shift(-1) * 100

    df['target'] = target

    # Drop rows with NaN in target
    df = df.dropna(subset=['target'])

    # Remove target from features
    feature_cols = [c for c in df.columns if c != 'target' and not c.endswith('_REGIME')]
    X = df[feature_cols].copy()
    y = df['target'].copy()

    # Fill remaining NaN with forward fill then backward fill
    X = X.ffill().bfill().fillna(0)

    # Create groups (simplified by column prefixes)
    groups = {}
    for col in X.columns:
        prefix = col.split('_')[0][:4]
        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append(col)

    # Ensure at least 5 groups
    if len(groups) < 5:
        groups = {f'g{i}': list(X.columns)[i::5] for i in range(5)}

    print(f"Loaded {len(X)} periods with {len(X.columns)} features")
    print(f"Features in groups: {len(groups)}")
    print(f"Date range: {X.index.min()} to {X.index.max()}")

    return X, y, groups

def test_model(X, y, groups, config, name):
    """Test a model configuration."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    try:
        backtester = WalkForwardBacktester(
            model_class=SSRFModel,
            initial_train_window=60,
            forecast_horizon=1,
            step_size=1
        )

        result = backtester.run(X, y, groups, config, verbose=False)

        metrics = result.metrics
        print(f"  Hit Ratio:      {metrics.get('hit_ratio', 0):.1%}")
        print(f"  Sharpe Ratio:   {metrics.get('sharpe_ratio', 0):.4f}")
        print(f"  Cumulative:     {metrics.get('cumulative_return', 0):.1%}")
        print(f"  Max Drawdown:   {metrics.get('max_drawdown', 0):.1%}")
        print(f"  R² OOS:         {metrics.get('r2_oos', 0):.4f}")
        print(f"  Volatility:     {metrics.get('volatility', 0):.2%}")

        return {
            'name': name,
            'hit_ratio': metrics.get('hit_ratio', 0),
            'sharpe': metrics.get('sharpe_ratio', 0),
            'cumulative': metrics.get('cumulative_return', 0),
            'maxdd': metrics.get('max_drawdown', 0),
            'r2_oos': metrics.get('r2_oos', 0)
        }
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            'name': name,
            'hit_ratio': 0,
            'sharpe': 0,
            'cumulative': 0,
            'maxdd': 0,
            'r2_oos': 0
        }

def main():
    print("="*70)
    print("REAL MARKET DATA TEST (FRED Cached Data 1980-2023)")
    print("="*70)

    # Load data
    X, y, groups = load_cached_fred_data()

    results = []

    # Test 1: ElasticNet (default)
    config1 = SSRFConfig(
        t_stat_threshold=1.5,
        n_factors=10,
        regime_window=12,
        elastic_net_alpha=0.001,
        elastic_net_l1_ratio=0.5,
        use_elastic_net_cv=True,
        model_type='elasticnet',
        prediction_scale=10.0
    )
    results.append(test_model(X, y, groups, config1, "ElasticNet (CV, α=0.001)"))

    # Test 2: ElasticNet (no CV, more L2)
    config2 = SSRFConfig(
        t_stat_threshold=1.5,
        n_factors=10,
        regime_window=12,
        elastic_net_alpha=0.01,
        elastic_net_l1_ratio=0.2,
        use_elastic_net_cv=False,
        model_type='elasticnet',
        prediction_scale=10.0
    )
    results.append(test_model(X, y, groups, config2, "ElasticNet (no CV, L1=0.2)"))

    # Test 3: Linear (OLS)
    config3 = SSRFConfig(
        t_stat_threshold=1.5,
        n_factors=10,
        regime_window=12,
        model_type='linear',
        prediction_scale=10.0
    )
    results.append(test_model(X, y, groups, config3, "Linear (OLS)"))

    # Test 4: ElasticNet higher alpha
    config4 = SSRFConfig(
        t_stat_threshold=1.5,
        n_factors=10,
        regime_window=12,
        elastic_net_alpha=0.05,
        elastic_net_l1_ratio=0.5,
        use_elastic_net_cv=False,
        model_type='elasticnet',
        prediction_scale=10.0
    )
    results.append(test_model(X, y, groups, config4, "ElasticNet (α=0.05)"))

    # Summary
    print("\n\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Model':30s} {'Hit%':7s} {'Sharpe':7s} {'Cumul%':7s} {'MaxDD%':7s}")
    print("-" * 65)

    df = pd.DataFrame(results)
    df = df.sort_values('sharpe', ascending=False)

    for _, row in df.iterrows():
        print(f"{row['name']:30s} {row['hit_ratio']:6.1%} {row['sharpe']:7.3f} "
              f"{row['cumulative']:7.1%} {row['maxdd']:7.1%}")

    # Save
    df.to_csv('/workspace/sp500_macro_forecast/real_data_test_results.csv', index=False)
    print(f"\nResults saved to: /workspace/sp500_macro_forecast/real_data_test_results.csv")

if __name__ == "__main__":
    main()
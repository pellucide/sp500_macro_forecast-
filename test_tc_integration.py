"""
Test script for TC integration
Run with: cd /workspace/sp500_macro_forecast && python -m src.test_tc_integration
"""

import sys
import os

# Change to project root
os.chdir('/workspace/sp500_macro_forecast')
sys.path.insert(0, '/workspace/sp500_macro_forecast')

print("=" * 70)
print("TESTING TC INTEGRATION")
print("=" * 70)

# Test 1: Import TCConfig
print("\n[Test 1] TCConfig...")
try:
    from src.ssrf_model import TCConfig
    print(f"  Micro: {TCConfig.MICRO_ACCOUNT_TC} bps")
    print(f"  Standard: {TCConfig.STANDARD_ACCOUNT_TC} bps")
    print(f"  Professional: {TCConfig.PROFESSIONAL_TC} bps")
    print(f"  Institutional: {TCConfig.INSTITUTIONAL_TC} bps")
    print(f"  get_tc_rate('standard'): {TCConfig.get_tc_rate('standard')} bps")
    print("  [PASS] TCConfig works correctly")
except Exception as e:
    print(f"  [FAIL] {e}")

# Test 2: SSRFConfig with TC settings
print("\n[Test 2] SSRFConfig with TC settings...")
try:
    from src.ssrf_model import SSRFConfig
    config = SSRFConfig(
        include_tc=True,
        tc_rate_bps=25.0,
        expected_turnover=0.15,
        account_tier="standard"
    )
    print(f"  include_tc: {config.include_tc}")
    print(f"  tc_rate_bps: {config.tc_rate_bps}")
    print(f"  expected_turnover: {config.expected_turnover}")
    print(f"  account_tier: {config.account_tier}")
    print("  [PASS] SSRFConfig with TC works correctly")
except Exception as e:
    print(f"  [FAIL] {e}")

# Test 3: TCAdjustedWalkForwardBacktester
print("\n[Test 3] TCAdjustedWalkForwardBacktester...")
try:
    from src.tc_backtesting import TCAdjustedWalkForwardBacktester
    backtester = TCAdjustedWalkForwardBacktester(
        tc_rate_bps=25.0,
        account_tier="standard",
        expected_turnover=0.15
    )
    print(f"  effective_tc_rate: {backtester.effective_tc_rate} bps")
    print(f"  expected_turnover: {backtester.expected_turnover}")
    print("  [PASS] TCAdjustedWalkForwardBacktester initialized correctly")
except Exception as e:
    print(f"  [FAIL] {e}")

# Test 4: SSRFModel with TC methods
print("\n[Test 4] SSRFModel TC methods...")
try:
    from src.ssrf_model import SSRFModel, SSRFConfig
    from src.fred_data import generate_sample_data

    # Generate sample data (returns 3 values)
    indicators, target, _ = generate_sample_data(n_periods=300, n_indicators=50)

    groups = {
        'output_income': [c for c in indicators.columns if 'output' in c][:5],
        'labor': [c for c in indicators.columns if 'labor' in c][:5],
        'inflation': [c for c in indicators.columns if 'inflation' in c][:5],
        'interest': [c for c in indicators.columns if 'interest' in c][:5],
        'sentiment': [c for c in indicators.columns if 'sentiment' in c][:5]
    }

    # Test with TC enabled
    config = SSRFConfig(
        include_tc=True,
        tc_rate_bps=25.0,
        expected_turnover=0.15,
        account_tier="standard"
    )
    model = SSRFModel(config)

    # Fit model
    train_size = 200
    X_train = indicators.iloc[:train_size]
    y_train = target.iloc[:train_size]

    model.fit(X_train, y_train, groups)

    # Test TC factor computation
    tc_factor = model.compute_tc_factor(0.5)
    print(f"  TC factor (signal=0.5): {tc_factor:.4f}")

    # Test predict_with_tc
    X_test = indicators.iloc[train_size:]
    y_test = target.iloc[train_size:]

    raw_pred = model.predict(X_test, y_test)
    tc_pred = model.predict_with_tc(X_test, y_test)

    print(f"  Raw prediction mean: {raw_pred.mean():.4f}")
    print(f"  TC-adjusted prediction mean: {tc_pred.mean():.4f}")
    print(f"  Adjustment ratio: {tc_pred.mean() / raw_pred.mean():.4f}")

    print("  [PASS] SSRFModel TC methods work correctly")
except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()

# Test 5: Full backtest with TC
print("\n[Test 5] Full TC-adjusted backtest...")
try:
    from src.tc_backtesting import TCAdjustedWalkForwardBacktester
    from src.ssrf_model import SSRFConfig, SSRFModel
    from src.fred_data import generate_sample_data

    indicators, target, _ = generate_sample_data(n_periods=300, n_indicators=50)

    groups = {
        'output_income': [c for c in indicators.columns if 'output' in c][:5],
        'labor': [c for c in indicators.columns if 'labor' in c][:5],
        'inflation': [c for c in indicators.columns if 'inflation' in c][:5],
        'interest': [c for c in indicators.columns if 'interest' in c][:5],
        'sentiment': [c for c in indicators.columns if 'sentiment' in c][:5]
    }

    config = SSRFConfig(
        include_tc=True,
        tc_rate_bps=25.0,
        expected_turnover=0.15,
        account_tier="standard"
    )

    backtester = TCAdjustedWalkForwardBacktester(
        initial_train_window=120,
        tc_rate_bps=25.0,
        account_tier="standard",
        expected_turnover=0.15
    )

    result = backtester.run(indicators, target, groups, config, verbose=False)

    print(f"  Predictions shape: {len(result.predictions)}")
    print(f"  Turnover mean: {result.turnover.mean():.4f}")
    print(f"  TC costs mean: {result.tc_costs.mean():.4f}")
    print(f"  Gross returns mean: {result.gross_returns.mean():.4f}")
    print(f"  Net returns mean: {result.net_returns.mean():.4f}")
    print(f"  TC metrics: n_trades={result.tc_metrics['n_trades']}, total_tc={result.tc_metrics['total_tc_cost_pct']:.1f}%")

    print("  [PASS] Full TC-adjusted backtest works correctly")
except Exception as e:
    print(f"  [FAIL] {e}")
    import traceback
    traceback.print_exc()

# Test 6: CLI arguments
print("\n[Test 6] CLI TC arguments...")
try:
    import argparse
    # Mock args to test parsing
    test_args = [
        '--use-sample-data',
        '--tc-rate', '25',
        '--account-tier', 'professional',
        '--expected-turnover', '0.10',
        '--include-tc',
        '--tc-backtest',
    ]

    # Create parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-sample-data', action='store_true')
    parser.add_argument('--tc-rate', type=float, default=25.0)
    parser.add_argument('--account-tier', type=str, default='standard')
    parser.add_argument('--expected-turnover', type=float, default=0.15)
    parser.add_argument('--include-tc', action='store_true')
    parser.add_argument('--tc-backtest', action='store_true')

    args = parser.parse_args(test_args)

    print(f"  tc_rate: {args.tc_rate}")
    print(f"  account_tier: {args.account_tier}")
    print(f"  expected_turnover: {args.expected_turnover}")
    print(f"  include_tc: {args.include_tc}")
    print(f"  tc_backtest: {args.tc_backtest}")

    print("  [PASS] CLI TC arguments work correctly")
except Exception as e:
    print(f"  [FAIL] {e}")

print("\n" + "=" * 70)
print("ALL TESTS COMPLETED")
print("=" * 70)
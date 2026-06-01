#!/usr/bin/env python3
import subprocess
import sys
import os

# Ensure we're in the right directory
os.chdir('/workspace/sp500_macro_forecast')
sys.path.insert(0, '/workspace/sp500_macro_forecast')

# Step 1: Quietly install dependencies
print("Installing dependencies: fredapi, yfinance...")
try:
    import subprocess
    result = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', 'fredapi', 'yfinance', '-q'],
        capture_output=True,
        timeout=60
    )
    print("✓ Dependencies installed\n")
except Exception as e:
    print(f"Warning: {e}\n")

# Step 2: Run the backtest
print("="*70)
print("FULL BACKTEST: Elastic Net vs Ridge Comparison")
print("="*70 + "\n")

try:
    import warnings
    warnings.filterwarnings('ignore')

    from src.config import DataConfig
    from src.fred_data import FREDDataLoader, DataProcessor
    from src.ssrf_model import SSRFModel, SSRFConfig
    from src.backtesting import WalkForwardBacktester
    from src.evaluation import MetricsCalculator
    import pandas as pd

    # Load FRED data
    print("Loading FRED data...")
    loader = FREDDataLoader(api_key=DataConfig.FRED_API_KEY)
    indicators = loader.fetch_all_indicators(start_date='1959-01-01')
    target = loader.fetch_spx_returns(start_date='1959-01-01')

    # Align data
    common_idx = indicators.index.intersection(target.index)
    indicators = indicators.loc[common_idx]
    target = target.loc[common_idx]
    print(f'Data: {len(indicators)} periods, {len(indicators.columns)} features')

    # Process data
    processor = DataProcessor()
    indicators = processor.handle_missing_values(indicators, method='ffill')
    indicators = processor.winsorize_outliers(indicators)
    groups = processor.create_category_groups(indicators)
    groups_dict = {k: v.columns.tolist() for k, v in groups.items()}
    print(f'Feature groups: {len(groups_dict)}\n')

    # Run backtest for each model type
    results_summary = []

    for model_type, name in [('elasticnet', 'Elastic Net'), ('linear', 'Ridge')]:
        print(f'{"="*70}')
        print(f'Running: {name}')
        print(f'{"="*70}')

        config = SSRFConfig(
            t_stat_threshold=1.5,
            n_factors=10,
            regime_window=12,
            elastic_net_alpha=0.001,
            elastic_net_l1_ratio=0.5 if model_type == 'elasticnet' else 0.0,
            use_elastic_net_cv=True,
            use_regime_detection=True,
            model_type=model_type,
        )

        backtester = WalkForwardBacktester(
            model_class=SSRFModel,
            initial_train_window=60,
            forecast_horizon=1,
            use_ct_restriction=True,
            step_size=1
        )

        print(f"Training and backtesting {name}...")
        result = backtester.run(indicators, target, groups_dict, config, verbose=False)

        calc = MetricsCalculator()
        metrics = calc.calculate_all(result.predictions, result.actual_returns, result.benchmark_predictions)

        print(f'\n{name} Results:')
        print(f'  Hit Ratio:     {metrics["hit_ratio"]:.2%}')
        print(f'  Sharpe Ratio:  {metrics["sharpe_ratio"]:.4f}')
        print(f'  R2 OOS:        {metrics["r2_oos"]:.4f}')
        print(f'  Cumulative:    {metrics["cumulative_return"]:.2%}')
        print(f'  Max Drawdown:  {metrics["max_drawdown"]:.2%}')
        print(f'  Volatility:    {metrics["volatility"]:.2%}')
        print(f'  Ann Return:    {metrics["annualized_return"]:.2%}\n')

        results_summary.append({
            'Model': name,
            'Hit Ratio': f'{metrics["hit_ratio"]:.2%}',
            'Sharpe': f'{metrics["sharpe_ratio"]:.4f}',
            'R2 OOS': f'{metrics["r2_oos"]:.4f}',
            'Cumul Return': f'{metrics["cumulative_return"]:.2%}',
            'Max DD': f'{metrics["max_drawdown"]:.2%}',
            'Volatility': f'{metrics["volatility"]:.2%}',
            'Ann Return': f'{metrics["annualized_return"]:.2%}',
        })

    # Summary table
    print('=' * 70)
    print('COMPARISON SUMMARY')
    print('=' * 70)
    df = pd.DataFrame(results_summary)
    print(df.to_string(index=False))

    # Save results
    print('\n' + '=' * 70)
    print('SAVING RESULTS')
    print('=' * 70)
    df.to_csv('backtest_results_elasticnet_vs_ridge.csv', index=False)
    print('Results saved to: backtest_results_elasticnet_vs_ridge.csv')

    # Git operations
    print('\nCommitting results to git...')
    try:
        subprocess.run(['git', 'add', '-A'], cwd='/workspace/sp500_macro_forecast', check=True, capture_output=True)
        subprocess.run(['git', 'commit', '-m', 'Add Elastic Net vs Ridge backtest comparison results'],
                      cwd='/workspace/sp500_macro_forecast', check=True, capture_output=True)
        print('✓ Results committed to git')
    except Exception as e:
        print(f'Note: Git commit skipped or failed - {e}')

    print('\n' + '=' * 70)
    print('BACKTEST COMPLETED SUCCESSFULLY')
    print('=' * 70)

except Exception as e:
    print(f"\nError occurred: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

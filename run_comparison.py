#!/usr/bin/env python
"""
Elastic Net vs Ridge Comparison Backtest
Installs dependencies and runs full walk-forward backtest
"""
import subprocess
import sys
import os

# Set working directory
os.chdir('/workspace/sp500_macro_forecast')

# Install dependencies
print("Installing dependencies: fredapi, yfinance...")
try:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'fredapi', 'yfinance', '-q'],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("Dependencies installed successfully.\n")
except subprocess.CalledProcessError as e:
    print(f"Warning: Could not install dependencies: {e}")
    print("Attempting to continue anyway...\n")

# Now import and run the backtest
try:
    import warnings
    warnings.filterwarnings('ignore')

    from src.config import DataConfig
    from src.fred_data import FREDDataLoader, DataProcessor
    from src.ssrf_model import SSRFModel, SSRFConfig
    from src.backtesting import WalkForwardBacktester
    from src.evaluation import MetricsCalculator
    import pandas as pd

    print('=' * 70)
    print('FULL BACKTEST: Elastic Net vs Ridge Comparison')
    print('=' * 70)

    # Load FRED data
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
    print(f'Feature groups: {len(groups_dict)}')

    # Run backtest for each model type
    results_summary = []

    for model_type, name in [('elasticnet', 'Elastic Net'), ('linear', 'Ridge')]:
        print(f'\n{"="*70}')
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
        print(f'  Ann Return:    {metrics["annualized_return"]:.2%}')

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
    print('\n' + '=' * 70)
    print('COMPARISON SUMMARY')
    print('=' * 70)
    df = pd.DataFrame(results_summary)
    print(df.to_string(index=False))

    # Save results to CSV
    print('\n' + '=' * 70)
    print('Saving results...')
    print('=' * 70)
    df.to_csv('backtest_results.csv', index=False)
    print('Results saved to: backtest_results.csv')

    # Git commit
    print('\nCommitting to git...')
    subprocess.run(['git', 'add', '-A'], cwd='/workspace/sp500_macro_forecast')
    subprocess.run(['git', 'commit', '-m', 'Add Elastic Net vs Ridge backtest comparison results'],
                   cwd='/workspace/sp500_macro_forecast', capture_output=True)
    print('Git commit completed.')

except Exception as e:
    print(f"Error running backtest: {e}")
    import traceback
    traceback.print_exc()

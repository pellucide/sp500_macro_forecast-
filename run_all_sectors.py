#!/usr/bin/env python
"""
Run SSRF model for all sectors with longer historical data.
"""
import subprocess
import time
import json
from pathlib import Path

# All available sectors
SECTORS = [
    'Materials', 'Energy', 'Financials', 'Industrials', 'Technology',
    'Utilities', 'Healthcare', 'ConsumerStaples', 'ConsumerDiscretionary',
    'RealEstate', 'CommunicationServices'
]

# Starting from 2000 for longer history
START_DATE = '2000-01-01'

results = {}

for sector in SECTORS:
    print(f"\n{'='*60}")
    print(f"Running SSRF for sector: {sector}")
    print(f"{'='*60}")

    try:
        cmd = [
            'python', '-m', 'src.main',
            '--start-date', START_DATE,
            '--train-window', '60',
            '--n-factors', '10',
            '--t-stat-threshold', '0.5',
            '--regime-window', '6',
            '--sector-rotation', sector,
            '--unregularized',
            '--no-cv'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        # Extract key metrics from output
        output = result.stdout + result.stderr

        # Parse key metrics
        metrics = {}

        # Direction accuracy
        if 'Direction Accuracy:' in output:
            for line in output.split('\n'):
                if 'Direction Accuracy:' in line:
                    acc = float(line.split(':')[1].strip().replace('%', ''))
                    metrics['direction_accuracy'] = acc

        # R² OOS
        if 'Campbell-Thompson R² OOS:' in output:
            for line in output.split('\n'):
                if 'Campbell-Thompson R² OOS:' in line:
                    r2 = float(line.split(':')[1].strip())
                    metrics['r2_oos'] = r2

        # Sharpe ratio
        if 'Sharpe Ratio:' in output:
            for line in output.split('\n'):
                if 'Sharpe Ratio:' in line:
                    sharpe = float(line.split(':')[1].strip())
                    metrics['sharpe_ratio'] = sharpe

        # Strategy return
        if 'Strategy Return:' in output:
            for line in output.split('\n'):
                if 'Strategy Return:' in line:
                    ret = float(line.split(':')[1].strip().replace('%', ''))
                    metrics['strategy_return'] = ret

        # Benchmark return
        if 'Benchmark Return:' in output:
            for line in output.split('\n'):
                if 'Benchmark Return:' in line:
                    ret = float(line.split(':')[1].strip().replace('%', ''))
                    metrics['benchmark_return'] = ret

        results[sector] = metrics
        print(f"  Direction Accuracy: {metrics.get('direction_accuracy', 'N/A')}%")
        print(f"  R² OOS: {metrics.get('r2_oos', 'N/A')}")
        print(f"  Sharpe: {metrics.get('sharpe_ratio', 'N/A')}")
        print(f"  Strategy Return: {metrics.get('strategy_return', 'N/A')}%")

        time.sleep(2)  # Rate limiting

    except Exception as e:
        print(f"  ERROR: {e}")
        results[sector] = {'error': str(e)}

# Print summary
print("\n" + "="*80)
print("SUMMARY: All Sectors Sector Rotation Results (SSRF with LinearRegression)")
print("="*80)
print(f"{'Sector':<25} {'Dir Acc %':<12} {'R² OOS':<10} {'Sharpe':<10} {'Strat Ret%':<12} {'Bench Ret%':<12}")
print("-"*80)

for sector, metrics in results.items():
    if 'error' in metrics:
        print(f"{sector:<25} ERROR")
    else:
        dir_acc = metrics.get('direction_accuracy', 0)
        r2 = metrics.get('r2_oos', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        strat = metrics.get('strategy_return', 0)
        bench = metrics.get('benchmark_return', 0)
        print(f"{sector:<25} {dir_acc:<12.2f} {r2:<10.4f} {sharpe:<10.2f} {strat:<12.2f} {bench:<12.2f}")

# Save results
with open('src/backtest/all_sectors_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to src/backtest/all_sectors_results.json")
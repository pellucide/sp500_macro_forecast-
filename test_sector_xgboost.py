"""
Multi-Sector XGBoost Test Script
Tests SSRF model with XGBoost on different market sectors
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from src.main import main

# Define sectors to test
SECTORS = [
    'Technology',
    'Financials',
    'Energy',
    'Materials',
    'Health_Care',
    'Consumer_Discretionary',
    'Industrials',
    'Utilities',
    'Real_Estate'
]

def run_sector_test(model_type='xgboost', n_periods=400, n_indicators=50):
    """Run SSRF test for a single sector."""
    print(f"\n{'='*60}")
    print(f"Testing Sector Rotation with {model_type.upper()}")
    print(f"{'='*60}")

    results = {}

    for sector in SECTORS:
        print(f"\n--- Sector: {sector} ---")

        try:
            result = main([
                '--use-sample-data',
                '--n-periods', str(n_periods),
                '--n-indicators', str(n_indicators),
                '--n-factors', '10',
                '--train-window', '60',
                '--t-stat-threshold', '1.5',
                '--alpha', '0.001',
                '--l1-ratio', '0.5',
                '--model-type', model_type,
                '--no-save',
                '--no-regime',
                '--no-ct-restriction',
                '--sector-rotation', sector
            ])

            metrics = result['metrics']
            results[sector] = {
                'r2_oos': metrics['r2_oos'],
                'hit_ratio': metrics['hit_ratio'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown']
            }

            print(f"  R² OOS: {metrics['r2_oos']:.4f}")
            print(f"  Direction Accuracy: {metrics['hit_ratio']:.2%}")
            print(f"  Sharpe: {metrics['sharpe_ratio']:.4f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results[sector] = {'error': str(e)}

    return results


def print_summary(results, model_type):
    """Print summary of sector test results."""
    print(f"\n{'='*70}")
    print(f"SECTOR ROTATION TEST SUMMARY - {model_type.upper()}")
    print(f"{'='*70}")

    # Print header
    print(f"\n{'Sector':<25} {'R² OOS':>10} {'Dir%':>8} {'Sharpe':>8} {'MaxDD':>8}")
    print("-" * 70)

    valid_results = []
    for sector, metrics in results.items():
        if 'error' not in metrics:
            r2 = metrics['r2_oos']
            hit = metrics['hit_ratio'] * 100
            sharpe = metrics['sharpe_ratio']
            maxdd = metrics['max_drawdown'] * 100

            print(f"{sector:<25} {r2:>10.4f} {hit:>7.2f}% {sharpe:>8.4f} {maxdd:>7.2f}%")
            valid_results.append(metrics)
        else:
            print(f"{sector:<25} {'ERROR':>10}")

    if valid_results:
        print("-" * 70)

        # Calculate averages
        avg_r2 = np.mean([r['r2_oos'] for r in valid_results])
        avg_hit = np.mean([r['hit_ratio'] for r in valid_results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in valid_results])
        avg_maxdd = np.mean([r['max_drawdown'] for r in valid_results])

        # Find best and worst
        best_sector = max(valid_results, key=lambda x: x['r2_oos'])
        worst_sector = min(valid_results, key=lambda x: x['r2_oos'])

        # Find sector names
        best_name = [s for s, m in results.items() if m.get('r2_oos') == best_sector['r2_oos']][0]
        worst_name = [s for s, m in results.items() if m.get('r2_oos') == worst_sector['r2_oos']][0]

        print(f"\n{'AVERAGE':<25} {avg_r2:>10.4f} {avg_hit:>7.2f}% {avg_sharpe:>8.4f} {avg_maxdd:>7.2f}%")
        print(f"\nBest performing sector: {best_name} (R² = {best_sector['r2_oos']:.4f})")
        print(f"Worst performing sector: {worst_name} (R² = {worst_sector['r2_oos']:.4f})")


if __name__ == "__main__":
    print("Starting Multi-Sector XGBoost Test")
    print("="*70)

    # Run test with XGBoost
    results = run_sector_test(model_type='xgboost')
    print_summary(results, 'xgboost')

    print("\n" + "="*70)
    print("Test complete!")
"""
Full Sector Analysis with Real Market Data
Tests XGBoost model across all available sectors
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.main import main

# Define all sectors to test (using config naming convention)
SECTORS = [
    'Technology',
    'Materials',
    'Energy',
    'Financials',
    'Industrials',
    'Utilities',
    'Consumer_Staples',
    'Consumer_Discretionary',
    'Health_Care',
    'Communication'
]

# Test parameters
N_PERIODS = 200
N_INDICATORS = 30
N_FACTORS = 5
TRAIN_WINDOW = 40
MODEL_TYPE = 'xgboost'


def run_full_sector_analysis():
    """Run complete sector analysis with real market data."""
    print("\n" + "=" * 80)
    print("FULL SECTOR ANALYSIS WITH REAL MARKET DATA")
    print("=" * 80)
    print(f"Model: {MODEL_TYPE.upper()}")
    print(f"Sectors: {len(SECTORS)}")
    print("=" * 80)

    results = {}
    successful_sectors = []
    failed_sectors = []

    for i, sector in enumerate(SECTORS):
        print(f"\n[{i+1}/{len(SECTORS)}] Testing {sector}...")

        try:
            result = main([
                '--use-sample-data',  # Still use sample data for FRED
                '--n-periods', str(N_PERIODS),
                '--n-indicators', str(N_INDICATORS),
                '--n-factors', str(N_FACTORS),
                '--train-window', str(TRAIN_WINDOW),
                '--t-stat-threshold', '1.5',
                '--alpha', '0.001',
                '--l1-ratio', '0.5',
                '--model-type', MODEL_TYPE,
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
                'max_drawdown': metrics['max_drawdown'],
                'cumulative_return': metrics['cumulative_return'],
                'volatility': metrics['volatility']
            }
            successful_sectors.append(sector)

            print(f"  ✓ R² OOS: {metrics['r2_oos']:.4f}")
            print(f"    Hit Ratio: {metrics['hit_ratio']:.2%}")
            print(f"    Sharpe: {metrics['sharpe_ratio']:.4f}")

        except Exception as e:
            results[sector] = {'error': str(e)}
            failed_sectors.append((sector, str(e)))
            print(f"  ✗ ERROR: {e}")

    return results, successful_sectors, failed_sectors


def print_summary(results, successful_sectors, failed_sectors):
    """Print comprehensive summary of results."""

    print("\n" + "=" * 80)
    print("SECTOR ANALYSIS SUMMARY")
    print("=" * 80)

    # Success summary
    print(f"\n✓ Successfully tested: {len(successful_sectors)}/{len(SECTORS)} sectors")

    if failed_sectors:
        print(f"\n✗ Failed sectors:")
        for sector, error in failed_sectors:
            print(f"  - {sector}: {error}")

    if successful_sectors:
        # Create results table
        print("\n" + "=" * 80)
        print("DETAILED RESULTS")
        print("=" * 80)
        print(f"\n{'Sector':<25} {'R² OOS':>10} {'Hit Ratio':>10} {'Sharpe':>10} {'Max DD':>10} {'Cum Ret':>10}")
        print("-" * 85)

        # Sort by R² OOS
        sorted_results = sorted(
            [(s, r) for s, r in results.items() if 'r2_oos' in r],
            key=lambda x: x[1]['r2_oos'],
            reverse=True
        )

        for sector, metrics in sorted_results:
            print(f"{sector:<25} {metrics['r2_oos']:>10.4f} "
                  f"{metrics['hit_ratio']:>9.2%} "
                  f"{metrics['sharpe_ratio']:>10.4f} "
                  f"{metrics['max_drawdown']:>9.2%} "
                  f"{metrics['cumulative_return']:>9.2%}")

        # Calculate statistics
        print("-" * 85)

        r2_values = [r['r2_oos'] for r in results.values() if 'r2_oos' in r]
        hit_values = [r['hit_ratio'] for r in results.values() if 'hit_ratio' in r]
        sharpe_values = [r['sharpe_ratio'] for r in results.values() if 'sharpe_ratio' in r]

        avg_r2 = np.mean(r2_values)
        avg_hit = np.mean(hit_values)
        avg_sharpe = np.mean(sharpe_values)

        print(f"{'AVERAGE':<25} {avg_r2:>10.4f} {avg_hit:>9.2%} {avg_sharpe:>10.4f}")

        # Best and worst
        print("\n" + "=" * 80)
        print("TOP & BOTTOM SECTORS")
        print("=" * 80)

        best = sorted_results[0]
        worst = sorted_results[-1]

        print(f"\n★ Best performing: {best[0]}")
        print(f"   R² OOS: {best[1]['r2_oos']:.4f}")
        print(f"   Hit Ratio: {best[1]['hit_ratio']:.2%}")
        print(f"   Sharpe: {best[1]['sharpe_ratio']:.4f}")

        print(f"\n☆ Worst performing: {worst[0]}")
        print(f"   R² OOS: {worst[1]['r2_oos']:.4f}")
        print(f"   Hit Ratio: {worst[1]['hit_ratio']:.2%}")
        print(f"   Sharpe: {worst[1]['sharpe_ratio']:.4f}")

        # R² distribution
        print("\n" + "=" * 80)
        print("R² OOS DISTRIBUTION")
        print("=" * 80)

        positive_r2 = [v for v in r2_values if v > 0]
        print(f"\n  Positive R² sectors: {len(positive_r2)}/{len(r2_values)}")
        print(f"  Negative R² sectors: {len(r2_values) - len(positive_r2)}/{len(r2_values)}")

        if positive_r2:
            print(f"  Average positive R²: {np.mean(positive_r2):.4f}")

    print("\n" + "=" * 80)


def save_results(results, successful_sectors, failed_sectors):
    """Save results to CSV."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create detailed DataFrame
    rows = []
    for sector, metrics in results.items():
        if 'error' not in metrics:
            rows.append({
                'Sector': sector,
                'R² OOS': metrics['r2_oos'],
                'Hit Ratio': metrics['hit_ratio'],
                'Sharpe Ratio': metrics['sharpe_ratio'],
                'Max Drawdown': metrics['max_drawdown'],
                'Cumulative Return': metrics['cumulative_return'],
                'Volatility': metrics['volatility'],
                'Status': 'Success'
            })
        else:
            rows.append({
                'Sector': sector,
                'R² OOS': np.nan,
                'Hit Ratio': np.nan,
                'Sharpe Ratio': np.nan,
                'Max Drawdown': np.nan,
                'Cumulative Return': np.nan,
                'Volatility': np.nan,
                'Status': f'Error: {metrics["error"]}'
            })

    df = pd.DataFrame(rows)
    filename = f"sector_analysis_{timestamp}.csv"
    df.to_csv(filename, index=False)
    print(f"\nResults saved to: {filename}")

    return df


if __name__ == "__main__":
    print("\n" + "🎯" * 20)
    print("S&P 500 SECTOR ROTATION ANALYSIS - XGBoost")
    print("🎯" * 20)

    # Run analysis
    results, successful, failed = run_full_sector_analysis()

    # Print summary
    print_summary(results, successful, failed)

    # Save results
    df = save_results(results, successful, failed)

    print("\n✓ Full sector analysis complete!")
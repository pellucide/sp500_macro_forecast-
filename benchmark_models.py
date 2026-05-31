"""
SSRF Model Benchmark Script
Compares all model types across all sectors
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.main import main
from src.config import DataConfig

# Configuration
SECTORS = list(DataConfig.SECTOR_ETFS.keys())[:-1]  # Exclude S&P_500
MODELS = ['linear', 'xgboost', 'ensemble']  # Only include working models
# MODELS = ['linear', 'xgboost', 'random_forest', 'catboost', 'mlp', 'ensemble']  # All models

# Test parameters
N_PERIODS = 200
N_INDICATORS = 30
N_FACTORS = 5
TRAIN_WINDOW = 40
T_STAT_THRESHOLD = 1.5


def run_benchmark() -> Dict[str, Dict[str, Dict]]:
    """
    Run benchmark comparing all models across all sectors.

    Returns:
        Dictionary with results for each model and sector
    """
    print("=" * 80)
    print("SSRF MODEL BENCHMARK")
    print("=" * 80)
    print(f"Models: {MODELS}")
    print(f"Sectors: {SECTORS}")
    print(f"Periods: {N_PERIODS}, Indicators: {N_INDICATORS}, Factors: {N_FACTORS}")
    print("=" * 80)

    results = {model: {} for model in MODELS}

    total_tests = len(MODELS) * len(SECTORS)
    current_test = 0

    for model in MODELS:
        print(f"\n{'='*40}")
        print(f"Testing model: {model.upper()}")
        print(f"{'='*40}")

        for sector in SECTORS:
            current_test += 1
            print(f"[{current_test}/{total_tests}] {sector}...", end=" ")

            try:
                result = main([
                    '--use-sample-data',
                    '--n-periods', str(N_PERIODS),
                    '--n-indicators', str(N_INDICATORS),
                    '--n-factors', str(N_FACTORS),
                    '--train-window', str(TRAIN_WINDOW),
                    '--t-stat-threshold', str(T_STAT_THRESHOLD),
                    '--model-type', model,
                    '--no-save',
                    '--no-regime',
                    '--no-ct-restriction',
                    '--sector-rotation', sector
                ])

                metrics = result['metrics']
                results[model][sector] = {
                    'r2_oos': metrics['r2_oos'],
                    'hit_ratio': metrics['hit_ratio'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'max_drawdown': metrics['max_drawdown'],
                    'cumulative_return': metrics['cumulative_return']
                }
                print(f"R²={metrics['r2_oos']:.4f}, Hit={metrics['hit_ratio']:.2%}")

            except Exception as e:
                results[model][sector] = {'error': str(e)}
                print(f"ERROR: {e}")

    return results


def generate_summary(results: Dict) -> pd.DataFrame:
    """Generate summary statistics and rankings."""

    # Create DataFrame
    rows = []
    for model, sectors in results.items():
        for sector, metrics in sectors.items():
            if 'error' not in metrics:
                rows.append({
                    'Model': model,
                    'Sector': sector,
                    'R² OOS': metrics['r2_oos'],
                    'Hit Ratio': metrics['hit_ratio'],
                    'Sharpe': metrics['sharpe_ratio'],
                    'Max DD': metrics['max_drawdown'],
                    'Cum Return': metrics['cumulative_return']
                })

    df = pd.DataFrame(rows)

    return df


def print_ranking(df: pd.DataFrame):
    """Print model and sector rankings."""

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)

    # Pivot for easier comparison
    r2_pivot = df.pivot(index='Sector', columns='Model', values='R² OOS')
    hit_pivot = df.pivot(index='Sector', columns='Model', values='Hit Ratio')

    print("\n📊 R² OOS by Sector and Model:")
    print("-" * 60)
    print(r2_pivot.round(4).to_string())

    print("\n📈 Hit Ratio by Sector and Model:")
    print("-" * 60)
    print((hit_pivot * 100).round(2).astype(str) + '%')

    # Average performance
    print("\n" + "=" * 80)
    print("AVERAGE PERFORMANCE (across all sectors)")
    print("=" * 80)

    avg_stats = df.groupby('Model').agg({
        'R² OOS': 'mean',
        'Hit Ratio': 'mean',
        'Sharpe': 'mean',
        'Max DD': 'mean'
    }).round(4)

    print(avg_stats.to_string())

    # Best combinations
    print("\n" + "=" * 80)
    print("TOP 10 MODEL-SECTOR COMBINATIONS")
    print("=" * 80)

    # Sort by R² OOS
    top_r2 = df.nlargest(10, 'R² OOS')[['Model', 'Sector', 'R² OOS', 'Hit Ratio']]
    print("\nBy R² OOS:")
    for i, row in top_r2.iterrows():
        print(f"  {row['Model']:12} | {row['Sector']:25} | R²={row['R² OOS']:>8.4f} | Hit={row['Hit Ratio']:.2%}")

    # Sort by Hit Ratio
    top_hit = df.nlargest(10, 'Hit Ratio')[['Model', 'Sector', 'R² OOS', 'Hit Ratio']]
    print("\nBy Hit Ratio:")
    for i, row in top_hit.iterrows():
        print(f"  {row['Model']:12} | {row['Sector']:25} | R²={row['R² OOS']:>8.4f} | Hit={row['Hit Ratio']:.2%}")

    # Best model per sector
    print("\n" + "=" * 80)
    print("BEST MODEL PER SECTOR")
    print("=" * 80)

    best_per_sector = df.loc[df.groupby('Sector')['R² OOS'].idxmax()][['Model', 'Sector', 'R² OOS', 'Hit Ratio']]
    for i, row in best_per_sector.iterrows():
        print(f"  {row['Sector']:25} → {row['Model']:12} (R²={row['R² OOS']:.4f})")

    # Overall best
    print("\n" + "=" * 80)
    print("OVERALL BEST")
    print("=" * 80)

    best_overall = df.loc[df['R² OOS'].idxmax()]
    print(f"  Model: {best_overall['Model']}")
    print(f"  Sector: {best_overall['Sector']}")
    print(f"  R² OOS: {best_overall['R² OOS']:.4f}")
    print(f"  Hit Ratio: {best_overall['Hit Ratio']:.2%}")

    return avg_stats


def save_results(df: pd.DataFrame, avg_stats: pd.DataFrame):
    """Save results to CSV files."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save detailed results
    detail_path = f"benchmark_results_{timestamp}.csv"
    df.to_csv(detail_path, index=False)
    print(f"\nDetailed results saved to: {detail_path}")

    # Save summary
    summary_path = f"benchmark_summary_{timestamp}.csv"
    avg_stats.to_csv(summary_path)
    print(f"Summary saved to: {summary_path}")


def main_benchmark():
    """Main benchmark execution."""

    print("\n" + "🎯" * 20)
    print("SSRF MODEL BENCHMARK - Comparing all models across all sectors")
    print("🎯" * 20 + "\n")

    # Run benchmark
    results = run_benchmark()

    # Generate summary
    df = generate_summary(results)

    # Print rankings
    avg_stats = print_ranking(df)

    # Save results
    save_results(df, avg_stats)

    return df, avg_stats


if __name__ == "__main__":
    results_df, summary = main_benchmark()
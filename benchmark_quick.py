"""
SSRF Model Benchmark Script (Quick Version)
Compares key models across key sectors
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.main import main
from src.config import DataConfig

# Quick configuration - reduced for speed
MODELS = ['linear', 'xgboost', 'ensemble']
SECTORS = ['Technology', 'Materials', 'Energy', 'Utilities', 'Financials']

N_PERIODS = 150
N_INDICATORS = 25
N_FACTORS = 5
TRAIN_WINDOW = 30


def main_benchmark():
    print("=" * 70)
    print("SSRF MODEL BENCHMARK (Quick)")
    print("=" * 70)

    results = {m: {} for m in MODELS}

    for model in MODELS:
        print(f"\n--- {model.upper()} ---")
        for sector in SECTORS:
            try:
                result = main([
                    '--use-sample-data',
                    '--n-periods', str(N_PERIODS),
                    '--n-indicators', str(N_INDICATORS),
                    '--n-factors', str(N_FACTORS),
                    '--train-window', str(TRAIN_WINDOW),
                    '--t-stat-threshold', '1.5',
                    '--model-type', model,
                    '--no-save',
                    '--no-regime',
                    '--no-ct-restriction',
                    '--sector-rotation', sector
                ])
                m = result['metrics']
                results[model][sector] = {'r2': m['r2_oos'], 'hit': m['hit_ratio']}
                print(f"  {sector:<20} R²={m['r2_oos']:>8.4f}  Hit={m['hit_ratio']:.2%}")
            except Exception as e:
                results[model][sector] = {'error': str(e)}
                print(f"  {sector:<20} ERROR")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Sector':<20}", end="")
    for m in MODELS:
        print(f"{m:>12}", end="")
    print()
    print("-" * 70)

    for sector in SECTORS:
        print(f"{sector:<20}", end="")
        for model in MODELS:
            if 'r2' in results[model].get(sector, {}):
                r2 = results[model][sector]['r2']
                print(f"{r2:>12.4f}", end="")
            else:
                print(f"{'ERROR':>12}", end="")
        print()

    # Average
    print("-" * 70)
    print(f"{'AVERAGE':<20}", end="")
    for model in MODELS:
        r2_vals = [v['r2'] for v in results[model].values() if 'r2' in v]
        avg = sum(r2_vals) / len(r2_vals) if r2_vals else 0
        print(f"{avg:>12.4f}", end="")
    print()

    return results


if __name__ == "__main__":
    main_benchmark()
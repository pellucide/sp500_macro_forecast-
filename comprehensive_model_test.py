"""
Comprehensive Model Comparison Test
Uses CLI approach for reliable testing
"""

import subprocess
import re
import pandas as pd
from datetime import datetime

def run_model(model_type, alpha=None, l1_ratio=None, prediction_scale=10.0, n_factors=10, t_stat=1.5):
    """Run model via CLI and extract metrics."""

    cmd = [
        'python', '-m', 'src.main',
        '--use-sample-data',
        '--n-periods', '200',
        '--model-type', model_type,
        '--prediction-scale', str(prediction_scale),
        '--n-factors', str(n_factors),
        '--t-stat-threshold', str(t_stat),
        '--no-save'
    ]

    if alpha is not None:
        cmd.extend(['--alpha', str(alpha)])
        cmd.append('--no-cv')  # Disable CV when setting alpha manually

    if l1_ratio is not None:
        cmd.extend(['--l1-ratio', str(l1_ratio)])

    try:
        result = subprocess.run(
            cmd,
            cwd='/workspace/sp500_macro_forecast',
            capture_output=True,
            text=True,
            timeout=120
        )

        # Parse output
        output = result.stdout + result.stderr

        hit_match = re.search(r'Direction Accuracy:\s*([\d.]+)%', output)
        sharpe_match = re.search(r'Sharpe Ratio:\s*([\d.\-]+)', output)
        cumul_match = re.search(r'Strategy Return:\s*([\d.\-]+)%', output)
        maxdd_match = re.search(r'Max Drawdown:\s*([\d.\-]+)%', output)
        r2_match = re.search(r"Campbell-Thompson R² OOS:\s*([\d.\-]+)", output)

        hit = float(hit_match.group(1)) / 100 if hit_match else 0
        sharpe = float(sharpe_match.group(1)) if sharpe_match else 0
        cumul = float(cumul_match.group(1)) / 100 if cumul_match else 0
        maxdd = float(maxdd_match.group(1)) / 100 if maxdd_match else 0
        r2 = float(r2_match.group(1)) if r2_match else 0

        return {
            'hit_ratio': hit,
            'sharpe_ratio': sharpe,
            'cumulative_return': cumul,
            'max_drawdown': maxdd,
            'r2_oos': r2,
            'error': None
        }
    except Exception as e:
        return {
            'hit_ratio': 0,
            'sharpe_ratio': 0,
            'cumulative_return': 0,
            'max_drawdown': 0,
            'r2_oos': 0,
            'error': str(e)[:50]
        }

def main():
    print("="*70)
    print("COMPREHENSIVE MODEL COMPARISON TEST")
    print(f"Started: {datetime.now()}")
    print("="*70)

    results = []

    # =========================================================================
    # Part 1: Model Type Comparison
    # =========================================================================
    print("\n\n" + "="*70)
    print("PART 1: MODEL TYPE COMPARISON (with 10x scaling)")
    print("="*70)

    model_types = ['elasticnet', 'linear', 'mlp']

    for model_type in model_types:
        print(f"\nTesting {model_type}...", end=" ")
        result = run_model(model_type, prediction_scale=10.0)
        result['model_type'] = model_type
        result['config'] = 'default'
        result['prediction_scale'] = 10.0
        results.append(result)
        print(f"Hit={result['hit_ratio']:.1%}, Sharpe={result['sharpe_ratio']:.3f}, Cumul={result['cumulative_return']:.1%}")

    # =========================================================================
    # Part 2: Prediction Scale Comparison (ElasticNet)
    # =========================================================================
    print("\n\n" + "="*70)
    print("PART 2: PREDICTION SCALE COMPARISON (ElasticNet)")
    print("="*70)

    scales = [1.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0]

    for scale in scales:
        print(f"\nTesting scale={scale}x...", end=" ")
        result = run_model('elasticnet', prediction_scale=scale)
        result['model_type'] = 'elasticnet'
        result['config'] = f'scale={scale}'
        result['prediction_scale'] = scale
        results.append(result)
        print(f"Hit={result['hit_ratio']:.1%}, Sharpe={result['sharpe_ratio']:.3f}, Cumul={result['cumulative_return']:.1%}, MaxDD={result['max_drawdown']:.1%}")

    # =========================================================================
    # Part 3: ElasticNet Alpha Comparison (Ridge-like search)
    # =========================================================================
    print("\n\n" + "="*70)
    print("PART 3: ELASTICNET ALPHA GRID SEARCH (no CV)")
    print("="*70)

    alphas = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]

    for alpha in alphas:
        print(f"\nTesting alpha={alpha}...", end=" ")
        result = run_model('elasticnet', alpha=alpha, prediction_scale=10.0)
        result['model_type'] = 'elasticnet'
        result['config'] = f'alpha={alpha}'
        result['prediction_scale'] = 10.0
        results.append(result)
        print(f"Hit={result['hit_ratio']:.1%}, Sharpe={result['sharpe_ratio']:.3f}, Cumul={result['cumulative_return']:.1%}")

    # =========================================================================
    # Part 4: L1 Ratio Comparison (Ridge vs Lasso vs ElasticNet)
    # =========================================================================
    print("\n\n" + "="*70)
    print("PART 4: L1 RATIO COMPARISON (alpha=0.01)")
    print("="*70)

    l1_ratios = [0.0, 0.2, 0.5, 0.8, 1.0]
    l1_names = {0.0: 'Ridge', 1.0: 'Lasso', 0.5: 'Equal', 0.2: 'More L2', 0.8: 'More L1'}

    for l1_ratio in l1_ratios:
        print(f"\nTesting L1={l1_ratio} ({l1_names.get(l1_ratio, 'Mixed')})...", end=" ")
        result = run_model('elasticnet', alpha=0.01, l1_ratio=l1_ratio, prediction_scale=10.0)
        result['model_type'] = 'elasticnet'
        result['config'] = f'L1_ratio={l1_ratio}'
        result['prediction_scale'] = 10.0
        results.append(result)
        print(f"Hit={result['hit_ratio']:.1%}, Sharpe={result['sharpe_ratio']:.3f}, Cumul={result['cumulative_return']:.1%}")

    # =========================================================================
    # Part 5: N_Factors Comparison
    # =========================================================================
    print("\n\n" + "="*70)
    print("PART 5: N_FACTORS COMPARISON")
    print("="*70)

    n_factors_list = [5, 10, 15, 20]

    for n_factors in n_factors_list:
        print(f"\nTesting n_factors={n_factors}...", end=" ")
        result = run_model('elasticnet', n_factors=n_factors, prediction_scale=10.0)
        result['model_type'] = 'elasticnet'
        result['config'] = f'n_factors={n_factors}'
        result['prediction_scale'] = 10.0
        results.append(result)
        print(f"Hit={result['hit_ratio']:.1%}, Sharpe={result['sharpe_ratio']:.3f}, Cumul={result['cumulative_return']:.1%}")

    # =========================================================================
    # Part 6: T-Stat Threshold Comparison
    # =========================================================================
    print("\n\n" + "="*70)
    print("PART 6: T-STAT THRESHOLD COMPARISON")
    print("="*70)

    t_stats = [1.0, 1.5, 2.0, 2.5, 3.0]

    for t_stat in t_stats:
        print(f"\nTesting t_stat={t_stat}...", end=" ")
        result = run_model('elasticnet', t_stat=t_stat, prediction_scale=10.0)
        result['model_type'] = 'elasticnet'
        result['config'] = f't_stat={t_stat}'
        result['prediction_scale'] = 10.0
        results.append(result)
        print(f"Hit={result['hit_ratio']:.1%}, Sharpe={result['sharpe_ratio']:.3f}, Cumul={result['cumulative_return']:.1%}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n\n" + "="*70)
    print("FINAL SUMMARY - TOP 15 BY SHARPE RATIO")
    print("="*70)

    df = pd.DataFrame(results)
    df = df.sort_values('sharpe_ratio', ascending=False)

    print(f"\n{'Model':12s} {'Config':20s} {'Scale':6s} {'Hit%':7s} {'Sharpe':7s} {'Cumul%':7s} {'MaxDD%':7s}")
    print("-" * 75)

    for _, row in df.head(15).iterrows():
        scale_str = f"{row['prediction_scale']:.1f}x" if row['prediction_scale'] else "N/A"
        config_str = row['config'][:20] if row['config'] else "N/A"
        print(f"{row['model_type']:12s} {config_str:20s} {scale_str:6s} "
              f"{row['hit_ratio']:6.1%} {row['sharpe_ratio']:7.3f} "
              f"{row['cumulative_return']:7.1%} {row['max_drawdown']:7.1%}")

    # Save results
    output_path = '/workspace/sp500_macro_forecast/model_comparison_results.csv'
    df.to_csv(output_path, index=False)
    print(f"\n\nResults saved to: {output_path}")
    print(f"\nCompleted: {datetime.now()}")

if __name__ == "__main__":
    main()

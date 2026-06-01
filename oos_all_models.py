"""
Out-of-Sample Test for All Model Types on Real FRED Data
Walk-forward validation with real economic indicators
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("OUT-OF-SAMPLE TEST: ALL MODEL TYPES ON REAL FRED DATA")
print("="*70)

# Load cached FRED data
print("\nLoading cached FRED data (1980-2026)...")
df = pd.read_csv('/workspace/sp500_macro_forecast/data/fred_cache/all_fred_data_enhanced.csv', index_col=0, parse_dates=True)
df = df.dropna(thresh=df.shape[1] * 0.5)

print(f"Data loaded: {len(df)} periods, {len(df.columns)} columns")
print(f"Date range: {df.index.min().strftime('%Y-%m')} to {df.index.max().strftime('%Y-%m')}")

# Create target: yield curve slope direction (proxy for market risk appetite)
target = (df['GS10'] - df['TB3MS']).shift(-1)
df['target'] = target
df = df.dropna(subset=['target'])

# Features (exclude regime columns and target)
feature_cols = [c for c in df.columns if c != 'target' and not c.endswith('_REGIME')]
X = df[feature_cols].ffill().bfill().fillna(0)
y = df['target'].values

print(f"Features: {len(feature_cols)} macroeconomic indicators")
print(f"Target: Yield Curve Slope direction (next month)")
print(f"OBS: 50% = random, >55% = meaningful, >60% = strong signal")

# Walk-forward test function
def walk_forward_test(X, y, model_fn, name, train_window=60, step=6):
    """Walk-forward out-of-sample test."""
    predictions = []
    actuals = []

    for i in range(train_window, len(X) - 1, step):
        X_train = X.iloc[:i]
        y_train = y[:i]

        X_test = X.iloc[i+1:min(i+step+1, len(X))]
        y_test = y[i+1:min(i+step+1, len(y))]

        if len(X_test) < 1:
            continue

        # Scale
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = model_fn()
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
        predictions.extend(pred)
        actuals.extend(y_test)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Direction accuracy
    valid_idx = (actuals[:-1] != 0) | (predictions[:-1] != 0)
    hit_ratio = np.mean(np.sign(predictions[:-1][valid_idx]) == np.sign(actuals[1:][valid_idx])) * 100

    # Sharpe (scaled predictions)
    scaled_pred = predictions * 10
    returns = scaled_pred * actuals
    sharpe = returns.mean() / returns.std() * np.sqrt(12) if returns.std() > 0 else 0

    # Avg prediction magnitude
    avg_pred = np.mean(np.abs(predictions))

    return {
        'name': name,
        'hit_ratio': hit_ratio,
        'sharpe': sharpe,
        'n_preds': len(predictions),
        'avg_pred': avg_pred
    }

results = []

# Import models
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge, Lasso

print("\n" + "="*70)
print("TESTING MODEL TYPES")
print("="*70)

# Test 1: ElasticNet α=0.01 (default)
print("\n1. ElasticNet (α=0.01, L1=0.5)")
r = walk_forward_test(X, y, lambda: ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=10000), "ElasticNet α=0.01")
print(f"   Hit: {r['hit_ratio']:.1f}%, Sharpe: {r['sharpe']:.3f}, Avg Pred: {r['avg_pred']:.4f}")
results.append(r)

# Test 2: ElasticNet α=0.05 (higher regularization)
print("\n2. ElasticNet (α=0.05, L1=0.5)")
r = walk_forward_test(X, y, lambda: ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=10000), "ElasticNet α=0.05")
print(f"   Hit: {r['hit_ratio']:.1f}%, Sharpe: {r['sharpe']:.3f}, Avg Pred: {r['avg_pred']:.4f}")
results.append(r)

# Test 3: ElasticNet α=0.001 (lower regularization)
print("\n3. ElasticNet (α=0.001, L1=0.5)")
r = walk_forward_test(X, y, lambda: ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000), "ElasticNet α=0.001")
print(f"   Hit: {r['hit_ratio']:.1f}%, Sharpe: {r['sharpe']:.3f}, Avg Pred: {r['avg_pred']:.4f}")
results.append(r)

# Test 4: Ridge (L1=0, pure L2)
print("\n4. Ridge (α=0.01)")
r = walk_forward_test(X, y, lambda: Ridge(alpha=0.01), "Ridge α=0.01")
print(f"   Hit: {r['hit_ratio']:.1f}%, Sharpe: {r['sharpe']:.3f}, Avg Pred: {r['avg_pred']:.4f}")
results.append(r)

# Test 5: Lasso (L1=1, sparse)
print("\n5. Lasso (α=0.01)")
r = walk_forward_test(X, y, lambda: Lasso(alpha=0.01, max_iter=10000), "Lasso α=0.01")
print(f"   Hit: {r['hit_ratio']:.1f}%, Sharpe: {r['sharpe']:.3f}, Avg Pred: {r['avg_pred']:.4f}")
results.append(r)

# Test 6: Linear (OLS)
print("\n6. Linear (OLS)")
r = walk_forward_test(X, y, lambda: LinearRegression(), "Linear (OLS)")
print(f"   Hit: {r['hit_ratio']:.1f}%, Sharpe: {r['sharpe']:.3f}, Avg Pred: {r['avg_pred']:.4f}")
results.append(r)

# Test 7: High regularization ElasticNet
print("\n7. ElasticNet (α=0.1, L1=0.3)")
r = walk_forward_test(X, y, lambda: ElasticNet(alpha=0.1, l1_ratio=0.3, max_iter=10000), "ElasticNet α=0.1")
print(f"   Hit: {r['hit_ratio']:.1f}%, Sharpe: {r['sharpe']:.3f}, Avg Pred: {r['avg_pred']:.4f}")
results.append(r)

# Test 8: Very high regularization
print("\n8. ElasticNet (α=0.5, L1=0.5)")
r = walk_forward_test(X, y, lambda: ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=10000), "ElasticNet α=0.5")
print(f"   Hit: {r['hit_ratio']:.1f}%, Sharpe: {r['sharpe']:.3f}, Avg Pred: {r['avg_pred']:.4f}")
results.append(r)

# Test 9: L1 heavy (Lasso-like)
print("\n9. ElasticNet (α=0.01, L1=0.8)")
r = walk_forward_test(X, y, lambda: ElasticNet(alpha=0.01, l1_ratio=0.8, max_iter=10000), "ElasticNet L1=0.8")
print(f"   Hit: {r['hit_ratio']:.1f}%, Sharpe: {r['sharpe']:.3f}, Avg Pred: {r['avg_pred']:.4f}")
results.append(r)

# Test 10: L2 heavy (Ridge-like)
print("\n10. ElasticNet (α=0.01, L1=0.2)")
r = walk_forward_test(X, y, lambda: ElasticNet(alpha=0.01, l1_ratio=0.2, max_iter=10000), "ElasticNet L1=0.2")
print(f"   Hit: {r['hit_ratio']:.1f}%, Sharpe: {r['sharpe']:.3f}, Avg Pred: {r['avg_pred']:.4f}")
results.append(r)

# Summary
print("\n" + "="*70)
print("SUMMARY - OUT-OF-SAMPLE RESULTS")
print("="*70)
print(f"\n{'Model':25s} {'Hit%':8s} {'Sharpe':8s} {'Avg Pred':10s} {'N Preds':8s}")
print("-" * 65)

df_results = pd.DataFrame(results)
df_results = df_results.sort_values('hit_ratio', ascending=False)

for _, row in df_results.iterrows():
    print(f"{row['name']:25s} {row['hit_ratio']:7.1f}% {row['sharpe']:8.3f} {row['avg_pred']:10.4f} {row['n_preds']:8d}")

# Best model
best = df_results.iloc[0]
print("\n" + "="*70)
print(f"BEST MODEL: {best['name']}")
print(f"  Hit Ratio: {best['hit_ratio']:.1f}%")
print(f"  Sharpe: {best['sharpe']:.3f}")
print(f"  Average Predictions: {best['avg_pred']:.4f}")
print("="*70)

# Save results
df_results.to_csv('/workspace/sp500_macro_forecast/oos_model_comparison.csv', index=False)
print(f"\nResults saved to: /workspace/sp500_macro_forecast/oos_model_comparison.csv")
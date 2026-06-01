"""Comprehensive test with real FRED data using walk-forward"""
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler

print("="*70)
print("REAL MARKET DATA TEST (FRED 1980-2025)")
print("Walk-Forward Validation with Real Economic Indicators")
print("="*70)

# Load data
print("\nLoading FRED cached data...")
df = pd.read_csv('/workspace/sp500_macro_forecast/data/fred_cache/all_fred_data_enhanced.csv', index_col=0, parse_dates=True)
df = df.dropna(thresh=df.shape[1] * 0.5)

# Target: next month yield curve slope direction
target = (df['GS10'] - df['TB3MS']).shift(-1)
df['target'] = target
df = df.dropna(subset=['target'])

# Features
feature_cols = [c for c in df.columns if c != 'target' and not c.endswith('_REGIME')]
X = df[feature_cols].ffill().bfill().fillna(0)
y = df['target'].values

print(f"Data: {len(X)} periods, {len(X.columns)} features")
print(f"Date Range: {X.index.min()} to {X.index.max()}")
print(f"Target: Yield Curve Slope Direction (next month)")

# Walk-forward test
def walk_forward_test(X, y, model_fn, name, train_window=60, step=6):
    """Walk-forward out-of-sample test."""
    predictions = []
    actuals = []

    for i in range(train_window, len(X) - 1, step):
        # Train on data up to i
        X_train = X.iloc[:i]
        y_train = y[:i]

        # Test on next step(s)
        X_test = X.iloc[i+1:i+step+1] if i+step+1 <= len(X) else X.iloc[i+1:i+2]
        y_test = y[i+1:i+step+1] if i+step+1 <= len(y) else y[i+1:i+2]

        if len(X_test) < 1:
            continue

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

    # Hit ratio (same direction)
    hit_ratio = np.mean(np.sign(predictions[:-1]) == np.sign(actuals[1:])) * 100

    # Sharpe (if we scale predictions)
    scaled_pred = predictions * 10  # Same scale as CLI
    returns = scaled_pred * actuals
    sharpe = returns.mean() / returns.std() * np.sqrt(12) if returns.std() > 0 else 0
    cumulative = (1 + returns/100).prod() - 1

    return {
        'name': name,
        'hit_ratio': hit_ratio,
        'sharpe': sharpe,
        'cumulative': cumulative * 100,
        'n_preds': len(predictions)
    }

results = []

# Test 1: ElasticNet α=0.01
print("\n=== Testing ElasticNet (α=0.01) ===")
r1 = walk_forward_test(X, y, lambda: ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000), "ElasticNet α=0.01")
print(f"  Hit Ratio: {r1['hit_ratio']:.1f}%, Sharpe: {r1['sharpe']:.3f}, Cumul: {r1['cumulative']:.1f}%")
results.append(r1)

# Test 2: ElasticNet α=0.001
print("\n=== Testing ElasticNet (α=0.001) ===")
r2 = walk_forward_test(X, y, lambda: ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000), "ElasticNet α=0.001")
print(f"  Hit Ratio: {r2['hit_ratio']:.1f}%, Sharpe: {r2['sharpe']:.3f}, Cumul: {r2['cumulative']:.1f}%")
results.append(r2)

# Test 3: ElasticNet α=0.05 (more regularization)
print("\n=== Testing ElasticNet (α=0.05) ===")
r3 = walk_forward_test(X, y, lambda: ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=5000), "ElasticNet α=0.05")
print(f"  Hit Ratio: {r3['hit_ratio']:.1f}%, Sharpe: {r3['sharpe']:.3f}, Cumul: {r3['cumulative']:.1f}%")
results.append(r3)

# Test 4: Ridge (L1=0)
print("\n=== Testing Ridge (L1=0) ===")
r4 = walk_forward_test(X, y, lambda: ElasticNet(alpha=0.01, l1_ratio=0, max_iter=5000), "Ridge α=0.01")
print(f"  Hit Ratio: {r4['hit_ratio']:.1f}%, Sharpe: {r4['sharpe']:.3f}, Cumul: {r4['cumulative']:.1f}%")
results.append(r4)

# Test 5: Lasso (L1=1)
print("\n=== Testing Lasso (L1=1) ===")
r5 = walk_forward_test(X, y, lambda: ElasticNet(alpha=0.01, l1_ratio=1.0, max_iter=5000), "Lasso α=0.01")
print(f"  Hit Ratio: {r5['hit_ratio']:.1f}%, Sharpe: {r5['sharpe']:.3f}, Cumul: {r5['cumulative']:.1f}%")
results.append(r5)

# Test 6: Linear (OLS)
print("\n=== Testing Linear (OLS) ===")
r6 = walk_forward_test(X, y, lambda: LinearRegression(), "Linear (OLS)")
print(f"  Hit Ratio: {r6['hit_ratio']:.1f}%, Sharpe: {r6['sharpe']:.3f}, Cumul: {r6['cumulative']:.1f}%")
results.append(r6)

# Summary
print("\n" + "="*70)
print("SUMMARY - Walk-Forward Out-of-Sample Results")
print("="*70)
print(f"\n{'Model':25s} {'Hit%':8s} {'Sharpe':8s} {'Cumul%':8s} {'N Preds':8s}")
print("-" * 65)

df = pd.DataFrame(results)
df = df.sort_values('hit_ratio', ascending=False)
for _, row in df.iterrows():
    print(f"{row['name']:25s} {row['hit_ratio']:7.1f}% {row['sharpe']:8.3f} {row['cumulative']:7.1f}% {row['n_preds']:8d}")

print("\n" + "="*70)
print("KEY INSIGHTS:")
print("="*70)
print(f"- Data: {len(X)} monthly observations from {X.index.min().strftime('%Y')} to {X.index.max().strftime('%Y')}")
print(f"- Features: {len(X.columns)} macroeconomic indicators from FRED-MD")
print(f"- Target: Yield curve slope direction (proxy for risk appetite)")
print(f"- 50% hit ratio = random; >55% = meaningful signal; >60% = strong signal")
"""Minimal real data test"""
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler

# Load data
print("Loading FRED data...")
df = pd.read_csv('/workspace/sp500_macro_forecast/data/fred_cache/all_fred_data_enhanced.csv', index_col=0, parse_dates=True)
df = df.dropna(thresh=df.shape[1] * 0.5)

# Target: next month yield curve slope
target = (df['GS10'] - df['TB3MS']).shift(-1)
df['target'] = target
df = df.dropna(subset=['target'])

# Features
feature_cols = [c for c in df.columns if c != 'target' and not c.endswith('_REGIME')]
X = df[feature_cols].ffill().bfill().fillna(0)
y = df['target'].values

print(f"Data: {len(X)} periods, {len(X.columns)} features")

# Simple train/test split
train_end = int(len(X) * 0.7)
X_train, X_test = X.iloc[:train_end], X.iloc[train_end:]
y_train, y_test = y[:train_end], y[train_end:]

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Test 1: ElasticNet
print("\n=== ElasticNet (α=0.01) ===")
model1 = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)
model1.fit(X_train_scaled, y_train)
pred1 = model1.predict(X_test_scaled)
# Direction accuracy
hit1 = np.mean(np.sign(pred1[:-1]) == np.sign(y_test[1:])) * 100
print(f"  Hit Ratio: {hit1:.1f}%")
print(f"  Coefs non-zero: {np.sum(model1.coef_ != 0)}/{len(model1.coef_)}")

# Test 2: Linear
print("\n=== Linear (OLS) ===")
model2 = LinearRegression()
model2.fit(X_train_scaled, y_train)
pred2 = model2.predict(X_test_scaled)
hit2 = np.mean(np.sign(pred2[:-1]) == np.sign(y_test[1:])) * 100
print(f"  Hit Ratio: {hit2:.1f}%")
print(f"  Coefs non-zero: {np.sum(model2.coef_ != 0)}/{len(model2.coef_)}")

# Test 3: ElasticNet smaller alpha
print("\n=== ElasticNet (α=0.001) ===")
model3 = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000)
model3.fit(X_train_scaled, y_train)
pred3 = model3.predict(X_test_scaled)
hit3 = np.mean(np.sign(pred3[:-1]) == np.sign(y_test[1:])) * 100
print(f"  Hit Ratio: {hit3:.1f}%")
print(f"  Coefs non-zero: {np.sum(model3.coef_ != 0)}/{len(model3.coef_)}")

print("\n=== Summary ===")
print(f"Real FRED Data (1980-~2025), Monthly, {len(X)} observations")
print(f"Target: Yield Curve Slope Direction")
print(f"Feature Groups: All FRED-MD indicators (59 features)")
print(f"\n{'Model':20s} {'Hit Ratio':12s} {'Non-zero Coefs':15s}")
print("-" * 50)
print(f"{'ElasticNet α=0.01':20s} {hit1:11.1f}% {np.sum(model1.coef_ != 0):14d}")
print(f"{'Linear (OLS)':20s} {hit2:11.1f}% {np.sum(model2.coef_ != 0):14d}")
print(f"{'ElasticNet α=0.001':20s} {hit3:11.1f}% {np.sum(model3.coef_ != 0):14d}")

print("\nNote: 50% hit ratio = random, >50% = better than random")
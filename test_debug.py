#!/usr/bin/env python3
"""
SSRF Test - DEBUG VERSION
"""
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*70)
print("SSRF DEBUG TEST")
print("="*70)

# Load data
spx = yf.download('^GSPC', start='1979-01-01', end='2026-06-01', progress=False)
spx_monthly = spx['Close'].resample('ME').last()
spx_returns = spx_monthly.pct_change().dropna() * 100
spx_returns.index = spx_returns.index.normalize()

fred = pd.read_csv('data/fred_cache/all_fred_data_enhanced.csv', index_col=0, parse_dates=True)
fred = fred.dropna(thresh=fred.shape[1] * 0.5)

feature_cols = [c for c in fred.columns if c not in ['GS10', 'TB3MS'] and not c.endswith('_REGIME')]
X = fred[feature_cols].ffill().bfill().fillna(0)
X.index = X.index.normalize()

common_idx = X.index.intersection(spx_returns.index)
X_aligned = X.loc[common_idx]
spx_aligned = spx_returns.loc[common_idx]

X_arr = X_aligned.values
y_arr = spx_aligned.values.flatten()
dates = X_aligned.index

print(f"\nData: {len(X_arr)} periods")

# Test simple ElasticNet
train_window = 60

# Test at i=60
i = train_window
X_train = X_arr[:i+1]
y_train = y_arr[1:i+1]  # y[1] to y[60]
X_test = X_arr[i+1:i+2]  # X[61]
y_actual = y_arr[i+1]  # y[61]

print(f"\nTrain shape: X={X_train.shape}, y={y_train.shape}")
print(f"Test shape: X={X_test.shape}")
print(f"y_actual = {y_actual:.4f}")

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nX_train_scaled mean: {X_train_scaled.mean():.6f}, std: {X_train_scaled.std():.6f}")
print(f"X_test_scaled mean: {X_test_scaled.mean():.6f}, std: {X_test_scaled.std():.6f}")
print(f"y_train mean: {y_train.mean():.4f}, std: {y_train.std():.4f}")

# Train model
model = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)
model.fit(X_train_scaled, y_train)

print(f"\nModel coefficients mean: {model.coef_.mean():.6f}")
print(f"Model coefficients std: {model.coef_.std():.6f}")
print(f"Non-zero coefficients: {np.sum(model.coef_ != 0)}")
print(f"Intercept: {model.intercept_:.6f}")

# Predict
pred = model.predict(X_test_scaled)[0]
print(f"\nPrediction: {pred:.6f}")
print(f"Prediction sign: {np.sign(pred)}")

# Check if prediction is always 0
predictions = []
for i in range(train_window, min(train_window + 10, len(y_arr) - 1)):
    X_train = X_arr[:i+1]
    y_train = y_arr[1:i+1]
    X_test = X_arr[i+1:i+2]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)
    model.fit(X_train_s, y_train)

    pred = model.predict(X_test_s)[0]
    predictions.append(pred)
    print(f"\ni={i}: pred={pred:.6f}, coef_std={model.coef_.std():.6f}")

print(f"\n\nAll predictions: {predictions}")
print(f"All zeros? {all(p == 0 for p in predictions)}")

# Now try without the shift
print("\n" + "="*70)
print("TEST WITHOUT SHIFT (y_train = y_arr[:i])")
print("="*70)

predictions2 = []
for i in range(train_window, min(train_window + 10, len(y_arr) - 1)):
    X_train = X_arr[:i]
    y_train = y_arr[:i]  # No shift
    X_test = X_arr[i:i+1]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)
    model.fit(X_train_s, y_train)

    pred = model.predict(X_test_s)[0]
    predictions2.append(pred)
    print(f"i={i}: pred={pred:.6f}, coef_std={model.coef_.std():.6f}")

print(f"\nAll predictions (no shift): {predictions2}")
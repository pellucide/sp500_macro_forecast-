#!/usr/bin/env python3
"""
SSRF vs Ridge Regression Comparison (FIXED)
pred[t] -> actual[t+1] for BOTH direction accuracy AND P&L
NO scaling - predictions are in original return space
"""
import numpy as np
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from src.test_utils import (
    load_spx_returns, load_fred_enhanced, get_feature_columns,
    align_features_and_target, calc_metrics, bootstrap_ci, t_test,
    print_header
)

np.random.seed(42)

print_header("SSRF vs RIDGE REGRESSION COMPARISON (FIXED)")

# Load data
spx_returns = load_spx_returns()
fred = load_fred_enhanced()

feature_cols = get_feature_columns(fred)
X = fred[feature_cols].ffill().bfill().fillna(0)

X_arr, y_arr, dates = align_features_and_target(X, spx_returns)

print(f"\nData: {len(X_arr)} periods ({dates[0].strftime('%Y-%m')} to {dates[-1].strftime('%Y-%m')})")
print(f"Mean monthly return: {np.mean(y_arr):.3f}%")

# Walk-forward OOS
train_window = 60

ssrf_preds = []
ridge_preds = []
actual_returns = []

for i in range(train_window, len(X_arr)):
    train_start = i - train_window
    X_train = X_arr[train_start:i]
    y_train = y_arr[train_start:i]
    X_test = X_arr[i:i+1]

    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # SSRF (ElasticNet)
        ssrf = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)
        ssrf.fit(X_train_scaled, y_train)
        ssrf_preds.append(ssrf.predict(X_test_scaled)[0])

        # Ridge
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_scaled, y_train)
        ridge_preds.append(ridge.predict(X_test_scaled)[0])

    except:
        ssrf_preds.append(0)
        ridge_preds.append(0)

    actual_returns.append(y_arr[i])

ssrf_preds = np.array(ssrf_preds)
ridge_preds = np.array(ridge_preds)
actual_returns = np.array(actual_returns)

print(f"Predictions: {len(ssrf_preds)} periods")

# CONSISTENT calculation: pred[t] -> actual[t]
pred_next = ssrf_preds[:-1]
actual_next = actual_returns[:-1]
ridge_pred_next = ridge_preds[:-1]

# Baselines
momentum_pred = np.zeros(len(actual_returns))
momentum_pred[1:] = np.sign(actual_returns[:-1])
hist_mean = np.mean(actual_returns)
hist_pred = np.full(len(actual_returns), hist_mean)
spx_total = (np.prod(1 + actual_returns/100) - 1) * 100

# Calculate metrics
print_header("RESULTS (pred[t] -> actual[t], NO SCALING)")

ssrf_metrics = calc_metrics(ssrf_preds[:-1], actual_returns[:-1])
ridge_metrics = calc_metrics(ridge_preds[:-1], actual_returns[:-1])
momentum_metrics = calc_metrics(momentum_pred[:-1], actual_returns[:-1])
hist_metrics = calc_metrics(hist_pred[:-1], actual_returns[:-1])

print(f"\n{'Model':<15} {'Hit%':>8} {'Total P&L':>12} {'Ann.Ret':>10} {'Sharpe':>8}")
print("-"*60)
print(f"{'SSRF':<15} {ssrf_metrics['hit_ratio']:>8.1f} {ssrf_metrics['total_pnl']:>12.2f} {ssrf_metrics['ann_return']:>10.2f} {ssrf_metrics['sharpe']:>8.3f}")
print(f"{'Ridge':<15} {ridge_metrics['hit_ratio']:>8.1f} {ridge_metrics['total_pnl']:>12.2f} {ridge_metrics['ann_return']:>10.2f} {ridge_metrics['sharpe']:>8.3f}")
print("-"*60)
print(f"{'Momentum':<15} {momentum_metrics['hit_ratio']:>8.1f} {momentum_metrics['total_pnl']:>12.2f} {momentum_metrics['ann_return']:>10.2f} {momentum_metrics['sharpe']:>8.3f}")
print(f"{'Hist Mean':<15} {hist_metrics['hit_ratio']:>8.1f} {hist_metrics['total_pnl']:>12.2f} {hist_metrics['ann_return']:>10.2f} {hist_metrics['sharpe']:>8.3f}")
print(f"{'SPX B&H':<15} {'N/A':>8} {spx_total:>12.1f} {'N/A':>10} {'N/A':>8}")

# Statistical significance
print_header("STATISTICAL SIGNIFICANCE")

for name, preds in [('SSRF', ssrf_preds), ('Ridge', ridge_preds)]:
    pred_n = preds[:-1]
    actual_n = actual_returns[:-1]
    pnl = pred_n * actual_n

    ci_low, ci_high = bootstrap_ci(pred_n, actual_n)
    t_stat, p_t = t_test(pred_n, actual_n)

    print(f"\n{name}:")
    print(f"  95% CI for Sharpe: [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"  t-test p-value: {p_t:.4f}")
    sig = "SIGNIFICANT" if p_t < 0.05 and ci_low > 0 else "NOT SIGNIFICANT"
    print(f"  {sig}")

# Verdict
print_header("VERDICT")
for m in [ssrf_metrics, ridge_metrics, momentum_metrics, hist_metrics]:
    verdict = "PASS" if m['hit_ratio'] > 50 and m['sharpe'] > 0 else "FAIL"
    print(f"{m['name']:<15}: {verdict}" if 'name' in m else f"Model: {verdict}")

# Note: calc_metrics doesn't return 'name', so we inline names
for name, m in [('SSRF', ssrf_metrics), ('Ridge', ridge_metrics),
                 ('Momentum', momentum_metrics), ('Hist Mean', hist_metrics)]:
    verdict = "PASS" if m['hit_ratio'] > 50 and m['sharpe'] > 0 else "FAIL"
    print(f"{name:<15}: {verdict}")

print("\n" + "="*70)

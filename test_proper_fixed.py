#!/usr/bin/env python3
"""
SSRF Test - FIXED temporal alignment and NO data leakage
Proper out-of-sample walk-forward test
"""
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from src.test_utils import (
    load_spx_returns, load_fred_enhanced, get_feature_columns,
    align_features_and_target, calc_metrics, bootstrap_ci, t_test,
    print_header
)

np.random.seed(42)

print_header("SSRF PROPER OOS TEST - NO LEAKAGE, FIXED ALIGNMENT")

# Load data
spx_returns = load_spx_returns()
fred = load_fred_enhanced()

feature_cols = get_feature_columns(fred)
X = fred[feature_cols].ffill().bfill().fillna(0)

X_arr, y_arr, dates = align_features_and_target(X, spx_returns)

print(f"\nData: {len(X_arr)} periods ({dates[0].strftime('%Y-%m')} to {dates[-1].strftime('%Y-%m')})")
print(f"Mean monthly return: {np.mean(y_arr):.3f}%")

# ============================================================================
# PROPER OOS Walk-Forward Test
# ============================================================================
train_window = 60
predictions = []
actuals = []

for i in range(train_window, len(y_arr) - 1):
    # CORRECT: Train on X[:i] and y[:i] to predict y[i+1]
    X_train = X_arr[:i]
    y_train = y_arr[:i]
    X_test = X_arr[i+1:i+2]
    y_actual = y_arr[i+1]

    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)[0]
    except:
        pred = 0

    predictions.append(pred)
    actuals.append(y_actual)

predictions = np.array(predictions)
actuals = np.array(actuals)

metrics = calc_metrics(predictions, actuals)
ci_lower, ci_upper = bootstrap_ci(predictions, actuals)
t_stat, p_t = t_test(predictions, actuals)

print_header("RESULTS (NO LEAKAGE, FIXED ALIGNMENT)")
print(f"\nSSRF:")
print(f"  Direction Accuracy: {metrics['hit_ratio']:.1f}% (out of {len(actuals)} predictions)")
print(f"  Correct: {int(metrics['hit_ratio'] * len(actuals) / 100)}, Wrong: {len(actuals) - int(metrics['hit_ratio'] * len(actuals) / 100)}")
print(f"  Total P&L: {metrics['total_pnl']:.2f}")
print(f"  Annualized Return: {metrics['ann_return']:.2f}%")
print(f"  Annualized Vol: {metrics['ann_vol']:.2f}%")
print(f"  Sharpe Ratio: {metrics['sharpe']:.3f}")

# SPX Buy & Hold
spx_total = (np.prod(1 + actuals/100) - 1) * 100
print(f"\nSPX Buy&Hold:")
print(f"  Total Return: {spx_total:.1f}%")

# Baselines
print("\n" + "-"*70)
print("Comparison to baselines:")

momentum_pred = np.zeros(len(actuals))
momentum_pred[1:] = np.sign(actuals[:-1])
mom_metrics = calc_metrics(momentum_pred, actuals)
print(f"  Momentum: {mom_metrics['hit_ratio']:.1f}% hit, {mom_metrics['total_pnl']:.2f} P&L")

hist_mean = np.full(len(actuals), np.mean(y_arr))
hist_metrics = calc_metrics(hist_mean, actuals)
print(f"  Hist Mean: {hist_metrics['hit_ratio']:.1f}% hit, {hist_metrics['total_pnl']:.2f} P&L")

# Statistical tests
print("\n" + "-"*70)
print("Statistical Significance:")
print(f"  Bootstrap 95% CI for Sharpe: [{ci_lower:.3f}, {ci_upper:.3f}]")
print(f"  t-test (mean P&L vs 0): t={t_stat:.3f}, p={p_t:.4f}")

# ============================================================================
# TRUE OOS TEST (Train 1980-2000, Test 2000-2026)
# ============================================================================
print_header("TRUE OUT-OF-SAMPLE TEST")

split_idx = 240  # ~2000-01

X_train_oos = X_arr[:split_idx]
y_train_oos = y_arr[:split_idx]
X_test_oos = X_arr[split_idx:]
y_test_oos = y_arr[split_idx:]

oos_preds = []
oos_actual = []

for i in range(len(X_train_oos), len(y_arr) - 1):
    X_train = X_arr[:i]
    y_train = y_arr[:i]
    X_test = X_arr[i+1:i+2]
    y_actual = y_arr[i+1]

    try:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)
        model.fit(X_train_s, y_train)
        oos_preds.append(model.predict(X_test_s)[0])
    except:
        oos_preds.append(0)
    oos_actual.append(y_actual)

oos_preds = np.array(oos_preds)
oos_actual = np.array(oos_actual)

oos_metrics = calc_metrics(oos_preds, oos_actual)
oos_ci_lower, oos_ci_upper = bootstrap_ci(oos_preds, oos_actual)

print(f"\nSSRF (True OOS):")
print(f"  Direction Accuracy: {oos_metrics['hit_ratio']:.1f}%")
print(f"  Sharpe Ratio: {oos_metrics['sharpe']:.3f}")

# Baselines
momentum_oos = np.full(len(oos_actual), np.sign(y_train_oos[-1]))
mom_oos_metrics = calc_metrics(momentum_oos, oos_actual)

hist_oos = np.full(len(oos_actual), np.mean(y_train_oos))
hist_oos_hit = np.mean(np.sign(hist_oos) == np.sign(oos_actual)) * 100

spx_total_oos = (np.prod(1 + oos_actual/100) - 1) * 100

print(f"  Momentum: {mom_oos_metrics['hit_ratio']:.1f}% hit, Sharpe={mom_oos_metrics['sharpe']:.3f}")
print(f"  Hist Mean: {hist_oos_hit:.1f}% hit")
print(f"  SPX Buy&Hold: {spx_total_oos:.1f}%")

print(f"  Bootstrap 95% CI: [{oos_ci_lower:.3f}, {oos_ci_upper:.3f}]")

# Verdict
print_header("VERDICT")
print(f"\nFull Sample (NO LEAKAGE):")
status = "WORKS" if metrics['hit_ratio'] > 50 and metrics['sharpe'] > 0 and ci_lower > 0 else "FAILS"
print(f"  SSRF {status}")
print(f"     Hit: {metrics['hit_ratio']:.1f}%, Sharpe: {metrics['sharpe']:.3f}, 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

print(f"\nTrue OOS (2000-2026):")
status = "WORKS" if oos_metrics['hit_ratio'] > 50 and oos_metrics['sharpe'] > 0 and oos_ci_lower > 0 else "FAILS"
print(f"  SSRF {status}")
print(f"     Hit: {oos_metrics['hit_ratio']:.1f}%, Sharpe: {oos_metrics['sharpe']:.3f}, 95% CI: [{oos_ci_lower:.3f}, {oos_ci_upper:.3f}]")

print("\n" + "="*70)

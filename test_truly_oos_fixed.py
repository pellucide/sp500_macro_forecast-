#!/usr/bin/env python3
"""
SSRF - TRULY OUT-OF-SAMPLE TEST (FIXED)
Proper walk-forward with correct temporal alignment
"""
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from src.test_utils import (
    load_spx_returns, load_fred_enhanced, get_feature_columns,
    align_features_and_target, calc_metrics, bootstrap_ci,
    print_header
)

np.random.seed(42)

print_header("SSRF TRULY OUT-OF-SAMPLE TEST (FIXED)")

# Load data
spx_returns = load_spx_returns()
fred = load_fred_enhanced()

feature_cols = get_feature_columns(fred)
X = fred[feature_cols].ffill().bfill().fillna(0)

X_arr, y_arr, dates = align_features_and_target(X, spx_returns)

print(f"\nData: {len(X_arr)} periods ({dates[0].strftime('%Y-%m')} to {dates[-1].strftime('%Y-%m')})")

# ============================================================================
# TEST 1: Expanding window walk-forward
# ============================================================================
print_header("TEST 1: EXPANDING WINDOW")

train_window = 60

exp_preds = []
exp_actual = []

for i in range(train_window, len(X_arr)):
    X_train = X_arr[:i]
    y_train = y_arr[:i]
    X_test = X_arr[i:i+1]
    y_actual = y_arr[i]

    try:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)
        model.fit(X_train_s, y_train)
        exp_preds.append(model.predict(X_test_s)[0])
    except:
        exp_preds.append(0)
    exp_actual.append(y_actual)

exp_preds = np.array(exp_preds)
exp_actual = np.array(exp_actual)
exp_metrics = calc_metrics(exp_preds, exp_actual)

print(f"SSRF (Expanding): Hit={exp_metrics['hit_ratio']:.1f}%, Sharpe={exp_metrics['sharpe']:.3f}")

# ============================================================================
# TEST 2: FIXED window walk-forward
# ============================================================================
print_header("TEST 2: FIXED WINDOW (60 months rolling)")

fix_preds = []
fix_actual = []

for i in range(train_window, len(X_arr)):
    train_start = i - train_window
    X_train = X_arr[train_start:i]
    y_train = y_arr[train_start:i]
    X_test = X_arr[i:i+1]
    y_actual = y_arr[i]

    try:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)
        model.fit(X_train_s, y_train)
        fix_preds.append(model.predict(X_test_s)[0])
    except:
        fix_preds.append(0)
    fix_actual.append(y_actual)

fix_preds = np.array(fix_preds)
fix_actual = np.array(fix_actual)
fix_metrics = calc_metrics(fix_preds, fix_actual)

print(f"SSRF (Fixed): Hit={fix_metrics['hit_ratio']:.1f}%, Sharpe={fix_metrics['sharpe']:.3f}")

# ============================================================================
# TEST 3: TRUE OOS (Train 1980-2000, Test 2000-2026)
# ============================================================================
print_header("TEST 3: TRUE OOS (Train 1980-2000, Test 2000-2026)")

split_idx = 240  # ~2000-01

oos_preds = []
oos_actual = []

for i in range(split_idx, len(y_arr) - 1):
    X_train = X_arr[:i+1]
    y_train = y_arr[:i+1]
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

print(f"SSRF (True OOS): Hit={oos_metrics['hit_ratio']:.1f}%, Sharpe={oos_metrics['sharpe']:.3f}")

# Baselines
momentum_oos = np.full(len(oos_actual), np.sign(y_arr[split_idx - 1]))
mom_metrics = calc_metrics(momentum_oos, oos_actual)

hist_oos = np.full(len(oos_actual), np.mean(y_arr[:split_idx]))
hist_hit = np.mean(np.sign(hist_oos) == np.sign(oos_actual)) * 100

spx_total = (np.prod(1 + oos_actual/100) - 1) * 100

print(f"Momentum (True OOS): Hit={mom_metrics['hit_ratio']:.1f}%, Sharpe={mom_metrics['sharpe']:.3f}")
print(f"Hist Mean (True OOS): Hit={hist_hit:.1f}%")
print(f"SPX Buy&Hold (True OOS): {spx_total:.1f}%")

# Bootstrap CI
print("\n--- Bootstrap 95% CI for True OOS ---")
ci_low, ci_high = bootstrap_ci(oos_preds, oos_actual)
print(f"SSRF 95% CI: [{ci_low:.3f}, {ci_high:.3f}]")

# ============================================================================
# SUMMARY
# ============================================================================
print_header("SUMMARY")
print(f"{'Test':<25} {'Hit%':>8} {'Sharpe':>8}")
print("-" * 45)
print(f"{'Expanding Window':<25} {exp_metrics['hit_ratio']:>8.1f} {exp_metrics['sharpe']:>8.3f}")
print(f"{'Fixed Window (60m)':<25} {fix_metrics['hit_ratio']:>8.1f} {fix_metrics['sharpe']:>8.3f}")
print(f"{'True OOS (2000-2026)':<25} {oos_metrics['hit_ratio']:>8.1f} {oos_metrics['sharpe']:>8.3f}")
print(f"{'Momentum (True OOS)':<25} {mom_metrics['hit_ratio']:>8.1f} {mom_metrics['sharpe']:>8.3f}")
print(f"{'SPX Buy&Hold':<25} {'N/A':>8} {spx_total:>8.1f}%")

print_header("VERDICT")

if oos_metrics['hit_ratio'] > 50 and oos_metrics['sharpe'] > 0 and ci_low > 0:
    print("SSRF PASSES TRUE OOS TEST")
elif oos_metrics['hit_ratio'] > 50 and oos_metrics['sharpe'] > 0:
    print("SSRF MARGINALLY PASSES - CI includes zero")
else:
    print("SSRF FAILS TRUE OOS TEST")
    print(f"  Hit: {oos_metrics['hit_ratio']:.1f}% (vs 50% random)")
    print(f"  Sharpe: {oos_metrics['sharpe']:.3f}")
    print(f"  95% CI: [{ci_low:.3f}, {ci_high:.3f}]")

print("\n" + "=" * 70)

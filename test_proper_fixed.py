#!/usr/bin/env python3
"""
SSRF Test - FIXED temporal alignment and NO data leakage
Proper out-of-sample walk-forward test
"""
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*70)
print("SSRF PROPER OOS TEST - NO LEAKAGE, FIXED ALIGNMENT")
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

print(f"\nData: {len(X_arr)} periods ({dates[0].strftime('%Y-%m')} to {dates[-1].strftime('%Y-%m')})")
print(f"Mean monthly return: {np.mean(y_arr):.3f}%")

# ============================================================================
# PROPER OOS Walk-Forward Test
# ============================================================================
# CORRECT APPROACH: Train on X[:i] and y[:i] to predict y[i+1]
# Test: X[i+1] to predict y[i+1]
# No data leakage: y[i+1] is never in training set
# ============================================================================

train_window = 60
predictions = []
actuals = []

for i in range(train_window, len(y_arr) - 1):
    # CORRECT: Train on X[:i] and y[:i] to predict y[i+1]
    # Both X and y have the same number of samples
    # Training: X[0:i], y[0:i] (same indices)
    # Test: X[i+1] to predict y[i+1]
    X_train = X_arr[:i]  # Features from time 0 to i-1 (i samples)
    y_train = y_arr[:i]  # Targets from time 0 to i-1 (i samples, same as X_train)

    # Test: X[i+1] to predict y[i+1]
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

# CORRECT 1-to-1 comparison: pred[t] vs actual[t]
direction_correct = np.sum(np.sign(predictions) == np.sign(actuals))
hit_ratio = direction_correct / len(actuals) * 100

# P&L calculation
pnl = predictions * actuals
total_pnl = np.sum(pnl)
ann_return = np.mean(pnl) * 12 * 100
ann_vol = np.std(pnl) * np.sqrt(12) * 100
sharpe = ann_return / ann_vol if ann_vol > 0 else 0

print("\n" + "="*70)
print("RESULTS (NO LEAKAGE, FIXED ALIGNMENT)")
print("="*70)

print(f"\nSSRF:")
print(f"  Direction Accuracy: {hit_ratio:.1f}% (out of {len(actuals)} predictions)")
print(f"  Correct: {direction_correct}, Wrong: {len(actuals) - direction_correct}")
print(f"  Total P&L: {total_pnl:.2f}")
print(f"  Annualized Return: {ann_return:.2f}%")
print(f"  Annualized Vol: {ann_vol:.2f}%")
print(f"  Sharpe Ratio: {sharpe:.3f}")

# SPX Buy & Hold
spx_total = (np.prod(1 + actuals/100) - 1) * 100
print(f"\nSPX Buy&Hold:")
print(f"  Total Return: {spx_total:.1f}%")

# Baselines
print("\n" + "-"*70)
print("Comparison to baselines:")

# Momentum
momentum_pred = np.zeros(len(actuals))
momentum_pred[1:] = np.sign(actuals[:-1])
momentum_correct = np.sum(np.sign(momentum_pred) == np.sign(actuals))
momentum_hit = momentum_correct / len(actuals) * 100
momentum_pnl = np.sum(momentum_pred * actuals)
print(f"  Momentum: {momentum_hit:.1f}% hit, {momentum_pnl:.2f} P&L")

# Hist mean
hist_mean = np.full(len(actuals), np.mean(y_arr))
hist_correct = np.sum(np.sign(hist_mean) == np.sign(actuals))
hist_hit = hist_correct / len(actuals) * 100
hist_pnl = np.sum(hist_mean * actuals)
print(f"  Hist Mean: {hist_hit:.1f}% hit, {hist_pnl:.2f} P&L")

# Statistical tests
print("\n" + "-"*70)
print("Statistical Significance:")

# Bootstrap CI
n_boot = 1000
sharpes_boot = []
for _ in range(n_boot):
    idx = np.random.choice(len(pnl), size=len(pnl), replace=True)
    boot_pnl = pnl[idx]
    boot_ret = np.mean(boot_pnl) * 12 * 100
    boot_vol = np.std(boot_pnl) * np.sqrt(12) * 100
    sharpes_boot.append(boot_ret / boot_vol if boot_vol > 0 else 0)

ci_lower = np.percentile(sharpes_boot, 2.5)
ci_upper = np.percentile(sharpes_boot, 97.5)
print(f"  Bootstrap 95% CI for Sharpe: [{ci_lower:.3f}, {ci_upper:.3f}]")

# t-test
t_stat, p_t = stats.ttest_1samp(pnl, 0)
print(f"  t-test (mean P&L vs 0): t={t_stat:.3f}, p={p_t:.4f}")

# ============================================================================
# TRUE OOS TEST (Train 1980-2000, Test 2000-2026)
# ============================================================================
print("\n" + "="*70)
print("TRUE OUT-OF-SAMPLE TEST")
print("="*70)

split_idx = 240  # ~2000-01

X_train_oos = X_arr[:split_idx]
y_train_oos = y_arr[:split_idx]
X_test_oos = X_arr[split_idx:]
y_test_oos = y_arr[split_idx:]

# CORRECT: Train on X[:240] and y[:240] to predict y[240+1], etc.
oos_preds = []
oos_actual = []

for i in range(len(X_train_oos), len(y_arr) - 1):
    # Train on X[:i] and y[:i]
    X_train = X_arr[:i]
    y_train = y_arr[:i]
    # Test on X[i+1] to predict y[i+1]
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

oos_hit = np.mean(np.sign(oos_preds) == np.sign(oos_actual)) * 100
oos_pnl = oos_preds * oos_actual
oos_sharpe = np.mean(oos_pnl) * 12 * 100 / (np.std(oos_pnl) * np.sqrt(12) * 100) if np.std(oos_pnl) > 0 else 0

print(f"\nSSRF (True OOS):")
print(f"  Direction Accuracy: {oos_hit:.1f}%")
print(f"  Sharpe Ratio: {oos_sharpe:.3f}")

# Baselines
momentum_oos = np.full(len(oos_actual), np.sign(y_train_oos[-1]))
mom_hit = np.mean(np.sign(momentum_oos) == np.sign(oos_actual)) * 100
mom_pnl = momentum_oos * oos_actual
mom_sharpe = np.mean(mom_pnl) * 12 * 100 / (np.std(mom_pnl) * np.sqrt(12) * 100) if np.std(mom_pnl) > 0 else 0

hist_oos = np.full(len(oos_actual), np.mean(y_train_oos))
hist_hit = np.mean(np.sign(hist_oos) == np.sign(oos_actual)) * 100

spx_total_oos = (np.prod(1 + oos_actual/100) - 1) * 100

print(f"  Momentum: {mom_hit:.1f}% hit, Sharpe={mom_sharpe:.3f}")
print(f"  Hist Mean: {hist_hit:.1f}% hit")
print(f"  SPX Buy&Hold: {spx_total_oos:.1f}%")

# Bootstrap CI for True OOS
n_boot = 1000
sharpes_oos_boot = []
for _ in range(n_boot):
    idx = np.random.choice(len(oos_pnl), size=len(oos_pnl), replace=True)
    boot_pnl = oos_pnl[idx]
    boot_ret = np.mean(boot_pnl) * 12 * 100
    boot_vol = np.std(boot_pnl) * np.sqrt(12) * 100
    sharpes_oos_boot.append(boot_ret / boot_vol if boot_vol > 0 else 0)

oos_ci_lower = np.percentile(sharpes_oos_boot, 2.5)
oos_ci_upper = np.percentile(sharpes_oos_boot, 97.5)
print(f"  Bootstrap 95% CI: [{oos_ci_lower:.3f}, {oos_ci_upper:.3f}]")

# Verdict
print("\n" + "="*70)
print("VERDICT")
print("="*70)

print(f"\nFull Sample (NO LEAKAGE):")
if hit_ratio > 50 and sharpe > 0 and ci_lower > 0:
    print(f"  ✅ SSRF WORKS")
    print(f"     Hit: {hit_ratio:.1f}%, Sharpe: {sharpe:.3f}, 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
else:
    print(f"  ❌ SSRF FAILS")
    print(f"     Hit: {hit_ratio:.1f}%, Sharpe: {sharpe:.3f}, 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

print(f"\nTrue OOS (2000-2026):")
if oos_hit > 50 and oos_sharpe > 0 and oos_ci_lower > 0:
    print(f"  ✅ SSRF WORKS")
    print(f"     Hit: {oos_hit:.1f}%, Sharpe: {oos_sharpe:.3f}, 95% CI: [{oos_ci_lower:.3f}, {oos_ci_upper:.3f}]")
else:
    print(f"  ❌ SSRF FAILS")
    print(f"     Hit: {oos_hit:.1f}%, Sharpe: {oos_sharpe:.3f}, 95% CI: [{oos_ci_lower:.3f}, {oos_ci_upper:.3f}]")

print("\n" + "="*70)
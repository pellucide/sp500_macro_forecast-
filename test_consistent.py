#!/usr/bin/env python3
"""
SSRF Test with CONSISTENT calculations
pred[t] → actual[t+1] for BOTH direction accuracy AND P&L
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

PREDICTION_SCALE = 10.0

print("="*70)
print(f"SSRF CONSISTENT TEST - SCALE={PREDICTION_SCALE}")
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
print(f"Std monthly return: {np.std(y_arr):.3f}%")

# Walk-forward OOS
train_window = 60
start_idx = train_window

model_preds = []
actual_returns = []  # actual[t] = return at time t

for i in range(start_idx, len(X_arr)):
    train_start = i - train_window
    X_train = X_arr[train_start:i]
    y_train = y_arr[train_start:i]
    X_test = X_arr[i:i+1]

    # Scale target
    y_train_scaled = y_train * PREDICTION_SCALE

    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=10000)
        model.fit(X_train_scaled, y_train_scaled)

        pred = model.predict(X_test_scaled)[0]
        model_preds.append(pred)
    except:
        model_preds.append(0)

    actual_returns.append(y_arr[i])

model_preds = np.array(model_preds)
actual_returns = np.array(actual_returns)

# CONSISTENT calculation: pred[t] → actual[t+1]
# pred[0] predicts for period 1, compare with actual[1]
# pred[1] predicts for period 2, compare with actual[2]
pred_for_next = model_preds[:-1]  # pred[0] to pred[n-2]
actual_next = actual_returns[1:]   # actual[1] to actual[n-1]

# Direction accuracy: did we predict the right direction?
direction_correct = np.sum(np.sign(pred_for_next) == np.sign(actual_next))
hit_ratio = direction_correct / len(actual_next) * 100

# P&L: what is our profit/loss from following predictions?
# We take position pred[t] at time t, earn pred[t] * actual[t+1]
pnl = pred_for_next * actual_next
total_pnl = np.sum(pnl) * 100  # percentage
ann_return = np.mean(pnl) * 12 * 100
ann_vol = np.std(pnl) * np.sqrt(12) * 100
sharpe = ann_return / ann_vol if ann_vol > 0 else 0

print("\n" + "="*70)
print("CONSISTENT METRICS (pred[t] → actual[t+1])")
print("="*70)

print(f"\nSSRF (Scale={PREDICTION_SCALE}):")
print(f"  Direction Accuracy: {hit_ratio:.1f}% (out of {len(actual_next)} predictions)")
print(f"  Correct: {direction_correct}, Wrong: {len(actual_next) - direction_correct}")
print(f"  Total P&L: {total_pnl:.1f}%")
print(f"  Annualized Return: {ann_return:.1f}%")
print(f"  Annualized Vol: {ann_vol:.1f}%")
print(f"  Sharpe Ratio: {sharpe:.3f}")

# SPX Buy & Hold for comparison
spx_total = (np.prod(1 + actual_returns/100) - 1) * 100
print(f"\nSPX Buy&Hold (buy at t=0, sell at t={len(actual_returns)-1}):")
print(f"  Total Return: {spx_total:.1f}%")

# Compare to baselines
print("\n" + "-"*70)
print("Comparison to baselines:")

# Random baseline (expected 50% accuracy)
random_hit = 50.0
random_pnl_mean = 0  # On average, random wins 0
print(f"  Random: ~{random_hit:.1f}% hit, ~{random_pnl_mean:.1f}% P&L")

# Momentum baseline
momentum_pred = np.sign(actual_returns[:-1])  # Predict today = yesterday's direction
momentum_correct = np.sum(np.sign(momentum_pred) == np.sign(actual_returns[1:]))
momentum_hit = momentum_correct / len(actual_returns[1:]) * 100
momentum_pnl = np.sum(momentum_pred * actual_returns[1:]) * 100
print(f"  Momentum: {momentum_hit:.1f}% hit, {momentum_pnl:.1f}% P&L")

# Hist mean baseline
hist_mean = np.mean(actual_returns[:-1])
hist_pred = np.full(len(actual_returns)-1, hist_mean)
hist_correct = np.sum(np.sign(hist_pred) == np.sign(actual_returns[1:]))
hist_hit = hist_correct / len(actual_returns[1:]) * 100
hist_pnl = np.sum(hist_pred * actual_returns[1:]) * 100
print(f"  Hist Mean: {hist_hit:.1f}% hit, {hist_pnl:.1f}% P&L")

# Statistical tests
print("\n" + "-"*70)
print("Statistical Significance (CONSISTENT):")

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

# Final verdict
print("\n" + "="*70)
print("VERDICT")
print("="*70)

if hit_ratio > 50 and sharpe > 0 and ci_lower > 0:
    print(f"\n✅ SSRF WORKS")
    print(f"  Hit: {hit_ratio:.1f}% (vs 50% random)")
    print(f"  Sharpe: {sharpe:.3f} (positive)")
    print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
else:
    print(f"\n❌ SSRF FAILS")
    print(f"  Hit: {hit_ratio:.1f}% (vs 50% random)")
    print(f"  Sharpe: {sharpe:.3f}")
    print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

print("\n" + "="*70)
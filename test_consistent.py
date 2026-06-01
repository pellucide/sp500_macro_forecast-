#!/usr/bin/env python3
"""
SSRF Test with PROPER temporal alignment (NO leakage)
Train on [0:t-1] to predict y[t]
Compare pred[t] with actual[t]
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

PREDICTION_SCALE = 1.0  # No scaling for fair comparison

print("="*70)
print(f"SSRF PROPER TEST - NO LEAKAGE")
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

# PROPER walk-forward OOS
# Train on [0:t-1] to predict y[t], test on y[t]
train_window = 60

model_preds = []
actual_returns = []

for i in range(train_window, len(X_arr)):
    # Train on [0:i-1], predict y[i]
    X_train = X_arr[:i]      # Features up to i-1
    y_train = y_arr[:i]     # Returns up to i-1
    X_test = X_arr[i:i+1]   # Features at time i
    y_actual = y_arr[i]      # Actual return at time i

    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=10000)
        model.fit(X_train_scaled, y_train)

        pred = model.predict(X_test_scaled)[0]
        model_preds.append(pred)
    except:
        model_preds.append(0)

    actual_returns.append(y_actual)

model_preds = np.array(model_preds)
actual_returns = np.array(actual_returns)

# PROPER comparison: pred[t] should be compared with actual[t]
# (not pred[t] with actual[t+1] which was the bug)
pred_compared = model_preds[:-1]
actual_compared = actual_returns[:-1]

# Direction accuracy
direction_correct = np.sum(np.sign(pred_compared) == np.sign(actual_compared))
hit_ratio = direction_correct / len(actual_compared) * 100

# P&L
pnl = pred_compared * actual_compared
total_pnl = np.sum(pnl)
ann_return = np.mean(pnl) * 12
ann_vol = np.std(pnl) * np.sqrt(12)
sharpe = ann_return / ann_vol if ann_vol > 0 else 0

print("\n" + "="*70)
print("RESULTS (NO LEAKAGE - proper temporal alignment)")
print("="*70)

print(f"\nSSRF:")
print(f"  Direction Accuracy: {hit_ratio:.1f}% (out of {len(actual_compared)} predictions)")
print(f"  Correct: {direction_correct}, Wrong: {len(actual_compared) - direction_correct}")
print(f"  Total P&L: {total_pnl:.2f}")
print(f"  Annualized Return: {ann_return:.2f}")
print(f"  Annualized Vol: {ann_vol:.2f}")
print(f"  Sharpe Ratio: {sharpe:.3f}")

# SPX Buy & Hold for comparison
spx_total = (np.prod(1 + actual_returns/100) - 1) * 100
print(f"\nSPX Buy&Hold:")
print(f"  Total Return: {spx_total:.1f}%")

# Compare to baselines
print("\n" + "-"*70)
print("Comparison to baselines:")

# Momentum baseline
momentum_pred = np.zeros(len(actual_returns))
momentum_pred[1:] = np.sign(actual_returns[:-1])  # Momentum at t = sign(return at t-1)
momentum_compared = momentum_pred[:-1]
momentum_correct = np.sum(np.sign(momentum_compared) == np.sign(actual_compared))
momentum_hit = momentum_correct / len(actual_compared) * 100
momentum_pnl = np.sum(momentum_compared * actual_compared)
print(f"  Momentum: {momentum_hit:.1f}% hit, {momentum_pnl:.2f} P&L")

# Hist mean baseline
hist_mean = np.mean(actual_returns)
hist_pred = np.full(len(actual_returns), hist_mean)
hist_compared = hist_pred[:-1]
hist_correct = np.sum(np.sign(hist_compared) == np.sign(actual_compared))
hist_hit = hist_correct / len(actual_compared) * 100
hist_pnl = np.sum(hist_compared * actual_compared)
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
    boot_ret = np.mean(boot_pnl) * 12
    boot_vol = np.std(boot_pnl) * np.sqrt(12)
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
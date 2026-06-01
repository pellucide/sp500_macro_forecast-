#!/usr/bin/env python3
"""
SSRF vs Ridge Regression Comparison (PROPER - NO LEAKAGE)
Uses lagged features: predict y[t] using X[t-1] (last month's FRED data)
"""
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*70)
print("SSRF vs RIDGE (NO LEAKAGE - Proper Temporal Split)")
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

# Walk-forward OOS with proper temporal split
# Train on periods [0:t] to predict y[t+1] using X[t+1]
train_window = 60

ssrf_preds = []
ridge_preds = []
actual_returns = []  # actual[t] = y_arr[t+1] (next period return)

for i in range(train_window, len(X_arr) - 1):
    train_start = i - train_window
    X_train = X_arr[train_start:i]      # Features up to time i-1
    y_train = y_arr[train_start + 1:i + 1]  # Returns from time 1 to i (predictors use past)
    X_test = X_arr[i + 1:i + 2]         # Next period features (at time i+1)

    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # SSRF (ElasticNet) - NO scaling
        ssrf = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=10000)
        ssrf.fit(X_train_scaled, y_train)
        ssrf_preds.append(ssrf.predict(X_test_scaled)[0])

        # Ridge - NO scaling
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_scaled, y_train)
        ridge_preds.append(ridge.predict(X_test_scaled)[0])

    except Exception as e:
        ssrf_preds.append(0)
        ridge_preds.append(0)

    actual_returns.append(y_arr[i + 1])  # actual at time i+1

# Convert to arrays
ssrf_preds = np.array(ssrf_preds)
ridge_preds = np.array(ridge_preds)
actual_returns = np.array(actual_returns)

print(f"Predictions: {len(ssrf_preds)} periods")
print(f"Actual returns: {len(actual_returns)} periods")

# Metric calculation: pred[t] → actual[t]
pred_next = ssrf_preds[:-1]
actual_next = actual_returns[:-1]

def calc_metrics(pred_next, actual_next, name):
    """Calculate metrics"""
    direction_correct = np.sum(np.sign(pred_next) == np.sign(actual_next))
    hit_ratio = direction_correct / len(actual_next) * 100

    pnl = pred_next * actual_next
    total_pnl = np.sum(pnl)
    ann_return = np.mean(pnl) * 12
    ann_vol = np.std(pnl) * np.sqrt(12)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    return {
        'name': name,
        'hit_ratio': hit_ratio,
        'total_pnl': total_pnl,
        'ann_return': ann_return,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
    }

print("\n" + "="*70)
print("RESULTS (NO LEAKAGE, proper temporal split)")
print("="*70)

ssrf_metrics = calc_metrics(ssrf_preds[:-1], actual_returns[:-1], 'SSRF')
ridge_metrics = calc_metrics(ridge_preds[:-1], actual_returns[:-1], 'Ridge')

# Momentum baseline: predict sign(y[t-1])
# Momentum at time t = sign(y[t-1])
momentum_pred = np.zeros(len(actual_returns))
momentum_pred[1:] = np.sign(actual_returns[:-1])
momentum_metrics = calc_metrics(momentum_pred[:-1], actual_returns[:-1], 'Momentum')

# Hist Mean: predict average return (from past only!)
hist_mean = np.mean(actual_returns)
hist_pred = np.full(len(actual_returns), hist_mean)
hist_metrics = calc_metrics(hist_pred[:-1], actual_returns[:-1], 'Hist Mean')

spx_total = (np.prod(1 + actual_returns/100) - 1) * 100

print(f"\n{'Model':<15} {'Hit%':>8} {'Total P&L':>12} {'Ann.Ret':>10} {'Sharpe':>8}")
print("-"*60)
print(f"{'SSRF':<15} {ssrf_metrics['hit_ratio']:>8.1f} {ssrf_metrics['total_pnl']:>12.2f} {ssrf_metrics['ann_return']:>10.2f} {ssrf_metrics['sharpe']:>8.3f}")
print(f"{'Ridge':<15} {ridge_metrics['hit_ratio']:>8.1f} {ridge_metrics['total_pnl']:>12.2f} {ridge_metrics['ann_return']:>10.2f} {ridge_metrics['sharpe']:>8.3f}")
print("-"*60)
print(f"{'Momentum':<15} {momentum_metrics['hit_ratio']:>8.1f} {momentum_metrics['total_pnl']:>12.2f} {momentum_metrics['ann_return']:>10.2f} {momentum_metrics['sharpe']:>8.3f}")
print(f"{'Hist Mean':<15} {hist_metrics['hit_ratio']:>8.1f} {hist_metrics['total_pnl']:>12.2f} {hist_metrics['ann_return']:>10.2f} {hist_metrics['sharpe']:>8.3f}")
print(f"{'SPX B&H':<15} {'N/A':>8} {spx_total:>12.1f} {'N/A':>10} {'N/A':>8}")

# Statistical significance
print("\n" + "="*70)
print("STATISTICAL SIGNIFICANCE")
print("="*70)

for name, preds in [('SSRF', ssrf_preds), ('Ridge', ridge_preds)]:
    pred_next = preds[:-1]
    actual_next = actual_returns[:-1]
    pnl = pred_next * actual_next

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
    t_stat, p_t = stats.ttest_1samp(pnl, 0)

    print(f"\n{name}:")
    print(f"  95% CI for Sharpe: [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"  t-test p-value: {p_t:.4f}")
    sig = "✓ SIGNIFICANT" if p_t < 0.05 and ci_lower > 0 else "✗ NOT SIGNIFICANT"
    print(f"  {sig}")

print("\n" + "="*70)
print("VERDICT")
print("="*70)

all_models = [ssrf_metrics, ridge_metrics, momentum_metrics, hist_metrics]
for m in all_models:
    verdict = "✅ PASS" if m['hit_ratio'] > 50 and m['sharpe'] > 0 else "❌ FAIL"
    print(f"{m['name']:<15}: {verdict}")

print("\n" + "="*70)
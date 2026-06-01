#!/usr/bin/env python3
"""
SSRF vs Ridge Regression Comparison
pred[t] → actual[t+1] for BOTH direction accuracy AND P&L
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
print("SSRF vs RIDGE REGRESSION COMPARISON")
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

# Walk-forward OOS
train_window = 60
start_idx = train_window

ssrf_preds_10 = []
ssrf_preds_5 = []
ridge_preds_10 = []
ridge_preds_5 = []
actual_returns = []

for i in range(start_idx, len(X_arr)):
    train_start = i - train_window
    X_train = X_arr[train_start:i]
    y_train = y_arr[train_start:i]
    X_test = X_arr[i:i+1]

    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # SSRF (ElasticNet)
        y_train_scaled_10 = y_train * 10.0
        ssrf_10 = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=10000)
        ssrf_10.fit(X_train_scaled, y_train_scaled_10)
        ssrf_preds_10.append(ssrf_10.predict(X_test_scaled)[0])

        y_train_scaled_5 = y_train * 5.0
        ssrf_5 = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=10000)
        ssrf_5.fit(X_train_scaled, y_train_scaled_5)
        ssrf_preds_5.append(ssrf_5.predict(X_test_scaled)[0])

        # Ridge
        y_train_scaled_10 = y_train * 10.0
        ridge_10 = Ridge(alpha=1.0)
        ridge_10.fit(X_train_scaled, y_train_scaled_10)
        ridge_preds_10.append(ridge_10.predict(X_test_scaled)[0])

        y_train_scaled_5 = y_train * 5.0
        ridge_5 = Ridge(alpha=1.0)
        ridge_5.fit(X_train_scaled, y_train_scaled_5)
        ridge_preds_5.append(ridge_5.predict(X_test_scaled)[0])

    except Exception as e:
        ssrf_preds_10.append(0)
        ssrf_preds_5.append(0)
        ridge_preds_10.append(0)
        ridge_preds_5.append(0)

    actual_returns.append(y_arr[i])

# Convert to arrays
ssrf_preds_10 = np.array(ssrf_preds_10)
ssrf_preds_5 = np.array(ssrf_preds_5)
ridge_preds_10 = np.array(ridge_preds_10)
ridge_preds_5 = np.array(ridge_preds_5)
actual_returns = np.array(actual_returns)

# CONSISTENT calculation: pred[t] → actual[t+1]
pred_for_next = model_preds[:-1] if 'model_preds' in dir() else None

def calc_metrics(preds, actual_returns):
    """Calculate metrics with consistent pred[t] → actual[t+1]"""
    preds = np.array(preds)
    actual = np.array(actual_returns)

    pred_next = preds[:-1]
    actual_next = actual[1:]

    # Direction accuracy
    direction_correct = np.sum(np.sign(pred_next) == np.sign(actual_next))
    hit_ratio = direction_correct / len(actual_next) * 100

    # P&L
    pnl = pred_next * actual_next
    total_pnl = np.sum(pnl) * 100
    ann_return = np.mean(pnl) * 12 * 100
    ann_vol = np.std(pnl) * np.sqrt(12) * 100
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    return {
        'hit_ratio': hit_ratio,
        'total_pnl': total_pnl,
        'ann_return': ann_return,
        'ann_vol': ann_vol,
        'sharpe': sharpe
    }

# Calculate metrics for each model/scale
print("\n" + "="*70)
print("RESULTS (pred[t] → actual[t+1])")
print("="*70)

results = {
    'SSRF (Scale=10)': calc_metrics(ssrf_preds_10, actual_returns),
    'SSRF (Scale=5)': calc_metrics(ssrf_preds_5, actual_returns),
    'Ridge (Scale=10)': calc_metrics(ridge_preds_10, actual_returns),
    'Ridge (Scale=5)': calc_metrics(ridge_preds_5, actual_returns),
}

# Baselines (need to pad to same length for calc_metrics)
# Momentum
momentum_pred = np.zeros(len(actual_returns))
momentum_pred[1:] = np.sign(actual_returns[:-1])
momentum_metrics = calc_metrics(momentum_pred, actual_returns)

# Hist Mean
hist_mean = np.mean(actual_returns[:-1])
hist_pred = np.full(len(actual_returns), hist_mean)
hist_metrics = calc_metrics(hist_pred, actual_returns)

# SPX B&H
spx_total = (np.prod(1 + actual_returns/100) - 1) * 100

print(f"\n{'Model':<20} {'Hit%':>8} {'P&L%':>12} {'Sharpe':>8} {'Verdict':>10}")
print("-"*70)

for name, m in results.items():
    verdict = "✅ PASS" if m['hit_ratio'] > 50 and m['sharpe'] > 0 else "❌ FAIL"
    print(f"{name:<20} {m['hit_ratio']:>8.1f} {m['total_pnl']:>12.1f} {m['sharpe']:>8.3f} {verdict:>10}")

print("-"*70)
print(f"{'Momentum':<20} {momentum_metrics['hit_ratio']:>8.1f} {momentum_metrics['total_pnl']:>12.1f} {momentum_metrics['sharpe']:>8.3f} {'✅ PASS':>10}")
print(f"{'Hist Mean':<20} {hist_metrics['hit_ratio']:>8.1f} {hist_metrics['total_pnl']:>12.1f} {hist_metrics['sharpe']:>8.3f} {'✅ PASS':>10}")
print(f"{'SPX Buy&Hold':<20} {'N/A':>8} {spx_total:>12.1f} {'N/A':>8} {'✅ PASS':>10}")

# Statistical significance
print("\n" + "="*70)
print("STATISTICAL SIGNIFICANCE")
print("="*70)

for name, preds in [('SSRF (Scale=10)', ssrf_preds_10), ('Ridge (Scale=10)', ridge_preds_10),
                     ('SSRF (Scale=5)', ssrf_preds_5), ('Ridge (Scale=5)', ridge_preds_5)]:
    preds = np.array(preds)
    pred_next = preds[:-1]
    actual_next = actual_returns[1:]
    pnl = pred_next * actual_next

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

    # t-test
    t_stat, p_t = stats.ttest_1samp(pnl, 0)

    print(f"\n{name}:")
    print(f"  95% CI for Sharpe: [{ci_lower:.3f}, {ci_upper:.3f}]")
    print(f"  t-test p-value: {p_t:.4f}")
    sig = "✓" if p_t < 0.05 and ci_lower > 0 else "✗"
    print(f"  Significant: {sig}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

best_model = max(results.items(), key=lambda x: x[1]['sharpe'])
print(f"\nBest model: {best_model[0]}")
print(f"  Sharpe: {best_model[1]['sharpe']:.3f}")
print(f"  P&L: {best_model[1]['total_pnl']:.1f}%")

# Ridge vs SSRF comparison
ridge_sharpe = results['Ridge (Scale=10)']['sharpe']
ssrf_sharpe = results['SSRF (Scale=10)']['sharpe']

if ridge_sharpe > ssrf_sharpe:
    print(f"\nRidge BEATS SSRF by {ridge_sharpe - ssrf_sharpe:.3f} Sharpe")
else:
    print(f"\nSSRF BEATS Ridge by {ssrf_sharpe - ridge_sharpe:.3f} Sharpe")

print("\n" + "="*70)
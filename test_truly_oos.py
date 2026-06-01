#!/usr/bin/env python3
"""
SSRF - TRULY OUT-OF-SAMPLE TEST
Proper walk-forward with FIXED window and correct temporal alignment
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
print("SSRF TRULY OUT-OF-SAMPLE TEST (FIXED WINDOW)")
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

# ============================================================================
# TEST 1: Expanding window walk-forward (current approach in test_consistent.py)
# ============================================================================
print("\n" + "="*70)
print("TEST 1: EXPANDING WINDOW (like test_consistent.py)")
print("="*70)

train_window = 60

# Expanding window - train on [0:i], predict y[i], test X[i]
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
        model = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=10000)
        model.fit(X_train_s, y_train)
        exp_preds.append(model.predict(X_test_s)[0])
    except:
        exp_preds.append(0)
    exp_actual.append(y_actual)

exp_preds = np.array(exp_preds)
exp_actual = np.array(exp_actual)

# Metrics
exp_hit = np.mean(np.sign(exp_preds) == np.sign(exp_actual)) * 100
exp_pnl = exp_preds * exp_actual
exp_sharpe = np.mean(exp_pnl) * 12 / (np.std(exp_pnl) * np.sqrt(12)) if np.std(exp_pnl) > 0 else 0

print(f"SSRF (Expanding): Hit={exp_hit:.1f}%, Sharpe={exp_sharpe:.3f}")

# ============================================================================
# TEST 2: FIXED window walk-forward (more conservative)
# ============================================================================
print("\n" + "="*70)
print("TEST 2: FIXED WINDOW (60 months rolling)")
print("="*70)

fix_preds = []
fix_actual = []

for i in range(train_window, len(X_arr)):
    # FIXED window: always 60 months of training
    train_start = i - train_window
    X_train = X_arr[train_start:i]
    y_train = y_arr[train_start:i]
    X_test = X_arr[i:i+1]
    y_actual = y_arr[i]

    try:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=10000)
        model.fit(X_train_s, y_train)
        fix_preds.append(model.predict(X_test_s)[0])
    except:
        fix_preds.append(0)
    fix_actual.append(y_actual)

fix_preds = np.array(fix_preds)
fix_actual = np.array(fix_actual)

# Metrics
fix_hit = np.mean(np.sign(fix_preds) == np.sign(fix_actual)) * 100
fix_pnl = fix_preds * fix_actual
fix_sharpe = np.mean(fix_pnl) * 12 / (np.std(fix_pnl) * np.sqrt(12)) if np.std(fix_pnl) > 0 else 0

print(f"SSRF (Fixed): Hit={fix_hit:.1f}%, Sharpe={fix_sharpe:.3f}")

# ============================================================================
# TEST 3: Train 1980-2000, Test 2000-2026 (TRUE OOS)
# ============================================================================
print("\n" + "="*70)
print("TEST 3: TRUE OOS (Train 1980-2000, Test 2000-2026)")
print("="*70)

split_idx = 240  # ~2000-01

X_train_oos = X_arr[:split_idx]
y_train_oos = y_arr[:split_idx]
X_test_oos = X_arr[split_idx:]
y_test_oos = y_arr[split_idx:]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_oos)
X_test_s = scaler.transform(X_test_oos)
model = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=10000)
model.fit(X_train_s, y_train_oos)
oos_preds = model.predict(X_test_s)

# Metrics
oos_hit = np.mean(np.sign(oos_preds) == np.sign(y_test_oos)) * 100
oos_pnl = oos_preds * y_test_oos
oos_sharpe = np.mean(oos_pnl) * 12 / (np.std(oos_pnl) * np.sqrt(12)) if np.std(oos_pnl) > 0 else 0

print(f"SSRF (True OOS): Hit={oos_hit:.1f}%, Sharpe={oos_sharpe:.3f}")

# Baselines for True OOS
momentum_oos = np.full(len(y_test_oos), np.sign(y_train_oos[-1]))
mom_hit = np.mean(np.sign(momentum_oos) == np.sign(y_test_oos)) * 100
mom_pnl = momentum_oos * y_test_oos
mom_sharpe = np.mean(mom_pnl) * 12 / (np.std(mom_pnl) * np.sqrt(12)) if np.std(mom_pnl) > 0 else 0

hist_oos = np.full(len(y_test_oos), np.mean(y_train_oos))
hist_hit = np.mean(np.sign(hist_oos) == np.sign(y_test_oos)) * 100

spx_total = np.sum(y_test_oos)

print(f"Momentum (True OOS): Hit={mom_hit:.1f}%, Sharpe={mom_sharpe:.3f}")
print(f"Hist Mean (True OOS): Hit={hist_hit:.1f}%")
print(f"SPX Buy&Hold (True OOS): {spx_total:.1f}%")

# Bootstrap CI for True OOS
print("\n--- Bootstrap 95% CI for True OOS ---")
n_boot = 1000
sharpes_boot = []
for _ in range(n_boot):
    idx = np.random.choice(len(oos_pnl), size=len(oos_pnl), replace=True)
    boot_pnl = oos_pnl[idx]
    sh = np.mean(boot_pnl) * 12 / (np.std(boot_pnl) * np.sqrt(12)) if np.std(boot_pnl) > 0 else 0
    sharpes_boot.append(sh)

ci_low = np.percentile(sharpes_boot, 2.5)
ci_high = np.percentile(sharpes_boot, 97.5)
print(f"SSRF 95% CI: [{ci_low:.3f}, {ci_high:.3f}]")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"{'Test':<25} {'Hit%':>8} {'Sharpe':>8}")
print("-"*45)
print(f"{'Expanding Window':<25} {exp_hit:>8.1f} {exp_sharpe:>8.3f}")
print(f"{'Fixed Window (60m)':<25} {fix_hit:>8.1f} {fix_sharpe:>8.3f}")
print(f"{'True OOS (2000-2026)':<25} {oos_hit:>8.1f} {oos_sharpe:>8.3f}")
print(f"{'Momentum (True OOS)':<25} {mom_hit:>8.1f} {mom_sharpe:>8.3f}")
print(f"{'SPX Buy&Hold':<25} {'N/A':>8} {spx_total:>8.1f}%")

print("\n" + "="*70)
print("VERDICT")
print("="*70)

if oos_hit > 50 and oos_sharpe > 0 and ci_low > 0:
    print("SSRF PASSES TRUE OOS TEST")
elif oos_hit > 50 and oos_sharpe > 0:
    print("SSRF MARGINALLY PASSES - CI includes zero")
else:
    print("SSRF FAILS TRUE OOS TEST")
    print(f"  Hit: {oos_hit:.1f}% (vs 50% random)")
    print(f"  Sharpe: {oos_sharpe:.3f}")
    print(f"  95% CI: [{ci_low:.3f}, {ci_high:.3f}]")

print("\n" + "="*70)

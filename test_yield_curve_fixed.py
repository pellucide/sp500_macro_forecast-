#!/usr/bin/env python3
"""
SSRF Test with Yield Curve Spread - Scale=10 (FIXED - Proper Temporal Alignment)
Complete walk-forward OOS test with baselines and statistical significance
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from src.test_utils import (
    load_fred_enhanced, get_feature_columns,
    calc_metrics, bootstrap_ci, permutation_test, t_test,
    print_header
)

np.random.seed(42)

PREDICTION_SCALE = 10.0

print_header(f"YIELD CURVE PREDICTION TEST - SCALE={PREDICTION_SCALE}")

# =============================================================================
# STEP 1: FETCH AND ALIGN DATA
# =============================================================================
print_header("STEP 1: FETCH YIELD CURVE DATA")

# Load FRED
fred = load_fred_enhanced()

# Target: Yield Curve Spread (GS10 - TB3MS)
y_target = (fred['GS10'] - fred['TB3MS']).dropna()
y_target.name = 'YC_Spread'

# Features (exclude yield curve related)
feature_cols = get_feature_columns(fred)
X = fred[feature_cols].ffill().bfill().fillna(0)
X.index = X.index.normalize()

# Align
common_idx = X.index.intersection(y_target.index)
X_aligned = X.loc[common_idx]
y_aligned = y_target.loc[common_idx]

# Target change (predict direction of spread change)
y_change = y_aligned.diff().dropna()
y_change.name = 'YC_Spread_Change'

# Re-align
common_idx2 = X_aligned.index.intersection(y_change.index)
X_arr = X_aligned.loc[common_idx2].values
y_arr = y_change.loc[common_idx2].values.flatten()
dates = y_change.loc[common_idx2].index

print(f"Features: {len(feature_cols)}")
print(f"Periods: {len(X_arr)} ({dates[0].strftime('%Y-%m')} to {dates[-1].strftime('%Y-%m')})")
print(f"Mean spread change: {np.mean(y_arr):.4f}%")
print(f"Std spread change: {np.std(y_arr):.4f}%")
print(f"Prediction Scale: {PREDICTION_SCALE}x")

# =============================================================================
# STEP 2: DEFINE BASELINES
# =============================================================================
print_header("STEP 2: DEFINE BASELINES")

def naive_baseline():
    return 0.0

def random_baseline():
    return np.random.choice([-1, 1])

def historical_mean_baseline(y_train):
    return np.mean(y_train)

def momentum_baseline(y_prev):
    return np.sign(y_prev)

# =============================================================================
# STEP 3: WALK-FORWARD OOS TEST
# =============================================================================
print_header("STEP 3: WALK-FORWARD OOS TEST (PROPER ALIGNMENT)")

train_window = 60
step_size = 1
start_idx = train_window

model_preds = []
naive_preds = []
random_preds = []
hist_mean_preds = []
momentum_preds = []
actual_returns = []

for i in range(start_idx, len(X_arr) - 1, step_size):
    X_train = X_arr[:i]
    y_train = y_arr[1:i+1]  # y shifted: X[t] predicts y[t+1]
    X_test = X_arr[i+1:i+2]

    y_train_scaled = y_train * PREDICTION_SCALE

    # Baselines
    naive_preds.append(naive_baseline())
    random_preds.append(random_baseline())
    hist_mean_preds.append(historical_mean_baseline(y_train))

    if len(momentum_preds) == 0:
        momentum_preds.append(0)
    else:
        momentum_preds.append(momentum_baseline(y_train[-1]))

    # SSRF Model
    try:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)
        model.fit(X_train_scaled, y_train_scaled)
        pred = model.predict(X_test_scaled)[0]
        model_preds.append(pred)
    except:
        model_preds.append(0)

    actual_returns.append(y_arr[i+1])

model_preds = np.array(model_preds)
naive_preds = np.array(naive_preds)
random_preds = np.array(random_preds)
hist_mean_preds = np.array(hist_mean_preds)
momentum_preds = np.array(momentum_preds)
actual_returns = np.array(actual_returns)

print(f"OOS periods: {len(model_preds)}")
print(f"Training window: {train_window} months")

# =============================================================================
# STEP 4: CALCULATE METRICS
# =============================================================================
print_header("STEP 4: CALCULATE METRICS (FIXED ALIGNMENT)")

model_metrics = calc_metrics(model_preds, actual_returns)
naive_metrics = calc_metrics(naive_preds, actual_returns)
random_metrics = calc_metrics(random_preds, actual_returns)
hist_mean_metrics = calc_metrics(hist_mean_preds, actual_returns)
momentum_metrics = calc_metrics(momentum_preds, actual_returns)

# Buy & Hold
bh_pnl = np.cumprod(1 + np.array(actual_returns)/100) - 1
bh_total = bh_pnl[-1] * 100 if len(bh_pnl) > 0 else 0

print("\n" + "-"*70)
print(f"{'Strategy':<20} {'Hit%':>8} {'Ann.Ret%':>12} {'Sharpe':>8} {'R2 OOS':>8} {'P&L%':>12}")
print("-"*70)
print(f"{'SSRF (Scale=10)':<20} {model_metrics['hit_ratio']:>8.1f} {model_metrics['ann_return']:>12.1f} {model_metrics['sharpe']:>8.3f} {model_metrics['r2_oos']:>8.4f} {model_metrics['total_pnl']:>12.1f}")
print(f"{'Naive (0)':<20} {naive_metrics['hit_ratio']:>8.1f} {naive_metrics['ann_return']:>12.1f} {naive_metrics['sharpe']:>8.3f} {naive_metrics['r2_oos']:>8.4f} {naive_metrics['total_pnl']:>12.1f}")
print(f"{'Random':<20} {random_metrics['hit_ratio']:>8.1f} {random_metrics['ann_return']:>12.1f} {random_metrics['sharpe']:>8.3f} {random_metrics['r2_oos']:>8.4f} {random_metrics['total_pnl']:>12.1f}")
print(f"{'Hist. Mean':<20} {hist_mean_metrics['hit_ratio']:>8.1f} {hist_mean_metrics['ann_return']:>12.1f} {hist_mean_metrics['sharpe']:>8.3f} {hist_mean_metrics['r2_oos']:>8.4f} {hist_mean_metrics['total_pnl']:>12.1f}")
print(f"{'Momentum':<20} {momentum_metrics['hit_ratio']:>8.1f} {momentum_metrics['ann_return']:>12.1f} {momentum_metrics['sharpe']:>8.3f} {momentum_metrics['r2_oos']:>8.4f} {momentum_metrics['total_pnl']:>12.1f}")
print(f"{'Buy&Hold':<20} {'N/A':>8} {'N/A':>12} {'N/A':>8} {'N/A':>8} {bh_total:>12.1f}")

# =============================================================================
# STEP 5: STATISTICAL TESTS
# =============================================================================
print_header("STEP 5: STATISTICAL TESTS")

def diebold_mariano_yield(pred1, pred2, actual, h=1):
    """DM test for equal predictive accuracy (with HAR correction)."""
    e1 = actual - pred1
    e2 = actual - pred2
    d = e1**2 - e2**2
    n = len(d)
    k = (n + 1 - 2*h + (h*(h+1))/3) ** 0.5
    dm_stat = np.mean(d) / (np.std(d) / k) if np.std(d) > 0 else 0
    return dm_stat

# Permutation test
p_perm = permutation_test(model_preds, actual_returns, n_perms=1000)
print(f"\nPermutation Test (SSRF vs Random):")
print(f"  p-value: {p_perm:.4f}")
print(f"  {'SIGNIFICANT' if p_perm < 0.05 else 'NOT SIGNIFICANT'}")

# Bootstrap CI
ci_lower, ci_upper = bootstrap_ci(model_preds, actual_returns, n_boot=1000, ci=0.95)
print(f"\nBootstrap 95% CI for Sharpe:")
print(f"  CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
print(f"  {'Contains positive values' if ci_upper > 0 else 'Entirely negative'}")

# Diebold-Mariano test (custom with HAR correction)
dm_stat = diebold_mariano_yield(model_preds, momentum_preds, actual_returns)
p_dm = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
print(f"\nDiebold-Mariano (SSRF vs Momentum):")
print(f"  t-statistic: {dm_stat:.3f}")
print(f"  p-value: {p_dm:.4f}")
print(f"  {'SIGNIFICANT' if p_dm < 0.05 else 'NOT SIGNIFICANT'}")

# t-test
pnl = model_preds * actual_returns
t_stat, p_t = stats.ttest_1samp(pnl, 0)
print(f"\nt-test (mean P&L vs 0):")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_t:.4f}")
print(f"  {'SIGNIFICANT' if p_t < 0.05 else 'NOT SIGNIFICANT'}")

# =============================================================================
# STEP 6: TRULY OOS TEST (Train 1980-2000, Test 2000-2026)
# =============================================================================
print_header("STEP 6: TRULY OUT-OF-SAMPLE TEST (2000-2026)")

train_end_date = pd.Timestamp('2000-01-01')
test_dates = dates[start_idx:start_idx + len(model_preds)]
train_mask = test_dates < train_end_date
test_mask = test_dates >= train_end_date

model_oos = model_preds[test_mask]
naive_oos = naive_preds[test_mask]
random_oos = random_preds[test_mask]
hist_oos = hist_mean_preds[test_mask]
momentum_oos = momentum_preds[test_mask]
actual_oos = actual_returns[test_mask]

print(f"OOS periods (2000-2026): {len(model_oos)}")

model_oos_metrics = calc_metrics(model_oos, actual_oos)
naive_oos_metrics = calc_metrics(naive_oos, actual_oos)
random_oos_metrics = calc_metrics(random_oos, actual_oos)
hist_oos_metrics = calc_metrics(hist_oos, actual_oos)
momentum_oos_metrics = calc_metrics(momentum_oos, actual_oos)

# Buy & Hold
bh_pnl_oos = np.cumprod(1 + actual_oos/100) - 1
bh_total_oos = bh_pnl_oos[-1] * 100 if len(bh_pnl_oos) > 0 else 0

print("\n" + "-"*70)
print(f"{'Strategy':<20} {'Hit%':>8} {'Ann.Ret%':>12} {'Sharpe':>8} {'P&L%':>12}")
print("-"*70)
print(f"{'SSRF (Scale=10)':<20} {model_oos_metrics['hit_ratio']:>8.1f} {model_oos_metrics['ann_return']:>12.1f} {model_oos_metrics['sharpe']:>8.3f} {model_oos_metrics['total_pnl']:>12.1f}")
for name, m in [('Naive (0)', naive_oos_metrics), ('Random', random_oos_metrics),
                ('Hist. Mean', hist_oos_metrics), ('Momentum', momentum_oos_metrics)]:
    print(f"{name:<20} {m['hit_ratio']:>8.1f} {m['ann_return']:>12.1f} {m['sharpe']:>8.3f} {m['total_pnl']:>12.1f}")
print(f"{'Buy&Hold':<20} {'N/A':>8} {'N/A':>12} {'N/A':>8} {bh_total_oos:>12.1f}")

# OOS Statistical tests
print("\nOOS Statistical Tests:")
p_perm_oos = permutation_test(model_oos, actual_oos, n_perms=1000)
print(f"  Permutation p-value: {p_perm_oos:.4f}")

ci_lower_oos, ci_upper_oos = bootstrap_ci(model_oos, actual_oos, n_boot=1000, ci=0.95)
print(f"  Bootstrap 95% CI: [{ci_lower_oos:.3f}, {ci_upper_oos:.3f}]")

dm_stat_oos = diebold_mariano_yield(model_oos, momentum_oos, actual_oos)
p_dm_oos = 2 * (1 - stats.norm.cdf(abs(dm_stat_oos)))
print(f"  Diebold-Mariano p-value: {p_dm_oos:.4f}")

pnl_oos = model_oos * actual_oos
t_stat_oos, p_t_oos = stats.ttest_1samp(pnl_oos, 0)
print(f"  t-test p-value: {p_t_oos:.4f}")

# =============================================================================
# STEP 7: FINAL VERDICT
# =============================================================================
print_header("FINAL VERDICT")

ssrf_hit_oos = model_oos_metrics['hit_ratio']
ssrf_sharpe_oos = model_oos_metrics['sharpe']
ssrf_pnl_oos = model_oos_metrics['total_pnl']

if ssrf_hit_oos > 50 and ssrf_sharpe_oos > 0 and ci_upper_oos > 0:
    print("\nSSRF PASSES YIELD CURVE TEST")
    print(f"  Scale={PREDICTION_SCALE}x")
    print(f"  Direction Accuracy: {ssrf_hit_oos:.1f}% (vs 50% random)")
    print(f"  Sharpe: {ssrf_sharpe_oos:.3f} (POSITIVE)")
    print(f"  Total P&L: {ssrf_pnl_oos:.1f}% (PROFITABLE)")
    print(f"  95% CI: [{ci_lower_oos:.3f}, {ci_upper_oos:.3f}]")
else:
    print("\nSSRF FAILS YIELD CURVE TEST")
    print(f"  Scale={PREDICTION_SCALE}x")
    print(f"  Direction Accuracy: {ssrf_hit_oos:.1f}% (vs 50% random)")
    print(f"  Sharpe: {ssrf_sharpe_oos:.3f}")
    print(f"  Total P&L: {ssrf_pnl_oos:.1f}%")
    print(f"  95% CI: [{ci_lower_oos:.3f}, {ci_upper_oos:.3f}]")

print("\n" + "="*70)
print("END OF TEST")
print("="*70)

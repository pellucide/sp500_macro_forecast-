"""
SSRF Test with S&P 500 Returns - Scale=20 (FIXED - Proper Temporal Alignment)
Complete walk-forward OOS test with baselines and statistical significance
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from src.test_utils import (
    load_spx_returns, load_fred_enhanced, get_feature_columns,
    align_features_and_target, calc_metrics, bootstrap_ci,
    permutation_test, diebold_mariano, t_test, print_header
)

np.random.seed(42)

PREDICTION_SCALE = 20.0

print_header(f"SSRF COMPREHENSIVE TEST - SCALE={PREDICTION_SCALE}")

# =============================================================================
# STEP 1: FETCH AND ALIGN DATA
# =============================================================================
print_header("STEP 1: FETCH S&P 500 RETURNS AND ALIGN DATA")

spx_returns = load_spx_returns()
fred = load_fred_enhanced()

feature_cols = get_feature_columns(fred)
X = fred[feature_cols].ffill().bfill().fillna(0)

X_arr, y_arr, dates = align_features_and_target(X, spx_returns)

print(f"Features: {len(feature_cols)}")
print(f"Periods: {len(X_arr)} ({dates[0].strftime('%Y-%m')} to {dates[-1].strftime('%Y-%m')})")
print(f"Mean monthly return: {np.mean(y_arr):.3f}%")
print(f"Std monthly return: {np.std(y_arr):.3f}%")
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
print_header("STEP 3: WALK-FORWARD OOS TEST (PROPER TEMPORAL ALIGNMENT)")

train_window = 60
step_size = 1
start_idx = train_window

model_preds = []
naive_preds = []
random_preds = []
hist_mean_preds = []
momentum_preds = []
actual_returns = []

print(f"Running walk-forward test...")
print(f"Train window: {train_window} months, Scale: {PREDICTION_SCALE}x")

for i in range(start_idx, len(y_arr) - 1, step_size):
    X_train = X_arr[:i]
    y_train = y_arr[:i]
    X_test = X_arr[i+1:i+2]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)
    model.fit(X_train_s, y_train)
    model_pred = model.predict(X_test_s)[0] * PREDICTION_SCALE

    # Baselines
    naive_preds.append(naive_baseline())
    np.random.seed(i)
    random_preds.append(random_baseline())
    hist_mean_preds.append(historical_mean_baseline(y_train))
    momentum_preds.append(momentum_baseline(y_train[-1]))

    model_preds.append(model_pred)
    actual_returns.append(y_arr[i+1])

model_preds = np.array(model_preds)
naive_preds = np.array(naive_preds)
random_preds = np.array(random_preds)
hist_mean_preds = np.array(hist_mean_preds)
momentum_preds = np.array(momentum_preds)
actual_returns = np.array(actual_returns)

print(f"Tested {len(actual_returns)} periods")

# =============================================================================
# STEP 4: METRICS CALCULATION
# =============================================================================
print_header("STEP 4: PERFORMANCE METRICS (FIXED ALIGNMENT)")

results = {
    'SSRF': calc_metrics(model_preds, actual_returns),
    'Naive (0)': calc_metrics(naive_preds, actual_returns),
    'Random': calc_metrics(random_preds, actual_returns),
    'Hist. Mean': calc_metrics(hist_mean_preds, actual_returns),
    'Momentum': calc_metrics(momentum_preds, actual_returns),
}

print(f"\n{'Strategy':<15} {'Hit%':>7} {'AnnRet%':>10} {'Sharpe':>8} {'R2 OOS':>8} {'Total P&L':>12}")
print("-" * 72)
for name, m in results.items():
    print(f"{name:<15} {m['hit_ratio']:>6.1f}% {m['ann_return']:>9.1f}% {m['sharpe']:>8.3f} {m['r2_oos']:>8.4f} {m['total_pnl']:>11.1f}%")

# =============================================================================
# STEP 5: STATISTICAL SIGNIFICANCE TESTS
# =============================================================================
print_header("STEP 5: STATISTICAL SIGNIFICANCE TESTS")

# SSRF Statistical Tests
print("\n--- SSRF Model Statistical Tests ---")
perm_p = permutation_test(model_preds, actual_returns)
print(f"Permutation Test (vs random): p={perm_p:.4f} {'***' if perm_p < 0.01 else '**' if perm_p < 0.05 else '*' if perm_p < 0.1 else ''}")

ci_low, ci_high = bootstrap_ci(model_preds, actual_returns)
print(f"Bootstrap 95% CI for Sharpe: [{ci_low:.3f}, {ci_high:.3f}]")

dm_stat, dm_p = diebold_mariano(model_preds, momentum_preds, actual_returns)
print(f"Diebold-Mariano (SSRF vs Momentum): t={dm_stat:.3f}, p={dm_p:.4f} {'***' if dm_p < 0.01 else '**' if dm_p < 0.05 else '*' if dm_p < 0.1 else ''}")

t_stat, t_p = t_test(model_preds, actual_returns)
print(f"t-test (mean P&L vs 0): t={t_stat:.3f}, p={t_p:.4f} {'***' if t_p < 0.01 else '**' if t_p < 0.05 else '*' if t_p < 0.1 else ''}")

# =============================================================================
# STEP 6: TRULY OUT-OF-SAMPLE TEST (Train 1980-2000, Test 2000-2026)
# =============================================================================
print_header("STEP 6: TRULY OUT-OF-SAMPLE (Train 1980-2000, Test 2000-2026)")

train_end = pd.Timestamp('2000-01-31')
test_start = pd.Timestamp('2000-02-28')

train_mask = dates <= train_end
test_mask = dates >= test_start

X_train_oos = X_arr[train_mask]
X_test_oos = X_arr[test_mask]
y_train_oos = y_arr[train_mask]
y_test_oos = y_arr[test_mask]

print(f"Training: {len(X_train_oos)} periods (up to {train_end.strftime('%Y-%m')})")
print(f"Testing: {len(y_test_oos)} periods ({test_start.strftime('%Y-%m')} to {dates[-1].strftime('%Y-%m')})")

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_oos)
X_test_s = scaler.transform(X_test_oos)
model = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)
model.fit(X_train_s, y_train_oos)
ssrf_oos_preds = model.predict(X_test_s) * PREDICTION_SCALE

# Baselines
naive_oos = np.full(len(y_test_oos), 0.0)
hist_mean_oos = np.full(len(y_test_oos), np.mean(y_train_oos))
momentum_oos = np.full(len(y_test_oos), np.sign(y_train_oos[-1]))
np.random.seed(42)
random_oos = np.array([random_baseline() for _ in range(len(y_test_oos))])

# OOS Metrics
print("\n--- Out-of-Sample Results ---")
oos_results = {
    'SSRF': calc_metrics(ssrf_oos_preds, y_test_oos),
    'Naive (0)': calc_metrics(naive_oos, y_test_oos),
    'Random': calc_metrics(random_oos, y_test_oos),
    'Hist. Mean': calc_metrics(hist_mean_oos, y_test_oos),
    'Momentum': calc_metrics(momentum_oos, y_test_oos),
}

print(f"\n{'Strategy':<15} {'Hit%':>7} {'AnnRet%':>10} {'Sharpe':>8} {'R2 OOS':>8} {'Total P&L':>12}")
print("-" * 72)
for name, m in oos_results.items():
    print(f"{name:<15} {m['hit_ratio']:>6.1f}% {m['ann_return']:>9.1f}% {m['sharpe']:>8.3f} {m['r2_oos']:>8.4f} {m['total_pnl']:>11.1f}%")

spx_total = np.sum(y_test_oos)
print(f"{'SPX Buy&Hold':<15} {'N/A':>7} {'N/A':>10} {'N/A':>8} {'N/A':>8} {spx_total:>11.1f}%")

# OOS Statistical Tests
print("\n--- Out-of-Sample SSRF Statistical Tests ---")
oos_perm_p = permutation_test(ssrf_oos_preds, y_test_oos)
print(f"Permutation Test (vs random): p={oos_perm_p:.4f} {'***' if oos_perm_p < 0.01 else '**' if oos_perm_p < 0.05 else '*' if oos_perm_p < 0.1 else ''}")

oos_ci_low, oos_ci_high = bootstrap_ci(ssrf_oos_preds, y_test_oos)
print(f"Bootstrap 95% CI for Sharpe: [{oos_ci_low:.3f}, {oos_ci_high:.3f}]")

oos_dm_stat, oos_dm_p = diebold_mariano(ssrf_oos_preds, momentum_oos, y_test_oos)
print(f"Diebold-Mariano (SSRF vs Momentum): t={oos_dm_stat:.3f}, p={oos_dm_p:.4f} {'***' if oos_dm_p < 0.01 else '**' if oos_dm_p < 0.05 else '*' if oos_dm_p < 0.1 else ''}")

oos_t_stat, oos_t_p = t_test(ssrf_oos_preds, y_test_oos)
print(f"t-test (mean P&L vs 0): t={oos_t_stat:.3f}, p={oos_t_p:.4f} {'***' if oos_t_p < 0.01 else '**' if oos_t_p < 0.05 else '*' if oos_t_p < 0.1 else ''}")

# =============================================================================
# FINAL CONCLUSION
# =============================================================================
print_header("FINAL CONCLUSION")

ssrf_oos_sharpe = oos_results['SSRF']['sharpe']
ssrf_oos_hit = oos_results['SSRF']['hit_ratio']
ssrf_oos_pnl = oos_results['SSRF']['total_pnl']

if ssrf_oos_sharpe > 0 and ssrf_oos_hit > 50:
    if oos_ci_low > 0:
        print("SSRF PASSES OUT-OF-SAMPLE TEST")
        print(f"   Scale={PREDICTION_SCALE}x")
        print(f"   Direction Accuracy: {ssrf_oos_hit:.1f}%")
        print(f"   Sharpe: {ssrf_oos_sharpe:.3f} (95% CI: [{oos_ci_low:.3f}, {oos_ci_high:.3f}])")
        print(f"   Total P&L: {ssrf_oos_pnl:+.1f}%")
    else:
        print("SSRF MARGINALLY PASSES - CI includes negative")
        print(f"   Direction Accuracy: {ssrf_oos_hit:.1f}%")
        print(f"   Sharpe: {ssrf_oos_sharpe:.3f}")
        print(f"   95% CI: [{oos_ci_low:.3f}, {oos_ci_high:.3f}]")
elif ssrf_oos_sharpe < 0:
    print("SSRF FAILS OUT-OF-SAMPLE TEST")
    print(f"   Scale={PREDICTION_SCALE}x")
    print(f"   Direction Accuracy: {ssrf_oos_hit:.1f}% (vs 50% random)")
    print(f"   Sharpe: {ssrf_oos_sharpe:.3f} (NEGATIVE)")
    print(f"   Total P&L: {ssrf_oos_pnl:+.1f}% (LOSES MONEY)")
    print(f"   95% CI: [{oos_ci_low:.3f}, {oos_ci_high:.3f}]")
else:
    print("SSRF INCONCLUSIVE")
    print(f"   Direction Accuracy: {ssrf_oos_hit:.1f}%")
    print(f"   Sharpe: {ssrf_oos_sharpe:.3f}")

print("\n" + "="*70)

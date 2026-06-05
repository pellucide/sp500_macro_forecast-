"""
SSRF Comprehensive Test Suite
Combines all 5 test scripts into one file for easier management.
Preserves each test's unique alignment strategy and model configuration.

Tests:
  1. Proper Alignment - Forward expanding window (X[:i] -> y[i+1])
  2. Truly OOS - Expanding + Fixed window + True OOS (2000-2026)
  3. Scale 20x - Forward expanding window with 20x prediction amplification
  4. Ridge Comparison - SSRF (ElasticNet) vs Ridge regression
  5. Yield Curve - Yield curve spread prediction with 10x scale
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from src.test_utils import (
    load_spx_returns, load_fred_enhanced, get_feature_columns,
    align_features_and_target, calc_metrics, bootstrap_ci,
    permutation_test, diebold_mariano, t_test, print_header,
    print_metrics_table, print_verdict
)

np.random.seed(42)

# Shared data (loaded once, used by all tests)
print_header("LOADING DATA")
spx_returns = load_spx_returns()
fred = load_fred_enhanced()
feature_cols = get_feature_columns(fred)
print(f"SPX returns: {len(spx_returns)} months ({spx_returns.index[0].strftime('%Y-%m')} to {spx_returns.index[-1].strftime('%Y-%m')})")
print(f"FRED features: {len(feature_cols)} indicators")
print(f"FRED date range: {fred.index[0].strftime('%Y-%m')} to {fred.index[-1].strftime('%Y-%m')}")


# =============================================================================
# HELPER: Common baselines
# =============================================================================
def _naive():
    return 0.0

def _random():
    return np.random.choice([-1, 1])

def _hist_mean(y_train):
    return np.mean(y_train)

def _momentum(y_prev):
    return np.sign(y_prev)


# =============================================================================
# TEST 1: PROPER ALIGNMENT (X[:i] -> y[i+1])
# =============================================================================
print_header("TEST 1: PROPER ALIGNMENT - FORWARD EXPANDING WINDOW (X[:i] -> y[i+1])")

X = fred[feature_cols].ffill().bfill().fillna(0)
X_arr, y_arr_raw, dates_raw = align_features_and_target(X, spx_returns)
# Temporal alignment: X[t] predicts y[t+1]
# Shift y so y_arr[i] = return at month after X_arr[i]
y_arr = y_arr_raw[1:]
X_arr = X_arr[:-1]
dates = dates_raw[1:]
print(f"\nData: {len(X_arr)} periods ({dates[0].strftime('%Y-%m')} to {dates[-1].strftime('%Y-%m')})")
print(f"Mean monthly return: {np.mean(y_arr):.3f}%")

# --- 1a: Full-sample walk-forward ---
train_window = 60
preds_1, actuals_1 = [], []

for i in tqdm(range(train_window, len(X_arr)), desc="Test 1a: Forward Walk"):
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
        preds_1.append(model.predict(X_test_s)[0])
    except:
        preds_1.append(0)
    actuals_1.append(y_actual)

preds_1 = np.array(preds_1)
actuals_1 = np.array(actuals_1)

m1 = calc_metrics(preds_1, actuals_1)
ci1_low, ci1_high = bootstrap_ci(preds_1, actuals_1)
t1_stat, t1_p = t_test(preds_1, actuals_1)

print(f"SSRF (Forward Expanding):")
print(f"  Direction Accuracy: {m1['hit_ratio']:.1f}% ({len(actuals_1)} predictions)")
print(f"  Total P&L: {m1['total_pnl']:.2f}")
print(f"  Ann Return: {m1['ann_return']:.2f}%, Ann Vol: {m1['ann_vol']:.2f}%")
print(f"  Sharpe Ratio: {m1['sharpe']:.3f}")

spx_bh_1 = (np.prod(1 + actuals_1 / 100) - 1) * 100
print(f"  SPX Buy&Hold: {spx_bh_1:.1f}%")
print(f"  Bootstrap 95% CI for Sharpe: [{ci1_low:.3f}, {ci1_high:.3f}]")
print(f"  t-test p-value: {t1_p:.4f}")
print_verdict("SSRF (Full Sample)", m1['hit_ratio'], m1['sharpe'], ci1_low, ci1_high)

# --- 1b: True OOS (Train 1980-2000, Test 2000-2026) ---
print_header("TEST 1b: TRUE OOS (2000-2026)")

split_idx = 240  # ~2000-01

oos_preds_1, oos_actual_1 = [], []
for i in tqdm(range(split_idx, len(X_arr)), desc="Test 1b: True OOS"):
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
        oos_preds_1.append(model.predict(X_test_s)[0])
    except:
        oos_preds_1.append(0)
    oos_actual_1.append(y_actual)

oos_preds_1 = np.array(oos_preds_1)
oos_actual_1 = np.array(oos_actual_1)

m1_oos = calc_metrics(oos_preds_1, oos_actual_1)
ci1_oos_low, ci1_oos_high = bootstrap_ci(oos_preds_1, oos_actual_1)

# Baselines
momentum_1 = np.full(len(oos_actual_1), np.sign(y_arr[split_idx - 1]))
mom1_m = calc_metrics(momentum_1, oos_actual_1)
hist_1 = np.full(len(oos_actual_1), np.mean(y_arr[:split_idx]))
hist1_hit = np.mean(np.sign(hist_1) == np.sign(oos_actual_1)) * 100
spx_bh_1_oos = (np.prod(1 + oos_actual_1 / 100) - 1) * 100

print(f"  SSRF:      Hit={m1_oos['hit_ratio']:.1f}%, Sharpe={m1_oos['sharpe']:.3f}")
print(f"  Momentum:  Hit={mom1_m['hit_ratio']:.1f}%, Sharpe={mom1_m['sharpe']:.3f}")
print(f"  Hist Mean: Hit={hist1_hit:.1f}%")
print(f"  SPX B&H:   {spx_bh_1_oos:.1f}%")
print(f"  SSRF 95% CI: [{ci1_oos_low:.3f}, {ci1_oos_high:.3f}]")
print_verdict("SSRF (True OOS)", m1_oos['hit_ratio'], m1_oos['sharpe'],
              ci1_oos_low, ci1_oos_high)


# =============================================================================
# TEST 2: TRULY OOS - THREE CONFIGURATIONS
# =============================================================================
print_header("TEST 2: THREE OOS CONFIGURATIONS")

# --- 2a: Expanding window (X[:i] -> y[i]) ---
preds_2a, actual_2a = [], []
for i in tqdm(range(train_window, len(X_arr)), desc="Test 2a: Expanding"):
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
        preds_2a.append(model.predict(X_test_s)[0])
    except:
        preds_2a.append(0)
    actual_2a.append(y_actual)

m2a = calc_metrics(np.array(preds_2a), np.array(actual_2a))

# --- 2b: Fixed window (60m rolling, X[t-60:t] -> y[t]) ---
preds_2b, actual_2b = [], []
for i in tqdm(range(train_window, len(X_arr)), desc="Test 2b: Fixed 60m"):
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
        preds_2b.append(model.predict(X_test_s)[0])
    except:
        preds_2b.append(0)
    actual_2b.append(y_actual)

m2b = calc_metrics(np.array(preds_2b), np.array(actual_2b))

# --- 2c: True OOS (Train 1980-2000, Test 2000-2026, forward alignment) ---
preds_2c, actual_2c = [], []
for i in tqdm(range(split_idx, len(X_arr)), desc="Test 2c: True OOS"):
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
        preds_2c.append(model.predict(X_test_s)[0])
    except:
        preds_2c.append(0)
    actual_2c.append(y_actual)

preds_2c = np.array(preds_2c)
actual_2c = np.array(actual_2c)
m2c = calc_metrics(preds_2c, actual_2c)

# Baselines for 2c
mom_2c = np.full(len(actual_2c), np.sign(y_arr[split_idx - 1]))
mom2c_m = calc_metrics(mom_2c, actual_2c)
hist_2c = np.full(len(actual_2c), np.mean(y_arr[:split_idx]))
hist2c_hit = np.mean(np.sign(hist_2c) == np.sign(actual_2c)) * 100
spx_bh_2c = (np.prod(1 + actual_2c / 100) - 1) * 100

ci2_low, ci2_high = bootstrap_ci(preds_2c, actual_2c)

print(f"\n{'Test':<25} {'Hit%':>8} {'Sharpe':>8}")
print("-" * 45)
print(f"{'Expanding Window':<25} {m2a['hit_ratio']:>8.1f} {m2a['sharpe']:>8.3f}")
print(f"{'Fixed Window (60m)':<25} {m2b['hit_ratio']:>8.1f} {m2b['sharpe']:>8.3f}")
print(f"{'True OOS (2000-2026)':<25} {m2c['hit_ratio']:>8.1f} {m2c['sharpe']:>8.3f}")
print(f"{'Momentum (True OOS)':<25} {mom2c_m['hit_ratio']:>8.1f} {mom2c_m['sharpe']:>8.3f}")
print(f"{'SPX Buy&Hold':<25} {'N/A':>8} {spx_bh_2c:>8.1f}%")
print(f"\nTrue OOS 95% CI: [{ci2_low:.3f}, {ci2_high:.3f}]")
print_verdict("SSRF (Test 2 True OOS)", m2c['hit_ratio'], m2c['sharpe'],
              ci2_low, ci2_high)


# =============================================================================
# TEST 3: SCALE 20X - PREDICTION AMPLIFICATION
# =============================================================================
print_header("TEST 3: SCALE 20X - PREDICTION AMPLIFICATION")

SCALE_3 = 20.0
print(f"Prediction scale: {SCALE_3}x")

# --- 3a: Full walk-forward ---
preds_3a = []
naive_3a_p, random_3a_p, hist_3a_p, momentum_3a_p = [], [], [], []
actual_3a = []

for i in tqdm(range(train_window, len(X_arr)), desc="Test 3a: Scale 20x"):
    X_train = X_arr[:i]
    y_train = y_arr[:i]
    X_test = X_arr[i:i+1]

    try:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        model = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)
        model.fit(X_train_s, y_train)
        preds_3a.append(model.predict(X_test_s)[0] * SCALE_3)
    except:
        preds_3a.append(0)

    naive_3a_p.append(_naive())
    np.random.seed(i)
    random_3a_p.append(_random())
    hist_3a_p.append(_hist_mean(y_train))
    momentum_3a_p.append(_momentum(y_train[-1]))
    actual_3a.append(y_arr[i])

preds_3a = np.array(preds_3a)
actual_3a = np.array(actual_3a)

results_3a = {
    'SSRF': calc_metrics(preds_3a, actual_3a),
    'Naive (0)': calc_metrics(np.array(naive_3a_p), actual_3a),
    'Random': calc_metrics(np.array(random_3a_p), actual_3a),
    'Hist. Mean': calc_metrics(np.array(hist_3a_p), actual_3a),
    'Momentum': calc_metrics(np.array(momentum_3a_p), actual_3a),
}

print_metrics_table(results_3a)

# Statistical tests
print("\n--- SSRF Statistical Tests ---")
perm_p_3a = permutation_test(preds_3a, actual_3a)
print(f"Permutation Test (vs random): p={perm_p_3a:.4f} {'SIG' if perm_p_3a < 0.05 else 'NOT SIG'}")

ci3a_low, ci3a_high = bootstrap_ci(preds_3a, actual_3a)
print(f"Bootstrap 95% CI for Sharpe: [{ci3a_low:.3f}, {ci3a_high:.3f}]")

dm3a_stat, dm3a_p = diebold_mariano(preds_3a, np.array(momentum_3a_p), actual_3a)
print(f"Diebold-Mariano (SSRF vs Momentum): t={dm3a_stat:.3f}, p={dm3a_p:.4f} {'SIG' if dm3a_p < 0.05 else 'NOT SIG'}")

t3a_stat, t3a_p = t_test(preds_3a, actual_3a)
print(f"t-test (mean P&L vs 0): t={t3a_stat:.3f}, p={t3a_p:.4f} {'SIG' if t3a_p < 0.05 else 'NOT SIG'}")

# --- 3b: True OOS (Train 1980-2000, Test 2000-2026) ---
train_end = pd.Timestamp('2000-01-31')
test_start = pd.Timestamp('2000-02-28')
train_mask_3 = dates <= train_end
test_mask_3 = dates >= test_start

X_train_3b = X_arr[train_mask_3]
X_test_3b = X_arr[test_mask_3]
y_train_3b = y_arr[train_mask_3]
y_test_3b = y_arr[test_mask_3]

scaler_3b = StandardScaler()
X_train_3b_s = scaler_3b.fit_transform(X_train_3b)
X_test_3b_s = scaler_3b.transform(X_test_3b)
model_3b = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)
model_3b.fit(X_train_3b_s, y_train_3b)
ssrf_3b = model_3b.predict(X_test_3b_s) * SCALE_3

naive_3b = np.full(len(y_test_3b), 0.0)
hist_3b = np.full(len(y_test_3b), np.mean(y_train_3b))
momentum_3b = np.full(len(y_test_3b), np.sign(y_train_3b[-1]))
np.random.seed(42)
random_3b = np.array([_random() for _ in range(len(y_test_3b))])

results_3b = {
    'SSRF': calc_metrics(ssrf_3b, y_test_3b),
    'Naive (0)': calc_metrics(naive_3b, y_test_3b),
    'Random': calc_metrics(random_3b, y_test_3b),
    'Hist. Mean': calc_metrics(hist_3b, y_test_3b),
    'Momentum': calc_metrics(momentum_3b, y_test_3b),
}

spx_3b = np.sum(y_test_3b)
print(f"\n--- Out-of-Sample Results (Scale={SCALE_3}x) ---")
print_metrics_table(results_3b)
print(f"{'SPX Buy&Hold':<20} {'N/A':>7} {'N/A':>9} {'N/A':>7} {'N/A':>7} {spx_3b:>11.1f}%")

# OOS statistical tests
print("\n--- OOS Statistical Tests ---")
oos_perm_3b = permutation_test(ssrf_3b, y_test_3b)
print(f"Permutation Test: p={oos_perm_3b:.4f} {'SIG' if oos_perm_3b < 0.05 else 'NOT SIG'}")

ci3b_low, ci3b_high = bootstrap_ci(ssrf_3b, y_test_3b)
print(f"Bootstrap 95% CI: [{ci3b_low:.3f}, {ci3b_high:.3f}]")

dm3b_stat, dm3b_p = diebold_mariano(ssrf_3b, momentum_3b, y_test_3b)
print(f"Diebold-Mariano (vs Momentum): t={dm3b_stat:.3f}, p={dm3b_p:.4f} {'SIG' if dm3b_p < 0.05 else 'NOT SIG'}")

t3b_stat, t3b_p = t_test(ssrf_3b, y_test_3b)
print(f"t-test: t={t3b_stat:.3f}, p={t3b_p:.4f} {'SIG' if t3b_p < 0.05 else 'NOT SIG'}")

m3b = results_3b['SSRF']
print_verdict(f"SSRF (Scale={SCALE_3}x OOS)", m3b['hit_ratio'], m3b['sharpe'],
              ci3b_low, ci3b_high,
              f"Total P&L: {m3b['total_pnl']:+.1f}%")


# =============================================================================
# TEST 4: SSRF vs RIDGE COMPARISON
# =============================================================================
print_header("TEST 4: SSRF vs RIDGE COMPARISON")

ssrf_4_p, ridge_4_p, actual_4 = [], [], []

for i in tqdm(range(train_window, len(X_arr)), desc="Test 4: SSRF vs Ridge"):
    train_start_4 = i - train_window
    X_train_4 = X_arr[train_start_4:i]
    y_train_4 = y_arr[train_start_4:i]
    X_test_4 = X_arr[i:i+1]

    try:
        scaler_4 = StandardScaler()
        X_train_4_s = scaler_4.fit_transform(X_train_4)
        X_test_4_s = scaler_4.transform(X_test_4)

        model_e = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)
        model_e.fit(X_train_4_s, y_train_4)
        ssrf_4_p.append(model_e.predict(X_test_4_s)[0])

        model_r = Ridge(alpha=1.0)
        model_r.fit(X_train_4_s, y_train_4)
        ridge_4_p.append(model_r.predict(X_test_4_s)[0])
    except:
        ssrf_4_p.append(0)
        ridge_4_p.append(0)

    actual_4.append(y_arr[i])

ssrf_4_p = np.array(ssrf_4_p)
ridge_4_p = np.array(ridge_4_p)
actual_4 = np.array(actual_4)

# Baselines
momentum_4 = np.zeros(len(actual_4))
momentum_4[1:] = np.sign(actual_4[:-1])
hist_4 = np.full(len(actual_4), np.mean(actual_4))
spx_bh_4 = (np.prod(1 + actual_4 / 100) - 1) * 100

ssrf_4_m = calc_metrics(ssrf_4_p, actual_4)
ridge_4_m = calc_metrics(ridge_4_p, actual_4)
mom_4_m = calc_metrics(momentum_4, actual_4)
hist_4_m = calc_metrics(hist_4, actual_4)

print(f"\n{'Model':<15} {'Hit%':>8} {'Total P&L':>12} {'Ann.Ret':>10} {'Sharpe':>8}")
print("-" * 60)
print(f"{'SSRF':<15} {ssrf_4_m['hit_ratio']:>8.1f} {ssrf_4_m['total_pnl']:>12.2f} {ssrf_4_m['ann_return']:>10.2f} {ssrf_4_m['sharpe']:>8.3f}")
print(f"{'Ridge':<15} {ridge_4_m['hit_ratio']:>8.1f} {ridge_4_m['total_pnl']:>12.2f} {ridge_4_m['ann_return']:>10.2f} {ridge_4_m['sharpe']:>8.3f}")
print("-" * 60)
print(f"{'Momentum':<15} {mom_4_m['hit_ratio']:>8.1f} {mom_4_m['total_pnl']:>12.2f} {mom_4_m['ann_return']:>10.2f} {mom_4_m['sharpe']:>8.3f}")
print(f"{'Hist Mean':<15} {hist_4_m['hit_ratio']:>8.1f} {hist_4_m['total_pnl']:>12.2f} {hist_4_m['ann_return']:>10.2f} {hist_4_m['sharpe']:>8.3f}")
print(f"{'SPX B&H':<15} {'N/A':>8} {spx_bh_4:>12.1f} {'N/A':>10} {'N/A':>8}")

# Statistical tests
for name, preds in [('SSRF', ssrf_4_p), ('Ridge', ridge_4_p)]:
    ci4_low, ci4_high = bootstrap_ci(preds, actual_4)
    t4_stat, t4_p = t_test(preds, actual_4)
    print(f"\n{name}:")
    print(f"  95% CI: [{ci4_low:.3f}, {ci4_high:.3f}], t-test p={t4_p:.4f}")

# Verdicts
for name, m in [('SSRF', ssrf_4_m), ('Ridge', ridge_4_m),
                ('Momentum', mom_4_m), ('Hist Mean', hist_4_m)]:
    ci4_low, _ = bootstrap_ci(m['total_pnl'] / len(actual_4) * np.ones(len(actual_4)),
                              actual_4) if False else (0, 0)  # skip CI for each
    v = "PASS" if m['hit_ratio'] > 50 and m['sharpe'] > 0 else "FAIL"
    print(f"{name:<15}: {v}")


# =============================================================================
# TEST 5: YIELD CURVE SPREAD PREDICTION
# =============================================================================
print_header("TEST 5: YIELD CURVE SPREAD PREDICTION")

SCALE_5 = 10.0
print(f"Prediction scale: {SCALE_5}x")

# Target: yield curve spread change
y_target = (fred['GS10'] - fred['TB3MS']).dropna()
y_target.name = 'YC_Spread'

X_5 = fred[get_feature_columns(fred)].ffill().bfill().fillna(0)

common_idx_5 = X_5.index.intersection(y_target.index)
X_5a = X_5.loc[common_idx_5]
y_5a = y_target.loc[common_idx_5]

y_change_5 = y_5a.diff().dropna()
y_change_5.name = 'YC_Spread_Change'

common_idx_5b = X_5a.index.intersection(y_change_5.index)
X_arr_5_raw = X_5a.loc[common_idx_5b].values
y_arr_5_raw = y_change_5.loc[common_idx_5b].values.flatten()
dates_5_raw = y_change_5.loc[common_idx_5b].index

# Temporal alignment: X[t] predicts y[t+1] (spread change next month)
X_arr_5 = X_arr_5_raw[:-1]
y_arr_5 = y_arr_5_raw[1:]
dates_5 = dates_5_raw[1:]

print(f"Periods: {len(X_arr_5)} ({dates_5[0].strftime('%Y-%m')} to {dates_5[-1].strftime('%Y-%m')})")
print(f"Mean spread change: {np.mean(y_arr_5):.4f}%")
print(f"Std spread change: {np.std(y_arr_5):.4f}%")

# Walk-forward
train_window_5 = 60
model_5_p, naive_5_p, random_5_p, hist_5_p, momentum_5_p = [], [], [], [], []
actual_5 = []

for i in tqdm(range(train_window_5, len(X_arr_5)), desc="Test 5: Yield Curve"):
    X_train_5 = X_arr_5[:i]
    y_train_5 = y_arr_5[:i] * SCALE_5
    X_test_5 = X_arr_5[i:i+1]

    naive_5_p.append(_naive())
    np.random.seed(i)
    random_5_p.append(_random())
    hist_5_p.append(_hist_mean(y_arr_5[:i]))
    momentum_5_p.append(_momentum(y_arr_5[i-1]) if len(momentum_5_p) > 0 else 0)

    try:
        scaler_5 = StandardScaler()
        X_train_5_s = scaler_5.fit_transform(X_train_5)
        X_test_5_s = scaler_5.transform(X_test_5)
        model_5 = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)
        model_5.fit(X_train_5_s, y_train_5)
        model_5_p.append(model_5.predict(X_test_5_s)[0])
    except:
        model_5_p.append(0)

    actual_5.append(y_arr_5[i])

model_5_p = np.array(model_5_p)
actual_5 = np.array(actual_5)

results_5 = {
    'SSRF': calc_metrics(model_5_p, actual_5),
    'Naive (0)': calc_metrics(np.array(naive_5_p), actual_5),
    'Random': calc_metrics(np.array(random_5_p), actual_5),
    'Hist. Mean': calc_metrics(np.array(hist_5_p), actual_5),
    'Momentum': calc_metrics(np.array(momentum_5_p), actual_5),
}

bh_total_5 = (np.prod(1 + np.array(actual_5) / 100) - 1) * 100 if len(actual_5) > 0 else 0

print_metrics_table(results_5)
print(f"{'Buy&Hold':<20} {'N/A':>7} {'N/A':>9} {'N/A':>7} {'N/A':>7} {bh_total_5:>11.1f}%")

# Diebold-Mariano (custom with HAR correction for yield)
def dm_yield(p1, p2, a, h=1):
    d = (a - p1)**2 - (a - p2)**2
    n = len(d)
    k = np.sqrt((n + 1 - 2*h + (h*(h+1))/3))
    if np.std(d) == 0 or k == 0:
        return 0, 1.0
    dm_stat = np.mean(d) / (np.std(d) / k)
    p_v = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_v

print("\n--- Statistical Tests ---")
p_perm_5 = permutation_test(model_5_p, actual_5, n_perms=1000)
print(f"Permutation Test: p={p_perm_5:.4f} {'SIG' if p_perm_5 < 0.05 else 'NOT SIG'}")

ci5_low, ci5_high = bootstrap_ci(model_5_p, actual_5)
print(f"Bootstrap 95% CI: [{ci5_low:.3f}, {ci5_high:.3f}]")

dm5_stat, dm5_p = dm_yield(model_5_p, np.array(momentum_5_p), actual_5)
print(f"Diebold-Mariano (vs Momentum): t={dm5_stat:.3f}, p={dm5_p:.4f} {'SIG' if dm5_p < 0.05 else 'NOT SIG'}")

pnl_5 = model_5_p * actual_5
t5_stat, t5_p = stats.ttest_1samp(pnl_5, 0)
print(f"t-test: t={t5_stat:.3f}, p={t5_p:.4f} {'SIG' if t5_p < 0.05 else 'NOT SIG'}")

# True OOS for yield curve
test_dates_5 = dates_5[train_window_5:train_window_5 + len(model_5_p)]
oos_mask_5 = test_dates_5 >= pd.Timestamp('2000-01-01')

print_header("TEST 5b: YIELD CURVE TRUE OOS (2000-2026)")
if np.sum(oos_mask_5) > 0:
    m5_oos = calc_metrics(model_5_p[oos_mask_5], actual_5[oos_mask_5])
    ci5o_low, ci5o_high = bootstrap_ci(model_5_p[oos_mask_5], actual_5[oos_mask_5])
    print(f"OOS periods: {np.sum(oos_mask_5)}")
    print(f"SSRF: Hit={m5_oos['hit_ratio']:.1f}%, Sharpe={m5_oos['sharpe']:.3f}")
    print(f"95% CI: [{ci5o_low:.3f}, {ci5o_high:.3f}]")
else:
    m5_oos = None
    print("No OOS periods available.")

# Final verdict for yield curve
if m5_oos is not None:
    print_verdict(f"SSRF Yield Curve (Scale={SCALE_5}x OOS)",
                  m5_oos['hit_ratio'], m5_oos['sharpe'],
                  ci5o_low, ci5o_high)
else:
    print_verdict(f"SSRF Yield Curve (Scale={SCALE_5}x Full)",
                  results_5['SSRF']['hit_ratio'], results_5['SSRF']['sharpe'],
                  ci5_low, ci5_high)


# =============================================================================
# FINAL SUMMARY TABLE
# =============================================================================
print_header("FINAL SUMMARY")
print(f"{'Test':<40} {'Hit%':>8} {'Sharpe':>8} {'95% CI Lower':>13} {'Verdict':>10}")
print("-" * 85)

def _verdict(hit, sh, ci_low):
    if hit > 50 and sh > 0 and ci_low > 0:
        return "PASS"
    elif hit > 50 and sh > 0:
        return "MARG"
    else:
        return "FAIL"

# Gather all results
rows = [
    ("1a: Proper Alignment (Full)", m1['hit_ratio'], m1['sharpe'], ci1_low),
    ("1b: Proper Alignment (OOS)", m1_oos['hit_ratio'], m1_oos['sharpe'], ci1_oos_low),
    ("2a: Expanding Window", m2a['hit_ratio'], m2a['sharpe'], 0),
    ("2b: Fixed Window (60m)", m2b['hit_ratio'], m2b['sharpe'], 0),
    ("2c: True OOS (Forward)", m2c['hit_ratio'], m2c['sharpe'], ci2_low),
    ("3a: Scale 20x (Full)", results_3a['SSRF']['hit_ratio'], results_3a['SSRF']['sharpe'], ci3a_low),
    ("3b: Scale 20x (OOS)", m3b['hit_ratio'], m3b['sharpe'], ci3b_low),
    ("4: SSRF vs Ridge (SSRF)", ssrf_4_m['hit_ratio'], ssrf_4_m['sharpe'], 0),
    ("4: SSRF vs Ridge (Ridge)", ridge_4_m['hit_ratio'], ridge_4_m['sharpe'], 0),
]

if m5_oos is not None:
    rows.append(("5b: Yield Curve (OOS)", m5_oos['hit_ratio'], m5_oos['sharpe'], ci5o_low))
else:
    rows.append(("5: Yield Curve (Full)", results_5['SSRF']['hit_ratio'], results_5['SSRF']['sharpe'], ci5_low))

for name, hit, sh, ci in rows:
    if ci != 0:
        print(f"{name:<40} {hit:>8.1f} {sh:>8.3f} {ci:>13.3f} {_verdict(hit, sh, ci):>10}")
    else:
        print(f"{name:<40} {hit:>8.1f} {sh:>8.3f} {'N/A':>13} {'N/A':>10}")

print("\n" + "=" * 70)
print("ALL TESTS COMPLETE")
print("=" * 70)

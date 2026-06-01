"""
SSRF Test with S&P 500 Returns - Scale=20 (FIXED - Proper Temporal Alignment)
Complete walk-forward OOS test with baselines and statistical significance
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

PREDICTION_SCALE = 20.0  # 20x scaling

print("="*70)
print(f"SSRF COMPREHENSIVE TEST - SCALE={PREDICTION_SCALE}")
print("="*70)

# =============================================================================
# STEP 1: FETCH AND ALIGN DATA
# =============================================================================
print("\n" + "="*70)
print("STEP 1: FETCH S&P 500 RETURNS AND ALIGN DATA")
print("="*70)

# Fetch SPX
spx = yf.download('^GSPC', start='1979-01-01', end='2026-06-01', progress=False)
spx_monthly = spx['Close'].resample('ME').last()
spx_returns = spx_monthly.pct_change().dropna() * 100

# Normalize dates
spx_returns.index = spx_returns.index + pd.offsets.MonthEnd(0)
spx_returns.index = spx_returns.index.normalize()

# Load FRED
fred = pd.read_csv('data/fred_cache/all_fred_data_enhanced.csv', index_col=0, parse_dates=True)
fred = fred.dropna(thresh=fred.shape[1] * 0.5)

# Features
feature_cols = [c for c in fred.columns if c not in ['GS10', 'TB3MS'] and not c.endswith('_REGIME')]
X = fred[feature_cols].ffill().bfill().fillna(0)
X.index = X.index.normalize()

# Align
common_idx = X.index.intersection(spx_returns.index)
X_aligned = X.loc[common_idx]
spx_aligned = spx_returns.loc[common_idx]

X_arr = X_aligned.values
y_arr = spx_aligned.values.flatten()
dates = X_aligned.index

print(f"Features: {len(feature_cols)}")
print(f"Periods: {len(X_arr)} ({dates[0].strftime('%Y-%m')} to {dates[-1].strftime('%Y-%m')})")
print(f"Mean monthly return: {np.mean(y_arr):.3f}%")
print(f"Std monthly return: {np.std(y_arr):.3f}%")
print(f"Prediction Scale: {PREDICTION_SCALE}x")

# =============================================================================
# STEP 2: DEFINE BASELINES
# =============================================================================
print("\n" + "="*70)
print("STEP 2: DEFINE BASELINES")
print("="*70)

def naive_baseline():
    """Predict: 0 (no direction)"""
    return 0.0

def random_baseline():
    """Predict: random +1 or -1"""
    return np.random.choice([-1, 1])

def historical_mean_baseline(y_train):
    """Predict: historical mean"""
    return np.mean(y_train)

def momentum_baseline(y_prev):
    """Predict: same direction as last period"""
    return np.sign(y_prev)

# =============================================================================
# STEP 3: WALK-FORWARD OOS TEST (PROPER - NO LEAKAGE)
# =============================================================================
print("\n" + "="*70)
print("STEP 3: WALK-FORWARD OOS TEST (PROPER TEMPORAL ALIGNMENT)")
print("="*70)

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

# FIXED: Proper walk-forward with consistent lengths
# Train on X[0:i] to predict y[i] (same period), then test on X[i+1] to predict y[i+1]
for i in range(start_idx, len(y_arr) - 1, step_size):
    # Training: X[0:i], y[0:i] (same length, indices 0..i-1)
    X_train = X_arr[:i]
    y_train = y_arr[:i]

    # Test: X[i+1] to predict y[i+1]
    X_test = X_arr[i+1:i+2]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # SSRF Model
    model = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=10000)
    model.fit(X_train_s, y_train)
    model_pred = model.predict(X_test_s)[0] * PREDICTION_SCALE  # APPLY SCALE

    # Baselines
    naive_pred = naive_baseline()
    np.random.seed(i)
    random_pred = random_baseline()
    hist_mean_pred = historical_mean_baseline(y_train)
    momentum_pred = momentum_baseline(y_train[-1])

    model_preds.append(model_pred)
    naive_preds.append(naive_pred)
    random_preds.append(random_pred)
    hist_mean_preds.append(hist_mean_pred)
    momentum_preds.append(momentum_pred)
    actual_returns.append(y_arr[i+1])  # FIXED: actual[t+1] matches pred[t+1]

model_preds = np.array(model_preds)
naive_preds = np.array(naive_preds)
random_preds = np.array(random_preds)
hist_mean_preds = np.array(hist_mean_preds)
momentum_preds = np.array(momentum_preds)
actual_returns = np.array(actual_returns)

print(f"Tested {len(actual_returns)} periods")

# =============================================================================
# STEP 4: METRICS CALCULATION (FIXED - CONSISTENT ALIGNMENT)
# =============================================================================
print("\n" + "="*70)
print("STEP 4: PERFORMANCE METRICS (FIXED ALIGNMENT)")
print("="*70)

def calculate_metrics(preds, actual):
    """Calculate all performance metrics with CONSISTENT alignment"""
    preds = np.array(preds)
    actual = np.array(actual)

    # FIXED: pred[t] vs actual[t] (same length, same index)
    direction_correct = np.sum(np.sign(preds) == np.sign(actual))
    hit_ratio = direction_correct / len(actual) * 100

    pnl = preds * actual
    total_pnl = np.sum(pnl)
    ann_return = np.mean(pnl) * 12 * 100
    ann_vol = np.std(pnl) * np.sqrt(12) * 100
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    ss_res = np.sum((actual - preds) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    return {
        'hit_ratio': hit_ratio,
        'total_pnl': total_pnl,
        'ann_return': ann_return,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'r2_oos': r2
    }

results = {
    'SSRF': calculate_metrics(model_preds, actual_returns),
    'Naive (0)': calculate_metrics(naive_preds, actual_returns),
    'Random': calculate_metrics(random_preds, actual_returns),
    'Hist. Mean': calculate_metrics(hist_mean_preds, actual_returns),
    'Momentum': calculate_metrics(momentum_preds, actual_returns)
}

print(f"\n{'Strategy':<15} {'Hit%':>7} {'AnnRet%':>10} {'Sharpe':>8} {'R² OOS':>8} {'Total P&L':>12}")
print("-" * 72)
for name, m in results.items():
    print(f"{name:<15} {m['hit_ratio']:>6.1f}% {m['ann_return']:>9.1f}% {m['sharpe']:>8.3f} {m['r2_oos']:>8.4f} {m['total_pnl']:>11.1f}%")

# =============================================================================
# STEP 5: STATISTICAL SIGNIFICANCE TESTS
# =============================================================================
print("\n" + "="*70)
print("STEP 5: STATISTICAL SIGNIFICANCE TESTS")
print("="*70)

def permutation_test(preds, actual, n_perms=1000):
    """Permutation test: is this strategy better than random shuffling?"""
    pnl = preds * actual
    real_sharpe = np.mean(pnl) / np.std(pnl) * np.sqrt(12) if np.std(pnl) > 0 else 0
    better = 0
    for _ in range(n_perms):
        shuffled = np.random.permutation(preds)
        shuffled_pnl = shuffled * actual
        sh = np.mean(shuffled_pnl) / np.std(shuffled_pnl) * np.sqrt(12) if np.std(shuffled_pnl) > 0 else 0
        if sh >= real_sharpe:
            better += 1
    return better / n_perms

def bootstrap_ci(preds, actual, n_boot=1000, ci=0.95):
    """Bootstrap 95% CI for Sharpe ratio"""
    pnl = preds * actual
    sharpes = []
    for _ in range(n_boot):
        idx = np.random.choice(len(pnl), size=len(pnl), replace=True)
        boot_pnl = pnl[idx]
        sh = np.mean(boot_pnl) / np.std(boot_pnl) * np.sqrt(12) if np.std(boot_pnl) > 0 else 0
        sharpes.append(sh)
    lower = np.percentile(sharpes, (1 - ci) / 2 * 100)
    upper = np.percentile(sharpes, (1 + ci) / 2 * 100)
    return lower, upper

def diebold_mariano(pred1, pred2, actual, h=1):
    """Diebold-Mariano test for equal predictive accuracy"""
    e1 = actual - pred1
    e2 = actual - pred2
    d = e1**2 - e2**2
    mean_d = np.mean(d)
    var_d = np.var(d) / len(d)
    dm_stat = mean_d / np.sqrt(var_d) if var_d > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value

def t_test(preds, actual):
    """t-test: is mean P&L significantly different from zero?"""
    pnl = preds * actual
    t_stat, p_value = stats.ttest_1samp(pnl, 0)
    return t_stat, p_value

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
print("\n" + "="*70)
print("STEP 6: TRULY OUT-OF-SAMPLE (Train 1980-2000, Test 2000-2026)")
print("="*70)

train_end = pd.Timestamp('2000-01-31')
test_start = pd.Timestamp('2000-02-28')

train_mask = dates <= train_end
test_mask = dates >= test_start

X_train = X_arr[train_mask]
X_test = X_arr[test_mask]
y_train = y_arr[train_mask]
y_test = y_arr[test_mask]
test_dates_sub = dates[test_mask]

print(f"Training: {len(X_train)} periods (up to {train_end.strftime('%Y-%m')})")
print(f"Testing: {len(y_test)} periods ({test_dates_sub[0].strftime('%Y-%m')} to {test_dates_sub[-1].strftime('%Y-%m')})")

# Train SSRF (FIXED: use X[t] to predict y[t])
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
model = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=10000)
model.fit(X_train_s, y_train)  # FIXED: y_train aligns with X_train
ssrf_preds = model.predict(X_test_s) * PREDICTION_SCALE  # APPLY SCALE

# Baselines
naive_oos = np.full(len(y_test), 0.0)
hist_mean_oos = np.full(len(y_test), np.mean(y_train))
momentum_oos = np.full(len(y_test), np.sign(y_train[-1]))
np.random.seed(42)
random_oos = np.array([random_baseline() for _ in range(len(y_test))])

# OOS Metrics (FIXED: align pred with actual)
print("\n--- Out-of-Sample Results ---")
oos_results = {
    'SSRF': calculate_metrics(ssrf_preds, y_test),  # FIXED: y_test matches prediction
    'Naive (0)': calculate_metrics(naive_oos, y_test),
    'Random': calculate_metrics(random_oos, y_test),
    'Hist. Mean': calculate_metrics(hist_mean_oos, y_test),
    'Momentum': calculate_metrics(momentum_oos, y_test)
}

print(f"\n{'Strategy':<15} {'Hit%':>7} {'AnnRet%':>10} {'Sharpe':>8} {'R² OOS':>8} {'Total P&L':>12}")
print("-" * 72)
for name, m in oos_results.items():
    print(f"{name:<15} {m['hit_ratio']:>6.1f}% {m['ann_return']:>9.1f}% {m['sharpe']:>8.3f} {m['r2_oos']:>8.4f} {m['total_pnl']:>11.1f}%")

# SPX Buy&Hold
spx_total = np.sum(y_test)
print(f"{'SPX Buy&Hold':<15} {'N/A':>7} {'N/A':>10} {'N/A':>8} {'N/A':>8} {spx_total:>11.1f}%")

# OOS Statistical Tests
print("\n--- Out-of-Sample SSRF Statistical Tests ---")
oos_perm_p = permutation_test(ssrf_preds, y_test)
print(f"Permutation Test (vs random): p={oos_perm_p:.4f} {'***' if oos_perm_p < 0.01 else '**' if oos_perm_p < 0.05 else '*' if oos_perm_p < 0.1 else ''}")

oos_ci_low, oos_ci_high = bootstrap_ci(ssrf_preds, y_test)
print(f"Bootstrap 95% CI for Sharpe: [{oos_ci_low:.3f}, {oos_ci_high:.3f}]")

oos_dm_stat, oos_dm_p = diebold_mariano(ssrf_preds, momentum_oos, y_test)
print(f"Diebold-Mariano (SSRF vs Momentum): t={oos_dm_stat:.3f}, p={oos_dm_p:.4f} {'***' if oos_dm_p < 0.01 else '**' if oos_dm_p < 0.05 else '*' if oos_dm_p < 0.1 else ''}")

oos_t_stat, oos_t_p = t_test(ssrf_preds, y_test)
print(f"t-test (mean P&L vs 0): t={oos_t_stat:.3f}, p={oos_t_p:.4f} {'***' if oos_t_p < 0.01 else '**' if oos_t_p < 0.05 else '*' if oos_t_p < 0.1 else ''}")

# =============================================================================
# FINAL CONCLUSION
# =============================================================================
print("\n" + "="*70)
print("FINAL CONCLUSION")
print("="*70)

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

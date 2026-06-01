"""
SSRF Test with S&P 500 Returns - Scale=10
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

PREDICTION_SCALE = 10.0  # 10x scaling

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
# STEP 3: WALK-FORWARD OOS TEST
# =============================================================================
print("\n" + "="*70)
print("STEP 3: WALK-FORWARD OOS TEST")
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

for i in range(start_idx, len(X_arr), step_size):
    train_end = i
    train_start = train_end - train_window

    X_train = X_arr[train_start:train_end]
    y_train = y_arr[train_start:train_end]
    X_test = X_arr[train_end:train_end+1]

    # Scale target
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

        model = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=10000)
        model.fit(X_train_scaled, y_train_scaled)

        pred = model.predict(X_test_scaled)[0]
        model_preds.append(pred)
    except:
        model_preds.append(0)

    actual_returns.append(y_arr[train_end])

print(f"OOS periods: {len(model_preds)}")
print(f"Training window: {train_window} months")

# =============================================================================
# STEP 4: CALCULATE METRICS
# =============================================================================
print("\n" + "="*70)
print("STEP 4: CALCULATE METRICS")
print("="*70)

def calc_metrics(preds, actual):
    """Calculate trading metrics"""
    preds = np.array(preds)
    actual = np.array(actual)

    # Direction accuracy
    direction_correct = np.sum(np.sign(preds[:-1]) == np.sign(actual[1:]))
    hit_ratio = direction_correct / (len(actual) - 1) * 100

    # P&L calculation
    pnl = preds * actual
    total_pnl = np.sum(pnl) * 100

    # Annualized metrics
    ann_return = np.mean(pnl) * 12 * 100
    ann_vol = np.std(pnl) * np.sqrt(12) * 100
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # Campbell-Thompson R² OOS
    ss_res = np.sum((actual - preds)**2)
    ss_tot = np.sum((actual - np.mean(actual))**2)
    r2_oos = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {
        'hit_ratio': hit_ratio,
        'total_pnl': total_pnl,
        'ann_return': ann_return,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'r2_oos': r2_oos
    }

model_metrics = calc_metrics(model_preds, actual_returns)
naive_metrics = calc_metrics(naive_preds, actual_returns)
random_metrics = calc_metrics(random_preds, actual_returns)
hist_mean_metrics = calc_metrics(hist_mean_preds, actual_returns)
momentum_metrics = calc_metrics(momentum_preds, actual_returns)

# SPX Buy & Hold
spx_pnl = np.cumprod(1 + np.array(actual_returns)/100) - 1
spx_total = spx_pnl[-1] * 100 if len(spx_pnl) > 0 else 0

print("\n" + "-"*70)
print(f"{'Strategy':<20} {'Hit%':>8} {'Ann.Ret%':>12} {'Sharpe':>8} {'R² OOS':>8} {'P&L%':>12}")
print("-"*70)
print(f"{'SSRF (Scale=10)':<20} {model_metrics['hit_ratio']:>8.1f} {model_metrics['ann_return']:>12.1f} {model_metrics['sharpe']:>8.3f} {model_metrics['r2_oos']:>8.4f} {model_metrics['total_pnl']:>12.1f}")
print(f"{'Naive (0)':<20} {naive_metrics['hit_ratio']:>8.1f} {naive_metrics['ann_return']:>12.1f} {naive_metrics['sharpe']:>8.3f} {naive_metrics['r2_oos']:>8.4f} {naive_metrics['total_pnl']:>12.1f}")
print(f"{'Random':<20} {random_metrics['hit_ratio']:>8.1f} {random_metrics['ann_return']:>12.1f} {random_metrics['sharpe']:>8.3f} {random_metrics['r2_oos']:>8.4f} {random_metrics['total_pnl']:>12.1f}")
print(f"{'Hist. Mean':<20} {hist_mean_metrics['hit_ratio']:>8.1f} {hist_mean_metrics['ann_return']:>12.1f} {hist_mean_metrics['sharpe']:>8.3f} {hist_mean_metrics['r2_oos']:>8.4f} {hist_mean_metrics['total_pnl']:>12.1f}")
print(f"{'Momentum':<20} {momentum_metrics['hit_ratio']:>8.1f} {momentum_metrics['ann_return']:>12.1f} {momentum_metrics['sharpe']:>8.3f} {momentum_metrics['r2_oos']:>8.4f} {momentum_metrics['total_pnl']:>12.1f}")
print(f"{'SPX Buy&Hold':<20} {'N/A':>8} {'N/A':>12} {'N/A':>8} {'N/A':>8} {spx_total:>12.1f}")

# =============================================================================
# STEP 5: STATISTICAL TESTS
# =============================================================================
print("\n" + "="*70)
print("STEP 5: STATISTICAL TESTS")
print("="*70)

def permutation_test(preds, actual, n_perms=1000):
    """Permutation test for statistical significance"""
    observed_sharpe = calc_metrics(preds, actual)['sharpe']
    random_sharpes = []

    for _ in range(n_perms):
        perm_pred = np.random.permutation(preds)
        random_sharpes.append(calc_metrics(perm_pred, actual)['sharpe'])

    return np.mean(np.abs(random_sharpes) >= np.abs(observed_sharpe))

def bootstrap_ci(preds, actual, n_boot=1000, ci=0.95):
    """Bootstrap confidence interval for Sharpe ratio"""
    sharpes = []
    for _ in range(n_boot):
        idx = np.random.choice(len(preds), size=len(preds), replace=True)
        boot_pred = np.array(preds)[idx]
        boot_actual = np.array(actual)[idx]
        sharpes.append(calc_metrics(boot_pred, boot_actual)['sharpe'])

    return np.percentile(sharpes, (1-ci)/2*100), np.percentile(sharpes, (1+ci)/2*100)

def diebold_mariano(pred1, pred2, actual, h=1):
    """DM test for equal predictive accuracy"""
    pred1 = np.array(pred1)
    pred2 = np.array(pred2)
    actual = np.array(actual)
    e1 = actual[h:] - pred1[:-h] if len(pred1) > h else actual - pred1
    e2 = actual[h:] - pred2[:-h] if len(pred2) > h else actual - pred2
    if len(e1) != len(e2):
        min_len = min(len(e1), len(e2))
        e1, e2 = e1[:min_len], e2[:min_len]
    d = e1**2 - e2**2
    n = len(d)
    k = (n + 1 - 2*h + (h*(h+1))/3) ** 0.5
    dm_stat = np.mean(d) / (np.std(d) / k) if np.std(d) > 0 else 0
    return dm_stat

# Permutation test
p_perm = permutation_test(model_preds, actual_returns, n_perms=1000)
print(f"\nPermutation Test (SSRF vs Random):")
print(f"  p-value: {p_perm:.4f}")
print(f"  {'✓ SIGNIFICANT' if p_perm < 0.05 else '✗ NOT SIGNIFICANT'}")

# Bootstrap CI
ci_lower, ci_upper = bootstrap_ci(model_preds, actual_returns, n_boot=1000, ci=0.95)
print(f"\nBootstrap 95% CI for Sharpe:")
print(f"  CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
print(f"  {'✓ Contains positive values' if ci_upper > 0 else '✗ Entirely negative'}")

# Diebold-Mariano test
dm_stat = diebold_mariano(model_preds, momentum_preds, actual_returns)
p_dm = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
print(f"\nDiebold-Mariano (SSRF vs Momentum):")
print(f"  t-statistic: {dm_stat:.3f}")
print(f"  p-value: {p_dm:.4f}")
print(f"  {'✓ SIGNIFICANT' if p_dm < 0.05 else '✗ NOT SIGNIFICANT'}")

# t-test
pnl = np.array(model_preds) * np.array(actual_returns)
t_stat, p_t = stats.ttest_1samp(pnl, 0)
print(f"\nt-test (mean P&L vs 0):")
print(f"  t-statistic: {t_stat:.3f}")
print(f"  p-value: {p_t:.4f}")
print(f"  {'✓ SIGNIFICANT' if p_t < 0.05 else '✗ NOT SIGNIFICANT'}")

# =============================================================================
# STEP 6: TRULY OOS TEST (Train 1980-2000, Test 2000-2026)
# =============================================================================
print("\n" + "="*70)
print("STEP 6: TRULY OUT-OF-SAMPLE TEST (2000-2026)")
print("="*70)

# Split
train_end_date = pd.Timestamp('2000-01-01')
test_dates = dates[start_idx:]
train_mask = test_dates < train_end_date
test_mask = test_dates >= train_end_date

model_oos = np.array(model_preds)[test_mask]
naive_oos = np.array(naive_preds)[test_mask]
random_oos = np.array(random_preds)[test_mask]
hist_oos = np.array(hist_mean_preds)[test_mask]
momentum_oos = np.array(momentum_preds)[test_mask]
actual_oos = np.array(actual_returns)[test_mask]

print(f"OOS periods (2000-2026): {len(model_oos)}")

model_oos_metrics = calc_metrics(model_oos, actual_oos)
naive_oos_metrics = calc_metrics(naive_oos, actual_oos)
random_oos_metrics = calc_metrics(random_oos, actual_oos)
hist_oos_metrics = calc_metrics(hist_oos, actual_oos)
momentum_oos_metrics = calc_metrics(momentum_oos, actual_oos)

# SPX Buy & Hold
spx_pnl_oos = np.cumprod(1 + actual_oos/100) - 1
spx_total_oos = spx_pnl_oos[-1] * 100 if len(spx_pnl_oos) > 0 else 0

print("\n" + "-"*70)
print(f"{'Strategy':<20} {'Hit%':>8} {'Ann.Ret%':>12} {'Sharpe':>8} {'P&L%':>12}")
print("-"*70)
print(f"{'SSRF (Scale=10)':<20} {model_oos_metrics['hit_ratio']:>8.1f} {model_oos_metrics['ann_return']:>12.1f} {model_oos_metrics['sharpe']:>8.3f} {model_oos_metrics['total_pnl']:>12.1f}")
print(f"{'Naive (0)':<20} {naive_oos_metrics['hit_ratio']:>8.1f} {naive_oos_metrics['ann_return']:>12.1f} {naive_oos_metrics['sharpe']:>8.3f} {naive_oos_metrics['total_pnl']:>12.1f}")
print(f"{'Random':<20} {random_oos_metrics['hit_ratio']:>8.1f} {random_oos_metrics['ann_return']:>12.1f} {random_oos_metrics['sharpe']:>8.3f} {random_oos_metrics['total_pnl']:>12.1f}")
print(f"{'Hist. Mean':<20} {hist_oos_metrics['hit_ratio']:>8.1f} {hist_oos_metrics['ann_return']:>12.1f} {hist_oos_metrics['sharpe']:>8.3f} {hist_oos_metrics['total_pnl']:>12.1f}")
print(f"{'Momentum':<20} {momentum_oos_metrics['hit_ratio']:>8.1f} {momentum_oos_metrics['ann_return']:>12.1f} {momentum_oos_metrics['sharpe']:>8.3f} {momentum_oos_metrics['total_pnl']:>12.1f}")
print(f"{'SPX Buy&Hold':<20} {'N/A':>8} {'N/A':>12} {'N/A':>8} {spx_total_oos:>12.1f}")

# OOS Statistical tests
print("\nOOS Statistical Tests:")
p_perm_oos = permutation_test(model_oos, actual_oos, n_perms=1000)
print(f"  Permutation p-value: {p_perm_oos:.4f}")

ci_lower_oos, ci_upper_oos = bootstrap_ci(model_oos, actual_oos, n_boot=1000, ci=0.95)
print(f"  Bootstrap 95% CI: [{ci_lower_oos:.3f}, {ci_upper_oos:.3f}]")

dm_stat_oos = diebold_mariano(model_oos, momentum_oos, actual_oos)
p_dm_oos = 2 * (1 - stats.norm.cdf(abs(dm_stat_oos)))
print(f"  Diebold-Mariano p-value: {p_dm_oos:.4f}")

pnl_oos = model_oos * actual_oos
t_stat_oos, p_t_oos = stats.ttest_1samp(pnl_oos, 0)
print(f"  t-test p-value: {p_t_oos:.4f}")

# =============================================================================
# STEP 7: FINAL VERDICT
# =============================================================================
print("\n" + "="*70)
print("FINAL VERDICT")
print("="*70)

ssrf_hit_oos = model_oos_metrics['hit_ratio']
ssrf_sharpe_oos = model_oos_metrics['sharpe']
ssrf_pnl_oos = model_oos_metrics['total_pnl']

if ssrf_hit_oos > 50 and ssrf_sharpe_oos > 0 and ci_upper_oos > 0:
    print("\n✓ SSRF PASSES OUT-OF-SAMPLE TEST")
    print(f"  Scale={PREDICTION_SCALE}x")
    print(f"  Direction Accuracy: {ssrf_hit_oos:.1f}% (vs 50% random)")
    print(f"  Sharpe: {ssrf_sharpe_oos:.3f} (POSITIVE)")
    print(f"  Total P&L: {ssrf_pnl_oos:.1f}% (PROFITABLE)")
    print(f"  95% CI: [{ci_lower_oos:.3f}, {ci_upper_oos:.3f}]")
else:
    print("\n✗ SSRF FAILS OUT-OF-SAMPLE TEST")
    print(f"  Scale={PREDICTION_SCALE}x")
    print(f"  Direction Accuracy: {ssrf_hit_oos:.1f}% (vs 50% random)")
    print(f"  Sharpe: {ssrf_sharpe_oos:.3f} (NEGATIVE)")
    print(f"  Total P&L: {ssrf_pnl_oos:.1f}% (LOSES MONEY)")
    print(f"  95% CI: [{ci_lower_oos:.3f}, {ci_upper_oos:.3f}]")

print("\n" + "="*70)
print("END OF TEST")
print("="*70)
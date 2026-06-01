"""
Comprehensive SSRF Test with S&P 500 Returns
Step 1: Fetch SPX data
Step 2: Create baselines (naive, random, historical mean)
Step 3: Statistical significance tests
Step 4: Truly out-of-sample test
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
print("COMPREHENSIVE SSRF TEST WITH S&P 500 RETURNS")
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
spx_returns.name = 'SPX_RETURN'

# Load FRED
fred = pd.read_csv('data/fred_cache/all_fred_data_enhanced.csv', index_col=0, parse_dates=True)
fred = fred.dropna(thresh=fred.shape[1] * 0.5)

# Features
feature_cols = [c for c in fred.columns if c not in ['GS10', 'TB3MS'] and not c.endswith('_REGIME')]
X = fred[feature_cols].ffill().bfill().fillna(0)

# Align
common_idx = X.index.intersection(spx_returns.index)
X_aligned = X.loc[common_idx]
spx_aligned = spx_returns.loc[common_idx]

print(f"Features: {len(feature_cols)}")
print(f"Aligned periods: {len(X_aligned)}")
print(f"Period: {X_aligned.index[0].strftime('%Y-%m')} to {X_aligned.index[-1].strftime('%Y-%m')}")

# Convert to arrays
X_arr = X_aligned.values
y_arr = spx_aligned.values.flatten()  # Ensure 1D array
dates = X_aligned.index

print(f"SPX Mean monthly return: {np.mean(y_arr):.3f}%")
print(f"SPX Std monthly return: {np.std(y_arr):.3f}%")

# =============================================================================
# STEP 2: DEFINE BASELINES
# =============================================================================
print("\n" + "="*70)
print("STEP 2: DEFINE BASELINES")
print("="*70)

def naive_baseline():
    """Predict: return = 0 (no direction)"""
    return 0.0

def random_baseline(y_val):
    """Predict: random +1 or -1 scaled by |y|"""
    return np.random.choice([-1, 1]) * np.abs(y_val)

def historical_mean_baseline(y_train):
    """Predict: use historical mean return"""
    return np.mean(y_train)

def momentum_baseline(y_prev):
    """Predict: same direction as last period"""
    return np.sign(y_prev)

# =============================================================================
# STEP 3: WALK-FORWARD OOS TEST
# =============================================================================
print("\n" + "="*70)
print("STEP 3: WALK-FORWARD OUT-OF-SAMPLE TEST")
print("="*70)

train_window = 60  # 5 years
step_size = 1
start_idx = train_window

model_preds = []
naive_preds = []
random_preds = []
hist_mean_preds = []
momentum_preds = []
actual = []
train_dates = []
test_dates = []

print(f"Running walk-forward test...")
print(f"Train window: {train_window} months")
print(f"Total test periods: {len(y_arr) - start_idx}")

for i in range(start_idx, len(y_arr) - 1, step_size):
    # Split
    X_train, X_test = X_arr[i - train_window:i], X_arr[i:i+1]
    y_train, y_test = y_arr[i - train_window:i], y_arr[i]

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # SSRF Model (ElasticNet)
    model = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=10000)
    model.fit(X_train_s, y_train)
    model_pred = model.predict(X_test_s)[0]

    # Baselines
    naive_pred = naive_baseline()
    np.random.seed(i)
    random_pred = random_baseline(y_test)
    hist_mean_pred = historical_mean_baseline(y_train)
    momentum_pred = momentum_baseline(y_train[-1])

    # Store
    model_preds.append(model_pred)
    naive_preds.append(naive_pred)
    random_preds.append(random_pred)
    hist_mean_preds.append(hist_mean_pred)
    momentum_preds.append(momentum_pred)
    actual.append(y_test)
    test_dates.append(dates[i])

# Convert to arrays
model_preds = np.array(model_preds)
naive_preds = np.array(naive_preds)
random_preds = np.array(random_preds)
hist_mean_preds = np.array(hist_mean_preds)
momentum_preds = np.array(momentum_preds)
actual = np.array(actual)

print(f"Tested {len(actual)} periods")

# =============================================================================
# STEP 4: STATISTICAL SIGNIFICANCE TESTS
# =============================================================================
print("\n" + "="*70)
print("STEP 4: STATISTICAL SIGNIFICANCE TESTS")
print("="*70)

def calculate_metrics(preds, actual):
    """Calculate performance metrics"""
    # Direction accuracy
    if len(preds) > 1:
        hit = np.mean(np.sign(preds[:-1]) == np.sign(actual[1:])) * 100
    else:
        hit = np.nan

    # P&L
    pnl = preds * actual
    total_pnl = np.sum(pnl)

    # Sharpe-like (mean / std of P&L)
    sharpe = np.mean(pnl) / np.std(pnl) * np.sqrt(12) if np.std(pnl) > 0 else 0

    # R² OOS
    ss_res = np.sum((actual - preds) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    return {
        'hit_ratio': hit,
        'total_pnl': total_pnl,
        'sharpe': sharpe,
        'r2_oos': r2,
        'mean_pred': np.mean(preds),
        'std_pred': np.std(preds)
    }

def permutation_test(model_preds, actual, n_permutations=1000):
    """Permutation test: is model significantly better than random?"""
    model_pnl = model_preds * actual
    model_sharpe = np.mean(model_pnl) / np.std(model_pnl) if np.std(model_pnl) > 0 else 0

    better_count = 0
    for _ in range(n_permutations):
        shuffled_preds = np.random.permutation(model_preds)
        shuffled_pnl = shuffled_preds * actual
        shuffled_sharpe = np.mean(shuffled_pnl) / np.std(shuffled_pnl) if np.std(shuffled_pnl) > 0 else 0
        if shuffled_sharpe >= model_sharpe:
            better_count += 1

    p_value = better_count / n_permutations
    return p_value

def bootstrap_ci(preds, actual, n_bootstrap=1000, ci=0.95):
    """Bootstrap confidence interval for Sharpe ratio"""
    pnl = preds * actual
    sharpe = np.mean(pnl) / np.std(pnl) * np.sqrt(12) if np.std(pnl) > 0 else 0

    sharpes = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(pnl), size=len(pnl), replace=True)
        boot_pnl = pnl[idx]
        boot_sharpe = np.mean(boot_pnl) / np.std(boot_pnl) * np.sqrt(12) if np.std(boot_pnl) > 0 else 0
        sharpes.append(boot_sharpe)

    lower = np.percentile(sharpes, (1 - ci) / 2 * 100)
    upper = np.percentile(sharpes, (1 + ci) / 2 * 100)

    return lower, upper, sharpe

# Calculate all metrics
print("\n--- Performance Metrics ---")
results = {
    'SSRF Model': calculate_metrics(model_preds, actual),
    'Naive (0)': calculate_metrics(naive_preds, actual),
    'Random': calculate_metrics(random_preds, actual),
    'Historical Mean': calculate_metrics(hist_mean_preds, actual),
    'Momentum': calculate_metrics(momentum_preds, actual)
}

print(f"\n{'Strategy':<20} {'Hit%':>8} {'Sharpe':>8} {'R² OOS':>8} {'Total P&L':>12}")
print("-" * 60)
for name, metrics in results.items():
    print(f"{name:<20} {metrics['hit_ratio']:>7.1f}% {metrics['sharpe']:>8.3f} {metrics['r2_oos']:>8.4f} {metrics['total_pnl']:>11.1f}%")

# Statistical tests
print("\n--- Statistical Significance Tests ---")

# Permutation test
print("\nPermutation Test (Is SSRF better than random shuffling?):")
p_value = permutation_test(model_preds, actual, n_permutations=1000)
print(f"  p-value: {p_value:.4f}")
print(f"  Significant at 5%: {'YES' if p_value < 0.05 else 'NO'}")

# Bootstrap CI
print("\nBootstrap 95% CI for Sharpe Ratio:")
ci_lower, ci_upper, sharpe = bootstrap_ci(model_preds, actual, n_bootstrap=1000)
print(f"  SSRF Sharpe: {sharpe:.3f}")
print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

# Diebold-Mariano test (simplified)
print("\nDiebold-Mariano Test (Is SSRF better than momentum?):")
from scipy.stats import norm

# Simplified DM test
ssrf_pnl = model_preds * actual
momentum_pnl = momentum_preds * actual

diff = ssrf_pnl[:-1] - momentum_pnl[:-1]
dm_stat = np.mean(diff) / (np.std(diff) / np.sqrt(len(diff))) if np.std(diff) > 0 else 0
dm_pvalue = 2 * (1 - norm.cdf(abs(dm_stat)))

print(f"  DM statistic: {dm_stat:.3f}")
print(f"  p-value: {dm_pvalue:.4f}")
print(f"  SSRF beats Momentum at 5%: {'YES' if dm_pvalue < 0.05 else 'NO'}")

# =============================================================================
# STEP 5: TRULY OUT-OF-SAMPLE TEST
# =============================================================================
print("\n" + "="*70)
print("STEP 5: TRULY OUT-OF-SAMPLE TEST (Train 1980-2000, Test 2000-2026)")
print("="*70)

# Split by time
train_end_date = pd.Timestamp('2000-01-31')
test_start_date = pd.Timestamp('2000-02-28')

train_mask = dates <= train_end_date
test_mask = dates >= test_start_date

X_train = X_arr[train_mask]
X_test = X_arr[test_mask]
y_train = y_arr[train_mask]
y_test = y_arr[test_mask]
test_dates_sub = dates[test_mask]

print(f"Training: {len(X_train)} periods (up to {train_end_date.strftime('%Y-%m')})")
print(f"Testing: {len(X_test)} periods ({test_dates_sub[0].strftime('%Y-%m')} to {test_dates_sub[-1].strftime('%Y-%m')})")

# Train on 1980-2000
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=10000)
model.fit(X_train_s, y_train)
oos_preds = model.predict(X_test_s)

# Momentum baseline
momentum_oos = np.sign(y_test)

# Calculate metrics
print("\n--- Out-of-Sample Results (2000-2026) ---")
oos_results = {
    'SSRF Model': calculate_metrics(oos_preds, y_test),
    'Momentum': calculate_metrics(momentum_oos, y_test),
    'Historical Mean': calculate_metrics(np.full(len(y_test), y_train.mean()), y_test)
}

print(f"\n{'Strategy':<20} {'Hit%':>8} {'Sharpe':>8} {'R² OOS':>8} {'Total P&L':>12}")
print("-" * 60)
for name, metrics in oos_results.items():
    print(f"{name:<20} {metrics['hit_ratio']:>7.1f}% {metrics['sharpe']:>8.3f} {metrics['r2_oos']:>8.4f} {metrics['total_pnl']:>11.1f}%")

# SPX Buy and Hold
spx_total = np.sum(y_test)
print(f"{'SPX Buy&Hold':<20} {'N/A':>8} {'N/A':>8} {'N/A':>8} {spx_total:>11.1f}%")

# Bootstrap CI for OOS
print("\nBootstrap 95% CI for OOS Sharpe:")
ci_lower_oos, ci_upper_oos, sharpe_oos = bootstrap_ci(oos_preds, y_test, n_bootstrap=1000)
print(f"  OOS Sharpe: {sharpe_oos:.3f}")
print(f"  95% CI: [{ci_lower_oos:.3f}, {ci_upper_oos:.3f}]")

# =============================================================================
# FINAL CONCLUSION
# =============================================================================
print("\n" + "="*70)
print("FINAL CONCLUSION")
print("="*70)
print()

ssrf_sharpe = results['SSRF Model']['sharpe']
ssrf_hit = results['SSRF Model']['hit_ratio']
naive_sharpe = results['Naive (0)']['sharpe']
momentum_sharpe = results['Momentum']['sharpe']

if p_value < 0.05 and ssrf_sharpe > momentum_sharpe:
    print("✅ SSRF MODEL SHOWS STATISTICALLY SIGNIFICANT IMPROVEMENT")
    print(f"   over random baseline (p={p_value:.4f})")
elif ssrf_sharpe > 0.5 and ssrf_hit > 52:
    print("⚠️ SSRF MODEL HAS MODEST PREDICTIVE POWER")
    print(f"   Direction accuracy: {ssrf_hit:.1f}%")
    print(f"   Sharpe ratio: {ssrf_sharpe:.3f}")
else:
    print("❌ SSRF MODEL DOES NOT SIGNIFICANTLY BEAT BASELINES")

print()
print(f"Full Sample Results:")
print(f"  SSRF Direction Accuracy: {ssrf_hit:.1f}%")
print(f"  SSRF Sharpe: {ssrf_sharpe:.3f}")
print(f"  Momentum Sharpe: {momentum_sharpe:.3f}")
print()
print(f"Out-of-Sample Results (2000-2026):")
print(f"  SSRF Direction Accuracy: {oos_results['SSRF Model']['hit_ratio']:.1f}%")
print(f"  SSRF Sharpe: {oos_results['SSRF Model']['sharpe']:.3f}")
print(f"  SPX Buy&Hold Total: {spx_total:.1f}%")

print("\n" + "="*70)
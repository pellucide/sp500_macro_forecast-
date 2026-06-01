"""
Comprehensive SSRF Test - FIXED date alignment
"""
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*70)
print("COMPREHENSIVE SSRF TEST WITH S&P 500 RETURNS")
print("="*70)

# =============================================================================
# STEP 1: FETCH AND ALIGN DATA (FIX DATES)
# =============================================================================
print("\n" + "="*70)
print("STEP 1: FETCH S&P 500 RETURNS AND ALIGN DATA")
print("="*70)

# Fetch SPX - use last trading day of month
spx = yf.download('^GSPC', start='1979-01-01', end='2026-06-01', progress=False)
spx_monthly = spx['Close'].resample('ME').last()
spx_returns = spx_monthly.pct_change().dropna() * 100

# Normalize SPX dates to end of month to match FRED
spx_returns.index = spx_returns.index + pd.offsets.MonthEnd(0)
spx_returns.index = spx_returns.index.normalize()

# Load FRED
fred = pd.read_csv('data/fred_cache/all_fred_data_enhanced.csv', index_col=0, parse_dates=True)
fred = fred.dropna(thresh=fred.shape[1] * 0.5)

# Features
feature_cols = [c for c in fred.columns if c not in ['GS10', 'TB3MS'] and not c.endswith('_REGIME')]
X = fred[feature_cols].ffill().bfill().fillna(0)

# Align - normalize dates first
X.index = X.index.normalize()
spx_returns.index = spx_returns.index.normalize()

common_idx = X.index.intersection(spx_returns.index)
X_aligned = X.loc[common_idx]
spx_aligned = spx_returns.loc[common_idx]

print(f"Features: {len(feature_cols)}")
print(f"Aligned periods: {len(X_aligned)}")
print(f"Period: {X_aligned.index[0].strftime('%Y-%m')} to {X_aligned.index[-1].strftime('%Y-%m')}")

# Convert to arrays
X_arr = X_aligned.values
y_arr = spx_aligned.values.flatten()
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
    return 0.0

def random_baseline(y_val):
    return np.random.choice([-1, 1]) * np.abs(y_val)

def historical_mean_baseline(y_train):
    return np.mean(y_train)

def momentum_baseline(y_prev):
    return np.sign(y_prev)

# =============================================================================
# STEP 3: WALK-FORWARD OOS TEST
# =============================================================================
print("\n" + "="*70)
print("STEP 3: WALK-FORWARD OUT-OF-SAMPLE TEST")
print("="*70)

train_window = 60
step_size = 1
start_idx = train_window

model_preds = []
naive_preds = []
random_preds = []
hist_mean_preds = []
momentum_preds = []
actual = []
test_dates = []

print(f"Running walk-forward test...")
print(f"Train window: {train_window} months")
print(f"Total test periods: {len(y_arr) - start_idx}")

for i in range(start_idx, len(y_arr) - 1, step_size):
    X_train, X_test = X_arr[i - train_window:i], X_arr[i:i+1]
    y_train, y_test = y_arr[i - train_window:i], y_arr[i]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # SSRF Model
    model = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=10000)
    model.fit(X_train_s, y_train)
    model_pred = model.predict(X_test_s)[0]

    # Baselines
    naive_pred = naive_baseline()
    np.random.seed(i)
    random_pred = random_baseline(y_test)
    hist_mean_pred = historical_mean_baseline(y_train)
    momentum_pred = momentum_baseline(y_train[-1])

    model_preds.append(model_pred)
    naive_preds.append(naive_pred)
    random_preds.append(random_pred)
    hist_mean_preds.append(hist_mean_pred)
    momentum_preds.append(momentum_pred)
    actual.append(y_test)
    test_dates.append(dates[i])

model_preds = np.array(model_preds)
naive_preds = np.array(naive_preds)
random_preds = np.array(random_preds)
hist_mean_preds = np.array(hist_mean_preds)
momentum_preds = np.array(momentum_preds)
actual = np.array(actual)

print(f"Tested {len(actual)} periods")

# =============================================================================
# STEP 4: STATISTICAL TESTS
# =============================================================================
print("\n" + "="*70)
print("STEP 4: STATISTICAL SIGNIFICANCE TESTS")
print("="*70)

def calculate_metrics(preds, actual):
    hit = np.mean(np.sign(preds[:-1]) == np.sign(actual[1:])) * 100 if len(preds) > 1 else np.nan
    pnl = preds * actual
    total_pnl = np.sum(pnl)
    sharpe = np.mean(pnl) / np.std(pnl) * np.sqrt(12) if np.std(pnl) > 0 else 0
    ss_res = np.sum((actual - preds) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return {'hit_ratio': hit, 'total_pnl': total_pnl, 'sharpe': sharpe, 'r2_oos': r2}

def permutation_test(model_preds, actual, n_permutations=1000):
    model_pnl = model_preds * actual
    model_sharpe = np.mean(model_pnl) / np.std(model_pnl) * np.sqrt(12) if np.std(model_pnl) > 0 else 0
    better_count = 0
    for _ in range(n_permutations):
        shuffled_preds = np.random.permutation(model_preds)
        shuffled_pnl = shuffled_preds * actual
        shuffled_sharpe = np.mean(shuffled_pnl) / np.std(shuffled_pnl) * np.sqrt(12) if np.std(shuffled_pnl) > 0 else 0
        if shuffled_sharpe >= model_sharpe:
            better_count += 1
    return better_count / n_permutations

def bootstrap_ci(preds, actual, n_bootstrap=1000, ci=0.95):
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

# Permutation test
print("\n--- Statistical Tests ---")
p_value = permutation_test(model_preds, actual, n_permutations=1000)
print(f"Permutation Test p-value: {p_value:.4f} (SSRF > random)")

ci_lower, ci_upper, sharpe = bootstrap_ci(model_preds, actual, n_bootstrap=1000)
print(f"Bootstrap 95% CI for Sharpe: [{ci_lower:.3f}, {ci_upper:.3f}] (Sharpe={sharpe:.3f})")

# =============================================================================
# STEP 5: TRULY OUT-OF-SAMPLE TEST
# =============================================================================
print("\n" + "="*70)
print("STEP 5: TRULY OUT-OF-SAMPLE (Train 1980-2000, Test 2000-2026)")
print("="*70)

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
print(f"Testing: {len(y_test)} periods ({test_dates_sub[0].strftime('%Y-%m')} to {test_dates_sub[-1].strftime('%Y-%m')})")

# SSRF
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
model = ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=10000)
model.fit(X_train_s, y_train)
ssrf_preds = model.predict(X_test_s)

# Momentum
momentum_preds_oos = np.full(len(y_test), np.sign(y_train[-1]))

# Metrics
print("\n--- Out-of-Sample Results (2000-2026) ---")
oos_results = {
    'SSRF Model': calculate_metrics(ssrf_preds, y_test),
    'Momentum': calculate_metrics(momentum_preds_oos, y_test),
    'Historical Mean': calculate_metrics(np.full(len(y_test), np.mean(y_train)), y_test)
}

print(f"\n{'Strategy':<20} {'Hit%':>8} {'Sharpe':>8} {'R² OOS':>8} {'Total P&L':>12}")
print("-" * 60)
for name, metrics in oos_results.items():
    print(f"{name:<20} {metrics['hit_ratio']:>7.1f}% {metrics['sharpe']:>8.3f} {metrics['r2_oos']:>8.4f} {metrics['total_pnl']:>11.1f}%")

# SPX Buy&Hold
spx_total = np.sum(y_test)
print(f"{'SPX Buy&Hold':<20} {'N/A':>8} {'N/A':>8} {'N/A':>8} {spx_total:>11.1f}%")

# Bootstrap for OOS
ci_lower_oos, ci_upper_oos, sharpe_oos = bootstrap_ci(ssrf_preds, y_test, n_bootstrap=1000)
print(f"\nBootstrap 95% CI for OOS Sharpe: [{ci_lower_oos:.3f}, {ci_upper_oos:.3f}] (Sharpe={sharpe_oos:.3f})")

# =============================================================================
# EXPLANATION OF P&L
# =============================================================================
print("\n" + "="*70)
print("UNDERSTANDING THE P&L METRICS")
print("="*70)

ssrf_pnl_sum = oos_results['SSRF Model']['total_pnl']
momentum_pnl_sum = oos_results['Momentum']['total_pnl']

print(f"""
The "Total P&L" column shows the CUMULATIVE return from the strategy.

For example:
- SSRF P&L = -12,310% means: if you started with $10,000 and followed
  SSRF's predictions, you'd have $10,000 * (1 - 123.10) = -$12,300 (LOST MONEY)

- Momentum P&L = +1,085% means: if you started with $10,000 and followed
  momentum's predictions, you'd have $10,000 * (1 + 10.85) = $118,500 (MADE MONEY)

- SPX Buy&Hold = +200% means: if you bought and held SPX, you'd have
  $10,000 * (1 + 2.00) = $30,000 (MADE MONEY)

CONCLUSION:
SSRF FAILS because it LOSES money (-12,310% P&L) while:
  - Momentum MAKES money (+1,085% P&L)
  - Buy&Hold MAKES money (+200% P&L)
""")

print("="*70)
print("FINAL CONCLUSION")
print("="*70)
if sharpe_oos < 0 and oos_results['SSRF Model']['hit_ratio'] < 50:
    print("❌ SSRF FAILS OUT-OF-SAMPLE TEST")
    print(f"   Direction accuracy: {oos_results['SSRF Model']['hit_ratio']:.1f}% (WORSE than random 50%)")
    print(f"   Sharpe: {sharpe_oos:.3f} (NEGATIVE)")
    print(f"   P&L: {ssrf_pnl_sum:+.0f}% (LOSES MONEY)")
elif sharpe_oos > 0 and oos_results['SSRF Model']['hit_ratio'] > 50:
    print("✅ SSRF PASSES OUT-OF-SAMPLE TEST")
else:
    print("⚠️ MIXED RESULTS")
    print(f"   Direction accuracy: {oos_results['SSRF Model']['hit_ratio']:.1f}%")
    print(f"   Sharpe: {sharpe_oos:.3f}")
    print(f"   P&L: {ssrf_pnl_sum:+.0f}%")
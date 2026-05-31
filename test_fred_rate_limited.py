"""
SSRF Test with Rate-Limited FRED Fetching
Fetches data sequentially to avoid rate limits
"""

import sys
import os
import time
import warnings

os.chdir('/workspace/sp500_macro_forecast')
sys.path.insert(0, '/workspace/sp500_macro_forecast')

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

print("=" * 75)
print("SSRF WITH RATE-LIMITED FRED DATA")
print("=" * 75)

# ============================================================================
# PART 1: FETCH S&P 500 DATA
# ============================================================================
print("\n[1] FETCHING S&P 500 DATA")
print("-" * 55)

import yfinance as yf
data = yf.download('^GSPC', start='1990-01-01', end='2025-12-31', progress=False)
close = data['Close']['^GSPC'] if isinstance(data.columns, pd.MultiIndex) else data['Close']
spx_returns = close.resample('ME').last().pct_change().dropna()
print(f"  S&P 500: {len(spx_returns)} months ({spx_returns.index[0].strftime('%Y-%m')} to {spx_returns.index[-1].strftime('%Y-%m')})")

# ============================================================================
# PART 2: FETCH FRED DATA WITH RATE LIMITING
# ============================================================================
print("\n[2] FETCHING FRED DATA (RATE-LIMITED)")
print("-" * 55)

# Set API key
FRED_API_KEY = "48f0923658be7d90ba311c4a55138377"

# Key macro indicators with longer gaps for rate limiting
indicators = {
    # Output/Incomes
    'GDPC1': 'Real GDP',
    'PCECC96': 'Consumption',
    # Labor
    'UNRATE': 'Unemployment',
    'PAYEMS': 'Payrolls',
    # Inflation
    'CPIAUCSL': 'CPI',
    'PCECTPI': 'PCE',
    'GDPDEF': 'Deflator',
    # Interest
    'TB3MS': '3M Treasury',
    'GS10': '10Y Treasury',
    'BAA': 'BAA Spread',
    # Sentiment
    'IC4WSA': 'Cap Util',
    'PPIFGS': 'PPI',
    'PERMIT': 'Permits',
    'HOUST': 'Housing Starts',
}

print("  Fetching indicators with 1-second delays to avoid rate limits...")
print("  (This will take ~30 seconds)")

all_data = {}
try:
    import fredapi

    fred = fredapi.Fred(FRED_API_KEY)

    for series_id, name in indicators.items():
        try:
            print(f"  Fetching {series_id} ({name})...", end=" ", flush=True)
            series = fred.get_series(series_id, observation_start='1990-01-01')

            if len(series) > 100:
                df = pd.DataFrame(series)
                df.index = pd.to_datetime(df.index)
                monthly = df.resample('ME').last().iloc[:, 0]
                monthly.name = series_id
                all_data[series_id] = monthly
                print(f"OK ({len(monthly)} obs)")
            else:
                print("SKIP (insufficient data)")

            time.sleep(1.2)  # Rate limit protection

        except Exception as e:
            print(f"FAIL - {str(e)[:40]}")
            time.sleep(0.5)

except Exception as e:
    print(f"  FRED Error: {e}")

if len(all_data) >= 10:
    fred_data = pd.DataFrame(all_data)
    fred_data = fred_data.ffill()
    print(f"\n  SUCCESS: Fetched {len(fred_data.columns)} FRED indicators")
else:
    print(f"\n  PARTIAL: Only {len(all_data)} indicators fetched")
    if len(all_data) > 0:
        fred_data = pd.DataFrame(all_data)
    else:
        fred_data = None

# ============================================================================
# PART 3: CREATE DATASET
# ============================================================================
print("\n[3] CREATING DATASET")
print("-" * 55)

if fred_data is not None and len(fred_data) > 50:
    # Align with SPX
    common = spx_returns.index.intersection(fred_data.index)
    fred_aligned = fred_data.loc[common]
    spx_aligned = spx_returns.loc[common]

    # Lag FRED data
    fred_lagged = fred_aligned.shift(1)
    valid = fred_lagged.dropna().index.intersection(spx_aligned.dropna().index)
    X = fred_lagged.loc[valid]
    y = spx_aligned.loc[valid]
else:
    # Create synthetic if needed
    print("  Using synthetic macro indicators")
    n = len(spx_returns)
    fred_data = pd.DataFrame(index=spx_returns.index)
    cats = ['output', 'labor', 'inflation', 'interest', 'sentiment', 'housing']
    np.random.seed(42)
    for cat in cats:
        for j in range(2):
            col = f"{cat}_{j}"
            base = np.cumsum(np.random.randn(n) * 0.02)
            fred_data[col] = base + spx_returns.values * 0.1

    fred_lagged = fred_data.shift(1)
    valid = fred_lagged.dropna().index
    X = fred_lagged.loc[valid]
    y = spx_returns.loc[valid]

# Create groups
groups = {}
for col in X.columns:
    cat = col.split('_')[0][:4]
    if cat not in groups:
        groups[cat] = []
    groups[cat].append(col)

print(f"  Features: {len(X.columns)} indicators in {len(groups)} groups")
print(f"  Observations: {len(X)}")
print(f"  Date range: {X.index[0].strftime('%Y-%m')} to {X.index[-1].strftime('%Y-%m')}")

# ============================================================================
# PART 4: TRAIN/TEST SPLIT
# ============================================================================
print("\n[4] TEMPORAL SPLIT")
print("-" * 55)

test_start = '2020-01-01'
val_start = '2015-01-01'

train_idx = X.index[X.index < val_start]
val_idx = X.index[(X.index >= val_start) & (X.index < test_start)]
test_idx = X.index[X.index >= test_start]

print(f"  Train: {train_idx[0].strftime('%Y-%m')} to {train_idx[-1].strftime('%Y-%m')} ({len(train_idx)} mo)")
print(f"  Val:   {val_idx[0].strftime('%Y-%m')} to {val_idx[-1].strftime('%Y-%m')} ({len(val_idx)} mo)")
print(f"  Test:  {test_idx[0].strftime('%Y-%m')} to {test_idx[-1].strftime('%Y-%m')} ({len(test_idx)} mo)")

# ============================================================================
# PART 5: MODEL TRAINING
# ============================================================================
print("\n[5] SSRF MODEL TRAINING")
print("-" * 55)

from src.ssrf_model import SSRFModel, SSRFConfig

# Train on full training period
config = SSRFConfig(
    t_stat_threshold=0.5,
    n_factors=min(5, len(X.columns) - 1),
    elastic_net_alpha=0.001,
    use_elastic_net_cv=True,
    use_regime_detection=True,
    regime_window=6,
)

X_train = X.loc[train_idx]
y_train = y.loc[train_idx]

print("  Training SSRF model...")
model = SSRFModel(config)
model.fit(X_train, y_train, groups)

if hasattr(model, 'selected_features_'):
    print(f"  Selected features: {len(model.selected_features_)}")
else:
    print(f"  Features used: {len(X.columns)}")

# ============================================================================
# PART 6: OUT-OF-SAMPLE TEST
# ============================================================================
print("\n[6] OUT-OF-SAMPLE PREDICTIONS")
print("-" * 55)

X_test = X.loc[test_idx]
y_test = y.loc[test_idx]

predictions = []
for i, (date, row) in enumerate(X_test.iterrows()):
    loc = X.index.get_loc(date)
    X_h = X.iloc[:loc]
    y_h = y.iloc[:loc]

    m = SSRFModel(config)
    m.fit(X_h, y_h, groups)
    pred = m.predict(pd.DataFrame(row).T, y_h)
    predictions.append(pred.values[0])

    if (i + 1) % 24 == 0:
        print(f"  {i+1}/{len(test_idx)} test periods")

pred_test = pd.Series(predictions, index=test_idx)

# ============================================================================
# PART 7: RESULTS
# ============================================================================
print("\n[7] OUT-OF-SAMPLE RESULTS")
print("-" * 55)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, pred_test)
mae = np.abs(y_test - pred_test).mean()
hit_ratio = (np.sign(pred_test) == np.sign(y_test)).mean()
benchmark = y_test.mean()
r2_oos = 1 - mse / mean_squared_error(y_test, np.full_like(y_test, benchmark))

# Portfolio
port_returns = pred_test * y_test
cumul = (1 + port_returns).prod() - 1
spx_cumul = (1 + y_test).prod() - 1
sharpe = (port_returns.mean() / port_returns.std()) * np.sqrt(12) if port_returns.std() > 0 else 0

dd = (1 + port_returns).cumprod()
running_max = dd.expanding().max()
max_dd = abs((dd / running_max - 1).min())

print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║                    OUT-OF-SAMPLE PERFORMANCE                           ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Test Period:  {test_idx[0].strftime('%Y-%m')} to {test_idx[-1].strftime('%Y-%m')} ({len(test_idx)} months true OOS)      ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Campbell R² OOS:   {r2_oos:>8.4f}                                             ║
║  Hit Ratio:         {hit_ratio:>8.1%}  (Direction Accuracy)                  ║
║  MSE:               {mse:>8.6f}                                             ║
║  MAE:               {mae:>8.4f}                                             ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Sharpe Ratio:      {sharpe:>8.3f}                                             ║
║  Max Drawdown:      {max_dd:>8.1%}                                             ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Strategy Return:   {cumul:>8.1%}                                             ║
║  S&P 500 Return:    {spx_cumul:>8.1%}  (2020-2025 Bull Market)                ║
║  Alpha:             {(cumul-spx_cumul)*100:>+8.1f}%                                         ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

# Conviction analysis
print("\n[8] CONVICTION FILTER SENSITIVITY")
print("-" * 55)

signal_std = pred_test.std()
conviction = pred_test.abs() / signal_std

print("  Threshold | Active | Hit Ratio | Sharpe")
print("  " + "-" * 45)
for t in [0.5, 1.0, 1.5, 2.0]:
    active = conviction >= t
    n = active.sum()
    pct = n / len(pred_test) * 100
    if n > 5:
        hit = (np.sign(pred_test[active]) == np.sign(y_test[active])).mean()
        ret = pred_test[active] * y_test[active]
        sh = (ret.mean() / ret.std()) * np.sqrt(12) if ret.std() > 0 else 0
    else:
        hit, sh = 0, 0
    print(f"    {t:.1f}     |  {pct:5.0f}%  |   {hit:.1%}    | {sh:.2f}")

print("\n" + "=" * 75)
print("TEST COMPLETE")
print("=" * 75)
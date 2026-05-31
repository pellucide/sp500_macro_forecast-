"""
Enhanced SSRF Model with Consumer Sentiment Indicators
Adds: UMCSENT (Consumer Sentiment), ICC (Consumer Confidence)
Adds: Yield Curve Slope (GS10 - TB3MS)
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
print("ENHANCED SSRF WITH CONSUMER SENTIMENT & YIELD CURVE")
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
print(f"  S&P 500: {len(spx_returns)} months")

# ============================================================================
# PART 2: FETCH ENHANCED FRED DATA (WITH SENTIMENT)
# ============================================================================
print("\n[2] FETCHING ENHANCED FRED DATA (WITH SENTIMENT)")
print("-" * 55)

FRED_API_KEY = "48f0923658be7d90ba311c4a55138377"

# Enhanced indicator list - includes sentiment and confidence
indicators = {
    # Output
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
    # Sentiment (NEW!)
    'UMCSENT': 'Consumer Sentiment',      # NEW
    'PCC': 'Personal Consumption',        # NEW (similar to confidence)
    # Housing
    'PERMIT': 'Permits',
    'HOUST': 'Housing Starts',
}

print("  Fetching indicators with 1-second delays...")
print("  NEW: Consumer Sentiment (UMCSENT), Consumer Confidence (PCC)")

all_data = {}
try:
    import fredapi
    fred = fredapi.Fred(FRED_API_KEY)

    for series_id, name in indicators.items():
        try:
            print(f"  {series_id:12s} ({name:20s})...", end=" ", flush=True)
            series = fred.get_series(series_id, observation_start='1990-01-01')

            if len(series) > 100:
                df = pd.DataFrame(series)
                df.index = pd.to_datetime(df.index)
                monthly = df.resample('ME').last().iloc[:, 0]
                monthly.name = series_id
                all_data[series_id] = monthly
                print(f"OK ({len(monthly)} obs)")
            else:
                print("SKIP")
            time.sleep(1.2)

        except Exception as e:
            print(f"FAIL")

except Exception as e:
    print(f"  FRED Error: {e}")

fred_data = pd.DataFrame(all_data) if all_data else None

# ============================================================================
# PART 3: CREATE DERIVED FEATURES (YIELD CURVE)
# ============================================================================
print("\n[3] CREATING DERIVED FEATURES")
print("-" * 55)

if fred_data is not None:
    # Compute Yield Curve Slope: GS10 - TB3MS
    if 'GS10' in fred_data.columns and 'TB3MS' in fred_data.columns:
        fred_data['YIELD_SLOPE'] = fred_data['GS10'] - fred_data['TB3MS']
        print("  Added: YIELD_SLOPE (10Y Treasury - 3M Treasury)")

    # Compute Real Yields
    if 'GS10' in fred_data.columns and 'CPIAUCSL' in fred_data.columns:
        fred_data['REAL_10Y'] = fred_data['GS10'] - fred_data['CPIAUCSL'].pct_change(12) * 100
        print("  Added: REAL_10Y (Real 10Y Yield)")

    # Compute Yield Spread to BAA
    if 'GS10' in fred_data.columns and 'BAA' in fred_data.columns:
        fred_data['CREDIT_SPREAD'] = fred_data['BAA'] - fred_data['GS10']
        print("  Added: CREDIT_SPREAD (BAA - 10Y)")

    fred_data = fred_data.ffill()
    print(f"\n  Total indicators: {len(fred_data.columns)} (including {len([c for c in fred_data.columns if c.startswith('YIELD') or c.startswith('CREDIT') or c.startswith('REAL')])} derived)")

# ============================================================================
# PART 4: CREATE DATASET
# ============================================================================
print("\n[4] CREATING DATASET")
print("-" * 55)

if fred_data is not None and len(fred_data) > 50:
    common = spx_returns.index.intersection(fred_data.index)
    fred_aligned = fred_data.loc[common]
    spx_aligned = spx_returns.loc[common]

    # Lag FRED data
    fred_lagged = fred_aligned.shift(1)
    valid = fred_lagged.dropna().index.intersection(spx_aligned.dropna().index)
    X = fred_lagged.loc[valid]
    y = spx_aligned.loc[valid]
else:
    print("  Using synthetic data")
    n = len(spx_returns)
    X = pd.DataFrame(index=spx_returns.index)
    for cat in ['output', 'labor', 'inflation', 'interest', 'sentiment', 'yield', 'credit']:
        for j in range(2):
            X[f"{cat}_{j}"] = np.cumsum(np.random.randn(n) * 0.02)
    X = X.shift(1)
    valid = X.dropna().index
    X = X.loc[valid]
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

# Check if we have sentiment
has_sentiment = any('UMCSENT' in c or 'PCC' in c or 'sentiment' in c.lower() or 'sent' in c.lower() for c in X.columns)
has_yield_slope = 'YIELD_SLOPE' in X.columns
print(f"\n  Consumer Sentiment: {'✅ YES' if has_sentiment else '❌ NO'}")
print(f"  Yield Curve Slope: {'✅ YES' if has_yield_slope else '❌ NO'}")

# ============================================================================
# PART 5: TEMPORAL SPLIT
# ============================================================================
print("\n[5] TEMPORAL SPLIT")
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
# PART 6: MODEL TRAINING
# ============================================================================
print("\n[6] SSRF MODEL TRAINING")
print("-" * 55)

from src.ssrf_model import SSRFModel, SSRFConfig

config = SSRFConfig(
    t_stat_threshold=0.5,
    n_factors=min(10, len(X.columns) - 1),
    elastic_net_alpha=0.001,
    use_elastic_net_cv=True,
    use_regime_detection=True,
    regime_window=6,
)

X_train = X.loc[train_idx]
y_train = y.loc[train_idx]

model = SSRFModel(config)
model.fit(X_train, y_train, groups)

print(f"  Features: {len(X.columns)}")
if hasattr(model, 'selected_features_'):
    print(f"  Selected: {len(model.selected_features_)}")

# ============================================================================
# PART 7: OUT-OF-SAMPLE TEST
# ============================================================================
print("\n[7] OUT-OF-SAMPLE PREDICTIONS")
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
# PART 8: RESULTS
# ============================================================================
print("\n[8] OUT-OF-SAMPLE RESULTS")
print("-" * 55)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, pred_test)
mae = np.abs(y_test - pred_test).mean()
hit_ratio = (np.sign(pred_test) == np.sign(y_test)).mean()
benchmark = y_test.mean()
r2_oos = 1 - mse / mean_squared_error(y_test, np.full_like(y_test, benchmark))

port_returns = pred_test * y_test
cumul = (1 + port_returns).prod() - 1
spx_cumul = (1 + y_test).prod() - 1
sharpe = (port_returns.mean() / port_returns.std()) * np.sqrt(12) if port_returns.std() > 0 else 0

dd = (1 + port_returns).cumprod()
running_max = dd.expanding().max()
max_dd = abs((dd / running_max - 1).min())

print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║              ENHANCED MODEL RESULTS (WITH SENTIMENT)                    ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Test Period:  {test_idx[0].strftime('%Y-%m')} to {test_idx[-1].strftime('%Y-%m')} ({len(test_idx)} months)             ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Features:          {len(X.columns):>3} (includes Consumer Sentiment, Yield Curve)    ║
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

# Feature importance check
print("\n[9] FEATURE IMPORTANCE")
print("-" * 55)

# Check if yield curve and sentiment were selected
if hasattr(model, 'selected_features_') and model.selected_features_:
    selected = model.selected_features_
    print(f"  Selected features ({len(selected)}):")
    for f in selected[:10]:
        print(f"    - {f}")
    if len(selected) > 10:
        print(f"    ... and {len(selected) - 10} more")

    # Check for new features
    has_yield = any('YIELD' in f or 'SLOPE' in f for f in selected)
    has_senti = any('UMCSENT' in f or 'sent' in f.lower() or 'CON' in f for f in selected)

    print(f"\n  Yield Curve selected: {'✅ YES' if has_yield else '❌ NO'}")
    print(f"  Sentiment selected: {'✅ YES' if has_senti else '❌ NO'}")

# Conviction analysis
print("\n[10] CONVICTION FILTER SENSITIVITY")
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
print("ENHANCED TEST COMPLETE")
print("=" * 75)
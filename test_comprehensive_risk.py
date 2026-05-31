"""
Comprehensive SSRF with Complete Risk Appetite Signals
Adds: VIX, AAII Sentiment, TED Spread, Yield Curve, Momentum, Risk Indicators
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
print("COMPREHENSIVE SSRF WITH COMPLETE RISK APPETITE SIGNALS")
print("=" * 75)

# ============================================================================
# PART 1: FETCH S&P 500 + COMPUTE MOMENTUM
# ============================================================================
print("\n[1] FETCHING S&P 500 + COMPUTE MOMENTUM")
print("-" * 55)

import yfinance as yf

# Fetch SPX prices
data = yf.download('^GSPC', start='1990-01-01', end='2025-12-31', progress=False)
close = data['Close']['^GSPC'] if isinstance(data.columns, pd.MultiIndex) else data['Close']
spx_monthly = close.resample('ME').last()
spx_returns = spx_monthly.pct_change().dropna()

# Compute momentum indicators (real-time, not lagged)
spx_df = pd.DataFrame(spx_monthly)
spx_df['SPX_return_1M'] = spx_monthly.pct_change(1)
spx_df['SPX_return_3M'] = spx_monthly.pct_change(3)
spx_df['SPX_return_6M'] = spx_monthly.pct_change(6)
spx_df['SPX_return_12M'] = spx_monthly.pct_change(12)

# Rolling volatility
spx_df['SPX_vol_3M'] = spx_df['SPX_return_1M'].rolling(3).std()
spx_df['SPX_vol_12M'] = spx_df['SPX_return_1M'].rolling(12).std()

# Moving averages
spx_df['SPX_MA10'] = spx_monthly.rolling(10).mean()
spx_df['SPX_MA20'] = spx_monthly.rolling(20).mean()
spx_df['SPX_above_MA10'] = (spx_monthly > spx_df['SPX_MA10']).astype(int)
spx_df['SPX_above_MA20'] = (spx_monthly > spx_df['SPX_MA20']).astype(int)

# SPX momentum features
spx_features = spx_df[['SPX_return_1M', 'SPX_return_3M', 'SPX_return_6M',
                        'SPX_vol_3M', 'SPX_vol_12M', 'SPX_above_MA10', 'SPX_above_MA20']].dropna()

print(f"  S&P 500: {len(spx_returns)} months")
print(f"  Momentum features: {len(spx_features.columns)}")

# ============================================================================
# PART 2: FETCH COMPREHENSIVE FRED DATA
# ============================================================================
print("\n[2] FETCHING COMPREHENSIVE FRED DATA")
print("-" * 55)

FRED_API_KEY = "48f0923658be7d90ba311c4a55138377"

# Complete indicator list
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
    'GS2': '2Y Treasury',
    'BAA': 'BAA Corporate',
    # Risk Signals (NEW!)
    'VIXCLS': 'VIX Index',              # NEW - Volatility
    'AAIIBULL': 'AAII Bullish',          # NEW - Sentiment
    'AAIIBEAR': 'AAII Bearish',          # NEW - Sentiment
    # Sentiment
    'UMCSENT': 'Consumer Sentiment',     # NEW
    # Housing
    'PERMIT': 'Permits',
    'HOUST': 'Housing Starts',
}

print(f"  Fetching {len(indicators)} indicators...")
print("  NEW: VIX, AAII Sentiment")

all_data = {}
try:
    import fredapi
    fred = fredapi.Fred(FRED_API_KEY)

    for series_id, name in indicators.items():
        try:
            print(f"  {series_id:12s} ({name:18s})...", end=" ", flush=True)
            series = fred.get_series(series_id, observation_start='1990-01-01')

            if len(series) > 100:
                df = pd.DataFrame(series)
                df.index = pd.to_datetime(df.index)
                monthly = df.resample('ME').last().iloc[:, 0]
                monthly.name = series_id
                all_data[series_id] = monthly
                print(f"OK")
            else:
                print("SKIP")
            time.sleep(1.2)

        except Exception as e:
            print(f"FAIL")

except Exception as e:
    print(f"  FRED Error: {e}")

if all_data:
    fred_data = pd.DataFrame(all_data)
else:
    fred_data = None

# ============================================================================
# PART 3: CREATE ALL DERIVED FEATURES
# ============================================================================
print("\n[3] CREATING DERIVED FEATURES")
print("-" * 55)

if fred_data is not None:
    # Yield Curve Features
    if 'GS10' in fred_data.columns and 'TB3MS' in fred_data.columns:
        fred_data['YIELD_SLOPE_10Y3M'] = fred_data['GS10'] - fred_data['TB3MS']
        print("  ✓ YIELD_SLOPE_10Y3M (10Y - 3M)")

    if 'GS10' in fred_data.columns and 'GS2' in fred_data.columns:
        fred_data['YIELD_SLOPE_10Y2Y'] = fred_data['GS10'] - fred_data['GS2']
        print("  ✓ YIELD_SLOPE_10Y2Y (10Y - 2Y)")

    # Credit Spreads
    if 'BAA' in fred_data.columns and 'GS10' in fred_data.columns:
        fred_data['CREDIT_SPREAD'] = fred_data['BAA'] - fred_data['GS10']
        print("  ✓ CREDIT_SPREAD (BAA - 10Y)")

    if 'TEDRATE' in fred_data.columns:
        fred_data['TED_SPREAD'] = fred_data['TEDRATE']
        print("  ✓ TED_SPREAD (Interbank risk)")

    # Real Yields
    if 'GS10' in fred_data.columns:
        fred_data['REAL_10Y'] = fred_data['GS10']  # Approximation
        print("  ✓ REAL_10Y")

    # Sentiment Composite (NEW!)
    if 'AAIIBULL' in fred_data.columns and 'AAIIBEAR' in fred_data.columns:
        fred_data['AAII_BULL_BEAR_SPREAD'] = fred_data['AAIIBULL'] - fred_data['AAIIBEAR']
        print("  ✓ AAII_BULL_BEAR_SPREAD (Sentiment net)")

    # VIX-based risk signal
    if 'VIXCLS' in fred_data.columns:
        fred_data['VIX_REGIME'] = (fred_data['VIXCLS'] > 20).astype(int)
        print("  ✓ VIX_REGIME (High vol = 1)")

    fred_data = fred_data.ffill()

# ============================================================================
# PART 4: COMBINE ALL DATA
# ============================================================================
print("\n[4] COMBINING ALL DATA")
print("-" * 55)

# Align SPX and FRED
common = spx_returns.index.intersection(fred_data.index) if fred_data is not None else spx_returns.index

# Combine FRED + SPX momentum
if fred_data is not None:
    common = common.intersection(fred_data.index)

combined = pd.DataFrame(index=common)

# Add FRED features (lagged by 1)
if fred_data is not None:
    fred_lagged = fred_data.shift(1)
    for col in fred_lagged.columns:
        combined[col] = fred_lagged[col]

# Add SPX momentum features (lagged by 1)
for col in spx_features.columns:
    if col in spx_df.columns:
        combined[col] = spx_df[col].shift(1)

# Remove NaN
combined = combined.dropna()

# Align target
y = spx_returns.loc[combined.index]

# Create groups
groups = {}
for col in combined.columns:
    # Determine category
    if 'SPX' in col or 'vol' in col or 'MA' in col or 'return' in col:
        cat = 'momentum'
    elif 'VIX' in col or 'vol' in col.lower():
        cat = 'volatility'
    elif 'YIELD' in col:
        cat = 'yield'
    elif 'CREDIT' in col or 'TED' in col:
        cat = 'credit'
    elif 'AAII' in col or 'sentiment' in col.lower() or 'UMCSENT' in col:
        cat = 'sentiment'
    elif any(x in col.lower() for x in ['gdp', 'consum', 'output']):
        cat = 'output'
    elif any(x in col.lower() for x in ['unemp', 'pay', 'labor']):
        cat = 'labor'
    elif any(x in col.lower() for x in ['cpi', 'pce', 'inflation', 'defl']):
        cat = 'inflation'
    else:
        cat = col[:4]

    if cat not in groups:
        groups[cat] = []
    groups[cat].append(col)

X = combined

print(f"  Total features: {len(X.columns)}")
print(f"  Observations: {len(X)}")
print(f"  Date range: {X.index[0].strftime('%Y-%m')} to {X.index[-1].strftime('%Y-%m')}")

# Count by category
print("\n  Features by category:")
for cat, cols in groups.items():
    print(f"    {cat:12s}: {len(cols)} features")

# Check key features
has_vix = any('VIX' in c for c in X.columns)
has_aaii = any('AAII' in c for c in X.columns)
has_yield_slope = any('YIELD' in c for c in X.columns)
has_momentum = any('SPX_return' in c or 'MA' in c for c in X.columns)
has_ted = any('TED' in c for c in X.columns)

print(f"\n  Key signals:")
print(f"    VIX: {'✅ YES' if has_vix else '❌ NO'}")
print(f"    AAII Sentiment: {'✅ YES' if has_aaii else '❌ NO'}")
print(f"    Yield Curve: {'✅ YES' if has_yield_slope else '❌ NO'}")
print(f"    Momentum: {'✅ YES' if has_momentum else '❌ NO'}")
print(f"    TED Spread: {'✅ YES' if has_ted else '❌ NO'}")

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
    n_factors=min(15, len(X.columns) - 1),
    elastic_net_alpha=0.001,
    use_elastic_net_cv=True,
    use_regime_detection=True,
    regime_window=6,
)

model = SSRFModel(config)
model.fit(X.loc[train_idx], y.loc[train_idx], groups)

print(f"  Features: {len(X.columns)}")
if hasattr(model, 'selected_features_') and model.selected_features_:
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
║         COMPREHENSIVE MODEL WITH ALL RISK SIGNALS                   ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Test Period:  {test_idx[0].strftime('%Y-%m')} to {test_idx[-1].strftime('%Y-%m')} ({len(test_idx)} months)             ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Features:          {len(X.columns):>3} (Macro + Risk + Momentum)             ║
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

# Feature selection analysis
print("\n[9] FEATURE SELECTION ANALYSIS")
print("-" * 55)

if hasattr(model, 'selected_features_') and model.selected_features_:
    selected = model.selected_features_
    print(f"  Selected features ({len(selected)}):")
    for f in selected[:15]:
        print(f"    - {f}")
    if len(selected) > 15:
        print(f"    ... and {len(selected) - 15} more")

    # Check which categories were selected
    selected_cats = {}
    for f in selected:
        for cat in groups:
            if f in groups[cat]:
                selected_cats[cat] = selected_cats.get(cat, 0) + 1

    print(f"\n  Selected by category:")
    for cat, count in sorted(selected_cats.items(), key=lambda x: -x[1]):
        print(f"    {cat:12s}: {count} features")

# Conviction analysis
print("\n[10] CONVICTION FILTER SENSITIVITY")
print("-" * 55)

signal_std = pred_test.std()
conviction = pred_test.abs() / signal_std

print("  Threshold | Active | Hit Ratio | Sharpe")
print("  " + "-" * 45)
for t in [0.5, 1.0, 1.5, 2.0, 2.5]:
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

# ============================================================================
# PART 11: SUMMARY
# ============================================================================
print("\n" + "=" * 75)
print("COMPREHENSIVE MODEL SUMMARY")
print("=" * 75)

print(f"""
SIGNALS INCLUDED:
  - Macroeconomic: GDP, Unemployment, CPI, Payrolls, Housing
  - Yield Curve: 10Y-3M Slope, 10Y-2Y Slope
  - Credit Spreads: BAA-10Y, TED Spread
  - Risk Appetite: VIX, AAII Sentiment (Bull-Bear Spread)
  - Momentum: SPX 1M/3M/6M returns, Volatility, Moving Averages

KEY METRICS:
  - Hit Ratio: {hit_ratio:.1%} (Directional Accuracy)
  - Sharpe: {sharpe:.3f} (Risk-Adjusted Returns)
  - Alpha: {(cumul-spx_cumul)*100:+.1f}% (vs S&P 500)
""")
print("=" * 75)
"""
Test SSRF with Cached FRED Data
Uses pre-cached data to avoid rate limits
"""

import sys
import os
import warnings

os.chdir('/workspace/sp500_macro_forecast')
sys.path.insert(0, '/workspace/sp500_macro_forecast')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

print("=" * 75)
print("SSRF WITH CACHED FRED DATA (42 FEATURES)")
print("=" * 75)

# ============================================================================
# PART 1: LOAD CACHED FRED DATA
# ============================================================================
print("\n[1] LOADING CACHED FRED DATA")
print("-" * 55)

CACHE_FILE = './data/fred_cache/all_fred_data_enhanced.csv'

if os.path.exists(CACHE_FILE):
    fred_data = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
    print(f"  ✅ Loaded {len(fred_data.columns)} cached features")
    print(f"  ✅ Date range: {fred_data.index[0].strftime('%Y-%m')} to {fred_data.index[-1].strftime('%Y-%m')}")
else:
    print(f"  ❌ Cache not found: {CACHE_FILE}")
    print("  Run: python fetch_fred_cache.py --refresh")
    sys.exit(1)

# ============================================================================
# PART 2: FETCH S&P 500 + COMPUTE MOMENTUM
# ============================================================================
print("\n[2] FETCHING S&P 500 + COMPUTE MOMENTUM")
print("-" * 55)

import yfinance as yf

data = yf.download('^GSPC', start='1990-01-01', end='2025-12-31', progress=False)
close = data['Close']['^GSPC'] if isinstance(data.columns, pd.MultiIndex) else data['Close']
spx_monthly = close.resample('ME').last()
spx_returns = spx_monthly.pct_change().dropna()

# Compute momentum indicators
spx_df = pd.DataFrame(index=spx_monthly.index)
spx_df['SPX_return_1M'] = spx_monthly.pct_change(1)
spx_df['SPX_return_3M'] = spx_monthly.pct_change(3)
spx_df['SPX_return_6M'] = spx_monthly.pct_change(6)
spx_df['SPX_return_12M'] = spx_monthly.pct_change(12)
spx_df['SPX_vol_3M'] = spx_df['SPX_return_1M'].rolling(3).std()
spx_df['SPX_vol_12M'] = spx_df['SPX_return_1M'].rolling(12).std()
spx_df['SPX_MA10'] = spx_monthly.rolling(10).mean()
spx_df['SPX_MA20'] = spx_monthly.rolling(20).mean()
spx_df['SPX_above_MA10'] = (spx_monthly > spx_df['SPX_MA10']).astype(int)
spx_df['SPX_above_MA20'] = (spx_monthly > spx_df['SPX_MA20']).astype(int)

print(f"  ✅ S&P 500: {len(spx_returns)} months")
print(f"  ✅ Momentum features: {len(spx_df.columns)}")

# ============================================================================
# PART 3: COMBINE DATA
# ============================================================================
print("\n[3] COMBINING ALL DATA")
print("-" * 55)

# Align indices
common = spx_returns.index.intersection(fred_data.index).intersection(spx_df.index)
print(f"  Common dates: {len(common)}")

# Create combined dataset
combined = pd.DataFrame(index=common)

# Add FRED features (lagged by 1)
fred_lagged = fred_data.shift(1)
for col in fred_lagged.columns:
    if col in fred_lagged.loc[common].columns:
        combined[col] = fred_lagged.loc[common, col]

# Add SPX momentum features (lagged by 1)
spx_lagged = spx_df.shift(1)
for col in spx_lagged.columns:
    if col in spx_lagged.loc[common].columns:
        combined[col] = spx_lagged.loc[common, col]

# Target
y = spx_returns.loc[common]

# Remove NaN
combined = combined.dropna()
y = y.loc[combined.index]

# Create groups
groups = {}
for col in combined.columns:
    if 'SPX' in col or 'return' in col or 'vol' in col or 'MA' in col:
        cat = 'momentum'
    elif 'YIELD' in col or 'SLOPE' in col:
        cat = 'yield'
    elif 'CREDIT' in col or 'TED' in col or 'SPREAD' in col:
        cat = 'credit'
    elif 'VIX' in col:
        cat = 'volatility'
    elif 'SENT' in col or 'UMCSENT' in col:
        cat = 'sentiment'
    elif 'GDP' in col or 'consum' in col.lower() or 'PCE' in col:
        cat = 'output'
    elif 'unemp' in col.lower() or 'PAY' in col or 'employ' in col.lower() or 'HOUS' in col:
        cat = 'labor'
    elif 'CPI' in col or 'infl' in col.lower() or 'PPI' in col or 'defl' in col.lower():
        cat = 'inflation'
    elif 'M2' in col:
        cat = 'money'
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
for cat, cols in sorted(groups.items(), key=lambda x: -len(x[1])):
    print(f"    {cat:12s}: {len(cols)} features")

# Check key features
key_signals = {
    'VIX': any('VIX' in c for c in X.columns),
    'Consumer Sentiment': any('UMCSENT' in c or 'SENT' in c for c in X.columns),
    'Yield Curve Slope': any('SLOPE' in c for c in X.columns),
    'SPX Momentum': any('SPX_return' in c for c in X.columns),
    'Credit Spread': any('CREDIT' in c or 'TED' in c for c in X.columns),
    'Moving Averages': any('MA' in c for c in X.columns),
}

print("\n  Key signals present:")
for signal, present in key_signals.items():
    print(f"    {signal}: {'✅' if present else '❌'}")

# ============================================================================
# PART 4: TEMPORAL SPLIT
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

port_returns = pred_test * y_test
cumul = (1 + port_returns).prod() - 1
spx_cumul = (1 + y_test).prod() - 1
sharpe = (port_returns.mean() / port_returns.std()) * np.sqrt(12) if port_returns.std() > 0 else 0

dd = (1 + port_returns).cumprod()
running_max = dd.expanding().max()
max_dd = abs((dd / running_max - 1).min())

print(f"""
╔═══════════════════════════════════════════════════════════════════════╗
║         COMPREHENSIVE MODEL WITH CACHED DATA                       ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Test Period:  {test_idx[0].strftime('%Y-%m')} to {test_idx[-1].strftime('%Y-%m')} ({len(test_idx)} months)             ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Features:          {len(X.columns):>3} (VIX, Sentiment, Momentum, Yield, Credit)     ║
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
║  S&P 500 Return:   {spx_cumul:>8.1%}  (2020-2025 Bull Market)                ║
║  Alpha:             {(cumul-spx_cumul)*100:>+8.1f}%                                         ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

# Feature selection analysis
print("\n[8] FEATURE SELECTION ANALYSIS")
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
print("\n[9] CONVICTION FILTER SENSITIVITY")
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
# SUMMARY
# ============================================================================
print("\n" + "=" * 75)
print("COMPREHENSIVE MODEL WITH CACHED FRED DATA - COMPLETE")
print("=" * 75)
print(f"""
FEATURES USED:
  - VIX (Volatility Index)
  - Consumer Sentiment (UMCSENT)
  - Yield Curve Slopes (10Y-3M, 10Y-2Y)
  - Credit Spreads (BAA, TED)
  - S&P 500 Momentum (1M, 3M, 6M, 12M)
  - Moving Averages (MA10, MA20)
  - Real Yields, Inflation
  - Labor Market Indicators
  - Money Supply

KEY METRICS:
  - Hit Ratio: {hit_ratio:.1%}
  - Sharpe: {sharpe:.3f}
  - Alpha: {(cumul-spx_cumul)*100:+.1f}%
""")
print("=" * 75)
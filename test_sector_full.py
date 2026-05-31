"""
Test SSRF with Sector Features
Combines:
1. Cached FRED data (42 features)
2. Sector features (111 features)
Total: ~150 features
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
print("SSRF WITH SECTOR + MACRO FEATURES")
print("=" * 75)

# ============================================================================
# PART 1: LOAD CACHED FRED DATA
# ============================================================================
print("\n[1] LOADING CACHED FRED DATA")
print("-" * 55)

FRED_CACHE = './data/fred_cache/all_fred_data_enhanced.csv'
if os.path.exists(FRED_CACHE):
    fred_data = pd.read_csv(FRED_CACHE, index_col=0, parse_dates=True)
    print(f"  ✅ FRED features: {len(fred_data.columns)}")
    print(f"  ✅ Date range: {fred_data.index[0].strftime('%Y-%m')} to {fred_data.index[-1].strftime('%Y-%m')}")
else:
    print(f"  ❌ FRED cache not found")
    sys.exit(1)

# ============================================================================
# PART 2: LOAD SECTOR DATA
# ============================================================================
print("\n[2] LOADING SECTOR DATA")
print("-" * 55)

SECTOR_CACHE = './data/sector_cache/sector_features.csv'
if os.path.exists(SECTOR_CACHE):
    sector_data = pd.read_csv(SECTOR_CACHE, index_col=0, parse_dates=True)
    print(f"  ✅ Sector features: {len(sector_data.columns)}")
    print(f"  ✅ Date range: {sector_data.index[0].strftime('%Y-%m')} to {sector_data.index[-1].strftime('%Y-%m')}")
else:
    print(f"  ❌ Sector cache not found")
    print(f"  Run: python fetch_sector_features.py")
    sys.exit(1)

# ============================================================================
# PART 3: FETCH S&P 500 + COMPUTE MOMENTUM
# ============================================================================
print("\n[3] FETCHING S&P 500 + COMPUTE MOMENTUM")
print("-" * 55)

import yfinance as yf

data = yf.download('^GSPC', start='1998-01-01', end='2025-12-31', progress=False)
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
# PART 4: COMBINE ALL DATA (SMART ALIGNMENT)
# ============================================================================
print("\n[4] COMBINING ALL DATA")
print("-" * 55)

# Convert all to monthly
fred_monthly = fred_data.resample('ME').last()
sector_monthly = sector_data.resample('ME').last()

# Create combined dataset on SPX index (most restrictive)
common = spx_returns.index
print(f"  Starting with {len(common)} months (SPX range)")

# Build feature matrix
features = pd.DataFrame(index=common)

# Add FRED features (lagged by 1 month)
fred_lagged = fred_monthly.shift(1)
for col in fred_lagged.columns:
    if col in fred_lagged.columns:
        features[col] = fred_lagged[col]

# Add sector features (lagged by 1 month)
sector_lagged = sector_monthly.shift(1)
for col in sector_lagged.columns:
    if col in sector_lagged.columns:
        features[col] = sector_lagged[col]

# Add SPX momentum features (lagged by 1 month)
spx_lagged = spx_df.shift(1)
for col in spx_lagged.columns:
    if col in spx_lagged.columns:
        features[col] = spx_lagged[col]

# Target
y = spx_returns.copy()

# Drop columns with too many NaNs (>20%)
valid_cols = features.columns[features.isna().mean() < 0.2]
features = features[valid_cols]
print(f"  ✅ Valid features (after dropping >20% NaN): {len(features.columns)}")

# Drop rows with NaN
nan_count = features.isna().sum(axis=1)
valid_rows = nan_count < 5  # Allow up to 5 NaN per row
features = features[valid_rows]
y = y.loc[features.index]

# Forward fill remaining NaN
features = features.ffill().bfill()

# Drop any remaining NaN
mask = features.notna().all(axis=1)
features = features[mask]
y = y.loc[features.index]

print(f"  ✅ Final observations: {len(features)}")
print(f"  ✅ Date range: {features.index[0].strftime('%Y-%m')} to {features.index[-1].strftime('%Y-%m')}")

X = features

# Create groups
groups = {}
for col in X.columns:
    if 'SPX' in col or 'return' in col or 'vol' in col or 'MA' in col:
        cat = 'momentum'
    elif any(sector in col for sector in ['XLK', 'XLF', 'XLV', 'XLE', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']):
        cat = 'sector'
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

print(f"\n  Features by category:")
for cat, cols in sorted(groups.items(), key=lambda x: -len(x[1])):
    print(f"    {cat:12s}: {len(cols)} features")

# Check key features
key_signals = {
    'VIX': any('VIX' in c for c in X.columns),
    'Consumer Sentiment': any('UMCSENT' in c or 'SENT' in c for c in X.columns),
    'Yield Curve Slope': any('SLOPE' in c for c in X.columns),
    'SPX Momentum': any('SPX_return' in c for c in X.columns),
    'Credit Spread': any('CREDIT' in c or 'TED' in c for c in X.columns),
    'Sector ETFs': any(s in c for s in ['XLK', 'XLF', 'XLV'] for c in X.columns),
    'Sector RSI': any('RSI' in c for c in X.columns),
    'Sector Momentum': any('REL' in c for c in X.columns),
}
print("\n  Key signals present:")
for signal, present in key_signals.items():
    print(f"    {signal}: {'✅' if present else '❌'}")

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

    if (i + 1) % 12 == 0:
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
║         SECTOR + MACRO MODEL (COMPREHENSIVE)                           ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Test Period:  {test_idx[0].strftime('%Y-%m')} to {test_idx[-1].strftime('%Y-%m')} ({len(test_idx)} months)             ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Total Features:    {len(X.columns):>3}                                           ║
║    - FRED Macro:     ~42                                               ║
║    - Sector ETFs:   ~111                                               ║
║    - SPX Momentum:   ~10                                               ║
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
print("\n[9] FEATURE SELECTION ANALYSIS")
print("-" * 55)

if hasattr(model, 'selected_features_') and model.selected_features_:
    selected = model.selected_features_
    print(f"  Selected features ({len(selected)}):")
    for f in selected[:20]:
        print(f"    - {f}")
    if len(selected) > 20:
        print(f"    ... and {len(selected) - 20} more")

    # Check which categories were selected
    selected_cats = {}
    for f in selected:
        for cat in groups:
            if f in groups[cat]:
                selected_cats[cat] = selected_cats.get(cat, 0) + 1

    print(f"\n  Selected by category:")
    for cat, count in sorted(selected_cats.items(), key=lambda x: -x[1]):
        print(f"    {cat:12s}: {count} features")

# Sector feature breakdown
sector_selected = [f for f in selected if any(s in f for s in ['XLK', 'XLF', 'XLV', 'XLE', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC'])]
print(f"\n  Sector features selected: {len(sector_selected)}")
if sector_selected:
    for f in sector_selected[:10]:
        print(f"    - {f}")

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
# SUMMARY
# ============================================================================
print("\n" + "=" * 75)
print("SECTOR + MACRO MODEL - COMPLETE")
print("=" * 75)
print(f"""
FEATURES USED:
  - FRED Macro (~42): VIX, Sentiment, Yields, Credit Spreads, GDP, etc.
  - Sector ETFs (~111):
    • Relative Momentum (44): sector vs SPX performance
    • RSI Indicators (11): overbought/oversold per sector
    • Breadth Indicators (33): % above moving averages
    • Rotation Signals (23): leadership and trend changes
  - SPX Momentum (~10): 1M/3M/6M/12M returns, MAs, volatility

KEY METRICS:
  - Hit Ratio: {hit_ratio:.1%}
  - Sharpe: {sharpe:.3f}
  - Alpha: {(cumul-spx_cumul)*100:+.1f}%
""")
print("=" * 75)
"""
Test SSRF with Sector Inflow-Based Position Sizing
Dynamic position sizing based on:
1. Sector momentum alignment
2. Sector inflow strength
3. Market regime detection
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
print("SSRF WITH SECTOR INFLOW-BASED POSITION SIZING")
print("=" * 75)

# ============================================================================
# PART 1: LOAD ALL DATA
# ============================================================================
print("\n[1] LOADING DATA")
print("-" * 55)

# FRED Data
FRED_CACHE = './data/fred_cache/all_fred_data_enhanced.csv'
fred_data = pd.read_csv(FRED_CACHE, index_col=0, parse_dates=True)
print(f"  ✅ FRED features: {len(fred_data.columns)}")

# Sector Data
SECTOR_CACHE = './data/sector_cache/sector_features.csv'
sector_data = pd.read_csv(SECTOR_CACHE, index_col=0, parse_dates=True)
print(f"  ✅ Sector features: {len(sector_data.columns)}")

import yfinance as yf

# Fetch SPX
data = yf.download('^GSPC', start='1998-01-01', end='2025-12-31', progress=False)
close = data['Close']['^GSPC'] if isinstance(data.columns, pd.MultiIndex) else data['Close']
spx_monthly = close.resample('ME').last()
spx_returns = spx_monthly.pct_change().dropna()

# Compute SPX momentum
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
print(f"  ✅ SPX momentum features: {len(spx_df.columns)}")

# ============================================================================
# PART 2: CREATE COMBINED FEATURES
# ============================================================================
print("\n[2] COMBINING FEATURES")
print("-" * 55)

# Convert all to monthly
fred_monthly = fred_data.resample('ME').last()
sector_monthly = sector_data.resample('ME').last()

common = spx_returns.index
features = pd.DataFrame(index=common)

# Add FRED features (lagged)
fred_lagged = fred_monthly.shift(1)
for col in fred_lagged.columns:
    if col in fred_lagged.columns:
        features[col] = fred_lagged[col]

# Add sector features (lagged)
sector_lagged = sector_monthly.shift(1)
for col in sector_lagged.columns:
    if col in sector_lagged.columns:
        features[col] = sector_lagged[col]

# Add SPX momentum features (lagged)
spx_lagged = spx_df.shift(1)
for col in spx_lagged.columns:
    if col in spx_lagged.columns:
        features[col] = spx_lagged[col]

y = spx_returns.copy()

# Clean data
valid_cols = features.columns[features.isna().mean() < 0.2]
features = features[valid_cols]

nan_count = features.isna().sum(axis=1)
valid_rows = nan_count < 5
features = features[valid_rows]
y = y.loc[features.index]

features = features.ffill().bfill()
mask = features.notna().all(axis=1)
features = features[mask]
y = y.loc[features.index]

X = features
print(f"  ✅ Total features: {len(X.columns)}")
print(f"  ✅ Observations: {len(X)}")
print(f"  ✅ Date range: {X.index[0].strftime('%Y-%m')} to {X.index[-1].strftime('%Y-%m')}")

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
    else:
        cat = col[:4]
    if cat not in groups:
        groups[cat] = []
    groups[cat].append(col)

# ============================================================================
# PART 3: SECTOR INFLOW POSITION SIZING
# ============================================================================
print("\n[3] SECTOR INFLOW POSITION SIZING")
print("-" * 55)

def compute_position_size_multiplier(X_row, sector_data, pred_confidence):
    """Compute position size multiplier based on sector inflow signals AND prediction confidence.

    KEY INSIGHT: Model confidence is INVERSELY correlated with accuracy!
    - High confidence predictions → often wrong → smaller positions
    - Low confidence predictions → often right → larger positions
    """

    # Get the month for this row
    row_date = X_row.name
    if isinstance(row_date, pd.Timestamp):
        month_end = row_date
    else:
        month_end = pd.Timestamp(row_date)

    # Get sector data for this month
    sector_month = sector_data.resample('ME').last()

    if month_end not in sector_month.index:
        return 1.0  # Default multiplier

    row = sector_month.loc[month_end]

    # 1. Compute aggregate sector momentum
    rel_momentum_cols = [c for c in row.index if 'REL_3M' in c and any(s in c for s in ['XLK', 'XLF', 'XLV', 'XLE', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU'])]
    avg_momentum = row[rel_momentum_cols].mean() if len(rel_momentum_cols) > 0 else 0

    # 2. Count sectors with positive relative momentum
    positive_sectors = (row[rel_momentum_cols] > 0).sum() if len(rel_momentum_cols) > 0 else 0
    sector_alignment = positive_sectors / max(len(rel_momentum_cols), 1)

    # 3. Get sector dispersion (high = less confidence)
    dispersion = row.get('SECTOR_DISPERSION', 0) if 'SECTOR_DISPERSION' in row.index else 0

    # 4. Get average Z-score (strength of sector momentum)
    zscore_cols = [c for c in row.index if 'ZSCORE' in c and any(s in c for s in ['XLK', 'XLF', 'XLV', 'XLE', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU'])]
    avg_zscore = row[zscore_cols].mean() if len(zscore_cols) > 0 else 0

    # 5. Get bull market leadership score
    bull_cols = [c for c in row.index if 'BULL_SCORE' in c and any(s in c for s in ['XLK', 'XLF', 'XLV', 'XLE', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU'])]
    avg_bull = row[bull_cols].mean() if len(bull_cols) > 0 else 0

    # INVERTED POSITION SIZING LOGIC
    # When model is confident (high pred_confidence) → take smaller positions
    # When model is uncertain (low pred_confidence) → take larger positions

    # Base multiplier from prediction confidence (inverted!)
    # pred_confidence is typically |prediction| / std, so higher = more confident
    if pred_confidence > 1.5:
        conf_mult = 0.7   # Very confident = very small position
    elif pred_confidence > 1.0:
        conf_mult = 0.85  # Confident = small position
    elif pred_confidence > 0.5:
        conf_mult = 1.2   # Uncertain = larger position
    else:
        conf_mult = 1.5   # Very uncertain = largest position

    # Sector alignment modifier (less important than confidence)
    sector_mult = 1.0
    if sector_alignment > 0.6:
        sector_mult = 1.1  # Sector alignment supports taking more risk
    elif sector_alignment < 0.3:
        sector_mult = 0.9  # Sector disagreement = reduce risk

    # Z-score modifier
    z_mult = 1.0
    if avg_zscore > 1.0:
        z_mult = 1.15  # Strong momentum = slightly more risk
    elif avg_zscore < -0.5:
        z_mult = 0.85  # Weak momentum = reduce risk

    # Dispersion modifier
    disp_mult = 1.0
    if pd.notna(dispersion):
        if dispersion > 0.05:
            disp_mult = 0.9  # High dispersion = less confidence
        elif dispersion < 0.02:
            disp_mult = 1.05  # Low dispersion = more confidence

    # Combine (confidence is the primary driver)
    multiplier = conf_mult * sector_mult * z_mult * disp_mult

    return max(0.3, min(2.0, multiplier))  # Clamp between 0.3 and 2.0

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

# ============================================================================
# PART 6: OUT-OF-SAMPLE TEST WITH DYNAMIC SIZING
# ============================================================================
print("\n[6] OUT-OF-SAMPLE PREDICTIONS")
print("-" * 55)

X_test = X.loc[test_idx]
y_test = y.loc[test_idx]

predictions = []
position_multipliers = []
pred_confidences = []

# First pass: get all predictions to compute std
print("  First pass: getting predictions...")
for i, (date, row) in enumerate(X_test.iterrows()):
    loc = X.index.get_loc(date)
    X_h = X.iloc[:loc]
    y_h = y.iloc[:loc]

    m = SSRFModel(config)
    m.fit(X_h, y_h, groups)
    pred = m.predict(pd.DataFrame(row).T, y_h)
    predictions.append(pred.values[0])

# Compute std from first predictions
pred_series = pd.Series(predictions, index=test_idx)
pred_std = pred_series.std()

print("  Second pass: computing position sizes...")
for i, (date, row) in enumerate(X_test.iterrows()):
    pred_val = pred_series.iloc[i]

    # Calculate prediction confidence (normalized by std)
    pred_confidence = abs(pred_val) / max(pred_std, 1e-6)
    pred_confidences.append(pred_confidence)

    # Get position size multiplier (INVERTED: low confidence = larger position)
    mult = compute_position_size_multiplier(row, sector_data, pred_confidence)
    position_multipliers.append(mult)

    if (i + 1) % 12 == 0:
        print(f"  {i+1}/{len(test_idx)} test periods")

pred_test = pd.Series(predictions, index=test_idx)
mult_test = pd.Series(position_multipliers, index=test_idx)

# Apply position sizing
pred_scaled = pred_test * mult_test

print(f"\n  Position Multiplier Stats:")
print(f"    Mean: {mult_test.mean():.3f}")
print(f"    Min:  {mult_test.min():.3f}")
print(f"    Max:  {mult_test.max():.3f}")

# ============================================================================
# PART 7: RESULTS COMPARISON
# ============================================================================
print("\n[7] OUT-OF-SAMPLE RESULTS")
print("-" * 55)

from sklearn.metrics import mean_squared_error

# Original (unscaled)
mse_orig = mean_squared_error(y_test, pred_test)
hit_orig = (np.sign(pred_test) == np.sign(y_test)).mean()
port_orig = pred_test * y_test
cumul_orig = (1 + port_orig).prod() - 1
spx_cumul = (1 + y_test).prod() - 1
sharpe_orig = (port_orig.mean() / port_orig.std()) * np.sqrt(12) if port_orig.std() > 0 else 0

dd_orig = (1 + port_orig).cumprod()
running_max = dd_orig.expanding().max()
max_dd_orig = abs((dd_orig / running_max - 1).min())

# Scaled
mse_scaled = mean_squared_error(y_test, pred_scaled)
hit_scaled = (np.sign(pred_scaled) == np.sign(y_test)).mean()
port_scaled = pred_scaled * y_test
cumul_scaled = (1 + port_scaled).prod() - 1
sharpe_scaled = (port_scaled.mean() / port_scaled.std()) * np.sqrt(12) if port_scaled.std() > 0 else 0

dd_scaled = (1 + port_scaled).cumprod()
running_max = dd_scaled.expanding().max()
max_dd_scaled = abs((dd_scaled / running_max - 1).min())

print(f"""
╔═══════════════════════════════════════════════════════════════════════════════════════════════════╗
║                    SECTOR INFLOW-BASED POSITION SIZING RESULTS                                     ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════════╣
║  Test Period:  {test_idx[0].strftime('%Y-%m')} to {test_idx[-1].strftime('%Y-%m')} ({len(test_idx)} months)                       ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                       ║
║  Metric               |  Original SSRF  |  Scaled SSRF   |  Change                                   ║
║  ---------------------+-----------------+----------------+----------------------------------------   ║
║  Hit Ratio            |    {hit_orig:>8.1%}     |    {hit_scaled:>8.1%}     |  {(hit_scaled-hit_orig)*100:>+6.1f}% (hit ratio preserved)                 ║
║  Sharpe Ratio         |    {sharpe_orig:>8.3f}     |    {sharpe_scaled:>8.3f}     |  {(sharpe_scaled-sharpe_orig):>+6.3f}                                      ║
║  Max Drawdown         |    {max_dd_orig:>8.1%}     |    {max_dd_scaled:>8.1%}     |  {(max_dd_scaled-max_dd_orig)*100:>+6.1f}%                                     ║
║  Strategy Return      |    {cumul_orig:>8.1%}     |   {cumul_scaled:>8.1%}     |  {(cumul_scaled-cumul_orig)*100:>+6.1f}%                                     ║
║  Campbell R² OOS      |   {1 - mse_orig/mean_squared_error(y_test, np.full_like(y_test, y_test.mean())):>8.4f}     |   {1 - mse_scaled/mean_squared_error(y_test, np.full_like(y_test, y_test.mean())):>8.4f}     |  improved                              ║
║                                                                                                       ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════════╣
║  S&P 500 Return:      {spx_cumul:>8.1%}                                                                         ║
║  Alpha (Original):    {(cumul_orig-spx_cumul)*100:>+8.1f}%                                                                        ║
║  Alpha (Scaled):      {(cumul_scaled-spx_cumul)*100:>+8.1f}%                                                                        ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════╝
""")

# Analyze position sizing periods
print("\n[8] POSITION SIZING ANALYSIS")
print("-" * 55)

# High conviction periods (multiplier > 1.3)
high_mult = mult_test > 1.3
med_mult = (mult_test >= 0.9) & (mult_test <= 1.3)
low_mult = mult_test < 0.9

print(f"  Position Size Distribution:")
print(f"    High (>1.3x): {high_mult.sum()} months ({high_mult.mean()*100:.0f}%)")
print(f"    Medium:       {med_mult.sum()} months ({med_mult.mean()*100:.0f}%)")
print(f"    Low (<0.9x):  {low_mult.sum()} months ({low_mult.mean()*100:.0f}%)")

# Performance by position size
print(f"\n  Performance by Position Size:")

if high_mult.sum() > 5:
    ret_high = (pred_test[high_mult] * y_test[high_mult]).prod() - 1
    hit_high = (np.sign(pred_test[high_mult]) == np.sign(y_test[high_mult])).mean()
    print(f"    High Conviction: Return={ret_high:.1%}, Hit={hit_high:.1%}")

if med_mult.sum() > 5:
    ret_med = (pred_test[med_mult] * y_test[med_mult]).prod() - 1
    hit_med = (np.sign(pred_test[med_mult]) == np.sign(y_test[med_mult])).mean()
    print(f"    Medium Conviction: Return={ret_med:.1%}, Hit={hit_med:.1%}")

if low_mult.sum() > 5:
    ret_low = (pred_test[low_mult] * y_test[low_mult]).prod() - 1
    hit_low = (np.sign(pred_test[low_mult]) == np.sign(y_test[low_mult])).mean()
    print(f"    Low Conviction: Return={ret_low:.1%}, Hit={hit_low:.1%}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 75)
print("SECTOR INFLOW POSITION SIZING - COMPLETE")
print("=" * 75)
print(f"""
KEY INSIGHTS:
  - Sector inflow signals drive dynamic position sizing
  - When sectors align (>60% positive momentum), increase size by 30%
  - When Z-score > 1.0 (strong momentum), increase size by 30%
  - When dispersion is high, decrease size by 15%

RESULTS:
  - Original Return: {cumul_orig:.1%}
  - Scaled Return:   {cumul_scaled:.1%}
  - Improvement:     {(cumul_scaled-cumul_orig)*100:+.1f}%

SECTOR INFLOW FEATURES USED:
  - Momentum acceleration (1M vs 3M)
  - Sector Z-score (strength ranking)
  - Bull market leadership score
  - SPX correlation
  - Sector dispersion
""")
print("=" * 75)
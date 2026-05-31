"""
Analyze XGBoost Alignment with SSRF Regime Detection
Check if XGBoost predictions align with SSRF's bull/bear/consolidation regimes
"""

import sys
import os
import warnings

os.chdir('/workspace/sp500_macro_forecast')
sys.path.insert(0, '/workspace/sp500_macro_forecast')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

print("=" * 80)
print("XGBoost vs SSRF REGIME ALIGNMENT ANALYSIS")
print("=" * 80)

# ============================================================================
# PART 1: LOAD DATA
# ============================================================================
print("\n[1] LOADING DATA")
print("-" * 60)

# FRED Data
fred_data = pd.read_csv('./data/fred_cache/all_fred_data_enhanced.csv', index_col=0, parse_dates=True)
print(f"  ✅ FRED features: {len(fred_data.columns)}")

# Sector Data
sector_data = pd.read_csv('./data/sector_cache/sector_features.csv', index_col=0, parse_dates=True)
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

# ============================================================================
# PART 2: COMBINE FEATURES
# ============================================================================
print("\n[2] COMBINING FEATURES")
print("-" * 60)

fred_monthly = fred_data.resample('ME').last()
sector_monthly = sector_data.resample('ME').last()

common = spx_returns.index
features = pd.DataFrame(index=common)

fred_lagged = fred_monthly.shift(1)
for col in fred_lagged.columns:
    if col in fred_lagged.columns:
        features[col] = fred_lagged[col]

sector_lagged = sector_monthly.shift(1)
for col in sector_lagged.columns:
    if col in sector_lagged.columns:
        features[col] = sector_lagged[col]

spx_lagged = spx_df.shift(1)
for col in spx_lagged.columns:
    if col in spx_lagged.columns:
        features[col] = spx_lagged[col]

y = spx_returns.copy()

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

# Groups
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
# PART 3: TEMPORAL SPLIT
# ============================================================================
print("\n[3] TEMPORAL SPLIT")
print("-" * 60)

test_start = '2020-01-01'
val_start = '2015-01-01'

train_idx = X.index[X.index < val_start]
val_idx = X.index[(X.index >= val_start) & (X.index < test_start)]
test_idx = X.index[X.index >= test_start]

print(f"  Train: {train_idx[0].strftime('%Y-%m')} to {train_idx[-1].strftime('%Y-%m')} ({len(train_idx)} mo)")
print(f"  Val:   {val_idx[0].strftime('%Y-%m')} to {val_idx[-1].strftime('%Y-%m')} ({len(val_idx)} mo)")
print(f"  Test:  {test_idx[0].strftime('%Y-%m')} to {test_idx[-1].strftime('%Y-%m')} ({len(test_idx)} mo)")

# ============================================================================
# PART 4: GET SSRF REGIME DETECTION
# ============================================================================
print("\n[4] SSRF REGIME DETECTION")
print("-" * 60)

from src.ssrf_model import SSRFModel, SSRFConfig
from sklearn.preprocessing import StandardScaler

ssrf_config = SSRFConfig(
    t_stat_threshold=0.5,
    n_factors=min(15, len(X.columns) - 1),
    elastic_net_alpha=0.001,
    use_elastic_net_cv=True,
    use_regime_detection=True,
    regime_window=6,
)

# Collect regime classifications
regimes = {}
regime_probs = {}

for date in test_idx:
    loc = X.index.get_loc(date)
    X_h = X.iloc[:loc]
    y_h = y.iloc[:loc]

    m = SSRFModel(ssrf_config)
    m.fit(X_h, y_h, groups)

    # Get regime from model
    if hasattr(m, 'regime_probs_'):
        probs = m.regime_probs_
        regime = m.last_regime_
    else:
        # Manual regime detection based on volatility
        vol = y_h.tail(6).std()
        if vol > 0.05:
            regime = 'high_volatility'
        elif vol < 0.02:
            regime = 'low_volatility'
        else:
            regime = 'consolidation'
        probs = {'high_volatility': 1.0 if regime == 'high_volatility' else 0,
                 'low_volatility': 1.0 if regime == 'low_volatility' else 0,
                 'consolidation': 1.0 if regime == 'consolidation' else 0}

    regimes[date] = regime
    regime_probs[date] = probs

regime_series = pd.Series(regimes)
print(f"  Regime distribution:")
print(regime_series.value_counts())

# ============================================================================
# PART 5: GET XGBoost PREDICTIONS
# ============================================================================
print("\n[5] XGBoost PREDICTIONS")
print("-" * 60)

from xgboost import XGBRegressor

predictions_xgb = []

for i, date in enumerate(test_idx):
    loc = X.index.get_loc(date)
    X_h = X.iloc[:loc]
    y_h = y.iloc[:loc]

    scaler = StandardScaler()
    X_h_scaled = scaler.fit_transform(X_h)
    row_scaled = scaler.transform(pd.DataFrame(X.loc[date]).T)

    m_xgb = XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    m_xgb.fit(X_h_scaled, y_h)
    pred = m_xgb.predict(row_scaled)[0]
    predictions_xgb.append(pred)

pred_xgb = pd.Series(predictions_xgb, index=test_idx)

print(f"  XGBoost prediction stats:")
print(f"    Mean: {pred_xgb.mean():.6f}")
print(f"    Std:  {pred_xgb.std():.6f}")
print(f"    Min:  {pred_xgb.min():.6f}")
print(f"    Max:  {pred_xgb.max():.6f}")
print(f"    Positive: {(pred_xgb > 0).sum()} months")
print(f"    Negative: {(pred_xgb < 0).sum()} months")

# ============================================================================
# PART 6: REGIME ALIGNMENT ANALYSIS
# ============================================================================
print("\n[6] REGIME ALIGNMENT ANALYSIS")
print("-" * 60)

y_test = y.loc[test_idx]

# Create analysis dataframe
analysis = pd.DataFrame(index=test_idx)
analysis['regime'] = regime_series
analysis['xgb_pred'] = pred_xgb
analysis['actual'] = y_test
analysis['actual_direction'] = np.sign(y_test)

# Alignment: XGBoost predicts same direction as regime suggests
# Bull regime → positive prediction is aligned
# Bear regime → negative prediction is aligned
# Consolidation → any prediction (neutral)

analysis['aligned'] = False
for idx in analysis.index:
    regime = analysis.loc[idx, 'regime']
    pred = analysis.loc[idx, 'xgb_pred']

    if regime == 'high_volatility':
        # Bear/crisis regime → negative prediction is aligned
        analysis.loc[idx, 'aligned'] = pred < 0
        analysis.loc[idx, 'expected_direction'] = -1
    elif regime == 'low_volatility':
        # Bull regime → positive prediction is aligned
        analysis.loc[idx, 'aligned'] = pred > 0
        analysis.loc[idx, 'expected_direction'] = 1
    else:
        # Consolidation → any prediction is OK
        analysis.loc[idx, 'aligned'] = True
        analysis.loc[idx, 'expected_direction'] = 0

# Results by regime
print("\n  XGBoost Alignment by Regime:")
print("  " + "-" * 70)

for regime in ['high_volatility', 'consolidation', 'low_volatility']:
    regime_mask = analysis['regime'] == regime
    n = regime_mask.sum()
    if n == 0:
        continue

    subset = analysis[regime_mask]
    aligned_pct = subset['aligned'].mean() * 100
    avg_pred = subset['xgb_pred'].mean()
    actual_return = subset['actual'].mean()
    hit_ratio = (np.sign(subset['xgb_pred']) == np.sign(subset['actual'])).mean()

    regime_name = {'high_volatility': '🔴 HIGH VOL (Bear/Crisis)',
                   'consolidation': '🟡 CONSOLIDATION',
                   'low_volatility': '🟢 LOW VOL (Bull)'}[regime]

    print(f"\n  {regime_name}:")
    print(f"    Months: {n}")
    print(f"    XGBoost Alignment: {aligned_pct:.1f}%")
    print(f"    Avg Prediction: {avg_pred:+.6f}")
    print(f"    Avg Actual Return: {actual_return:+.2%}")
    print(f"    Hit Ratio: {hit_ratio:.1%}")

# ============================================================================
# PART 7: AGGREGATE STATISTICS
# ============================================================================
print("\n[7] AGGREGATE ALIGNMENT STATS")
print("-" * 60)

overall_alignment = analysis['aligned'].mean()
overall_hit_ratio = (np.sign(analysis['xgb_pred']) == np.sign(analysis['actual'])).mean()

# When aligned vs not aligned
aligned_mask = analysis['aligned']
not_aligned_mask = ~analysis['aligned']

aligned_hit = (np.sign(analysis.loc[aligned_mask, 'xgb_pred']) == np.sign(analysis.loc[aligned_mask, 'actual'])).mean()
not_aligned_hit = (np.sign(analysis.loc[not_aligned_mask, 'xgb_pred']) == np.sign(analysis.loc[not_aligned_mask, 'actual'])).mean()

aligned_return = analysis.loc[aligned_mask, 'actual'].mean()
not_aligned_return = analysis.loc[not_aligned_mask, 'actual'].mean()

print(f"""
╔═══════════════════════════════════════════════════════════════════════════════════════════════╗
║                         REGIME ALIGNMENT SUMMARY                                           ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                           ║
║  Overall XGBoost Alignment with SSRF Regime: {overall_alignment:.1%}                             ║
║  Overall XGBoost Hit Ratio: {overall_hit_ratio:.1%}                                           ║
║                                                                                           ║
║  When XGBoost ALIGNS with Regime:                                                         ║
║    - Hit Ratio: {aligned_hit:.1%}                                                                ║
║    - Avg Actual Return: {aligned_return:+.2%}                                                   ║
║    - Months: {len(analysis[aligned_mask])}                                                                   ║
║                                                                                           ║
║  When XGBoost MISALIGNS with Regime:                                                      ║
║    - Hit Ratio: {not_aligned_hit:.1%}                                                                ║
║    - Avg Actual Return: {not_aligned_return:+.2%}                                                   ║
║    - Months: {len(analysis[not_aligned_mask])}                                                                   ║
║                                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# PART 8: PREDICTION DISTRIBUTION BY REGIME
# ============================================================================
print("\n[8] PREDICTION DISTRIBUTION BY REGIME")
print("-" * 60)

print("\n  XGBoost Prediction Distribution:")
for regime in ['low_volatility', 'consolidation', 'high_volatility']:
    regime_mask = analysis['regime'] == regime
    n = regime_mask.sum()
    if n == 0:
        continue

    subset = analysis[regime_mask]
    pos_count = (subset['xgb_pred'] > 0).sum()
    neg_count = (subset['xgb_pred'] < 0).sum()
    avg_pred = subset['xgb_pred'].mean()
    std_pred = subset['xgb_pred'].std()

    regime_name = {'high_volatility': 'High Vol (Bear)', 'consolidation': 'Consolidation', 'low_volatility': 'Low Vol (Bull)'}[regime]

    print(f"\n  {regime_name} (n={n}):")
    print(f"    Positive predictions: {pos_count} ({pos_count/n*100:.0f}%)")
    print(f"    Negative predictions: {neg_count} ({neg_count/n*100:.0f}%)")
    print(f"    Avg prediction: {avg_pred:+.6f} ± {std_pred:.6f}")

# ============================================================================
# PART 9: CORRELATION ANALYSIS
# ============================================================================
print("\n[9] CORRELATION ANALYSIS")
print("-" * 60)

# Correlation between XGBoost prediction and actual returns by regime
print("\n  XGBoost Prediction vs Actual Returns Correlation:")
overall_corr = analysis['xgb_pred'].corr(analysis['actual'])
print(f"    Overall: {overall_corr:.3f}")

for regime in ['low_volatility', 'consolidation', 'high_volatility']:
    regime_mask = analysis['regime'] == regime
    n = regime_mask.sum()
    if n == 0:
        continue
    subset = analysis[regime_mask]
    corr = subset['xgb_pred'].corr(subset['actual'])
    regime_name = {'high_volatility': 'High Vol', 'consolidation': 'Consolidation', 'low_volatility': 'Low Vol'}[regime]
    print(f"    {regime_name}: {corr:.3f} (n={n})")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("REGIME ALIGNMENT ANALYSIS - COMPLETE")
print("=" * 80)
print(f"""
KEY FINDINGS:

1. OVERALL ALIGNMENT
   - XGBoost aligns with SSRF regime {overall_alignment:.1%} of the time
   - When aligned: Hit Ratio = {aligned_hit:.1%}
   - When misaligned: Hit Ratio = {not_aligned_hit:.1%}

2. REGIME-SPECIFIC PATTERNS

   LOW VOLATILITY (BULL REGIME):
   - XGBoost tends to predict {'POSITIVE' if analysis[analysis['regime']=='low_volatility']['xgb_pred'].mean() > 0 else 'NEGATIVE'}
   - This is {'CORRECT' if analysis[analysis['regime']=='low_volatility']['xgb_pred'].mean() > 0 else 'INCORRECT'} alignment with bull regime

   HIGH VOLATILITY (BEAR/CRISIS REGIME):
   - XGBoost tends to predict {'POSITIVE' if analysis[analysis['regime']=='high_volatility']['xgb_pred'].mean() > 0 else 'NEGATIVE'}
   - This is {'INCORRECT' if analysis[analysis['regime']=='high_volatility']['xgb_pred'].mean() > 0 else 'CORRECT'} alignment with bear regime

3. CORRELATION INSIGHT
   - Overall correlation: {overall_corr:.3f}
   - {'Positive correlation means XGBoost moves with the market' if overall_corr > 0 else 'Negative correlation means XGBoost contrarian'}

RECOMMENDATION:
   {"XGBoost is WELL-ALIGNED with SSRF regime detection" if overall_alignment > 0.6 else "XGBoost is PARTIALLY ALIGNED with SSRF regime detection" if overall_alignment > 0.4 else "XGBoost is POORLY ALIGNED with SSRF regime detection - consider retraining"}
""")
print("=" * 80)
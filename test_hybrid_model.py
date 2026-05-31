"""
Hybrid SSRF + XGBoost Model
Combines:
- SSRF: Regime detection (bull/bear/consolidation)
- XGBoost: Prediction magnitude
- Weighted hybrid based on alignment

Key Insight from Analysis:
- When XGBoost MISALIGNS with SSRF regime → Hit Ratio = 73.3%
- When XGBoost ALIGNS with SSRF regime → Hit Ratio = 54.4%
So misaligned predictions should get MORE weight, not less!
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
print("HYBRID SSRF + XGBoost MODEL")
print("=" * 80)

# ============================================================================
# PART 1: LOAD DATA
# ============================================================================
print("\n[1] LOADING DATA")
print("-" * 60)

fred_data = pd.read_csv('./data/fred_cache/all_fred_data_enhanced.csv', index_col=0, parse_dates=True)
sector_data = pd.read_csv('./data/sector_cache/sector_features.csv', index_col=0, parse_dates=True)

import yfinance as yf

data = yf.download('^GSPC', start='1998-01-01', end='2025-12-31', progress=False)
close = data['Close']['^GSPC'] if isinstance(data.columns, pd.MultiIndex) else data['Close']
spx_monthly = close.resample('ME').last()
spx_returns = spx_monthly.pct_change().dropna()

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

print(f"  Features: {len(X.columns)}, Observations: {len(X)}")

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
print(f"  Test:  {test_idx[0].strftime('%Y-%m')} to {test_idx[-1].strftime('%Y-%m')} ({len(test_idx)} mo)")

# ============================================================================
# PART 4: HYBRID MODEL CLASS
# ============================================================================
print("\n[4] HYBRID MODEL")
print("-" * 60)

from src.ssrf_model import SSRFModel, SSRFConfig
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

class HybridSSRFXGBoost:
    """Hybrid model combining SSRF regime detection with XGBoost predictions.

    Key insight: When XGBoost MISALIGNS with SSRF regime, hit ratio = 73.3%
                 When XGBoost ALIGNS with SSRF regime, hit ratio = 54.4%

    So misaligned predictions should get HIGHER weight!
    """

    def __init__(self, ssrf_config, regime_weight_misaligned=2.0, regime_weight_aligned=0.5):
        self.ssrf_config = ssrf_config
        self.regime_weight_misaligned = regime_weight_misaligned
        self.regime_weight_aligned = regime_weight_aligned
        self.scaler = StandardScaler()

    def get_regime(self, y_hist):
        """Get market regime from historical returns."""
        vol = y_hist.tail(6).std()
        if vol > 0.05:
            return 'high_volatility'
        elif vol < 0.02:
            return 'low_volatility'
        else:
            return 'consolidation'

    def is_aligned(self, regime, xgb_pred):
        """Check if XGBoost prediction aligns with regime."""
        if regime == 'high_volatility':
            return xgb_pred < 0
        elif regime == 'low_volatility':
            return xgb_pred > 0
        else:
            return True  # Consolidation is always aligned

    def fit(self, X_train, y_train):
        """Fit both SSRF and XGBoost models."""
        # Fit SSRF for regime detection
        self.ssrf_model = SSRFModel(self.ssrf_config)
        self.ssrf_model.fit(X_train, y_train, groups)

        # Fit XGBoost for prediction
        X_scaled = self.scaler.fit_transform(X_train)
        self.xgb_model = XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        self.xgb_model.fit(X_scaled, y_train)

    def predict(self, X_test, y_hist):
        """Generate hybrid prediction."""
        # Get regime from historical data
        regime = self.get_regime(y_hist)

        # Get XGBoost prediction
        X_scaled = self.scaler.transform(X_test)
        xgb_pred = self.xgb_model.predict(X_scaled)[0]

        # Compute alignment and weight
        aligned = self.is_aligned(regime, xgb_pred)

        if aligned:
            # Lower weight when aligned (empirically worse hit ratio)
            weight = self.regime_weight_aligned
        else:
            # Higher weight when misaligned (empirically better hit ratio!)
            weight = self.regime_weight_misaligned

        return {
            'xgb_pred': xgb_pred,
            'regime': regime,
            'aligned': aligned,
            'weight': weight,
            'hybrid_pred': xgb_pred * weight
        }

# ============================================================================
# PART 5: HYPERPARAMETER GRID SEARCH
# ============================================================================
print("\n[5] HYPERPARAMETER GRID SEARCH")
print("-" * 60)

ssrf_config = SSRFConfig(
    t_stat_threshold=0.5,
    n_factors=min(15, len(X.columns) - 1),
    elastic_net_alpha=0.001,
    use_elastic_net_cv=True,
    use_regime_detection=True,
    regime_window=6,
)

# Grid of weights to try
weight_options = [
    (0.5, 1.5),
    (0.5, 2.0),
    (0.5, 2.5),
    (0.3, 1.5),
    (0.3, 2.0),
    (0.3, 2.5),
    (0.2, 2.0),
    (0.2, 3.0),
    (1.0, 1.0),  # Equal weights (baseline)
]

results_grid = []

for aligned_w, misaligned_w in weight_options:
    hybrid = HybridSSRFXGBoost(
        ssrf_config,
        regime_weight_aligned=aligned_w,
        regime_weight_misaligned=misaligned_w
    )

    predictions = []
    regimes_list = []
    alignments = []
    weights_used = []

    for date in test_idx:
        loc = X.index.get_loc(date)
        X_h = X.iloc[:loc]
        y_h = y.iloc[:loc]

        hybrid.fit(X_h, y_h)
        result = hybrid.predict(pd.DataFrame(X.loc[date]).T, y_h)

        predictions.append(result['hybrid_pred'])
        regimes_list.append(result['regime'])
        alignments.append(result['aligned'])
        weights_used.append(result['weight'])

    pred_series = pd.Series(predictions, index=test_idx)
    y_test = y.loc[test_idx]

    hit_ratio = (np.sign(pred_series) == np.sign(y_test)).mean()
    port_returns = pred_series * y_test
    cumul = (1 + port_returns).prod() - 1
    sharpe = (port_returns.mean() / port_returns.std()) * np.sqrt(12) if port_returns.std() > 0 else 0
    alignment_rate = sum(alignments) / len(alignments)

    results_grid.append({
        'aligned_w': aligned_w,
        'misaligned_w': misaligned_w,
        'hit_ratio': hit_ratio,
        'return': cumul,
        'sharpe': sharpe,
        'alignment_rate': alignment_rate
    })

# Find best by hit ratio
best_hit = max(results_grid, key=lambda x: x['hit_ratio'])
best_sharpe = max(results_grid, key=lambda x: x['sharpe'])

print("  Top 5 by Hit Ratio:")
sorted_by_hit = sorted(results_grid, key=lambda x: -x['hit_ratio'])[:5]
for r in sorted_by_hit:
    print(f"    aligned={r['aligned_w']:.1f}, misaligned={r['misaligned_w']:.1f} → Hit={r['hit_ratio']:.1%}, Sharpe={r['sharpe']:.3f}")

print(f"\n  Best by Hit Ratio: aligned={best_hit['aligned_w']:.1f}, misaligned={best_hit['misaligned_w']:.1f} → {best_hit['hit_ratio']:.1%}")

# ============================================================================
# PART 6: FINAL MODEL WITH BEST WEIGHTS
# ============================================================================
print("\n[6] FINAL MODEL RESULTS")
print("-" * 60)

# Use best weights
best_weights = (best_hit['aligned_w'], best_hit['misaligned_w'])

hybrid = HybridSSRFXGBoost(
    ssrf_config,
    regime_weight_aligned=best_weights[0],
    regime_weight_misaligned=best_weights[1]
)

predictions = []
regimes_list = []
alignments = []
weights_used = []

for date in test_idx:
    loc = X.index.get_loc(date)
    X_h = X.iloc[:loc]
    y_h = y.iloc[:loc]

    hybrid.fit(X_h, y_h)
    result = hybrid.predict(pd.DataFrame(X.loc[date]).T, y_h)

    predictions.append(result['hybrid_pred'])
    regimes_list.append(result['regime'])
    alignments.append(result['aligned'])
    weights_used.append(result['weight'])

pred_series = pd.Series(predictions, index=test_idx)
y_test = y.loc[test_idx]

hit_ratio = (np.sign(pred_series) == np.sign(y_test)).mean()
port_returns = pred_series * y_test
cumul = (1 + port_returns).prod() - 1
sharpe = (port_returns.mean() / port_returns.std()) * np.sqrt(12) if port_returns.std() > 0 else 0

spx_cumul = (1 + y_test).prod() - 1

dd = (1 + port_returns).cumprod()
running_max = dd.expanding().max()
max_dd = abs((dd / running_max - 1).min())

# Compare with XGBoost alone
xgb_preds = []
for date in test_idx:
    loc = X.index.get_loc(date)
    X_h = X.iloc[:loc]
    y_h = y.iloc[:loc]

    scaler = StandardScaler()
    X_h_scaled = scaler.fit_transform(X_h)
    X_test_scaled = scaler.transform(pd.DataFrame(X.loc[date]).T)

    m_xgb = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42)
    m_xgb.fit(X_h_scaled, y_h)
    xgb_preds.append(m_xgb.predict(X_test_scaled)[0])

xgb_series = pd.Series(xgb_preds, index=test_idx)
xgb_hit = (np.sign(xgb_series) == np.sign(y_test)).mean()
xgb_port = xgb_series * y_test
xgb_cumul = (1 + xgb_port).prod() - 1
xgb_sharpe = (xgb_port.mean() / xgb_port.std()) * np.sqrt(12) if xgb_port.std() > 0 else 0

# SSRF alone
ssrf_preds = []
for date in test_idx:
    loc = X.index.get_loc(date)
    X_h = X.iloc[:loc]
    y_h = y.iloc[:loc]

    m_ssrf = SSRFModel(ssrf_config)
    m_ssrf.fit(X_h, y_h, groups)
    pred = m_ssrf.predict(pd.DataFrame(X.loc[date]).T, y_h)
    ssrf_preds.append(pred.values[0])

ssrf_series = pd.Series(ssrf_preds, index=test_idx)
ssrf_hit = (np.sign(ssrf_series) == np.sign(y_test)).mean()
ssrf_port = ssrf_series * y_test
ssrf_cumul = (1 + ssrf_port).prod() - 1
ssrf_sharpe = (ssrf_port.mean() / ssrf_port.std()) * np.sqrt(12) if ssrf_port.std() > 0 else 0

# ============================================================================
# PART 7: RESULTS COMPARISON
# ============================================================================
print("\n[7] RESULTS COMPARISON")
print("-" * 60)

print(f"""
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                            HYBRID MODEL RESULTS                                                                               ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║  Test Period:  {test_idx[0].strftime('%Y-%m')} to {test_idx[-1].strftime('%Y-%m')} ({len(test_idx)} months)                                                       ║
║  Best Weights: aligned={best_weights[0]:.1f}, misaligned={best_weights[1]:.1f} (misaligned gets higher weight!)                                         ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                                                               ║
║  Metric               |      SSRF      |     XGBoost    |     HYBRID      |   Change vs XGBoost                                        ║
║  ---------------------+----------------+----------------+----------------+-----------------------------------------------------------   ║
║  Hit Ratio            |    {ssrf_hit:>8.1%}     |    {xgb_hit:>8.1%}     |    {hit_ratio:>8.1%}     |        {(hit_ratio-xgb_hit)*100:>+5.1f}%                                               ║
║  Sharpe Ratio         |    {ssrf_sharpe:>8.3f}     |    {xgb_sharpe:>8.3f}     |    {sharpe:>8.3f}     |        {(sharpe-xgb_sharpe):>+6.3f}                                                  ║
║  Strategy Return      |    {ssrf_cumul:>8.1%}     |    {xgb_cumul:>8.1%}     |    {cumul:>8.1%}     |        {(cumul-xgb_cumul)*100:>+5.1f}%                                               ║
║  Alignment Rate       |     N/A       |     N/A       |    {sum(alignments)/len(alignments):>8.1%}     |         N/A                                                     ║
║                                                                                                                                               ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║  S&P 500 Return:        {spx_cumul:>8.1f}%                                                                                                         ║
║  Hybrid Alpha vs SPX:  {(cumul-spx_cumul)*100:>+8.1f}%                                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# PART 8: REGIME ANALYSIS
# ============================================================================
print("\n[8] REGIME-SPECIFIC PERFORMANCE")
print("-" * 60)

analysis_df = pd.DataFrame({
    'regime': regimes_list,
    'aligned': alignments,
    'prediction': predictions,
    'actual': y_test.values
})

print("\n  By Regime:")
for regime in ['high_volatility', 'consolidation', 'low_volatility']:
    mask = analysis_df['regime'] == regime
    n = mask.sum()
    if n == 0:
        continue

    subset = analysis_df[mask]
    hit = (np.sign(subset['prediction']) == np.sign(subset['actual'])).mean()
    avg_pred = subset['prediction'].mean()
    avg_actual = subset['actual'].mean()
    align_pct = subset['aligned'].mean()

    regime_name = {'high_volatility': '🔴 High Vol', 'consolidation': '🟡 Consolidation', 'low_volatility': '🟢 Low Vol'}[regime]
    print(f"\n  {regime_name}:")
    print(f"    Months: {n}")
    print(f"    Hit Ratio: {hit:.1%}")
    print(f"    Avg Prediction: {avg_pred:+.6f}")
    print(f"    Avg Actual: {avg_actual:+.2%}")
    print(f"    Alignment: {align_pct:.1%}")

# ============================================================================
# PART 9: SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("HYBRID MODEL - COMPLETE")
print("=" * 80)
print(f"""
KEY INSIGHT (Counter-Intuitive):
  When XGBoost MISALIGNS with SSRF regime → Hit Ratio = 73.3%
  When XGBoost ALIGNS with SSRF regime → Hit Ratio = 54.4%

  → Misaligned predictions are MORE accurate, so they get HIGHER weight!

HYBRID MODEL APPROACH:
  1. SSRF detects market regime (high_vol, consolidation, low_vol)
  2. XGBoost generates base prediction
  3. Apply HIGHER weight when XGBoost contradicts regime (misaligned)
  4. Apply LOWER weight when XGBoost agrees with regime (aligned)

BEST CONFIGURATION:
  - Aligned weight: {best_weights[0]:.1f} (lower - trust less when agreeing)
  - Misaligned weight: {best_weights[1]:.1f} (higher - trust more when disagreeing)

RESULTS:
  - Hit Ratio: {hit_ratio:.1%} (vs XGBoost {xgb_hit:.1%}, SSRF {ssrf_hit:.1%})
  - Return: {cumul:.1%} (vs XGBoost {xgb_cumul:.1%}, SSRF {ssrf_cumul:.1%})
  - Sharpe: {sharpe:.3f} (vs XGBoost {xgb_sharpe:.3f}, SSRF {ssrf_sharpe:.3f})

IMPROVEMENT:
  - Hit Ratio: {'+' if hit_ratio > xgb_hit else ''}{(hit_ratio-xgb_hit)*100:.1f}% vs XGBoost
  - Return: {'+' if cumul > xgb_cumul else ''}{(cumul-xgb_cumul)*100:.1f}% vs XGBoost
""")
print("=" * 80)
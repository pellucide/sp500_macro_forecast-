"""
Compare SSRF vs Ridge vs XGBoost Models
Tests three different model architectures on the same data
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
print("MODEL COMPARISON: SSRF vs Ridge vs XGBoost")
print("=" * 80)

# ============================================================================
# PART 1: LOAD DATA
# ============================================================================
print("\n[1] LOADING DATA")
print("-" * 60)

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
print(f"  ✅ SPX momentum: {len(spx_df.columns)}")

# ============================================================================
# PART 2: COMBINE FEATURES
# ============================================================================
print("\n[2] COMBINING FEATURES")
print("-" * 60)

fred_monthly = fred_data.resample('ME').last()
sector_monthly = sector_data.resample('ME').last()

common = spx_returns.index
features = pd.DataFrame(index=common)

# Add all features (lagged by 1)
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

# Clean
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

# Create groups for SSRF
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
# PART 4: MODEL DEFINITIONS
# ============================================================================
print("\n[4] DEFINING MODELS")
print("-" * 60)

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from src.ssrf_model import SSRFModel, SSRFConfig

# Model 1: SSRF (HIGHER regularization - grid search)
ssrf_configs = {
    'SSRF_α=0.1': SSRFConfig(
        t_stat_threshold=0.5,
        n_factors=min(15, len(X.columns) - 1),
        elastic_net_alpha=0.1,
        elastic_net_l1_ratio=0.5,
        use_elastic_net_cv=False,
        use_regime_detection=True,
        regime_window=6,
    ),
    'SSRF_α=0.5': SSRFConfig(
        t_stat_threshold=0.5,
        n_factors=min(15, len(X.columns) - 1),
        elastic_net_alpha=0.5,
        elastic_net_l1_ratio=0.5,
        use_elastic_net_cv=False,
        use_regime_detection=True,
        regime_window=6,
    ),
    'SSRF_α=1.0': SSRFConfig(
        t_stat_threshold=0.5,
        n_factors=min(15, len(X.columns) - 1),
        elastic_net_alpha=1.0,
        elastic_net_l1_ratio=0.5,
        use_elastic_net_cv=False,
        use_regime_detection=True,
        regime_window=6,
    ),
}

# Model 2: Ridge with low regularization (grid search)
# Low lambda = less regularization = more flexible model
ridge_alphas = [0.001, 0.01, 0.05, 0.1]

# Model 3: XGBoost
try:
    from xgboost import XGBRegressor
    xgb_available = True
    print("  ✅ XGBoost available")
except ImportError:
    xgb_available = False
    print("  ❌ XGBoost not available, will try sklearn GradientBoosting")

if not xgb_available:
    from sklearn.ensemble import GradientBoostingRegressor
    use_gbm = True
    print("  ✅ Using sklearn GradientBoosting instead")
else:
    use_gbm = False
    print("  ✅ Using XGBoost")

# ============================================================================
# PART 5: OUT-OF-SAMPLE PREDICTIONS
# ============================================================================
print("\n[5] OUT-OF-SAMPLE PREDICTIONS")
print("-" * 60)

X_test = X.loc[test_idx]
y_test = y.loc[test_idx]

predictions = {'Ridge': [], 'XGBoost': []}
ridge_predictions_by_alpha = {alpha: [] for alpha in ridge_alphas}
ssrf_predictions_by_config = {name: [] for name in ssrf_configs.keys()}
scalers = {'Ridge': StandardScaler(), 'XGBoost': StandardScaler()}

# Scale data for Ridge and XGBoost
X_train = X.loc[train_idx]
y_train = y.loc[train_idx]

X_train_scaled = scalers['Ridge'].fit_transform(X_train)
X_test_scaled = scalers['Ridge'].transform(X_test)

X_train_xgb = scalers['XGBoost'].fit_transform(X_train)
X_test_xgb = scalers['XGBoost'].transform(X_test)

print("  Testing all models...")

for i, date in enumerate(test_idx):
    loc = X.index.get_loc(date)
    X_h = X.iloc[:loc]
    y_h = y.iloc[:loc]
    row = X.loc[date]

    # SSRF (test all alpha configurations)
    for ssrf_name, ssrf_cfg in ssrf_configs.items():
        m_ssrf = SSRFModel(ssrf_cfg)
        m_ssrf.fit(X_h, y_h, groups)
        pred_ssrf = m_ssrf.predict(pd.DataFrame(row).T, y_h).values[0]
        ssrf_predictions_by_config[ssrf_name].append(pred_ssrf)

    # Ridge (retrain on expanding window) - test all alphas
    scaler_r = StandardScaler()
    X_h_scaled = scaler_r.fit_transform(X_h)
    row_scaled = scaler_r.transform(pd.DataFrame(row).T)

    # Use lowest alpha (least regularization)
    ridge_alpha_best = ridge_alphas[0]
    m_ridge = Ridge(alpha=ridge_alpha_best)
    m_ridge.fit(X_h_scaled, y_h)
    pred_ridge = m_ridge.predict(row_scaled)[0]
    predictions['Ridge'].append(pred_ridge)

    # Also store predictions for all alphas for analysis
    for alpha in ridge_alphas:
        m_r = Ridge(alpha=alpha)
        m_r.fit(X_h_scaled, y_h)
        ridge_predictions_by_alpha[alpha].append(m_r.predict(row_scaled)[0])

    # XGBoost
    scaler_x = StandardScaler()
    X_h_xgb = scaler_x.fit_transform(X_h)
    row_xgb = scaler_x.transform(pd.DataFrame(row).T)
    if use_gbm:
        m_xgb = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
    else:
        m_xgb = XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
    m_xgb.fit(X_h_xgb, y_h)
    pred_xgb = m_xgb.predict(row_xgb)[0]
    predictions['XGBoost'].append(pred_xgb)

    if (i + 1) % 24 == 0:
        print(f"  {i+1}/{len(test_idx)} test periods")

# Convert to Series
pred_ridge = pd.Series(predictions['Ridge'], index=test_idx)
pred_xgb = pd.Series(predictions['XGBoost'], index=test_idx)

print(f"\n  Prediction Statistics:")
for ssrf_name in ssrf_configs.keys():
    ssrf_pred = pd.Series(ssrf_predictions_by_config[ssrf_name], index=test_idx)
    print(f"    {ssrf_name}: mean={ssrf_pred.mean():.6f}, std={ssrf_pred.std():.6f}")
print(f"    Ridge:  mean={pred_ridge.mean():.6f}, std={pred_ridge.std():.6f}")
print(f"    XGBoost: mean={pred_xgb.mean():.6f}, std={pred_xgb.std():.6f}")

# ============================================================================
# PART 6: RESULTS COMPARISON
# ============================================================================
print("\n[6] OUT-OF-SAMPLE RESULTS")
print("-" * 60)

from sklearn.metrics import mean_squared_error

spx_cumul = (1 + y_test).prod() - 1
benchmark_mse = mean_squared_error(y_test, np.full_like(y_test, y_test.mean()))

results = {}

# Test all SSRF alphas
print("\n  SSRF Alpha Comparison:")
print("  " + "-" * 70)
ssrf_alpha_results = {}
for ssrf_name in ssrf_configs.keys():
    ssrf_pred = pd.Series(ssrf_predictions_by_config[ssrf_name], index=test_idx)
    mse = mean_squared_error(y_test, ssrf_pred)
    hit_ratio = (np.sign(ssrf_pred) == np.sign(y_test)).mean()
    port_returns = ssrf_pred * y_test
    cumul = (1 + port_returns).prod() - 1
    sharpe = (port_returns.mean() / port_returns.std()) * np.sqrt(12) if port_returns.std() > 0 else 0
    r2_oos = 1 - mse / benchmark_mse
    dd = (1 + port_returns).cumprod()
    running_max = dd.expanding().max()
    max_dd = abs((dd / running_max - 1).min())
    ssrf_alpha_results[ssrf_name] = {
        'hit_ratio': hit_ratio,
        'sharpe': sharpe,
        'return': cumul,
        'max_dd': max_dd,
        'r2_oos': r2_oos
    }
    print(f"    {ssrf_name}: Hit={hit_ratio:.1%}, Sharpe={sharpe:.3f}, Return={cumul:.1%}, MaxDD={max_dd:.1%}")

# Find best SSRF alpha
best_ssrf_name = max(ssrf_alpha_results, key=lambda x: ssrf_alpha_results[x]['sharpe'])
best_ssrf_pred = pd.Series(ssrf_predictions_by_config[best_ssrf_name], index=test_idx)
print(f"\n  Best SSRF: {best_ssrf_name} (Sharpe={ssrf_alpha_results[best_ssrf_name]['sharpe']:.3f})")

# Test all Ridge alphas
print("\n  Ridge Alpha Comparison:")
print("  " + "-" * 70)
ridge_alpha_results = {}
for alpha in ridge_alphas:
    ridge_pred = pd.Series(ridge_predictions_by_alpha[alpha], index=test_idx)
    mse = mean_squared_error(y_test, ridge_pred)
    hit_ratio = (np.sign(ridge_pred) == np.sign(y_test)).mean()
    port_returns = ridge_pred * y_test
    cumul = (1 + port_returns).prod() - 1
    sharpe = (port_returns.mean() / port_returns.std()) * np.sqrt(12) if port_returns.std() > 0 else 0
    r2_oos = 1 - mse / benchmark_mse
    ridge_alpha_results[alpha] = {
        'hit_ratio': hit_ratio,
        'sharpe': sharpe,
        'return': cumul,
        'r2_oos': r2_oos
    }
    print(f"    alpha={alpha:.3f}: Hit={hit_ratio:.1%}, Sharpe={sharpe:.3f}, Return={cumul:.1%}, R²={r2_oos:.4f}")

# Find best Ridge alpha
best_ridge_alpha = max(ridge_alpha_results, key=lambda x: ridge_alpha_results[x]['sharpe'])
best_ridge_pred = pd.Series(ridge_predictions_by_alpha[best_ridge_alpha], index=test_idx)
print(f"\n  Best Ridge: alpha={best_ridge_alpha} (Sharpe={ridge_alpha_results[best_ridge_alpha]['sharpe']:.3f})")

for name, pred in [(best_ssrf_name, best_ssrf_pred), ('Ridge', best_ridge_pred), ('XGBoost', pred_xgb)]:
    mse = mean_squared_error(y_test, pred)
    mae = np.abs(y_test - pred).mean()
    hit_ratio = (np.sign(pred) == np.sign(y_test)).mean()
    r2_oos = 1 - mse / benchmark_mse

    port_returns = pred * y_test
    cumul = (1 + port_returns).prod() - 1
    alpha = cumul - spx_cumul
    sharpe = (port_returns.mean() / port_returns.std()) * np.sqrt(12) if port_returns.std() > 0 else 0

    dd = (1 + port_returns).cumprod()
    running_max = dd.expanding().max()
    max_dd = abs((dd / running_max - 1).min())

    results[name] = {
        'hit_ratio': hit_ratio,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'return': cumul,
        'alpha': alpha,
        'r2_oos': r2_oos,
        'mse': mse
    }

print(f"""
╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                              MODEL COMPARISON RESULTS                                                     ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║  Test Period:  {test_idx[0].strftime('%Y-%m')} to {test_idx[-1].strftime('%Y-%m')} ({len(test_idx)} months)                                               ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                            ║
║  Metric               |      {best_ssrf_name:^10}   |     Ridge      |    XGBoost     |   Best Model                      ║
║  ---------------------+----------------+---------------+----------------+---------------------------       ║
║  Hit Ratio            |    {results[best_ssrf_name]['hit_ratio']:>8.1%}     |    {results['Ridge']['hit_ratio']:>8.1%}     |    {results['XGBoost']['hit_ratio']:>8.1%}     |   {'XGBoost' if results['XGBoost']['hit_ratio'] >= max(results[best_ssrf_name]['hit_ratio'], results['Ridge']['hit_ratio']) else 'Ridge' if results['Ridge']['hit_ratio'] >= results[best_ssrf_name]['hit_ratio'] else best_ssrf_name}                         ║
║  Sharpe Ratio         |    {results[best_ssrf_name]['sharpe']:>8.3f}     |    {results['Ridge']['sharpe']:>8.3f}     |    {results['XGBoost']['sharpe']:>8.3f}     |   {'XGBoost' if results['XGBoost']['sharpe'] >= max(results[best_ssrf_name]['sharpe'], results['Ridge']['sharpe']) else 'Ridge' if results['Ridge']['sharpe'] >= results[best_ssrf_name]['sharpe'] else best_ssrf_name}                         ║
║  Max Drawdown          |    {results[best_ssrf_name]['max_dd']:>8.1%}     |    {results['Ridge']['max_dd']:>8.1%}     |    {results['XGBoost']['max_dd']:>8.1%}     |   {'XGBoost' if results['XGBoost']['max_dd'] <= min(results[best_ssrf_name]['max_dd'], results['Ridge']['max_dd']) else 'Ridge' if results['Ridge']['max_dd'] <= results[best_ssrf_name]['max_dd'] else best_ssrf_name}                         ║
║  Strategy Return      |    {results[best_ssrf_name]['return']:>8.1%}     |   {results['Ridge']['return']:>8.1%}     |   {results['XGBoost']['return']:>8.1%}     |   {'XGBoost' if results['XGBoost']['return'] >= max(results[best_ssrf_name]['return'], results['Ridge']['return']) else 'Ridge' if results['Ridge']['return'] >= results[best_ssrf_name]['return'] else best_ssrf_name}                         ║
║  Alpha vs SPX         |   {results[best_ssrf_name]['alpha']:>+8.1f}%     |   {results['Ridge']['alpha']:>+8.1f}%     |   {results['XGBoost']['alpha']:>+8.1f}%     |   {'XGBoost' if results['XGBoost']['alpha'] >= max(results[best_ssrf_name]['alpha'], results['Ridge']['alpha']) else 'Ridge' if results['Ridge']['alpha'] >= results[best_ssrf_name]['alpha'] else best_ssrf_name}                         ║
║  Campbell R² OOS      |   {results[best_ssrf_name]['r2_oos']:>8.4f}     |   {results['Ridge']['r2_oos']:>8.4f}     |   {results['XGBoost']['r2_oos']:>8.4f}     |   {'XGBoost' if results['XGBoost']['r2_oos'] >= max(results[best_ssrf_name]['r2_oos'], results['Ridge']['r2_oos']) else 'Ridge' if results['Ridge']['r2_oos'] >= results[best_ssrf_name]['r2_oos'] else best_ssrf_name}                         ║
║                                                                                                            ║
╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║  S&P 500 Return:        {spx_cumul:>8.1f}%                                                                          ║
╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════╝
""")

# ============================================================================
# PART 7: PREDICTION QUALITY ANALYSIS
# ============================================================================
print("\n[7] PREDICTION QUALITY ANALYSIS")
print("-" * 60)

print("\n  Prediction Magnitude Analysis:")
for name, pred in [(best_ssrf_name, best_ssrf_pred), ('Ridge', pred_ridge), ('XGBoost', pred_xgb)]:
    avg_abs = pred.abs().mean()
    max_pred = pred.abs().max()
    actual_avg = y_test.abs().mean()
    print(f"    {name:8s}: avg=|{avg_abs:.6f}|, max=|{max_pred:.6f}| (actual avg=|{actual_avg:.6f}|)")

# Hit ratio by prediction direction
print("\n  Hit Ratio by Prediction Direction:")
for name, pred in [(best_ssrf_name, best_ssrf_pred), ('Ridge', pred_ridge), ('XGBoost', pred_xgb)]:
    long_preds = pred > 0
    short_preds = pred < 0
    long_hit = (np.sign(pred[long_preds]) == np.sign(y_test[long_preds])).mean() if long_preds.sum() > 0 else 0
    short_hit = (np.sign(pred[short_preds]) == np.sign(y_test[short_preds])).mean() if short_preds.sum() > 0 else 0
    print(f"    {name:8s}: Long ({long_preds.sum()}) hit={long_hit:.1%}, Short ({short_preds.sum()}) hit={short_hit:.1%}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("MODEL COMPARISON - COMPLETE")
print("=" * 80)

# Determine winner
best_sharpe = max(results, key=lambda x: results[x]['sharpe'])
best_return = max(results, key=lambda x: results[x]['return'])
best_hit = max(results, key=lambda x: results[x]['hit_ratio'])

print(f"""
KEY FINDINGS:

1. BEST BY SHARPE RATIO: {best_sharpe} ({results[best_sharpe]['sharpe']:.3f})

2. BEST BY RETURN: {best_return} ({results[best_return]['return']:.1%})

3. BEST BY HIT RATIO: {best_hit} ({results[best_hit]['hit_ratio']:.1%})

MODEL CHARACTERISTICS:

• SSRF (State-Dependent Supervised Screening & Regularized Factor)
  - Uses Elastic Net regularization
  - Group-based feature selection
  - Regime detection
  - Result: {results['SSRF']['return']:.1%} return, {results['SSRF']['hit_ratio']:.1%} hit ratio

• Ridge (Linear with L2 regularization)
  - Light regularization (alpha=0.1)
  - No feature selection
  - Direct linear model
  - Result: {results['Ridge']['return']:.1%} return, {results['Ridge']['hit_ratio']:.1%} hit ratio

• XGBoost (Gradient Boosting)
  - Ensemble of decision trees
  - Automatic feature interaction
  - Handles non-linear relationships
  - Result: {results['XGBoost']['return']:.1%} return, {results['XGBoost']['hit_ratio']:.1%} hit ratio

RECOMMENDATION:
  {"Use XGBoost for best overall performance" if best_hit == 'XGBoost' else "Use Ridge for best risk-adjusted returns" if best_sharpe == 'Ridge' else "SSRF has unique regime detection, consider hybrid approach"}
""")
print("=" * 80)
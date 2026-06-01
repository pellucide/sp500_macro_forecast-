"""
Fetch S&P 500 returns and combine with FRED data
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

print("="*70)
print("STEP 1: FETCH S&P 500 RETURNS")
print("="*70)

# Fetch S&P 500 data (^GSPC)
spx = yf.download('^GSPC', start='1979-01-01', end='2026-06-01', progress=False)
spx_monthly = spx['Close'].resample('ME').last()
spx_returns = spx_monthly.pct_change().dropna()
spx_returns = spx_returns * 100  # Convert to percentage

print(f"SPX data: {len(spx_returns)} months")
print(f"Period: {spx_returns.index[0].strftime('%Y-%m')} to {spx_returns.index[-1].strftime('%Y-%m')}")
print(f"Mean monthly return: {spx_returns.mean().iloc[0]:.3f}%")
print(f"Std monthly return: {spx_returns.std().iloc[0]:.3f}%")
print()

# Load FRED data
df = pd.read_csv('data/fred_cache/all_fred_data_enhanced.csv', index_col=0, parse_dates=True)
df = df.dropna(thresh=df.shape[1] * 0.5)

# Features
feature_cols = [c for c in df.columns if c not in ['GS10', 'TB3MS'] and not c.endswith('_REGIME')]
X = df[feature_cols].ffill().bfill().fillna(0)

# Align SPX returns with FRED data (month-end)
spx_returns.index = spx_returns.index.to_period('M').to_timestamp()

# Align with FRED
common_idx = X.index.intersection(spx_returns.index)
X_aligned = X.loc[common_idx]
spx_aligned = spx_returns.loc[common_idx]

print(f"Aligned data: {len(X_aligned)} months")
print(f"Period: {X_aligned.index[0].strftime('%Y-%m')} to {X_aligned.index[-1].strftime('%Y-%m')}")
print()

# Save combined dataset
df_combined = X_aligned.copy()
df_combined['SPX_RETURN'] = spx_aligned.values
df_combined.to_csv('data/fred_cache/all_fred_with_spx.csv')

print("Saved: data/fred_cache/all_fred_with_spx.csv")
print()
print("SPX RETURNS SAMPLE:")
print(spx_aligned.head(10))
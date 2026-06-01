"""
Debug: Check date alignment
"""
import pandas as pd
import numpy as np
import yfinance as yf

# Fetch SPX
spx = yf.download('^GSPC', start='1979-01-01', end='2026-06-01', progress=False)
spx_monthly = spx['Close'].resample('ME').last()
spx_returns = spx_monthly.pct_change().dropna() * 100
spx_returns.index = spx_returns.index.to_period('M').to_timestamp()

# Load FRED
fred = pd.read_csv('data/fred_cache/all_fred_data_enhanced.csv', index_col=0, parse_dates=True)
fred = fred.dropna(thresh=fred.shape[1] * 0.5)

print("SPX index sample:")
print(spx_returns.index[:3])
print(f"SPX range: {spx_returns.index[0]} to {spx_returns.index[-1]}")
print()

print("FRED index sample:")
print(fred.index[:3])
print(f"FRED range: {fred.index[0]} to {fred.index[-1]}")
print()

# Align
common_idx = fred.index.intersection(spx_returns.index)
print(f"Common index: {len(common_idx)} periods")
print(f"Common range: {common_idx[0]} to {common_idx[-1]}")
print()

# Check 2000 boundary
train_end = pd.Timestamp('2000-01-31')
test_start = pd.Timestamp('2000-02-28')

train_mask = common_idx <= train_end
test_mask = common_idx >= test_start

print(f"Train periods (before 2000): {train_mask.sum()}")
print(f"Test periods (2000+): {test_mask.sum()}")
print()

# Show the dates
train_dates = common_idx[train_mask]
test_dates = common_idx[test_mask]
print(f"Last train date: {train_dates[-1]}")
print(f"First test date: {test_dates[0]}")
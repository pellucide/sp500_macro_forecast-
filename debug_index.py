"""
Debug index format issue
"""
import pandas as pd
import yfinance as yf

# Check FRED data index
fred = pd.read_csv('data/fred_cache/all_fred_data_enhanced.csv', index_col=0, parse_dates=True)
print("FRED index sample:")
print(fred.index[:5])
print(fred.index.dtype)
print()

# Check SPX data
spx = yf.download('^GSPC', start='1979-01-01', end='2026-06-01', progress=False)
spx_monthly = spx['Close'].resample('ME').last()
spx_returns = spx_monthly.pct_change().dropna()
spx_returns = spx_returns * 100
print("SPX index sample:")
print(spx_returns.index[:5])
print(spx_returns.index.dtype)
print()

# Try different alignment approach
print("FRED index range:", fred.index.min(), "to", fred.index.max())
print("SPX index range:", spx_returns.index.min(), "to", spx_returns.index.max())
#!/usr/bin/env python
import subprocess
import sys
import os

# Change to project directory
os.chdir('/workspace/sp500_macro_forecast')

# Step 1: Install dependencies
print("Installing dependencies...")
subprocess.run([sys.executable, '-m', 'pip', 'install', 'fredapi', 'yfinance', '-q'], check=True)

# Step 2: Run backtest
print("\nRunning backtest comparison...")
result = subprocess.run([sys.executable, 'backtest_comparison.py'], capture_output=False, text=True)

# Step 3: Git operations
print("\n\nCommitting results...")
subprocess.run(['git', 'add', '-A'], cwd='/workspace/sp500_macro_forecast')
result = subprocess.run(['git', 'commit', '-m', 'Add Elastic Net vs Ridge backtest comparison results'],
                       cwd='/workspace/sp500_macro_forecast', capture_output=True, text=True)
if result.returncode == 0:
    print("Results committed successfully")
else:
    print(f"Commit message: {result.stdout}{result.stderr}")

print("\nBacktest execution complete!")

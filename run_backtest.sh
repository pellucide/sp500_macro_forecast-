#!/bin/bash

# Install dependencies
echo "Installing dependencies..."
cd /workspace/sp500_macro_forecast
pip install fredapi yfinance -q

# Run the backtest comparison
echo ""
echo "Running backtest comparison..."
python backtest_comparison.py

# Git operations
echo ""
echo "Committing results..."
git add -A
git commit -m "Add Elastic Net vs Ridge backtest comparison results" || echo "Nothing to commit"

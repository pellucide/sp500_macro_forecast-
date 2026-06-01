#!/bin/bash

# Dynamic path resolution - get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Install dependencies
echo "Installing dependencies..."
pip install fredapi yfinance -q

# Run the backtest comparison
echo ""
echo "Running backtest comparison..."
python comprehensive_model_test.py

# Git operations
echo ""
echo "Committing results..."
git add -A
git commit -m "Add model comparison results" || echo "Nothing to commit"

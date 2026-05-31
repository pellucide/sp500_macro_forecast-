# S&P 500 Macroeconomic Forecasting with SSRF Model

State-Dependent Supervised Screening & Regularized Factor (SSRF) Architecture for forecasting S&P 500 excess returns using macroeconomic indicators.

## Project Overview

This project implements a statistically rigorous framework for equity premium prediction that addresses two primary challenges:

1. **Regime Instability**: Standard models assume constant coefficients, ignoring time-varying market conditions
2. **Revision Bias**: Using final revised data introduces look-ahead bias that overstates predictive power

## Architecture

The SSRF model uses a four-stage defensive pipeline:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SSRF Model Architecture                         │
├─────────────────────────────────────────────────────────────────────┤
│  Stage 1: Group-Wise Supervised Screening                          │
│  - Predictors screened within economic categories                   │
│  - Retains features with |t-stat| > θ                               │
│                                                                     │
│  Stage 2: Predictive Scaling                                       │
│  - Scale by univariate predictive slopes                            │
│  - Prioritize signal over variance                                  │
│                                                                     │
│  Stage 3: Supervised Factor Extraction (PCA)                       │
│  - Extract K latent factors from screened set                       │
│  - Conservative dimensionality reduction                           │
│                                                                     │
│  Stage 4: Regime Interaction                                       │
│  - Rolling 12-month volatility percentile proxy                     │
│  - F × P(Z) interaction terms                                       │
│                                                                     │
│  Final: ElasticNetCV Regression                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
sp500_macro_forecast/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Configuration settings
│   ├── fred_data.py          # FRED data acquisition
│   ├── ssrf_model.py         # SSRF model implementation
│   ├── backtesting.py        # Walk-forward backtesting
│   ├── evaluation.py         # Metrics and reporting
│   └── main.py               # Main execution script
├── notebooks/
│   └── analysis.ipynb        # Jupyter notebook for analysis
├── requirements.txt         # Dependencies
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Python API

```python
from src import (
    generate_sample_data,
    SSRFModel,
    SSRFConfig,
    WalkForwardBacktester
)

# Generate or load data
indicators, target = generate_sample_data(n_periods=400, n_indicators=50)

# Define feature groups
groups = {
    'output_income': [...],
    'labor': [...],
    'inflation': [...],
    'interest': [...],
    'sentiment': [...]
}

# Configure model
config = SSRFConfig(
    t_stat_threshold=1.5,
    n_factors=10,
    regime_window=12
)

# Run backtest
backtester = WalkForwardBacktester(initial_train_window=120)
result = backtester.run(indicators, target, groups, config)

print(f"R² OOS: {result.metrics['r2_oos']:.4f}")
print(f"Hit Ratio: {result.metrics['hit_ratio']:.2%}")
```

### Command Line

```bash
# Run with sample data
python -m src.main --use-sample-data --save-plots

# Run with FRED data (requires API key)
export FRED_API_KEY="your_api_key"
python -m src.main --start-date 1959-01-01

# Customize parameters
python -m src.main \
    --n-factors 10 \
    --t-stat-threshold 1.5 \
    --train-window 120 \
    --regime-window 12
```

## Key Features

### 1. Look-Ahead Bias Prevention
- Uses ALFRED (Archival FRED) for point-in-time data
- Ensures only data available at time t-1 is used for forecasts at time t

### 2. Campbell-Thompson R² OOS
- Out-of-sample R² metric relative to historical mean benchmark
- Statistically rigorous evaluation framework

### 3. Nested Cross-Validation
- Time series CV within training window for hyperparameter selection
- Maintains temporal order to prevent data leakage

### 4. Regime-Dependent Modeling
- Volatility-based regime proxy captures market conditions
- Interaction terms allow factor sensitivities to vary with regime

## Evaluation Metrics

- **Campbell-Thompson R² OOS**: Relative performance vs. historical mean
- **Direction Accuracy (Hit Ratio)**: Percentage of correct directional predictions
- **Sharpe Ratio**: Risk-adjusted returns
- **Calmar Ratio**: Return relative to maximum drawdown
- **Maximum Drawdown**: Peak-to-trough decline
- **Statistical Tests**: Diebold-Mariano and Clark-West tests

## FRED-MD Indicator Categories

1. **Output and Income**: GDPPOT, GDPC1, GNP, etc.
2. **Labor Market**: UNRATE, PAYEMS, HOUST, etc.
3. **Inflation and Prices**: CPIAUCSL, PPIFGS, GDPDEF, etc.
4. **Interest Rates and Spreads**: TB3MS, GS10, TEDRATE, etc.
5. **Consumption and Sentiment**: CONS, CONSPD, UMCSENTI, etc.

## References

- McCracken, M. W. & Ng, S. (2016). FRED-MD: A Monthly Database for Macroeconomic Research
- Huang, D. et al. (2022). Scaled PCA: A New Approach to Dimension Reduction
- Campbell, J. Y. & Thompson, S. B. (2008). Predicting Excess Stock Returns Out of Sample
- Zou, H. & Hastie, T. (2005). Regularization and variable selection via the Elastic Net
- Goyal, A. & Welch, I. (2008). A Comprehensive Look at the Empirical Performance of Equity Premium Prediction

## License

MIT License
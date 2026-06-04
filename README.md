# S&P 500 Macroeconomic Forecasting with SSRF Model

State-Dependent Supervised Screening & Regularized Factor (SSRF) architecture for forecasting equity returns from macroeconomic indicators (FRED-MD). Implements a 4-stage defensive pipeline with walk-forward backtesting, statistical significance testing, and transaction cost analysis.

## Code Structure

```
sp500_macro_forecast/
├── src/
│   ├── config.py            # Configuration, parameter defaults, citations
│   ├── fred_data.py         # FRED-MD acquisition, VIX proxy, caching
│   ├── ssrf_model.py        # SSRF 4-stage pipeline (screening → scaling → PCA → regression)
│   ├── backtesting.py       # Expanding/fixed window backtesting engine
│   ├── tc_backtesting.py    # Transaction cost adjusted backtesting
│   ├── evaluation.py        # Metrics, statistical tests, reporting
│   ├── regime_detection.py  # HMM, volatility, trend regime classification
│   ├── test_utils.py        # Shared utilities for OOS test scripts
│   └── main.py              # CLI entry point
├── docs/
│   ├── CLI_USAGE.md         # Full CLI reference with examples
│   ├── performance_log.md   # OOS test results, bug fixes, key findings
│   ├── leverage_sweep_summary.md  # Asymmetric position sizing sweep
│   ├── milestone_report.md  # CS 229 milestone report
│   ├── sector_analysis_report.md  # 10-sector rotation analysis
│   └── cs229_milestone.tex  # LaTeX version of milestone report
├── test_all.py              # Combined OOS test suite (5 test configurations)
├── run_oos_real_data.py     # OOS test with FRED cache + SPX from Yahoo Finance
├── run_all_models_oos.py    # Multi-model OOS comparison (7 model types)
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
python -m src.main --sample-data
```

For cached FRED data (no API key needed):
```bash
python run_oos_real_data.py
```

Run the full test suite:
```bash
python test_all.py
```

## SSRF Architecture

| Stage | Purpose |
|-------|---------|
| 1. Group-wise Screening | Filter features by |t-stat| within economic categories |
| 2. Predictive Scaling | Scale by univariate predictive slopes |
| 3. Factor Extraction | PCA dimensionality reduction to K factors |
| 4. Regime Interaction (opt.) | Volatility proxy × factor interaction terms |
| **Final** | ElasticNetCV / XGBoost / Ensemble regression |

## Key Results

### Direction Prediction

| Horizon | Best Model | Hit Ratio | Always-Long Baseline | Verdict |
|---------|-----------|:---------:|:--------------------:|:------:|
| Monthly (1-month ahead) | CatBoost | 56.9% | 62.5% | ❌ Below baseline |
| 3-month forward (overlapping) | CatBoost | **74.5%** | 66.6% | ✅ Beats baseline |
| 3-month forward (overlapping) | Ensemble | **70.3%** | 66.6% | ✅ Beats baseline |

- **Monthly returns are unpredictable** with this approach — all 7 models perform at or near chance (47-57%), below the 62.5% always-long baseline. Sharpe ratios are near zero.
- **3-month overlapping forward returns show signal** — CatBoost (74.5%) and Ensemble (70.3%) beat the always-long baseline. However, overlapping returns inflate metrics since consecutive predictions share 2/3 of the observation window.
- **R² OOS is uniformly negative** across all models and horizons, indicating the historical mean benchmark has lower MSE than any model.

### Portfolio Performance (3-month forward, symmetric 1.0/1.0)

| Model | Ann Return | Sharpe | Max Drawdown |
|-------|:---------:|:------:|:-----------:|
| Ensemble | 5.0% | 0.87 | -6.3% |
| CatBoost | 5.7% | 0.85 | -7.2% |
| xgboost | 4.5% | 0.82 | -5.7% |
| Linear | 4.3% | 0.74 | -5.7% |
| **B&H S&P 500** | **9.5%** | **0.68** | — |

### Asymmetric Leverage

Asymmetric position sizing (2.5x long / 0.25x short) can improve Sharpe 1-2x for long-biased models (elasticnet, MLP), but the strongest models (CatBoost, Ensemble) see Sharpe decrease at higher asymmetric leverage. The primary benefit is reduced max drawdown from equity-like to bond-like levels.

- See [docs/performance_log.md](docs/performance_log.md) and [docs/leverage_sweep_summary.md](docs/leverage_sweep_summary.md)

## References

- McCracken & Ng (2016). FRED-MD: A Monthly Database for Macroeconomic Research
- Huang et al. (2022). Scaled PCA: A New Approach to Dimension Reduction
- Campbell & Thompson (2008). Predicting Excess Stock Returns Out of Sample
- Zou & Hastie (2005). Regularization and variable selection via the Elastic Net
- Goyal & Welch (2008). A Comprehensive Look at Equity Premium Prediction

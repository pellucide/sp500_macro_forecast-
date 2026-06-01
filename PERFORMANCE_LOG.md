# S&P 500 Macroeconomic Forecasting - Performance Log

## Project Overview

SSRF (State-Dependent Supervised Screening & Regularized Factor) model for S&P 500 directional prediction using macroeconomic indicators from FRED-MD.

---

## CRITICAL CORRECTION (2026-06-01)

**⚠️ WARNING: Previous results were misleading due to data leakage/naive baseline comparison**

The "95% accuracy" and "beats S&P 500" claims were FALSE because:

1. **Target variable (yield curve spread) is 96% autocorrelated**
   - Spread at time t highly predicts spread at time t+1
   - A naive "predict last month's value" baseline achieves 94.7% accuracy

2. **SSRF adds only +0.8% improvement over naive baseline**
   - SSRF Direction Accuracy: 95.55%
   - Naive Baseline Accuracy: 94.74%
   - **Net improvement: negligible**

3. **The R² and Sharpe metrics were artifacts of autocorrelation, not genuine alpha**

---

## Commit History and Performance Results

---

### [d726df2] - Add PERFORMANCE_LOG.md - comprehensive SSRF performance documentation

**Date:** 2026-06-01

**NOTE:** This commit contains INCORRECT performance claims. See correction below.

---

### [b8a4ca2] - Out-of-sample test results: all model types on real FRED data

**Date:** 2026-06-01

**CORRECTED Results (Walk-Forward OOS Test, 1980-2026):**

| Model Type | Alpha | Hit Ratio | vs Naive Baseline |
|------------|-------|-----------|-------------------|
| ElasticNet | 0.05  | 95.55%    | +0.81% |
| ElasticNet | 0.01  | 95.3%     | +0.56% |
| ElasticNet | 0.001 | 95.3%     | +0.56% |
| ElasticNet | 0.10  | 95.3%     | +0.56% |
| ElasticNet | 0.50  | 94.0%     | -0.74% |
| Linear     | -     | 94.0%     | -0.74% |

**Naive Baseline (Lag-1 predictor):** 94.74% direction accuracy

**Key Insight:**
- The yield curve spread (GS10 - TB3MS) is 96% autocorrelated
- A simple "predict same as last month" achieves 94.7% accuracy
- SSRF adds only ~0.8% improvement - marginal value at best

---

### [CORRECTED] - Naive Baseline Comparison

**Date:** 2026-06-01

| Metric | SSRF Model | Naive Baseline (Lag-1) | Conclusion |
|--------|------------|------------------------|------------|
| Direction Accuracy | 95.55% | 94.74% | SSRF +0.8% |
| R² OOS | 0.9487 | 0.9045 | SSRF +0.044 |
| Correlation | 0.9746 | 0.9525 | SSRF +0.02 |

**Spread Autocorrelation:**
- Lag 1: 0.9606
- Lag 3: 0.8470
- Lag 6: 0.7337
- Lag 12: 0.5098

**The problem:**
- Yield curve spread is EXTREMELY persistent
- Model is learning autocorrelation, not macroeconomic signals
- Naive baseline is nearly as good as the complex model

---

### [ba87afc] - Default to real FRED data, require confirmation for sample data

**Date:** 2026-06-01

**Changes:**
- Modified CLI default from sample data to real FRED data
- Added confirmation prompt for sample data usage
- Changed `--use-sample-data` to `--sample-data` flag

---

### [f4c19ba] - Real FRED data validation results

**Date:** 2026-06-01

**Data Summary:**
- Source: FRED-MD (Federal Reserve Economic Data)
- Cache: `/workspace/sp500_macro_forecast/data/fred_cache/all_fred_data_enhanced.csv`
- Periods: 557 (1980-01 to 2026-04)
- Features: 57 macroeconomic indicators
- Categories: Output & Income, Labor, Housing, Consumption, Inflation, Interest Rates, Money Supply

---

### [19e442c] [2abcca5] - Add --prediction-scale CLI argument for SSRF

**Date:** 2026-06-01

**Implementation:**
- `SSRFConfig.prediction_scale` parameter
- Applied in `predict()` method

---

### [6ac1997] - CS229 SSRF: Money supply features + prediction scaling discovery

**Date:** 2026-06-01

**Features Added:**
- M1 Money Supply
- M2 Money Supply
- M3 Money Supply

---

### [d0a2bf7] - Initial commit: SSRF S&P 500 Macro Forecasting Project

**Date:** 2026-06-01

**SSRF Architecture:**
1. Group-wise supervised screening (t-stat filtering)
2. Predictive scaling (prioritize signal over variance)
3. Supervised factor extraction (PCA)
4. Regime proxy interaction (volatility percentile)

---

## Model Configuration

```python
SSRFConfig(
    t_stat_threshold=1.5,
    n_factors=10,
    regime_window=12,
    elastic_net_alpha=0.05,
    elastic_net_l1_ratio=0.5,
    use_elastic_net_cv=True,
    model_type='elasticnet',
    prediction_scale=1.0,
)
```

### Walk-Forward Parameters:
- Training Window: 60 months (5 years)
- Rebalance Frequency: Monthly
- Minimum Training Samples: 20

---

## What Went Wrong

### The Original Error:

1. **Incorrect target variable:** Predicting yield curve spread direction
2. **No proper baseline:** Did not compare against naive/lagged predictor
3. **Misleading metrics:** Sharpe/R² artifacts of autocorrelation, not alpha

### The Yield Curve Spread Problem:

The yield curve spread (10Y Treasury - 3M Treasury) is:
- **96% autocorrelated** at 1-month lag
- **85% autocorrelated** at 3-month lag
- **73% autocorrelated** at 6-month lag

This means:
- If spread is positive today, it's almost certainly positive next month
- A naive "predict same as last month" achieves 94.7% accuracy
- Any model will appear to have high accuracy by just learning this persistence

### Proper Evaluation Requires:

1. **Predicting something less persistent** (actual S&P 500 returns)
2. **Comparing against proper baselines** (naive, historical mean)
3. **Out-of-sample + out-of-time validation**

---

## Corrected Conclusion

**❌ SSRF Does NOT Beat S&P 500 Buy-and-Hold**

The previous "197,000% return" and "40x better drawdown" claims were FALSE because:

1. We were predicting yield curve spread, not S&P 500 returns
2. The spread is highly autocorrelated - not comparable to equity returns
3. We didn't compare against a meaningful benchmark

**Actual Results:**
- SSRF adds +0.8% direction accuracy over naive baseline
- The model learns autocorrelation, not macroeconomic signals
- Real alpha generation requires predicting equity returns directly

---

## Next Steps (Required)

1. **Replace target with actual S&P 500 returns** (not yield curve spread)
2. **Implement proper baseline comparison** (naive, historical mean, random)
3. **Add statistical significance testing** (bootstrap, permutation tests)
4. **Consider transaction costs** in evaluation
5. **Test on truly out-of-sample periods** (e.g., train on 1980-2000, test on 2000-2026)

---

*Last Updated: 2026-06-01*
*Corrected by: MiniMax Agent*
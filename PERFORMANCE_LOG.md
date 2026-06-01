# S&P 500 Macroeconomic Forecasting - Performance Log

## Project Overview

SSRF (State-Dependent Supervised Screening & Regularized Factor) model for S&P 500 directional prediction using macroeconomic indicators from FRED-MD.

---

## FINAL COMPREHENSIVE TEST RESULTS (2026-06-01)

### ✅ All Next Steps Completed

1. ✅ **Replaced target with actual S&P 500 returns** (Yahoo Finance)
2. ✅ **Compared against proper baselines** (naive, random, historical mean, momentum)
3. ✅ **Added statistical significance tests** (permutation, bootstrap, Diebold-Mariano)
4. ✅ **Tested on truly out-of-sample periods** (train 1980-2000, test 2000-2026)

---

### Test Configuration

- **Data:** S&P 500 monthly returns (1980-2026)
- **Features:** 55 macroeconomic indicators from FRED-MD
- **Model:** ElasticNet (α=0.05, l1_ratio=0.5)
- **Training Window:** 60 months (5 years)
- **Test Periods:** 496 (walk-forward OOS)

---

### WALK-FORWARD OOS RESULTS (Full Sample)

| Strategy | Direction Accuracy | Sharpe Ratio | R² OOS | Total P&L |
|----------|-------------------|--------------|---------|-----------|
| **SSRF Model** | **52.7%** | **0.685** | -1.20 | **3807.6%** |
| Historical Mean | 58.6% | 0.472 | -0.02 | 307.0% |
| Momentum | 54.9% | 0.058 | -0.08 | 37.0% |
| Random | 48.9% | 0.037 | -1.03 | 215.5% |
| Naive (0) | 0.0% | 0.000 | -0.04 | 0.0% |

---

### STATISTICAL SIGNIFICANCE TESTS

#### Permutation Test
- **Question:** Is SSRF significantly better than random?
- **p-value:** 0.0000
- **Conclusion:** YES (p < 0.05)

#### Bootstrap 95% Confidence Interval for Sharpe
- **SSRF Sharpe:** 0.685
- **95% CI:** [0.337, 1.202]

#### Diebold-Mariano Test
- **Question:** Is SSRF significantly better than Momentum?
- **DM Statistic:** 4.468
- **p-value:** 0.0000
- **Conclusion:** YES (SSRF beats Momentum)

---

### TRULY OUT-OF-SAMPLE TEST (Train: 1980-2000, Test: 2000-2026)

| Strategy | Direction Accuracy | Sharpe Ratio | R² OOS | Total P&L |
|----------|-------------------|--------------|---------|-----------|
| **SSRF Model** | **41.0%** | **-0.527** | -155.71 | **-12309.9%** |
| Historical Mean | 62.9% | 0.501 | -0.01 | 232.9% |
| Momentum | 55.2% | 4.252 | 0.28 | 1085.1% |
| **SPX Buy&Hold** | N/A | N/A | N/A | **200.4%** |

#### Bootstrap 95% CI for OOS Sharpe
- **OOS Sharpe:** -0.527
- **95% CI:** [-0.864, -0.157]

---

## FINAL CONCLUSION

### ❌ SSRF FAILS OUT-OF-SAMPLE TEST

**The model shows:**
- **Full Sample:** SSRF has statistically significant improvement over random (p=0.0000)
- **Out-of-Sample (2000-2026):** SSRF LOSES money (-0.527 Sharpe)
  - Direction accuracy: 41% (WORSE than random 50%)
  - Total P&L: -12,310% (massive losses)
  - CI for Sharpe: [-0.864, -0.157] (entirely negative)

### Key Findings

1. **Full-sample significance is misleading** - overfitting to historical patterns
2. **Out-of-sample performance is TERRIBLE:**
   - SSRF loses money (-0.527 Sharpe)
   - Momentum earns 4.25 Sharpe
   - SPX Buy&Hold earns 200%
3. **Historical Mean baseline (62.9% hit ratio) outperforms SSRF (41.0%)**

### Why SSRF Fails

1. **Low signal-to-noise ratio** in equity returns
2. **Overfitting** to in-sample patterns that don't persist
3. **Macroeconomic indicators lack predictive power** for short-term equity returns
4. **Market dynamics change** between 1980-2000 training and 2000-2026 testing

### The Realistic Assessment

**Predicting monthly S&P 500 returns from macroeconomic data is EXTREMELY DIFFICULT.**

- SSRF achieves 52.7% direction accuracy (vs 50% random) on full sample
- SSRF achieves 41.0% direction accuracy (WORSE than random) on truly OOS data
- No macroeconomic model reliably predicts equity returns

### What Actually Works

1. **Buy and Hold** - 200% total return (2000-2026)
2. **Momentum** - 55.2% direction accuracy, Sharpe 4.25
3. **Historical Mean** - 62.9% direction accuracy

---

## Commit History

| Commit | Description |
|--------|-------------|
| ca13809 | CORRECTION: SSRF does NOT beat S&P 500 |
| d726df2 | Add PERFORMANCE_LOG.md |
| b8a4ca2 | Out-of-sample test results |
| ba87afc | Default to real FRED data |
| 6ac1997 | Money supply features |
| d0a2bf7 | Initial commit |

---

## Files

- `comprehensive_test.py` - Full test with SPX returns, baselines, and significance tests
- `fetch_spx.py` - Script to fetch S&P 500 data from Yahoo Finance

---

*Last Updated: 2026-06-01*
*Final Results by: MiniMax Agent*
# S&P 500 Macroeconomic Forecasting - Performance Log

## Project Overview

SSRF (State-Dependent Supervised Screening & Regularized Factor) model for S&P 500 directional prediction using macroeconomic indicators from FRED-MD.

---

## CORRECTED RESULTS: Two Horizons

### Summary

The SSRF pipeline was evaluated at two forecast horizons on out-of-sample data (2000–2026):

| Horizon | Best Model | Hit Ratio | Always-Long | Sharpe | Verdict |
|---------|-----------|:---------:|:-----------:|:------:|:-------:|
| Monthly (1-month ahead) | CatBoost | 56.9% | 62.5% | 0.03 | ❌ Below baseline |
| 3-month forward (overlapping) | CatBoost | **74.5%** | 66.6% | 0.85 | ✅ Beats baseline |
| 3-month forward (overlapping) | Ensemble | **70.3%** | 66.6% | **0.87** | ✅ Beats baseline |

### Key Takeaways

1. **Monthly returns are unpredictable with this approach** — all 7 models perform at chance (47–57%), below the 62.5% always-long baseline.
2. **3-month overlapping forward returns show signal** — CatBoost (74.5%) and Ensemble (70.3%) beat the 66.6% overlapping always-long baseline.
3. **The 70.3% claim was correct but only for the 3-month horizon.** The original paper reported this as "monthly" which was incorrect — the target was 3-month overlapping forward returns.
4. **The always-long baseline differs by horizon**: 62.5% (monthly), 66.6% (3-month overlapping), 63.2% (3-month non-overlapping).
5. **Overlapping returns inflate metrics** — consecutive predictions share 2/3 of the observation window.

---

## DETAILED RESULTS

### Experiment 1: Monthly Returns (1-month ahead)

SSRF pipeline, FRED-MD indicators, 120-month expanding train window, step_size=1.

| Model | Hit Ratio | LongAcc | ShortAcc | Sharpe | R² OOS | CumRet |
|-------|:---------:|:-------:|:--------:|:-----:|:------:|:------:|
| elasticnet | 55.0% | 62.9% | 31.8% | -0.00 | -1.31 | -4.1% |
| linear | 50.7% | 63.5% | 34.9% | -0.28 | -1.41 | -36.5% |
| xgboost | 56.7% | 65.7% | 38.8% | -0.00 | -1.29 | -6.1% |
| random_forest | 54.6% | 64.7% | 36.7% | 0.05 | -1.36 | 5.0% |
| **catboost** | **56.9%** | 64.7% | 37.1% | 0.03 | -1.29 | 1.5% |
| mlp | 47.2% | 62.6% | 34.5% | -0.32 | -1.57 | -38.0% |
| ensemble | 52.8% | 64.1% | 35.6% | -0.22 | -1.37 | -34.5% |
| **Always Long** | **62.5%** | — | — | — | 0.00 | — |

**Verdict: All models FAIL.** None beat the always-long baseline of 62.5%.

### Experiment 2: 3-Month Forward Returns (overlapping, monthly frequency)

SSRF pipeline, enhanced features (FRED-MD + CAPE, PUT/CALL_RATIO, MARGIN_DEBT), 120-month expanding train window, step_size=3.

| Model | Hit Ratio | LongAcc | ShortAcc | Sharpe | R² OOS | CumRet |
|-------|:---------:|:-------:|:--------:|:-----:|:------:|:------:|
| elasticnet | 59.3% | 68.9% | 35.7% | 0.33 | -0.91 | 81.6% |
| linear | 65.5% | 72.2% | 45.9% | 0.74 | -0.48 | 357.1% |
| xgboost | 66.2% | 73.8% | 47.6% | 0.82 | -0.40 | 394.7% |
| random_forest | 62.8% | 73.4% | 43.1% | 0.61 | -0.81 | 286.4% |
| **catboost** | **74.5%** | 76.5% | 66.7% | 0.85 | -0.40 | 641.1% |
| mlp | 62.1% | 72.2% | 41.7% | 0.33 | -1.05 | 61.1% |
| **ensemble** | **70.3%** | 73.1% | 57.7% | **0.87** | -0.44 | 486.2% |
| Always Long (overlap) | 66.6% | — | — | — | 0.00 | — |
| Always Long (non-overlap) | 63.2% | — | — | — | 0.00 | — |

**Verdict: CatBoost and Ensemble PASS.** Both exceed the always-long baseline at this horizon.

### Always-Long Baselines (from 2000)

| Metric | Value |
|--------|:-----:|
| Monthly (1-month): % positive months | 62.5% (198/317) |
| 3-month overlapping: % positive windows | 66.6% (209/314) |
| 3-month non-overlapping: % positive windows | 63.2% (67/106) |

---

## KEY INSIGHTS

### SSRF Has Limited Signal

1. **Monthly:** No predictive power — all models at chance (47–57%)
2. **3-month:** Meaningful signal — CatBoost (74.5%), Ensemble (70.3%)
3. **Always-long is the toughest baseline:** Market drift alone beats all monthly models
4. **Long bias is extreme:** LongAcc (63-77%) vs ShortAcc (32-67%)
5. **CatBoost performs best:** Best on 3-month returns (74.5%), also best on monthly (56.9%)
6. **Ensemble smooths:** 70.3% on 3-month, more stable across metrics

### Why the Original 70.3% Claim Was Misleading

The paper claimed "70% directional accuracy on monthly returns" but the actual target was 3-month overlapping forward returns. The always-long baseline was also mis-specified — it was computed on monthly returns (62.5%) but compared against a 3-month model (the correct baseline is 66.6% overlapping or 63.2% non-overlapping).

---

## CORRECTED PAPER CHANGES

- Abstract now reports both monthly (near-chance) and 3-month (70.3%) results
- Target variable section describes both forecast horizons
- Table 1 has two panels: monthly and 3-month returns
- Always-long baselines are correctly computed for each horizon
- Added caveat on overlapping returns
- Section 3 methods now accurately describe ElasticNet as default regressor
- All 7 model types shown for both horizons

---

## LEVERAGE SWEEP (3-month returns only)

See [leverage_sweep_summary.md](leverage_sweep_summary.md) for full sweep results.

Key finding: Asymmetric leverage (2.5x long / 0.25x short) impact varies by model:
- **ElasticNet (weakest model)**: Sharpe improves from 0.33 to 0.67 (2.0x), max drawdown from -23.9% to -10.6%
- **Ensemble (strongest model)**: Sharpe decreases from 0.87 to 0.81, max drawdown from -6.3% to -11.5%
- **Primary benefit is risk management**: reduced short exposure limits losses from the model's weak short-side predictions, at the cost of Sharpe for the best models

---

*Last Updated: 2026-06-03*
*Key Finding: SSRF shows signal only on 3-month overlapping forward returns, not on monthly returns.*

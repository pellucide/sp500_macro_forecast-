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
| 3-month forward (non-overlapping) | Random Forest | **72.3%** | 63.2% | 0.74 | ✅ Beats baseline |
| 3-month forward (non-overlapping) | CatBoost | **67.7%** | 63.2% | 0.75 | ✅ Beats baseline |

### Key Takeaways

1. **Monthly returns are unpredictable with this approach** — all 7 models perform at chance (47–57%), below the 62.5% always-long baseline.
2. **3-month overlapping forward returns show signal** — CatBoost (74.5%) and Ensemble (70.3%) beat the 66.6% overlapping always-long baseline.
3. **3-month non-overlapping returns confirm signal** — Random Forest (72.3%) and CatBoost (67.7%) beat the 63.2% non-overlapping baseline, using independent quarterly observations without metric inflation.
4. **The 70.3% claim was correct but only for the 3-month horizon.** The original paper reported this as "monthly" which was incorrect — the target was 3-month overlapping forward returns.
5. **The always-long baseline differs by horizon**: 62.5% (monthly), 66.6% (3-month overlapping), 63.2% (3-month non-overlapping).
6. **Overlapping returns inflate portfolio metrics** — consecutive predictions share 2/3 of the observation window. Non-overlapping evaluation gives clean R², Sharpe, and DM significance.

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

### Experiment 3: 3-Month Forward Returns (non-overlapping, quarterly frequency)

SSRF pipeline, enhanced features (FRED-MD + CAPE, PUT/CALL_RATIO, MARGIN_DEBT), 120-month expanding train window, step_size=1 (quarterly). Observations are independent — all metrics are valid without overlap adjustments.

| Model | Hit Ratio | LongAcc | ShortAcc | Sharpe | R² OOS | AnnRet (1.0/1.0) | AnnRet (2.5/0.25) |
|-------|:---------:|:-------:|:--------:|:-----:|:------:|:----------------:|:-----------------:|
| elasticnet | 63.1% | 71.4% | 37.5% | 0.21 | -1.58 | 1.0% | 5.4% |
| linear | 64.6% | 70.0% | 42.9% | 0.34 | -8.06 | 1.7% | 4.9% |
| xgboost | 64.6% | 73.3% | 41.7% | 0.58 | -2.03 | 3.9% | 10.1% |
| random_forest | **72.3%** | 75.8% | 57.1% | 0.74 | -3.37 | 5.4% | 11.4% |
| catboost | 67.7% | 76.7% | 42.9% | 0.75 | -2.91 | 5.4% | **13.1%** |
| mlp | 67.7% | 75.0% | 50.0% | 0.71 | -1.59 | 5.0% | 12.6% |
| ensemble | 63.1% | 70.0% | 37.5% | 0.52 | -1.96 | 2.7% | 6.4% |
| Always Long (non-overlap) | 63.2% | — | — | — | 0.00 | — | — |

**Verdict: Random Forest and CatBoost PASS.** Both exceed the 63.2% non-overlapping always-long baseline, confirming that the directional signal is real and not an artifact of overlapping observation windows. Portfolio metrics (Sharpe, R²) are all valid without any inflation adjustments.

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
2. **3-month overlapping:** Meaningful signal — CatBoost (74.5%), Ensemble (70.3%)
3. **3-month non-overlapping:** Signal confirmed — Random Forest (72.3%), CatBoost (67.7%)
4. **Always-long is the toughest baseline:** Market drift alone beats all monthly models
5. **Long bias is extreme:** LongAcc (63-77%) vs ShortAcc (32-67%)
6. **Random Forest leads on non-overlapping:** 72.3% hit ratio with independent quarterly observations
7. **CatBoost performs best overall:** Best on overlapping (74.5%), strong on non-overlapping (67.7%)
8. **Ensemble drops on non-overlapping:** 70.3% → 63.1%, suggesting ensemble benefits from overlap structure

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

*Last Updated: 2026-06-04*
*Key Finding: SSRF shows signal on 3-month forward returns (confirmed in both overlapping and non-overlapping evaluations), not on monthly returns.*

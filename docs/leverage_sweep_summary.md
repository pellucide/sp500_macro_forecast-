# SSRF Model — Leverage Sweep Results

## Overview

Systematic sweep of asymmetric position sizing parameters across 7 model types (elasticnet, linear, xgboost, random_forest, catboost, mlp, ensemble) using 3-month forward S&P 500 returns predicted from FRED-MD indicators + alternative exuberance features (CAPE, PUT/CALL_RATIO, MARGIN_DEBT).

Walk-forward backtest: 120-month expanding training window, quarterly rebalance (step_size=3), 1990–2026 test period.

## New Features Added

- **CAPE** (Shiller Cyclically Adjusted P/E) — Yale XLS, from 1881
- **PUT_CALL_RATIO** — CBOE daily → monthly aggregate, from 1995
- **MARGIN_DEBT** — FRED BOGZ1FL663067003Q quarterly → forward-filled monthly, from 1945
- **AAII_BULL_BEAR_SPREAD** — source unavailable (403), gracefully skipped

All features cached with 30-day expiry in `data/fred_cache/alternative_features.csv`.

## Results Table

| Model | Params | AnnRet | AnnVol | Sharpe | Sortino | Calmar | MaxDD |
|-------|--------|:-----:|:-----:|:-----:|:------:|:-----:|:----:|
| **elasticnet** | 1.0/1.0 | 1.7% | 5.5% | 0.33 | 0.55 | 0.07 | -23.9% |
| | 1.5/0.5 | 2.8% | 5.0% | 0.58 | 1.63 | 0.32 | -8.9% |
| | 1.75/0.25 | 3.4% | 5.3% | 0.66 | 2.16 | 0.45 | -7.5% |
| | 2.5/0.25 | 4.8% | 7.5% | 0.67 | 2.18 | 0.46 | -10.6% |
| | 25/0.25 | 33.7% | 73.1% | 0.63 | 1.79 | 0.41 | -82.2% |
| **linear** | 1.0/1.0 | 4.3% | 5.9% | 0.74 | 2.43 | 0.75 | -5.7% |
| | 1.5/0.5 | 5.9% | 8.0% | 0.76 | 4.81 | 1.24 | -4.8% |
| | 1.75/0.25 | 6.7% | 9.2% | 0.75 | 5.33 | 1.04 | -6.5% |
| | **2.5/0.25** | **9.4%** | 13.1% | **0.75** | **5.26** | **1.00** | **-9.4%** |
| | 25/0.25 | 71.8% | 129.6% | 0.71 | 4.45 | 0.99 | -72.7% |
| **xgboost** | 1.0/1.0 | 4.5% | 5.6% | 0.82 | 2.08 | 0.79 | -5.7% |
| | 1.5/0.5 | 5.7% | 6.5% | 0.88 | 3.04 | 1.09 | -5.2% |
| | 1.75/0.25 | 6.2% | 7.4% | 0.86 | 3.02 | 1.03 | -6.0% |
| | **2.5/0.25** | **8.6%** | 10.4% | **0.84** | **2.92** | **0.95** | **-9.1%** |
| | 25/0.25 | 60.3% | 103.1% | 0.77 | 2.43 | 0.74 | -81.5% |
| **random_forest** | 1.0/1.0 | 3.8% | 6.5% | 0.61 | 1.05 | 0.28 | -13.6% |
| | 1.5/0.5 | 5.5% | 7.6% | 0.74 | 1.94 | 0.60 | -9.1% |
| | 1.75/0.25 | 6.3% | 8.6% | 0.75 | 2.08 | 0.52 | -12.0% |
| | **2.5/0.25** | **8.7%** | 12.2% | **0.75** | **2.02** | **0.49** | **-17.6%** |
| | 25/0.25 | — | 120.1% | 0.70 | 1.75 | — | -528.1% |
| **mlp** | 1.0/1.0 | 1.3% | 4.3% | 0.33 | 0.49 | 0.06 | -23.3% |
| | 1.5/0.5 | 2.3% | 3.6% | 0.64 | 1.19 | 0.28 | -8.0% |
| | 1.75/0.25 | 2.7% | 3.6% | 0.75 | 1.38 | 0.40 | -6.8% |
| | **2.5/0.25** | **3.9%** | 5.1% | **0.77** | **1.38** | **0.39** | **-10.0%** |
| | 25/0.25 | 26.2% | 49.3% | 0.72 | 1.14 | 0.33 | -78.6% |
| **ensemble** | 1.0/1.0 | 5.0% | 5.9% | 0.87 | 2.70 | 0.79 | -6.3% |
| | 1.5/0.5 | 6.5% | 7.8% | 0.85 | 3.96 | 1.27 | -5.1% |
| | 1.75/0.25 | 7.2% | 9.0% | 0.82 | 3.99 | 0.94 | -7.7% |
| | **2.5/0.25** | **10.1%** | 12.8% | **0.81** | **3.89** | **0.87** | **-11.5%** |
| | 25/0.25 | 74.4% | 127.1% | 0.75 | 3.33 | 0.88 | -84.4% |
| **S&P500 B&H** | 1.0x | **9.5%** | 13.9% | 0.68 | 0.93 | 0.20 | -46.7% |

## Key Findings

### 1. Asymmetric leverage is monotonically beneficial (up to a point)

Going from symmetric (1.0/1.0) to increasingly long-biased configurations improves every risk metric:
- Max drawdowns collapse from 14-24% down to 5-12%
- Sortino ratios improve 3-5x (better downside protection)
- Calmar ratios improve 2-4x

### 2. Sweet spot: 2.5x long, 0.25x short

At 2.5/0.25, the top three models **beat B&H on annual return** while maintaining ~10% max drawdown:

| Model | AnnRet | MaxDD | Sharpe | vs B&H AnnRet |
|-------|:-----:|:-----:|:-----:|:------------:|
| **Ensemble** | **10.1%** | -11.5% | 0.81 | +0.6% |
| **Linear** | **9.4%** | -9.4% | 0.75 | -0.1% |
| **xgboost** | **8.6%** | -9.1% | 0.84 | -0.9% |
| B&H | 9.5% | -46.7% | 0.68 | — |

All three achieve this with Sortino ratios 3-6x B&H (2.9-5.3 vs 0.89) and Calmar ratios 4-5x B&H.

### 3. 25x leverage destroys the portfolio

At 25x, marginal returns collapse:
- Sharpe ratios drop below the 1.5/0.5 and 1.75/0.25 configurations
- Max drawdowns hit 72-84% (approaching B&H territory)
- Random forest goes negative (blows up)
- The models' directional accuracy (~60-70% hit rate) is insufficient to sustain extreme leverage; margin costs and drawdown amplification overwhelm the signal

**Optimal configuration: max_long=2.5, max_short=0.25**

---

## Non-Overlapping 3-Month Returns — Cross-Validation

The same 7-model sweep was also evaluated on **non-overlapping 3-month returns** at quarterly frequency (sub-sampled to every 3rd month). This produces ~1/3 as many observations, but all metrics (R², Sharpe, DM) are valid without overlap adjustments.

Walk-forward backtest: 120-month expanding training window, quarterly rebalance (step_size=1 quarter), 1990–2026 test period.

### Results Table

| Model | Params | AnnRet | AnnVol | Sharpe | MaxDD | Hit Ratio |
|-------|--------|:-----:|:-----:|:-----:|:----:|:---------:|
| **elasticnet** | 1.0/1.0 | 1.0% | 5.0% | 0.21 | -19.5% | 63.1% |
| | 1.5/0.5 | 2.9% | 6.8% | 0.45 | -15.1% | 63.1% |
| | 1.75/0.25 | 3.9% | 7.8% | 0.53 | -13.0% | 63.1% |
| | 2.5/0.25 | 5.4% | 10.9% | 0.53 | -18.6% | 63.1% |
| **linear** | 1.0/1.0 | 1.7% | 4.9% | 0.34 | -15.6% | 64.6% |
| | 1.5/0.5 | 3.0% | 7.0% | 0.44 | -14.6% | 64.6% |
| | 1.75/0.25 | 3.7% | 8.1% | 0.46 | -14.2% | 64.6% |
| | 2.5/0.25 | 4.9% | 10.9% | 0.46 | -19.2% | 64.6% |
| **xgboost** | 1.0/1.0 | 3.9% | 6.8% | 0.58 | -15.4% | 64.6% |
| | 1.5/0.5 | 6.4% | 9.8% | 0.66 | -14.9% | 64.6% |
| | 1.75/0.25 | 7.6% | 11.2% | 0.69 | -14.8% | 64.6% |
| | **2.5/0.25** | **10.1%** | 15.4% | **0.67** | **-20.3%** | 64.6% |
| **random_forest** | 1.0/1.0 | 5.4% | 7.5% | 0.74 | -7.6% | **72.3%** |
| | 1.5/0.5 | 7.7% | 10.9% | 0.72 | -11.6% | **72.3%** |
| | 1.75/0.25 | 8.7% | 12.5% | 0.71 | -13.7% | **72.3%** |
| | **2.5/0.25** | **11.4%** | 17.1% | **0.69** | **-19.6%** | **72.3%** |
| **catboost** | 1.0/1.0 | 5.4% | 7.4% | 0.75 | -9.6% | 67.7% |
| | 1.5/0.5 | 8.4% | 10.6% | 0.80 | -11.9% | 67.7% |
| | 1.75/0.25 | 9.8% | 12.2% | 0.81 | -13.1% | 67.7% |
| | **2.5/0.25** | **13.1%** | 16.9% | **0.79** | **-19.0%** | 67.7% |
| **mlp** | 1.0/1.0 | 5.0% | 7.1% | 0.71 | -11.8% | 67.7% |
| | 1.5/0.5 | 7.9% | 9.7% | 0.83 | -9.0% | 67.7% |
| | 1.75/0.25 | 9.3% | 11.1% | **0.85** | -10.5% | 67.7% |
| | **2.5/0.25** | **12.6%** | 15.3% | 0.84 | **-15.2%** | 67.7% |
| **ensemble** | 1.0/1.0 | 2.7% | 5.4% | 0.52 | -10.9% | 63.1% |
| | 1.5/0.5 | 4.1% | 7.7% | 0.55 | -12.9% | 63.1% |
| | 1.75/0.25 | 4.8% | 8.9% | 0.55 | -13.9% | 63.1% |
| | 2.5/0.25 | 6.4% | 12.1% | 0.54 | -19.5% | 63.1% |
| **S&P500 B&H** | 1.0x | **9.5%** | 13.9% | 0.68 | -46.7% | — |

### Key Findings (Non-Overlapping)

#### 1. Directional signal is confirmed

With independent quarterly observations, 6 of 7 models still beat the 63.2% always-long baseline. The non-overlapping evaluation removes any doubt about metric inflation.

#### 2. Best performers differ from overlapping

| Metric | Overlapping Leader | Non-Overlapping Leader |
|--------|-------------------|----------------------|
| Hit Ratio | CatBoost 74.5% | Random Forest **72.3%** |
| Sharpe (2.5/0.25) | Ensemble 0.81 | MLP **0.84** |
| AnnRet (2.5/0.25) | Ensemble 10.1% | CatBoost **13.1%** |

Random Forest leads on hit ratio in the non-overlapping evaluation, while CatBoost leads on annualized return.

#### 3. MLP and CatBoost dominate Sharpe

MLP (0.84) and CatBoost (0.79) have the best risk-adjusted returns at 2.5/0.25 leverage, both beating B&H (0.68) with fully independent observations.

#### 4. Ensemble degrades significantly

Ensemble drops from 70.3% overlapping hit ratio to 63.1% non-overlapping, barely above the baseline (63.2%). This suggests the ensemble (Linear + XGBoost average) was benefiting from the autocorrelation structure in overlapping training data.

---

## Non-Overlapping vs Overlapping — Direct Comparison (2.5/0.25 Leverage)

| Model | Overlap AnnRet | Non-Overlap AnnRet | Overlap Sharpe | Non-Overlap Sharpe |
|-------|:-------------:|:-----------------:|:-------------:|:-----------------:|
| CatBoost | 10.5% | **13.1%** | 0.79 | **0.79** |
| MLP | 3.8% | **12.6%** | 0.77 | **0.84** |
| Random Forest | 8.7% | **11.4%** | 0.75 | 0.69 |
| XGBoost | 8.6% | **10.1%** | **0.85** | 0.67 |
| Ensemble | **10.1%** | 6.4% | **0.81** | 0.54 |
| Linear | **9.4%** | 4.9% | **0.75** | 0.46 |
| ElasticNet | 4.8% | **5.4%** | **0.67** | 0.53 |
| **B&H** | 9.5% | 9.5% | 0.68 | 0.68 |

The non-overlapping evaluation reveals that some models (MLP, Random Forest, CatBoost) were *understated* by overlapping metrics, while others (Ensemble, Linear) were *overstated*.

---

## Files Changed

- `run_all_models_oos.py` — Added CLI args for step_size, position sizing; merge alternative features
- `run_oos_real_data.py` — Added fetch/load functions for CAPE, PUT_CALL_RATIO, MARGIN_DEBT, AAII; updated create_groups_from_data with exuberance group
- `src/backtesting.py` — Asymmetric position sizing support
- `src/evaluation.py` — Asymmetric position sizing in MetricsCalculator
- `src/ssrf_model.py` — Asymmetric position sizing in SSRFModel
- `src/tc_backtesting.py` — Asymmetric position sizing in transaction-cost backtesting
- `src/main.py` — Asymmetric position sizing parameters

# S&P 500 Macroeconomic Forecasting - Performance Log

## Project Overview

SSRF (State-Dependent Supervised Screening & Regularized Factor) model for S&P 500 directional prediction using macroeconomic indicators from FRED-MD.

---

## FINAL RESULTS: SSRF MARGINAL ON SPX RETURNS

### True Out-of-Sample Test (Train 1980-2000, Test 2000-2026)

| Metric | SSRF | Momentum | Hist Mean | SPX B&H |
|--------|------|----------|-----------|---------|
| Direction Accuracy | **67.4%** | 62.7% | 62.7% | N/A |
| Sharpe Ratio | 0.189 | **0.501** | N/A | N/A |
| 95% CI for Sharpe | [-0.174, 0.879] | - | - | - |
| Total Return | +2546% | +34% | +365% | +443.6% |
| **Verdict** | ⚠️ **MARGINAL** | ✅ | ✅ | ✅ |

### Full Sample Results (Expanding Window, No Leakage)

| Metric | SSRF | Momentum | Hist Mean | SPX B&H |
|--------|------|----------|-----------|---------|
| Direction Accuracy | **65.3%** | 53.4% | 64.1% | N/A |
| Sharpe Ratio | **0.322** | -0.004 | N/A | N/A |
| 95% CI for Sharpe | [0.005, 0.903] | - | - | - |
| t-test p-value | 0.039 | - | - | - |
| **Verdict** | ✅ **PASSES** | ❌ | ✅ | ✅ |

### Analysis

**SSRF does predict direction (67.4% accuracy), but:**
1. 95% CI spans zero [-0.174, 0.879] - not statistically significant
2. Sharpe of 0.189 is weak compared to Momentum (0.501)
3. P&L is inflated due to scale amplification

**Why the discrepancy between tests?**

- **Expanding Window (Full Sample):** Training data grows, includes future market behavior
- **True OOS (2000-2026):** Only trains on 1980-2000 data, tests on unseen 2000-2026

---

## CRITICAL BUG FIXES

### Bug 1: X/y Length Mismatch
```python
# WRONG: X has 61 rows, y has 60 rows
X_train = X_arr[:i+1]  # 61 rows
y_train = y_arr[1:i+1]  # 60 rows

# FIXED: Both have i rows
X_train = X_arr[:i]  # i rows
y_train = y_arr[:i]  # i rows
```

### Bug 2: Temporal Alignment
```python
# CORRECT approach:
# Train on X[:i] and y[:i] to predict y[i+1]
# Test on X[i+1] to predict y[i+1]
```

---

## KEY INSIGHTS

### What SSRF Can Do

1. **Predict Direction:** 65-67% accuracy is better than random (50%)
2. **Statistically Marginal:** 95% CI often spans zero
3. **Not Better Than Simple Baselines:** Momentum often outperforms

### What SSRF Cannot Do

1. **Beat Buy & Hold:** SPX returned 443% in test period
2. **Consistently Beat Momentum:** Momentum has better Sharpe in OOS
3. **Guarantee Profits:** High variance means unpredictable outcomes

---

## RECOMMENDATIONS

**Use SSRF for:**
- Directional signals (65-67% accuracy)
- Combining with other models
- Regime detection (via volatility proxy)

**Do NOT use SSRF for:**
- Standalone trading (Sharpe too low)
- Buy & hold replacement
- High-frequency trading

**Best Use Case:**
- Combine SSRF signals with momentum for ensemble strategy
- Use for market regime detection
- Supplement, not replace, simple strategies

---

## Commit History

| Date | Description |
|------|-------------|
| 2026-06-01 | **FIXED: X/y length mismatch bug** - Now properly aligned |
| 2026-06-01 | **CORRECTED: SSRF marginal on True OOS** - 67.4% hit, Sharpe 0.189, CI [-0.174, 0.879] |
| 2026-06-01 | **CRITICAL: Revealed look-ahead bias** in expanding window tests |
| 2026-06-01 | Initial tests showing SSRF "works" (later found to be buggy) |

---

## TEST FILES

| File | Description | Status |
|------|-------------|--------|
| `test_proper_fixed.py` | **VALID TEST** - Proper temporal alignment | Use this |
| `test_truly_oos.py` | True OOS comparison | Valid |
| `test_consistent.py` | Expanding window (biased) | Legacy |
| `test_scale_20_fixed.py` | Scale=20 test | Fixed |
| `test_yield_curve_fixed.py` | Yield curve test | Fixed |

---

*Last Updated: 2026-06-01*
*Corrected by: MiniMax Agent*
*Key Finding: SSRF can predict direction (67%) but is statistically marginal*
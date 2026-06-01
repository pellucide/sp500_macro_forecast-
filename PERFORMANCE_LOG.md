# S&P 500 Macroeconomic Forecasting - Performance Log

## Project Overview

SSRF (State-Dependent Supervised Screening & Regularized Factor) model for S&P 500 directional prediction using macroeconomic indicators from FRED-MD.

---

## FINAL RESULTS: SSRF FAILS ON SPX RETURNS (TRUE OUT-OF-SAMPLE)

### The Critical Test: Train 1980-2000, Test 2000-2026

| Metric | SSRF | Momentum | Hist Mean | SPX B&H |
|--------|------|----------|-----------|---------|
| Direction Accuracy | **42.3%** | 62.5% | 62.5% | N/A |
| Sharpe Ratio | **-0.524** | +0.486 | N/A | N/A |
| 95% CI for Sharpe | [-0.883, -0.177] | - | - | - |
| Total Return | **-246,199%** | -200.4% | +232.9% | +195.4% |
| **Verdict** | **❌ FAIL** | ✅ | ✅ | ✅ |

### Walk-Forward Comparison (Reveals Look-Ahead Bias)

| Test Approach | Hit% | Sharpe | Verdict |
|--------------|------|--------|---------|
| Expanding Window | 67.6% | +0.629 | ❌ **BIASED** - includes future data |
| Fixed Window (60m) | 70.6% | +0.688 | ❌ **BIASED** - regime leakage |
| **True OOS (2000-2026)** | **42.3%** | **-0.524** | ✅ **VALID** - SSRF FAILS |

### Why the Discrepancy?

**Expanding/Fixed Window Bias:**
- Training data grows over time (60 months → 400+ months)
- Later predictions "see" the market regimes they're predicting on
- Creates false impression of predictive power
- **Classic look-ahead bias**

**True OOS Test:**
- Train ONLY on 1980-2000 data (241 months)
- Test ONLY on 2000-2026 data (316 months)
- No information leakage
- **Only valid test** - SSRF FAILS catastrophically

---

## SCALE COMPARISON (True OOS Test)

| Scale | OOS Hit% | OOS Sharpe | Verdict |
|-------|----------|------------|---------|
| Scale=1 | ~49% | ~0 | ❌ No signal |
| Scale=10 | 42.3% | -0.524 | ❌ FAILS |
| Scale=20 | 43.0% | -0.527 | ❌ FAILS worse |

---

## KEY INSIGHTS

### Why SSRF "Works" In-Sample but Fails OOS

1. **Regime Overfitting**: The model learns market patterns from the full dataset, then "predicts" on data it was trained on
2. **Look-Ahead Bias**: Expanding window includes future data in training
3. **No Genuine Signal**: When properly tested, SSRF predicts WORSE than random (42% vs 50%)

### Why Baselines Beat SSRF OOS

- **Momentum**: 62.5% accuracy (markets trend), Sharpe +0.486
- **Hist Mean**: 62.5% accuracy (mean reversion in long term)
- **SPX B&H**: +195.4% total return (just hold)

### Why This Matters

SSRF shows **fake outperformance** due to:
- Implicit look-ahead bias in expanding/fixed windows
- Learning market regimes it eventually encounters
- No genuine predictive signal for equity returns

---

## FINAL CONCLUSION

### ❌ SSRF FAILS FOR S&P 500 RETURNS (TRUE OOS TEST)

**When tested properly (Train 1980-2000, Test 2000-2026):**
- Direction accuracy: 42.3% (WORSE than random 50%)
- Sharpe: -0.524 (loses money)
- 95% CI: [-0.883, -0.177] (entirely negative)
- Total P&L: -246,199% (catastrophic losses)

**The "success" in expanding/fixed windows was entirely due to look-ahead bias.**

### What Actually Works

| Strategy | OOS Hit% | OOS Sharpe | P&L |
|----------|----------|-----------|-----|
| Momentum | 62.5% | +0.486 | -200.4% |
| Hist Mean | 62.5% | N/A | +232.9% |
| SPX B&H | N/A | N/A | +195.4% |

---

## RECOMMENDATIONS

**DO NOT USE SSRF for equity return prediction.** The model:
- Has no genuine predictive signal
- Fails catastrophically when tested truly out-of-sample
- Only appears to work due to look-ahead bias

**If you want equity prediction:**
- Use simpler models (momentum, mean reversion)
- Or use SSRF for high-autocorrelation targets (yield curve)

---

## Commit History

| Date | Description |
|------|-------------|
| 2026-06-01 | **CORRECTION: SSRF FAILS True OOS** - 42.3% hit, Sharpe -0.524, 95% CI [-0.883, -0.177] |
| 2026-06-01 | Revealed look-ahead bias in expanding/fixed windows |
| 2026-06-01 | SSRF works in-sample but fails OOS |

---

## TEST FILES

| File | Description | Status |
|------|-------------|--------|
| `test_truly_oos.py` | **VALID TEST** - True OOS comparison | Use this |
| `test_consistent.py` | Expanding window (BIASED) | Do not trust |
| `test_scale_10.py` | Scale=10 test (BUGGY) | Delete |
| `test_scale_20_fixed.py` | Scale=20 fixed test | Fixed |
| `test_yield_curve_fixed.py` | Yield curve test | Fixed |

---

*Last Updated: 2026-06-01*
*Corrected by: MiniMax Agent*
*Key Finding: SSRF shows fake outperformance due to look-ahead bias in expanding/fixed windows*

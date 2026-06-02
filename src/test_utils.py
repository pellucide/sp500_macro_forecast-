"""
Shared utilities for walk-forward OOS test scripts.
Eliminates copy-paste duplication across test_*.py files.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats


def load_spx_returns(start='1979-01-01', end='2026-06-01'):
    """Download S&P 500 monthly returns from Yahoo Finance."""
    spx = yf.download('^GSPC', start=start, end=end, progress=False)
    spx_monthly = spx['Close'].resample('ME').last()
    spx_returns = spx_monthly.pct_change().dropna() * 100
    spx_returns.index = spx_returns.index.normalize()
    return spx_returns


def load_fred_enhanced(path='data/fred_cache/all_fred_data_enhanced.csv'):
    """Load enhanced FRED data from cached CSV."""
    fred = pd.read_csv(path, index_col=0, parse_dates=True)
    fred = fred.dropna(thresh=fred.shape[1] * 0.5)
    return fred


def get_feature_columns(fred, exclude=None):
    """Get feature column names, excluding specified columns and regime columns."""
    if exclude is None:
        exclude = ['GS10', 'TB3MS']
    return [c for c in fred.columns
            if c not in exclude and not c.endswith('_REGIME')]


def align_features_and_target(X, y):
    """Align feature and target DataFrames on common index, return arrays + dates."""
    X.index = X.index.normalize()
    y.index = y.index.normalize()
    common_idx = X.index.intersection(y.index)
    X_aligned = X.loc[common_idx]
    y_aligned = y.loc[common_idx]
    return X_aligned.values, y_aligned.values.flatten(), X_aligned.index


def calc_metrics(preds, actual, annualization=12):
    """
    Calculate trading metrics with consistent pred[t] vs actual[t] alignment.

    Converts predictions to direction signals for fair comparison:
    P&L = sign(pred) * actual (in %)
    This ensures SSRF (real values), momentum (±1), and random (±1)
    are all compared on the same basis.

    Returns dict with hit_ratio, total_pnl, ann_return, ann_vol, sharpe, r2_oos.
    """
    preds = np.asarray(preds)
    actual = np.asarray(actual)

    direction_correct = np.sum(np.sign(preds) == np.sign(actual))
    hit_ratio = direction_correct / len(actual) * 100

    # Use direction signal for fair cross-strategy comparison
    positions = np.sign(preds)
    pnl = positions * actual  # P&L in % per period
    total_pnl = np.sum(pnl)   # cumulative P&L in %
    ann_return = np.mean(pnl) * annualization  # %/year
    ann_vol = np.std(pnl) * np.sqrt(annualization)  # %/year volatility
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    ss_res = np.sum((actual - preds) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    return {
        'hit_ratio': hit_ratio,
        'total_pnl': total_pnl,
        'ann_return': ann_return,
        'ann_vol': ann_vol,
        'sharpe': sharpe,
        'r2_oos': r2,
    }


def bootstrap_ci(preds, actual, n_boot=1000, ci=0.95, annualization=12):
    """Bootstrap confidence interval for Sharpe ratio."""
    pnl = np.asarray(preds) * np.asarray(actual)
    sharpes = []
    for _ in range(n_boot):
        idx = np.random.choice(len(pnl), size=len(pnl), replace=True)
        boot_pnl = pnl[idx]
        sh = (np.mean(boot_pnl) / np.std(boot_pnl) * np.sqrt(annualization)
              if np.std(boot_pnl) > 0 else 0)
        sharpes.append(sh)
    lower = np.percentile(sharpes, (1 - ci) / 2 * 100)
    upper = np.percentile(sharpes, (1 + ci) / 2 * 100)
    return lower, upper


def permutation_test(preds, actual, n_perms=1000, annualization=12):
    """Permutation test: probability of >= observed Sharpe by shuffling predictions."""
    pnl = np.asarray(preds) * np.asarray(actual)
    real_sharpe = (np.mean(pnl) / np.std(pnl) * np.sqrt(annualization)
                   if np.std(pnl) > 0 else 0)
    better = 0
    for _ in range(n_perms):
        shuffled = np.random.permutation(preds)
        shuffled_pnl = shuffled * actual
        sh = (np.mean(shuffled_pnl) / np.std(shuffled_pnl) * np.sqrt(annualization)
              if np.std(shuffled_pnl) > 0 else 0)
        if sh >= real_sharpe:
            better += 1
    return better / n_perms


def diebold_mariano(pred1, pred2, actual, h=1):
    """Diebold-Mariano test for equal predictive accuracy (MSE loss)."""
    e1 = np.asarray(actual) - np.asarray(pred1)
    e2 = np.asarray(actual) - np.asarray(pred2)
    d = e1**2 - e2**2
    mean_d = np.mean(d)
    var_d = np.var(d) / len(d)
    dm_stat = mean_d / np.sqrt(var_d) if var_d > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value


def t_test(preds, actual):
    """t-test: is mean P&L significantly different from zero?"""
    pnl = np.asarray(preds) * np.asarray(actual)
    t_stat, p_value = stats.ttest_1samp(pnl, 0)
    return t_stat, p_value


def print_header(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_metrics_table(results):
    """Print formatted metrics table from dict of {name: metrics_dict}."""
    print(f"\n{'Strategy':<20} {'Hit%':>8} {'AnnRet%':>10} {'Sharpe':>8} {'R2 OOS':>8} {'Total P&L':>12}")
    print("-" * 72)
    for name, m in results.items():
        print(f"{name:<20} {m['hit_ratio']:>7.1f}% {m['ann_return']:>9.1f}% {m['sharpe']:>8.3f} {m.get('r2_oos', 0):>8.4f} {m['total_pnl']:>11.1f}%")


def print_verdict(name, hit_ratio, sharpe, ci_lower, ci_upper, extra=""):
    """Print PASS/FAIL verdict based on hit ratio, Sharpe, and CI."""
    if hit_ratio > 50 and sharpe > 0 and ci_lower > 0:
        status = "PASSES"
    elif hit_ratio > 50 and sharpe > 0:
        status = "MARGINALLY PASSES - CI includes zero"
    elif sharpe < 0:
        status = "FAILS (negative Sharpe)"
    else:
        status = "INCONCLUSIVE"
    print(f"\n{name}: {status}")
    print(f"  Direction Accuracy: {hit_ratio:.1f}%")
    print(f"  Sharpe: {sharpe:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")
    if extra:
        print(f"  {extra}")

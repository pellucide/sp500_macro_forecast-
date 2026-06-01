"""
Main Execution Script for S&P 500 Macroeconomic Forecasting Project
State-Dependent Supervised Screening & Regularized Factor (SSRF) Architecture
"""

import warnings
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import (
    PROJECT_ROOT, DATA_DIR, MODELS_DIR, BACKTEST_DIR,
    DataConfig, ModelConfig, BacktestConfig
)
from .fred_data import (
    FREDDataLoader, DataProcessor,
    download_fred_data, generate_sample_data
)
from .ssrf_model import SSRFModel, SSRFConfig
from .backtesting import WalkForwardBacktester
from .tc_backtesting import TCAdjustedWalkForwardBacktester
from .evaluation import MetricsCalculator, generate_report

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_environment(args: argparse.Namespace) -> dict:
    """
    Setup environment and configuration.

    Args:
        args: Command line arguments

    Returns:
        Dictionary with environment settings
    """
    env = {
        'use_sample_data': args.use_sample_data or not DataConfig.FRED_API_KEY,
        'output_dir': Path(args.output_dir) if args.output_dir else BACKTEST_DIR,
        'save_results': not args.no_save,
        'verbose': args.verbose
    }

    env['output_dir'].mkdir(parents=True, exist_ok=True)

    return env


def load_data(env: dict, args: argparse.Namespace) -> tuple:
    """
    Load and prepare data.

    Args:
        env: Environment dictionary
        args: Command line arguments

    Returns:
        Tuple of (indicators, target, groups)
    """
    logger.info("Loading data...")

    if env['use_sample_data']:
        logger.info("Using generated sample data (no FRED API key)")
        indicators, base_target, groups = generate_sample_data(
            n_periods=args.n_periods,
            n_indicators=args.n_indicators,
            seed=args.seed
        )
        logger.info(f"Generated {len(indicators)} periods of data")
        logger.info(f"Created {len(groups)} feature groups")
        for group, features in groups.items():
            logger.info(f"  {group}: {len(features)} features")

        # Handle sector rotation for sample data
        if args.sector_rotation:
            logger.info(f"Creating synthetic sector rotation target for: {args.sector_rotation}")
            # Create sector-specific variation for sample data
            # Each sector has different sensitivity to underlying factors
            sector_weights = np.random.randn(args.n_indicators) * 0.02
            sector_signal = (indicators.values @ sector_weights)

            # Add sector-specific pattern
            np.random.seed(hash(args.sector_rotation) % (2**32))
            sector_noise = np.random.randn(len(indicators)) * 0.08
            target = pd.Series(
                sector_signal + sector_noise,
                index=indicators.index,
                name=f'{args.sector_rotation}_relative'
            )
        else:
            target = base_target
    else:
        # Map config sector names to fetcher names
        sector_name_map = {
            'Consumer_Staples': 'ConsumerStaples',
            'Consumer_Discretionary': 'ConsumerDiscretionary',
            'Communication': 'CommunicationServices',
            'Health_Care': 'Healthcare'
        }
        fetcher_sector = sector_name_map.get(args.sector_rotation, args.sector_rotation)

        logger.info("Downloading FRED-MD data...")
        loader = FREDDataLoader(api_key=DataConfig.FRED_API_KEY)

        indicators = loader.fetch_all_indicators(
            start_date=args.start_date,
            end_date=args.end_date
        )

        # Get target: sector rotation or S&P 500 returns
        if args.sector_rotation:
            logger.info(f"Predicting sector rotation for: {args.sector_rotation}")

            # Use Yahoo Finance for sector ETF data (FRED doesn't have ETF data)
            sector_returns = loader.fetch_sector_returns_yfinance(
                start_date=args.start_date,
                end_date=args.end_date
            )

            # Check using fetcher_sector name (mapped from config name)
            if len(sector_returns) > 0 and fetcher_sector in sector_returns.columns:
                target = sector_returns[fetcher_sector]
                logger.info(f"Using {args.sector_rotation} ({fetcher_sector}) relative returns as target")
            else:
                logger.warning(f"Sector {args.sector_rotation} not found. Using SP500.")
                target = loader.fetch_spx_returns(
                    start_date=args.start_date,
                    end_date=args.end_date
                )
        else:
            target = loader.fetch_spx_returns(
                start_date=args.start_date,
                end_date=args.end_date
            )

        # Align data
        common_idx = indicators.index.intersection(target.index)
        indicators = indicators.loc[common_idx]
        target = target.loc[common_idx]

        logger.info(f"Downloaded {len(indicators.columns)} indicators")

    # Process data
    processor = DataProcessor()
    indicators = processor.handle_missing_values(indicators, method='ffill')
    indicators = processor.winsorize_outliers(indicators)

    # For sample data, we already have groups from generate_sample_data
    # For FRED data, create groups from indicator categories
    if not env['use_sample_data']:
        groups = processor.create_category_groups(indicators)
        groups_dict = {k: v.columns.tolist() for k, v in groups.items()}
    else:
        # Use the groups we already created from generate_sample_data
        groups_dict = groups

    logger.info(f"Using {len(groups_dict)} feature groups")
    for group, features in groups_dict.items():
        logger.info(f"  {group}: {len(features)} features")

    return indicators, target, groups_dict


def run_backtest(
    indicators: pd.DataFrame,
    target: pd.Series,
    groups: dict,
    env: dict,
    args: argparse.Namespace
) -> dict:
    """
    Run the walk-forward backtest.

    Args:
        indicators: Feature DataFrame
        target: Target series
        groups: Feature groups
        env: Environment dictionary
        args: Command line arguments

    Returns:
        Dictionary with backtest results
    """
    logger.info("Starting walk-forward backtest...")

    # Configure model
    model_config = SSRFConfig(
        t_stat_threshold=args.t_stat_threshold,
        n_factors=args.n_factors,
        regime_window=args.regime_window,
        elastic_net_alpha=args.alpha,
        elastic_net_l1_ratio=args.l1_ratio,
        use_elastic_net_cv=not args.no_cv,
        use_regime_detection=not args.no_regime,
        model_type=args.model_type,
        # Transaction cost settings
        include_tc=args.include_tc,
        tc_rate_bps=args.tc_rate,
        expected_turnover=args.expected_turnover,
        account_tier=args.account_tier,
        # Conviction filtering settings
        conviction_filter_enabled=args.conviction_filter,
        min_conviction_threshold=args.conviction_threshold,
        # Prediction scaling
        prediction_scale=args.prediction_scale,
    )

    # Create model class wrapper to enable Ridge if requested
    if args.unregularized:
        class SSRFModelUnregularized(SSRFModel):
            def __init__(self, config=None):
                self._use_ridge = True
                super().__init__(config)

        model_class = SSRFModelUnregularized
    else:
        model_class = SSRFModel

    # Check if using TC-adjusted backtester
    if args.tc_backtest:
        logger.info(f"Using TC-adjusted backtester with {args.account_tier} tier ({args.tc_rate} bps)")
        backtester = TCAdjustedWalkForwardBacktester(
            model_class=model_class,
            initial_train_window=args.train_window,
            forecast_horizon=1,
            use_ct_restriction=not args.no_ct_restriction,
            step_size=args.step_size,
            tc_rate_bps=args.tc_rate,
            account_tier=args.account_tier,
            expected_turnover=args.expected_turnover,
        )
        # Set conviction filtering parameters
        backtester.conviction_filter_enabled = args.conviction_filter
        backtester.min_conviction_threshold = args.conviction_threshold
        result = backtester.run(
            indicators, target, groups, model_config, verbose=env['verbose']
        )

        return {
            'predictions': result.predictions,
            'tc_adjusted_predictions': result.tc_adjusted_predictions,
            'actual_returns': result.actual_returns,
            'benchmark': result.benchmark_predictions,
            'turnover': result.turnover,
            'tc_costs': result.tc_costs,
            'net_returns': result.net_returns,
            'gross_returns': result.gross_returns,
            'metrics': result.metrics,
            'tc_metrics': result.tc_metrics,
            'dates': result.dates,
            'train_windows': result.train_windows,
            'use_tc_adjusted': True,
        }
    else:
        backtester = WalkForwardBacktester(
            model_class=model_class,
            initial_train_window=args.train_window,
            forecast_horizon=1,
            use_ct_restriction=not args.no_ct_restriction,
            step_size=args.step_size
        )

        result = backtester.run(
            indicators, target, groups, model_config, verbose=env['verbose']
        )

        return {
            'predictions': result.predictions,
            'actual_returns': result.actual_returns,
            'benchmark': result.benchmark_predictions,
            'metrics': result.metrics,
            'dates': result.dates,
            'train_windows': result.train_windows,
            'use_tc_adjusted': False,
        }


def evaluate_results(
    results: dict,
    env: dict,
    args: argparse.Namespace
) -> dict:
    """
    Calculate comprehensive metrics and generate report.

    Args:
        results: Backtest results
        env: Environment dictionary
        args: Command line arguments

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Calculating evaluation metrics...")

    # Use backtester metrics
    metrics = results['metrics']

    # Add statistical tests if requested
    if args.statistical_tests:
        from .evaluation import StatisticalTests

        # Diebold-Mariano test
        dm_stat, dm_pval = StatisticalTests.dm_test(
            results['actual_returns'].values,
            results['predictions'].values,
            results['benchmark'].values
        )

        # Clark-West test
        cw_stat, cw_pval = StatisticalTests.cw_test(
            results['actual_returns'].values,
            results['predictions'].values,
            results['benchmark'].values
        )

        metrics['dm_statistic'] = dm_stat
        metrics['dm_p_value'] = dm_pval
        metrics['cw_statistic'] = cw_stat
        metrics['cw_p_value'] = cw_pval

    # R² OOS confidence interval
    from .evaluation import StatisticalTests
    r2_lower, r2_upper = StatisticalTests.out_of_sample_r2_confidence_interval(
        metrics['r2_oos'],
        len(results['predictions'])
    )
    metrics['r2_oos_ci_lower'] = r2_lower
    metrics['r2_oos_ci_upper'] = r2_upper

    return metrics


def save_results(
    results: dict,
    metrics: dict,
    env: dict,
    args: argparse.Namespace
):
    """
    Save results to files.

    Args:
        results: Backtest results
        metrics: Evaluation metrics
        env: Environment dictionary
        args: Command line arguments
    """
    if not env['save_results']:
        return

    logger.info("Saving results...")

    output_dir = env['output_dir']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save predictions
    pred_df = pd.DataFrame({
        'date': results['dates'],
        'actual': results['actual_returns'].values,
        'predicted': results['predictions'].values,
        'benchmark': results['benchmark'].values
    })
    pred_path = output_dir / f"predictions_{timestamp}.csv"
    pred_df.to_csv(pred_path, index=False)
    logger.info(f"Predictions saved to {pred_path}")

    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.index = [timestamp]
    metrics_path = output_dir / f"metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_path)
    logger.info(f"Metrics saved to {metrics_path}")

    # Generate and save report
    report = generate_report(
        metrics,
        model_name="SSRF",
        additional_info={
            'Training Window': f"{args.train_window} months",
            'N Factors': args.n_factors,
            'T-stat Threshold': args.t_stat_threshold,
            'Regime Window': args.regime_window
        }
    )

    report_path = output_dir / f"report_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")

    # Save plots
    if args.save_plots:
        try:
            from .backtesting import plot_predictions

            fig = plot_predictions(
                results,
                save_path=output_dir / f"predictions_plot_{timestamp}.png"
            )
            if fig:
                logger.info(f"Plot saved to {output_dir / f'predictions_plot_{timestamp}.png'}")
        except Exception as e:
            logger.warning(f"Failed to save plot: {e}")


def print_summary(metrics: dict, args: argparse.Namespace, results: dict = None):
    """Print summary of results."""
    print("\n" + "=" * 60)
    print("SUMMARY: S&P 500 Macroeconomic Forecasting Results")
    print("=" * 60)

    print("\n📊 Statistical Performance:")
    print(f"   Campbell-Thompson R² OOS: {metrics.get('r2_oos', metrics.get('r2_oos_tc_adjusted', 0)):.4f}")
    print(f"   Direction Accuracy: {metrics.get('hit_ratio', metrics.get('hit_ratio_tc_adjusted', 0)):.2%}")

    print("\n📈 Risk-Adjusted Returns:")
    print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', metrics.get('sharpe_net', 0)):.4f}")
    print(f"   Calmar Ratio: {metrics.get('calmar_ratio', 0):.4f}")

    print("\n📉 Drawdown Analysis:")
    print(f"   Max Drawdown: {metrics.get('max_drawdown', metrics.get('max_drawdown_net', 0)):.2%}")
    print(f"   Benchmark Max Drawdown: {metrics.get('benchmark_max_drawdown', 0):.2%}")

    print("\n💰 Cumulative Performance:")
    print(f"   Strategy Return: {metrics.get('cumulative_return', metrics.get('cumulative_net', 0)):.2%}")
    print(f"   Benchmark Return: {metrics.get('benchmark_cumulative_return', metrics.get('spx_cumulative', 0)):.2%}")
    print(f"   Strategy Volatility: {metrics.get('volatility', metrics.get('volatility_net', 0)):.2%}")

    # Print TC metrics if using TC-adjusted backtest
    if results and results.get('use_tc_adjusted'):
        print("\n💸 Transaction Cost Metrics:")
        tc_metrics = results.get('tc_metrics', {})
        print(f"   Account Tier: {tc_metrics.get('account_tier', 'standard')}")
        print(f"   Effective TC Rate: {tc_metrics.get('effective_tc_rate_bps', 0):.1f} bps")
        print(f"   Number of Trades: {tc_metrics.get('n_trades', 0)}")
        print(f"   Total TC Cost: {tc_metrics.get('total_tc_cost_pct', 0):.1f}%")
        print(f"   Gross Annual Return: {metrics.get('ann_return_gross', 0)*100:.2f}%")
        print(f"   Net Annual Return: {metrics.get('ann_return_net', 0)*100:.2f}%")
        print(f"   TC Drag: {metrics.get('tc_drag', 0)*100:.2f}%")
        print(f"   vs S&P 500 (Net): {metrics.get('vs_spx_net', 0)*100:.2f}%")

    # Print conviction filter settings if enabled
    if hasattr(args, 'conviction_filter') and args.conviction_filter:
        print("\n🎯 Conviction Filter:")
        print(f"   Enabled: Yes")
        print(f"   Threshold: {args.conviction_threshold:.2f}")

    if args.statistical_tests:
        print("\n🧪 Statistical Tests:")
        print(f"   Diebold-Mariano: t={metrics.get('dm_statistic', 0):.4f}, p={metrics.get('dm_p_value', 1):.4f}")
        print(f"   Clark-West: t={metrics.get('cw_statistic', 0):.4f}, p={metrics.get('cw_p_value', 1):.4f}")

    print("\n" + "=" * 60)


def main(args=None):
    """
    Main execution function.

    Args:
        args: Optional arguments (for testing)

    Returns:
        Dictionary with results and metrics
    """
    parser = argparse.ArgumentParser(
        description="S&P 500 Macroeconomic Forecasting with SSRF Model"
    )

    # Data options
    parser.add_argument('--use-sample-data', action='store_true',
                        help='Use sample data instead of FRED')
    parser.add_argument('--n-periods', type=int, default=400,
                        help='Number of periods for sample data')
    parser.add_argument('--n-indicators', type=int, default=50,
                        help='Number of indicators for sample data')
    parser.add_argument('--n-factors', type=int, default=10,
                        help='Number of factors to extract')
    parser.add_argument('--start-date', type=str, default='1959-01-01',
                        help='Start date for FRED data')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date for FRED data')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--sector-rotation', type=str, default=None,
                        help='Predict specific sector (Materials, Energy, Financials, Technology, etc.)')

    # Model options
    parser.add_argument('--train-window', type=int, default=60,
                        help='Initial training window (months)')
    parser.add_argument('--t-stat-threshold', type=float, default=1.5,
                        help='T-statistic threshold for screening')
    parser.add_argument('--regime-window', type=int, default=12,
                        help='Regime proxy window (months)')
    parser.add_argument('--alpha', type=float, default=0.001,
                        help='Elastic Net alpha')
    parser.add_argument('--l1-ratio', type=float, default=0.5,
                        help='Elastic Net L1 ratio')
    parser.add_argument('--prediction-scale', type=float, default=1.0,
                        help='Scale factor for predictions (default: 1.0, recommended: 5-20)')
    parser.add_argument('--no-cv', action='store_true',
                        help='Disable cross-validation for regularization')
    parser.add_argument('--unregularized', action='store_true',
                        help='Use unregularized linear regression (Ridge with tiny alpha)')
    parser.add_argument('--no-ct-restriction', action='store_true',
                        help='Disable Campbell-Thompson restriction')
    parser.add_argument('--no-regime', action='store_true',
                        help='Disable regime detection features')
    parser.add_argument('--step-size', type=int, default=1,
                        help='Walk-forward step size (months)')
    parser.add_argument('--model-type', type=str, default='elasticnet',
                        choices=['elasticnet', 'linear', 'xgboost', 'random_forest', 'catboost', 'mlp', 'ensemble'],
                        help='Final regression model type')

    # Transaction cost options
    parser.add_argument('--tc-rate', type=float, default=25.0,
                        help='Transaction cost rate in basis points (default: 25)')
    parser.add_argument('--account-tier', type=str, default='standard',
                        choices=['micro', 'standard', 'professional', 'institutional'],
                        help='Account tier for TC calculation (default: standard)')
    parser.add_argument('--expected-turnover', type=float, default=0.15,
                        help='Expected portfolio turnover rate (default: 0.15 = 15%)')
    parser.add_argument('--include-tc', action='store_true',
                        help='Include transaction costs in predictions')
    parser.add_argument('--tc-backtest', action='store_true',
                        help='Use TC-adjusted backtester')

    # Conviction filtering options
    parser.add_argument('--conviction-filter', action='store_true',
                        help='Enable high-conviction signal filtering')
    parser.add_argument('--conviction-threshold', type=float, default=1.0,
                        help='Minimum conviction threshold (default: 1.0, z-score style)')

    # Output options
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results to files')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save prediction plots')
    parser.add_argument('--statistical-tests', action='store_true',
                        help='Run statistical significance tests')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress')

    parsed_args = parser.parse_args(args)

    # Suppress warnings
    warnings.filterwarnings('ignore')

    # Setup
    env = setup_environment(parsed_args)

    logger.info("=" * 60)
    logger.info("S&P 500 Macroeconomic Forecasting with SSRF Model")
    logger.info("=" * 60)

    # Load data
    indicators, target, groups = load_data(env, parsed_args)

    # Run backtest
    results = run_backtest(indicators, target, groups, env, parsed_args)

    # Evaluate
    metrics = evaluate_results(results, env, parsed_args)

    # Save
    save_results(results, metrics, env, parsed_args)

    # Print summary (pass results for TC metrics)
    print_summary(metrics, parsed_args, results)

    return {
        'results': results,
        'metrics': metrics
    }


if __name__ == "__main__":
    main()
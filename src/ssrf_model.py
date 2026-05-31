"""
State-Dependent Supervised Screening & Regularized Factor (SSRF) Architecture
Four-stage defensive pipeline for low SNR environments
Includes regime detection for state-dependent forecasting.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
import logging

from sklearn.linear_model import LinearRegression, ElasticNetCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats

from .config import ModelConfig, DataConfig
from .regime_detection import (
    RegimeConfig, RegimeType, MarketRegimeDetector, create_regime_features
)

logger = logging.getLogger(__name__)


# =============================================================================
# Transaction Cost Configuration
# =============================================================================
class TCConfig:
    """Configuration for transaction cost modeling."""
    # Base TC rates in basis points
    MICRO_ACCOUNT_TC = 50.0   # < $10k
    STANDARD_ACCOUNT_TC = 25.0  # $10k - $100k
    PROFESSIONAL_TC = 15.0    # $1M - $10M
    INSTITUTIONAL_TC = 5.0    # > $10M

    # Default tier
    DEFAULT_TIER = "standard"  # Can be: micro, standard, professional, institutional

    @classmethod
    def get_tc_rate(cls, tier: str = None) -> float:
        """Get TC rate based on account tier."""
        tier = tier or cls.DEFAULT_TIER
        rates = {
            "micro": cls.MICRO_ACCOUNT_TC,
            "standard": cls.STANDARD_ACCOUNT_TC,
            "professional": cls.PROFESSIONAL_TC,
            "institutional": cls.INSTITUTIONAL_TC,
        }
        return rates.get(tier.lower(), cls.STANDARD_ACCOUNT_TC)


@dataclass
class SSRFConfig:
    """Configuration for SSRF model."""
    t_stat_threshold: float = ModelConfig.SCREENING_T_STAT_THRESHOLD
    n_factors: int = ModelConfig.N_FACTORS
    regime_window: int = ModelConfig.REGIME_WINDOW
    elastic_net_alpha: float = ModelConfig.ELASTIC_NET_ALPHA
    elastic_net_l1_ratio: float = ModelConfig.ELASTIC_NET_L1_RATIO
    use_elastic_net_cv: bool = ModelConfig.USE_ELASTIC_NET_CV
    n_inner_cv_folds: int = ModelConfig.N_INNER_CV_FOLDS
    # Regime detection settings
    use_regime_detection: bool = True
    regime_n_regimes: int = 3
    regime_smoothing_window: int = 3
    # Final regression model type: 'elasticnet', 'linear', 'xgboost', 'random_forest'
    model_type: str = 'elasticnet'
    # Transaction cost settings
    include_tc: bool = False  # Include TC factor in prediction adjustment
    tc_rate_bps: float = 25.0  # Transaction cost rate in basis points
    expected_turnover: float = 0.15  # Expected turnover rate (15%)
    account_tier: str = "standard"  # Account tier for TC calculation

    # High-conviction settings
    min_conviction_threshold: float = 0.0  # Minimum signal strength to trade (0-1 scale)
    # When to act only on high conviction:
    # 0.0 = trade on all signals (standard behavior)
    # 0.5 = only trade when |signal| > 0.5 std deviations
    # 0.75 = only trade when very confident signals
    conviction_filter_enabled: bool = False  # Enable high-conviction filtering

    # Prediction scaling
    # SSRF predictions are often very small due to heavy regularization.
    # This parameter scales up predictions to capture more return.
    # Default: 1.0 (no scaling)
    # Recommended range: 5.0 - 20.0 for most applications
    prediction_scale: float = 1.0  # Multiply predictions by this factor


@dataclass
class ModelState:
    """State variables for the SSRF model."""
    selected_features: Dict[str, List[str]]
    scaling_factors: Dict[str, np.ndarray]
    scaler: StandardScaler
    pca: PCA
    mean_factors_train: np.ndarray
    volatility_percentiles_train: np.ndarray
    final_model: Optional[ElasticNetCV] = None
    coefficients: Optional[np.ndarray] = None
    # Regime detection state
    regime_detector: Optional[MarketRegimeDetector] = None
    regime_features: Optional[pd.DataFrame] = None


class GroupwiseScreen:
    """
    Stage 1: Group-Wise Supervised Screening

    Screens predictors within economic categories to prevent category dominance.
    Retains predictors with univariate t-statistics above threshold θ.
    """

    def __init__(self, t_stat_threshold: float = 1.5):
        """
        Initialize screen.

        Args:
            t_stat_threshold: Minimum |t-stat| for retention
        """
        self.t_stat_threshold = t_stat_threshold

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Dict[str, List[str]]
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[str]]]:
        """
        Perform group-wise screening.

        Args:
            X: Feature DataFrame
            y: Target variable
            groups: Dictionary mapping category to list of features

        Returns:
            Tuple of (screened features by group, selected feature names)
        """
        selected = {}
        selected_features = []

        for group_name, features in groups.items():
            # Filter to features that exist in X
            available = [f for f in features if f in X.columns]
            if not available:
                logger.debug(f"No features available for group {group_name}")
                selected[group_name] = pd.DataFrame(index=X.index)
                continue

            X_group = X[available]

            # Compute univariate t-statistics
            t_stats = self._compute_t_statistics(X_group, y)

            # Select features with |t-stat| > threshold
            # If no features pass, take the best one anyway
            mask = np.abs(t_stats) > self.t_stat_threshold
            selected_idx = np.where(mask)[0]

            # If no features pass threshold, take top feature by |t-stat|
            if len(selected_idx) == 0 and len(t_stats) > 0:
                selected_idx = [np.argmax(np.abs(t_stats))]

            selected_names = [available[i] for i in selected_idx]

            if selected_names:
                selected[group_name] = X_group[selected_names].copy()
                selected_features.extend(selected_names)
                logger.info(
                    f"Group '{group_name}': {len(selected_names)}/{len(available)} "
                    f"features retained (threshold={self.t_stat_threshold:.2f})"
                )
            else:
                # Don't add empty DataFrame - skip empty groups
                logger.debug(f"Group '{group_name}': 0/{len(available)} features retained (skipping)")

        # Filter out empty groups
        selected = {k: v for k, v in selected.items() if len(v.columns) > 0}

        return selected, selected_features

    def _compute_t_statistics(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> np.ndarray:
        """
        Compute univariate t-statistics for each feature.

        Args:
            X: Feature DataFrame
            y: Target variable

        Returns:
            Array of t-statistics
        """
        # Align X and y
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X_valid = X.loc[valid_idx]
        y_valid = y.loc[valid_idx]

        if len(X_valid) < 10:
            return np.zeros(len(X.columns))

        t_stats = np.zeros(len(X.columns))

        for i, col in enumerate(X.columns):
            try:
                # Univariate regression
                reg = LinearRegression()
                X_col = X_valid[[col]].values
                reg.fit(X_col, y_valid.values)

                # Compute t-statistic
                residuals = y_valid.values - reg.predict(X_col)
                n = len(residuals)
                mse = np.sum(residuals ** 2) / (n - 2)

                # Standard error of slope
                x_mean = X_col.mean()
                x_var = np.sum((X_col - x_mean) ** 2)
                se = np.sqrt(mse / x_var)

                if se > 0:
                    t_stats[i] = reg.coef_[0] / se
            except Exception as e:
                logger.debug(f"Error computing t-stat for {col}: {e}")
                t_stats[i] = 0

        return t_stats


class PredictiveScaler:
    """
    Stage 2: Predictive Scaling

    Scales retained predictors by their univariate predictive slopes
    to prioritize signal over variance.
    """

    def __init__(self):
        self.slopes = {}

    def fit_transform(
        self,
        X: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Apply predictive scaling.

        Args:
            X: Screened feature DataFrame

        Returns:
            Tuple of (scaled features, scaling factors)
        """
        if X.empty or X.shape[1] == 0:
            return X, np.array([])

        # Standardize first
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Compute and apply predictive slopes
        slopes = np.std(X.values, axis=0)
        slopes[slopes == 0] = 1  # Avoid division by zero

        self.slopes = {col: slopes[i] for i, col in enumerate(X.columns)}
        X_scaled_df = pd.DataFrame(
            X_scaled,
            index=X.index,
            columns=X.columns
        )

        return X_scaled_df, slopes

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted scaling to new data.

        Args:
            X: Feature DataFrame

        Returns:
            Scaled features
        """
        if X.empty or len(self.slopes) == 0:
            return X

        # Standardize
        X_scaled = self.scaler.transform(X)

        # Apply scaling
        # We need to access the scaler - but it was created in fit_transform
        # Store it as instance variable
        if not hasattr(self, 'scaler') or self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(X)

        return pd.DataFrame(X_scaled, index=X.index, columns=X.columns)


class SupervisedFactorExtractor:
    """
    Stage 3: Supervised Factor Extraction

    Uses PCA to extract latent factors from the screened/scaled feature set.
    """

    def __init__(self, n_factors: int = 10):
        """
        Initialize factor extractor.

        Args:
            n_factors: Number of latent factors to extract
        """
        self.n_factors = n_factors
        self.pca = PCA(n_components=n_factors)
        self.mean_factors_train = None

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Extract factors from features.

        Args:
            X: Scaled feature DataFrame

        Returns:
            DataFrame with extracted factors
        """
        if X.empty or X.shape[1] == 0:
            logger.warning("No features available for factor extraction")
            return pd.DataFrame(index=X.index, columns=[f'factor_{i}' for i in range(self.n_factors)])

        if X.shape[1] < self.n_factors:
            logger.warning(
                f"Only {X.shape[1]} features available for {self.n_factors} factors. "
                f"Reducing factors."
            )
            actual_factors = X.shape[1]
            self.pca = PCA(n_components=actual_factors)
            self.n_factors = actual_factors

        # Fit PCA
        X_array = X.values
        valid_mask = ~np.isnan(X_array).any(axis=1)

        if valid_mask.sum() < self.n_factors:
            logger.error("Insufficient valid observations for PCA")
            return pd.DataFrame(index=X.index, columns=[f'factor_{i}' for i in range(self.n_factors)])

        X_valid = X_array[valid_mask]

        try:
            factors = self.pca.fit_transform(X_valid)
            # Store the mean of factors (mean of each principal component across samples)
            self.mean_factors_train = factors.mean(axis=0)
        except Exception as e:
            logger.error(f"PCA failed: {e}")
            return pd.DataFrame(index=X.index, columns=[f'factor_{i}' for i in range(self.n_factors)])

        # Create DataFrame
        factor_names = [f'factor_{i}' for i in range(self.n_factors)]
        factors_df = pd.DataFrame(index=X.index, columns=factor_names)
        factors_df.loc[X.index[valid_mask]] = factors

        # Fill NaN rows with forward/backward fill
        if (~valid_mask).sum() > 0:
            factors_df = factors_df.ffill().bfill()

        logger.info(
            f"Extracted {self.n_factors} factors. "
            f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.2%}"
        )

        # Store factor names for later use
        self._factor_names = factor_names

        return factors_df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using fitted PCA.

        Args:
            X: Feature DataFrame

        Returns:
            DataFrame with transformed factors
        """
        if X.empty or X.shape[1] == 0:
            return pd.DataFrame(index=X.index, columns=[f'factor_{i}' for i in range(self.n_factors)])

        X_array = X.values

        # Handle case where feature count doesn't match training
        expected_features = self.pca.n_features_in_
        actual_features = X_array.shape[1]

        if actual_features != expected_features:
            logger.warning(
                f"Feature count mismatch: expected {expected_features}, got {actual_features}. "
                f"Padding with zeros."
            )
            # Pad with zeros if we have fewer features
            if actual_features < expected_features:
                padding = np.zeros((X_array.shape[0], expected_features - actual_features))
                X_array = np.hstack([X_array, padding])
            # Truncate if we have more features
            else:
                X_array = X_array[:, :expected_features]

        try:
            factors = self.pca.transform(X_array)
        except Exception as e:
            logger.error(f"PCA transform failed: {e}")
            return pd.DataFrame(index=X.index, columns=[f'factor_{i}' for i in range(self.n_factors)])

        factor_names = [f'factor_{i}' for i in range(self.n_factors)]
        return pd.DataFrame(factors, index=X.index, columns=factor_names)


class RegimeProxy:
    """
    Stage 4: Interaction with Causal Soft-Regime Proxy

    Computes rolling volatility percentile as regime proxy
    and creates interaction terms.
    """

    def __init__(self, regime_window: int = 12):
        """
        Initialize regime proxy.

        Args:
            regime_window: Rolling window for volatility computation (months)
        """
        self.regime_window = regime_window
        self.training_percentiles = None

    def compute_rolling_volatility(
        self,
        returns: pd.Series,
        window: Optional[int] = None
    ) -> pd.Series:
        """
        Compute rolling volatility of returns.

        Args:
            returns: Return series
            window: Rolling window (defaults to self.regime_window)

        Returns:
            Rolling volatility series
        """
        window = window or self.regime_window
        return returns.rolling(window=window).std()

    def compute_percentile_rank(
        self,
        values: pd.Series,
        reference: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Compute percentile rank of values relative to reference distribution.

        Args:
            values: Values to rank
            reference: Reference distribution for ranking

        Returns:
            Percentile ranks (0-1)
        """
        if reference is None:
            reference = values

        # Store training distribution for prediction phase
        if self.training_percentiles is None:
            self.training_percentiles = reference.dropna().values

        # Vectorized percentile rank computation
        # Use expanding window for historical percentile
        result = pd.Series(index=values.index, dtype=float)

        for i in range(len(values)):
            current_value = values.iloc[i]
            current_date = values.index[i]

            # Get reference data up to current date
            if current_date in reference.index:
                ref_data = reference.loc[:current_date].dropna()
            else:
                ref_data = self.training_percentiles

            if len(ref_data) > 0:
                # Percentile rank: percentage of values below current
                result.iloc[i] = (ref_data < current_value).mean()
            else:
                result.iloc[i] = 0.5  # Default to median

        return result

    def create_interaction_terms(
        self,
        factors: pd.DataFrame,
        regime_proxy: pd.Series
    ) -> pd.DataFrame:
        """
        Create interaction terms between factors and regime proxy.
        Centers factors to ensure orthogonality.

        Args:
            factors: Factor DataFrame
            regime_proxy: Regime proxy series

        Returns:
            DataFrame with main effects and interactions
        """
        if factors.empty:
            return pd.DataFrame(index=regime_proxy.index)

        # Center factors using training mean
        if hasattr(self, 'mean_factors') and self.mean_factors is not None:
            centered_factors = factors.values - self.mean_factors
        else:
            centered_factors = factors.values - factors.mean()

        # Create interaction terms: F * P(Z)
        interaction = centered_factors * regime_proxy.values.reshape(-1, 1)

        # Combine main effects and interactions
        factor_names = factors.columns.tolist()
        interaction_names = [f'{name}_interaction' for name in factor_names]

        result = pd.DataFrame(index=factors.index)
        for i, name in enumerate(factor_names):
            result[name] = centered_factors[:, i]
            result[interaction_names[i]] = interaction[:, i]

        return result

    def set_training_volatility_percentiles(self, percentiles: np.ndarray):
        """Store training volatility percentiles for reference."""
        self.training_percentiles = percentiles

    def set_mean_factors(self, mean_factors: np.ndarray):
        """Store mean factors for centering."""
        self.mean_factors = mean_factors


class SSRFModel:
    """
    Complete State-Dependent Supervised Screening & Regularized Factor Model

    Implements the full four-stage pipeline:
    1. Group-wise supervised screening
    2. Predictive scaling
    3. Supervised factor extraction (PCA)
    4. Interaction modeling with regime proxy
    """

    def __init__(self, config: Optional[SSRFConfig] = None):
        """
        Initialize SSRF model.

        Args:
            config: Model configuration
        """
        self.config = config or SSRFConfig()
        self.state = None

        # Initialize _use_ridge flag before checking it
        # Subclasses should set _use_ridge = True before calling super().__init__()
        # to enable unregularized LinearRegression
        self._use_ridge = getattr(self, '_use_ridge', False)

        # Initialize components
        self.screen = GroupwiseScreen(self.config.t_stat_threshold)
        self.scaler = PredictiveScaler()
        self.factor_extractor = SupervisedFactorExtractor(self.config.n_factors)
        self.regime_proxy = RegimeProxy(self.config.regime_window)

        # Model for final regression based on model_type
        self.model = self._create_final_model()

    def _create_final_model(self):
        """
        Create the final regression model based on configuration.

        Returns:
            Fitted model instance
        """
        model_type = self.config.model_type.lower()

        # Check for legacy _use_ridge flag (from subclass SSRFModelUnregularized)
        if self._use_ridge:
            logger.info("Using unregularized LinearRegression")
            return LinearRegression()

        if model_type == 'linear':
            logger.info("Using LinearRegression")
            return LinearRegression()

        elif model_type == 'xgboost':
            logger.info("Using XGBoost (shallow trees for low SNR)")
            try:
                from xgboost import XGBRegressor
                return XGBRegressor(
                    n_estimators=100,
                    max_depth=3,  # Shallow trees for low SNR
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=5,  # Prevent overfitting
                    reg_alpha=0.1,  # L1 regularization
                    reg_lambda=1.0,  # L2 regularization
                    random_state=42,
                    verbosity=0
                )
            except ImportError:
                logger.warning("XGBoost not available, falling back to LinearRegression")
                return LinearRegression()

        elif model_type == 'random_forest':
            logger.info("Using Random Forest")
            try:
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_leaf=10,  # Prevent overfitting
                    min_samples_split=20,
                    random_state=42
                )
            except ImportError:
                logger.warning("RandomForest not available, falling back to LinearRegression")
                return LinearRegression()

        elif model_type == 'catboost':
            logger.info("Using CatBoost (shallow trees for low SNR)")
            try:
                from catboost import CatBoostRegressor
                return CatBoostRegressor(
                    iterations=100,
                    depth=3,  # Shallow trees for low SNR
                    learning_rate=0.05,
                    l2_leaf_reg=3,  # L2 regularization
                    random_seed=42,
                    verbose=False,
                    allow_writing_files=False
                )
            except ImportError:
                logger.warning("CatBoost not available, falling back to LinearRegression")
                return LinearRegression()

        elif model_type == 'mlp':
            logger.info("Using MLP with strong regularization for low SNR")
            try:
                from sklearn.neural_network import MLPRegressor
                return MLPRegressor(
                    hidden_layer_sizes=(16, 8),  # Very small architecture for low SNR
                    activation='tanh',  # tanh is often more stable than relu for financial data
                    solver='adam',
                    alpha=1.0,  # STRONG L2 regularization (100x more than default)
                    learning_rate='adaptive',
                    learning_rate_init=0.001,  # Slower learning
                    max_iter=1000,
                    early_stopping=True,
                    validation_fraction=0.15,  # More validation data
                    n_iter_no_change=20,  # Longer patience
                    tol=1e-5,  # Tighter tolerance
                    random_state=42
                )
            except ImportError:
                logger.warning("MLP not available, falling back to LinearRegression")
                return LinearRegression()

        elif model_type == 'ensemble':
            logger.info("Using Ensemble: Linear + XGBoost (simple average)")
            from sklearn.ensemble import VotingRegressor
            try:
                from xgboost import XGBRegressor
                linear = LinearRegression()
                xgb = XGBRegressor(
                    n_estimators=50,  # Fewer estimators for ensemble
                    max_depth=2,  # Even shallower in ensemble
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=10,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    random_state=42,
                    verbosity=0
                )
                # VotingRegressor averages predictions
                return VotingRegressor([
                    ('linear', linear),
                    ('xgb', xgb)
                ])
            except ImportError:
                logger.warning("XGBoost not available, using ElasticNet only")
                if self.config.use_elastic_net_cv:
                    return ElasticNetCV(
                        l1_ratio=self.config.elastic_net_l1_ratio,
                        alphas=np.logspace(-5, 0, 50),
                        cv=self.config.n_inner_cv_folds,
                        random_state=42,
                        max_iter=5000
                    )
                else:
                    return LinearRegression()

        elif model_type == 'elasticnet':
            if self.config.use_elastic_net_cv:
                logger.info("Using ElasticNet with cross-validation")
                return ElasticNetCV(
                    l1_ratio=self.config.elastic_net_l1_ratio,
                    alphas=np.logspace(-5, 0, 50),
                    cv=self.config.n_inner_cv_folds,
                    random_state=42,
                    max_iter=5000
                )
            else:
                from sklearn.linear_model import ElasticNet
                logger.info("Using ElasticNet without cross-validation")
                return ElasticNet(
                    alpha=self.config.elastic_net_alpha,
                    l1_ratio=self.config.elastic_net_l1_ratio,
                    random_state=42,
                    max_iter=5000
                )

        else:
            logger.warning(f"Unknown model type '{model_type}', using LinearRegression")
            return LinearRegression()

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Dict[str, List[str]]
    ) -> 'SSRFModel':
        """
        Fit the SSRF model.

        Args:
            X: Feature DataFrame
            y: Target variable
            groups: Feature groups for screening

        Returns:
            self
        """
        logger.info("Fitting SSRF model...")

        # Store original data for regime proxy calculation
        self._X_train = X.copy()
        self._y_train = y.copy()
        self._groups = groups

        # Stage 1: Group-wise screening
        logger.info("Stage 1: Group-wise supervised screening")
        screened, selected_features = self.screen.fit_transform(X, y, groups)
        logger.info(f"Selected {len(selected_features)} features total")

        # Check if any features were selected
        if len(selected_features) == 0 or len(screened) == 0:
            logger.warning("No features passed screening. Using benchmark prediction.")
            self.state = ModelState(
                selected_features={},
                scaling_factors={},
                scaler=StandardScaler(),
                pca=PCA(n_components=self.config.n_factors),
                mean_factors_train=np.zeros(self.config.n_factors),
                volatility_percentiles_train=np.zeros(self.config.regime_window)
            )
            return self

        # Combine screened features
        X_screened = pd.concat(screened.values(), axis=1)

        # Store the selected feature names for consistent prediction
        self._selected_feature_names = selected_features

        # Stage 2: Predictive scaling
        logger.info("Stage 2: Predictive scaling")
        X_scaled, scaling_factors = self.scaler.fit_transform(X_screened)

        # Stage 3: Factor extraction
        logger.info("Stage 3: Supervised factor extraction")
        factors = self.factor_extractor.fit_transform(X_scaled)

        # Store n_factors actually used (may be less if fewer features)
        self._n_factors_actual = factors.shape[1]

        # Compute regime proxy
        logger.info("Stage 4: Computing regime proxy")
        rolling_vol = self.regime_proxy.compute_rolling_volatility(y, self.config.regime_window)
        regime_p = self.regime_proxy.compute_percentile_rank(rolling_vol, rolling_vol)

        # Store training percentiles and mean factors
        self.regime_proxy.set_training_volatility_percentiles(
            regime_p.dropna().values
        )
        self.regime_proxy.set_mean_factors(
            self.factor_extractor.mean_factors_train
        )

        # Create interaction terms
        X_final = self.regime_proxy.create_interaction_terms(factors, regime_p)

        # Handle any remaining NaN
        X_final = X_final.fillna(0)

        # Optional: Add regime detection features
        regime_features = None
        if self.config.use_regime_detection:
            logger.info("Detecting market regimes...")
            try:
                regime_config = RegimeConfig(
                    n_regimes=self.config.regime_n_regimes,
                    min_regime_duration=self.config.regime_smoothing_window
                )
                regime_detector = MarketRegimeDetector(regime_config)
                regime_detector.fit(y)

                # Detect regimes in training data
                regimes = regime_detector.detect(y)

                # Create regime features
                regime_features = create_regime_features(y, regimes)

                # Add regime features to X_final
                for col in regime_features.columns:
                    if col not in X_final.columns:
                        X_final[col] = regime_features[col].values

                # Log regime distribution
                regime_dist = regime_detector.get_regime_distribution(regimes)
                logger.info(f"Regime distribution: {regime_dist}")

                self._regime_detector = regime_detector

            except Exception as e:
                logger.warning(f"Regime detection failed: {e}. Continuing without regime features.")
                self._regime_detector = None
        else:
            self._regime_detector = None

        # Store the feature names from training for prediction alignment
        self._training_feature_names = X_final.columns.tolist()

        # Final regression with cross-validation
        logger.info("Fitting final ElasticNet model")
        valid_mask = ~(X_final.isna().any(axis=1) | y.isna())
        X_final_valid = X_final.loc[valid_mask]
        y_valid = y.loc[valid_mask]

        if len(X_final_valid) < 20:
            logger.error("Insufficient training samples")
            return self

        # Fit the model
        self.model.fit(X_final_valid.values, y_valid.values)

        # Store model state - use getattr for models without coef_ (tree-based models)
        self.state = ModelState(
            selected_features={k: list(v.columns) for k, v in screened.items() if not v.empty},
            scaling_factors={col: scaling_factors[i] for i, col in enumerate(X_screened.columns) if col in selected_features},
            scaler=StandardScaler(),  # Placeholder
            pca=self.factor_extractor.pca,
            mean_factors_train=self.factor_extractor.mean_factors_train,
            volatility_percentiles_train=regime_p.dropna().values,
            final_model=self.model,
            coefficients=getattr(self.model, 'coef_', None),
            regime_detector=getattr(self, '_regime_detector', None),
            regime_features=regime_features
        )

        # Log non-zero coefficients (only for linear models)
        if hasattr(self.model, 'coef_'):
            logger.info(f"Model fitted. Non-zero coefficients: {np.sum(self.model.coef_ != 0)}")
        else:
            logger.info(f"Model fitted ({self.config.model_type}). Tree-based model - no coefficient sparsity metric.")

        return self

    def predict(self, X: pd.DataFrame, y_for_regime: Optional[pd.Series] = None) -> pd.Series:
        """
        Generate predictions using the fitted model.

        Args:
            X: Feature DataFrame for prediction
            y_for_regime: Returns series for regime proxy computation

        Returns:
            Predictions series
        """
        if self.state is None:
            raise ValueError("Model must be fitted before prediction")

        # Stage 1: Filter to only the features that were selected during training
        all_selected = []
        for feats in self.state.selected_features.values():
            all_selected.extend(feats)

        # Use only selected features that exist in X
        X_filtered = X[[c for c in all_selected if c in X.columns]]

        # If some features are missing, that's okay - we still use what we have
        # The PCA will handle the dimension mismatch

        # Stage 2: Scaling
        X_scaled, _ = self.scaler.fit_transform(X_filtered)

        # Stage 3: Factor extraction
        factors = self.factor_extractor.transform(X_scaled)

        # Stage 4: Regime proxy and interactions
        if y_for_regime is not None:
            rolling_vol = self.regime_proxy.compute_rolling_volatility(
                y_for_regime, self.config.regime_window
            )
            regime_p_full = self.regime_proxy.compute_percentile_rank(rolling_vol, rolling_vol)
            # For single prediction, get the regime proxy for the test date
            regime_p = regime_p_full.iloc[[-1]]  # Last value corresponds to test date
        else:
            # Use training distribution
            regime_p = pd.Series([0.5], index=X.index)

        X_final = self.regime_proxy.create_interaction_terms(factors, regime_p)
        X_final = X_final.fillna(0)

        # Add regime features if regime detection was enabled during training
        if self.config.use_regime_detection and self.state.regime_detector is not None:
            try:
                # Detect current regime
                if y_for_regime is not None:
                    current_regime = self.state.regime_detector.get_current_regime(y_for_regime)
                else:
                    current_regime = 'unknown'

                # Get regime features from training state
                if self.state.regime_features is not None:
                    # Add last regime feature value for current prediction
                    for col in self.state.regime_features.columns:
                        if col not in X_final.columns:
                            X_final[col] = self.state.regime_features[col].iloc[-1]

                logger.debug(f"Current regime: {current_regime}")

            except Exception as e:
                logger.warning(f"Could not add regime features: {e}")

        # Use the exact feature names from training
        expected_features = getattr(self, '_training_feature_names', None)
        if expected_features is None:
            # Fallback: rebuild expected features
            feature_names = [c for col_list in self.state.selected_features.values() for c in col_list]
            n_factors = getattr(self, '_n_factors_actual', self.config.n_factors)
            factor_names = [f'factor_{i}' for i in range(n_factors)]
            interaction_names = [f'{name}_interaction' for name in factor_names]
            expected_features = feature_names + factor_names + interaction_names

        # Add missing columns and align order
        for col in expected_features:
            if col not in X_final.columns:
                X_final[col] = 0

        # Ensure correct order and columns
        final_cols = [c for c in expected_features if c in X_final.columns]
        X_final = X_final[final_cols]

        # If we have extra columns (shouldn't happen but handle it)
        if len(X_final.columns) < len(expected_features):
            # Try to match based on number of columns
            X_final = X_final.reindex(columns=expected_features, fill_value=0)

        # Predict
        try:
            predictions = self.model.predict(X_final.values)
        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
            # Return zero predictions as fallback
            predictions = np.zeros(len(X))

        # Apply prediction scaling
        if self.config.prediction_scale != 1.0:
            predictions = predictions * self.config.prediction_scale
            logger.debug(f"Applied prediction scale: {self.config.prediction_scale}")

        return pd.Series(predictions, index=X.index, name='prediction')

    def compute_tc_factor(self, signal_magnitude: float = 1.0) -> float:
        """
        Compute transaction cost factor for signal adjustment.

        Args:
            signal_magnitude: Estimated signal magnitude (0-1 scale)

        Returns:
            TC factor (0-1) representing signal reduction due to TC
        """
        if not self.config.include_tc:
            return 1.0

        # Get TC rate based on account tier
        tc_rate = TCConfig.get_tc_rate(self.config.account_tier)

        # TC factor = exp(-turnover * tc_rate * signal_magnitude)
        # This naturally reduces signal more when:
        # - Turnover is high
        # - TC rate is high
        # - Signal magnitude is high (larger trades)
        tc_factor = np.exp(-self.config.expected_turnover * tc_rate / 10000 * signal_magnitude)

        # Alternatively, use simple linear reduction
        # tc_factor = 1.0 - min(self.config.expected_turnover * tc_rate / 10000, 0.5)

        return max(tc_factor, 0.5)  # Don't reduce signal by more than 50%

    def predict_with_tc(self, X: pd.DataFrame, y_for_regime: Optional[pd.Series] = None) -> pd.Series:
        """
        Generate TC-adjusted predictions.

        Args:
            X: Feature DataFrame for prediction
            y_for_regime: Returns series for regime proxy computation

        Returns:
            TC-adjusted predictions series
        """
        # Get raw predictions
        raw_predictions = self.predict(X, y_for_regime)

        if not self.config.include_tc:
            return raw_predictions

        # Compute signal magnitude (relative to prediction std)
        signal_magnitude = raw_predictions.abs() / (raw_predictions.std() + 1e-8)
        signal_magnitude = signal_magnitude.clip(0, 1)  # Normalize to [0, 1]

        # Apply TC factor to each prediction
        tc_adjusted = []
        for i in range(len(raw_predictions)):
            tc_factor = self.compute_tc_factor(signal_magnitude.iloc[i])
            tc_adjusted.append(raw_predictions.iloc[i] * tc_factor)

        return pd.Series(tc_adjusted, index=raw_predictions.index, name='tc_adjusted_prediction')

    def predict_high_conviction(
        self,
        X: pd.DataFrame,
        y_for_regime: Optional[pd.Series] = None,
        threshold: float = None
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate high-conviction predictions only.

        Only activates positions when signal exceeds the conviction threshold.
        This reduces turnover and transaction costs.

        Args:
            X: Feature DataFrame for prediction
            y_for_regime: Returns series for regime proxy computation
            threshold: Conviction threshold (0-1). Uses config if None.

        Returns:
            Tuple of (high_conviction_predictions, conviction_scores)
            - predictions: Signals only where conviction >= threshold
            - conviction_scores: Signal strength scores (0-1)
        """
        # Get raw predictions
        raw_predictions = self.predict(X, y_for_regime)

        # Compute conviction scores (normalized signal strength)
        pred_std = raw_predictions.std()
        if pred_std > 1e-8:
            conviction_scores = raw_predictions.abs() / pred_std
        else:
            conviction_scores = raw_predictions.abs()

        # Normalize to [0, 1]
        conviction_scores = conviction_scores.clip(0, 1)

        # Use config threshold if not specified
        if threshold is None:
            threshold = self.config.min_conviction_threshold

        # Apply conviction filter
        high_conviction_predictions = raw_predictions.copy()
        high_conviction_predictions[conviction_scores < threshold] = 0

        # Log statistics
        n_signals = (conviction_scores >= threshold).sum()
        pct_signals = n_signals / len(conviction_scores) * 100
        logger.info(
            f"High-conviction filter: {n_signals}/{len(conviction_scores)} "
            f"({pct_signals:.1f}%) signals above threshold {threshold}"
        )

        return high_conviction_predictions, conviction_scores


def nested_time_series_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_class: type,
    n_splits: int = 5,
    train_size: int = 120
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    """
    Perform nested time series cross-validation.

    Args:
        X: Feature DataFrame
        y: Target series
        n_splits: Number of CV splits
        train_size: Minimum training window size

    Returns:
        List of (train_X, val_X, train_y, val_y) tuples
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=12)
    splits = list(tscv.split(X))

    # Adjust for minimum training size
    valid_splits = []
    for train_idx, val_idx in splits:
        if len(train_idx) >= train_size:
            valid_splits.append((
                X.iloc[train_idx],
                X.iloc[val_idx],
                y.iloc[train_idx],
                y.iloc[val_idx]
            ))

    return valid_splits


# =============================================================================
# Utility Functions
# =============================================================================
def compute_factor_returns(
    factors: pd.DataFrame,
    returns: pd.Series
) -> pd.DataFrame:
    """
    Compute returns attributed to each factor.

    Args:
        factors: Factor DataFrame
        returns: Actual returns

    Returns:
        Factor-attributed returns
    """
    # Compute correlation of each factor with returns
    correlations = factors.corrwith(returns)

    # Normalize factors
    factors_norm = (factors - factors.mean()) / factors.std()

    # Factor returns as product of factor exposure and correlation
    factor_returns = factors_norm.mul(correlations, axis=1)

    return factor_returns


def evaluate_factor_importance(
    model: SSRFModel
) -> pd.DataFrame:
    """
    Evaluate importance of extracted factors.

    Args:
        model: Fitted SSRF model

    Returns:
        DataFrame with factor importance metrics
    """
    if model.state is None:
        return pd.DataFrame()

    importance = pd.DataFrame({
        'feature': model.state.selected_features.get('factors', []),
        'coefficient': model.state.coefficients[:len(model.state.selected_features.get('factors', []))]
    })

    importance['abs_coefficient'] = importance['coefficient'].abs()
    importance = importance.sort_values('abs_coefficient', ascending=False)

    return importance


if __name__ == "__main__":
    # Example usage
    print("SSRF Model - Sample Usage")
    print("=" * 50)

    # Generate sample data
    from .fred_data import generate_sample_data

    indicators, target = generate_sample_data(n_periods=300, n_indicators=50)

    # Define groups
    groups = {
        'output_income': [c for c in indicators.columns if 'output' in c],
        'labor': [c for c in indicators.columns if 'labor' in c],
        'inflation': [c for c in indicators.columns if 'inflation' in c],
        'interest': [c for c in indicators.columns if 'interest' in c],
        'sentiment': [c for c in indicators.columns if 'sentiment' in c]
    }

    # Initialize and fit model
    config = SSRFConfig(t_stat_threshold=1.5, n_factors=5)
    model = SSRFModel(config)

    # Split data
    train_size = 200
    X_train, X_test = indicators.iloc[:train_size], indicators.iloc[train_size:]
    y_train, y_test = target.iloc[:train_size], target.iloc[train_size:]

    # Fit
    model.fit(X_train, y_train, groups)

    # Predict
    predictions = model.predict(X_test, y_test)

    print(f"\nPrediction statistics:")
    print(f"  Mean: {predictions.mean():.4f}")
    print(f"  Std:  {predictions.std():.4f}")
    print(f"  Min:  {predictions.min():.4f}")
    print(f"  Max:  {predictions.max():.4f}")
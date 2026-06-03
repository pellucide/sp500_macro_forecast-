"""
Market Regime Detection Module
Provides state-dependent regime classification for the SSRF architecture.
Supports multiple regime detection methods including HMM, volatility-based, and trend-based regimes.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

from . import ensure_series

logger = logging.getLogger(__name__)


class RegimeType(Enum):
    """Enumeration of possible market regimes."""
    BULL = "bull"
    BEAR = "bear"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    CONSOLIDATION = "consolidation"
    RECOVERY = "recovery"
    UNKNOWN = "unknown"


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""
    # Volatility-based regime thresholds
    vol_high_percentile: float = 75.0  # High volatility threshold (percentile)
    vol_low_percentile: float = 25.0   # Low volatility threshold (percentile)

    # Trend detection
    trend_window: int = 12  # Months for trend calculation
    trend_threshold: float = 0.0  # Threshold for trend vs mean reversion

    # Drawdown detection
    drawdown_threshold: float = -0.10  # 10% drawdown triggers bear regime
    recovery_threshold: float = 0.10   # 10% recovery from trough triggers recovery regime

    # HMM configuration
    n_regimes: int = 3  # Number of hidden regimes for HMM
    use_hmm: bool = True

    # Regime smoothing
    min_regime_duration: int = 3  # Minimum months in a regime (prevent flickering)


class VolatilityRegimeDetector:
    """
    Simple volatility-based regime detection.

    Classifies regimes based on rolling volatility percentiles:
    - High Volatility: Volatility > 75th percentile
    - Low Volatility: Volatility < 25th percentile
    - Normal: Between 25th and 75th percentile
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self.training_vol_percentiles = None

    def fit(self, returns: pd.Series) -> 'VolatilityRegimeDetector':
        """
        Fit the detector on training data.

        Args:
            returns: Training return series

        Returns:
            self
        """
        returns = ensure_series(returns, "returns")

        # Compute rolling volatility
        vol = returns.rolling(window=12).std()

        # Compute percentile thresholds from training data
        vol_clean = vol.dropna()
        self.training_vol_percentiles = {
            'high': np.percentile(vol_clean, self.config.vol_high_percentile),
            'low': np.percentile(vol_clean, self.config.vol_low_percentile),
            'median': np.percentile(vol_clean, 50)
        }

        logger.info(f"Fitted volatility regime detector. "
                   f"High threshold: {self.training_vol_percentiles['high']:.4f}, "
                   f"Low threshold: {self.training_vol_percentiles['low']:.4f}")

        return self

    def detect(self, returns: pd.Series) -> pd.Series:
        """
        Detect regimes from return series.

        Args:
            returns: Return series

        Returns:
            Series with regime labels
        """
        vol = returns.rolling(window=12).std()

        regimes = pd.Series(index=returns.index, dtype=object)

        # Classify based on thresholds
        high_mask = vol > self.training_vol_percentiles['high']
        low_mask = vol < self.training_vol_percentiles['low']

        regimes[high_mask] = RegimeType.HIGH_VOLATILITY.value
        regimes[low_mask] = RegimeType.LOW_VOLATILITY.value
        regimes[~(high_mask | low_mask)] = RegimeType.CONSOLIDATION.value

        return regimes


class TrendRegimeDetector:
    """
    Trend-based regime detection using moving averages.

    Classifies regimes based on price momentum:
    - Trend Up: Price above moving average with positive momentum
    - Trend Down: Price below moving average with negative momentum
    - Consolidation: Price near moving average
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self.training_momentum = None

    def fit(self, returns: pd.Series) -> 'TrendRegimeDetector':
        """
        Fit the detector on training data.

        Args:
            returns: Training return series

        Returns:
            self
        """
        returns = ensure_series(returns, "returns")

        # Compute momentum
        momentum = returns.rolling(window=self.config.trend_window).sum()

        # Store training distribution
        self.training_momentum = momentum.dropna()
        self.momentum_threshold = self.config.trend_threshold

        return self

    def detect(self, returns: pd.Series) -> pd.Series:
        """
        Detect regimes from return series.

        Args:
            returns: Return series

        Returns:
            Series with regime labels
        """
        momentum = returns.rolling(window=self.config.trend_window).sum()

        regimes = pd.Series(index=returns.index, dtype=object)

        # Classify based on momentum
        regimes[momentum > self.momentum_threshold] = RegimeType.TREND_UP.value
        regimes[momentum < -self.momentum_threshold] = RegimeType.TREND_DOWN.value
        regimes[(momentum >= -self.momentum_threshold) &
                (momentum <= self.momentum_threshold)] = RegimeType.CONSOLIDATION.value

        return regimes


class DrawdownRegimeDetector:
    """
    Drawdown-based regime detection.

    Classifies regimes based on drawdown from peak:
    - Bear: Drawdown > threshold
    - Recovery: Rising from trough
    - Normal: Within normal range
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self.peak = None

    def fit(self, returns: pd.Series) -> 'DrawdownRegimeDetector':
        """
        Fit the detector on training data.

        Args:
            returns: Training return series

        Returns:
            self
        """
        returns = ensure_series(returns, "returns")

        # Compute cumulative returns
        cumulative = (1 + returns).cumprod()

        # Compute drawdown from peak
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak

        self.max_drawdown_training = drawdown.min()
        self.peak_level = 1.0

        return self

    def detect(self, returns: pd.Series) -> pd.Series:
        """
        Detect regimes from return series.

        Args:
            returns: Return series

        Returns:
            Series with regime labels
        """
        cumulative = (1 + returns).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak

        regimes = pd.Series(index=returns.index, dtype=object)

        # Classify based on drawdown
        regimes[drawdown < -abs(self.config.drawdown_threshold)] = RegimeType.BEAR.value
        regimes[drawdown >= -abs(self.config.drawdown_threshold)] = RegimeType.CONSOLIDATION.value

        return regimes


class HiddenMarkovRegimeDetector:
    """
    Hidden Markov Model (HMM) based regime detection.

    Uses Gaussian HMM to detect hidden market regimes.
    States typically correspond to:
    - Bull market (low volatility, positive returns)
    - Bear market (high volatility, negative returns)
    - Transition/Recovery (moderate volatility)
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self.model = None
        self.trained_states = None

    def fit(self, returns: pd.Series) -> 'HiddenMarkovRegimeDetector':
        """
        Fit HMM on training data.

        Args:
            returns: Training return series

        Returns:
            self
        """
        returns = ensure_series(returns, "returns")

        try:
            from hmmlearn import hmm

            # Prepare data - need 2D array (n_samples, n_features)
            X_raw = returns.values
            valid_mask = ~np.isnan(X_raw)
            X = X_raw[valid_mask].reshape(-1, 1)  # Keep 2D shape for hmmlearn

            if len(X) < 50:
                logger.warning("Insufficient data for HMM training. Using fallback.")
                return self

            # Fit Gaussian HMM
            self.model = hmm.GaussianHMM(
                n_components=self.config.n_regimes,
                covariance_type="full",
                n_iter=1000,
                random_state=42
            )

            self.model.fit(X)

            # Store X for use in _analyze_states
            self._X_train = X

            # Analyze fitted states
            self.trained_states = self._analyze_states(returns)

            logger.info(f"Fitted HMM with {self.config.n_regimes} regimes")

        except ImportError:
            logger.warning("hmmlearn not available. HMM regime detection disabled.")
            self.config.use_hmm = False
        except Exception as e:
            logger.warning(f"HMM fitting failed: {e}. Using fallback.")
            self.config.use_hmm = False

        return self

    def _analyze_states(self, returns: pd.Series) -> Dict[int, Dict]:
        """Analyze HMM states to determine regime characteristics."""
        if self.model is None:
            return {}

        # Use stored X from fit() if available, otherwise create it
        if hasattr(self, '_X_train') and self._X_train is not None:
            X = self._X_train
        else:
            X_raw = returns.values
            valid_mask = ~np.isnan(X_raw)
            X = X_raw[valid_mask].reshape(-1, 1)

        states = self.model.predict(X)

        state_stats = {}
        for state in range(self.config.n_regimes):
            mask = states == state
            state_returns = X[mask, 0]  # Get column 0 of 2D array

            state_stats[state] = {
                'mean': np.mean(state_returns),
                'std': np.std(state_returns),
                'count': len(state_returns)
            }

        # Classify states based on characteristics
        # Sort by mean return (descending)
        sorted_states = sorted(state_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
        n_states = len(sorted_states)

        regime_mapping = {}
        for rank, (state, stats) in enumerate(sorted_states):
            if rank == 0:
                regime_mapping[state] = RegimeType.BULL.value
            elif rank == n_states - 1:
                regime_mapping[state] = RegimeType.BEAR.value
            elif n_states == 3:
                # Only use consolidation label for 3-regime case
                regime_mapping[state] = RegimeType.CONSOLIDATION.value
            else:
                # For n != 3 regimes, label intermediate states by volatility
                # High volatility intermediate states closer to BEAR behavior
                vol = stats['std']
                vol_threshold = np.mean([sorted_states[0][1]['std'], sorted_states[-1][1]['std']])
                regime_mapping[state] = RegimeType.BEAR.value if vol > vol_threshold else RegimeType.BULL.value

        return regime_mapping

    def detect(self, returns: pd.Series) -> pd.Series:
        """
        Detect regimes using HMM.

        Args:
            returns: Return series

        Returns:
            Series with regime labels
        """
        if self.model is None or not self.config.use_hmm:
            # Fallback to simple regime detection
            return self._fallback_detection(returns)

        try:
            X_raw = returns.values
            valid_mask = ~np.isnan(X_raw)
            X = X_raw[valid_mask].reshape(-1, 1)

            states = self.model.predict(X)

            regimes = pd.Series(index=returns.index, dtype=object)
            valid_idx = returns.index[valid_mask]

            for i, (idx, state) in enumerate(zip(valid_idx, states)):
                if state in self.trained_states:
                    regimes[idx] = self.trained_states[state]
                else:
                    regimes[idx] = RegimeType.UNKNOWN.value

            return regimes

        except Exception as e:
            logger.warning(f"HMM detection failed: {e}. Using fallback.")
            return self._fallback_detection(returns)

    def _fallback_detection(self, returns: pd.Series) -> pd.Series:
        """Fallback regime detection when HMM is unavailable."""
        vol = returns.rolling(window=12).std()
        vol_median = vol.median()

        regimes = pd.Series(index=returns.index, dtype=object)

        # Simple volatility-based classification
        regimes[vol > vol_median] = RegimeType.HIGH_VOLATILITY.value
        regimes[vol <= vol_median] = RegimeType.LOW_VOLATILITY.value

        return regimes


class MarketRegimeDetector:
    """
    Combined Market Regime Detector

    Integrates multiple regime detection methods and provides
    a unified regime classification for the SSRF architecture.
    """

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()

        # Initialize individual detectors
        self.vol_detector = VolatilityRegimeDetector(config)
        self.trend_detector = TrendRegimeDetector(config)
        self.drawdown_detector = DrawdownRegimeDetector(config)
        self.hmm_detector = HiddenMarkovRegimeDetector(config)

        self.is_fitted = False
        self.current_regime = RegimeType.UNKNOWN

    def fit(self, returns: pd.Series) -> 'MarketRegimeDetector':
        """
        Fit all regime detectors on training data.

        Args:
            returns: Training return series

        Returns:
            self
        """
        logger.info("Fitting regime detectors...")

        self.vol_detector.fit(returns)
        self.trend_detector.fit(returns)
        self.drawdown_detector.fit(returns)
        self.hmm_detector.fit(returns)

        self.is_fitted = True

        logger.info("Regime detectors fitted successfully")

        return self

    def detect(self, returns: pd.Series) -> pd.Series:
        """
        Detect market regimes using combined methods.

        Args:
            returns: Return series

        Returns:
            Series with regime labels
        """
        if not self.is_fitted:
            raise ValueError("Regime detector must be fitted before detection")

        # Get individual regime classifications
        vol_regimes = self.vol_detector.detect(returns)
        trend_regimes = self.trend_detector.detect(returns)

        if self.config.use_hmm:
            hmm_regimes = self.hmm_detector.detect(returns)
        else:
            hmm_regimes = vol_regimes  # Fallback to volatility

        # Combine regimes using voting/scoring
        combined_regimes = self._combine_regimes(
            vol_regimes, trend_regimes, hmm_regimes
        )

        # Apply smoothing to prevent regime flickering
        combined_regimes = self._smooth_regimes(combined_regimes)

        return combined_regimes

    def _combine_regimes(
        self,
        vol_regimes: pd.Series,
        trend_regimes: pd.Series,
        hmm_regimes: pd.Series
    ) -> pd.Series:
        """
        Combine regime classifications using weighted voting.

        Primary weight goes to HMM, secondary to volatility.
        """
        combined = pd.Series(index=vol_regimes.index, dtype=object)

        for idx in vol_regimes.index:
            votes = {
                RegimeType.BULL.value: 0,
                RegimeType.BEAR.value: 0,
                RegimeType.HIGH_VOLATILITY.value: 0,
                RegimeType.LOW_VOLATILITY.value: 0,
                RegimeType.TREND_UP.value: 0,
                RegimeType.TREND_DOWN.value: 0,
                RegimeType.CONSOLIDATION.value: 0,
                RegimeType.RECOVERY.value: 0,
                RegimeType.UNKNOWN.value: 0
            }

            # Count votes
            if vol_regimes[idx] in votes:
                votes[vol_regimes[idx]] += 1
            if trend_regimes[idx] in votes:
                votes[trend_regimes[idx]] += 1
            if hmm_regimes[idx] in votes:
                votes[hmm_regimes[idx]] += 2  # Double weight for HMM

            # Select most voted regime
            combined[idx] = max(votes, key=votes.get)

        return combined

    def _smooth_regimes(self, regimes: pd.Series) -> pd.Series:
        """
        Apply smoothing to prevent regime flickering.

        Replaces short regimes with the surrounding regime.
        """
        if len(regimes) < self.config.min_regime_duration:
            return regimes

        smoothed = regimes.copy()
        regime_counts = {}

        # Count consecutive regimes
        current_regime = regimes.iloc[0]
        current_count = 1
        i = 0  # ensure `i` is defined for final-regime check below

        for i in range(1, len(regimes)):
            if regimes.iloc[i] == current_regime:
                current_count += 1
            else:
                # Check if previous regime was too short
                if current_count < self.config.min_regime_duration:
                    # Find next different regime
                    next_regime = None
                    for j in range(i, len(regimes)):
                        if regimes.iloc[j] != current_regime:
                            next_regime = regimes.iloc[j]
                            break

                    # Replace short regime with next regime
                    if next_regime:
                        for k in range(i - current_count, i):
                            smoothed.iloc[k] = next_regime

                current_regime = regimes.iloc[i]
                current_count = 1

        # Check if final regime was too short (loop only checks on regime transitions)
        if current_count < self.config.min_regime_duration:
            # Find previous different regime to merge with
            prev_regime = None
            for j in range(i - current_count - 1, -1, -1):
                if regimes.iloc[j] != current_regime:
                    prev_regime = regimes.iloc[j]
                    break

            # Replace short final regime with previous regime
            if prev_regime:
                for k in range(i - current_count, i + 1):
                    smoothed.iloc[k] = prev_regime

        return smoothed

    def get_current_regime(self, returns: pd.Series) -> str:
        """
        Get the current (most recent) regime.

        Args:
            returns: Return series

        Returns:
            Current regime label
        """
        regimes = self.detect(returns)
        return regimes.iloc[-1]

    def get_regime_distribution(self, regimes: pd.Series) -> Dict[str, float]:
        """
        Get the distribution of regimes in the series.

        Args:
            regimes: Regime series from detect()

        Returns:
            Dictionary mapping regime to percentage of time
        """
        value_counts = regimes.value_counts()
        total = len(regimes)

        return {
            regime: (count / total) * 100
            for regime, count in value_counts.items()
        }


def create_regime_features(
    returns: pd.Series,
    regimes: pd.Series,
    lookback: int = 12
) -> pd.DataFrame:
    """
    Create regime-conditioned features for the SSRF model.

    Args:
        returns: Return series
        regimes: Regime labels
        lookback: Window for feature computation

    Returns:
        DataFrame with regime features
    """
    features = pd.DataFrame(index=returns.index)

    # Regime indicator variables (one-hot encoding)
    unique_regimes = regimes.unique()

    for regime in unique_regimes:
        features[f'regime_{regime}'] = (regimes == regime).astype(float)

    # Regime persistence (how long in current regime)
    regime_persistence = pd.Series(0.0, index=returns.index)
    current_regime = regimes.iloc[0]
    count = 0

    for i in range(len(regimes)):
        if regimes.iloc[i] == current_regime:
            count += 1
        else:
            current_regime = regimes.iloc[i]
            count = 1
        regime_persistence.iloc[i] = count

    features['regime_persistence'] = regime_persistence

    # Regime volatility metrics
    for regime in unique_regimes:
        mask = regimes == regime
        regime_returns = returns[mask]
        if len(regime_returns) > 0:
            features[f'vol_{regime}'] = returns.rolling(lookback).std() * mask
        else:
            features[f'vol_{regime}'] = 0

    # Time in current regime (normalized)
    features['regime_time_fraction'] = (
        regime_persistence / regime_persistence.rolling(lookback).max()
    )

    return features


# Export main class
__all__ = [
    'RegimeConfig',
    'RegimeType',
    'VolatilityRegimeDetector',
    'TrendRegimeDetector',
    'DrawdownRegimeDetector',
    'HiddenMarkovRegimeDetector',
    'MarketRegimeDetector',
    'create_regime_features'
]
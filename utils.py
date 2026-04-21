"""
utils.py — Validation and helper functions for Form Trend API

Responsibilities:
- Pre-flight validation of input data before regression runs
- Edge case detection (zero variance, all-identical scores)
- Weight generation for exponentially weighted regression
- Rounding helpers for clean response output
"""

from typing import List, Tuple
import math


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DECAY_RATE = 0.85  # Fixed internal hyperparameter — not exposed to users
MIN_SCORES = 3
MAX_SCORES = 30
MIN_SCORE_VALUE = 0.0
MAX_SCORE_VALUE = 100.0
MIN_VARIANCE_EPSILON = 1e-6  # Below this, scores are considered identical

# Trend classification thresholds (on normalized_slope x R-squared)
RISING_THRESHOLD = 0.02    # +2% change per match relative to mean → Rising
DECLINING_THRESHOLD = -0.02  # -2% change per match relative to mean → Declining

# R-squared confidence thresholds
HIGH_CONFIDENCE_R2 = 0.7
MODERATE_CONFIDENCE_R2 = 0.4


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_scores(scores: List[float]) -> None:
    """
    Run all pre-flight checks on the scores list.
    Raises ValueError with a descriptive message on any failure.
    Called by services.py before any math runs.
    """
    if scores is None or len(scores) == 0:
        raise ValueError("Performance scores list cannot be empty.")

    if len(scores) < MIN_SCORES:
        raise ValueError(
            f"Minimum {MIN_SCORES} scores required for regression. "
            f"Received {len(scores)}."
        )

    if len(scores) > MAX_SCORES:
        raise ValueError(
            f"Maximum {MAX_SCORES} scores allowed to keep form trend recent. "
            f"Received {len(scores)}."
        )

    for i, score in enumerate(scores):
        if score is None:
            raise ValueError(f"Score at position {i+1} is None. All scores must be numeric.")
        if math.isnan(score) or math.isinf(score):
            raise ValueError(f"Score at position {i+1} is not a valid number (NaN or Inf).")
        if score < MIN_SCORE_VALUE or score > MAX_SCORE_VALUE:
            raise ValueError(
                f"Score {score} at position {i+1} is out of valid range "
                f"[{MIN_SCORE_VALUE}, {MAX_SCORE_VALUE}]."
            )


def is_zero_variance(scores: List[float]) -> bool:
    """
    Returns True if all scores are effectively identical (variance < epsilon).
    In this case regression is undefined (division by zero in SS_total).
    The service layer handles this as a special 'Stable' early return.
    """
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    return variance < MIN_VARIANCE_EPSILON


# ---------------------------------------------------------------------------
# Weight Generation
# ---------------------------------------------------------------------------

def generate_exponential_weights(n: int, decay_rate: float) -> List[float]:
    """
    Generate exponential decay weights for n matches.

    The most recent match (index n-1) gets weight = 1.0.
    Going backwards, each match is multiplied by decay_rate.

    Formula:
        raw_weight[i] = decay_rate ^ (n - 1 - i)   for i = 0, 1, ..., n-1

    Weights are then normalized so they sum to 1.0 (for numerical stability).

    Args:
        n          : number of matches
        decay_rate : float in [0.5, 1.0]. 1.0 = equal weights (plain OLS).
                     0.85 = recommended (recent form weighted more).

    Returns:
        List of n normalized weights, oldest to newest.

    Example (n=5, decay_rate=0.85):
        raw  = [0.85^4, 0.85^3, 0.85^2, 0.85^1, 0.85^0]
             = [0.522,  0.614,  0.722,  0.850,  1.000]
        normalized = each / sum → sums to 1.0
    """
    raw_weights = [decay_rate ** (n - 1 - i) for i in range(n)]
    total = sum(raw_weights)
    normalized = [round(w / total, 6) for w in raw_weights]
    return normalized


# ---------------------------------------------------------------------------
# Weighted Statistics Helpers
# ---------------------------------------------------------------------------

def weighted_mean(values: List[float], weights: List[float]) -> float:
    """
    Compute weighted mean: sum(w_i * v_i) / sum(w_i)
    Weights are assumed normalized (sum = 1), so denominator = 1.
    """
    return sum(w * v for w, v in zip(weights, values))


def weighted_covariance(
    x: List[float],
    y: List[float],
    weights: List[float],
    mean_x: float,
    mean_y: float
) -> float:
    """
    Weighted covariance between x and y:
        Cov_w(x, y) = sum(w_i * (x_i - mean_x) * (y_i - mean_y))
    This is the numerator of the weighted slope formula.
    """
    return sum(
        w * (xi - mean_x) * (yi - mean_y)
        for w, xi, yi in zip(weights, x, y)
    )


def weighted_variance(
    x: List[float],
    weights: List[float],
    mean_x: float
) -> float:
    """
    Weighted variance of x:
        Var_w(x) = sum(w_i * (x_i - mean_x)^2)
    This is the denominator of the weighted slope formula.
    """
    return sum(
        w * (xi - mean_x) ** 2
        for w, xi in zip(weights, x)
    )


# ---------------------------------------------------------------------------
# Classification Helpers
# ---------------------------------------------------------------------------

def classify_trend(trend_score: float) -> str:
    """
    Classify the final trend_score into a label.

    trend_score = normalized_slope x R-squared
    Thresholds represent a 2% per-match change relative to mean score.

    Returns: 'Rising', 'Declining', or 'Stable'
    """
    if trend_score > RISING_THRESHOLD:
        return "Rising"
    elif trend_score < DECLINING_THRESHOLD:
        return "Declining"
    else:
        return "Stable"


def classify_confidence(r_squared: float) -> str:
    """
    Translate R-squared into a human-readable confidence level.

    High     : R2 >= 0.7  → trend line fits data well, interpretation reliable
    Moderate : R2 >= 0.4  → some noise but trend detectable
    Low      : R2 <  0.4  → high variance, trend label should be treated cautiously
    """
    if r_squared >= HIGH_CONFIDENCE_R2:
        return "High"
    elif r_squared >= MODERATE_CONFIDENCE_R2:
        return "Moderate"
    else:
        return "Low"


def build_interpretation(
    player_name: str,
    trend_label: str,
    slope_beta1: float,
    r_squared: float,
    confidence: str,
    n: int
) -> str:
    """
    Build a human-readable interpretation string summarizing the analysis.
    """
    name = player_name if player_name else "The player"
    direction_phrase = {
        "Rising": f"improving at approximately {abs(slope_beta1):.2f} points per match",
        "Declining": f"declining at approximately {abs(slope_beta1):.2f} points per match",
        "Stable": "performing consistently without a significant upward or downward trend",
    }[trend_label]

    return (
        f"{name} is {direction_phrase} over the last {n} matches analyzed. "
        f"The trend line explains {r_squared * 100:.1f}% of the performance variance "
        f"(R-squared = {r_squared:.3f}), indicating {confidence.lower()} confidence "
        f"in this trend direction."
    )


# ---------------------------------------------------------------------------
# Rounding Helper
# ---------------------------------------------------------------------------

def round_float(value: float, decimals: int = 4) -> float:
    """Round a float to specified decimal places for clean API output."""
    return round(value, decimals)

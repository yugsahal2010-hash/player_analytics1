"""
services.py — Core analytics logic for Form Trend API

Scientific Approach: Exponentially Weighted Linear Regression (EWLR)

Why EWLR over plain OLS?
    Plain OLS treats a match from 3 months ago equally to last week's match.
    For "form" — which is inherently about recent trajectory — recent matches
    must carry more weight. EWLR applies exponential decay weights so that
    the regression slope is pulled more strongly by recent performance.

Mathematical Pipeline:
    1. Generate exponential decay weights (recent = higher weight)
    2. Compute weighted means of x (match index) and y (scores)
    3. Compute weighted slope β₁ via weighted least squares formula
    4. Compute weighted intercept β₀
    5. Compute predicted values ŷᵢ from the regression line
    6. Compute SS_total and SS_residual → derive R²
    7. Normalize slope by weighted mean score (scale-independence)
    8. Compute trend_score = normalized_slope × R²
    9. Classify: Rising / Stable / Declining
    10. Classify confidence from R²
    11. Build human-readable interpretation
"""

from typing import List
from schemas import FormTrendRequest, FormTrendResponse, DerivedVariables
from utils import (
    DEFAULT_DECAY_RATE,
    validate_scores,
    is_zero_variance,
    generate_exponential_weights,
    weighted_mean,
    weighted_covariance,
    weighted_variance,
    classify_trend,
    classify_confidence,
    build_interpretation,
    round_float,
)


def compute_form_trend(request: FormTrendRequest) -> FormTrendResponse:
    """
    Main service function. Orchestrates the full analytical pipeline.

    Args:
        request : FormTrendRequest — validated Pydantic model from the endpoint

    Returns:
        FormTrendResponse — structured response with all metrics and derived variables
    """

    scores: List[float] = request.performance_scores
    decay_rate: float = DEFAULT_DECAY_RATE  # Fixed at 0.85 — not user-facing
    n: int = len(scores)

    # ------------------------------------------------------------------
    # Step 0: Pre-flight validation
    # ------------------------------------------------------------------
    validate_scores(scores)

    # ------------------------------------------------------------------
    # Step 1: Handle zero-variance edge case
    # All scores identical → no trend possible, return Stable immediately
    # ------------------------------------------------------------------
    if is_zero_variance(scores):
        mean_score = scores[0]
        flat_weights = [round_float(1.0 / n, 6)] * n
        x_indices = list(range(1, n + 1))

        derived = DerivedVariables(
            x_indices=x_indices,
            weights=flat_weights,
            weighted_mean_x=round_float(weighted_mean([float(x) for x in x_indices], flat_weights)),
            weighted_mean_y=round_float(mean_score),
            slope_beta1=0.0,
            intercept_beta0=round_float(mean_score),
            ss_total=0.0,
            ss_residual=0.0,
            r_squared=1.0,  # Perfect fit — line IS the data
            normalized_slope=0.0,
        )

        return FormTrendResponse(
            player_id=request.player_id,
            player_name=request.player_name,
            trend_label="Stable",
            trend_score=0.0,
            slope_beta1=0.0,
            r_squared=1.0,
            confidence="High",
            num_matches_used=n,
            derived_variables=derived,
            interpretation=(
                f"{request.player_name or 'The player'} has posted identical scores "
                f"across all {n} matches. Performance is perfectly stable with no trend."
            ),
        )

    # ------------------------------------------------------------------
    # Step 2: Build time axis
    # x = [1, 2, 3, ..., n] — match indices (oldest=1, most recent=n)
    # ------------------------------------------------------------------
    x: List[float] = [float(i) for i in range(1, n + 1)]
    y: List[float] = scores

    # ------------------------------------------------------------------
    # Step 3: Generate exponential decay weights
    # Most recent match → weight closest to 1.0
    # decay_rate=0.85 means each older match is 85% as important as the next
    # ------------------------------------------------------------------
    weights: List[float] = generate_exponential_weights(n, decay_rate)

    # ------------------------------------------------------------------
    # Step 4: Compute weighted means
    # mean_x = Σ(wᵢ × xᵢ),  mean_y = Σ(wᵢ × yᵢ)
    # ------------------------------------------------------------------
    mean_x: float = weighted_mean(x, weights)
    mean_y: float = weighted_mean(y, weights)

    # ------------------------------------------------------------------
    # Step 5: Compute weighted slope β₁
    #
    # Weighted Least Squares formula:
    #   β₁ = Σ[wᵢ(xᵢ - mean_x)(yᵢ - mean_y)] / Σ[wᵢ(xᵢ - mean_x)²]
    #
    # Numerator   = weighted covariance of x and y
    # Denominator = weighted variance of x
    # ------------------------------------------------------------------
    cov_xy: float = weighted_covariance(x, y, weights, mean_x, mean_y)
    var_x: float = weighted_variance(x, weights, mean_x)

    if var_x < 1e-10:
        # Should never happen with distinct x indices, but guard anyway
        raise ValueError("Weighted variance of x indices is near zero. Cannot compute slope.")

    beta1: float = cov_xy / var_x

    # ------------------------------------------------------------------
    # Step 6: Compute intercept β₀
    #   β₀ = mean_y - β₁ × mean_x
    # ------------------------------------------------------------------
    beta0: float = mean_y - beta1 * mean_x

    # ------------------------------------------------------------------
    # Step 7: Compute predicted values ŷᵢ = β₀ + β₁ × xᵢ
    # ------------------------------------------------------------------
    y_predicted: List[float] = [beta0 + beta1 * xi for xi in x]

    # ------------------------------------------------------------------
    # Step 8: Compute R²
    #
    #   SS_total    = Σ wᵢ(yᵢ - mean_y)²   ← total weighted variance
    #   SS_residual = Σ wᵢ(yᵢ - ŷᵢ)²       ← unexplained weighted variance
    #   R²          = 1 - SS_residual / SS_total
    #
    # R² tells us: "what fraction of the score variance is explained by
    # the linear time trend?" High R² = trend is consistent and reliable.
    # ------------------------------------------------------------------
    ss_total: float = sum(
        w * (yi - mean_y) ** 2
        for w, yi in zip(weights, y)
    )
    ss_residual: float = sum(
        w * (yi - yhat) ** 2
        for w, yi, yhat in zip(weights, y, y_predicted)
    )

    # Guard: ss_total should be > 0 here (zero variance caught above)
    r_squared: float = 1.0 - (ss_residual / ss_total) if ss_total > 1e-10 else 1.0

    # Clamp R² to [0, 1] — floating point edge cases can produce tiny negatives
    r_squared = max(0.0, min(1.0, r_squared))

    # ------------------------------------------------------------------
    # Step 9: Normalize slope
    #
    #   normalized_slope = β₁ / mean_y
    #
    # This converts the raw slope into a fraction of average score,
    # making it comparable across players and scoring systems.
    # e.g., normalized_slope = 0.03 means +3% improvement per match.
    # ------------------------------------------------------------------
    normalized_slope: float = beta1 / mean_y if abs(mean_y) > 1e-10 else 0.0

    # ------------------------------------------------------------------
    # Step 10: Compute final trend_score
    #
    #   trend_score = normalized_slope × R²
    #
    # This is the key composite metric. It penalizes a high slope that
    # comes from noisy/inconsistent data (low R²). A player who jumps
    # 60→90→40→95 has a high slope but low R² → trend_score stays modest.
    # ------------------------------------------------------------------
    trend_score: float = normalized_slope * r_squared

    # ------------------------------------------------------------------
    # Step 11: Classify trend label and confidence
    # ------------------------------------------------------------------
    trend_label: str = classify_trend(trend_score)
    confidence: str = classify_confidence(r_squared)

    # ------------------------------------------------------------------
    # Step 12: Build interpretation
    # ------------------------------------------------------------------
    interpretation: str = build_interpretation(
        player_name=request.player_name,
        trend_label=trend_label,
        slope_beta1=beta1,
        r_squared=r_squared,
        confidence=confidence,
        n=n,
    )

    # ------------------------------------------------------------------
    # Step 13: Assemble response
    # ------------------------------------------------------------------
    derived = DerivedVariables(
        x_indices=list(range(1, n + 1)),
        weights=[round_float(w, 6) for w in weights],
        weighted_mean_x=round_float(mean_x),
        weighted_mean_y=round_float(mean_y),
        slope_beta1=round_float(beta1),
        intercept_beta0=round_float(beta0),
        ss_total=round_float(ss_total),
        ss_residual=round_float(ss_residual),
        r_squared=round_float(r_squared),
        normalized_slope=round_float(normalized_slope),
    )

    return FormTrendResponse(
        player_id=request.player_id,
        player_name=request.player_name,
        trend_label=trend_label,
        trend_score=round_float(trend_score),
        slope_beta1=round_float(beta1),
        r_squared=round_float(r_squared),
        confidence=confidence,
        num_matches_used=n,
        derived_variables=derived,
        interpretation=interpretation,
    )

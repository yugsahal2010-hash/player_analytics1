from schemas import FormTrendRequest, FormTrendResponse, TrendDetails
from utils import (
    exponential_weights,
    weighted_mean,
    weighted_variance,
    weighted_covariance,
    classify_trend,
    classify_confidence,
)


def compute_form_trend(request: FormTrendRequest):
    scores = request.performance_scores
    n = len(scores)

    x = list(range(1, n + 1))
    weights = exponential_weights(n)

    mean_x = weighted_mean(x, weights)
    mean_y = weighted_mean(scores, weights)

    var_x = weighted_variance(x, weights, mean_x)
    cov_xy = weighted_covariance(x, scores, weights, mean_x, mean_y)

    slope = cov_xy / var_x if var_x != 0 else 0
    intercept = mean_y - slope * mean_x

    predicted = [intercept + slope * xi for xi in x]

    ss_total = sum(w * (y - mean_y) ** 2 for y, w in zip(scores, weights))
    ss_residual = sum(w * (y - yhat) ** 2 for y, yhat, w in zip(scores, predicted, weights))

    r_squared = 1 - (ss_residual / ss_total) if ss_total != 0 else 1
    r_squared = max(0, min(1, r_squared))

    normalized_slope = slope / mean_y if mean_y != 0 else 0
    trend_score = normalized_slope * r_squared

    trend_label = classify_trend(trend_score)
    confidence = classify_confidence(r_squared)

    player_name = request.player_name or "The player"

    interpretation = (
        f"{player_name} is {trend_label.lower()} "
        f"with {confidence.lower()} confidence over the last {n} matches."
    )

    return FormTrendResponse(
        player_id=request.player_id,
        player_name=request.player_name,
        trend_label=trend_label,
        trend_score=round(trend_score, 4),
        slope=round(slope, 4),
        r_squared=round(r_squared, 4),
        confidence=confidence,
        matches_used=n,
        details=TrendDetails(
            weights=[round(w, 4) for w in weights],
            slope=round(slope, 4),
            intercept=round(intercept, 4),
            r_squared=round(r_squared, 4),
            normalized_slope=round(normalized_slope, 4),
        ),
        interpretation=interpretation,
    )

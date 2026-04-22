def exponential_weights(n, decay=0.85):
    raw = [decay ** (n - 1 - i) for i in range(n)]
    total = sum(raw)
    return [w / total for w in raw]


def weighted_mean(values, weights):
    return sum(v * w for v, w in zip(values, weights))


def weighted_variance(values, weights, mean):
    return sum(w * (v - mean) ** 2 for v, w in zip(values, weights))


def weighted_covariance(x, y, weights, mean_x, mean_y):
    return sum(
        w * (xi - mean_x) * (yi - mean_y)
        for xi, yi, w in zip(x, y, weights)
    )


def classify_trend(score):
    if score > 0.02:
        return "Rising"
    elif score < -0.02:
        return "Declining"
    return "Stable"


def classify_confidence(r_squared):
    if r_squared >= 0.7:
        return "High"
    elif r_squared >= 0.4:
        return "Moderate"
    return "Low"

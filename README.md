# Form Trend API — Handover Documentation

## API Name
**Form Trend (Regression)**
Route: `POST /api/v1/player/form-trend`

---

## Objective
This API identifies whether a player's recent performance is **Rising**, **Stable**, or **Declining** over time. It does not rely on simple averages. Instead, it fits a weighted regression line through the player's ordered match scores, where recent matches carry more weight than older ones. The output is a composite trend score that combines both the **direction** (slope) and the **consistency** (R-squared) of the trend.

---

## Scientific Principle Used

**Exponentially Weighted Linear Regression (EWLR)**

Plain Ordinary Least Squares (OLS) regression assigns equal importance to all data points. For sports form analysis, this is inappropriate — a match from 3 months ago should not weigh as much as last week's match.

EWLR solves this by assigning exponentially decaying weights to matches:

```
weight[i] = decay_rate ^ (n - 1 - i)     (oldest = lowest weight)
```

The weighted slope formula:

```
β₁ = Σ[wᵢ(xᵢ - x̄_w)(yᵢ - ȳ_w)] / Σ[wᵢ(xᵢ - x̄_w)²]
```

The final metric:

```
trend_score = (β₁ / ȳ_w) × R²
```

This penalizes noisy/inconsistent trends and rewards consistent directional movement.

---

## Input Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `player_id` | string | Yes | Unique player identifier |
| `player_name` | string | No | Display name (used in interpretation text) |
| `performance_scores` | List[float] | Yes | Ordered scores oldest→newest. Range [0,100]. Min 3, max 30. |
| `decay_rate` | float | No | Exponential decay rate [0.5–1.0]. Default: 0.85. Use 1.0 for plain OLS. |

---

## Derived Variables and Their Meaning

| Variable | Formula | Meaning |
|---|---|---|
| `x_indices` | [1, 2, ..., n] | Time axis — match number in sequence |
| `weights` | decay_rate^(n-1-i), normalized | How much each match contributes to regression |
| `weighted_mean_x` | Σ(wᵢ × xᵢ) | Weighted center of match indices |
| `weighted_mean_y` | Σ(wᵢ × yᵢ) | Weighted average score (recent-biased) |
| `slope_beta1` | Weighted covariance(x,y) / Weighted variance(x) | Raw points gained/lost per match |
| `intercept_beta0` | mean_y - β₁ × mean_x | Where the regression line crosses x=0 |
| `ss_total` | Σ wᵢ(yᵢ - mean_y)² | Total weighted variance in scores |
| `ss_residual` | Σ wᵢ(yᵢ - ŷᵢ)² | Variance unexplained by the trend line |
| `r_squared` | 1 - SS_residual / SS_total | Consistency/reliability of the trend (0–1) |
| `normalized_slope` | β₁ / mean_y | Slope as % of average score — scale-independent |

---

## Final Output Fields

| Field | Type | Meaning |
|---|---|---|
| `trend_label` | string | "Rising", "Stable", or "Declining" |
| `trend_score` | float | normalized_slope × R². Core composite metric. |
| `slope_beta1` | float | Raw slope — points per match from regression |
| `r_squared` | float | How well the trend line fits the data (0–1) |
| `confidence` | string | "High" (R²≥0.7), "Moderate" (R²≥0.4), "Low" (R²<0.4) |
| `num_matches_used` | int | Number of scores used |
| `derived_variables` | object | Full breakdown of all intermediate computed values |
| `interpretation` | string | Human-readable summary of the analysis |

### Trend Classification Thresholds

| trend_score | Label |
|---|---|
| > +0.02 | Rising |
| < -0.02 | Declining |
| Between ±0.02 | Stable |

A threshold of ±0.02 represents a 2% change per match relative to the player's average score — the minimum meaningful change in sports performance analytics.

---

## Example Request

```json
POST /api/v1/player/form-trend
Content-Type: application/json

{
  "player_id": "P001",
  "player_name": "Rohit Sharma",
  "performance_scores": [55.0, 60.0, 58.0, 67.0, 72.0, 75.0, 80.0],
  "decay_rate": 0.85
}
```

---

## Example Response

```json
{
  "player_id": "P001",
  "player_name": "Rohit Sharma",
  "trend_label": "Rising",
  "trend_score": 0.0389,
  "slope_beta1": 4.0476,
  "r_squared": 0.9421,
  "confidence": "High",
  "num_matches_used": 7,
  "derived_variables": {
    "x_indices": [1, 2, 3, 4, 5, 6, 7],
    "weights": [0.053, 0.063, 0.074, 0.087, 0.102, 0.12, 0.141],
    "weighted_mean_x": 5.0812,
    "weighted_mean_y": 72.3156,
    "slope_beta1": 4.0476,
    "intercept_beta0": 51.7098,
    "ss_total": 184.2341,
    "ss_residual": 10.6812,
    "r_squared": 0.9421,
    "normalized_slope": 0.0413
  },
  "interpretation": "Rohit Sharma is improving at approximately 4.05 points per match over the last 7 matches analyzed. The trend line explains 94.2% of the performance variance (R-squared = 0.942), indicating high confidence in this trend direction."
}
```

---

## Validation Errors

| Condition | HTTP Status | Error Message |
|---|---|---|
| Fewer than 3 scores | 400 | "At least 3 performance scores are required for trend analysis." |
| More than 30 scores | 400 | "Maximum 30 scores allowed..." |
| Score outside [0, 100] | 400 | "Score {x} at position {i} is out of valid range [0, 100]." |
| NaN or Inf in scores | 400 | "Score at position {i} is not a valid number (NaN or Inf)." |
| Missing required field | 422 | Pydantic schema validation error |
| decay_rate outside [0.5, 1.0] | 422 | Pydantic field validation error |

---

## Assumptions Made

1. **Score range is [0, 100].** If the upstream scoring system uses a different range (e.g. 0–50), either normalize scores before calling this API or adjust the validation bounds in `utils.py`.

2. **Scores are ordered oldest to newest.** The calling system must ensure chronological ordering. This API does not sort or infer order.

3. **Default decay_rate = 0.85.** This means each older match is 85% as important as the one after it. This is a standard exponential smoothing parameter used in time-series sports analytics.

4. **Trend threshold = ±0.02.** A 2% per-match change relative to mean score is treated as the minimum meaningful trend. This can be tuned by the integration team if domain-specific thresholds are known.

5. **Maximum 30 matches.** Form is a short-term concept. Using 30+ matches risks picking up seasonal trends rather than current form.

---

## Proposed New Model Fields

If the webapp's `PlayerMatch` or `MatchPerformance` model does not yet store a normalized performance score, the following field is recommended:

```
PlayerMatchPerformance:
    player_id         : str
    match_id          : str
    match_date        : datetime   ← required to ensure correct chronological ordering
    performance_score : float      ← normalized [0–100] composite score for that match
```

The `match_date` field is critical — without it, the calling system cannot guarantee score ordering, which invalidates the time-axis assumption of the regression.

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Test via Swagger UI
Open: `http://localhost:8000/docs`

### 4. Test via curl
```bash
curl -X POST http://localhost:8000/api/v1/player/form-trend \
  -H "Content-Type: application/json" \
  -d '{
    "player_id": "P001",
    "player_name": "Rohit Sharma",
    "performance_scores": [55.0, 60.0, 58.0, 67.0, 72.0, 75.0, 80.0],
    "decay_rate": 0.85
  }'
```

### 5. Health check
```bash
curl http://localhost:8000/health
```

---

## File Structure

```
form_trend_api/
├── main.py           # FastAPI app and route definitions
├── schemas.py        # Pydantic request and response models
├── services.py       # Core EWLR analytics logic
├── utils.py          # Validation, weight generation, classification helpers
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

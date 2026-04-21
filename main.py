"""
main.py — FastAPI app and route definitions for Form Trend API

API Name  : Form Trend (Regression)
Version   : 1.0.0
Author    : Student 5 — Meta Analytics
Sprint    : Sprint 2 — Player Analytics
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from schemas import FormTrendRequest, FormTrendResponse
from services import compute_form_trend

# ---------------------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Form Trend API",
    description=(
        "Identifies whether a player's recent performance trend is Rising, Stable, "
        "or Declining using Exponentially Weighted Linear Regression. "
        "Returns a trend score, R-squared reliability measure, slope, and full "
        "derived variable breakdown."
    ),
    version="1.0.0",
    contact={
        "name": "Student 5 — Meta Analytics",
    },
    openapi_tags=[
        {
            "name": "Form Trend",
            "description": "Player performance trend analysis using weighted regression.",
        }
    ],
)


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------

@app.get(
    "/health",
    tags=["Health"],
    summary="Health check",
    description="Returns 200 OK if the API is running.",
)
def health_check():
    return {"status": "ok", "api": "Form Trend API", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Main Endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/api/v1/player/form-trend",
    response_model=FormTrendResponse,
    tags=["Form Trend"],
    summary="Compute player form trend",
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Trend analysis computed successfully."},
        400: {"description": "Validation error — invalid input data."},
        422: {"description": "Schema validation error — malformed request body."},
        500: {"description": "Internal server error during computation."},
    },
)
def get_form_trend(request: FormTrendRequest) -> FormTrendResponse:
    """
    Analyze a player's recent performance trend using Exponentially Weighted
    Linear Regression (EWLR).

    **What this API does:**
    - Takes an ordered list of recent performance scores (oldest → newest)
    - Applies exponential decay weights so recent matches matter more
    - Fits a weighted least squares regression line through the scores
    - Returns slope (β₁), R-squared, normalized slope, and a composite trend score
    - Classifies the trend as **Rising**, **Stable**, or **Declining**

    **Scientific principle:**
    Exponentially Weighted Least Squares regression, where:
    - `trend_score = normalized_slope × R²`
    - Normalized slope captures direction and magnitude relative to mean score
    - R² captures consistency/reliability of the trend

    **Threshold logic:**
    - `trend_score > 0.02`  → Rising
    - `trend_score < -0.02` → Declining
    - Otherwise            → Stable
    """
    try:
        result = compute_form_trend(request)
        return result

    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during trend computation: {str(e)}",
        )

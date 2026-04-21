from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class FormTrendRequest(BaseModel):
    player_id: str = Field(..., description="Unique identifier for the player")
    player_name: Optional[str] = Field(None, description="Player name (optional, for display)")
    performance_scores: List[float] = Field(
        ...,
        description=(
            "Ordered list of recent performance scores from oldest to newest. "
            "Each score should be in the range [0, 100]. Minimum 3 scores required, maximum 30."
        )
    )
    @field_validator("performance_scores")
    @classmethod
    def validate_scores_list(cls, v):
        if len(v) < 3:
            raise ValueError("At least 3 performance scores are required for trend analysis.")
        if len(v) > 30:
            raise ValueError("Maximum 30 scores allowed. Form trend focuses on recent performance.")
        for score in v:
            if score < 0 or score > 100:
                raise ValueError(f"Score {score} is out of valid range [0, 100].")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "player_id": "P001",
                "player_name": "Rohit Sharma",
                "performance_scores": [55.0, 60.0, 58.0, 67.0, 72.0, 75.0, 80.0]
            }
        }
    }


class DerivedVariables(BaseModel):
    x_indices: List[int] = Field(..., description="Match index values used as the time axis (1-based)")
    weights: List[float] = Field(..., description="Exponential decay weights assigned to each match")
    weighted_mean_x: float = Field(..., description="Weighted mean of match indices")
    weighted_mean_y: float = Field(..., description="Weighted mean of performance scores")
    slope_beta1: float = Field(..., description="Raw regression slope (beta1) — change in score per match")
    intercept_beta0: float = Field(..., description="Regression intercept (beta0) — baseline score value")
    ss_total: float = Field(..., description="Total sum of squares — total variance in scores")
    ss_residual: float = Field(..., description="Residual sum of squares — variance not explained by the line")
    r_squared: float = Field(..., description="R-squared — proportion of variance explained by the trend line (0 to 1)")
    normalized_slope: float = Field(..., description="Slope normalized by weighted mean score — scale-independent trend rate")


class FormTrendResponse(BaseModel):
    player_id: str
    player_name: Optional[str]
    trend_label: str = Field(
        ...,
        description="Final trend interpretation: 'Rising', 'Stable', or 'Declining'"
    )
    trend_score: float = Field(
        ...,
        description=(
            "Final composite trend score = normalized_slope x R-squared. "
            "Combines direction and consistency. Positive = rising, negative = declining."
        )
    )
    slope_beta1: float = Field(..., description="Raw slope — points gained or lost per match on average")
    r_squared: float = Field(..., description="R-squared reliability of the trend line (0 to 1)")
    confidence: str = Field(
        ...,
        description="Confidence in trend reliability: 'High' (R2>=0.7), 'Moderate' (R2>=0.4), 'Low' (R2<0.4)"
    )
    num_matches_used: int = Field(..., description="Number of match scores used in analysis")
    derived_variables: DerivedVariables = Field(..., description="All intermediate computed variables")
    interpretation: str = Field(..., description="Human-readable summary of the trend analysis")

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class FormTrendRequest(BaseModel):
    player_id: str
    player_name: Optional[str] = None
    performance_scores: List[float]

    @field_validator("performance_scores")
    @classmethod
    def validate_scores(cls, v):
        if len(v) < 3:
            raise ValueError("At least 3 scores required")
        if len(v) > 30:
            raise ValueError("Maximum 30 scores allowed")
        for score in v:
            if score < 0 or score > 100:
                raise ValueError("Scores must be between 0 and 100")
        return v


class TrendDetails(BaseModel):
    weights: List[float]
    slope: float
    intercept: float
    r_squared: float
    normalized_slope: float


class FormTrendResponse(BaseModel):
    player_id: str
    player_name: Optional[str]
    trend_label: str
    trend_score: float
    slope: float
    r_squared: float
    confidence: str
    matches_used: int
    details: TrendDetails
    interpretation: str

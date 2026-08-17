"""Pydantic schemas for the FastAPI application."""

from pydantic import BaseModel, Field, ConfigDict
from typing import Dict

class PredictionRequest(BaseModel):
    """Schema for a single defect prediction request."""
    loc: float = Field(..., description="Lines of Code")
    cyclomatic_complexity: float = Field(..., description="McCabe's Cyclomatic Complexity")
    halstead_volume: float = Field(..., description="Halstead Volume metric")
    # Add other key metrics used in your model...
    
    # Pydantic V2 compliant configuration
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "loc": 150.0,
                "cyclomatic_complexity": 12.0,
                "halstead_volume": 450.5
            }
        }
    )

class PredictionResponse(BaseModel):
    """Schema for the prediction response."""
    defect_probability: float = Field(..., description="Probability of a defect existing (0.0 to 1.0)")
    prediction: int = Field(..., description="Binary prediction (1 for defect, 0 for clean)")
    shap_values: Dict[str, float] = Field(default_factory=dict, description="SHAP values for local explainability")
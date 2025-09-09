from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ClothingItem(BaseModel):
    item_type: str
    confidence: float
    bounding_box: List[int]

class DominantColor(BaseModel):
    color_name: str
    rgb: List[int]
    hex: str
    percentage: float

class ConfidenceScores(BaseModel):
    overall: float
    style: float
    color: float

class ClothingAnalysisResponse(BaseModel):
    status: str
    clothing_items: List[ClothingItem]
    style_classification: str
    formality: str
    texture: str
    dominant_colors: List[DominantColor]
    color_distribution: Dict[str, float]
    detailed_attributes: Dict[str, Any]
    confidence_scores: ConfidenceScores

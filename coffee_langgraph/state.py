from __future__ import annotations
from typing import TypedDict, Literal, Optional, Dict, Any

Impact = Literal["bullish", "bearish", "neutral"]

class Signal(TypedDict, total=False):
    impact: Impact
    confidence: float
    rationale: str
    features: Dict[str, Any]

class PredictivePrice(TypedDict, total=False):
    horizon_days: int
    predicted_return_pct: float  # e.g., +1.2 => +1.2% over horizon
    source: str  # e.g., 'timesfm', 'timegpt', 'csv', 'manual'

class GraphState(TypedDict, total=False):
    # Inputs
    user_request: str
    period_days: int
    weather_csv: Optional[str]
    news_csv: Optional[str]
    logistics_csv: Optional[str]
    forecast_csv: Optional[str]
    last_price_csv: Optional[str]
    predictive_price: Optional[PredictivePrice]

    # Agent outputs
    geospatial_signal: Optional[Signal]
    web_news_signal: Optional[Signal]
    logistics_signal: Optional[Signal]

    # Final
    decision: Optional[Dict[str, Any]]

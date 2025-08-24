from __future__ import annotations
from typing import Dict, Any, Optional
from coffee_langgraph.state import Signal, PredictivePrice
from coffee_langgraph import config

def impact_to_score(impact: str) -> float:
    return {"bullish": 1.0, "neutral": 0.0, "bearish": -1.0}.get(impact, 0.0)

def combine_signals(geo: Optional[Signal], web: Optional[Signal], logi: Optional[Signal], pp: Optional[PredictivePrice]) -> Dict[str, Any]:
    # Convert impacts to scores
    geo_s = impact_to_score(geo.get("impact")) if geo else 0.0
    web_s = impact_to_score(web.get("impact")) if web else 0.0
    logi_s = impact_to_score(logi.get("impact")) if logi else 0.0
    pred_s = 0.0
    if pp and (pp.get("predicted_return_pct") is not None):
        # map predicted return to -1..1 using a squashing function
        r = float(pp["predicted_return_pct"]) / 2.0  # 2% maps to ±1 approx
        pred_s = max(-1.0, min(1.0, r))

    # Weighted sum
    total = (geo_s * config.WEIGHTS["geospatial"] +
             web_s * config.WEIGHTS["web_news"] +
             logi_s * config.WEIGHTS["logistics"] +
             pred_s * config.WEIGHTS["predictive"])

    if total >= config.BULLISH_THRESHOLD:
        stance = "bullish"
    elif total <= config.BEARISH_THRESHOLD:
        stance = "bearish"
    else:
        stance = "neutral"

    details = {
        "weights": config.WEIGHTS,
        "component_scores": {
            "geospatial": geo_s,
            "web_news": web_s,
            "logistics": logi_s,
            "predictive": pred_s,
        },
        "total_score": round(total, 3),
        "stance": stance
    }
    return details

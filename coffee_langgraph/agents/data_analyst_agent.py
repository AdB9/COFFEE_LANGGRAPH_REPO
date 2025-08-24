from __future__ import annotations
from typing import Dict, Any, Optional
import pandas as pd
from ..state import GraphState, PredictivePrice
from ..utils.io import read_csv_if_exists
from ..utils.scoring import combine_signals

def load_predictive_price(state: GraphState) -> Optional[PredictivePrice]:
    # Command-line/manual override has priority
    if state.get("predictive_price") and state["predictive_price"].get("predicted_return_pct") is not None:
        return state["predictive_price"]

    # Try forecast CSV + last price CSV to compute predicted return
    fc_path = state.get("forecast_csv") or "data/sample/price_forecast_sample.csv"
    lp_path = state.get("last_price_csv") or "data/sample/last_price.csv"
    fc = read_csv_if_exists(fc_path)
    lp = read_csv_if_exists(lp_path)
    if fc is None or lp is None or fc.empty or lp.empty:
        return None

    # Use horizon matching period_days if available; else compute 2-day horizon
    horizon_days = int(state.get("period_days") or 2)
    fc["ds"] = pd.to_datetime(fc["ds"]).dt.date
    lp_date = pd.to_datetime(lp["date"]).dt.date.iloc[-1]
    last_close = float(lp["last_close"].iloc[-1])
    # Take the forecast value at horizon_days ahead (first row is t+1)
    if len(fc) >= horizon_days:
        yhat = float(fc.iloc[horizon_days-1]["yhat"])
    else:
        yhat = float(fc.iloc[-1]["yhat"])
    ret_pct = (yhat - last_close) / last_close * 100.0
    return {"horizon_days": horizon_days, "predicted_return_pct": ret_pct, "source": "csv"}

def data_analyst_agent(state: GraphState) -> Dict[str, Any]:
    # Only aggregate when all three signals exist
    if not all([state.get("geospatial_signal"), state.get("web_news_signal"), state.get("logistics_signal")]):
        # Return state unchanged; aggregator will be called again after other branches
        return {}

    pp = state.get("predictive_price") or load_predictive_price(state)
    details = combine_signals(state.get("geospatial_signal"), state.get("web_news_signal"), state.get("logistics_signal"), pp)

    rationale = []
    for k, v in details["component_scores"].items():
        if abs(v) > 0.15:
            tag = "bullish" if v > 0 else ("bearish" if v < 0 else "neutral")
            rationale.append(f"{k}={tag} ({v:+.2f})")
    if pp:
        rationale.append(f"predictive_return={pp['predicted_return_pct']:+.2f}% over {pp['horizon_days']}d")

    return {
        "decision": {
            "stance": details["stance"],
            "score": details["total_score"],
            "weights": details["weights"],
            "components": details["component_scores"],
            "predictive_price": pp,
            "rationale": "; ".join(rationale) or "Signals mixed; defaulting to neutral."
        }
    }

from __future__ import annotations
import pandas as pd
from typing import Dict, Any
from ..state import GraphState
from ..utils.io import read_csv_if_exists
from .. import config

# Convert numpy/pandas scalars -> native Python types (msgpack-safe)
def to_native(obj):
    try:
        import numpy as np
        np_generic = (np.generic,)
    except Exception:
        np_generic = tuple()
    if isinstance(obj, np_generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(v) for v in obj]
    return obj

def geospatial_agent(state: GraphState) -> Dict[str, Any]:
    """
    Heuristic geospatial signal using CSV:
    Columns: date,region,min_temp_c,precip_14d_mm,precip_14d_norm_mm,ndvi_anom
    """
    path = state.get("weather_csv") or "data/sample/weather_sample.csv"
    df = read_csv_if_exists(path)
    if df is None or df.empty:
        return {"geospatial_signal": {
            "impact": "neutral", "confidence": 0.3, "rationale": "No weather data found.", "features": {}
        }}

    df["date"] = pd.to_datetime(df["date"])
    latest_date = df["date"].max()
    latest_df = df[df["date"] == latest_date]

    frost = (latest_df["min_temp_c"] <= config.FROST_TEMP_C).any()
    deficit = ((latest_df["precip_14d_norm_mm"] - latest_df["precip_14d_mm"]) >= config.DRYNESS_MM_DEFICIT).any()

    ndvi_series = latest_df.get("ndvi_anom", None)
    ndvi_avg = 0.0
    ndvi_neg = False
    if ndvi_series is not None:
        ndvi_avg = float(pd.to_numeric(ndvi_series, errors="coerce").mean())
        ndvi_neg = ndvi_avg < 0

    score = 0
    rationale_bits = []
    if bool(frost):
        score += 1; rationale_bits.append("Frost risk detected in at least one region.")
    if bool(deficit):
        score += 1; rationale_bits.append("14d precipitation deficit exceeds threshold.")
    if ndvi_neg:
        score += 1; rationale_bits.append("Negative NDVI anomaly suggests plant stress.")

    if score >= 2:
        impact, conf = "bullish", 0.75
    elif score == 1:
        impact, conf = "bullish", 0.55
    else:
        impact, conf = "neutral", 0.4

    signal = {
        "impact": impact,
        "confidence": float(conf),
        "rationale": " ".join(rationale_bits) or "No significant geospatial risk signals.",
        "features": {
            "latest_date": str(getattr(latest_date, "date", lambda: latest_date)()),
            "regions": [str(r) for r in latest_df["region"].astype(str).unique().tolist()],
            "min_temp_c": float(pd.to_numeric(latest_df["min_temp_c"], errors="coerce").min()),
            "precip_14d_mm_sum": float(pd.to_numeric(latest_df["precip_14d_mm"], errors="coerce").sum()),
            "precip_deficit_any": bool(deficit),
            "ndvi_anom_avg": float(ndvi_avg),
        }
    }
    return {"geospatial_signal": to_native(signal)}

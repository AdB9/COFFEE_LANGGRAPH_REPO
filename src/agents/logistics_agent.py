from __future__ import annotations
import pandas as pd
from typing import Dict, Any
from ..state import GraphState
from ..utils.io import read_csv_if_exists
from .. import config

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

def logistics_agent(state: GraphState) -> Dict[str, Any]:
    """
    Heuristic logistics signal using CSV:
    Columns: date,port,vessels_waiting,avg_wait_time_days,freight_index,disruptions
    """
    path = state.get("logistics_csv") or "data/sample/logistics_sample.csv"
    df = read_csv_if_exists(path)
    if df is None or df.empty:
        return {"logistics_signal": {
            "impact": "neutral", "confidence": 0.35, "rationale": "No logistics data.", "features": {}
        }}

    df["date"] = pd.to_datetime(df["date"])
    latest_date = df["date"].max()
    latest_df = df[df["date"] == latest_date]

    wait_spike = bool((pd.to_numeric(latest_df["avg_wait_time_days"], errors="coerce") >= float(config.WAIT_DAYS_SPIKE)).any())
    freight_spike = bool((pd.to_numeric(latest_df["freight_index"], errors="coerce") >= float(config.FREIGHT_INDEX_SPIKE)).any())

    disruption_col = latest_df.get("disruptions", None)
    has_disruption = False
    if disruption_col is not None:
        has_disruption = bool(disruption_col.fillna("").astype(str).str.len().gt(0).any())

    score = sum([wait_spike, freight_spike, has_disruption])
    if score >= 2:
        impact, conf = "bullish", 0.65
        reason = "Congestion/freight/disruption elevated (supply delays)."
    elif score == 1:
        impact, conf = "bullish", 0.55
        reason = "Some logistics stress indicators elevated."
    else:
        impact, conf = "neutral", 0.45
        reason = "No major logistics stress."

    signal = {
        "impact": impact,
        "confidence": float(conf),
        "rationale": reason,
        "features": {
            "latest_date": str(getattr(latest_date, "date", lambda: latest_date)()),
            "ports": [str(p) for p in latest_df["port"].astype(str).unique().tolist()],
            "avg_wait_time_max": float(pd.to_numeric(latest_df["avg_wait_time_days"], errors="coerce").max()),
            "freight_index_max": float(pd.to_numeric(latest_df["freight_index"], errors="coerce").max()),
            "disruptions_any": bool(has_disruption),
        }
    }
    return {"logistics_signal": to_native(signal)}

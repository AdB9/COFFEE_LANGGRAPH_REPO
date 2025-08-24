from __future__ import annotations
import pandas as pd
from typing import Dict, Any
from src.state import GraphState
from src.utils.io import read_csv_if_exists
from src.utils.data_fetchers import fetch_santos_arrivals_table


def _to_native(o):
    try:
        import numpy as np
        if isinstance(o, np.generic): return o.item()
    except Exception: pass
    if isinstance(o, dict): return {k: _to_native(v) for k, v in o.items()}
    if isinstance(o, list): return [_to_native(v) for v in o]
    return o

def logistics_agent(state: GraphState) -> Dict[str, Any]:
    path = state.get("logistics_csv")
    df = read_csv_if_exists(path) if path else fetch_santos_arrivals_table()

    if df is None or df.empty:
        return {"logistics_signal": {"impact":"neutral","confidence":0.4,"rationale":"No logistics data available.","features":{}}}

    txt = df.apply(lambda s: s.astype(str).str.lower())
    is_container = txt.apply(lambda s: s.str.contains("conteiner|container|tecon|btp|msc|maersk", regex=True))
    is_agri = txt.apply(lambda s: s.str.contains("cafe|coffee|acucar|sugar", regex=True))

    container_hits = int(is_container.any(axis=1).sum())
    agro_hits = int(is_agri.any(axis=1).sum())

    score, bits = 0, []
    if container_hits >= 20: score += 1; bits.append("High container arrival density at Santos.")
    if agro_hits >= 5: score += 1; bits.append("Elevated agribulk movements near Santos.")

    if score >= 2: impact, conf = "bullish", 0.65
    elif score == 1: impact, conf = "neutral", 0.50
    else: impact, conf = "neutral", 0.45

    signal = {
        "impact": impact, "confidence": float(conf),
        "rationale": " ".join(bits) or "No significant congestion signals from Santos schedule.",
        "features": {"container_arrivals_count": int(container_hits), "agri_related_count": int(agro_hits)}
    }
    return {"logistics_signal": _to_native(signal)}

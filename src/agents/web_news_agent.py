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

def web_news_agent(state: GraphState) -> Dict[str, Any]:
    """
    Heuristic web/news signal using CSV:
    Columns: date,lang,keyword,count,avg_sentiment
    """
    path = state.get("news_csv") or "data/sample/news_sample.csv"
    df = read_csv_if_exists(path)
    if df is None or df.empty:
        return {"web_news_signal": {
            "impact": "neutral", "confidence": 0.3, "rationale": "No news/social data.", "features": {}
        }}

    df["date"] = pd.to_datetime(df["date"])
    latest_date = df["date"].max()
    latest_df = df[df["date"] == latest_date]

    total_count = float(pd.to_numeric(latest_df["count"], errors="coerce").sum())
    avg_sent = float(pd.to_numeric(latest_df.get("avg_sentiment", 0), errors="coerce").mean())

    frost_like = latest_df[latest_df["keyword"].astype(str).str.contains("frost|geada|helada", case=False, na=False)]
    drought_like = latest_df[latest_df["keyword"].astype(str).str.contains("drought|seca|sequía", case=False, na=False)]
    supply_neg_count = float(
        pd.to_numeric(frost_like["count"], errors="coerce").sum()
        + pd.to_numeric(drought_like["count"], errors="coerce").sum()
    )

    if supply_neg_count >= float(config.NEWS_COUNT_SPIKE) and avg_sent <= float(config.NEG_SENTIMENT):
        impact, conf = "bullish", 0.7
        reason = "Supply-risk news spike with negative sentiment."
    elif total_count >= float(config.NEWS_COUNT_SPIKE) and avg_sent > 0.1:
        impact, conf = "bearish", 0.6
        reason = "High volume with positive sentiment (possible bumper/relief news)."
    else:
        impact, conf = "neutral", 0.45
        reason = "No clear volume/sentiment extremes."

    signal = {
        "impact": impact,
        "confidence": float(conf),
        "rationale": reason,
        "features": {
            "latest_date": str(getattr(latest_date, "date", lambda: latest_date)()),
            "total_count": float(total_count),
            "avg_sentiment": float(avg_sent),
            "supply_neg_count": float(supply_neg_count),
        }
    }
    return {"web_news_signal": to_native(signal)}

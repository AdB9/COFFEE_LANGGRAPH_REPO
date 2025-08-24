from __future__ import annotations
import pandas as pd
from typing import Dict, Any
from ..state import GraphState
from ..utils.io import read_csv_if_exists
from ..utils.data_fetchers import fetch_gdelt_docs

TERMS = [
    "coffee OR arabica OR robusta",
    "(frost OR geada OR helada OR cold snap OR freeze)",
    "(drought OR seca OR sequía)",
    "(strike OR protest OR blockade)",
    "(shipping OR freight OR port OR congestion)",
    "(El Niño OR La Niña)",
]

def _to_native(o):
    try:
        import numpy as np
        if isinstance(o, np.generic): return o.item()
    except Exception: pass
    if isinstance(o, dict): return {k: _to_native(v) for k, v in o.items()}
    if isinstance(o, list): return [_to_native(v) for v in o]
    return o

def web_news_agent(state: GraphState) -> Dict[str, Any]:
    path = state.get("news_csv")
    period_days = int(state.get("period_days", 2))

    # CSV override if provided
    if path:
        df = read_csv_if_exists(path)
        if df is not None and not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            latest_date = df["date"].max()
            latest_df = df[df["date"] == latest_date]
            total_count = float(pd.to_numeric(latest_df["count"], errors="coerce").sum())
            avg_sent = float(pd.to_numeric(latest_df.get("avg_sentiment", 0), errors="coerce").mean())
            features = {"latest_date": str(latest_date.date()), "total_count": total_count, "avg_sentiment": avg_sent}
        else:
            df = None

    if not path or df is None or df.empty:
        end = pd.Timestamp.utcnow()
        start = end - pd.Timedelta(days=period_days)
        q = " ".join(TERMS)
        gd = fetch_gdelt_docs(q, start.strftime("%Y%m%d%H%M%S"), end.strftime("%Y%m%d%H%M%S"), max_records=250)
        if gd is None or gd.empty:
            return {"web_news_signal": {"impact":"neutral","confidence":0.4,"rationale":"No recent relevant news.","features":{}}}
        tone = float(gd.get("tone", pd.Series([0.0]*len(gd))).astype(float).mean())
        features = {
            "latest_date": str(pd.to_datetime(gd["seendate"]).max().date()) if "seendate" in gd else end.strftime("%Y-%m-%d"),
            "total_count": float(len(gd)),
            "avg_tone": tone,
        }
        avg_sent = tone
        total_count = float(len(gd))

    # Simple mapping: negative news tone / high volume => bullish (supply risk)
    if total_count >= 100 and avg_sent <= -0.1:
        impact, conf, reason = "bullish", 0.65, "High volume of supply-risk coverage with negative tone."
    elif total_count >= 100 and avg_sent > 0.1:
        impact, conf, reason = "bearish", 0.55, "High volume with positive tone (relief/bumper signals)."
    else:
        impact, conf, reason = "neutral", 0.45, "Mixed/limited signals."

    signal = {"impact": impact, "confidence": float(conf), "rationale": reason, "features": features}
    return {"web_news_signal": _to_native(signal)}

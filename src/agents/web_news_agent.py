from __future__ import annotations
import pandas as pd
from typing import Dict, Any
from src.state import GraphState
from src.utils.io import read_csv_if_exists
from src.utils.data_fetchers import fetch_gdelt_docs

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
    features = {}
    avg_sent, total_count = 0.0, 0.0

    # CSV override if provided
    if path:
        df = read_csv_if_exists(path)
        if df is not None and not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            latest_date = df["date"].max()
            latest_df = df[df["date"] == latest_date]
            total_count = float(pd.to_numeric(latest_df["count"], errors="coerce").sum())
            avg_sent = float(pd.to_numeric(latest_df.get("avg_sentiment", 0), errors="coerce").mean())
            features = {
                "latest_date": str(latest_date.date()),
                "total_count": total_count,
                "avg_sentiment": avg_sent,
            }
        else:
            df = None

    if not path or df is None or df.empty:
        end = pd.Timestamp.utcnow()
        start = end - pd.Timedelta(days=period_days)
        q = " ".join(TERMS)

        try:
            gd = fetch_gdelt_docs(
                q,
                start.strftime("%Y%m%d%H%M%S"),
                end.strftime("%Y%m%d%H%M%S"),
                max_records=150,   # lower to reduce 429s
            )
        except Exception as e:
            gd = pd.DataFrame()

        if gd is None or gd.empty:
            # graceful fallback
            return {
                "web_news_signal": {
                    "impact": "neutral",
                    "confidence": 0.40,
                    "rationale": "News API throttled or no recent articles; defaulting to neutral.",
                    "features": {"latest_date": str(end.date()), "total_count": 0.0, "avg_tone": 0.0},
                }
            }

        tone_series = gd["tone"] if "tone" in gd.columns else pd.Series([0.0] * len(gd))
        avg_sent = float(pd.to_numeric(tone_series, errors="coerce").fillna(0).mean())
        total_count = float(len(gd))
        features = {
            "latest_date": str(pd.to_datetime(gd["seendate"]).max().date()) if "seendate" in gd else end.strftime("%Y-%m-%d"),
            "total_count": total_count,
            "avg_tone": avg_sent,
        }

    # Heuristic mapping
    if total_count >= 100 and avg_sent <= -0.1:
        impact, conf, reason = "bullish", 0.65, "High volume of supply-risk coverage with negative tone."
    elif total_count >= 100 and avg_sent > 0.1:
        impact, conf, reason = "bearish", 0.55, "High volume with positive tone (relief/bumper signals)."
    else:
        impact, conf, reason = "neutral", 0.45, "Mixed or limited signals."

    signal = {
        "impact": impact,
        "confidence": float(conf),
        "rationale": reason,
        "features": features,
    }
    return {"web_news_signal": _to_native(signal)}

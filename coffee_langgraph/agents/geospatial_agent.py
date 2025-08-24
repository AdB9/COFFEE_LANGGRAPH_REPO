from __future__ import annotations
import pandas as pd
from typing import Dict, Any
from ..state import GraphState
from ..utils.io import read_csv_if_exists
from ..utils.data_fetchers import fetch_open_meteo_daily, fetch_modis_ndvi_ornl
from .. import config

def _to_native(o):
    try:
        import numpy as np
        if isinstance(o, np.generic): return o.item()
    except Exception: pass
    if isinstance(o, dict): return {k: _to_native(v) for k, v in o.items()}
    if isinstance(o, list): return [_to_native(v) for v in o]
    return o

def _fetch_region_frames(regions, start_date, end_date):
    frames = []
    for r in regions:
        lat, lon = float(r["lat"]), float(r["lon"])
        wx = fetch_open_meteo_daily(lat, lon, start_date, end_date)
        ndvi = fetch_modis_ndvi_ornl(lat, lon, start_date, end_date)
        df = wx.merge(ndvi, on="date", how="left"); df["region"] = r["name"]; frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def geospatial_agent(state: GraphState) -> Dict[str, Any]:
    path = state.get("weather_csv")
    period_days = int(state.get("period_days", 2))
    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=period_days + 14)  # 2w context

    if path:
        df = read_csv_if_exists(path)
    else:
        regions = state.get("regions") or [{"name":"Sul de Minas, Brazil","lat":-21.5,"lon":-45.0}]
        df = _fetch_region_frames(regions, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    if df is None or df.empty:
        return {"geospatial_signal": {"impact":"neutral","confidence":0.3,"rationale":"No geospatial data fetched.","features":{}}}

    latest_date = df["date"].max()
    latest_df = df[df["date"] == latest_date].copy()
    frost = (latest_df["tmin_c"] <= config.FROST_TEMP_C).any()
    precip_14 = df[df["date"] >= latest_date - pd.Timedelta(days=14)].groupby("region")["precip_mm"].sum().mean()
    ndvi_avg = latest_df.get("ndvi", pd.Series(dtype=float)).mean(skipna=True)

    score, bits = 0, []
    if bool(frost): score += 1; bits.append("Frost risk in at least one region.")
    if float(precip_14 or 0) < float(config.PRECIP_14D_MIN_MM): score += 1; bits.append("14d precip below threshold (dryness).")
    if pd.notnull(ndvi_avg) and ndvi_avg < 0.0: score += 1; bits.append("Negative NDVI anomaly.")

    if score >= 2: impact, conf = "bullish", 0.70
    elif score == 1: impact, conf = "bullish", 0.55
    else: impact, conf = "neutral", 0.45

    signal = {
        "impact": impact, "confidence": float(conf), "rationale": " ".join(bits) or "No strong geospatial anomalies.",
        "features": {
            "latest_date": str(pd.to_datetime(latest_date).date()),
            "regions": list(map(str, latest_df["region"].unique().tolist())),
            "tmin_min_c": float(pd.to_numeric(latest_df["tmin_c"], errors="coerce").min()),
            "precip_14d_mm_mean": float(precip_14 or 0),
            "ndvi_mean": float(ndvi_avg) if pd.notnull(ndvi_avg) else None,
        }
    }
    return {"geospatial_signal": _to_native(signal)}

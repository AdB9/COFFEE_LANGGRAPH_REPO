from __future__ import annotations
import pandas as pd
from typing import Dict, Any
from coffee_langgraph.state import GraphState
from coffee_langgraph.utils.io import read_csv_if_exists
from coffee_langgraph.utils.data_fetchers import fetch_open_meteo_daily, fetch_modis_ndvi_ornl
from coffee_langgraph import config as CFG  # <- keep this alias

# Safe defaults if config module is missing some attrs
FROST_TMIN_C = getattr(CFG, "FROST_TMIN_C", 2.0)
PRECIP_14D_MIN_MM = getattr(CFG, "PRECIP_14D_MIN_MM", 25.0)
NDVI_ANOM_BULLISH_MAX = getattr(CFG, "NDVI_ANOM_BULLISH_MAX", -0.10)

def _to_native(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
    except Exception:
        pass
    if isinstance(o, dict):
        return {k: _to_native(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_to_native(v) for v in o]
    return o

def _fetch_region_frames(regions, start_date, end_date):
    frames = []
    for r in regions:
        lat, lon = float(r["lat"]), float(r["lon"])
        wx = fetch_open_meteo_daily(lat, lon, start_date, end_date)
        if wx is None or wx.empty:
            continue

        # NDVI: resilient; if empty, fill with NaN so merge works
        ndvi = fetch_modis_ndvi_ornl(lat, lon, start_date, end_date)
        if ndvi is None or ndvi.empty:
            ndvi = pd.DataFrame({"date": wx["date"], "ndvi": pd.Series([pd.NA] * len(wx))})

        # Ensure datetime before merge
        wx["date"] = pd.to_datetime(wx["date"], errors="coerce")
        ndvi["date"] = pd.to_datetime(ndvi["date"], errors="coerce")

        df = wx.merge(ndvi, on="date", how="left")
        df["region"] = r["name"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def geospatial_agent(state: GraphState) -> Dict[str, Any]:
    path = state.get("weather_csv")
    period_days = int(state.get("period_days", 2))
    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=period_days + 14)  # 2w context window

    if path:
        df = read_csv_if_exists(path)
    else:
        regions = state.get("regions") or [{"name": "Sul de Minas, Brazil", "lat": -21.5, "lon": -45.0}]
        df = _fetch_region_frames(regions, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    if df is None or df.empty:
        return {"geospatial_signal": {"impact": "neutral", "confidence": 0.30,
                                      "rationale": "No geospatial data fetched.", "features": {}}}

    # Types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["tmin_c"] = pd.to_numeric(df.get("tmin_c"), errors="coerce")
    df["precip_mm"] = pd.to_numeric(df.get("precip_mm"), errors="coerce")
    if "ndvi" in df.columns:
        df["ndvi"] = pd.to_numeric(df["ndvi"], errors="coerce")

    latest_date = df["date"].max()
    latest_df = df[df["date"] == latest_date].copy()

    # 14d window aggregates
    window14 = df[df["date"] >= (latest_date - pd.Timedelta(days=14))]
    precip_14 = float(window14.groupby("region")["precip_mm"].sum(min_count=1).mean()) if not window14.empty else 0.0

    # Frost anywhere in the latest day
    frost = bool((latest_df["tmin_c"] <= FROST_TMIN_C).any())

    # NDVI anomaly: last 14d mean vs overall mean (if available)
    if "ndvi" in df.columns and df["ndvi"].notna().any():
        ndvi_14 = window14["ndvi"].mean(skipna=True)
        ndvi_all = df["ndvi"].mean(skipna=True)
        ndvi_anom = float(ndvi_14 - ndvi_all) if pd.notna(ndvi_14) and pd.notna(ndvi_all) else None
    else:
        ndvi_anom = None

    # Scoring
    score, reasons = 0, []
    if frost:
        score += 1; reasons.append("Frost risk in at least one region.")
    if precip_14 < PRECIP_14D_MIN_MM:
        score += 1; reasons.append("14-day precipitation below threshold (dryness).")
    if ndvi_anom is not None and ndvi_anom <= NDVI_ANOM_BULLISH_MAX:
        score += 1; reasons.append("Negative NDVI anomaly (vegetation stress).")

    if score >= 2:
        impact, conf = "bullish", 0.70
    elif score == 1:
        impact, conf = "bullish", 0.55
    else:
        impact, conf = "neutral", 0.45

    features = {
        "latest_date": str(latest_date.date()),
        "regions": [str(x) for x in latest_df["region"].dropna().unique().tolist()],
        "tmin_min_c": float(pd.to_numeric(latest_df["tmin_c"], errors="coerce").min()),
        "precip_14d_mm_mean": float(precip_14),
        "ndvi_anom": ndvi_anom if ndvi_anom is not None else None,
    }

    signal = {
        "impact": impact,
        "confidence": float(conf),
        "rationale": " ".join(reasons) if reasons else "No strong geospatial anomalies.",
        "features": features,
    }
    return {"geospatial_signal": _to_native(signal)}

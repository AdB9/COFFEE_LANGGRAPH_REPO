from __future__ import annotations
from typing import Tuple
import requests, pandas as pd

def _daterange(start: str, end: str) -> Tuple[str, str]:
    s = pd.to_datetime(start).date().isoformat()
    e = pd.to_datetime(end).date().isoformat()
    return s, e

# --- Geospatial ---

def fetch_open_meteo_daily(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    # Open-Meteo Historical / ERA5-Land
    url = "https://archive-api.open-meteo.com/v1/archive"  # docs: open-meteo.com
    start, end = _daterange(start_date, end_date)
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start, "end_date": end,
        "daily": ["temperature_2m_min","precipitation_sum"],
        "timezone": "UTC",
    }
    r = requests.get(url, params=params, timeout=60); r.raise_for_status()
    daily = r.json().get("daily", {})
    if not daily: return pd.DataFrame()
    df = pd.DataFrame(daily)
    df["time"] = pd.to_datetime(df["time"])
    return df.rename(columns={"time":"date","temperature_2m_min":"tmin_c","precipitation_sum":"precip_mm"})

def fetch_modis_ndvi_ornl(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    # ORNL TESViS MODIS subset API (NDVI from MOD13Q1.061)
    url = "https://modis.ornl.gov/rst/api/v1/subset"
    start, end = _daterange(start_date, end_date)
    params = {
        "product": "MOD13Q1.061", "latitude": lat, "longitude": lon,
        "startDate": start, "endDate": end,
        "kmAboveBelow": 0, "kmLeftRight": 0
    }
    r = requests.get(url, params=params, timeout=90); r.raise_for_status()
    ts = []
    for d in r.json().get("subset", []):
        if "NDVI" in (d.get("band") or ""):
            val = d.get("value")
            if isinstance(val, list):
                try: val = sum(val)/len(val)
                except Exception: continue
            ts.append({"date": pd.to_datetime(d.get("calendar_date")), "ndvi": float(val) * 0.0001})
    return pd.DataFrame(ts).sort_values("date") if ts else pd.DataFrame()

# --- Logistics (Santos port page as congestion proxy) ---

def fetch_santos_arrivals_table() -> pd.DataFrame:
    # Parse table from Santos Port Authority "Scheduled arrivals"
    tables = pd.read_html("https://www.portodesantos.com.br/en/ship-tracker/scheduled-arrivals/")
    return tables[0] if tables else pd.DataFrame()

# --- News / Web ---

def fetch_gdelt_docs(query: str, start_datetime: str, end_datetime: str, max_records: int = 250) -> pd.DataFrame:
    # GDELT 2.0 Doc API: returns JSON list of articles
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query, "mode": "ArtList", "format": "JSON",
        "maxrecords": max_records, "sort": "DateDesc",
        "startdatetime": start_datetime, "enddatetime": end_datetime
    }
    r = requests.get(url, params=params, timeout=60); r.raise_for_status()
    df = pd.DataFrame(r.json().get("articles", []))
    if "seendate" in df.columns:
        df["seendate"] = pd.to_datetime(df["seendate"])
    return df

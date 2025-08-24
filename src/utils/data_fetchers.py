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
    """
    Robust wrapper for ORNL MODIS NDVI.
    - Returns EMPTY DataFrame on 404/429/any error so callers can continue.
    - MOD13Q1 is a 16-day composite; very recent end dates often aren't available yet.
    """
    import time
    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter

    url = "https://modis.ornl.gov/rst/api/v1/subset"
    start = pd.to_datetime(start_date).date().isoformat()
    end = pd.to_datetime(end_date).date().isoformat()
    params = {
        "product": "MOD13Q1.061",
        "latitude": lat,
        "longitude": lon,
        "startDate": start,
        "endDate": end,
        "kmAboveBelow": 0,
        "kmLeftRight": 0,
    }

    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(
        total=3, connect=2, read=2, backoff_factor=1.0,
        status_forcelist=[404, 429, 500, 502, 503, 504],
        allowed_methods=["GET"], raise_on_status=False
    )))

    try:
        r = s.get(url, params=params, timeout=90)
        # If still not OK (e.g., 404 for not-yet-available composite), return empty
        if r.status_code != 200:
            return pd.DataFrame()
        data = r.json().get("subset", [])
        ts = []
        for d in data:
            # Some records have band names; NDVI band values are scaled by 1e4
            band = (d.get("band") or "").upper()
            if "NDVI" in band:
                val = d.get("value")
                if isinstance(val, list):
                    try:
                        val = sum(val) / max(1, len(val))
                    except Exception:
                        continue
                try:
                    val = float(val) * 0.0001
                except Exception:
                    continue
                ts.append({"date": pd.to_datetime(d.get("calendar_date")), "ndvi": val})
        return pd.DataFrame(ts).sort_values("date") if ts else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# --- Logistics (Santos port page as congestion proxy) ---

def fetch_santos_arrivals_table() -> pd.DataFrame:
    """
    Scrape Santos Port scheduled-arrivals table using ONLY stdlib (html.parser).
    No lxml/bs4/html5lib required.
    """
    import requests
    from html.parser import HTMLParser

    url = "https://www.portodesantos.com.br/en/ship-tracker/scheduled-arrivals/"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    html = r.text

    class TableParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.in_table = False
            self.in_thead = False
            self.in_tbody = False
            self.in_tr = False
            self.in_th = False
            self.in_td = False
            self.headers = []
            self.current_row = []
            self.rows = []
            self._buf = []

        def handle_starttag(self, tag, attrs):
            if tag == "table" and not self.in_table:
                self.in_table = True
            elif self.in_table and tag == "thead":
                self.in_thead = True
            elif self.in_table and tag == "tbody":
                self.in_tbody = True
            elif self.in_table and tag == "tr":
                self.in_tr = True
                self.current_row = []
            elif self.in_table and tag == "th":
                self.in_th = True
                self._buf = []
            elif self.in_table and tag == "td":
                self.in_td = True
                self._buf = []

        def handle_endtag(self, tag):
            if tag == "th" and self.in_th:
                text = "".join(self._buf).strip()
                self.headers.append(text)
                self.in_th = False
            elif tag == "td" and self.in_td:
                text = "".join(self._buf).strip()
                self.current_row.append(text)
                self.in_td = False
            elif tag == "tr" and self.in_tr:
                # Only commit body rows (avoid header tr if any)
                if self.current_row:
                    self.rows.append(self.current_row)
                self.in_tr = False
            elif tag == "thead":
                self.in_thead = False
            elif tag == "tbody":
                self.in_tbody = False
            elif tag == "table":
                self.in_table = False

        def handle_data(self, data):
            if (self.in_th or self.in_td) and data:
                self._buf.append(data)

    parser = TableParser()
    parser.feed(html)

    # Build DataFrame
    if not parser.rows:
        return pd.DataFrame()

    # If header count matches row width, set columns; else fallback to default ints
    if parser.headers and len(parser.headers) == len(parser.rows[0]):
        df = pd.DataFrame(parser.rows, columns=parser.headers)
    else:
        df = pd.DataFrame(parser.rows)

    return df


# --- News / Web ---

def fetch_gdelt_docs(query: str, start_datetime: str, end_datetime: str, max_records: int = 150) -> pd.DataFrame:
    """
    Fetch GDELT Doc API results with retries and graceful fallback on 429s.
    Returns an empty DataFrame on persistent errors so callers can degrade to neutral.
    """
    import time
    from urllib3.util.retry import Retry
    from requests.adapters import HTTPAdapter

    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "JSON",
        "maxrecords": max_records,        # keep modest to reduce throttling
        "sort": "DateDesc",
        "startdatetime": start_datetime,
        "enddatetime": end_datetime,
    }

    s = requests.Session()
    retries = Retry(
        total=4,
        connect=3,
        read=3,
        backoff_factor=1.2,              # 0s, 1.2s, 2.4s, 3.6s
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))

    try:
        r = s.get(url, params=params, timeout=60)
        if r.status_code == 429:
            # final defensive wait + one last light request
            time.sleep(2.5)
            params["maxrecords"] = min(50, max_records)
            r = s.get(url, params=params, timeout=60)
        r.raise_for_status()
        data = r.json().get("articles", [])
        df = pd.DataFrame(data)
        if "seendate" in df.columns:
            df["seendate"] = pd.to_datetime(df["seendate"], errors="coerce")
        # Normalize tone to float if present
        if "tone" in df.columns:
            df["tone"] = pd.to_numeric(df["tone"], errors="coerce")
        return df
    except Exception:
        # On any persistent error, degrade gracefully
        return pd.DataFrame()


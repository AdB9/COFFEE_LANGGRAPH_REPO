# src/agents/web_news_agent.py
# src/agents/web_news_agent.py
from __future__ import annotations
import os, time, unicodedata, re, json
from typing import Dict, Any, List
import pandas as pd
import requests

from src.state import GraphState
from src.utils.io import read_csv_if_exists

# ---------- Brave News Search ----------
BRAVE_NEWS_ENDPOINT = "https://api.search.brave.com/res/v1/news/search"
QUERY_GROUPS: List[str] = [
    "(coffee OR arabica OR robusta)",
    "(frost OR freeze OR \"cold snap\" OR geada OR helada)",
    "(drought OR seca OR sequia)",
    "(strike OR protest OR blockade)",
    "(shipping OR freight OR port OR congestion)",
    "(\"El Nino\" OR \"La Nina\")",
]

def _ascii(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")

def _build_query() -> str:
    q = " AND ".join(QUERY_GROUPS)
    q = re.sub(r"\s+", " ", q).strip()
    return q

def _brave_news(q: str, start: pd.Timestamp, end: pd.Timestamp, country: str = "us", count: int = 30) -> pd.DataFrame:
    token = os.getenv("BRAVE_API_KEY") or os.getenv("BRAVE_SEARCH_API_KEY")
    if not token:
        raise RuntimeError("Set BRAVE_API_KEY in your environment.")
    count = min(int(count), 50)

    # Freshness helper
    days = (end - start).days
    if days <= 1:
        freshness = "pd"
    elif days <= 7:
        freshness = "pw"
    elif days <= 31:
        freshness = "pm"
    else:
        freshness = f"{start.strftime('%Y-%m-%d')}to{end.strftime('%Y-%m-%d')}"

    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": token,
        "User-Agent": "coffee-langgraph/0.1",
    }
    params = {
        "q": _ascii(q),
        "country": country,
        "search_lang": "en",
        "ui_lang": "en-US",
        "count": count,
        "freshness": freshness,
        "extra_snippets": "true",
        "safesearch": "moderate",
    }

    # Retry on 429/5xx
    backoff = 1.0
    for _ in range(5):
        r = requests.get(BRAVE_NEWS_ENDPOINT, headers=headers, params=params, timeout=30)
        if r.status_code == 200:
            data = r.json() or {}
            items = data.get("results") or []
            rows = []
            for it in items:
                # NOTE: Brave returns:
                #   "age": "4 days ago"  (string)
                #   "page_age": "2025-08-19T18:38:16" (ISO timestamp string)
                published = it.get("page_age") or it.get("age")
                meta_url = it.get("meta_url") or {}
                if isinstance(meta_url, str):
                    source_host = meta_url
                else:
                    source_host = (meta_url or {}).get("hostname")

                rows.append({
                    "title": it.get("title"),
                    "url": it.get("url"),
                    "published": published,             # string; we'll parse later
                    "source": source_host,
                    "snippet": it.get("description"),
                    "extra_snippets": it.get("extra_snippets") or [],
                })
            return pd.DataFrame(rows)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(backoff); backoff *= 2; continue
        r.raise_for_status()

    return pd.DataFrame()

# ---------- Simple heuristic tone (backup) ----------
NEG_KW = ("frost", "freeze", "strike", "blockade", "congestion", "delay", "shortage", "drought", "crop loss", "damage")
POS_KW = ("bumper", "surplus", "recovery", "relief", "rainfall", "improves")

def _kw_score(s: str) -> int:
    s = (s or "").lower()
    return sum(w in s for w in NEG_KW) - sum(w in s for w in POS_KW)

# ---------- LLM Assessor ----------
def _assess_with_llm(df: pd.DataFrame, period_days: int) -> Dict[str, Any]:
    """
    Uses an LLM to assess market impact from news. Returns a dict:
    {
      "impact": "bullish|bearish|neutral",
      "confidence": float 0-1,
      "rationale": "...",
      "drivers": [list of short bullets],
      "top_urls": [url1, url2, url3]
    }
    Falls back to {} if no model/key available or errors occur.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {}  # no LLM available

    try:
        # Lazy import so the package is optional
        from langchain_openai import ChatOpenAI
    except Exception:
        return {}

    # Build compact evidence (cap length)
    sample = df.copy()
    # Prefer most recent; parse published when possible
    sample["published_dt"] = pd.to_datetime(sample["published"], errors="coerce")
    sample = sample.sort_values(["published_dt"], ascending=False, na_position="last").head(20)

    evidence_lines = []
    for _, r in sample.iterrows():
        line = f"- {str(r.get('published') or '')} | {r.get('source') or ''} | {r.get('title') or ''} :: {r.get('snippet') or ''} | {r.get('url') or ''}"
        # keep each line fairly short
        evidence_lines.append(line[:500])

    system = (
        "You are a commodities analyst specializing in coffee. "
        "Read the headlines/snippets below (last {d} days). "
        "Decide whether the net impact on coffee prices is BULLISH, BEARISH, or NEUTRAL for the next short period. "
        "Weigh weather (frost/drought), supply (harvests, inventories), logistics (ports, freight), and macro demand. "
        "Be conservative; avoid overreacting to a single article."
    ).format(d=period_days)

    user = (
        "Evidence:\n{ev}\n\n"
        "Return STRICT JSON with keys: impact (bullish|bearish|neutral), confidence (0-1), "
        "rationale (<= 50 words), drivers (array of <=4 short bullets), top_urls (array of up to 3 URLs).\n"
        "Example:\n"
        "{{\"impact\":\"bullish\",\"confidence\":0.68,\"rationale\":\"...\",\"drivers\":[\"...\"],\"top_urls\":[\"...\"]}}"
    ).format(ev="\n".join(evidence_lines))

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=api_key)
    msg = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
    text = (msg.content or "").strip()

    try:
        j = json.loads(text)
        # Basic validation
        if isinstance(j, dict) and "impact" in j and "confidence" in j:
            # normalize impact
            imp = str(j.get("impact", "")).lower().strip()
            if imp not in ("bullish", "bearish", "neutral"):
                return {}
            j["impact"] = imp
            j["confidence"] = float(j.get("confidence") or 0.5)
            j["rationale"] = str(j.get("rationale") or "")[:300]
            j["drivers"] = list(j.get("drivers") or [])[:4]
            j["top_urls"] = list(j.get("top_urls") or [])[:3]
            return j
    except Exception:
        pass

    return {}

# ---------- Main node ----------
def web_news_agent(state: GraphState) -> Dict[str, Any]:
    path = state.get("news_csv")
    period_days = int(state.get("period_days", 2))
    end = pd.Timestamp.utcnow()
    start = end - pd.Timedelta(days=period_days)

    # (1) CSV override
    if path:
        df = read_csv_if_exists(path)
        if df is not None and not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            latest_date = df["date"].max()
            latest_df = df[df["date"] == latest_date]
            total = float(pd.to_numeric(latest_df["count"], errors="coerce").sum())
            avg = float(pd.to_numeric(latest_df.get("avg_sentiment", 0), errors="coerce").mean())
            features = {"latest_date": str(latest_date.date()), "total_count": total, "avg_sentiment": avg}
            # Basic heuristic classification
            if total >= 100 and avg <= -0.1:
                impact, conf, reason = "bullish", 0.6, "High volume with negative tone (supply risk)."
            elif total >= 100 and avg > 0.1:
                impact, conf, reason = "bearish", 0.55, "High volume with positive tone (relief/surplus)."
            else:
                impact, conf, reason = "neutral", 0.45, "Mixed or limited signals."
            return {"web_news_signal": {
                "impact": impact, "confidence": conf, "rationale": reason, "features": features
            }}

    # (2) Live Brave
    q = _build_query()
    df = _brave_news(q, start, end, country="us", count=30)

    if df is None or df.empty:
        return {"web_news_signal": {
            "impact": "neutral", "confidence": 0.4,
            "rationale": "No recent relevant news or API limit reached.", "features": {}
        }}

    # Heuristic tone proxy
    df["score"] = df["snippet"].map(_kw_score)
    total_count = float(len(df))
    avg_sent = float(df["score"].mean())
    latest_date = pd.to_datetime(df["published"], errors="coerce").max()

    features = {
        "latest_date": str((latest_date or end).date()),
        "total_count": total_count,
        "avg_tone_proxy": avg_sent,
        "top_sources": list(df["source"].value_counts().head(5).index),
        "sample_titles": df["title"].head(5).fillna("").tolist(),
    }

    # Heuristic impact
    if total_count >= 100 and avg_sent <= -0.1:
        h_impact, h_conf, h_reason = "bullish", 0.65, "High volume of supply-risk coverage with negative tone."
    elif total_count >= 100 and avg_sent > 0.1:
        h_impact, h_conf, h_reason = "bearish", 0.55, "High volume with positive tone (relief/bumper signals)."
    else:
        h_impact, h_conf, h_reason = "neutral", 0.45, "Mixed/limited signals."

    # (3) LLM assessor (optional)
    llm_judgment = _assess_with_llm(df, period_days)
    if llm_judgment:
        # Blend LLM confidence with heuristic for stability
        blended_conf = float((h_conf + float(llm_judgment.get("confidence", 0.5))) / 2.0)
        reason = f"{llm_judgment.get('rationale','').strip()} (LLM) | {h_reason} (heuristic)"
        features["llm_drivers"] = llm_judgment.get("drivers", [])
        features["llm_top_urls"] = llm_judgment.get("top_urls", [])
        impact = llm_judgment.get("impact", h_impact)
        return {"web_news_signal": {
            "impact": impact, "confidence": blended_conf, "rationale": reason, "features": features
        }}

    # Fallback: heuristic only
    return {"web_news_signal": {
        "impact": h_impact, "confidence": h_conf, "rationale": h_reason, "features": features
    }}

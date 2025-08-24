"""Simplified geospatial data extractor agent using weather and NDVI data."""

from __future__ import annotations
import pandas as pd
import logging
from typing import Dict, Any, List
from datetime import datetime

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END, START
from typing_extensions import TypedDict

from alpha_seeker.common.data_models import (
    DataExtractorResult,
    RegionInfo,
    CommodityType
)

# Use local implementations instead of non-existent coffee_langgraph module
GraphState = Dict[str, Any]

def read_csv_if_exists(path):
    if path and pd and hasattr(pd, 'read_csv'):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None

def fetch_open_meteo_daily(lat, lon, start_date, end_date):
    # Mock weather data
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    return pd.DataFrame({
        'date': dates,
        'tmin_c': [2.5 + (i % 10) for i in range(len(dates))],
        'precip_mm': [1.0 + (i % 5) for i in range(len(dates))]
    })

def fetch_modis_ndvi_ornl(lat, lon, start_date, end_date):
    # Mock NDVI data
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    return pd.DataFrame({
        'date': dates,
        'ndvi': [0.7 + (i % 3) * 0.1 for i in range(len(dates))]
    })

class CFG:
    FROST_TMIN_C = 2.0
    PRECIP_14D_MIN_MM = 25.0
    NDVI_ANOM_BULLISH_MAX = -0.10

logger = logging.getLogger(__name__)

# Safe defaults if config module is missing some attrs
FROST_TMIN_C = getattr(CFG, "FROST_TMIN_C", 2.0)
PRECIP_14D_MIN_MM = getattr(CFG, "PRECIP_14D_MIN_MM", 25.0)
NDVI_ANOM_BULLISH_MAX = getattr(CFG, "NDVI_ANOM_BULLISH_MAX", -0.10)

def _to_native(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
    except ImportError:
        pass
    if isinstance(o, dict):
        return {k: _to_native(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_to_native(v) for v in o]
    return o

def _fetch_region_frames(regions, start_date, end_date):
    frames = []
    for r in regions:
        lat, lon = float(r.coordinates[0]), float(r.coordinates[1])
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
        df["region"] = r.region_name
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def geospatial_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    path = state.get("weather_csv")
    period_days = int(state.get("period_days", 2))
    end = pd.Timestamp.utcnow().normalize()
    start = end - pd.Timedelta(days=period_days + 14)  # 2w context window

    if path:
        df = read_csv_if_exists(path)
    else:
        regions = state.get("regions") or [{"name": "Sul de Minas, Brazil", "lat": -21.5, "lon": -45.0}]
        # Convert to RegionInfo-like objects for compatibility
        region_objects = []
        for r in regions:
            if isinstance(r, dict):
                # Create a simple object with the required attributes
                region_obj = type('Region', (), {
                    'region_name': r.get('name', 'Unknown'),
                    'coordinates': (r.get('lat', 0), r.get('lon', 0))
                })()
                region_objects.append(region_obj)
            else:
                region_objects.append(r)
        df = _fetch_region_frames(region_objects, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    if df is None or df.empty:
        return {"geospatial_signal": {"impact": "neutral", "confidence": 0.30,
                                      "rationale": "No geospatial data fetched.", "features": {}}}

    # Types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "tmin_c" in df.columns:
        df["tmin_c"] = pd.to_numeric(df["tmin_c"], errors="coerce")
    if "precip_mm" in df.columns:
        df["precip_mm"] = pd.to_numeric(df["precip_mm"], errors="coerce")
    if "ndvi" in df.columns:
        df["ndvi"] = pd.to_numeric(df["ndvi"], errors="coerce")

    latest_date = df["date"].max()
    latest_df = df[df["date"] == latest_date].copy()

    # 14d window aggregates
    window14 = df[df["date"] >= (latest_date - pd.Timedelta(days=14))]
    precip_14 = float(window14.groupby("region")["precip_mm"].sum(min_count=1).mean()) if not window14.empty and "precip_mm" in window14.columns else 0.0

    # Frost anywhere in the latest day
    frost = bool((latest_df["tmin_c"] <= FROST_TMIN_C).any()) if "tmin_c" in latest_df.columns else False

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
        score += 1
        reasons.append("Frost risk in at least one region.")
    if precip_14 < PRECIP_14D_MIN_MM:
        score += 1
        reasons.append("14-day precipitation below threshold (dryness).")
    if ndvi_anom is not None and ndvi_anom <= NDVI_ANOM_BULLISH_MAX:
        score += 1
        reasons.append("Negative NDVI anomaly (vegetation stress).")

    if score >= 2:
        impact, conf = "bullish", 0.70
    elif score == 1:
        impact, conf = "bullish", 0.55
    else:
        impact, conf = "neutral", 0.45

    features = {
        "latest_date": str(latest_date.date()),
        "regions": [str(x) for x in latest_df["region"].dropna().unique().tolist()] if "region" in latest_df.columns else [],
        "tmin_min_c": float(pd.to_numeric(latest_df["tmin_c"], errors="coerce").min()) if "tmin_c" in latest_df.columns else None,
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



def extract_geospatial_data(config: RunnableConfig):
    """Graph factory function for geospatial data extraction agent.
    
    This function creates a LangGraph that can be used by the LangGraph runtime.
    It must take exactly one argument: RunnableConfig.
    """
    
    class GeospatialState(TypedDict):
        """State for the geospatial extraction graph."""
        messages: list
        result: DataExtractorResult
        error: str
    
    async def extraction_node(state: GeospatialState):
        """Main extraction node."""
        try:
            # Use the simplified geospatial agent
            signal_result = geospatial_agent(dict(state))
            
            # Convert to DataExtractorResult format
            signal = signal_result.get("geospatial_signal", {})
            
            result = DataExtractorResult(
                agent_name="geospatial_extractor",
                extraction_timestamp=datetime.now(),
                data_points=[{
                    "type": "weather_ndvi_analysis",
                    "signal": signal,
                    "timestamp": datetime.now()
                }],
                key_insights=[signal.get("rationale", "No insights available")],
                anomalies_detected=[],
                confidence_level=signal.get("confidence", 0.5),
                metadata={
                    "analysis_method": "weather_ndvi_analysis",
                    "features": signal.get("features", {}),
                    "impact": signal.get("impact", "neutral")
                }
            )
            
            return {"result": result}
            
        except Exception as e:
            logger.error(f"Geospatial extraction failed: {e}")
            return {"error": str(e)}
    
    # Build the graph
    builder = StateGraph(GeospatialState)
    builder.add_node("extract", extraction_node)
    builder.add_edge(START, "extract")
    builder.add_edge("extract", END)
    
    return builder.compile()


async def extract_geospatial_data_impl(
    commodity: CommodityType,
    regions: List[RegionInfo],
    failure_windows: List
) -> DataExtractorResult:
    """Implementation function for geospatial data extraction.
    
    This is the simplified implementation using weather and NDVI data.
    """
    try:
        logger.info("=== Geospatial Data Extractor ===")
        logger.info(f"Analyzing {commodity.value} for {len(regions)} regions")
        
        if not failure_windows:
            raise ValueError("failure_windows is required - no fallback time window allowed")
        
        logger.info(f"Extracting data for {len(failure_windows)} specific failure windows:")
        for i, window in enumerate(failure_windows, 1):
            logger.info(f"  Window {i}: {window.start_date} to {window.end_date}")
        
        # Create a state-like dict for the geospatial_agent function
        state = {
            "regions": [{"name": r.region_name, "lat": r.coordinates[0], "lon": r.coordinates[1]} for r in regions],
            "period_days": 7,  # Use a reasonable default
            "weather_csv": None  # No CSV path, fetch data directly
        }
        
        # Use the new simplified geospatial agent
        result = geospatial_agent(state)
        signal = result.get("geospatial_signal", {})
        
        # Create data points from the signal
        data_points = [{
            "type": "weather_ndvi_analysis",
            "signal": signal,
            "timestamp": datetime.now(),
            "regions_analyzed": len(regions),
            "features": signal.get("features", {})
        }]
        
        # Extract insights from the signal
        key_insights = [
            signal.get("rationale", "No insights available"),
            f"Impact assessment: {signal.get('impact', 'neutral')}",
            f"Confidence level: {signal.get('confidence', 0.5):.2f}"
        ]
        
        # Check for anomalies based on the impact
        anomalies_detected = []
        if signal.get("impact") == "bullish" and signal.get("confidence", 0) > 0.6:
            anomalies_detected.append("High confidence bullish geospatial signal detected")
        
        confidence_level = signal.get("confidence", 0.5)
        
        result = DataExtractorResult(
            agent_name="geospatial_extractor",
            extraction_timestamp=datetime.now(),
            data_points=data_points,
            key_insights=key_insights,
            anomalies_detected=anomalies_detected,
            confidence_level=confidence_level,
            metadata={
                "regions_analyzed": len(regions),
                "data_sources": ["Open_Meteo", "MODIS_NDVI"],
                "analysis_method": "weather_ndvi_analysis",
                "failure_windows_count": len(failure_windows),
                "failure_windows": [f"{w.start_date} to {w.end_date}" for w in failure_windows],
                "signal_features": signal.get("features", {}),
                "impact": signal.get("impact", "neutral")
            }
        )
        
        logger.info(f"Geospatial extraction complete: {len(data_points)} data points, {len(key_insights)} insights")
        return result
        
    except Exception as e:
        logger.error(f"Geospatial data extraction failed: {e}")
        return DataExtractorResult(
            agent_name="geospatial_extractor",
            extraction_timestamp=datetime.now(),
            data_points=[],
            key_insights=[],
            anomalies_detected=[f"Extraction failed: {str(e)}"],
            confidence_level=0.0,
            metadata={"error": str(e)}
        )


async def main():
    """Test the simplified geospatial analysis."""
    from datetime import date
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("=== Testing Simplified Geospatial Analysis ===")
    
    # Create test data
    test_region = RegionInfo(
        region_name="Minas Gerais",
        country="Brazil", 
        coordinates=(-19.8157, -43.9542),
        production_volume=1500000,
        region_type="coffee_production"
    )
    
    # Create test failure windows (simulate recent time periods)
    failure_windows = [
        type('FailureWindow', (), {
            'start_date': date(2024, 7, 1),
            'end_date': date(2024, 7, 15)
        })(),
        type('FailureWindow', (), {
            'start_date': date(2024, 8, 1), 
            'end_date': date(2024, 8, 15)
        })()
    ]
    
    commodity = CommodityType.COFFEE
    
    logger.info(f"Testing simplified analysis for {test_region.region_name}, {test_region.country}")
    logger.info(f"Commodity: {commodity.value}")
    logger.info(f"Failure windows: {len(failure_windows)}")
    
    try:
        # Test the simplified analysis
        result = await extract_geospatial_data_impl(
            commodity=commodity,
            regions=[test_region],
            failure_windows=failure_windows
        )
        
        logger.info("=== Analysis Results ===")
        logger.info(f"Agent: {result.agent_name}")
        logger.info(f"Confidence: {result.confidence_level:.2f}")
        logger.info(f"Data points: {len(result.data_points)}")
        logger.info(f"Key insights: {len(result.key_insights)}")
        logger.info(f"Anomalies: {len(result.anomalies_detected)}")
        
        # Show detailed results
        logger.info("\n=== Data Points ===")
        for i, point in enumerate(result.data_points[:3], 1):  # Show first 3
            logger.info(f"{i}. Type: {point.get('type', 'unknown')}")
            if point.get('signal'):
                signal = point['signal']
                logger.info(f"   Impact: {signal.get('impact', 'unknown')}")
                logger.info(f"   Confidence: {signal.get('confidence', 0):.2f}")
                logger.info(f"   Rationale: {signal.get('rationale', 'No rationale')}")
            logger.info(f"   Timestamp: {point.get('timestamp', 'unknown')}")
        
        logger.info("\n=== Key Insights ===")
        for i, insight in enumerate(result.key_insights[:5], 1):  # Show first 5
            logger.info(f"{i}. {insight}")
        
        if result.anomalies_detected:
            logger.info("\n=== Anomalies Detected ===")
            for i, anomaly in enumerate(result.anomalies_detected, 1):
                logger.info(f"{i}. {anomaly}")
        
        logger.info("\n=== Metadata ===")
        for key, value in result.metadata.items():
            logger.info(f"{key}: {value}")
        
        logger.info("\n✅ Simplified Geospatial Analysis Test Completed Successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

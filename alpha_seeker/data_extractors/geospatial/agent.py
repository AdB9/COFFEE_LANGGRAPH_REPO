"""Geospatial data extractor agent with VLM-based satellite image analysis."""

import logging
import os
import base64
import asyncio
from io import BytesIO
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

# New imports for Sentinel Hub and image processing
import numpy as np
from PIL import Image
from sentinelhub.config import SHConfig
from sentinelhub.api.process import SentinelHubRequest
from sentinelhub.data_collections import DataCollection
from sentinelhub.constants import MimeType, CRS
from sentinelhub.geometry import BBox
from sentinelhub.geo_utils import bbox_to_dimensions
from dotenv import load_dotenv

from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, START

from alpha_seeker.common.data_models import (
    DataExtractorResult,
    GeospatialData,
    GeospatialDataType,
    RegionInfo,
    CommodityType
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# --- Sentinel Hub Configuration ---
def get_sentinel_config() -> SHConfig:
    """Get Sentinel Hub configuration with credentials from environment variables."""
    config = SHConfig()
    
    # Set credentials from environment variables
    sh_client_id = os.getenv('SH_CLIENT_ID')
    sh_client_secret = os.getenv('SH_CLIENT_SECRET')
    
    if sh_client_id and sh_client_secret:
        config.sh_client_id = sh_client_id
        config.sh_client_secret = sh_client_secret
        logger.info(f"✅ Sentinel Hub credentials loaded successfully (Client ID: {sh_client_id[:8]}...)")
    else:
        logger.warning("SH_CLIENT_ID or SH_CLIENT_SECRET environment variables are not set.")
        logger.error("Sentinel Hub credentials not found. Please set SH_CLIENT_ID and SH_CLIENT_SECRET environment variables.")
    
    return config



async def _analyze_satellite_imagery_with_vlm(
    region: RegionInfo,
    start_date: datetime,
    end_date: datetime,
    commodity: CommodityType,
    window_idx: int = 0
) -> Dict[str, Any]:
    """Analyze satellite imagery using Vision Language Model for the specified region and time window."""
    
    try:
        logger.info(f"Analyzing satellite imagery for {region.region_name}, {region.country} ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")
        
        # Get satellite imagery from Sentinel Hub
        satellite_images = await _get_satellite_imagery_from_sentinelhub(region, start_date, end_date)
        
        if not satellite_images:
            logger.warning(f"No satellite imagery available for {region.region_name}")
            return _generate_fallback_analysis(region, start_date, end_date, window_idx)
        
        # Initialize VLM
        vlm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", # Updated to a powerful VLM
            temperature=0.1,
            max_output_tokens=2048
        )
        
        # Analyze each satellite image
        all_analyses = []
        for img_data in satellite_images:
            analysis = await _analyze_single_image_with_vlm(vlm, img_data, region, commodity)
            if analysis:
                all_analyses.append(analysis)
        
        # Synthesize all analyses into final result
        return _synthesize_vlm_analyses(all_analyses, region, start_date, end_date, window_idx, commodity)
        
    except Exception as e:
        logger.error(f"VLM satellite analysis failed for {region.region_name}: {e}")
        import traceback
        traceback.print_exc()
        return _generate_fallback_analysis(region, start_date, end_date, window_idx)


async def _get_satellite_imagery_from_sentinelhub(region: RegionInfo, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
    """Get Sentinel-2 satellite imagery for the specified region and time period."""
    
    # Define a bounding box around the region's coordinates (approx. 10km x 10km)
    lat, lon = region.coordinates
    bbox_size = 0.05  # degrees
    bbox_coords = (lon - bbox_size, lat - bbox_size, lon + bbox_size, lat + bbox_size)
    bbox = BBox(bbox_coords, crs=CRS.WGS84)
    size = bbox_to_dimensions(bbox, resolution=10) # 10m resolution for Sentinel-2

    # Simple evalscript to return a true-color (RGB) image
    evalscript_true_color = """
    //VERSION=3
    function setup() {
        return {
            input: ["B04", "B03", "B02"],
            output: { bands: 3 }
        };
    }
    function evaluatePixel(sample) {
        return [2.5 * sample.B04, 2.5 * sample.B03, 2.5 * sample.B02];
    }
    """

    images = []
    
    # Create a request for each week in the time window to get a time-series
    current_date = start_date
    while current_date <= end_date:
        request_start_date = current_date
        request_end_date = min(current_date + timedelta(days=7), end_date)
        time_interval = (request_start_date.strftime('%Y-%m-%d'), request_end_date.strftime('%Y-%m-%d'))
        
        logger.info(f"Requesting Sentinel-2 data for {time_interval}")
        
        # Get Sentinel Hub configuration with credentials
        config = get_sentinel_config()
        
        request = SentinelHubRequest(
            evalscript=evalscript_true_color,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=time_interval,
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
            bbox=bbox,
            size=size,
            config=config,
        )
        
        try:
            # get_data returns a list of images, we expect one mosaic
            image_data_list = await asyncio.to_thread(request.get_data)
            if image_data_list and len(image_data_list[0]) > 0:
                image_array = image_data_list[0]
                images.append({
                    "image_array": image_array,
                    "date": request_start_date,
                    "type": "true_color_sentinel2_l2a",
                    "source": "Copernicus/Sentinel-2",
                    "resolution": "10m"
                })
        except Exception as e:
            logger.warning(f"Could not retrieve image for {time_interval}: {e}")

        current_date += timedelta(days=7)
        if len(images) >= 4:  # Limit to 4 images per analysis
            break
            
    return images





async def _analyze_single_image_with_vlm(
    vlm: ChatGoogleGenerativeAI,
    img_data: Dict[str, Any],
    region: RegionInfo,
    commodity: CommodityType
) -> Optional[Dict[str, Any]]:
    """Analyze a single satellite image numpy array using the VLM."""
    
    try:
        image_array = img_data["image_array"]
        
        # Convert numpy array to a PNG image in memory
        image = Image.fromarray(image_array)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        
        # Encode image to base64
        image_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Create VLM analysis prompt
        prompt = f"""You are an expert remote sensing analyst specializing in agriculture. Analyze this satellite image of {region.region_name}, {region.country} captured around {img_data['date'].strftime('%Y-%m-%d')}.

This region is critical for {commodity.value} production. Your detailed analysis is required to identify factors that could impact market prices.

Focus on these key areas:
1.  **Vegetation Health & Vigor**: Assess the greenness and density of vegetation in agricultural plots. Use the Normalized Difference Vegetation Index (NDVI) principles if possible (dark green vs. light green/brown). Are there signs of widespread stress, disease, or drought?
2.  **Soil Moisture & Water Bodies**: Observe the color of bare soil (darker indicates moisture). Are rivers, lakes, or reservoirs full or depleted compared to expectations?
3.  **Cloud Cover & Atmospheric Conditions**: Note any significant cloud cover, haze, or shadows that might obscure the view or indicate recent weather events (e.g., heavy rain).
4.  **Land Use Patterns**: Confirm the presence of agricultural activity. Are there any visible signs of recent harvesting, planting, or land clearing?

Provide a structured analysis:
- **Overall Assessment**: A one-sentence summary of the conditions.
- **Key Observations**: A bulleted list of specific, quantifiable observations (e.g., "Approximately 20% of the visible farmland appears stressed with a brownish tint," or "River levels seem adequate for this time of year.").
- **Potential Impact on {commodity.value}**: Explain how your observations could logically affect the upcoming {commodity.value} yield (e.g., "Widespread vegetation stress suggests a potentially lower-than-expected harvest, which could lead to supply constraints.").
- **Confidence Level**: Rate your confidence in this analysis from 0.0 (uncertain) to 1.0 (highly confident).

Image metadata:
- Date: {img_data['date'].strftime('%Y-%m-%d')}
- Source: {img_data['source']}
- Resolution: {img_data['resolution']}
"""

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                }
            ]
        )
        
        response = await vlm.ainvoke([message])
        analysis_text = response.content
        
        logger.info(f"VLM analysis completed for {region.region_name} on {img_data['date'].strftime('%Y-%m-%d')}")
        
        return {
            "date": img_data["date"],
            "analysis": analysis_text,
            "image_metadata": {k: v for k, v in img_data.items() if k != 'image_array'}, # Exclude array
            "region": region.region_name,
            "country": region.country
        }
        
    except Exception as e:
        logger.error(f"VLM analysis failed for single image: {e}")
        return None


def _synthesize_vlm_analyses(
    analyses: List[Dict[str, Any]], 
    region: RegionInfo, 
    start_date: datetime, 
    end_date: datetime,
    window_idx: int,
    commodity: CommodityType
) -> Dict[str, Any]:
    """Synthesize multiple VLM analyses into structured data points and insights."""
    
    if not analyses:
        return _generate_fallback_analysis(region, start_date, end_date, window_idx)
    
    data_points = []
    insights = []
    anomalies = []
    
    window_context = f" (Window {window_idx + 1}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})" if window_idx > 0 or start_date != end_date else ""
    
    # Extract key information from VLM analyses
    for analysis in analyses:
        analysis_text = analysis["analysis"]
        analysis_date = analysis["date"]
        
        # Create data points from VLM analysis
        data_points.append({
            "type": "vlm_satellite_analysis",
            "region": region.region_name,
            "analysis_text": analysis_text,
            "timestamp": analysis_date,
            "coordinates": region.coordinates,
            "source": "VLM_Gemini_2.0_Flash",
            "image_source": analysis["image_metadata"]["source"]
        })
        
        # Extract insights from analysis text (simple keyword-based extraction)
        analysis_lower = analysis_text.lower()
        
        if "stress" in analysis_lower or "drought" in analysis_lower:
            insights.append(f"VLM detected vegetation stress indicators in {region.region_name} on {analysis_date.strftime('%Y-%m-%d')}")
        
        if "healthy" in analysis_lower and "vegetation" in analysis_lower:
            insights.append(f"VLM observed healthy vegetation patterns in {region.region_name} on {analysis_date.strftime('%Y-%m-%d')}")
        
        if "cloud" in analysis_lower and ("cover" in analysis_lower or "coverage" in analysis_lower):
            insights.append(f"VLM noted significant cloud coverage affecting visibility in {region.region_name} on {analysis_date.strftime('%Y-%m-%d')}")
        
        # Detect anomalies
        if any(keyword in analysis_lower for keyword in ["anomal", "unusual", "extreme", "severe"]):
            anomalies.append(f"VLM detected unusual conditions in {region.region_name} during satellite analysis{window_context}")
        
        if "flood" in analysis_lower or "excessive" in analysis_lower:
            anomalies.append(f"VLM identified potential flooding or excessive moisture in {region.region_name} on {analysis_date.strftime('%Y-%m-%d')}")
    
    # Add summary insights
    insights.append(f"Completed VLM satellite analysis for {region.region_name}{window_context} using {len(analyses)} satellite images")
    insights.append(f"Analysis covers {commodity.value} production region with detailed vegetation and environmental assessment")
    
    return {
        "data_points": data_points,
        "insights": insights,
        "anomalies": anomalies
    }


def _generate_fallback_analysis(region: RegionInfo, start_date: datetime, end_date: datetime, window_idx: int) -> Dict[str, Any]:
    """Generate fallback analysis when VLM analysis fails."""
    
    window_context = f" (Window {window_idx + 1}: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})" if window_idx > 0 or start_date != end_date else ""
    
    return {
        "data_points": [{
            "type": "fallback_analysis",
            "region": region.region_name,
            "message": "VLM satellite analysis unavailable - using fallback method",
            "timestamp": start_date,
            "coordinates": region.coordinates
        }],
        "insights": [f"Satellite analysis for {region.region_name}{window_context} requires manual review - VLM analysis failed"],
        "anomalies": [f"Unable to complete automated satellite analysis for {region.region_name}{window_context}"]
    }


def extract_geospatial_data(config: RunnableConfig):
    """Graph factory function for geospatial data extraction agent.
    
    This function creates a LangGraph that can be used by the LangGraph runtime.
    It must take exactly one argument: RunnableConfig.
    """
    from langgraph.graph import StateGraph, END, START
    from typing_extensions import TypedDict
    
    class GeospatialState(TypedDict):
        """State for the geospatial extraction graph."""
        messages: list
        result: DataExtractorResult
        error: str
    
    async def extraction_node(state: GeospatialState):
        """Main extraction node."""
        try:
            # For now, create a simple extraction with mock data
            # In a real implementation, this would parse input from messages
            # and extract parameters like commodity, regions, time windows
            
            from alpha_seeker.common.data_models import CommodityType, RegionInfo
            
            # Mock data for demonstration
            commodity = CommodityType.COFFEE
            regions = [
                RegionInfo(
                    region_name="Minas Gerais",
                    country="Brazil",
                    coordinates=(19.8157, 43.9542),
                    production_volume=1500000,
                    region_type="state"
                )
            ]
            # Create test failure windows
            from datetime import date
            failure_windows = [
                type('FailureWindow', (), {
                    'start_date': date.today() - timedelta(days=30),
                    'end_date': date.today() - timedelta(days=16)
                })(),
                type('FailureWindow', (), {
                    'start_date': date.today() - timedelta(days=15),
                    'end_date': date.today() - timedelta(days=1)
                })()
            ]
            
            result = await extract_geospatial_data_impl(
                commodity=commodity,
                regions=regions,
                failure_windows=failure_windows
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


# Rename the original function to avoid conflicts
async def extract_geospatial_data_impl(
    commodity: CommodityType,
    regions: List[RegionInfo],
    failure_windows: List
) -> DataExtractorResult:
    """Implementation function for geospatial data extraction.
    
    This is the original function renamed to avoid conflicts with the graph factory.
    """
    try:
        logger.info("=== Geospatial Data Extractor ===")
        logger.info(f"Analyzing {commodity.value} for {len(regions)} regions")
        
        # Always require failure windows - no fallback to generic time window
        if not failure_windows:
            raise ValueError("failure_windows is required - no fallback time window allowed")
        
        logger.info(f"Extracting data for {len(failure_windows)} specific failure windows:")
        for i, window in enumerate(failure_windows, 1):
            logger.info(f"  Window {i}: {window.start_date} to {window.end_date}")
        
        # TODO: Implement actual geospatial data extraction
        # This would integrate with:
        # - NASA Earth Observation data
        # - USGS satellite imagery
        # - Weather API services (OpenWeatherMap, NOAA)
        # - Agricultural monitoring services
        # - Remote sensing platforms
        
        data_points = []
        key_insights = []
        anomalies_detected = []
        
        for region in regions:
            logger.info(f"Processing region: {region.region_name}, {region.country}")
            
            # Extract data for each specific failure window using VLM satellite analysis
            for window_idx, window in enumerate(failure_windows):
                window_start = datetime.combine(window.start_date, datetime.min.time())
                window_end = datetime.combine(window.end_date, datetime.min.time())
                
                # Analyze satellite imagery with VLM for this specific window
                region_data = await _analyze_satellite_imagery_with_vlm(
                    region, window_start, window_end, commodity, window_idx
                )
                data_points.extend(region_data["data_points"])
                key_insights.extend(region_data["insights"])
                anomalies_detected.extend(region_data["anomalies"])
        
        confidence_level = 0.75  # TODO: Calculate based on data quality
        
        result = DataExtractorResult(
            agent_name="geospatial_extractor",
            extraction_timestamp=datetime.now(),
            data_points=data_points,
            key_insights=key_insights,
            anomalies_detected=anomalies_detected,
            confidence_level=confidence_level,
            metadata={
                "regions_analyzed": len(regions),
                "data_sources": ["Copernicus/Sentinel-2", "VLM_Gemini_1.5_Flash"],
                "analysis_method": "VLM_satellite_imagery_analysis",
                "failure_windows_count": len(failure_windows),
                "failure_windows": [f"{w.start_date} to {w.end_date}" for w in failure_windows],
                "vlm_model": "gemini-1.5-flash",
                "satellite_imagery_source": "Sentinel_Hub"
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
    """Test the VLM-based geospatial analysis with a random region."""
    from datetime import date
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info("=== Testing VLM Geospatial Analysis ===")
    
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
    
    logger.info(f"Testing VLM analysis for {test_region.region_name}, {test_region.country}")
    logger.info(f"Commodity: {commodity.value}")
    logger.info(f"Failure windows: {len(failure_windows)}")
    
    try:
        # Test the VLM analysis
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
            if point.get('analysis_text'):
                logger.info(f"   Analysis: {point['analysis_text'][:200]}...")
            logger.info(f"   Source: {point.get('source', 'unknown')}")
            logger.info(f"   Timestamp: {point.get('timestamp', 'unknown')}")
        
        logger.info("\n=== Key Insights ===")
        for i, insight in enumerate(result.key_insights[:5], 1):  # Show first 5
            logger.info(f"{i}. {insight}")
        
        if result.anomalies_detected:
            logger.info("\n=== Anomalies Detected ===")
            for i, anomaly in enumerate(result.anomalies_detected, 1):
                logger.info(f"{i}. {anomaly}")
        
        logger.info(f"\n=== Metadata ===")
        for key, value in result.metadata.items():
            logger.info(f"{key}: {value}")
        
        logger.info("\n✅ VLM Geospatial Analysis Test Completed Successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

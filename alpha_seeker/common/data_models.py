"""Common data models for the Alpha Seeker system."""

from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime

from pydantic import BaseModel, Field


class CommodityType(Enum):
    """Supported commodity types."""
    
    COFFEE = "coffee"
    SUGAR = "sugar"
    COCOA = "cocoa"
    WHEAT = "wheat"
    CORN = "corn"
    SOYBEANS = "soybeans"


class SearchResultType(Enum):
    """Type of search result."""
    
    NEWS_ARTICLE = "news_article"
    RESEARCH_PAPER = "research_paper"
    MARKET_REPORT = "market_report"
    GOVERNMENT_DATA = "government_data"
    SOCIAL_MEDIA = "social_media"
    GENERAL_WEB = "general_web"


class GeospatialDataType(Enum):
    """Type of geospatial data."""
    
    WEATHER_PATTERNS = "weather_patterns"
    SOIL_MOISTURE = "soil_moisture"
    VEGETATION_HEALTH = "vegetation_health"
    SATELLITE_IMAGERY = "satellite_imagery"
    PRECIPITATION = "precipitation"
    TEMPERATURE = "temperature"


class LogisticsDataType(Enum):
    """Type of logistics data."""
    
    PORT_CONGESTION = "port_congestion"
    SHIPPING_RATES = "shipping_rates"
    SUPPLY_CHAIN_DISRUPTION = "supply_chain_disruption"
    TRANSPORTATION_COSTS = "transportation_costs"
    INVENTORY_LEVELS = "inventory_levels"


class CommodityInput(BaseModel):
    """Input specification for the Alpha Seeker system."""
    
    commodity: CommodityType = Field(description="The commodity to analyze")
    analysis_period_days: int = Field(default=90, description="Number of days to analyze")
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence for indicators")


class RegionInfo(BaseModel):
    """Information about a commodity production region."""
    
    region_name: str = Field(description="Name of the region")
    country: str = Field(description="Country name")
    coordinates: tuple[float, float] = Field(description="Latitude and longitude")
    production_volume: float = Field(default=0.0, description="Production volume (to be filled by agents)")
    region_type: str = Field(default="production_area", description="Type of region (production_area, port, etc.)")
    # Optional legacy fields for backward compatibility
    production_percentage: Optional[float] = Field(default=None, description="Percentage of global production")
    key_factors: List[str] = Field(default_factory=list, description="Key factors affecting this region")


class WebSearchResult(BaseModel):
    """A single web search result."""
    
    title: str = Field(description="Title of the web page")
    url: str = Field(description="URL of the web page")
    snippet: str = Field(description="Brief description or snippet from the page")
    content: str = Field(default="", description="Full content if available")
    search_type: SearchResultType = Field(default=SearchResultType.GENERAL_WEB, description="Type of search result")
    relevance_score: float = Field(default=0.0, description="Relevance score for this result")
    timestamp: Optional[datetime] = Field(default=None, description="When this content was published")


class GeospatialData(BaseModel):
    """Geospatial data point."""
    
    data_type: GeospatialDataType = Field(description="Type of geospatial data")
    region: str = Field(description="Geographic region")
    coordinates: tuple[float, float] = Field(description="Latitude and longitude")
    value: float = Field(description="Data value")
    unit: str = Field(description="Unit of measurement")
    timestamp: datetime = Field(description="When this data was recorded")
    confidence: float = Field(default=0.0, description="Confidence in the data quality")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class LogisticsData(BaseModel):
    """Logistics data point."""
    
    data_type: LogisticsDataType = Field(description="Type of logistics data")
    location: str = Field(description="Location (port, route, etc.)")
    value: float = Field(description="Data value")
    unit: str = Field(description="Unit of measurement")
    timestamp: datetime = Field(description="When this data was recorded")
    impact_level: str = Field(description="Impact level: low, medium, high")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ResearchQuery(BaseModel):
    """A research query with associated web search results."""
    
    query: str = Field(description="The research query")
    search_results: List[WebSearchResult] = Field(
        default_factory=list,
        description="Web search results for this query"
    )
    context: str = Field(default="", description="Additional context for this query")
    priority: int = Field(default=1, description="Priority level (1=highest, 5=lowest)")


class TimeWindow(BaseModel):
    """Time window for analysis."""
    
    start_date: datetime = Field(description="Start date of the time window")
    end_date: datetime = Field(description="End date of the time window")
    lookback_days: int = Field(default=20, description="Number of days to look back from anomaly date")


class ModelPredictionError(BaseModel):
    """A model prediction error from evaluation_results.csv."""
    
    date: datetime = Field(description="Date of the prediction")
    actual: float = Field(description="Actual price")
    predicted: float = Field(description="Predicted price")
    absolute_delta: float = Field(description="Absolute difference")
    percentage_error: float = Field(description="Percentage error")
    is_huge_difference: bool = Field(description="Whether this is flagged as a huge difference")


class AlphaIndicator(BaseModel):
    """A proposed alpha indicator for improving model predictions."""
    
    name: str = Field(description="Short, descriptive name for the indicator")
    description: str = Field(description="Clear explanation of what the indicator measures")
    reasoning: str = Field(description="Justification for why this indicator could be predictive")
    confidence_score: float = Field(
        ge=0.0, le=1.0, 
        description="Confidence score between 0.0 and 1.0"
    )
    suggested_data_source: str = Field(description="Concrete suggestion on where to acquire this data")
    category: str = Field(description="Category of the indicator (e.g., weather, economic, social)")
    implementation_difficulty: int = Field(
        ge=1, le=5,
        description="Implementation difficulty (1=easy, 5=very difficult)"
    )
    correlation_evidence: str = Field(description="Evidence of correlation with price movements")
    time_window_effectiveness: List[str] = Field(description="Time windows where this indicator would be most effective")


class DataExtractorResult(BaseModel):
    """Result from a data extractor agent."""
    
    agent_name: str = Field(description="Name of the data extractor agent")
    extraction_timestamp: datetime = Field(description="When the extraction was performed")
    data_points: List[Dict[str, Any]] = Field(description="Extracted data points")
    key_insights: List[str] = Field(description="Key insights from the data")
    anomalies_detected: List[str] = Field(description="Any anomalies or unusual patterns detected")
    confidence_level: float = Field(description="Overall confidence in the extracted data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AnalysisContext(BaseModel):
    """Context for the data analyst combining all inputs."""
    
    commodity: CommodityType = Field(description="The commodity being analyzed")
    time_window: TimeWindow = Field(description="Analysis time window")
    model_errors: List[ModelPredictionError] = Field(description="Model prediction errors")
    geospatial_data: Optional[DataExtractorResult] = Field(description="Geospatial analysis results")
    logistics_data: Optional[DataExtractorResult] = Field(description="Logistics analysis results")
    web_news_data: Optional[DataExtractorResult] = Field(description="Web news analysis results")
    selected_regions: List[RegionInfo] = Field(description="Selected regions for analysis")
    price_history: List[Dict[str, Any]] = Field(description="Historical price data from dataset.csv")
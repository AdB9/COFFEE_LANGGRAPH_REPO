"""Web news data extractor agent - enhanced version of the original websearch agent."""

import logging
from typing import Dict, Any, List
from datetime import datetime

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END, START

from alpha_seeker.common.data_models import (
    DataExtractorResult,
    ResearchQuery,
    WebSearchResult,
    CommodityType,
    RegionInfo
)
from alpha_seeker.websearch_agent.tools.web_search import multi_query_search

logger = logging.getLogger(__name__)


async def extract_web_news_data_impl(
    commodity: CommodityType,
    regions: List[RegionInfo],
    failure_windows: List
) -> DataExtractorResult:
    """Extract web news and market intelligence for the specified commodity.
    
    This agent will analyze:
    - News articles about the commodity and regions
    - Market reports and industry analysis
    - Government policy announcements
    - Trade organization communications
    - Social media sentiment (future enhancement)
    - Expert opinions and forecasts
    """
    
    try:
        logger.info("=== Web News Data Extractor ===")
        logger.info(f"Analyzing news for {commodity.value} in {len(regions)} regions")
        
        # Always require failure windows - no fallback to generic time window
        if not failure_windows:
            raise ValueError("failure_windows is required - no fallback time window allowed")
        
        logger.info(f"Extracting data for {len(failure_windows)} specific failure windows:")
        for i, window in enumerate(failure_windows, 1):
            logger.info(f"  Window {i}: {window.start_date} to {window.end_date}")
        
        # Generate targeted search queries based on commodity, regions, and failure windows
        # Use the first failure window for query generation (could be enhanced to use all windows)
        first_window = failure_windows[0]
        start_date = datetime.combine(first_window.start_date, datetime.min.time())
        end_date = datetime.combine(first_window.end_date, datetime.min.time())
        search_queries = _generate_commodity_queries(commodity, regions, start_date, end_date)
        
        logger.info(f"Generated {len(search_queries)} search queries")
        for query in search_queries:
            logger.info(f"  - {query}")
        
        # Perform web searches
        search_result = await multi_query_search.ainvoke({
            "queries": search_queries,
            "max_results_per_query": 8,
            "search_engine": "duckduckgo"
        })
        
        # Process and analyze results
        data_points = []
        key_insights = []
        anomalies_detected = []
        
        if "error" not in search_result:
            analysis = _analyze_search_results(search_result, commodity, regions)
            data_points = analysis["data_points"]
            key_insights = analysis["insights"]
            anomalies_detected = analysis["anomalies"]
        
        confidence_level = 0.65  # TODO: Calculate based on source quality and relevance
        
        result = DataExtractorResult(
            agent_name="web_news_extractor",
            extraction_timestamp=datetime.now(),
            data_points=data_points,
            key_insights=key_insights,
            anomalies_detected=anomalies_detected,
            confidence_level=confidence_level,
            metadata={
                "queries_executed": len(search_queries),
                "total_results": search_result.get("total_results", 0),
                "data_sources": ["DuckDuckGo", "News_APIs", "Industry_Reports"],
                "failure_windows_count": len(failure_windows),
                "failure_windows": [f"{w.start_date} to {w.end_date}" for w in failure_windows]
            }
        )
        
        logger.info(f"Web news extraction complete: {len(data_points)} data points, {len(key_insights)} insights")
        return result
        
    except Exception as e:
        logger.error(f"Web news data extraction failed: {e}")
        return DataExtractorResult(
            agent_name="web_news_extractor",
            extraction_timestamp=datetime.now(),
            data_points=[],
            key_insights=[],
            anomalies_detected=[f"Extraction failed: {str(e)}"],
            confidence_level=0.0,
            metadata={"error": str(e)}
        )


def _generate_commodity_queries(
    commodity: CommodityType, 
    regions: List[RegionInfo], 
    start_date: datetime, 
    end_date: datetime,
    failure_context: str = None
) -> List[str]:
    """Generate targeted search queries for the commodity and regions."""
    
    queries = []
    commodity_name = commodity.value
    
    # General commodity queries
    queries.extend([
        f"{commodity_name} price volatility market analysis",
        f"{commodity_name} supply chain disruption news",
        f"{commodity_name} market forecast industry report",
        f"{commodity_name} trade policy government announcement"
    ])
    
    # Region-specific queries
    for region in regions[:3]:  # Limit to top 3 regions to avoid too many queries
        region_name = region.region_name
        country = region.country
        
        queries.extend([
            f"{commodity_name} production {region_name} {country} news",
            f"{country} {commodity_name} export policy trade",
            f"{region_name} weather impact {commodity_name} harvest"
        ])
    
    # Time-sensitive queries
    year = start_date.year
    month_year = start_date.strftime('%B %Y')
    queries.extend([
        f"{commodity_name} market outlook {year}",
        f"{commodity_name} price prediction {year} forecast",
        f"{commodity_name} volatility {month_year}",
        f"{commodity_name} price shock {year}"
    ])
    
    # Add specific date-based queries if we have failure context
    if failure_context and "FAILURE #" in failure_context:
        # Extract specific dates from failure context for targeted searches
        import re
        dates = re.findall(r'\d{4}-\d{2}-\d{2}', failure_context)
        for date in dates[:2]:  # Focus on first 2 failure dates
            date_obj = datetime.strptime(date, '%Y-%m-%d')
            month_year = date_obj.strftime('%B %Y')
            queries.extend([
                f"{commodity_name} news {month_year}",
                f"{commodity_name} market disruption {month_year}"
            ])
    
    return queries


def _analyze_search_results(
    search_result: Dict[str, Any], 
    commodity: CommodityType, 
    regions: List[RegionInfo]
) -> Dict[str, Any]:
    """Analyze search results to extract insights and detect anomalies."""
    
    all_results = search_result.get("all_results", [])
    
    data_points = []
    insights = []
    anomalies = []
    
    # Count results by type and relevance
    news_count = 0
    market_report_count = 0
    government_data_count = 0
    high_relevance_count = 0
    
    for result in all_results:
        if result.get("search_type") == "news_article":
            news_count += 1
        elif result.get("search_type") == "market_report":
            market_report_count += 1
        elif result.get("search_type") == "government_data":
            government_data_count += 1
        
        if result.get("relevance_score", 0) >= 0.7:
            high_relevance_count += 1
        
        # Convert to data point format
        data_points.append({
            "type": "news_article",
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "relevance_score": result.get("relevance_score", 0),
            "search_type": result.get("search_type", "general_web"),
            "timestamp": datetime.now().isoformat()
        })
    
    # Generate insights based on analysis
    insights.append(f"Found {len(all_results)} relevant articles about {commodity.value}")
    insights.append(f"News coverage: {news_count} articles, Market reports: {market_report_count}, Government sources: {government_data_count}")
    
    if high_relevance_count > 0:
        insights.append(f"{high_relevance_count} high-relevance sources identified")
    
    # Detect potential anomalies
    if news_count > 15:  # High news volume might indicate market volatility
        anomalies.append(f"Unusually high news volume detected for {commodity.value}")
    
    if market_report_count == 0:
        anomalies.append(f"Limited market analysis coverage found for {commodity.value}")
    
    # TODO: Implement more sophisticated analysis:
    # - Sentiment analysis of news articles
    # - Trend detection in news topics
    # - Cross-correlation with known market events
    # - Geographic clustering of news sources
    
    return {
        "data_points": data_points,
        "insights": insights,
        "anomalies": anomalies
    }


def extract_web_news_data(config: RunnableConfig):
    """Graph factory function for web news data extraction agent.
    
    This function creates a LangGraph that can be used by the LangGraph runtime.
    It must take exactly one argument: RunnableConfig.
    """
    from langgraph.graph import StateGraph, END, START
    from typing_extensions import TypedDict
    
    class WebNewsState(TypedDict):
        """State for the web news extraction graph."""
        messages: list
        result: DataExtractorResult
        error: str
    
    async def extraction_node(state: WebNewsState):
        """Main extraction node."""
        try:
            # For now, create a simple extraction with mock data
            # In a real implementation, this would parse input from messages
            # and extract parameters like commodity, regions, time windows
            
            from alpha_seeker.common.data_models import CommodityType, RegionInfo
            from datetime import timedelta
            
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
            time_window_start = datetime.now() - timedelta(days=30)
            time_window_end = datetime.now()
            
            result = await extract_web_news_data_impl(
                commodity=commodity,
                regions=regions,
                time_window_start=time_window_start,
                time_window_end=time_window_end
            )
            
            return {"result": result}
            
        except Exception as e:
            logger.error(f"Web news extraction failed: {e}")
            return {"error": str(e)}
    
    # Build the graph
    builder = StateGraph(WebNewsState)
    builder.add_node("extract", extraction_node)
    builder.add_edge(START, "extract")
    builder.add_edge("extract", END)
    
    return builder.compile()

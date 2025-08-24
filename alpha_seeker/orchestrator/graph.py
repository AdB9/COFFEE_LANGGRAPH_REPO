"""Main orchestrator graph for the Alpha Seeker system."""

import asyncio
import logging
from typing import cast

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from alpha_seeker.orchestrator.state import OrchestratorState
from alpha_seeker.orchestrator.nodes.input_processor import input_processor
from alpha_seeker.orchestrator.nodes.region_selector import region_selector
from alpha_seeker.orchestrator.nodes.failure_analyzer import failure_analyzer
from alpha_seeker.data_extractors.geospatial.agent import extract_geospatial_data_impl
from alpha_seeker.data_extractors.web_news.agent import extract_web_news_data_impl
from alpha_seeker.data_analyst.nodes.alpha_generator import alpha_generator

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def geospatial_extractor_node(state: OrchestratorState, config: RunnableConfig) -> dict:
    """Wrapper node for geospatial data extraction."""
    try:
        if not state.commodity_input or not state.selected_regions:
            return {"error": "Missing commodity input or regions for geospatial extraction"}
        
        # Always require failure windows - no fallback
        if not state.failure_windows:
            return {"error": "failure_windows is required for geospatial extraction - no fallback allowed"}
        
        result = await extract_geospatial_data_impl(
            commodity=state.commodity_input.commodity,
            regions=state.selected_regions,
            failure_windows=state.failure_windows
        )
        
        return {
            "geospatial_results": result,
            "current_agent": "geospatial_extractor"
        }
    except Exception as e:
        logger.error(f"Geospatial extraction node failed: {e}")
        return {"error": f"Geospatial extraction failed: {str(e)}"}




async def web_news_extractor_node(state: OrchestratorState, config: RunnableConfig) -> dict:
    """Wrapper node for web news data extraction."""
    try:
        if not state.commodity_input or not state.selected_regions:
            return {"error": "Missing commodity input or regions for web news extraction"}
        
        # Always require failure windows - no fallback
        if not state.failure_windows:
            return {"error": "failure_windows is required for web news extraction - no fallback allowed"}
        
        result = await extract_web_news_data_impl(
            commodity=state.commodity_input.commodity,
            regions=state.selected_regions,
            failure_windows=state.failure_windows
        )
        
        return {
            "web_news_results": result,
            "current_agent": "web_news_extractor"
        }
    except Exception as e:
        logger.error(f"Web news extraction node failed: {e}")
        return {"error": f"Web news extraction failed: {str(e)}"}


# Configuration class for the orchestrator
class OrchestratorConfiguration:
    """Configuration for the orchestrator."""
    
    def __init__(
        self,
        model: str = "google_genai:gemini-2.0-flash",
        analysis_depth: str = "comprehensive",
        parallel_extraction: bool = True,
        **kwargs
    ):
        self.model = model
        self.analysis_depth = analysis_depth
        self.parallel_extraction = parallel_extraction
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @classmethod
    def from_runnable_config(cls, config):
        """Create configuration from runnable config."""
        configurable = (config or {}).get("configurable", {})
        return cls(**configurable)


# Build the main orchestrator graph
builder = StateGraph(OrchestratorState)

# Add nodes to the graph
builder.add_node("input", input_processor)
builder.add_node("region_selector", region_selector)
builder.add_node("failure_analyzer", failure_analyzer)
builder.add_node("geospatial", geospatial_extractor_node)
builder.add_node("web_news", web_news_extractor_node)
builder.add_node("data_analyst", alpha_generator)

# Add edges following the diagram structure
builder.add_edge(START, "input")
builder.add_edge("input", "region_selector")
builder.add_edge("region_selector", "failure_analyzer")

# Parallel data extraction (after failure analysis)
builder.add_edge("failure_analyzer", "geospatial")
builder.add_edge("failure_analyzer", "web_news")

# All extractors feed into data analyst
builder.add_edge("geospatial", "data_analyst")
builder.add_edge("web_news", "data_analyst")

# End after analysis
builder.add_edge("data_analyst", END)

# Compile the graph
graph = builder.compile()
graph.name = "AlphaSeekerOrchestrator"


async def main():
    """Test the complete Alpha Seeker orchestrator."""
    logger.info("=== Testing Alpha Seeker Orchestrator ===")
    
    # Test configuration
    test_config = cast(RunnableConfig, {
        "configurable": {
            "thread_id": "alpha_seeker_test_001",
            "model": "google_genai:gemini-2.0-flash",
            "analysis_depth": "comprehensive",
            "parallel_extraction": True
        }
    })
    
    # Test queries
    test_queries = [
        "Analyze coffee market for alpha indicators over 90 days",
        "Find alpha indicators for coffee price prediction failures",
        "Sugar market analysis for supply chain disruption indicators",
        "Coffee production anomalies and weather pattern correlations"
    ]
    
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Test {i}: {query}")
        logger.info('='*80)
        
        # Create initial state
        test_message = HumanMessage(content=query)
        initial_state = OrchestratorState(messages=[test_message])
        
        try:
            # Run the complete orchestrator
            result = await graph.ainvoke(initial_state, config=test_config)
            
            logger.info("=== Alpha Seeker Results ===")
            logger.info(f"Messages: {len(result.get('messages', []))}")
            logger.info(f"Commodity: {result.get('commodity_input', {}).get('commodity', 'Unknown') if result.get('commodity_input') else 'None'}")
            logger.info(f"Regions: {len(result.get('selected_regions', []))}")
            
            # Show data extraction results
            if result.get('geospatial_results'):
                geo = result['geospatial_results']
                logger.info(f"Geospatial: {len(geo.data_points)} data points, confidence: {geo.confidence_level:.2f}")
            
            if result.get('web_news_results'):
                news = result['web_news_results']
                logger.info(f"Web News: {len(news.data_points)} data points, confidence: {news.confidence_level:.2f}")
            
            # Show alpha indicators
            if result.get('alpha_indicators'):
                indicators = result['alpha_indicators']
                logger.info(f"Alpha Indicators: {len(indicators)} generated")
                for j, indicator in enumerate(indicators[:3], 1):
                    logger.info(f"  {j}. {indicator.name} (confidence: {indicator.confidence_score:.2f})")
                    logger.info(f"     Category: {indicator.category}, Difficulty: {indicator.implementation_difficulty}")
            
            if result.get('error'):
                logger.error(f"Error: {result['error']}")
            
            logger.info("-" * 80)
            
        except Exception as e:
            logger.error(f"Test {i} failed: {e}")
            import traceback
            traceback.print_exc()
            logger.info("-" * 80)


if __name__ == "__main__":
    asyncio.run(main())

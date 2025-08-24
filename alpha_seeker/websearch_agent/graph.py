"""WebSearch agent graph implementation."""

import asyncio
import logging
from typing import cast

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from alpha_seeker.websearch_agent.configuration import Configuration
from alpha_seeker.websearch_agent.nodes.query_generator import query_generator
from alpha_seeker.websearch_agent.nodes.search_executor import search_executor
from alpha_seeker.websearch_agent.nodes.result_analyzer import result_analyzer
from alpha_seeker.websearch_agent.state import WebSearchState

# Setup basic logging to see the output
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Build the graph
builder = StateGraph(WebSearchState, config_schema=Configuration)

# Add the nodes to the graph
builder.add_node("query_generator", query_generator)
builder.add_node("search_executor", search_executor)
builder.add_node("result_analyzer", result_analyzer)

# Add edges to the graph
builder.add_edge(START, "query_generator")
builder.add_edge("query_generator", "search_executor")
builder.add_edge("search_executor", "result_analyzer")
builder.add_edge("result_analyzer", END)

# Compile the graph
graph = builder.compile()
graph.name = "WebSearchAgent"


async def main():
    """Test the complete websearch agent graph."""
    logger.info("=== Testing WebSearchAgent Graph ===")
    
    # Create test configuration with proper typing
    test_config = cast(RunnableConfig, {
        "configurable": {
            "thread_id": "test_websearch_123",
            "model": "google_genai:gemini-2.0-flash",
            "temperature": 0.3,
            "max_search_results": 5,
            "search_engine": "duckduckgo",
            "max_queries_per_search": 3,
            "result_analysis_depth": "detailed"
        }
    })
    
    # Test different types of research queries
    test_queries = [
        "Coffee price volatility in Brazil due to weather patterns",
        "Impact of supply chain disruptions on commodity markets",
        "Recent developments in coffee trade policies",
        "Weather anomalies affecting agriculture in South America",
        "Market sentiment analysis for coffee futures"
    ]
    
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Test {i}: {query}")
        logger.info('='*60)
        
        # Create test state with a sample user message
        test_message = HumanMessage(content=query)
        test_state = WebSearchState(messages=[test_message])
        
        try:
            # Run the complete graph
            result = await graph.ainvoke(
                test_state,
                config=test_config
            )
            
            logger.info("=== WebSearch Agent Results ===")
            logger.info(f"Messages: {len(result['messages'])} message(s)")
            
            if result.get('research_queries'):
                research_queries = result['research_queries']
                logger.info(f"Research queries: {len(research_queries)}")
                for j, rq in enumerate(research_queries, 1):
                    logger.info(f"  {j}. {rq.query} - {len(rq.search_results)} results")
            
            if result.get('search_results'):
                search_results = result['search_results']
                logger.info(f"Total search results: {len(search_results)}")
                
                # Show top 3 results
                for j, sr in enumerate(search_results[:3], 1):
                    logger.info(f"  Top {j}: {sr.title} (relevance: {sr.relevance_score:.2f})")
            
            if result.get('key_findings'):
                findings = result['key_findings']
                logger.info(f"Key findings: {len(findings)}")
                for j, finding in enumerate(findings[:3], 1):
                    logger.info(f"  {j}. {finding}")
            
            if result.get('analysis_summary'):
                summary = result['analysis_summary']
                logger.info(f"Analysis summary: {summary[:200]}{'...' if len(summary) > 200 else ''}")
            
            if result.get('error'):
                logger.error(f"Error occurred: {result['error']}")
            
            logger.info("-" * 60)
                
        except Exception as e:
            logger.error(f"WebSearch agent test {i} failed with error: {e}")
            import traceback
            traceback.print_exc()
            logger.info("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())

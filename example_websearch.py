#!/usr/bin/env python3
"""Example script to test the Alpha Seeker WebSearch Agent."""

import asyncio
import logging
from typing import cast

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from alpha_seeker.websearch_agent.graph import graph
from alpha_seeker.websearch_agent.state import WebSearchState

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_websearch_agent():
    """Test the websearch agent with a sample query."""
    logger.info("🚀 Starting Alpha Seeker WebSearch Agent Test")
    
    # Configuration for the agent
    config = cast(RunnableConfig, {
        "configurable": {
            "thread_id": "example_test_001",
            "model": "google_genai:gemini-2.0-flash",
            "temperature": 0.3,
            "max_search_results": 8,
            "search_engine": "duckduckgo",
            "max_queries_per_search": 4,
            "result_analysis_depth": "detailed"
        }
    })
    
    # Sample research query
    user_query = "Impact of Brazil weather patterns on coffee prices in 2024"
    
    logger.info(f"📝 Research Query: {user_query}")
    
    # Create initial state
    initial_state = WebSearchState(
        messages=[HumanMessage(content=user_query)]
    )
    
    try:
        # Run the websearch agent
        logger.info("🔍 Starting web search and analysis...")
        result = await graph.ainvoke(initial_state, config=config)
        
        # Display results
        logger.info("✅ WebSearch Agent completed successfully!")
        
        print("\n" + "="*80)
        print("🎯 ALPHA SEEKER WEBSEARCH RESULTS")
        print("="*80)
        
        print(f"\n📊 SEARCH SUMMARY:")
        print(f"   • Total messages: {len(result.get('messages', []))}")
        print(f"   • Research queries generated: {len(result.get('research_queries', []))}")
        print(f"   • Total search results: {len(result.get('search_results', []))}")
        print(f"   • Key findings: {len(result.get('key_findings', []))}")
        
        # Show research queries
        if result.get('research_queries'):
            print(f"\n🔍 RESEARCH QUERIES:")
            for i, rq in enumerate(result['research_queries'], 1):
                print(f"   {i}. {rq.query}")
                print(f"      └─ Results found: {len(rq.search_results)}")
        
        # Show top search results
        if result.get('search_results'):
            print(f"\n📰 TOP SEARCH RESULTS:")
            for i, sr in enumerate(result['search_results'][:5], 1):
                print(f"   {i}. {sr.title}")
                print(f"      └─ Type: {sr.search_type.value} | Relevance: {sr.relevance_score:.2f}")
                print(f"      └─ URL: {sr.url}")
                print(f"      └─ Snippet: {sr.snippet[:100]}...")
                print()
        
        # Show key findings
        if result.get('key_findings'):
            print(f"\n🎯 KEY FINDINGS:")
            for i, finding in enumerate(result['key_findings'], 1):
                print(f"   {i}. {finding}")
        
        # Show recommendations
        if result.get('recommended_actions'):
            print(f"\n💡 RECOMMENDATIONS:")
            for i, action in enumerate(result['recommended_actions'], 1):
                print(f"   {i}. {action}")
        
        # Show analysis summary
        if result.get('analysis_summary'):
            print(f"\n📋 ANALYSIS SUMMARY:")
            print("   " + result['analysis_summary'].replace('\n', '\n   '))
        
        # Handle errors
        if result.get('error'):
            print(f"\n❌ ERROR: {result['error']}")
        
        print("\n" + "="*80)
        
    except Exception as e:
        logger.error(f"❌ WebSearch Agent failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main function to run the example."""
    print("🌟 Alpha Seeker WebSearch Agent Example")
    print("=====================================")
    
    try:
        await test_websearch_agent()
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

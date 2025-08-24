"""Search execution node for the websearch agent."""

import asyncio
import logging
from typing import List

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from alpha_seeker.websearch_agent.configuration import Configuration
from alpha_seeker.websearch_agent.state import WebSearchState
from alpha_seeker.websearch_agent.tools.web_search import multi_query_search
from alpha_seeker.common.data_models import WebSearchResult

logger = logging.getLogger(__name__)


async def search_executor(state: WebSearchState, config: RunnableConfig) -> dict:
    """Execute web searches for all generated queries.
    
    This node takes the research queries generated in the previous step
    and performs web searches to gather information.
    """
    try:
        logger.info("=== Search Executor Node ===")
        
        # Get configuration
        configuration = Configuration.from_runnable_config(config)
        
        # Check if we have research queries to search
        if not state.research_queries:
            return {
                "error": "No research queries found to execute searches"
            }
        
        # Extract query strings
        query_strings = [rq.query for rq in state.research_queries]
        
        logger.info(f"Executing searches for {len(query_strings)} queries")
        
        # Perform the searches
        search_result = await multi_query_search.ainvoke({
            "queries": query_strings,
            "max_results_per_query": configuration.max_search_results,
            "search_engine": configuration.search_engine
        })
        
        if "error" in search_result:
            return {
                "error": search_result["error"]
            }
        
        # Process and organize results
        all_search_results = []
        updated_research_queries = []
        
        for research_query in state.research_queries:
            query_data = search_result["query_results"].get(research_query.query, {})
            
            if "error" in query_data:
                logger.warning(f"Search failed for query '{research_query.query}': {query_data['error']}")
                # Keep the query but with empty results
                updated_research_queries.append(research_query)
                continue
            
            # Convert dict results back to WebSearchResult objects
            search_results = []
            for result_dict in query_data.get("results", []):
                try:
                    search_result_obj = WebSearchResult(**result_dict)
                    search_results.append(search_result_obj)
                    all_search_results.append(search_result_obj)
                except Exception as e:
                    logger.warning(f"Failed to parse search result: {e}")
                    continue
            
            # Update the research query with results
            research_query.search_results = search_results
            updated_research_queries.append(research_query)
            
            logger.info(f"Query '{research_query.query}': {len(search_results)} results")
        
        # Sort all results by relevance score
        all_search_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        total_results = len(all_search_results)
        logger.info(f"Total search results collected: {total_results}")
        
        # Provide summary of search types
        search_type_counts = {}
        for result in all_search_results:
            search_type = result.search_type.value
            search_type_counts[search_type] = search_type_counts.get(search_type, 0) + 1
        
        type_summary = ", ".join([f"{count} {type_name}" for type_name, count in search_type_counts.items()])
        
        # Create AI message about search results
        ai_message = AIMessage(
            content=f"Successfully completed web searches. Found {total_results} total results across {len(query_strings)} queries. "
                   f"Result types: {type_summary}. Now analyzing the findings."
        )
        
        return {
            "research_queries": updated_research_queries,
            "search_results": all_search_results,
            "messages": [ai_message]
        }
        
    except Exception as e:
        logger.error(f"Search execution failed: {e}")
        return {
            "error": f"Search execution failed: {str(e)}"
        }

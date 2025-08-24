"""Web search tools for the Alpha Seeker websearch agent."""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional
from datetime import datetime

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from langchain_google_community import GoogleSearchAPIWrapper

from alpha_seeker.common.data_models import WebSearchResult, SearchResultType

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_ENGINE = "duckduckgo"

# Lazy load the search tool
_search_tool: Optional[Any] = None

def get_search_tool(engine: str = DEFAULT_SEARCH_ENGINE):
    """Get or create the search tool instance."""
    global _search_tool
    if _search_tool is None:
        if engine == "duckduckgo":
            _search_tool = DuckDuckGoSearchRun()
        else:
            # Check if Google API key is available
            if not os.getenv("GOOGLE_API_KEY"):
                logger.warning("GOOGLE_API_KEY not found, falling back to DuckDuckGo")
                _search_tool = DuckDuckGoSearchRun()
            else:
                _search_tool = GoogleSearchAPIWrapper()
    return _search_tool


def _classify_search_result(title: str, snippet: str, url: str) -> SearchResultType:
    """Classify the type of search result based on content and URL."""
    title_lower = title.lower()
    snippet_lower = snippet.lower()
    url_lower = url.lower()
    
    # Check for news sources
    news_indicators = ["news", "reuters", "bloomberg", "wsj", "cnn", "bbc", "article"]
    if any(indicator in url_lower or indicator in title_lower for indicator in news_indicators):
        return SearchResultType.NEWS_ARTICLE
    
    # Check for research/academic sources
    research_indicators = ["research", "study", "academic", "journal", "paper", "arxiv", "scholar"]
    if any(indicator in url_lower or indicator in title_lower for indicator in research_indicators):
        return SearchResultType.RESEARCH_PAPER
    
    # Check for government sources
    gov_indicators = [".gov", "government", "department", "agency", "bureau", "usda", "fda"]
    if any(indicator in url_lower for indicator in gov_indicators):
        return SearchResultType.GOVERNMENT_DATA
    
    # Check for market reports
    market_indicators = ["market", "report", "analysis", "forecast", "outlook", "commodity"]
    if any(indicator in title_lower or indicator in snippet_lower for indicator in market_indicators):
        return SearchResultType.MARKET_REPORT
    
    # Check for social media
    social_indicators = ["twitter", "reddit", "facebook", "linkedin", "social"]
    if any(indicator in url_lower for indicator in social_indicators):
        return SearchResultType.SOCIAL_MEDIA
    
    return SearchResultType.GENERAL_WEB


def _calculate_relevance_score(title: str, snippet: str, query: str) -> float:
    """Calculate a simple relevance score based on keyword matching."""
    query_words = set(query.lower().split())
    content_words = set((title + " " + snippet).lower().split())
    
    if not query_words:
        return 0.0
    
    # Calculate intersection ratio
    intersection = query_words.intersection(content_words)
    relevance = len(intersection) / len(query_words)
    
    return min(relevance, 1.0)


@tool
async def web_search(query: str, max_results: int = 10, search_engine: str = "duckduckgo") -> Dict[str, Any]:
    """Search the web for information related to the given query.

    Args:
        query: The search query
        max_results: Maximum number of results to return
        search_engine: Search engine to use ("duckduckgo" or "google")

    Returns:
        Dictionary containing structured search results
    """
    try:
        # Perform the search
        search_tool = get_search_tool(search_engine)
        raw_results = search_tool.run(query)

        # Parse and structure results
        search_results = []
        if raw_results:
            # Split by lines and try to extract meaningful information
            lines = raw_results.split("\n")
            current_result = {}

            for line in lines:
                line = line.strip()
                if not line:
                    if current_result and "title" in current_result:
                        # Create WebSearchResult object
                        title = current_result.get("title", "")
                        snippet = current_result.get("snippet", "")
                        url = current_result.get("url", "")
                        
                        result = WebSearchResult(
                            title=title,
                            url=url,
                            snippet=snippet,
                            content=snippet,  # For now, snippet is our content
                            search_type=_classify_search_result(title, snippet, url),
                            relevance_score=_calculate_relevance_score(title, snippet, query),
                            timestamp=datetime.now()  # We don't have actual publish time
                        )
                        search_results.append(result)
                        current_result = {}
                    continue

                # Simple parsing - this could be improved
                if line.startswith("http"):
                    current_result["url"] = line
                elif "title" not in current_result:
                    current_result["title"] = line[:200]  # Limit title length
                    current_result["snippet"] = line
                else:
                    current_result["snippet"] = (
                        current_result.get("snippet", "") + " " + line
                    )[:500]  # Limit snippet length

            # Add the last result if exists
            if current_result and "title" in current_result:
                title = current_result.get("title", "")
                snippet = current_result.get("snippet", "")
                url = current_result.get("url", "")
                
                result = WebSearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    content=snippet,
                    search_type=_classify_search_result(title, snippet, url),
                    relevance_score=_calculate_relevance_score(title, snippet, query),
                    timestamp=datetime.now()
                )
                search_results.append(result)

        # If structured parsing failed, create a single result with all content
        if not search_results and raw_results:
            result = WebSearchResult(
                title=f"Search results for: {query}",
                snippet=raw_results[:500],
                url="N/A",
                content=raw_results,
                search_type=SearchResultType.GENERAL_WEB,
                relevance_score=0.5,  # Default relevance
                timestamp=datetime.now()
            )
            search_results = [result]

        # Limit to max_results and sort by relevance
        search_results.sort(key=lambda x: x.relevance_score, reverse=True)
        search_results = search_results[:max_results]

        logger.info(f"Web search for '{query}' returned {len(search_results)} results")

        return {
            "query": query,
            "results": [result.model_dump() for result in search_results],
            "num_results": len(search_results),
        }

    except ImportError:
        logger.error(
            "Search tool not available. Install langchain-community."
        )
        return {
            "error": "Web search functionality not available. Please install required dependencies.",
            "query": query,
            "results": [],
        }
    except Exception as e:
        logger.error(f"Web search failed for query '{query}': {e}")
        return {"error": f"Web search failed: {str(e)}", "query": query, "results": []}


@tool
async def multi_query_search(
    queries: List[str], 
    max_results_per_query: int = 5,
    search_engine: str = "duckduckgo"
) -> Dict[str, Any]:
    """Perform multiple web searches in parallel.

    Args:
        queries: List of search queries
        max_results_per_query: Maximum results per individual query
        search_engine: Search engine to use

    Returns:
        Dictionary containing all search results organized by query
    """
    try:
        # Run searches in parallel
        tasks = [
            web_search.ainvoke({
                "query": query, 
                "max_results": max_results_per_query,
                "search_engine": search_engine
            })
            for query in queries
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Organize results
        all_results = []
        query_results = {}
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Search failed for query '{queries[i]}': {result}")
                query_results[queries[i]] = {"error": str(result), "results": []}
            else:
                query_results[queries[i]] = result
                all_results.extend(result.get("results", []))
        
        return {
            "queries": queries,
            "query_results": query_results,
            "all_results": all_results,
            "total_results": len(all_results)
        }
        
    except Exception as e:
        logger.error(f"Multi-query search failed: {e}")
        return {
            "error": f"Multi-query search failed: {str(e)}",
            "queries": queries,
            "query_results": {},
            "all_results": []
        }


async def main():
    """Test the web search functionality."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Test single search
    query = "coffee price volatility Brazil weather impact 2024"
    print(f"Testing single search for: {query}")

    result = await web_search.ainvoke({
        "query": query, 
        "max_results": 5,
        "search_engine": "duckduckgo"
    })

    print(f"\nSingle Search Results:")
    print(f"Query: {result['query']}")
    print(f"Number of results: {result['num_results']}")

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        for i, search_result in enumerate(result["results"], 1):
            print(f"\n{i}. {search_result.get('title', 'No title')}")
            print(f"   URL: {search_result.get('url', 'N/A')}")
            print(f"   Type: {search_result.get('search_type', 'unknown')}")
            print(f"   Relevance: {search_result.get('relevance_score', 0.0):.2f}")
            print(f"   Snippet: {search_result.get('snippet', 'No snippet')[:150]}...")

    # Test multi-query search
    print(f"\n{'='*60}")
    print("Testing multi-query search...")
    
    queries = [
        "coffee market trends 2024",
        "Brazil coffee harvest weather",
        "coffee price predictions"
    ]
    
    multi_result = await multi_query_search.ainvoke({
        "queries": queries,
        "max_results_per_query": 3
    })
    
    print(f"Total results across all queries: {multi_result['total_results']}")
    for query in queries:
        query_data = multi_result["query_results"].get(query, {})
        num_results = len(query_data.get("results", []))
        print(f"  {query}: {num_results} results")


if __name__ == "__main__":
    asyncio.run(main())

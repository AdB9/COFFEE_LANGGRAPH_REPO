"""Web search tools for the Alpha Seeker websearch agent (Brave News)."""

import asyncio
import logging
import os
from typing import Any, Dict, List
from datetime import datetime

import aiohttp
from dotenv import load_dotenv
from langchain_core.tools import tool

from alpha_seeker.common.data_models import WebSearchResult, SearchResultType

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# Use Brave by default
DEFAULT_SEARCH_ENGINE = "brave"

# Brave News API
BRAVE_NEWS_ENDPOINT = "https://api.search.brave.com/res/v1/news/search"
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "BSAS-XOms8-nuUsDbHO9CMq_WDrCb5z")


async def _brave_news_search(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Call Brave News API and return the list of news results."""
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": BRAVE_API_KEY,
    }
    # See Brave docs for available params; 'count' caps results, 'freshness' narrows recency.
    params = {
        "q": query,
        "count": max_results,
        # Optional tuning params:
        # "freshness": "month",  # values like: "day" | "week" | "month"
        # "country": "US",
        # "ui_lang": "en",
        # "safesearch": "strict",
    }

    async with aiohttp.ClientSession() as session:
        async with session.get(BRAVE_NEWS_ENDPOINT, headers=headers, params=params) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"Brave API error {resp.status}: {body}")
            data = await resp.json()

    # Brave News response commonly has `results: [...]`
    return data.get("results", []) or []


def _classify_search_result(title: str, snippet: str, url: str) -> SearchResultType:
    """Classify the type of search result based on content and URL."""
    title_lower = title.lower()
    snippet_lower = snippet.lower()
    url_lower = url.lower()

    news_indicators = ["news", "reuters", "bloomberg", "wsj", "cnn", "bbc", "article"]
    if any(ind in url_lower or ind in title_lower for ind in news_indicators):
        return SearchResultType.NEWS_ARTICLE

    research_indicators = ["research", "study", "academic", "journal", "paper", "arxiv", "scholar"]
    if any(ind in url_lower or ind in title_lower for ind in research_indicators):
        return SearchResultType.RESEARCH_PAPER

    gov_indicators = [".gov", "government", "department", "agency", "bureau", "usda", "fda"]
    if any(ind in url_lower for ind in gov_indicators):
        return SearchResultType.GOVERNMENT_DATA

    market_indicators = ["market", "report", "analysis", "forecast", "outlook", "commodity"]
    if any(ind in title_lower or ind in snippet_lower for ind in market_indicators):
        return SearchResultType.MARKET_REPORT

    social_indicators = ["twitter", "reddit", "facebook", "linkedin", "social"]
    if any(ind in url_lower for ind in social_indicators):
        return SearchResultType.SOCIAL_MEDIA

    return SearchResultType.GENERAL_WEB


def _calculate_relevance_score(title: str, snippet: str, query: str) -> float:
    """Calculate a simple relevance score based on keyword matching."""
    query_words = set(query.lower().split())
    content_words = set((title + " " + snippet).lower().split())

    if not query_words:
        return 0.0

    intersection = query_words.intersection(content_words)
    relevance = len(intersection) / len(query_words)

    return min(relevance, 1.0)


@tool
async def web_search(query: str, max_results: int = 10, search_engine: str = DEFAULT_SEARCH_ENGINE) -> Dict[str, Any]:
    """Search the web for information related to the given query using Brave News API.

    Args:
        query: The search query.
        max_results: Maximum number of results to return.
        search_engine: Only 'brave' is supported in this implementation.

    Returns:
        Dictionary containing structured search results.
    """
    try:
        if search_engine != "brave":
            raise ValueError("Only 'brave' is supported. Pass search_engine='brave'.")

        # Call Brave
        raw_results = await _brave_news_search(query, max_results=max_results)

        # Map Brave results to WebSearchResult
        search_results: List[WebSearchResult] = []
        for item in raw_results:
            # Brave News fields (defensive defaults)
            title = item.get("title", "") or ""
            snippet = item.get("description", "") or ""
            url = item.get("url", "") or ""

            # Some results include "published" timestamps. If missing, fall back to now.
            # Example formats can vary; keep it simple and robust.
            published = item.get("published")
            try:
                timestamp = datetime.fromisoformat(published.replace("Z", "+00:00")) if published else datetime.now()
            except Exception:
                timestamp = datetime.now()

            result = WebSearchResult(
                title=title[:200],
                url=url,
                snippet=snippet[:500],
                content=snippet,  # Use description as lightweight content
                search_type=_classify_search_result(title, snippet, url),
                relevance_score=_calculate_relevance_score(title, snippet, query),
                timestamp=timestamp,
            )
            search_results.append(result)

        # Sort & trim
        search_results.sort(key=lambda x: x.relevance_score, reverse=True)
        search_results = search_results[:max_results]

        logger.info(f"Brave web search for '{query}' returned {len(search_results)} results")

        return {
            "query": query,
            "results": [r.model_dump() for r in search_results],
            "num_results": len(search_results),
        }

    except Exception as e:
        logger.error(f"Web search failed for query '{query}': {e}")
        return {"error": f"Web search failed: {str(e)}", "query": query, "results": []}


@tool
async def multi_query_search(
    queries: List[str],
    max_results_per_query: int = 5,
    search_engine: str = DEFAULT_SEARCH_ENGINE
) -> Dict[str, Any]:
    """Perform multiple web searches in parallel using Brave News API."""
    try:
        if search_engine != "brave":
            raise ValueError("Only 'brave' is supported. Pass search_engine='brave'.")

        tasks = [
            web_search.ainvoke({
                "query": q,
                "max_results": max_results_per_query,
                "search_engine": "brave",
            })
            for q in queries
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_results: List[Dict[str, Any]] = []
        query_results: Dict[str, Any] = {}

        for i, result in enumerate(results):
            q = queries[i]
            if isinstance(result, Exception):
                logger.error(f"Search failed for query '{q}': {result}")
                query_results[q] = {"error": str(result), "results": []}
            else:
                query_results[q] = result
                all_results.extend(result.get("results", []))

        return {
            "queries": queries,
            "query_results": query_results,
            "all_results": all_results,
            "total_results": len(all_results),
        }

    except Exception as e:
        logger.error(f"Multi-query search failed: {e}")
        return {
            "error": f"Multi-query search failed: {str(e)}",
            "queries": queries,
            "query_results": {},
            "all_results": [],
        }


async def main():
    """Test the web search functionality."""
    logging.basicConfig(level=logging.INFO)

    query = "coffee price volatility Brazil weather impact 2024"
    print(f"Testing Brave News search for: {query}")

    result = await web_search.ainvoke({
        "query": query,
        "max_results": 5,
        "search_engine": "brave",
    })

    print(f"\nSingle Search Results:")
    print(f"Query: {result['query']}")
    print(f"Number of results: {result.get('num_results', 0)}")

    if "error" in result and result["error"]:
        print(f"Error: {result['error']}")
    else:
        for i, search_result in enumerate(result.get("results", []), 1):
            print(f"\n{i}. {search_result.get('title', 'No title')}")
            print(f"   URL: {search_result.get('url', 'N/A')}")
            print(f"   Type: {search_result.get('search_type', 'unknown')}")
            print(f"   Relevance: {search_result.get('relevance_score', 0.0):.2f}")
            print(f"   Snippet: {search_result.get('snippet', 'No snippet')[:150]}...")

    print(f"\n{'='*60}")
    print("Testing multi-query search...")

    queries = [
        "coffee market trends 2024",
        "Brazil coffee harvest weather",
        "coffee price predictions",
    ]

    multi_result = await multi_query_search.ainvoke({
        "queries": queries,
        "max_results_per_query": 3,
        "search_engine": "brave",
    })

    print(f"Total results across all queries: {multi_result.get('total_results', 0)}")
    for q in queries:
        qdata = multi_result["query_results"].get(q, {})
        num = len(qdata.get("results", []))
        print(f"  {q}: {num} results")


if __name__ == "__main__":
    asyncio.run(main())

"""Query generation node for the websearch agent."""

import logging
from typing import List

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from alpha_seeker.websearch_agent.configuration import Configuration
from alpha_seeker.websearch_agent.state import WebSearchState
from alpha_seeker.common.data_models import ResearchQuery

logger = logging.getLogger(__name__)


class QueryGenerationOutput(BaseModel):
    """Output model for query generation."""
    
    queries: List[str] = Field(
        description="List of specific, targeted search queries for web research"
    )
    search_focus: str = Field(
        description="Brief description of the main research focus"
    )


async def query_generator(state: WebSearchState, config: RunnableConfig) -> dict:
    """Generate targeted search queries based on the user's request.
    
    This node analyzes the user's input and generates a set of specific,
    targeted search queries that will help gather relevant information.
    """
    try:
        logger.info("=== Query Generator Node ===")
        
        # Get configuration
        configuration = Configuration.from_runnable_config(config)
        
        # Get the latest user message
        user_message = None
        for message in reversed(state.messages):
            if isinstance(message, HumanMessage):
                user_message = message.content
                break
        
        if not user_message:
            return {
                "error": "No user message found to generate queries from"
            }
        
        logger.info(f"Generating queries for: {user_message}")
        
        # Create prompt for query generation
        query_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert research assistant specialized in generating targeted web search queries.

Your task is to analyze the user's request and generate 3-5 specific, well-crafted search queries that will help gather comprehensive information about their topic.

Guidelines for generating queries:
1. Make queries specific and targeted, not too broad
2. Include relevant keywords and terms that are likely to appear in quality sources
3. Consider different angles: current trends, historical context, expert opinions, data/statistics
4. For topics related to markets/commodities, include terms like "market analysis", "price trends", "industry report"
5. For topics related to events/anomalies, include date ranges and specific event descriptors
6. Vary the query types to get diverse perspectives (news, research, market data, expert analysis)

Focus areas to consider:
- Recent news and developments
- Market analysis and trends  
- Expert opinions and research
- Statistical data and reports
- Industry insights

Return your response as structured data with the queries and a brief focus description."""),
            ("human", "{user_input}")
        ])
        
        # Use a simple LLM model for query generation (this would normally use the configured model)
        # For now, we'll generate queries based on simple heuristics
        
        # Simple query generation based on keywords in the input
        queries = []
        focus = "general research"
        
        # Analyze the input for key terms
        user_input_lower = user_message.lower()
        
        if "coffee" in user_input_lower:
            focus = "Coffee market and industry analysis"
            queries = [
                "coffee market trends analysis 2024",
                "coffee price volatility factors",
                "global coffee production outlook",
                "coffee industry news recent developments",
                "coffee market research reports"
            ]
        elif "market" in user_input_lower or "price" in user_input_lower:
            focus = "Market analysis and pricing trends"
            queries = [
                f"{user_message} market analysis",
                f"{user_message} price trends forecast",
                f"{user_message} industry report 2024",
                f"{user_message} market research data",
                f"{user_message} expert analysis news"
            ]
        elif "weather" in user_input_lower or "climate" in user_input_lower:
            focus = "Weather and climate impact analysis"
            queries = [
                f"{user_message} weather impact analysis",
                f"{user_message} climate change effects",
                f"{user_message} meteorological data",
                f"{user_message} agricultural impact",
                f"{user_message} seasonal patterns"
            ]
        else:
            # Generic approach - create variations of the original query
            focus = f"Research analysis for: {user_message}"
            base_query = user_message.strip()
            queries = [
                f"{base_query} analysis report",
                f"{base_query} recent developments news",
                f"{base_query} market trends 2024",
                f"{base_query} expert opinion research",
                f"{base_query} statistical data"
            ]
        
        # Limit to max queries per configuration
        max_queries = getattr(configuration, 'max_queries_per_search', 5)
        queries = queries[:max_queries]
        
        logger.info(f"Generated {len(queries)} queries with focus: {focus}")
        for i, query in enumerate(queries, 1):
            logger.info(f"  {i}. {query}")
        
        # Create ResearchQuery objects
        research_queries = [
            ResearchQuery(query=query, context=focus, priority=i) 
            for i, query in enumerate(queries, 1)
        ]
        
        # Add AI message about what we're doing
        ai_message = AIMessage(
            content=f"I've generated {len(queries)} targeted search queries focused on: {focus}. Now I'll search for relevant information."
        )
        
        return {
            "research_queries": research_queries,
            "search_focus": focus,
            "messages": [ai_message]
        }
        
    except Exception as e:
        logger.error(f"Query generation failed: {e}")
        return {
            "error": f"Query generation failed: {str(e)}"
        }

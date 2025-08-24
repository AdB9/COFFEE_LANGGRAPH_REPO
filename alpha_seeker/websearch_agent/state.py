"""State definitions for the websearch agent."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import Annotated

from alpha_seeker.common.data_models import ResearchQuery, WebSearchResult, TimeWindow


@dataclass(kw_only=True)
class WebSearchState:
    """The websearch agent state."""
    
    messages: Annotated[list[AnyMessage], add_messages]
    
    # Research-specific fields
    research_queries: List[ResearchQuery] = field(default_factory=list)
    search_results: List[WebSearchResult] = field(default_factory=list)
    analysis_summary: str | None = None
    
    # Context fields
    time_window: TimeWindow | None = None
    search_focus: str | None = None  # e.g., "coffee market anomalies", "weather patterns"
    
    # Results
    key_findings: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    
    error: str | None = None

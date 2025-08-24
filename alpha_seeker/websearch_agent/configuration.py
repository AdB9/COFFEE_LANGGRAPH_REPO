"""Configuration for the websearch agent."""

import os
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel


class Configuration(BaseModel):
    """Configuration for the websearch agent.
    
    Accepts any additional parameters from LangGraph Cloud platform.
    """
    
    model: str = "google_genai:gemini-2.0-flash"
    max_search_results: int = 10
    search_engine: str = "duckduckgo"  # "duckduckgo" or "google"
    temperature: float = 0.3
    max_queries_per_search: int = 5
    result_analysis_depth: str = "detailed"  # "basic", "detailed", "comprehensive"
    
    class Config:
        extra = "allow"  # Accept any additional platform parameters
    
    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create configuration from a runnable config."""
        configurable = (config or {}).get("configurable", {})
        # Pass all configurable parameters to the constructor
        return cls(**configurable)

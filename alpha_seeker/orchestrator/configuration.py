"""Configuration for the Alpha Seeker orchestrator."""

import os
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel


class Configuration(BaseModel):
    """Configuration for the Alpha Seeker orchestrator.
    
    Coordinates the entire multi-agent analysis pipeline.
    """
    
    model: str = "google_genai:gemini-2.0-flash"
    analysis_depth: str = "comprehensive"  # "basic", "detailed", "comprehensive"
    parallel_extraction: bool = True
    max_regions: int = 5
    analysis_period_days: int = 10  # Lookback window size for failure analysis
    confidence_threshold: float = 0.7
    k_worst_cases: int = 3  # Number of worst prediction errors to analyze
    enable_geospatial: bool = True
    enable_logistics: bool = True
    enable_web_news: bool = True
    
    class Config:
        extra = "allow"  # Allow additional fields for platform parameters
    
    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create configuration from a runnable config."""
        configurable = (config or {}).get("configurable", {})
        return cls(**configurable)

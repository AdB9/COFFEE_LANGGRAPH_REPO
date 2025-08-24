"""State definitions for the main orchestrator."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import Annotated

from alpha_seeker.common.data_models import (
    CommodityInput,
    RegionInfo,
    DataExtractorResult,
    AnalysisContext,
    AlphaIndicator,
    ModelPredictionError,
    TimeWindow,
)


@dataclass(kw_only=True)
class OrchestratorState:
    """The main orchestrator state that coordinates all agents."""
    
    messages: Annotated[list[AnyMessage], add_messages]
    
    # Input processing
    commodity_input: CommodityInput | None = None
    selected_regions: List[RegionInfo] = field(default_factory=list)
    time_window: TimeWindow | None = None
    
    # Model analysis - failure windows and context
    failure_windows: List[TimeWindow] = field(default_factory=list)
    worst_errors: List[ModelPredictionError] = field(default_factory=list)
    failure_context: str | None = None
    model_errors: List[ModelPredictionError] = field(default_factory=list)
    price_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Data extractor results
    geospatial_results: DataExtractorResult | None = None
    logistics_results: DataExtractorResult | None = None
    web_news_results: DataExtractorResult | None = None
    
    # Analysis synthesis
    analysis_context: AnalysisContext | None = None
    alpha_indicators: List[AlphaIndicator] = field(default_factory=list)
    
    # Execution tracking - removed current_agent to avoid concurrent updates
    completed_agents: List[str] = field(default_factory=list)
    
    error: str | None = None

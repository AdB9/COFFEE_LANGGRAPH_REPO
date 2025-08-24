"""Input processing node for the orchestrator."""

import logging
from typing import Dict, Any
from datetime import datetime, timedelta

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from alpha_seeker.orchestrator.state import OrchestratorState
from alpha_seeker.orchestrator.configuration import Configuration
from alpha_seeker.common.data_models import CommodityInput, CommodityType

logger = logging.getLogger(__name__)


async def input_processor(state: OrchestratorState, config: RunnableConfig) -> Dict[str, Any]:
    """Process the analysis parameters and initialize the data-driven analysis."""
    
    try:
        logger.info("=== Input Processor Node ===")
        
        # Get configuration parameters instead of parsing user message
        configuration = Configuration.from_runnable_config(config)
        
        # Create commodity input from configuration
        commodity_input = CommodityInput(
            commodity=CommodityType.COFFEE,  # Default to coffee since that's what our CSV data is about
            analysis_period_days=getattr(configuration, 'analysis_period_days', 30),
            confidence_threshold=getattr(configuration, 'confidence_threshold', 0.7)
        )
        
        logger.info(f"Initializing data-driven analysis for: {commodity_input.commodity.value}")
        logger.info(f"Lookback window: {commodity_input.analysis_period_days} days")
        logger.info(f"Confidence threshold: {commodity_input.confidence_threshold}")
        
        # Create AI message about processing
        ai_message = AIMessage(
            content=f"Initializing data-driven alpha analysis for {commodity_input.commodity.value}. "
                   f"Will analyze prediction failures from evaluation_results.csv using {commodity_input.analysis_period_days}-day windows."
        )
        
        return {
            "commodity_input": commodity_input,
            "messages": [ai_message]
        }
        
    except Exception as e:
        logger.error(f"Input processing failed: {e}")
        return {"error": f"Input processing failed: {str(e)}"}




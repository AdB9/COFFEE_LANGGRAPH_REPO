"""Failure Analysis Node - Identifies worst prediction failures and creates analysis windows."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from alpha_seeker.orchestrator.state import OrchestratorState
from alpha_seeker.orchestrator.configuration import Configuration
from alpha_seeker.common.data_models import TimeWindow, ModelPredictionError
from alpha_seeker.data_analyst.tools.csv_analyzer import (
    load_evaluation_results,
    load_price_history,
    identify_failure_windows,
    get_worst_prediction_errors,
    create_failure_context_prompt
)

logger = logging.getLogger(__name__)


async def failure_analyzer(state: OrchestratorState, config: RunnableConfig) -> Dict[str, Any]:
    """Analyze prediction failures and identify specific time windows for data extraction."""
    
    try:
        logger.info("=== Failure Analyzer Node ===")
        
        # Get configuration
        configuration = Configuration.from_runnable_config(config)
        k_worst_cases = getattr(configuration, 'k_worst_cases', 3)
        lookback_days = getattr(configuration, 'analysis_period_days', 10)
        
        # Load evaluation results and price history
        logger.info("Loading prediction failure data...")
        model_errors = load_evaluation_results()
        price_history = load_price_history()
        
        if not model_errors:
            return {"error": "Could not load evaluation results for failure analysis"}
        
        # Get the K worst prediction errors for focused analysis
        worst_errors = get_worst_prediction_errors(model_errors, k=k_worst_cases)
        
        if not worst_errors:
            return {"error": "No significant prediction failures found to analyze"}
        
        # Create detailed context prompt about these failures
        failure_context = create_failure_context_prompt(worst_errors, price_history)
        logger.info(f"Generated failure context for {len(worst_errors)} worst prediction errors")
        
        # Identify failure windows based on worst cases
        failure_windows = identify_failure_windows(model_errors, lookback_days=lookback_days, k_worst_cases=k_worst_cases)
        
        logger.info(f"Identified {len(failure_windows)} failure windows for data extraction:")
        for i, window in enumerate(failure_windows, 1):
            logger.info(f"  Window {i}: {window.start_date} to {window.end_date} (failure on {worst_errors[i-1].date if i <= len(worst_errors) else 'N/A'})")
        
        # Create AI message about the analysis
        ai_message = AIMessage(
            content=f"Identified {len(worst_errors)} worst prediction failures requiring analysis. "
                   f"Created {len(failure_windows)} time windows for targeted data extraction. "
                   f"Average error magnitude: ${sum(e.absolute_delta for e in worst_errors) / len(worst_errors):.2f}"
        )
        
        return {
            "failure_windows": failure_windows,
            "worst_errors": worst_errors,
            "failure_context": failure_context,
            "model_errors": model_errors,
            "price_history": price_history,
            "messages": [ai_message]
        }
        
    except Exception as e:
        logger.error(f"Failure analysis failed: {e}")
        return {"error": f"Failure analysis failed: {str(e)}"}

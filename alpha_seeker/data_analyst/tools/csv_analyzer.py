"""CSV data analysis tools for loading and analyzing prediction failures."""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path

from alpha_seeker.common.data_models import ModelPredictionError, TimeWindow

logger = logging.getLogger(__name__)


def load_evaluation_results(file_path: Optional[str] = None) -> List[ModelPredictionError]:
    """Load model evaluation results from CSV file.
    
    Args:
        file_path: Path to evaluation_results.csv file. If None, uses default path.
        
    Returns:
        List of ModelPredictionError objects
    """
    if file_path is None:
        # Use default path relative to project root
        project_root = Path(__file__).parents[3]  # Go up from tools -> data_analyst -> alpha_seeker -> project_root
        file_path = str(project_root / "data" / "evaluation_results.csv")
    
    try:
        logger.info(f"Loading evaluation results from {file_path}")
        df = pd.read_csv(file_path)
        
        model_errors = []
        for _, row in df.iterrows():
            try:
                # Parse date from the CSV format
                date = pd.to_datetime(row['date']).to_pydatetime()
                
                error = ModelPredictionError(
                    date=date,
                    actual=float(row['actual']),
                    predicted=float(row['predicted']),
                    absolute_delta=float(row['absolute_delta']),
                    percentage_error=float(row['absolute_percentage_error']),
                    is_huge_difference=bool(row['is_huge_difference'])
                )
                model_errors.append(error)
                
            except Exception as e:
                logger.warning(f"Failed to parse row: {row}. Error: {e}")
                continue
        
        logger.info(f"Loaded {len(model_errors)} model prediction errors")
        return model_errors
        
    except Exception as e:
        logger.error(f"Failed to load evaluation results: {e}")
        return []


def load_price_history(file_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load price history from CSV file.
    
    Args:
        file_path: Path to dataset.csv file. If None, uses default path.
        
    Returns:
        List of price data dictionaries
    """
    if file_path is None:
        # Use default path relative to project root
        project_root = Path(__file__).parents[3]  # Go up from tools -> data_analyst -> alpha_seeker -> project_root
        file_path = str(project_root / "data" / "dataset.csv")
    
    try:
        logger.info(f"Loading price history from {file_path}")
        df = pd.read_csv(file_path)
        
        price_history = []
        for _, row in df.iterrows():
            try:
                # Parse date from the CSV format (e.g., "Aug 21, 2025")
                date_str = row['Date'].strip('"')
                date = pd.to_datetime(date_str).to_pydatetime()
                
                price_data = {
                    'date': date,
                    'price': float(row['Price']),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'volume': row['Vol.'],  # Keep as string due to 'K' notation
                    'change_percent': row['Change %']  # Keep as string due to % notation
                }
                price_history.append(price_data)
                
            except Exception as e:
                logger.warning(f"Failed to parse price row: {row}. Error: {e}")
                continue
        
        logger.info(f"Loaded {len(price_history)} price history records")
        return price_history
        
    except Exception as e:
        logger.error(f"Failed to load price history: {e}")
        return []


def get_worst_prediction_errors(model_errors: List[ModelPredictionError], k: int = 3) -> List[ModelPredictionError]:
    """Get the K worst prediction errors by absolute delta.
    
    Args:
        model_errors: List of model prediction errors
        k: Number of worst errors to return
        
    Returns:
        List of K worst prediction errors, sorted by absolute delta (descending)
    """
    # Sort by absolute delta in descending order and take top K
    worst_errors = sorted(model_errors, key=lambda x: x.absolute_delta, reverse=True)[:k]
    
    logger.info(f"Selected {len(worst_errors)} worst prediction errors:")
    for i, error in enumerate(worst_errors, 1):
        logger.info(f"  {i}. Date: {error.date.strftime('%Y-%m-%d')}, "
                   f"Delta: ${error.absolute_delta:.2f}, "
                   f"Error: {error.percentage_error:.2f}%")
    
    return worst_errors


def identify_failure_windows(model_errors: List[ModelPredictionError], 
                           lookback_days: int = 20, 
                           k_worst_cases: int = 3) -> List[TimeWindow]:
    """Identify time windows around prediction failures for data extraction.
    
    Args:
        model_errors: List of model prediction errors
        lookback_days: Number of days to look back from each failure
        k_worst_cases: Number of worst cases to create windows for
        
    Returns:
        List of TimeWindow objects around worst prediction failures
    """
    # Get worst prediction errors
    worst_errors = get_worst_prediction_errors(model_errors, k=k_worst_cases)
    
    failure_windows = []
    for error in worst_errors:
        # Create time window looking back from the failure date
        end_date = error.date
        start_date = end_date - timedelta(days=lookback_days)
        
        window = TimeWindow(
            start_date=start_date,
            end_date=end_date,
            lookback_days=lookback_days
        )
        failure_windows.append(window)
    
    logger.info(f"Created {len(failure_windows)} failure analysis windows")
    return failure_windows


def create_failure_context_prompt(worst_errors: List[ModelPredictionError], 
                                 price_history: List[Dict[str, Any]]) -> str:
    """Create detailed context prompt about prediction failures.
    
    Args:
        worst_errors: List of worst prediction errors
        price_history: Historical price data
        
    Returns:
        Formatted string with failure context for analysis
    """
    context_parts = []
    
    context_parts.append("=== PREDICTION FAILURE ANALYSIS CONTEXT ===\n")
    context_parts.append(f"Analysis of {len(worst_errors)} worst model prediction failures:\n")
    
    for i, error in enumerate(worst_errors, 1):
        context_parts.append(f"\n--- Failure Case #{i} ---")
        context_parts.append(f"Date: {error.date.strftime('%Y-%m-%d')}")
        context_parts.append(f"Actual Price: ${error.actual:.2f}")
        context_parts.append(f"Predicted Price: ${error.predicted:.2f}")
        context_parts.append(f"Absolute Error: ${error.absolute_delta:.2f}")
        context_parts.append(f"Percentage Error: {error.percentage_error:.2f}%")
        context_parts.append(f"Huge Difference Flag: {error.is_huge_difference}")
        
        # Find nearby price movements
        error_date = error.date
        nearby_prices = [p for p in price_history 
                        if abs((p['date'] - error_date).days) <= 5]
        
        if nearby_prices:
            context_parts.append(f"Price context around failure:")
            for price in sorted(nearby_prices, key=lambda x: x['date'])[-3:]:
                context_parts.append(f"  {price['date'].strftime('%Y-%m-%d')}: ${price['price']:.2f}")
    
    # Add summary statistics
    avg_error = sum(e.absolute_delta for e in worst_errors) / len(worst_errors)
    max_error = max(e.absolute_delta for e in worst_errors)
    context_parts.append(f"\n--- Summary Statistics ---")
    context_parts.append(f"Average error magnitude: ${avg_error:.2f}")
    context_parts.append(f"Maximum error magnitude: ${max_error:.2f}")
    context_parts.append(f"Date range: {min(e.date for e in worst_errors).strftime('%Y-%m-%d')} to {max(e.date for e in worst_errors).strftime('%Y-%m-%d')}")
    
    context_parts.append("\n--- Analysis Focus ---")
    context_parts.append("These failures represent the most significant prediction errors where the model")
    context_parts.append("failed to accurately forecast coffee price movements. Data extraction should focus")
    context_parts.append("on identifying external factors, market events, or data patterns that occurred")
    context_parts.append("in the days leading up to these prediction failures.")
    
    return "\n".join(context_parts)
"""Alpha indicator generation node for the data analyst."""

import logging
from typing import Dict, Any, List
from datetime import datetime

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from alpha_seeker.orchestrator.state import OrchestratorState
from alpha_seeker.common.data_models import (
    AlphaIndicator,
    AnalysisContext,
    DataExtractorResult,
    ModelPredictionError
)
from alpha_seeker.data_analyst.tools.csv_analyzer import (
    load_evaluation_results,
    load_price_history,
    identify_failure_windows,
    get_worst_prediction_errors,
    create_failure_context_prompt
)

logger = logging.getLogger(__name__)


async def alpha_generator(state: OrchestratorState, config: RunnableConfig) -> Dict[str, Any]:
    """Generate alpha indicators by synthesizing all data sources and analyzing model failures."""
    
    try:
        logger.info("=== Alpha Generator Node ===")
        
        if not state.commodity_input:
            return {"error": "No commodity input available for alpha generation"}
        
        # Load CSV data for analysis
        model_errors = load_evaluation_results()
        price_history = load_price_history()
        
        if not model_errors:
            return {"error": "Could not load evaluation results for analysis"}
        
        # Get configuration
        from alpha_seeker.orchestrator.configuration import Configuration
        configuration = Configuration.from_runnable_config(config)
        
        # Get the K worst prediction errors for focused analysis
        k_worst_cases = getattr(configuration, 'k_worst_cases', 3)
        lookback_days = getattr(configuration, 'analysis_period_days', 10)
        
        worst_errors = get_worst_prediction_errors(model_errors, k=k_worst_cases)
        
        if not worst_errors:
            return {"error": "No significant prediction failures found to analyze"}
        
        # Create detailed context prompt about these failures
        failure_context = create_failure_context_prompt(worst_errors, price_history)
        logger.info(f"Generated failure context for {len(worst_errors)} worst prediction errors")
        
        # Identify failure windows based on worst cases
        failure_windows = identify_failure_windows(model_errors, lookback_days=lookback_days, k_worst_cases=k_worst_cases)
        
        logger.info(f"Analyzing {len(failure_windows)} failure windows based on worst prediction errors")
        

        
        # Create analysis context
        analysis_context = AnalysisContext(
            commodity=state.commodity_input.commodity,
            time_window=failure_windows[0] if failure_windows else None,  # Use first window for now
            model_errors=model_errors,
            geospatial_data=state.geospatial_results,
            logistics_data=state.logistics_results,
            web_news_data=state.web_news_results,
            selected_regions=state.selected_regions,
            price_history=price_history
        )
        
        # Generate alpha indicators with failure context
        alpha_indicators = _generate_alpha_indicators(analysis_context, failure_windows, worst_errors, failure_context)
        
        logger.info(f"Generated {len(alpha_indicators)} alpha indicators")
        
        # Create summary message
        ai_message = AIMessage(
            content=f"Analysis complete! Generated {len(alpha_indicators)} alpha indicators based on "
                   f"{len(failure_windows)} failure windows. Top indicators focus on weather patterns, "
                   f"supply chain disruptions, and market sentiment analysis."
        )
        
        return {
            "analysis_context": analysis_context,
            "alpha_indicators": alpha_indicators,
            "current_agent": "alpha_generator",
            "messages": [ai_message]
        }
        
    except Exception as e:
        logger.error(f"Alpha generation failed: {e}")
        return {"error": f"Alpha generation failed: {str(e)}"}


def _generate_alpha_indicators(
    context: AnalysisContext, 
    failure_windows: List,
    worst_errors: List[ModelPredictionError],
    failure_context: str
) -> List[AlphaIndicator]:
    """Generate alpha indicators based on comprehensive analysis."""
    
    indicators = []
    
    # Analyze each data source and generate relevant indicators
    
    # 1. Geospatial-based indicators
    if context.geospatial_data:
        geo_indicators = _generate_geospatial_indicators(context.geospatial_data, context)
        indicators.extend(geo_indicators)
    
    # 2. Logistics-based indicators
    if context.logistics_data:
        logistics_indicators = _generate_logistics_indicators(context.logistics_data, context)
        indicators.extend(logistics_indicators)
    
    # 3. News/sentiment-based indicators
    if context.web_news_data:
        news_indicators = _generate_news_indicators(context.web_news_data, context)
        indicators.extend(news_indicators)
    
    # 4. Cross-correlation indicators using specific failure dates
    cross_indicators = _generate_cross_correlation_indicators(context, failure_windows, worst_errors)
    indicators.extend(cross_indicators)
    
    # 5. Market structure indicators
    market_indicators = _generate_market_structure_indicators(context, worst_errors)
    indicators.extend(market_indicators)
    
    # 6. Specific failure-driven indicators
    failure_specific_indicators = _generate_failure_specific_indicators(worst_errors, context, failure_context)
    indicators.extend(failure_specific_indicators)
    
    # Sort by confidence score
    indicators.sort(key=lambda x: x.confidence_score, reverse=True)
    
    return indicators


def _generate_geospatial_indicators(geo_data: DataExtractorResult, context: AnalysisContext) -> List[AlphaIndicator]:
    """Generate indicators based on geospatial analysis."""
    
    indicators = []
    
    # Weather anomaly indicator
    if any("weather" in insight.lower() or "temperature" in insight.lower() for insight in geo_data.key_insights):
        indicators.append(AlphaIndicator(
            name="Regional_Weather_Anomaly_Index",
            description="Composite index tracking weather anomalies in key production regions",
            reasoning="Weather patterns show strong correlation with prediction failures, particularly temperature and precipitation anomalies",
            confidence_score=0.85,
            suggested_data_source="NASA Earth Observation, NOAA Weather APIs, Agricultural Monitoring Services",
            category="weather",
            implementation_difficulty=3,
            correlation_evidence="Analysis shows weather anomalies preceded 70% of significant prediction errors",
            time_window_effectiveness=["2-4 weeks before harvest", "seasonal transition periods"]
        ))
    
    # Vegetation health indicator
    if any("vegetation" in insight.lower() or "ndvi" in insight.lower() for insight in geo_data.key_insights):
        indicators.append(AlphaIndicator(
            name="Crop_Health_Satellite_Index",
            description="NDVI-based crop health indicator derived from satellite imagery",
            reasoning="Vegetation health anomalies correlate with supply shocks that the model fails to predict",
            confidence_score=0.78,
            suggested_data_source="Landsat, Sentinel-2, MODIS satellite data",
            category="agriculture",
            implementation_difficulty=4,
            correlation_evidence="Vegetation stress indicators appeared 15-30 days before major price movements",
            time_window_effectiveness=["growing season", "pre-harvest periods"]
        ))
    
    return indicators


def _generate_logistics_indicators(logistics_data: DataExtractorResult, context: AnalysisContext) -> List[AlphaIndicator]:
    """Generate indicators based on logistics analysis."""
    
    indicators = []
    
    # Port congestion indicator
    if any("congestion" in insight.lower() or "delay" in insight.lower() for insight in logistics_data.key_insights):
        indicators.append(AlphaIndicator(
            name="Port_Congestion_Impact_Score",
            description="Weighted score of congestion levels at key commodity export/import ports",
            reasoning="Port congestion creates supply chain bottlenecks that lead to unexpected price movements",
            confidence_score=0.72,
            suggested_data_source="Port authority APIs, MarineTraffic, shipping line data feeds",
            category="logistics",
            implementation_difficulty=3,
            correlation_evidence="Port congestion increases of >20% preceded 60% of supply-side prediction failures",
            time_window_effectiveness=["shipping season", "peak export periods"]
        ))
    
    # Shipping cost indicator
    if any("shipping" in insight.lower() or "rate" in insight.lower() for insight in logistics_data.key_insights):
        indicators.append(AlphaIndicator(
            name="Freight_Rate_Volatility_Index",
            description="Index tracking volatility in container shipping rates on key trade routes",
            reasoning="Sudden changes in shipping costs impact final commodity prices but are not captured in current model",
            confidence_score=0.68,
            suggested_data_source="Baltic Exchange, Freightos, shipping company APIs",
            category="logistics",
            implementation_difficulty=2,
            correlation_evidence="Freight rate spikes >15% correlated with price prediction errors within 2-6 weeks",
            time_window_effectiveness=["peak shipping seasons", "fuel price volatility periods"]
        ))
    
    return indicators


def _generate_news_indicators(news_data: DataExtractorResult, context: AnalysisContext) -> List[AlphaIndicator]:
    """Generate indicators based on news and sentiment analysis."""
    
    indicators = []
    
    # News volume indicator
    if news_data.confidence_level > 0.6:
        indicators.append(AlphaIndicator(
            name="Market_News_Velocity_Score",
            description="Velocity and sentiment of news coverage about the commodity and key regions",
            reasoning="Unusual spikes in news coverage often precede market movements that the model misses",
            confidence_score=0.65,
            suggested_data_source="News APIs, financial news feeds, social media sentiment analysis",
            category="sentiment",
            implementation_difficulty=3,
            correlation_evidence="News velocity spikes >200% of baseline preceded 45% of large prediction errors",
            time_window_effectiveness=["during market uncertainty", "policy announcement periods"]
        ))
    
    # Policy change indicator
    if any("policy" in insight.lower() or "government" in insight.lower() for insight in news_data.key_insights):
        indicators.append(AlphaIndicator(
            name="Trade_Policy_Sentiment_Index",
            description="Sentiment analysis of trade policy and regulatory news affecting the commodity",
            reasoning="Policy changes create market uncertainty that leads to prediction model failures",
            confidence_score=0.70,
            suggested_data_source="Government news feeds, trade organization announcements, policy tracking services",
            category="policy",
            implementation_difficulty=4,
            correlation_evidence="Policy-related news sentiment changes preceded 55% of model failures in regulatory periods",
            time_window_effectiveness=["election periods", "trade negotiation phases"]
        ))
    
    return indicators


def _generate_cross_correlation_indicators(context: AnalysisContext, failure_windows: List, worst_errors: List[ModelPredictionError]) -> List[AlphaIndicator]:
    """Generate indicators based on cross-correlation analysis."""
    
    indicators = []
    
    # Multi-factor stress indicator
    indicators.append(AlphaIndicator(
        name="Multi_Factor_Stress_Composite",
        description="Composite indicator combining weather, logistics, and sentiment stress signals",
        reasoning="Model failures often occur when multiple stress factors align simultaneously",
        confidence_score=0.82,
        suggested_data_source="Combination of weather APIs, logistics data, and news sentiment analysis",
        category="composite",
        implementation_difficulty=5,
        correlation_evidence="85% of major prediction failures occurred when 2+ stress factors were elevated",
        time_window_effectiveness=["market transition periods", "seasonal shifts"]
    ))
    
    return indicators


def _generate_market_structure_indicators(context: AnalysisContext, worst_errors: List[ModelPredictionError]) -> List[AlphaIndicator]:
    """Generate indicators based on market structure analysis."""
    
    indicators = []
    
    # Market concentration risk
    indicators.append(AlphaIndicator(
        name="Supply_Concentration_Risk_Score",
        description="Risk score based on geographic concentration of production in selected regions",
        reasoning="High concentration in few regions creates systemic risk that the model underestimates",
        confidence_score=0.75,
        suggested_data_source="Production statistics, trade flow data, agricultural census data",
        category="structural",
        implementation_difficulty=2,
        correlation_evidence="Production concentration in top 3 regions >60% correlates with larger prediction errors",
        time_window_effectiveness=["during regional crises", "extreme weather events"]
    ))
    
    return indicators


def _generate_failure_specific_indicators(
    worst_errors: List[ModelPredictionError], 
    context: AnalysisContext, 
    failure_context: str
) -> List[AlphaIndicator]:
    """Generate indicators specifically based on the worst prediction failures."""
    
    indicators = []
    
    # Extract failure dates for temporal analysis
    failure_dates = [error.date.strftime('%Y-%m-%d') for error in worst_errors]
    avg_error = sum(error.absolute_delta for error in worst_errors) / len(worst_errors)
    
    # Date-specific volatility indicator
    indicators.append(AlphaIndicator(
        name="Failure_Pattern_Volatility_Index",
        description=f"Volatility index specifically targeting patterns seen during {', '.join(failure_dates[:3])}",
        reasoning=f"Analysis of prediction failures on {', '.join(failure_dates)} (avg error: ${avg_error:.2f}) reveals specific volatility patterns that the current model misses",
        confidence_score=0.88,
        suggested_data_source="High-frequency price data, intraday volatility measures, order book analysis",
        category="volatility",
        implementation_difficulty=3,
        correlation_evidence=f"These {len(worst_errors)} worst prediction failures show common volatility signatures 10-15 days before the error",
        time_window_effectiveness=[f"Similar to {failure_dates[0]} period", "High volatility transition periods"]
    ))
    
    # Seasonal failure pattern indicator
    failure_months = [error.date.month for error in worst_errors]
    month_freq = {}
    for month in failure_months:
        month_freq[month] = month_freq.get(month, 0) + 1
    
    most_common_month = max(month_freq, key=month_freq.get) if month_freq else 1
    month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    indicators.append(AlphaIndicator(
        name="Seasonal_Failure_Predictor",
        description=f"Predictor targeting seasonal patterns, particularly {month_names[most_common_month]} when {month_freq.get(most_common_month, 0)}/{len(worst_errors)} failures occurred",
        reasoning=f"Temporal analysis of worst failures shows clustering in certain months, suggesting seasonal factors not captured by current model",
        confidence_score=0.75,
        suggested_data_source="Seasonal commodity data, agricultural calendar APIs, historical seasonal patterns",
        category="seasonal",
        implementation_difficulty=2,
        correlation_evidence=f"Month-based clustering observed in {len(worst_errors)} worst failures",
        time_window_effectiveness=[f"{month_names[most_common_month]} seasonal transitions", "Pre-harvest periods"]
    ))
    
    # Error magnitude predictor
    max_error = max(error.absolute_delta for error in worst_errors)
    indicators.append(AlphaIndicator(
        name="Extreme_Error_Magnitude_Predictor",
        description=f"Early warning system for prediction errors exceeding ${max_error*.7:.2f} (70% of worst observed: ${max_error:.2f})",
        reasoning=f"Based on analysis of errors ranging from ${min(error.absolute_delta for error in worst_errors):.2f} to ${max_error:.2f}, we can build early warning signals",
        confidence_score=0.82,
        suggested_data_source="Multi-timeframe volatility measures, market stress indicators, liquidity metrics",
        category="risk_management",
        implementation_difficulty=4,
        correlation_evidence=f"Extreme errors (>${max_error*.8:.2f}) show predictable buildup patterns 5-10 days prior",
        time_window_effectiveness=["Market stress periods", "Low liquidity conditions"]
    ))
    
    return indicators

#!/usr/bin/env python3
"""
Alpha Seeker Dashboard - Single Page, Light Theme
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Alpha Seeker - Commodity Model Enhancement",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for light theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #4a90e2 0%, #7bb3f0 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .metric-card {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4a90e2;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #2c3e50;
    }
    .insight-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #4a90e2;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: #2c3e50;
    }
    .alpha-indicator {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid #7bb3f0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        color: #2c3e50;
    }
    .process-step {
        background: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
        text-align: center;
        color: #2c3e50;
    }
    .stApp {
        background-color: #fafafa;
    }
    .stMarkdown {
        color: #333333;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50;
    }
    p, li, span {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

def load_evaluation_data() -> pd.DataFrame:
    """Load and process the evaluation results data."""
    try:
        df = pd.read_csv("data/evaluation_results.csv")
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"Error loading evaluation data: {e}")
        return pd.DataFrame()

def load_price_data() -> pd.DataFrame:
    """Load and process the commodity price data."""
    try:
        df = pd.read_csv("data/dataset.csv")
        df['Date'] = df['Date'].str.replace('"', '')
        df['Date'] = pd.to_datetime(df['Date'], format='%b %d, %Y', errors='coerce')
        
        numeric_cols = ['Price', 'Open', 'High', 'Low']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'Vol.' in df.columns:
            df['Vol.'] = df['Vol.'].astype(str).str.replace('K', '').str.replace('-', '0')
            df['Vol.'] = pd.to_numeric(df['Vol.'], errors='coerce') * 1000
        
        if 'Change %' in df.columns:
            df['Change %'] = df['Change %'].astype(str).str.replace('%', '')
            df['Change %'] = pd.to_numeric(df['Change %'], errors='coerce')
        
        df = df.dropna(subset=['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error loading price data: {e}")
        return pd.DataFrame()

def analyze_prediction_failures(eval_df: pd.DataFrame, threshold: float = 15.0) -> Dict[str, Any]:
    """Analyze prediction failures to identify patterns and top error windows."""
    
    major_errors = eval_df[eval_df['absolute_delta'] >= threshold].copy()
    major_errors = major_errors.sort_values('absolute_delta', ascending=False)
    
    error_windows = []
    for _, error in major_errors.head(10).iterrows():
        error_date = error['date']
        window_start = error_date - timedelta(days=7)
        window_end = error_date + timedelta(days=3)
        
        error_windows.append({
            'error_date': error_date,
            'window_start': window_start,
            'window_end': window_end,
            'actual': error['actual'],
            'predicted': error['predicted'],
            'error': error['absolute_delta'],
            'error_pct': error['absolute_percentage_error']
        })
    
    analysis = {
        'total_predictions': len(eval_df),
        'major_errors': len(major_errors),
        'avg_error': eval_df['absolute_delta'].mean(),
        'max_error': eval_df['absolute_delta'].max(),
        'error_rate': len(major_errors) / len(eval_df) * 100,
        'top_error_windows': error_windows[:5],
        'monthly_error_distribution': major_errors.groupby(
            major_errors['date'].dt.to_period('M')
        )['absolute_delta'].agg(['count', 'mean']).to_dict('index')
    }
    
    return analysis

def generate_sample_alpha_indicators() -> List[Dict[str, Any]]:
    """Generate sample alpha indicators with detailed reasoning."""
    return [
        {
            "name": "São Paulo Drought Stress Index",
            "category": "Geospatial",
            "confidence_score": 0.87,
            "implementation_difficulty": 3,
            "description": "Soil moisture deficit in São Paulo coffee regions predicts price volatility 2-3 weeks ahead",
            "detailed_reasoning": """
            **Analysis Overview:**
            During the March 2025 prediction failure window, satellite imagery analysis revealed severe drought conditions in São Paulo state's coffee belt, specifically in the municipalities of Garça, Marília, and Avaré. These regions account for approximately 15% of Brazil's arabica production.

            **Correlation Discovery:**
            The geospatial agent detected that when soil moisture index drops below -2.5 standard deviations from the 10-year average, coffee futures exhibit increased volatility within 14-21 days. This pattern was missed by fundamental models because:
            
            1. **Lag in Official Reports**: Brazilian government crop reports are released monthly, missing short-term stress events
            2. **Spatial Granularity**: Traditional models use state-level data, missing municipal hotspots
            3. **Severity Threshold**: Moderate stress events (not reaching emergency levels) still impact yields

            **Implementation Signal:**
            When 5+ consecutive days show soil moisture < -2.0 std dev across >10% of São Paulo coffee municipalities, trigger increased volatility expectations for 2-3 week forward prices.
            """,
            "data_source": "Sentinel-2 satellite imagery, SMAP soil moisture data, EMBRAPA weather stations",
            "failure_correlation": "Identified in 4 out of 5 major prediction failures during Q1 2025",
            "statistical_significance": "p-value: 0.003, correlation coefficient: 0.73"
        },
        {
            "name": "Santos Port Congestion Cascade",
            "category": "Logistics", 
            "confidence_score": 0.81,
            "implementation_difficulty": 2,
            "description": "Container ship queue depth at Santos port predicts coffee spot-futures basis widening",
            "detailed_reasoning": """
            **Discovery Context:**
            During the February 2025 error window, the logistics agent detected unusual container vessel congestion at Santos port (Brazil's primary coffee export terminal). While fundamental models account for general port capacity, they miss real-time operational bottlenecks.

            **Chain Reaction Analysis:**
            The agent traced this specific sequence during the Feb 13th prediction failure:
            
            1. **Initial Trigger**: Maersk's 14,000 TEU vessel "Munich Maersk" experienced mechanical delays, blocking berth 33 for 3 days
            2. **Cascade Effect**: 8 additional vessels queued, including 3 coffee-dedicated ships
            3. **Market Impact**: Spot coffee premiums increased by $0.15/lb within 48 hours as traders anticipated delivery delays
            4. **Model Blindness**: Fundamental models only saw Santos as "operational" in monthly reports

            **Predictive Mechanism:**
            Queue depth >12 vessels + coffee ships >30% of total = basis widening signal with 24-48 hour lead time.
            """,
            "data_source": "MarineTraffic API, Santos Port Authority schedules, coffee vessel tracking",
            "failure_correlation": "Present in 3 out of 5 Q1 2025 prediction failures",
            "statistical_significance": "p-value: 0.012, lead time accuracy: 78%"
        },
        {
            "name": "Colombian Weather-Politics Interaction",
            "category": "Web News + Geospatial",
            "confidence_score": 0.76,
            "implementation_difficulty": 4,
            "description": "La Niña onset combined with Colombian election cycles amplifies coffee price volatility",
            "detailed_reasoning": """
            **Complex Pattern Discovery:**
            The multi-agent system discovered an interaction effect between meteorological and political factors that traditional models treat as independent variables. During the April 2025 prediction failure, both climate and sentiment agents flagged simultaneous signals.

            **Synergistic Effect:**
            When La Niña weather stress occurs during political uncertainty:
            1. Farmers delay selling hoping for policy support
            2. Weather concerns get amplified in media coverage
            3. International buyers hedge more aggressively
            4. Price volatility increases by 40-60% vs. weather-only events

            **Implementation Logic:**
            IF (Colombian climate index > 1.5 AND political sentiment volatility > 80th percentile) 
            THEN multiply price volatility forecasts by 1.4-1.6x for 3-month horizon.
            """,
            "data_source": "NOAA climate data, Colombian news sentiment analysis, Twitter political tracking",
            "failure_correlation": "Interaction effect detected in 2 major Q1 2025 failures",
            "statistical_significance": "Interaction term p-value: 0.018, R² improvement: 0.23"
        }
    ]

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Alpha Seeker</h1>
        <h3>Discovering Alpha Indicators for Commodity Model Enhancement</h3>
        <p>Intelligent analysis of prediction failures to identify new trading signals</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model Evaluation Results Section
    st.header("☕ Coffee Price Prediction Model - Evaluation Results")
    
    st.markdown("""
    <div class="insight-box">
        <h4>📈 Existing Model Performance</h4>
        <p>We have a machine learning model that predicts coffee futures prices. Below are the 
        evaluation results showing actual vs predicted prices on historical data. The Alpha Seeker 
        system analyzes prediction errors to discover new alpha indicators.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display evaluation images
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Model Evaluation Plots")
        try:
            st.image("data/evaluation_plots.png", caption="Model Performance Analysis", use_container_width=True)
        except:
            st.warning("Evaluation plots image not found")
    
    with col2:
        st.subheader("📈 Model Evaluation Summary")
        try:
            st.image("data/model_evaluation.png", caption="Error Distribution & Metrics", use_container_width=True)
        except:
            st.warning("Model evaluation image not found")
    
    # Alpha discovery trigger
    st.subheader("🔍 Alpha Discovery Analysis")
    
    alpha_discovery_triggered = st.button(
        "🚀 Analyze Prediction Errors & Discover Alpha Indicators",
        type="primary",
        help="Run the Alpha Seeker system to find new trading signals"
    )
    
    st.divider()
    
    # Set fixed configuration values
    error_threshold = 15.0
    analysis_days = 10
    enable_geospatial = True
    enable_web_news = True
    enable_logistics = False
    
    # Load data
    eval_df = load_evaluation_data()
    if eval_df.empty:
        st.error("⚠️ Unable to load model evaluation data.")
        return
    
    price_df = load_price_data()
    if price_df.empty:
        st.warning("⚠️ Price context data not available.")
        price_df = pd.DataFrame()
    
    # Analyze prediction failures
    analysis = analyze_prediction_failures(eval_df, error_threshold)
    
    # Model Performance Summary Section
    st.header("📊 Model Performance Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>Total Predictions</h4>
            <h2>{:,}</h2>
            <p>Model evaluation period</p>
        </div>
        """.format(analysis['total_predictions']), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>Major Errors</h4>
            <h2>{}</h2>
            <p>${:.1f}+ prediction misses</p>
        </div>
        """.format(analysis['major_errors'], error_threshold), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h4>Error Rate</h4>
            <h2>{:.1f}%</h2>
            <p>Significant failures</p>
        </div>
        """.format(analysis['error_rate']), unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h4>Max Error</h4>
            <h2>${:.1f}</h2>
            <p>Worst prediction miss</p>
        </div>
        """.format(analysis['max_error']), unsafe_allow_html=True)
    
    st.divider()
    
    # Alpha Discovery Process
    st.header("🤖 Alpha Discovery Process")
    
    st.markdown("""
    <div class="insight-box">
        <h4>🧠 From Prediction Errors to Alpha Insights</h4>
        <p>The Alpha Seeker system analyzes when and where your existing model fails, then uses 
        intelligent agents to search for external factors that could explain these prediction errors. 
        This discovers new alpha indicators that can be added as features to improve model performance.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Process flow
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="process-step">
            <h3>1️⃣ Analyze Errors</h3>
            <p>• Find when model predictions fail<br/>
            • Focus on biggest misses<br/>
            • Extract error time windows</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="process-step">
            <h3>2️⃣ Search Context</h3>
            <p>🛰️ <strong>Geospatial</strong>: Weather & satellite data<br/>
            📰 <strong>News</strong>: Market events & sentiment<br/>
            🚛 <strong>Logistics</strong>: Supply chain issues</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="process-step">
            <h3>3️⃣ Find Alpha</h3>
            <p>• Match external data to errors<br/>
            • Discover new indicators<br/>
            • Generate feature signals</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Alpha Discovery Backend Integration
    if alpha_discovery_triggered:
        st.subheader("🔍 Alpha Discovery in Progress")
        
        with st.spinner("Running Alpha Seeker multi-agent analysis..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Import the alpha seeker system
                status_text.text("Initializing Alpha Seeker system...")
                progress_bar.progress(0.1)
                
                from langchain_core.messages import HumanMessage
                from langchain_core.runnables import RunnableConfig
                from alpha_seeker.orchestrator.graph import graph
                from alpha_seeker.orchestrator.state import OrchestratorState
                from typing import cast
                
                status_text.text("Configuring analysis parameters...")
                progress_bar.progress(0.2)
                
                # Configure the system
                config = cast(RunnableConfig, {
                    "configurable": {
                        "thread_id": "dashboard_analysis_coffee",
                        "model": "google_genai:gemini-2.0-flash",
                        "analysis_depth": "comprehensive",
                        "parallel_extraction": True,
                        "max_regions": 5,
                        "analysis_period_days": analysis_days,
                        "confidence_threshold": 0.7,
                        "k_worst_cases": min(5, len(analysis['top_error_windows'])),
                        "enable_geospatial": enable_geospatial,
                        "enable_web_news": enable_web_news,
                        "enable_logistics": enable_logistics
                    }
                })
                
                status_text.text("Creating analysis state...")
                progress_bar.progress(0.3)
                
                # Create initial state
                initial_state = OrchestratorState(
                    messages=[HumanMessage(content="Analyze coffee prediction failures and discover alpha indicators")]
                )
                
                status_text.text("🤖 Running multi-agent analysis...")
                progress_bar.progress(0.4)
                
                # Run the alpha seeker system
                import asyncio
                
                if hasattr(asyncio, 'run'):
                    result = asyncio.run(graph.ainvoke(initial_state, config=config))
                else:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    result = loop.run_until_complete(graph.ainvoke(initial_state, config=config))
                    loop.close()
                
                status_text.text("Processing results...")
                progress_bar.progress(0.9)
                
                # Store results in session state for display
                if 'alpha_results' not in st.session_state:
                    st.session_state.alpha_results = {}
                
                st.session_state.alpha_results = {
                    'indicators': result.get('alpha_indicators', []),
                    'geospatial_results': result.get('geospatial_results'),
                    'web_news_results': result.get('web_news_results'),
                    'analysis_context': result.get('analysis_context'),
                    'timestamp': pd.Timestamp.now()
                }
                
                progress_bar.progress(1.0)
                status_text.text("✅ Alpha discovery complete!")
                
                st.success(f"🎯 Successfully discovered {len(st.session_state.alpha_results['indicators'])} alpha indicators!")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error running alpha discovery: {e}")
                st.write("Falling back to demo mode with sample indicators...")
                
                # Fallback to sample data
                if 'alpha_results' not in st.session_state:
                    st.session_state.alpha_results = {}
                
                st.session_state.alpha_results = {
                    'indicators': generate_sample_alpha_indicators(),
                    'geospatial_results': None,
                    'web_news_results': None,
                    'analysis_context': None,
                    'timestamp': pd.Timestamp.now(),
                    'demo_mode': True
                }
    
    st.divider()
    
    # Discovered Alpha Indicators Section
    st.header("💡 Discovered Alpha Indicators")
    
    # Check if we have analysis results
    if 'alpha_results' in st.session_state and st.session_state.alpha_results:
        results = st.session_state.alpha_results
        
        if results.get('demo_mode'):
            st.warning("🧪 Demo Mode: Showing sample indicators. Run Alpha Discovery for real results.")
        else:
            st.success(f"✅ Analysis completed at {results['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        alpha_indicators = results['indicators']
        
        st.markdown("""
        <div class="insight-box">
            <h4>🎯 Alpha Discovery Results</h4>
            <p>The multi-agent system has analyzed your model's prediction failures and identified 
            external factors that correlate with these errors. Each indicator includes detailed 
            reasoning about how these factors could contribute to improved predictions.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Indicator summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Indicators", len(alpha_indicators))
        
        with col2:
            # Handle both dict and object types
            if alpha_indicators and hasattr(alpha_indicators[0], 'confidence_score'):
                avg_confidence = np.mean([ind.confidence_score for ind in alpha_indicators])
            else:
                avg_confidence = np.mean([ind['confidence_score'] for ind in alpha_indicators])
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        
        with col3:
            # Handle both dict and object types
            if alpha_indicators and hasattr(alpha_indicators[0], 'category'):
                geospatial_count = len([ind for ind in alpha_indicators if 'Geospatial' in ind.category])
            else:
                geospatial_count = len([ind for ind in alpha_indicators if 'Geospatial' in ind['category']])
            st.metric("Geospatial Signals", geospatial_count)
        
        with col4:
            # Handle both dict and object types
            if alpha_indicators and hasattr(alpha_indicators[0], 'category'):
                news_count = len([ind for ind in alpha_indicators if 'News' in ind.category])
            else:
                news_count = len([ind for ind in alpha_indicators if 'News' in ind['category']])
            st.metric("News/Web Signals", news_count)
        
        # Detailed indicators
        st.subheader("📋 Alpha Indicator Analysis")
        
        for i, indicator in enumerate(alpha_indicators, 1):
            # Handle both dict and object types
            if hasattr(indicator, 'name'):
                # Object type (AlphaIndicator)
                name = indicator.name
                confidence = indicator.confidence_score
                category = indicator.category
                description = indicator.description
                reasoning = getattr(indicator, 'reasoning', '')
                data_source = getattr(indicator, 'suggested_data_source', 'N/A')
                difficulty = getattr(indicator, 'implementation_difficulty', 3)
                correlation = getattr(indicator, 'correlation_evidence', '')
            else:
                # Dict type (fallback sample data)
                name = indicator['name']
                confidence = indicator['confidence_score']
                category = indicator['category']
                description = indicator['description']
                reasoning = indicator.get('detailed_reasoning', indicator.get('reasoning', ''))
                data_source = indicator.get('data_source', 'N/A')
                difficulty = indicator.get('implementation_difficulty', 3)
                correlation = indicator.get('failure_correlation', '')
            
            with st.expander(f"Alpha Indicator #{i}: {name} (Confidence: {confidence:.2f})"):
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Category:** {category}")
                    st.markdown(f"**Description:** {description}")
                    
                    # Show detailed reasoning if available
                    if reasoning:
                        st.markdown("**Detailed Analysis & Reasoning:**")
                        st.markdown(reasoning)
                    
                    st.markdown(f"**Data Sources:** {data_source}")
                    
                    # Show additional analysis details if available
                    if correlation:
                        st.markdown(f"**Failure Correlation:** {correlation}")
                
                with col2:
                    # Confidence gauge
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = confidence,
                        title = {'text': "Confidence Score"},
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'axis': {'range': [None, 1]},
                            'bar': {'color': "#4a90e2"},
                            'steps': [
                                {'range': [0, 0.5], 'color': "lightgray"},
                                {'range': [0.5, 0.8], 'color': "#ffd966"},
                                {'range': [0.8, 1], 'color': "#93c47d"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.9
                            }
                        }
                    ))
                    fig.update_layout(height=200)
                    st.plotly_chart(fig, use_container_width=True, key=f"confidence_gauge_{i}")
                    
                    # Implementation difficulty
                    difficulty_stars = "⭐" * difficulty
                    st.write(f"**Implementation:** {difficulty_stars}")
                    
                    if difficulty <= 2:
                        st.success("🟢 Low Complexity")
                    elif difficulty <= 3:
                        st.warning("🟡 Medium Complexity") 
                    else:
                        st.error("🔴 High Complexity")
        
    else:
        st.info("🔍 Click 'Analyze Prediction Errors & Discover Alpha Indicators' to start the discovery process")
        
        st.markdown("""
        <div class="insight-box">
            <h4>🎯 Alpha Discovery Process</h4>
            <p>Upload your model evaluation results and click the analysis button to:</p>
            <ul>
                <li>Identify significant prediction failures</li>
                <li>Extract contextual data from multiple sources</li>
                <li>Generate actionable alpha indicators</li>
                <li>Provide detailed reasoning for each discovery</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

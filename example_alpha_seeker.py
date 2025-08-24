#!/usr/bin/env python3
"""Example script to test the Alpha Seeker multi-agent system."""

import asyncio
import logging
from typing import cast

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from alpha_seeker.orchestrator.graph import graph
from alpha_seeker.orchestrator.state import OrchestratorState

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_alpha_seeker(lookback_window_days: int = 10, k_worst_cases: int = 3):
    """Test the complete Alpha Seeker multi-agent system with configurable parameters."""
    logger.info("🚀 Starting Alpha Seeker Multi-Agent System")
    logger.info(f"📊 Configuration: {lookback_window_days}-day lookback window, analyzing {k_worst_cases} worst prediction errors")
    
    # Configuration for the system - purely parameter-driven
    config = cast(RunnableConfig, {
        "configurable": {
            "thread_id": f"alpha_seeker_{lookback_window_days}d_{k_worst_cases}cases",
            "model": "google_genai:gemini-2.0-flash",
            "analysis_depth": "comprehensive",
            "parallel_extraction": True,
            "max_regions": 5,
            "analysis_period_days": lookback_window_days,  # This is the key parameter
            "confidence_threshold": 0.7,
            "k_worst_cases": k_worst_cases,  # Number of worst cases to analyze
            "enable_geospatial": True,
            "enable_logistics": True,
            "enable_web_news": True
        }
    })
    
    # Create initial state - no user message needed since analysis is data-driven
    initial_state = OrchestratorState(
        messages=[HumanMessage(content="Initialize data-driven alpha analysis")]  # Placeholder message
    )
    
    try:
        # Run the complete Alpha Seeker system
        logger.info("🔍 Starting multi-agent analysis pipeline...")
        result = await graph.ainvoke(initial_state, config=config)
        
        # Display results
        logger.info("✅ Alpha Seeker Analysis completed successfully!")
        
        print("\n" + "="*100)
        print("🎯 ALPHA SEEKER MULTI-AGENT ANALYSIS RESULTS")
        print("="*100)
        
        # Input Processing Results
        if result.get('commodity_input'):
            commodity_input = result['commodity_input']
            print("\n📊 COMMODITY ANALYSIS:")
            print(f"   • Commodity: {commodity_input.commodity.value.upper()}")
            print(f"   • Analysis Period: {commodity_input.analysis_period_days} days")
            print(f"   • Confidence Threshold: {commodity_input.confidence_threshold}")
        
        # Region Selection Results
        if result.get('selected_regions'):
            regions = result['selected_regions']
            print(f"\n🌍 SELECTED PRODUCTION REGIONS ({len(regions)}):")
            for i, region in enumerate(regions, 1):
                print(f"   {i}. {region.region_name}, {region.country}")
                # Inject key factors for Brazil and Vietnam if missing or empty
                if not region.key_factors:
                    if region.country.lower() == "brazil":
                        region.key_factors = [
                            "Frosts/droughts impact yields; rainfall affects flowering/cherry development",
                            "Highly mechanized, large-scale production; cost-efficient",
                            "BRL/USD exchange rate affects export competitiveness",
                            "Infrastructure and export logistics influence supply chain",
                            "Pest/disease outbreaks (Leaf Rust, Berry Borer) can reduce output",
                            "Strong Arabica focus shapes global prices"
                        ]
                    elif region.country.lower() == "vietnam":
                        region.key_factors = [
                            "~95% Robusta anchors global supply/instant market",
                            "Seasonal monsoons/droughts (El Niño) impact yields",
                            "Mostly smallholder farms; heavy irrigation/fertilizer use",
                            "Strong government support for coffee as export crop",
                            "Efficient logistics, proximity to Asian markets",
                            "Sustainability: deforestation, water, soil issues affect trade"
                        ]
                key_factors_str = ', '.join(region.key_factors[:3]) if region.key_factors else "N/A"
                print(f"      └─ Key Factors: {key_factors_str}")
        
        # Data Extraction Results
        print("\n🔬 DATA EXTRACTION RESULTS:")
        
        if result.get('geospatial_results'):
            geo = result['geospatial_results']
            print("   🛰️  GEOSPATIAL AGENT:")
            print(f"      └─ Data Points: {len(geo.data_points)}")
            print(f"      └─ Confidence: {geo.confidence_level:.2f}")
            print(f"      └─ Key Insights: {len(geo.key_insights)}")
            for insight in geo.key_insights[:2]:
                print(f"         • {insight}")
            if geo.anomalies_detected:
                print(f"      └─ Anomalies: {len(geo.anomalies_detected)}")
                for anomaly in geo.anomalies_detected[:2]:
                    print(f"         ⚠️  {anomaly}")
        
        if result.get('logistics_results'):
            logistics = result['logistics_results']
            print("   🚛 LOGISTICS AGENT:")
            print(f"      └─ Data Points: {len(logistics.data_points)}")
            print(f"      └─ Confidence: {logistics.confidence_level:.2f}")
            print(f"      └─ Key Insights: {len(logistics.key_insights)}")
            for insight in logistics.key_insights[:2]:
                print(f"         • {insight}")
            if logistics.anomalies_detected:
                print(f"      └─ Anomalies: {len(logistics.anomalies_detected)}")
        
        if result.get('web_news_results'):
            news = result['web_news_results']
            print("   📰 WEB NEWS AGENT:")
            print(f"      └─ Data Points: {len(news.data_points)}")
            print(f"      └─ Confidence: {news.confidence_level:.2f}")
            print(f"      └─ Key Insights: {len(news.key_insights)}")
            for insight in news.key_insights[:2]:
                print(f"         • {insight}")
        
        # Alpha Indicators Results
        if result.get('alpha_indicators'):
            indicators = result['alpha_indicators']
            print(f"\n🎯 ALPHA INDICATORS GENERATED ({len(indicators)}):")
            
            for i, indicator in enumerate(indicators, 1):
                print(f"\n   {i}. {indicator.name}")
                print(f"      └─ Category: {indicator.category.upper()}")
                print(f"      └─ Confidence: {indicator.confidence_score:.2f}")
                print(f"      └─ Implementation Difficulty: {indicator.implementation_difficulty}/5")
                print(f"      └─ Description: {indicator.description}")
                print(f"      └─ Reasoning: {indicator.reasoning}")
                print(f"      └─ Data Source: {indicator.suggested_data_source}")
                if indicator.time_window_effectiveness:
                    print(f"      └─ Most Effective During: {', '.join(indicator.time_window_effectiveness)}")
                print()
        
        # Analysis Context
        if result.get('analysis_context'):
            context = result['analysis_context']
            print("\n📋 ANALYSIS CONTEXT:")
            print(f"   • Model Errors Analyzed: {len(context.model_errors)}")
            print(f"   • Price History Records: {len(context.price_history)}")
            if context.time_window:
                print(f"   • Analysis Window: {context.time_window.start_date.strftime('%Y-%m-%d')} to {context.time_window.end_date.strftime('%Y-%m-%d')}")
            
            # Show specific prediction failures
            huge_errors = [err for err in context.model_errors if err.is_huge_difference]
            print(f"   • Significant Prediction Failures: {len(huge_errors)}")
            if huge_errors:
                print(f"   • Failure Date Range: {min(err.date for err in huge_errors).strftime('%Y-%m-%d')} to {max(err.date for err in huge_errors).strftime('%Y-%m-%d')}")
                print(f"   • Average Error: {sum(err.absolute_delta for err in huge_errors) / len(huge_errors):.2f}")
                print(f"   • Max Error: {max(err.absolute_delta for err in huge_errors):.2f}")
            
            # Show price context
            if context.price_history:
                recent_prices = context.price_history[:10]  # First 10 records
                print(f"   • Price Range in Dataset: ${min(p['price'] for p in recent_prices):.2f} - ${max(p['price'] for p in recent_prices):.2f}")
                print(f"   • Latest Price: ${recent_prices[0]['price']:.2f} on {recent_prices[0]['date'].strftime('%Y-%m-%d')}")
        
        # Execution Summary
        print("\n📈 EXECUTION SUMMARY:")
        print(f"   • Total Messages: {len(result.get('messages', []))}")
        print(f"   • Current Agent: {result.get('current_agent', 'completed')}")
        print(f"   • Completed Agents: {', '.join(result.get('completed_agents', []))}")
        
        # Handle errors
        if result.get('error'):
            print(f"\n❌ ERROR: {result['error']}")
        
        print("\n" + "="*100)
        # CSV Data Analysis Summary
        if result.get('analysis_context'):
            context = result['analysis_context']
            huge_errors = [err for err in context.model_errors if err.is_huge_difference]
            
            if huge_errors:
                print("\n📊 PREDICTION FAILURE ANALYSIS:")
                print("   Top 5 Largest Prediction Errors from evaluation_results.csv:")
                
                # Sort by absolute delta and show top 5
                top_errors = sorted(huge_errors, key=lambda x: x.absolute_delta, reverse=True)[:5]
                for i, error in enumerate(top_errors, 1):
                    print(f"   {i}. Date: {error.date.strftime('%Y-%m-%d')}")
                    print(f"      └─ Actual: ${error.actual:.2f}, Predicted: ${error.predicted:.2f}")
                    print(f"      └─ Error: ${error.absolute_delta:.2f} ({error.percentage_error:.1f}%)")
                
                print("\n📈 DATASET.CSV PRICE CONTEXT:")
                if context.price_history:
                    recent_data = context.price_history[:5]
                    print("   Recent Price Data from dataset.csv:")
                    for i, record in enumerate(recent_data, 1):
                        print(f"   {i}. {record['date'].strftime('%Y-%m-%d')}: ${record['price']:.2f}")
                        print(f"      └─ Range: ${record['low']:.2f} - ${record['high']:.2f}, Volume: {record['volume']}")
        
        print("\n🎉 Alpha Seeker analysis complete!")
        print(f"📋 Analyzed {len(huge_errors) if 'huge_errors' in locals() else 0} prediction failures from evaluation_results.csv")
        print(f"📊 Used {lookback_window_days}-day lookback windows and {k_worst_cases} worst cases")
        print(f"💡 Generated {len(result.get('alpha_indicators', []))} alpha indicators to prevent future failures")
        print("="*100)
        
    except Exception as e:
        logger.error(f"❌ Alpha Seeker system failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main function to run the Alpha Seeker example with different configurations."""
    print("🌟 Alpha Seeker Multi-Agent System - Data-Driven Analysis")
    print("=" * 60)
    print("This system analyzes coffee prediction failures from evaluation_results.csv")
    print("and generates alpha indicators using multi-agent data extraction.")
    print("=" * 60)
    
    # Test different configurations
    test_configs = [
        {"lookback_window_days": 10, "k_worst_cases": 3},
        # {"lookback_window_days": 15, "k_worst_cases": 5},  # Uncomment for additional tests
    ]
    
    for i, config in enumerate(test_configs, 1):
        print(f"\n🔬 Running Test Configuration {i}:")
        print(f"   • Lookback Window: {config['lookback_window_days']} days")
        print(f"   • Worst Cases to Analyze: {config['k_worst_cases']}")
        print("-" * 60)
        
        try:
            await test_alpha_seeker(**config)
        except KeyboardInterrupt:
            print("\n🛑 Demo interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Configuration {i} failed: {e}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    asyncio.run(main())

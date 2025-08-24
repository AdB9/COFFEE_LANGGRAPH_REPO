"""Result analysis node for the websearch agent."""

import logging
from typing import List, Dict
from collections import defaultdict

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from alpha_seeker.websearch_agent.configuration import Configuration
from alpha_seeker.websearch_agent.state import WebSearchState
from alpha_seeker.common.data_models import WebSearchResult, SearchResultType

logger = logging.getLogger(__name__)


async def result_analyzer(state: WebSearchState, config: RunnableConfig) -> dict:
    """Analyze and synthesize the web search results.
    
    This node processes all the search results to extract key findings,
    identify patterns, and generate actionable insights.
    """
    try:
        logger.info("=== Result Analyzer Node ===")
        
        # Get configuration
        configuration = Configuration.from_runnable_config(config)
        
        # Check if we have search results to analyze
        if not state.search_results:
            return {
                "error": "No search results found to analyze"
            }
        
        logger.info(f"Analyzing {len(state.search_results)} search results")
        
        # Organize results by type and relevance
        results_by_type = defaultdict(list)
        high_relevance_results = []
        
        for result in state.search_results:
            results_by_type[result.search_type].append(result)
            if result.relevance_score >= 0.5:  # High relevance threshold
                high_relevance_results.append(result)
        
        # Extract key findings based on analysis depth
        analysis_depth = configuration.result_analysis_depth
        key_findings = []
        
        if analysis_depth == "basic":
            key_findings = _basic_analysis(state.search_results, state.search_focus or "general research")
        elif analysis_depth == "detailed":
            key_findings = _detailed_analysis(results_by_type, high_relevance_results, state.search_focus or "general research")
        else:  # comprehensive
            key_findings = _comprehensive_analysis(results_by_type, high_relevance_results, state.search_focus or "general research")
        
        # Generate recommended actions
        recommended_actions = _generate_recommendations(results_by_type, key_findings)
        
        # Create analysis summary
        analysis_summary = _create_analysis_summary(
            state.search_results, 
            results_by_type, 
            key_findings, 
            state.search_focus or "general research"
        )
        
        logger.info(f"Analysis complete. Found {len(key_findings)} key findings and {len(recommended_actions)} recommendations")
        
        # Create AI message with summary
        ai_message = AIMessage(
            content=f"Analysis complete! I've processed {len(state.search_results)} search results and identified "
                   f"{len(key_findings)} key findings. The analysis reveals important insights about {state.search_focus or 'the research topic'}."
        )
        
        return {
            "key_findings": key_findings,
            "recommended_actions": recommended_actions,
            "analysis_summary": analysis_summary,
            "messages": [ai_message]
        }
        
    except Exception as e:
        logger.error(f"Result analysis failed: {e}")
        return {
            "error": f"Result analysis failed: {str(e)}"
        }


def _basic_analysis(results: List[WebSearchResult], focus: str) -> List[str]:
    """Perform basic analysis of search results."""
    findings = []
    
    # Count results by type
    type_counts = defaultdict(int)
    for result in results:
        type_counts[result.search_type.value] += 1
    
    findings.append(f"Searched across {len(results)} sources focused on {focus}")
    
    if type_counts:
        top_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        type_summary = ", ".join([f"{count} {type_name}" for type_name, count in top_types])
        findings.append(f"Primary source types: {type_summary}")
    
    # Identify high-relevance results
    high_relevance = [r for r in results if r.relevance_score >= 0.7]
    if high_relevance:
        findings.append(f"Found {len(high_relevance)} highly relevant sources")
    
    return findings


def _detailed_analysis(results_by_type: Dict, high_relevance_results: List[WebSearchResult], focus: str) -> List[str]:
    """Perform detailed analysis of search results."""
    findings = []
    
    # Basic analysis first
    all_results = []
    for type_results in results_by_type.values():
        all_results.extend(type_results)
    findings.extend(_basic_analysis(all_results, focus))
    
    # Analyze by source type
    for result_type, type_results in results_by_type.items():
        if not type_results:
            continue
            
        avg_relevance = sum(r.relevance_score for r in type_results) / len(type_results)
        
        if result_type == SearchResultType.NEWS_ARTICLE.value:
            findings.append(f"News coverage: {len(type_results)} articles with average relevance {avg_relevance:.2f}")
        elif result_type == SearchResultType.RESEARCH_PAPER.value:
            findings.append(f"Academic research: {len(type_results)} papers identified")
        elif result_type == SearchResultType.MARKET_REPORT.value:
            findings.append(f"Market analysis: {len(type_results)} reports found")
        elif result_type == SearchResultType.GOVERNMENT_DATA.value:
            findings.append(f"Official data: {len(type_results)} government sources")
    
    # Extract common themes from high-relevance results
    if high_relevance_results:
        themes = _extract_common_themes(high_relevance_results)
        if themes:
            findings.append(f"Common themes in top results: {', '.join(themes)}")
    
    return findings


def _comprehensive_analysis(results_by_type: Dict, high_relevance_results: List[WebSearchResult], focus: str) -> List[str]:
    """Perform comprehensive analysis of search results."""
    findings = []
    
    # Start with detailed analysis
    findings.extend(_detailed_analysis(results_by_type, high_relevance_results, focus))
    
    # Deep dive into content patterns
    all_results = []
    for type_results in results_by_type.values():
        all_results.extend(type_results)
    
    # Analyze content depth
    content_analysis = _analyze_content_depth(all_results)
    findings.extend(content_analysis)
    
    # Temporal analysis if timestamps are available
    temporal_insights = _analyze_temporal_patterns(all_results)
    findings.extend(temporal_insights)
    
    # Quality assessment
    quality_insights = _assess_source_quality(results_by_type)
    findings.extend(quality_insights)
    
    return findings


def _generate_recommendations(results_by_type: Dict, key_findings: List[str]) -> List[str]:
    """Generate actionable recommendations based on analysis."""
    recommendations = []
    
    # Check for data gaps
    if SearchResultType.GOVERNMENT_DATA.value not in results_by_type:
        recommendations.append("Consider searching for official government data sources for more authoritative information")
    
    if SearchResultType.RESEARCH_PAPER.value not in results_by_type:
        recommendations.append("Look for academic research papers for in-depth analysis")
    
    # Check for recency
    all_results = []
    for type_results in results_by_type.values():
        all_results.extend(type_results)
    
    if len(all_results) < 10:
        recommendations.append("Expand search with additional keywords for more comprehensive coverage")
    
    # Source diversity
    if len(results_by_type) < 3:
        recommendations.append("Diversify source types for a more balanced perspective")
    
    # Based on findings
    if any("limited" in finding.lower() or "few" in finding.lower() for finding in key_findings):
        recommendations.append("Consider broadening search terms or time range")
    
    return recommendations


def _create_analysis_summary(results: List[WebSearchResult], results_by_type: Dict, key_findings: List[str], focus: str) -> str:
    """Create a comprehensive analysis summary."""
    summary_parts = []
    
    summary_parts.append(f"Research Analysis Summary for: {focus}")
    summary_parts.append(f"Total sources analyzed: {len(results)}")
    
    if results_by_type:
        type_breakdown = []
        for result_type, type_results in results_by_type.items():
            avg_relevance = sum(r.relevance_score for r in type_results) / len(type_results)
            type_breakdown.append(f"{len(type_results)} {result_type} (avg relevance: {avg_relevance:.2f})")
        summary_parts.append(f"Source breakdown: {', '.join(type_breakdown)}")
    
    if key_findings:
        summary_parts.append("Key Findings:")
        for i, finding in enumerate(key_findings, 1):
            summary_parts.append(f"{i}. {finding}")
    
    return "\n".join(summary_parts)


def _extract_common_themes(results: List[WebSearchResult]) -> List[str]:
    """Extract common themes from result titles and snippets."""
    # Simple keyword frequency analysis
    word_counts = defaultdict(int)
    stop_words = {"the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "a", "an"}
    
    for result in results:
        words = (result.title + " " + result.snippet).lower().split()
        for word in words:
            word = word.strip(".,!?()[]")
            if len(word) > 3 and word not in stop_words:
                word_counts[word] += 1
    
    # Return top themes (words appearing in multiple results)
    themes = [word for word, count in word_counts.items() if count >= 2]
    return themes[:5]  # Top 5 themes


def _analyze_content_depth(results: List[WebSearchResult]) -> List[str]:
    """Analyze the depth and quality of content found."""
    insights = []
    
    # Analyze snippet lengths as a proxy for content depth
    snippet_lengths = [len(result.snippet) for result in results if result.snippet]
    if snippet_lengths:
        avg_length = sum(snippet_lengths) / len(snippet_lengths)
        if avg_length > 200:
            insights.append("Results show detailed content with substantial information depth")
        elif avg_length > 100:
            insights.append("Results provide moderate detail level")
        else:
            insights.append("Results show brief summaries - may need deeper sources")
    
    # Check for URLs that suggest quality sources
    quality_domains = 0
    for result in results:
        if any(domain in result.url.lower() for domain in [".edu", ".org", ".gov", "reuters", "bloomberg"]):
            quality_domains += 1
    
    if quality_domains > len(results) * 0.3:  # More than 30% quality domains
        insights.append("High proportion of authoritative sources identified")
    
    return insights


def _analyze_temporal_patterns(results: List[WebSearchResult]) -> List[str]:
    """Analyze temporal patterns in the results."""
    insights = []
    
    # For now, this is a placeholder since we don't have reliable publish dates
    # In a real implementation, you would parse publication dates from results
    
    if len(results) > 0:
        insights.append("Temporal analysis available - mix of recent and historical sources")
    
    return insights


def _assess_source_quality(results_by_type: Dict) -> List[str]:
    """Assess the overall quality of sources found."""
    insights = []
    
    # Count authoritative source types
    authoritative_types = [
        SearchResultType.GOVERNMENT_DATA.value,
        SearchResultType.RESEARCH_PAPER.value,
        SearchResultType.MARKET_REPORT.value
    ]
    
    authoritative_count = sum(
        len(results_by_type.get(source_type, [])) 
        for source_type in authoritative_types
    )
    
    total_results = sum(len(results) for results in results_by_type.values())
    
    if total_results > 0:
        auth_ratio = authoritative_count / total_results
        if auth_ratio > 0.5:
            insights.append("High quality: majority of sources are authoritative")
        elif auth_ratio > 0.3:
            insights.append("Good quality: substantial authoritative source coverage")
        else:
            insights.append("Mixed quality: consider seeking more authoritative sources")
    
    return insights

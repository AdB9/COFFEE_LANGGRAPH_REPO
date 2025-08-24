"""Region selection node for the orchestrator."""

import logging
import re
from typing import Dict, Any, List, Optional

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from alpha_seeker.orchestrator.state import OrchestratorState
from alpha_seeker.common.data_models import RegionInfo, CommodityType
from alpha_seeker.websearch_agent.tools.web_search import multi_query_search

logger = logging.getLogger(__name__)


async def region_selector(state: OrchestratorState, config: RunnableConfig) -> Dict[str, Any]:
    """Select key production regions for the specified commodity using web search."""
    
    try:
        logger.info("=== Region Selector Node ===")
        
        if not state.commodity_input:
            return {"error": "No commodity input available for region selection"}
        
        commodity = state.commodity_input.commodity
        logger.info(f"Searching for major production regions for {commodity.value}")
        
        # Generate search queries to find production regions
        search_queries = _generate_region_search_queries(commodity)
        logger.info(f"Generated {len(search_queries)} search queries for region discovery")
        
        # Perform web searches to find production regions
        search_result = await multi_query_search.ainvoke({
            "queries": search_queries,
            "max_results_per_query": 8,
            "search_engine": "duckduckgo"
        })
        
        # Parse and extract regions from search results
        selected_regions = await _extract_regions_from_search_results(
            search_result, commodity
        )
        
        if not selected_regions:
            logger.warning(f"No regions found from web search for {commodity.value}, using fallback")
            # Only use minimal fallback if search completely fails
            selected_regions = _get_fallback_regions(commodity)
        
        logger.info(f"Selected {len(selected_regions)} regions for {commodity.value}")
        for region in selected_regions:
            logger.info(f"  - {region.region_name}, {region.country}")
        
        # Create AI message about region selection
        region_names = [f"{r.region_name} ({r.country})" for r in selected_regions]
        ai_message = AIMessage(
            content=f"Identified {len(selected_regions)} key production regions for {commodity.value} "
                   f"through web search: {', '.join(region_names)}. These regions are major global producers."
        )
        
        return {
            "selected_regions": selected_regions,
            "messages": [ai_message]
        }
        
    except Exception as e:
        logger.error(f"Region selection failed: {e}")
        return {"error": f"Region selection failed: {str(e)}"}


def _generate_region_search_queries(commodity: CommodityType) -> List[str]:
    """Generate search queries to find major production regions for a commodity."""
    commodity_name = commodity.value.lower()
    
    queries = [
        f"major {commodity_name} producing regions world",
        f"largest {commodity_name} producers by country",
        f"top {commodity_name} production areas globally",
        f"{commodity_name} growing regions climate conditions",
        f"{commodity_name} production statistics by region",
        f"world {commodity_name} production map regions",
        f"leading {commodity_name} producing countries 2024"
    ]
    
    return queries


async def _extract_regions_from_search_results(
    search_result: Dict[str, Any], 
    commodity: CommodityType
) -> List[RegionInfo]:
    """Extract production regions from web search results."""
    regions = []
    all_results = search_result.get("all_results", [])
    
    # Track found regions to avoid duplicates
    found_regions = set()
    
    # Common patterns for extracting regions and countries
    region_patterns = [
        # Pattern: "Region, Country" or "Region (Country)"
        r'([A-Z][a-zA-Z\s]+),\s*([A-Z][a-zA-Z\s]+)',
        r'([A-Z][a-zA-Z\s]+)\s*\(([A-Z][a-zA-Z\s]+)\)',
        # Pattern: Country names followed by region indicators
        r'([A-Z][a-zA-Z\s]+)\s+(?:region|state|province|area)',
    ]
    
    # Country-specific region keywords
    country_regions = {
        'brazil': ['minas gerais', 'são paulo', 'bahia', 'espírito santo', 'paraná'],
        'vietnam': ['central highlands', 'dak lak', 'gia lai', 'kon tum'],
        'colombia': ['huila', 'cauca', 'nariño', 'tolima', 'quindío'],
        'ethiopia': ['kaffa', 'sidamo', 'yirgacheffe', 'harrar'],
        'guatemala': ['antigua', 'cobán', 'huehuetenango'],
        'indonesia': ['sumatra', 'java', 'sulawesi'],
        'honduras': ['copán', 'santa bárbara', 'el paraíso'],
        'india': ['karnataka', 'kerala', 'tamil nadu', 'maharashtra'],
        'china': ['yunnan', 'guangxi', 'hainan'],
        'ivory coast': ["côte d'ivoire", 'abidjan', 'yamoussoukro'],
        'ghana': ['ashanti', 'western', 'central'],
        'ecuador': ['manabí', 'los ríos', 'guayas']
    }
    
    logger.info(f"Processing {len(all_results)} search results for region extraction")
    
    for result in all_results:
        content = f"{result.get('title', '')} {result.get('snippet', '')}".lower()
        
        # Extract regions using patterns
        for pattern in region_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if len(match) == 2:
                    region_name, country = match
                    region_key = f"{region_name.strip()}, {country.strip()}".lower()
                    
                    if region_key not in found_regions and len(region_name.strip()) > 2:
                        found_regions.add(region_key)
                        
                        regions.append(RegionInfo(
                            region_name=region_name.strip().title(),
                            country=country.strip().title(),
                            coordinates=_get_approximate_coordinates(
                                region_name.strip(), country.strip()
                            ),
                            production_volume=0,  # To be filled by downstream agents
                            region_type="production_area"
                        ))
        
        # Look for country-specific regions
        for country, region_list in country_regions.items():
            if country in content:
                for region in region_list:
                    if region in content:
                        region_key = f"{region}, {country}".lower()
                        if region_key not in found_regions:
                            found_regions.add(region_key)
                            
                            regions.append(RegionInfo(
                                region_name=region.title(),
                                country=country.title(),
                                coordinates=_get_approximate_coordinates(region, country),
                                production_volume=0,
                                region_type="production_area"
                            ))
    
    # Limit to top regions and sort by relevance (for now, just limit)
    regions = regions[:8]
    
    logger.info(f"Extracted {len(regions)} unique regions from search results")
    return regions


def _get_approximate_coordinates(region: str, country: str) -> tuple:
    """Get approximate coordinates for a region. This is a simple lookup."""
    # Basic coordinate lookup - in production, you'd use a geocoding service
    coordinate_map = {
        # Coffee regions
        ('minas gerais', 'brazil'): (-19.8157, -43.9542),
        ('são paulo', 'brazil'): (-23.5505, -46.6333),
        ('bahia', 'brazil'): (-12.9714, -38.5014),
        ('central highlands', 'vietnam'): (12.2677, 108.0051),
        ('huila', 'colombia'): (2.5358, -75.5223),
        ('kaffa', 'ethiopia'): (7.2525, 36.2425),
        ('antigua', 'guatemala'): (14.5586, -90.7342),
        ('sumatra', 'indonesia'): (0.5897, 101.3431),
        ('karnataka', 'india'): (15.3173, 75.7139),
        ('yunnan', 'china'): (25.0000, 101.0000),
        
        # Default country centers
        ('brazil', ''): (-14.2350, -51.9253),
        ('vietnam', ''): (14.0583, 108.2772),
        ('colombia', ''): (4.5709, -74.2973),
        ('ethiopia', ''): (9.1450, 40.4897),
        ('guatemala', ''): (15.7835, -90.2308),
        ('indonesia', ''): (-0.7893, 113.9213),
        ('india', ''): (20.5937, 78.9629),
        ('china', ''): (35.8617, 104.1954),
    }
    
    # Try exact match first
    key = (region.lower(), country.lower())
    if key in coordinate_map:
        return coordinate_map[key]
    
    # Try country default
    key = (country.lower(), '')
    if key in coordinate_map:
        return coordinate_map[key]
    
    # Default coordinates
    return (0.0, 0.0)


def _get_fallback_regions(commodity: CommodityType) -> List[RegionInfo]:
    """Minimal fallback regions if web search completely fails."""
    if commodity == CommodityType.COFFEE:
        return [
            RegionInfo(
                region_name="Major Coffee Region",
                country="Brazil",
                coordinates=(-19.8157, -43.9542),
                production_volume=0,
                region_type="fallback"
            ),
            RegionInfo(
                region_name="Major Coffee Region",
                country="Vietnam",
                coordinates=(12.2677, 108.0051),
                production_volume=0,
                region_type="fallback"
            )
        ]
    else:
        return [
            RegionInfo(
                region_name="Primary Production Region",
                country="Unknown",
                coordinates=(0.0, 0.0),
                production_volume=0,
                region_type="fallback"
            )
        ]


async def main():
    """Example of running the region_selector node with coffee input."""
    import asyncio
    from langchain_core.runnables import RunnableConfig
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=== Region Selector Example ===")
    print("Running region selector for COFFEE...")
    
    # Create input for coffee analysis
    from alpha_seeker.common.data_models import CommodityInput, CommodityType
    commodity_input = CommodityInput(
        commodity=CommodityType.COFFEE,
        analysis_period_days=90,
        confidence_threshold=0.7
    )
    
    # Create the orchestrator state
    state = OrchestratorState(
        messages=[],
        commodity_input=commodity_input
    )
    
    # Configuration for the node
    config = RunnableConfig()
    
    try:
        # Run the region selector node
        print("Searching for major coffee production regions...")
        result = await region_selector(state, config)
        
        if "error" in result:
            print(f"❌ Error occurred: {result['error']}")
            return
        
        # Display results
        selected_regions = result.get("selected_regions", [])
        messages = result.get("messages", [])
        
        print(f"\n✅ Successfully found {len(selected_regions)} coffee production regions:")
        print("-" * 60)
        
        for i, region in enumerate(selected_regions, 1):
            print(f"{i}. {region.region_name}")
            print(f"   Country: {region.country}")
            print(f"   Coordinates: {region.coordinates[0]:.4f}, {region.coordinates[1]:.4f}")
            print(f"   Type: {region.region_type}")
            print()
        
        if messages:
            print("📝 Generated message:")
            print(f"   {messages[0].content}")
        
        print("\n" + "=" * 60)
        print("Example completed successfully!")
        
    except Exception as e:
        print(f"❌ Example failed with exception: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

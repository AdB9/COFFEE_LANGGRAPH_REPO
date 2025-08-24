# Alpha Seeker

A multi-agent LangGraph system for discovering alpha indicators through automated research.

## 🎯 Project Vision

Alpha Seeker is designed to enhance time series forecasting models by automatically discovering new, high-impact features called "alpha indicators" for the commodity market. The system identifies periods of poor model performance and conducts automated research to understand potential causes.

## 🏗️ Architecture

The project follows a modular, agent-based architecture using LangGraph:

```
alpha_seeker/
├── common/                    # Shared data models and utilities
│   └── data_models.py        # Pydantic models for data structures
├── websearch_agent/          # Web search research agent
│   ├── nodes/               # Graph nodes (processing steps)
│   │   ├── query_generator.py    # Generates targeted search queries
│   │   ├── search_executor.py    # Executes web searches
│   │   └── result_analyzer.py    # Analyzes and synthesizes results
│   ├── tools/               # Agent tools
│   │   └── web_search.py         # Web search implementations
│   ├── configuration.py     # Agent configuration
│   ├── state.py            # Agent state management
│   └── graph.py            # Main agent graph definition
└── example_websearch.py    # Example usage script
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd /path/to/COFFEE_LANGGRAPH_REPO
   ```

2. **Create and activate virtual environment:**
   ```bash
   uv venv
   source .venv/bin/activate
   ```

3. **Install the project in editable mode:**
   ```bash
   uv pip install -e .
   ```

### Basic Usage

**Run the example websearch agent:**
```bash
python example_websearch.py
```

This will demonstrate the complete websearch workflow:
1. **Query Generation** - Converts user input into targeted search queries
2. **Search Execution** - Performs web searches using DuckDuckGo (default)
3. **Result Analysis** - Analyzes findings and generates insights

## 🔧 Configuration

### Environment Variables

Copy `env_template` to `.env` and configure:

```bash
cp env_template .env
```

Key configuration options:
- `GOOGLE_API_KEY` - For Google Search API (optional, falls back to DuckDuckGo)
- `DEFAULT_SEARCH_ENGINE` - "duckduckgo" or "google"
- `MAX_SEARCH_RESULTS` - Maximum results per query

### Agent Configuration

The websearch agent supports these configuration parameters:

```python
config = {
    "configurable": {
        "model": "google_genai:gemini-2.0-flash",
        "max_search_results": 8,
        "search_engine": "duckduckgo",
        "max_queries_per_search": 4,
        "result_analysis_depth": "detailed"  # "basic", "detailed", "comprehensive"
    }
}
```

## 📊 WebSearch Agent Features

### Query Generation
- Automatically generates 3-5 targeted search queries
- Adapts to different research domains (coffee markets, weather, general topics)
- Considers multiple angles: news, research, market data, expert analysis

### Search Execution
- Parallel execution of multiple queries
- Support for DuckDuckGo and Google Search APIs
- Automatic result classification by type (news, research, market reports, etc.)
- Relevance scoring based on keyword matching

### Result Analysis
- Three analysis depths: basic, detailed, comprehensive
- Source quality assessment
- Common theme extraction
- Actionable recommendations
- Comprehensive analysis summaries

### Result Types Supported
- 📰 News Articles
- 📑 Research Papers
- 📊 Market Reports
- 🏛️ Government Data
- 💬 Social Media
- 🌐 General Web Sources

## 🔍 Example Output

```
🎯 ALPHA SEEKER WEBSEARCH RESULTS
================================================================================

📊 SEARCH SUMMARY:
   • Total messages: 4
   • Research queries generated: 4
   • Total search results: 8
   • Key findings: 5

🔍 RESEARCH QUERIES:
   1. coffee market trends analysis 2024
   2. coffee price volatility factors
   3. global coffee production outlook
   4. coffee industry news recent developments

📰 TOP SEARCH RESULTS:
   1. Coffee Market Analysis Report 2024
      └─ Type: market_report | Relevance: 0.85
      └─ URL: https://example.com/coffee-report
      └─ Snippet: Comprehensive analysis of global coffee market trends...

🎯 KEY FINDINGS:
   1. Searched across 8 sources focused on Coffee market and industry analysis
   2. High quality: majority of sources are authoritative
   3. Recent coverage shows increased volatility due to weather patterns
   4. Market reports indicate supply chain concerns in Brazil
   5. Expert analysis suggests diversification strategies
```

## 🛠️ Development

### Project Structure
This project follows the same patterns as the `lang_graph` reference implementation:

- **Modular Design** - Each agent is self-contained
- **StateGraph Architecture** - Uses LangGraph for workflow orchestration
- **Configuration Management** - Flexible, environment-based configuration
- **Tool-based Architecture** - Reusable tools across different agents

### Adding New Agents
To add new agents (geospatial, social media, etc.), follow the websearch agent pattern:

1. Create agent directory under `alpha_seeker/`
2. Implement `state.py`, `configuration.py`, `graph.py`
3. Add nodes in `nodes/` directory
4. Add tools in `tools/` directory
5. Update `pyproject.toml` with any new dependencies

### Testing
Run the example scripts to test functionality:
```bash
python example_websearch.py
```

## 🔮 Future Roadmap

Based on the Requirements.md, the next agents to implement are:

1. **Geospatial Agent** - Satellite imagery analysis for crop conditions
2. **Social Media Agent** - Sentiment analysis and trending topic detection
3. **Orchestrator Agent** - Coordinates multiple agents for comprehensive research
4. **Alpha Proposer Agent** - Generates alpha indicators from research findings

## 📝 Dependencies

Key dependencies:
- `langgraph` - Graph-based agent orchestration
- `langchain` - LLM integration and tools
- `duckduckgo-search` - Web search functionality
- `pydantic` - Data validation and modeling
- `pandas` - Data analysis
- `python-dotenv` - Environment configuration

## 🤝 Contributing

Follow the existing patterns:
1. Maintain the modular architecture
2. Use Pydantic models for data structures
3. Follow the StateGraph pattern for agent workflows
4. Add comprehensive logging and error handling
5. Include example usage scripts

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

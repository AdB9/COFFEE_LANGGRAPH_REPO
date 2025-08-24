# Coffee LangGraph – Multi-Agent Signals → Bullish/Bearish

This repository scaffolds a **LangGraph** project for the **Coffee** commodity with this flow:

```
[input node]
      ├──▶ [Geospatial Agent] ─┐
      ├──▶ [Web/News Agent] ───┼──▶ [Data Analyst Agent] ───▶ [Bullish / Bearish / Neutral]
      └──▶ [Logistics Agent] ──┘
                      ↑
        (+ Predictive Price input)
```

- The **Input node** receives the user statement/context.
- Three agents run in **parallel**:
  - **Geospatial Agent** (weather/frost/drought/NDVI heuristics)
  - **Web/News Agent** (news/social volume proxy heuristics)
  - **Logistics Agent** (ports, freight, congestion heuristics)
- Their signals feed the **Data Analyst Agent**, which **also** ingests a **predictive price** (file or CLI input), and outputs **Bullish/Bearish/Neutral** with a rationale.

> Works **without API keys** using simple heuristics and sample CSVs. You can plug LLMs later.

---

## Quickstart

### Using uv (Recommended)

```bash
# 1) Install with uv (fastest Python package manager)
uv sync

# 2) Run the graph on sample data (period = 2 days)
uv run coffee-langgraph --period-days 2

# Optional: provide your own predictive return (percentage, e.g., +1.5 means +1.5% over horizon)
uv run coffee-langgraph --period-days 2 --predicted-return 1.5

# Optional: run with your own CSVs
uv run coffee-langgraph \
    --weather-csv data/sample/weather_sample.csv \
    --news-csv data/sample/news_sample.csv \
    --logistics-csv data/sample/logistics_sample.csv \
    --forecast-csv data/sample/price_forecast_sample.csv \
    --last-price-csv data/sample/last_price.csv
```

### Using pip (Alternative)

```bash
# 1) Create venv and install deps
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 2) Run the graph on sample data (period = 2 days)
coffee-langgraph --period-days 2

# Or run directly with Python module
python -m coffee_langgraph.main --period-days 2
```

Outputs are saved under `outputs/` as JSON (full state dump) and printed to console.

---

## Repository layout

```
coffee_langgraph/
  __init__.py            # Package initialization
  main.py                # CLI entrypoint
  graph_builder.py       # LangGraph wiring
  state.py               # Typed state definitions
  config.py              # Weights & thresholds
  utils/
    scoring.py           # Combine signals into final stance
    io.py                # CSV helpers & validation
  agents/
    geospatial_agent.py  # Heuristic geospatial signal
    web_news_agent.py    # Heuristic web/news signal
    logistics_agent.py   # Heuristic logistics signal
    data_analyst_agent.py# Aggregator (uses predictive price too)
data/sample/             # Sample CSVs you can replace with your own
outputs/                 # Run artifacts
pyproject.toml           # Modern Python packaging & dependencies
```

---

## Development Setup

```bash
# Install with development dependencies
uv sync --extra dev

# Run formatting and linting
uv run black coffee_langgraph/
uv run isort coffee_langgraph/
uv run flake8 coffee_langgraph/
uv run mypy coffee_langgraph/

# Run tests (if any)
uv run pytest
```

## Customizing for LLMs

This skeleton is **LLM-free by default** (for portability). If you want to prompt an LLM inside any agent:
- Add your provider to `pyproject.toml` dependencies (e.g., `langchain-openai`, `langchain-anthropic`, `boto3` for Bedrock).
- Create a simple wrapper in `coffee_langgraph/llm/provider.py` and call it from the agent file(s).
- Use `LANGGRAPH_CHECKPOINT` to persist between steps if you add `interrupt()` for human-in-the-loop.

---

## Notes

- **Not investment advice.** This is a **hackathon scaffold** with toy heuristics, meant to be extended.
- The **parallelism** is modeled with LangGraph branches; the `Data Analyst` node aggregates once all three signals exist.
- Tweak **weights/thresholds** in `src/config.py` to calibrate behavior quickly.

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

```bash
# 1) Create venv and install deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2) Run the graph on sample data (period = 2 days)
python -m src.main --period-days 2

# Optional: provide your own predictive return (percentage, e.g., +1.5 means +1.5% over horizon)
python -m src.main --period-days 2 --predicted-return 1.5

# Optional: run with your own CSVs
python -m src.main --weather-csv data/sample/weather_sample.csv                    --news-csv data/sample/news_sample.csv                    --logistics-csv data/sample/logistics_sample.csv                    --forecast-csv data/sample/price_forecast_sample.csv                    --last-price-csv data/sample/last_price.csv
```

Outputs are saved under `outputs/` as JSON (full state dump) and printed to console.

---

## Repository layout

```
src/
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
```

---

## Customizing for LLMs

This skeleton is **LLM-free by default** (for portability). If you want to prompt an LLM inside any agent:
- Add your provider to `requirements.txt` (e.g., `langchain-openai`, `langchain-anthropic`, `boto3` for Bedrock).
- Create a simple wrapper in `src/llm/provider.py` and call it from the agent file(s).
- Use `LANGGRAPH_CHECKPOINT` to persist between steps if you add `interrupt()` for human-in-the-loop.

---

## Notes

- **Not investment advice.** This is a **hackathon scaffold** with toy heuristics, meant to be extended.
- The **parallelism** is modeled with LangGraph branches; the `Data Analyst` node aggregates once all three signals exist.
- Tweak **weights/thresholds** in `src/config.py` to calibrate behavior quickly.

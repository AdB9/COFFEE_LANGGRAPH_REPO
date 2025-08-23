from __future__ import annotations
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from .state import GraphState
from .agents.geospatial_agent import geospatial_agent
from .agents.web_news_agent import web_news_agent
from .agents.logistics_agent import logistics_agent
from .agents.data_analyst_agent import data_analyst_agent

def input_node(state: GraphState) -> Dict[str, Any]:
    # Pass through; could normalize inputs here
    return {}

def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("input", input_node)
    graph.add_node("geospatial", geospatial_agent)
    graph.add_node("web_news", web_news_agent)
    graph.add_node("logistics", logistics_agent)
    graph.add_node("data_analyst", data_analyst_agent)

    graph.set_entry_point("input")

    # Parallel fan-out from input
    graph.add_edge("input", "geospatial")
    graph.add_edge("input", "web_news")
    graph.add_edge("input", "logistics")

    # Fan-in to aggregator
    graph.add_edge("geospatial", "data_analyst")
    graph.add_edge("web_news", "data_analyst")
    graph.add_edge("logistics", "data_analyst")

    graph.add_edge("data_analyst", END)
    return graph

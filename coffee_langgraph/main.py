from __future__ import annotations
import json, time, os
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.json import JSON

from coffee_langgraph.state import GraphState
from coffee_langgraph.graph_builder import build_graph

app = typer.Typer(add_completion=False)
console = Console()

def _pretty_print_decision(result: dict):
    decision = result.get("decision", {})
    stance = decision.get("stance", "n/a")
    score = decision.get("score", "n/a")
    components = decision.get("components", {})
    weights = decision.get("weights", {})
    pred = decision.get("predictive_price", {})
    rationale = decision.get("rationale", "")

    table = Table(title="Final Decision", show_lines=True)
    table.add_column("Field", style="bold cyan")
    table.add_column("Value", overflow="fold")
    table.add_row("Stance", str(stance))
    table.add_row("Score", str(score))
    table.add_row("Components", json.dumps(components, indent=2))
    table.add_row("Weights", json.dumps(weights, indent=2))
    table.add_row("Predictive Price", json.dumps(pred, indent=2))
    table.add_row("Rationale", rationale or "-")
    console.print(table)

@app.command()
def main(
    period_days: int = typer.Option(2, help="Horizon (days) the analyst should consider."),
    user_request: str = typer.Option("Analyze coffee price direction for the next period.", help="Free-form instruction/context."),
    # CSV overrides
    weather_csv: str = typer.Option(None, help="Path to weather/geospatial CSV."),
    news_csv: str = typer.Option(None, help="Path to news/social CSV."),
    logistics_csv: str = typer.Option(None, help="Path to logistics CSV."),
    forecast_csv: str = typer.Option(None, help="Path to forecast CSV (columns: ds,yhat)."),
    last_price_csv: str = typer.Option(None, help="Path to last price CSV (columns: date,last_close)."),
    # Optional manual predictive return
    predicted_return: float = typer.Option(None, help="Manual predicted return in % over the horizon (e.g., 1.5)."),
    # Runtime options
    stream: bool = typer.Option(True, help="Stream node updates to the console."),
    save_json: bool = typer.Option(True, help="Save final state JSON into outputs/"),
):
    # Build initial state
    initial_state: GraphState = {
        "period_days": int(period_days),
        "user_request": user_request,
        "weather_csv": weather_csv,
        "news_csv": news_csv,
        "logistics_csv": logistics_csv,
        "forecast_csv": forecast_csv,
        "last_price_csv": last_price_csv,
    }
    if predicted_return is not None:
        initial_state["predictive_price"] = {
            "horizon_days": int(period_days),
            "predicted_return_pct": float(predicted_return),
            "source": "manual",
        }

    # Build & compile graph (no checkpointer)
    graph = build_graph()
    app_graph = graph.compile()

    # Save graph visualization (if possible)
    try:
        image_bytes = app_graph.get_graph().draw_mermaid_png()
        
        # Save the image to a file
        with open("graph.png", "wb") as f:
            f.write(image_bytes)
        
        console.print("[green]Graph visualization saved to graph.png[/green]")
    except Exception as e:
        console.print(f"[yellow]Could not generate or save graph image: {e}[/yellow]")

    console.print(Panel.fit("[bold]Coffee LangGraph[/bold]", title="Run", border_style="green"))

    # Execute
    if stream:
        console.print("[bold cyan]Streaming node updates...[/bold cyan]")
        last_value = None
        for event in app_graph.stream(initial_state, stream_mode="values"):
            for node_name, val in event.items():
                console.print(Panel.fit(JSON.from_data(val), title=f"Update from: {node_name}", border_style="cyan"))
                last_value = val
        result = last_value or {}
    else:
        result = app_graph.invoke(initial_state)

    console.print()
    _pretty_print_decision(result)

    if save_json:
        os.makedirs("outputs", exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = f"outputs/run_{ts}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        console.print(f"[green]Saved state to[/green] {out_path}")

    return 0

if __name__ == "__main__":
    app()
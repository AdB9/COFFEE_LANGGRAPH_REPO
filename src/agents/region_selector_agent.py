from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd
from src.state import GraphState


COFFEE_REGIONS = [
    # Brazil (Arabica)
    {"name": "Sul de Minas, Brazil", "lat": -21.5, "lon": -45.0, "type": "arabica"},
    {"name": "Cerrado Mineiro, Brazil", "lat": -18.7, "lon": -47.8, "type": "arabica"},
    {"name": "Mogiana, Brazil", "lat": -20.0, "lon": -47.0, "type": "arabica"},
    # Vietnam (Robusta)
    {"name": "Dak Lak, Vietnam", "lat": 12.67, "lon": 108.05, "type": "robusta"},
    {"name": "Gia Lai, Vietnam", "lat": 13.98, "lon": 108.00, "type": "robusta"},
    {"name": "Lam Dong, Vietnam", "lat": 11.94, "lon": 108.44, "type": "arabica/robusta"},
    # Colombia
    {"name": "Huila, Colombia", "lat": 2.78, "lon": -75.30, "type": "arabica"},
    {"name": "Antioquia, Colombia", "lat": 6.57, "lon": -75.00, "type": "arabica"},
    # Ethiopia
    {"name": "Oromia, Ethiopia", "lat": 7.5, "lon": 39.0, "type": "arabica"},
    # Indonesia
    {"name": "Aceh, Indonesia", "lat": 4.4, "lon": 96.8, "type": "arabica"},
]

def region_selector_agent(state: GraphState) -> Dict[str, Any]:
    period_days = int(state.get("period_days", 2))
    now = pd.Timestamp.utcnow()
    month = int(now.month)
    regions: List[dict] = []

    def add(names: List[str]):
        for n in names:
            m = next((r for r in COFFEE_REGIONS if r["name"].startswith(n)), None)
            if m and m not in regions:
                regions.append(m)

    # Heuristics by seasonality
    if month in (5, 6, 7, 8):  # May–Aug: Brazil frost risk
        add(["Sul de Minas, Brazil", "Cerrado Mineiro, Brazil", "Mogiana, Brazil"])
        add(["Aceh, Indonesia"])
    elif month in (12, 1, 2, 3, 4):  # Vietnam dry season/harvest stress
        add(["Dak Lak, Vietnam", "Gia Lai, Vietnam", "Lam Dong, Vietnam"])
        add(["Huila, Colombia"])
    else:  # shoulder: diversify globally
        add(["Sul de Minas, Brazil", "Dak Lak, Vietnam", "Huila, Colombia", "Oromia, Ethiopia"])

    return {"regions": regions[:5]}

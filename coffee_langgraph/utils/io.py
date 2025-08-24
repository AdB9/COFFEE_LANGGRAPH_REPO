from __future__ import annotations
import os
import pandas as pd
from typing import Optional

def read_csv_if_exists(path: Optional[str]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)

def latest_row(df: pd.DataFrame, date_col: str = None):
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        return df.sort_values(date_col).iloc[-1].to_dict()
    return df.iloc[-1].to_dict()

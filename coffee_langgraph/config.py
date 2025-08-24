from __future__ import annotations

# Simple weights for combining signals (sum to 1.0)
WEIGHTS = {
    "geospatial": 0.40,
    "web_news": 0.30,
    "logistics": 0.20,
    "predictive": 0.10,
}

# Thresholds for final stance
BULLISH_THRESHOLD = 0.20   # > +0.20 => bullish
BEARISH_THRESHOLD = -0.20  # < -0.20 => bearish

# Heuristic cutoffs for geospatial features
FROST_TEMP_C = 2.0  # <= 2C => frost risk
DRYNESS_MM_DEFICIT = 10.0  # precip deficit threshold over 14d

# News spike thresholds
NEWS_COUNT_SPIKE = 20
NEG_SENTIMENT = -0.1  # average sentiment below this considered negative

# Logistics thresholds
WAIT_DAYS_SPIKE = 3.0
FREIGHT_INDEX_SPIKE = 1.1  # baseline ~1.0

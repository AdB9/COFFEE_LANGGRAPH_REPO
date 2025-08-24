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

# src/config.py
import os

# Geospatial thresholds
FROST_TMIN_C = float(os.getenv("FROST_TMIN_C", 2.0))                 # °C
PRECIP_14D_MIN_MM = float(os.getenv("PRECIP_14D_MIN_MM", 25.0))      # mm over last 14 days
NDVI_ANOM_BULLISH_MAX = float(os.getenv("NDVI_ANOM_BULLISH_MAX", -0.10))  # avg anomaly <= this → stress
NDVI_ANOM_BEARISH_MIN = float(os.getenv("NDVI_ANOM_BEARISH_MIN", 0.08))   # avg anomaly >= this → lush

# Web/news thresholds
NEWS_HIGH_VOLUME = float(os.getenv("NEWS_HIGH_VOLUME", 100))
NEWS_TONE_POS = float(os.getenv("NEWS_TONE_POS", 0.10))
NEWS_TONE_NEG = float(os.getenv("NEWS_TONE_NEG", -0.10))

# Logistics thresholds
FREIGHT_INDEX_ELEVATED = float(os.getenv("FREIGHT_INDEX_ELEVATED", 1.25))
PORT_WAIT_DAYS_ELEVATED = float(os.getenv("PORT_WAIT_DAYS_ELEVATED", 3.5))

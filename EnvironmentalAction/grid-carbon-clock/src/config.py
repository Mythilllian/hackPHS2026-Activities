"""Shared paths and constants."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"

TZ = "America/New_York"
TARGET = "ci_g_kwh"  # carbon intensity, g CO2 per kWh


def hour_start_local(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Turn EIA's end-of-hour UTC stamps into the local hour each value actually covers.

    EIA labels each row with the END of the hour (05:00 UTC covers 04:00-05:00 UTC), so we
    subtract an hour before converting to Eastern time. Plots and the app use this convention.
    """
    return index.tz_convert(TZ) - pd.Timedelta(hours=1)

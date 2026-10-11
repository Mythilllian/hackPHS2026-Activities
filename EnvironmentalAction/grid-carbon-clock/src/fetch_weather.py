"""Open-Meteo weather for six PJM cities (history and forecast), requested in UTC and cached.

Weather data by Open-Meteo.com (CC BY 4.0).
"""
import time

import pandas as pd
import requests

from config import PROCESSED, RAW

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
VARS = ["temperature_2m", "wind_speed_100m", "shortwave_radiation", "cloud_cover"]
# name: (lat, lon). Temperature stands in for demand, radiation/cloud for solar, 100 m wind for wind power.
CITIES = {
    "phl": (39.95, -75.17),
    "newark": (40.74, -74.17),
    "dc": (38.90, -77.04),
    "pit": (40.44, -80.00),
    "col": (39.96, -83.00),
    "chi": (41.88, -87.63),
}


def _get(url, params, retries=4):
    for i in range(retries):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        time.sleep(5 * (i + 1))  # back off on 429 / 5xx
    r.raise_for_status()


def _to_df(js, city):
    h = js["hourly"]
    df = pd.DataFrame(h)
    df["time"] = pd.to_datetime(df["time"]).dt.tz_localize("UTC")
    return df.set_index("time").add_prefix(f"{city}_")


def fetch_history(start: str, end: str) -> pd.DataFrame:
    path = RAW / "weather_history.parquet"
    if path.exists():
        return pd.read_parquet(path)
    frames = []
    for city, (lat, lon) in CITIES.items():
        parts = []
        for y in range(pd.Timestamp(start).year, pd.Timestamp(end).year + 1):
            s = max(pd.Timestamp(start), pd.Timestamp(f"{y}-01-01")).date()
            e = min(pd.Timestamp(end), pd.Timestamp(f"{y}-12-31")).date()
            js = _get(ARCHIVE_URL, dict(latitude=lat, longitude=lon, start_date=s, end_date=e,
                                        hourly=",".join(VARS), timezone="UTC"))
            parts.append(_to_df(js, city))
            print(city, y, "ok")
            time.sleep(1)
        frames.append(pd.concat(parts))
    df = pd.concat(frames, axis=1)
    df.to_parquet(path)
    return df


def fetch_forecast(days: int = 4, past_days: int = 14) -> pd.DataFrame:
    """Recent observed + forecast weather for all cities (UTC). Never cached here; callers decide."""
    frames = []
    for city, (lat, lon) in CITIES.items():
        js = _get(FORECAST_URL, dict(latitude=lat, longitude=lon, hourly=",".join(VARS),
                                     forecast_days=days, past_days=past_days, timezone="UTC"))
        frames.append(_to_df(js, city))
    df = pd.concat(frames, axis=1)
    df.to_parquet(PROCESSED / "weather_recent.parquet")
    return df


if __name__ == "__main__":
    # The archive API lags about 5 days behind today.
    end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=6)).date().isoformat()
    h = fetch_history("2018-07-01", end)
    print(h.shape, h.index.min(), h.index.max(), "NaN%:", round(h.isna().mean().mean() * 100, 2))
    f = fetch_forecast()
    print(f.shape, f.index.min(), f.index.max())

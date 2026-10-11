"""Gives the app a carbon forecast for any date.

How a forecast is made:
  1. The model takes the calendar, weather in six cities, and the carbon intensity from 2, 3 and 7 days ago.
  2. For past dates, all of that exists, so one model call predicts every hour.
  3. For future dates, the 2-day-old values are not published yet, so we go one day at a time and use the
     model's own earlier predictions as the lags. Errors build up with distance (see horizon_eval.py).
  4. Weather comes from Open-Meteo's 16-day forecast, then from typical weather for that date in past years.
"""
from dataclasses import dataclass
from datetime import date as Date

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import OUTPUTS, PROCESSED, TARGET, TZ, hour_start_local
from features import add_features
from fetch_weather import CITIES, fetch_forecast

WEATHER_CACHE = PROCESSED / "weather_recent.parquet"
LAG_BUFFER = pd.Timedelta(days=8)   # history needed to build the 7-day lag
MAX_OUTLOOK_DAYS = 180              # how far ahead the app goes


@dataclass
class DayForecast:
    day: Date
    pred: np.ndarray      # 48 values: g CO2/kWh for each local hour of `day`, then of the next day
    actual: np.ndarray    # same layout; NaN where the grid data does not exist yet
    split: str            # was this day in the model's "train", "validation", "test" data, or "unseen"?
    kind: str             # "history", "forecast" (real weather forecast) or "outlook" (typical weather)
    horizon_days: int     # days after the latest published grid data (0 or less = already known)
    cached: bool          # True if the live weather call failed and saved weather was used


class CarbonService:
    def __init__(self, live: bool = True):
        self.booster = lgb.Booster(model_file=str(OUTPUTS / "model.txt"))
        self.feature_names = self.booster.feature_name()

        # Training table: carbon intensity and weather for 2018 to early October 2026.
        data = pd.read_parquet(PROCESSED / "dataset.parquet")
        self.history_weather = data[[c for c in data.columns if c.split("_", 1)[0] in CITIES]]
        end = data[TARGET].dropna().index.max()
        self.test_start, self.val_start = end - pd.Timedelta(days=30), end - pd.Timedelta(days=60)

        # All carbon intensity ever published, which runs a few days past the training table.
        grid = pd.read_parquet(PROCESSED / "grid_ci.parquet")
        self.actual = grid[TARGET].where(~grid["outlier"])
        self.last_known = self.actual.dropna().index.max()

        # Typical weather for each (month, day, hour), used when the real forecast runs out.
        w = self.history_weather
        self.typical_weather = w.groupby([w.index.month, w.index.day, w.index.hour]).mean()

        self.history = pd.DataFrame({"pred": self._predict_rows(data[TARGET], w, w.index), "actual": data[TARGET]})
        self.future = pd.DataFrame()   # filled in by refresh_live
        self.forecast_end, self.cached = None, False
        if live:
            self.refresh_live()
        self._build_table()

    # ---- the model ----------------------------------------------------------------------------
    def _predict_rows(self, ci: pd.Series, weather: pd.DataFrame, rows: pd.DatetimeIndex) -> pd.Series:
        """Predict the given hours. Lags are read from `ci`, so it must hold real or already-predicted values."""
        window = pd.date_range(rows[0] - LAG_BUFFER, rows[-1] + pd.Timedelta(days=1), freq="h", tz="UTC")
        df = pd.DataFrame({TARGET: ci.reindex(window)}, index=window).join(weather)
        X = add_features(df).loc[rows, self.feature_names]
        return pd.Series(self.booster.predict(X), index=rows)

    def predict_ahead(self, ci: pd.Series, weather: pd.DataFrame) -> pd.Series:
        """Predict every hour after the last known one, a day at a time, feeding predictions back in as lags."""
        known_end = ci.dropna().index.max()
        rows = weather.index[weather.index > known_end]
        filled = ci.reindex(pd.date_range(known_end - pd.Timedelta(days=10), rows.max(), freq="h", tz="UTC"))
        for i in range(0, len(rows), 24):
            day = rows[i:i + 24]
            filled.loc[day] = self._predict_rows(filled, weather, day)
        return filled.loc[rows]

    # ---- live weather -------------------------------------------------------------------------
    def refresh_live(self):
        """Predict from now to about six months out. Falls back to saved weather if offline."""
        try:
            weather = fetch_forecast(days=16, past_days=14)
        except Exception:
            if not WEATHER_CACHE.exists():
                return
            weather, self.cached = pd.read_parquet(WEATHER_CACHE), True
        self.forecast_end = weather.index.max()

        # Past the real forecast, fill in typical weather for that date.
        end = pd.Timestamp.now(tz="UTC").normalize() + pd.Timedelta(days=MAX_OUTLOOK_DAYS + 2)
        later = pd.date_range(self.forecast_end + pd.Timedelta(hours=1), end, freq="h", tz="UTC")
        typical = self.typical_weather.reindex(list(zip(later.month, later.day, later.hour)))
        typical.index = later
        weather = pd.concat([weather, typical])

        recent = weather.index[weather.index <= self.last_known]   # lags exist: predict directly
        pred = pd.concat([self._predict_rows(self.actual, weather, recent), self.predict_ahead(self.actual, weather)])
        self.future = pd.DataFrame({"pred": pred, "actual": self.actual.reindex(pred.index)})

    def _build_table(self):
        """One table of predictions by local hour: history first, then whatever comes after it."""
        table = self.history
        if not self.future.empty:
            table = pd.concat([table, self.future[self.future.index > table.index.max()]])
        table.index = hour_start_local(table.index)
        self.table = table

    # ---- asking for one day ---------------------------------------------------------------------
    def available_range(self) -> tuple[Date, Date]:
        """First and last date with a (nearly) full day of predictions."""
        hours_per_day = pd.Series(1, index=self.table.index).groupby(self.table.index.date).count()
        full = hours_per_day[hours_per_day >= 23].index
        return full.min(), full.max()

    @staticmethod
    def _day_vector(column: pd.Series, day: Date) -> np.ndarray:
        """24 values by wall-clock hour. Daylight-saving days have 23 or 25 hours, so average duplicates and fill one gap."""
        hours = column[column.index.date == day]
        by_hour = hours.groupby(hours.index.hour).mean().reindex(range(24))
        return by_hour.interpolate(limit=1, limit_area="inside").to_numpy()

    def day(self, day: Date) -> DayForecast:
        first, last = self.available_range()
        if not first <= day <= last:
            raise ValueError(f"Pick a date between {first} and {last}.")
        tomorrow = day + pd.Timedelta(days=1)
        pred = np.r_[self._day_vector(self.table["pred"], day), self._day_vector(self.table["pred"], tomorrow)]
        actual = np.r_[self._day_vector(self.table["actual"], day), self._day_vector(self.table["actual"], tomorrow)]

        noon = pd.Timestamp(day, tz=TZ) + pd.Timedelta(hours=12)
        in_history = noon <= self.history.index.max()
        if noon <= self.val_start:
            split = "train"
        elif noon <= self.test_start:
            split = "validation"
        else:
            split = "test" if in_history else "unseen"
        if in_history:
            kind = "history"
        else:
            kind = "outlook" if noon > self.forecast_end else "forecast"
        horizon = (day - self.last_known.tz_convert(TZ).date()).days
        return DayForecast(day, pred, actual, split, kind, horizon, self.cached and not in_history)

    def recent_accuracy(self):
        """(hours, average error) on days after the training data ended whose real values have since been published."""
        if self.future.empty:
            return None
        recent = self.future[self.future.index > self.history.index.max()].dropna()
        return (len(recent), float((recent["pred"] - recent["actual"]).abs().mean())) if len(recent) else None

    # ---- the Explore tab ------------------------------------------------------------------------
    def actual_between(self, start: Date, end: Date, daytype: str = "All days") -> pd.Series:
        """Real carbon intensity between two dates, indexed by local hour, optionally weekdays or weekends only."""
        a = self.actual.dropna()
        s = pd.Series(a.to_numpy(), index=hour_start_local(a.index))
        s = s[(s.index.date >= start) & (s.index.date <= end)]
        if daytype == "Weekdays":
            s = s[s.index.dayofweek < 5]
        elif daytype == "Weekends":
            s = s[s.index.dayofweek >= 5]
        return s

    def typical_day(self, start: Date, end: Date, daytype: str = "All days") -> pd.DataFrame:
        """Average carbon intensity by hour of day, with the 10th and 90th percentiles."""
        by_hour = self.actual_between(start, end, daytype).groupby(lambda t: t.hour)
        return pd.DataFrame({"mean": by_hour.mean(), "p10": by_hour.quantile(0.1), "p90": by_hour.quantile(0.9)})

    def daily_means(self, start: Date, end: Date) -> pd.Series:
        s = self.actual_between(start, end)
        return s.groupby(s.index.date).mean()

"""Feature engineering, the time-based train/val/test split, and the naive baselines."""
import holidays
import numpy as np
import pandas as pd

from config import PROCESSED, TARGET, TZ
from fetch_weather import CITIES as _CITY_COORDS, VARS as WEATHER_VARS

CITIES = list(_CITY_COORDS)
# Hard rule: every lag is at least 48 hours old. EIA publishes its CO2 estimates 1-2 days late,
# so at forecast time the model can never see the most recent 48 hours of carbon intensity.
LAGS_H = [48, 72, 168]
MIN_LAG_H = 48


def add_features(df: pd.DataFrame, with_lags: bool = True) -> pd.DataFrame:
    """Build the model inputs from a UTC-indexed frame of weather (plus TARGET when with_lags)."""
    out = pd.DataFrame(index=df.index)
    local = df.index.tz_convert(TZ)
    out["hour"] = local.hour
    out["dow"] = local.dayofweek
    out["month"] = local.month
    us_hol = holidays.US(years=range(local.year.min(), local.year.max() + 1))
    out["is_holiday"] = pd.Index(local.date).map(lambda d: d in us_hol).astype(int)
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    doy = local.dayofyear
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    for c in CITIES:
        for v in WEATHER_VARS:
            out[f"{c}_{v}"] = df[f"{c}_{v}"]
    # Within-day shape: each weather value minus that local day's mean, so the model can tell
    # "this hour is sunnier or windier than the rest of the day" directly.
    day = pd.Index(local.date)
    for c in CITIES:
        for v in WEATHER_VARS:
            col = df[f"{c}_{v}"]
            out[f"{c}_{v}_anom"] = col.values - col.groupby(day).transform("mean").values
    if with_lags:
        s = df[TARGET]
        for h in LAGS_H:
            assert h >= MIN_LAG_H
            out[f"ci_lag{h}h"] = s.shift(h)
        # Shape of the day 2 and 7 days ago: lagged value minus that lagged day's mean.
        # The day means are built from shifted (48h+ old) values, so the lag rule still holds.
        for h in (48, 168):
            lag = s.shift(h)
            out[f"ci_shape_lag{h}h"] = lag.values - lag.groupby(day).transform("mean").values
        # Mean of the 24-hour window that ends 48 hours ago.
        out["ci_roll24_lag48h"] = s.shift(MIN_LAG_H).rolling(24, min_periods=12).mean()
    return out


def make_splits(X: pd.DataFrame, y: pd.Series):
    """Split by time: train = everything before the last 60 days, validation = next 30, test = final 30."""
    end = y.index.max()
    test_start = end - pd.Timedelta(days=30)
    val_start = end - pd.Timedelta(days=60)
    tr = y.index <= val_start
    va = (y.index > val_start) & (y.index <= test_start)
    te = y.index > test_start
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


def mae(a, b): return float(np.mean(np.abs(a - b)))
def mape(a, b): return float(np.mean(np.abs((a - b) / a)) * 100)


if __name__ == "__main__":
    df = pd.read_parquet(PROCESSED / "dataset.parquet")
    X = add_features(df)
    y = df[TARGET]
    keep = y.notna()
    X, y = X[keep], y[keep]
    (Xtr, ytr), (Xva, yva), (Xte, yte) = make_splits(X, y)
    for n, a in [("train", ytr), ("val", yva), ("test", yte)]:
        print(f"{n}: {len(a)} rows  {a.index.min()} -> {a.index.max()}")
    print("n features:", X.shape[1])
    print(list(X.columns))
    print("feature NaN% (train):", Xtr.isna().mean().round(3)[lambda s: s > 0].to_dict())
    pd.concat([X, y], axis=1).to_parquet(PROCESSED / "features.parquet")
    # Baselines on the test set, only where the lagged value exists.
    for name, col in [("same hour 2 days ago", "ci_lag48h"), ("same hour 7 days ago", "ci_lag168h")]:
        m = Xte[col].notna()
        print(f"baseline {name}: n={int(m.sum())}  MAE={mae(yte[m], Xte.loc[m, col]):.2f} g/kWh  "
              f"MAPE={mape(yte[m], Xte.loc[m, col]):.2f}%")

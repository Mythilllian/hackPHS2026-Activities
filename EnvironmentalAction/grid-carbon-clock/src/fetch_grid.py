"""Load the EIA Form 930 PJM workbook and compute hourly carbon intensity (g CO2/kWh).

Units (from the workbook's 'Notes' sheet):
  CO2 Emissions Generated  -> metric tons per hour
  Generation (MWh per hour) -> 'Positive Generation' (sum of positive fuel-type generation)
  tons/MWh * 1000 = g/kWh
'UTC time' is the END of the hour. We keep it as timezone-aware UTC.
"""
import pandas as pd

from config import PROCESSED, RAW

RAW_XLSX = RAW / "PJM.xlsx"
RAW_CACHE = RAW / "pjm_hourly_raw.parquet"
OUT = PROCESSED / "grid_ci.parquet"

FUEL_COLS = ["NG: COL", "NG: NG", "NG: NUC", "NG: OIL", "NG: GEO", "NG: WAT", "NG: PS",
             "NG: SUN", "NG: SNB", "NG: WND", "NG: WNB", "NG: BAT", "NG: OES", "NG: UES", "NG: OTH", "NG: UNK"]


def load_raw() -> pd.DataFrame:
    if RAW_CACHE.exists():
        return pd.read_parquet(RAW_CACHE)
    df = pd.read_excel(RAW_XLSX, sheet_name="Published Hourly Data")
    df.to_parquet(RAW_CACHE)
    return df


def build() -> pd.DataFrame:
    raw = load_raw()
    df = pd.DataFrame({
        "ts_utc": pd.to_datetime(raw["UTC time"]).dt.tz_localize("UTC"),
        "co2_t": raw["CO2 Emissions Generated"],
        "gen_mwh": raw["Positive Generation"],
        "demand_mwh": raw["Demand"],
    })
    for c in FUEL_COLS:
        df[c.replace("NG: ", "gen_")] = raw[c]
    df = df.dropna(subset=["co2_t", "gen_mwh"]).set_index("ts_utc").sort_index()
    df = df[~df.index.duplicated()]
    df["ci_g_kwh"] = df["co2_t"] / df["gen_mwh"] * 1000
    # Cross-check against EIA's own intensity column (lbs/kWh x 453.592 g/lb).
    eia = raw.set_index(pd.to_datetime(raw["UTC time"]).dt.tz_localize("UTC"))[
        "CO2 Emissions Intensity for Generated Electricity"]
    df["ci_eia_check"] = eia.reindex(df.index) * 453.592
    # Flag zero/negative generation and implausible intensities.
    df["outlier"] = (df["gen_mwh"] <= 0) | ~df["ci_g_kwh"].between(100, 1200)
    return df


if __name__ == "__main__":
    df = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT)
    print(df.shape, df.index.min(), df.index.max())
    print(df["ci_g_kwh"].describe())
    print("outliers:", int(df["outlier"].sum()))
    print("max |ci - eia| g/kWh:", (df["ci_g_kwh"] - df["ci_eia_check"]).abs().max())

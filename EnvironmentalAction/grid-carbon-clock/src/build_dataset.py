"""Join grid carbon intensity with weather on the UTC hour, fill short gaps, flag outliers."""
import pandas as pd

from config import PROCESSED, TARGET
from fetch_grid import build as build_grid
from fetch_weather import fetch_history

OUT = PROCESSED / "dataset.parquet"
MAX_GAP_H = 6  # interpolate only gaps up to 6 hours


def main():
    grid = build_grid()
    # outliers become NaN so they are treated like any other gap (the flag column stays).
    grid.loc[grid["outlier"], TARGET] = float("nan")
    full_idx = pd.date_range(grid.index.min(), grid.index.max(), freq="h", tz="UTC")
    grid = grid.reindex(full_idx)
    grid["outlier"] = grid["outlier"].fillna(False).astype(bool)
    n_missing_before = int(grid[TARGET].isna().sum())
    grid["ci_filled"] = grid[TARGET].isna()
    grid[TARGET] = grid[TARGET].interpolate(limit=MAX_GAP_H, limit_area="inside")
    grid["ci_filled"] &= grid[TARGET].notna()
    wx = fetch_history("2018-07-01", "2026-10-04")
    df = grid.join(wx, how="inner")
    n_total = len(df)
    n_missing_after = int(df[TARGET].isna().sum())
    df.to_parquet(OUT)
    print("rows (grid+weather overlap):", n_total)
    print("range:", df.index.min(), "->", df.index.max())
    print("grid hours missing before fill:", n_missing_before, "| interpolated:", int(df["ci_filled"].sum()),
          "| still NaN:", n_missing_after, f"({100*n_missing_after/n_total:.2f}%)")
    print("outliers flagged:", int(df["outlier"].sum()))
    print("weather NaN %:", round(wx.isna().mean().mean() * 100, 3))
    print(df[TARGET].describe().round(1))


if __name__ == "__main__":
    main()

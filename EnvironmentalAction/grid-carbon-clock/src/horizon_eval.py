"""How fast does accuracy fall with distance? Replays the recent past as if it were the future.

From each start date the model only sees grid data up to that date, then predicts 30 days ahead,
feeding on its own earlier predictions. Weather is the OBSERVED weather, so real forecasts will do
somewhat worse than this. The baseline is the average daily profile of the 28 days before the start.
Results go to outputs/horizon_accuracy.json and feed the disclaimers in the app.
"""
import json

import numpy as np
import pandas as pd

from config import OUTPUTS, PROCESSED, TARGET, hour_start_local
from service import CarbonService
from train import daily_windows

BUCKETS = [("1-2 days", 1, 2), ("3-7 days", 3, 7), ("8-14 days", 8, 14), ("15-30 days", 15, 30)]
HORIZON_DAYS = 30


def main():
    svc = CarbonService(live=False)
    data = pd.read_parquet(PROCESSED / "dataset.parquet")
    ci, weather = data[TARGET], svc.history_weather
    starts = pd.date_range(svc.val_start, svc.test_start, freq="3D")
    rows = []
    for s in starts:
        known = ci[ci.index <= s]
        wx = weather[weather.index <= s + pd.Timedelta(days=HORIZON_DAYS)]
        pred = svc.predict_ahead(known, wx)
        base_profile = known[known.index > s - pd.Timedelta(days=28)].groupby(lambda t: t.hour).mean()
        r = pd.DataFrame({"pred": pred, "base": base_profile.reindex(pred.index.hour).to_numpy(),
                          "actual": ci.reindex(pred.index)})
        r["day"] = ((r.index - known.dropna().index.max()) / pd.Timedelta(hours=1) - 1) // 24 + 1
        r["start"] = s
        rows.append(r.dropna())
    allr = pd.concat(rows)
    out = {"starts": len(starts), "horizon_days": HORIZON_DAYS, "buckets": {}}
    for name, lo, hi in BUCKETS:
        b = allr[(allr["day"] >= lo) & (allr["day"] <= hi)]
        hits = {"pred": [], "base": []}
        for _, g in b.groupby("start"):
            a = daily_windows(g["actual"])
            for k in hits:
                p = daily_windows(g[k])
                days = a.index.intersection(p.index)
                hits[k] += list(np.abs(a.loc[days].values.argmin(1) - p.loc[days].values.argmin(1)) <= 1)
        out["buckets"][name] = {
            "mae_model": float((b.pred - b.actual).abs().mean()),
            "mae_baseline": float((b.base - b.actual).abs().mean()),
            "within_1h_model": float(np.mean(hits["pred"])), "within_1h_baseline": float(np.mean(hits["base"])),
            "hours": int(len(b))}
    (OUTPUTS / "horizon_accuracy.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

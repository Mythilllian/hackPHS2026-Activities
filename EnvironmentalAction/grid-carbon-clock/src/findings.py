"""Numbers for the Findings page, all computed from the data (nothing typed in by hand)."""
from datetime import timedelta

import numpy as np
import pandas as pd

import impact
from config import PROCESSED, TZ, hour_start_local
from plots import best_window

FUELS = {"Natural gas": ["gen_NG"], "Coal": ["gen_COL"], "Nuclear": ["gen_NUC"], "Wind and solar": ["gen_WND", "gen_SUN"]}


def compute(service) -> dict:
    out = {}
    actual = service.actual.dropna()
    s = pd.Series(actual.to_numpy(), index=hour_start_local(actual.index))
    out.update(hours=len(s), mean=s.mean(), low=s.min(), high=s.max(), first=s.index.min().date(), last=s.index.max().date())

    # By year, and by season. Winter 2019 means December 2018 to February 2019.
    yearly = s.groupby(s.index.year).mean()
    summer = s[s.index.month.isin([7, 8])]
    summer = summer.groupby(summer.index.year).mean()
    winter = s[s.index.month.isin([12, 1, 2])]
    winter_year = winter.index.year + (winter.index.month == 12)
    full = winter.groupby(winter_year).count() > 2000          # skip winters we only partly have
    winter = winter.groupby(winter_year).mean()[full]
    out["years"] = [(y, yearly.get(y), summer.get(y), winter.get(y)) for y in range(s.index.min().year, s.index.max().year + 1)]

    # Daily pattern over the whole record and weekday vs weekend.
    typical = service.typical_day(out["first"], out["last"])["mean"].to_numpy()
    clean, dirty = best_window(typical), best_window(-typical)
    out.update(clean=clean, dirty=(dirty[0], -dirty[1]))
    last_year = service.typical_day(out["last"] - timedelta(days=365), out["last"])["mean"].to_numpy()
    cy, dy = best_window(last_year), best_window(-last_year)
    out.update(clean_year=cy, dirty_year=(dy[0], -dy[1]))
    out["weekday"] = service.actual_between(out["first"], out["last"], "Weekdays").mean()
    out["weekend"] = service.actual_between(out["first"], out["last"], "Weekends").mean()

    # What the grid is made of: first 12 months of data against the latest 12.
    grid = pd.read_parquet(PROCESSED / "grid_ci.parquet")
    def shares(frame):
        totals = {name: frame[cols].clip(lower=0).sum().sum() for name, cols in FUELS.items()}
        everything = frame[[c for c in frame.columns if c.startswith("gen_") and c != "gen_mwh"]].clip(lower=0).sum().sum()
        return {name: t / everything for name, t in totals.items()}
    out["mix_then"], out["mix_now"] = shares(grid.iloc[:8760]), shares(grid.iloc[-8760:])

    # What each appliance would have saved per run over the test month, using the real grid in hindsight.
    first = service.test_start.tz_convert(TZ).date() + timedelta(days=1)
    last = service.history.index.max().tz_convert(TZ).date() - timedelta(days=1)
    days = [service.day(first + timedelta(days=i)) for i in range((last - first).days + 1)]
    out["appliances"], out["test_days"] = [], len(days)
    for a in impact.APPLIANCES.values():
        results = []
        for d in days:
            plan = impact.plan(d.pred, a)
            score = impact.hindsight(d.actual, plan) if plan else None
            if score:
                results.append(score)
        got, best = np.array(results).T
        out["appliances"].append(dict(name=a.name, kwh=a.kwh, followed=got.mean(), perfect=best.mean(),
                                      days_helped=float((got > 0).mean()), n=len(results)))
    return out

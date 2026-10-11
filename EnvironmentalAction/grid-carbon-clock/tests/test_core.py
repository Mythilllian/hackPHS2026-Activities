"""Fast checks on the rules the project depends on. Run with:  .venv/bin/python -m pytest"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import features
import impact


def _frame(n=24 * 30):
    idx = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    df = pd.DataFrame({features.TARGET: np.arange(n, dtype=float)}, index=idx)
    for c in features.CITIES:
        for v in features.WEATHER_VARS:
            df[f"{c}_{v}"] = 1.0
    return df


def test_lags_are_never_newer_than_48_hours():
    df = _frame()
    X = features.add_features(df)
    # The target equals its row number, so a lag of h hours shows up as row - h.
    for col in [c for c in X.columns if c.startswith("ci_lag")]:
        h = int(col.removeprefix("ci_lag").removesuffix("h"))
        assert h >= 48
        assert (X[col].dropna() == (np.arange(len(df)) - h)[X[col].notna()]).all()
    assert X["ci_roll24_lag48h"].dropna().iloc[0] <= np.arange(len(df))[-1] - 48


def test_changing_the_last_two_days_does_not_change_the_features_for_them():
    """Leakage check: features for the final 47 hours must not depend on the target's last 47 hours."""
    df = _frame()
    changed = df.copy()
    changed.iloc[-47:, changed.columns.get_loc(features.TARGET)] = 1e6
    a, b = features.add_features(df).iloc[-47:], features.add_features(changed).iloc[-47:]
    pd.testing.assert_frame_equal(a, b)


def test_split_is_chronological_and_has_no_overlap():
    idx = pd.date_range("2026-01-01", periods=24 * 200, freq="h", tz="UTC")
    y = pd.Series(1.0, index=idx)
    (_, ytr), (_, yva), (_, yte) = features.make_splits(pd.DataFrame(index=idx), y)
    assert ytr.index.max() < yva.index.min() <= yva.index.max() < yte.index.min()
    assert len(yva) == len(yte) == 30 * 24


def test_each_appliance_picks_its_own_window():
    # A grid that is dirtiest at 6 PM, cleanest at 1 PM and 3 AM next day.
    vals = np.full(48, 340.0)
    vals[13:16] = 300.0
    vals[27:31] = 305.0
    vals[18:21] = 380.0
    plans = {name: impact.plan(vals, a) for name, a in impact.APPLIANCES.items()}
    assert plans["Laundry (washer + dryer)"].best_start == 13   # daytime-only appliance
    assert plans["EV charging"].best_start >= 24                # overnight appliance
    assert len({p.saved_kg for p in plans.values()}) == len(plans)


def test_plan_respects_the_allowed_hours_and_missing_data():
    a = impact.APPLIANCES["EV charging"]
    vals = np.r_[np.full(24, 330.0), np.full(24, np.nan)]
    p = impact.plan(vals, a)
    assert p is not None and p.best_start + a.duration_h <= 24     # next day unknown, so stay inside today
    assert impact.plan(np.full(10, 300.0), a) is None              # nothing fits


def test_no_savings_reported_when_waiting_is_worse():
    vals = np.full(48, 300.0)
    p = impact.plan(vals, impact.APPLIANCES["Dishwasher"])
    assert p.saved_kg == 0.0

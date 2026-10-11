"""When should an appliance run, and how much CO2 does waiting save?

Each appliance has a run time, a window when it is allowed to run, and a time people usually start it.
We compare the best start inside the window with the usual start. The numbers below are rough
assumptions, not measurements: edit them in the app or here.

Hours count from midnight of the chosen day, so 24 is midnight and 31 is 7 AM the next morning.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Appliance:
    name: str
    kwh: float            # energy per run
    duration_h: int       # hours one run takes (power is assumed constant)
    earliest_start: int   # first hour a run may begin
    finish_by: int        # hour the run must be done by
    usual_start: int      # when people normally start it


APPLIANCES = {a.name: a for a in [
    Appliance("EV charging", kwh=30.0, duration_h=4, earliest_start=18, finish_by=31, usual_start=18),
    Appliance("Dishwasher", kwh=1.5, duration_h=2, earliest_start=19, finish_by=30, usual_start=20),
    Appliance("Laundry (washer + dryer)", kwh=3.0, duration_h=3, earliest_start=8, finish_by=22, usual_start=18),
    Appliance("Water heater", kwh=4.5, duration_h=2, earliest_start=0, finish_by=29, usual_start=6),
    Appliance("Custom", kwh=1.0, duration_h=1, earliest_start=0, finish_by=47, usual_start=0)
]}

NJ_EV_COUNT = 280150


@dataclass(frozen=True)
class Plan:
    appliance: Appliance
    best_start: int
    best_g_kwh: float               # average g CO2/kWh during the best run
    usual_g_kwh: float | None       # same for the usual start (None if it has no forecast)
    saved_kg: float | None          # CO2 saved per run by waiting


def run_averages(vals, hours):
    """Average carbon intensity of a run starting at each hour. NaN if any hour in the run is unknown."""
    return np.array([np.mean(vals[s:s + hours]) for s in range(len(vals) - hours + 1)])


def usable_starts(a: Appliance, averages):
    """Start hours inside the appliance's window that we have data for."""
    last_start = min(a.finish_by, len(averages) + a.duration_h - 1) - a.duration_h
    return [s for s in range(a.earliest_start, last_start + 1) if not np.isnan(averages[s])]


def plan(vals, a: Appliance) -> Plan | None:
    """Best start for appliance `a` given hourly g/kWh from midnight (24 or 48 values), or None if nothing fits."""
    averages = run_averages(vals, a.duration_h)
    starts = usable_starts(a, averages)
    if not starts:
        return None
    best = min(starts, key=lambda s: averages[s])
    if a.usual_start not in starts:
        return Plan(a, best, float(averages[best]), None, None)
    usual = float(averages[a.usual_start])
    return Plan(a, best, float(averages[best]), usual, max(usual - averages[best], 0.0) * a.kwh / 1000)


def hindsight(actual_vals, p: Plan):
    """Score a plan against what the grid really did.

    Returns (kg saved by following the plan, kg a perfect pick would have saved), or None if unknown.
    """
    a = p.appliance
    averages = run_averages(actual_vals, a.duration_h)
    starts = usable_starts(a, averages)
    if p.usual_g_kwh is None or p.best_start not in starts or a.usual_start not in starts:
        return None
    usual = averages[a.usual_start]
    to_kg = a.kwh / 1000
    return (usual - averages[p.best_start]) * to_kg, (usual - min(averages[s] for s in starts)) * to_kg

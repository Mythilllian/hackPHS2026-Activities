"""The four charts the app draws, in one dark style. (Report figures live in report_figures.py.)"""
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch

from config import ROOT

# ---- colors ---------------------------------------------------------------------------------------
CARD, TEXT, MUTED, GRID = "#121c24", "#e6edf3", "#8b9bab", "#243341"
BLUE, TEAL, ORANGE = "#4cc2ff", "#2dd4bf", "#fb923c"
# Clean to dirty: teal, pale yellow, vermillion. Readable with common forms of color blindness.
CLEAN_TO_DIRTY = LinearSegmentedColormap.from_list("clean_dirty", ["#14b8a6", "#f4e8a3", "#e8590c"])

font_file = ROOT / "assets" / "Inter-Regular.otf"
if font_file.exists():
    font_manager.fontManager.addfont(str(font_file))
    plt.rcParams["font.family"] = "Inter"
plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13, "legend.fontsize": 11,
    "figure.facecolor": CARD, "axes.facecolor": CARD, "savefig.facecolor": CARD,
    "text.color": TEXT, "axes.labelcolor": TEXT, "xtick.color": TEXT, "ytick.color": TEXT,
    "axes.edgecolor": GRID, "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False,
})


def hour_label(h: int) -> str:
    """0 -> '12 AM', 13 -> '1 PM'. Hours past 23 wrap to the next morning."""
    h %= 24
    return f"{(h % 12) or 12} {'AM' if h < 12 else 'PM'}"


def best_window(values, width: int = 3):
    """Start hour and average of the cleanest `width`-hour stretch in 24 hourly values."""
    averages = [np.mean(values[i:i + width]) for i in range(24 - width + 1)]
    start = int(np.argmin(averages))
    return start, float(averages[start])


# ---- 1. the Carbon Clock ----------------------------------------------------------------------------
def carbon_clock(values, title: str, start_hour: int = 0, run=None):
    """24 wedges starting at the top, clockwise. Wedge length and color show g CO2/kWh.

    `values` are 24 hourly values beginning at `start_hour` (0 = midnight at the top).
    `run` = (first wedge, hours) outlines a run; without it we outline the cleanest 3 hours.
    """
    values = np.asarray(values, float)
    norm = Normalize(values.min() - 10, values.max() + 10)
    if run is None:
        first, hours = best_window(values)[0], 3
    else:
        first, hours = run
    average = float(np.mean(values[first:first + hours]))

    fig = plt.figure(figsize=(7.5, 7.5))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    width = 2 * np.pi / 24
    ax.bar(np.arange(24) * width + width / 2, values, width=width * 0.94,
           color=CLEAN_TO_DIRTY(norm(values)), edgecolor=CARD, linewidth=1)

    # Outline the run (or the cleanest 3 hours).
    top = values[first:first + hours].max() + 4
    arc = np.linspace(first * width, (first + hours) * width, 60)
    ax.plot(arc, np.full_like(arc, top), color=TEXT, lw=3.5)
    for edge in (first * width, (first + hours) * width):
        ax.plot([edge, edge], [0, top], color=TEXT, lw=3.5)

    ax.set_ylim(0, values.max() * 1.12)
    ax.set_xticks(np.arange(0, 2 * np.pi, np.pi / 4))
    ax.set_xticklabels([hour_label(start_hour + h) for h in range(0, 24, 3)])
    ax.set_yticks([100, 200, 300, 400])
    ax.set_yticklabels([])
    ax.grid(True)
    ax.spines["polar"].set_visible(False)
    what = "Cleanest 3 h" if run is None else f"Best {hours}-hour run"
    ax.set_title(f"{title}\n{what}: {hour_label(start_hour + first)} to {hour_label(start_hour + first + hours)} ({average:.0f} g CO₂/kWh)", pad=22)
    colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CLEAN_TO_DIRTY), ax=ax, shrink=0.55, pad=0.08)
    colorbar.set_label("g CO₂/kWh")
    colorbar.outline.set_visible(False)
    fig.tight_layout()
    return fig


# ---- 2. one appliance's run window ------------------------------------------------------------------
def plan_timeline(day, plan, title: str):
    """Forecast (and actual, if known) with the allowed, recommended and usual run windows shaded."""
    a = plan.appliance
    hours = min(max(24, a.finish_by) + 1, 48)
    x = np.arange(hours)
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.axvspan(a.earliest_start, min(a.finish_by, hours), color="#1b2a36", zorder=0)
    ax.step(x, day.pred[:hours], where="post", color=BLUE, lw=2.5, label="Forecast")
    if not np.isnan(day.actual[:hours]).all():
        ax.step(x, day.actual[:hours], where="post", color=TEXT, lw=1.6, ls=":", label="Actual")
    ax.axvspan(plan.best_start, plan.best_start + a.duration_h, color=TEAL, alpha=0.35)
    if plan.usual_g_kwh is not None:
        ax.axvspan(a.usual_start, a.usual_start + a.duration_h, facecolor="none", edgecolor=ORANGE, hatch="//", lw=1.5)
    ax.axvline(24, color=MUTED, lw=1, ls="--")  # midnight
    ax.set_xticks(range(0, hours, 3))
    ax.set_xticklabels([hour_label(h) for h in range(0, hours, 3)])
    ax.set_xlim(0, hours - 1)
    ax.set_ylabel("g CO₂/kWh")
    ax.set_title(title, loc="left")
    ax.grid(axis="y")
    handles, _ = ax.get_legend_handles_labels()
    handles += [Patch(facecolor="#1b2a36", label="Allowed to run"), Patch(facecolor=TEAL, alpha=0.35, label="Best start"),
                Patch(facecolor="none", edgecolor=ORANGE, hatch="//", label="Usual start")]
    ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout()
    return fig


# ---- 3 and 4. the Explore tab -----------------------------------------------------------------------
def profile_plot(profile: pd.DataFrame, title: str):
    """Average carbon intensity by hour of day, with a band for the 10th to 90th percentile."""
    start, _ = best_window(profile["mean"].to_numpy())
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.fill_between(profile.index, profile["p10"], profile["p90"], color=BLUE, alpha=0.18, label="10th to 90th percentile")
    ax.plot(profile.index, profile["mean"], color=BLUE, lw=2.6, label="Average")
    ax.axvspan(start - 0.5, start + 2.5, color=TEAL, alpha=0.3, label="Cleanest 3 h")
    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([hour_label(h) for h in range(0, 24, 3)])
    ax.set_ylabel("g CO₂/kWh")
    ax.set_title(title, loc="left")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout()
    return fig


def trend_plot(daily: pd.Series, title: str):
    """Daily average carbon intensity with a 30-day rolling average."""
    dates = pd.to_datetime(daily.index)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(dates, daily.to_numpy(), color="#4b5b6b", lw=0.8, label="Daily average")
    ax.plot(dates, daily.rolling(30, min_periods=10).mean().to_numpy(), color=BLUE, lw=2.4, label="30-day average")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    ax.set_ylabel("g CO₂/kWh")
    ax.set_title(title, loc="left")
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    return fig

"""Static figures for the README and slides, saved to outputs/. Run:  python report_figures.py"""
import json

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from config import OUTPUTS, PROCESSED, TARGET, TZ, hour_start_local
from plots import BLUE, CLEAN_TO_DIRTY, ORANGE, TEXT, carbon_clock

DPI = 200


def hour_by_day(series: pd.Series) -> pd.DataFrame:
    """Rows = local hour 0 to 23, columns = local date."""
    local = hour_start_local(series.index)
    table = pd.DataFrame({"v": series.to_numpy(), "date": local.normalize().tz_localize(None), "hour": local.hour})
    return table.pivot_table(index="hour", columns="date", values="v", aggfunc="mean")


def clock_example():
    """The Carbon Clock for the last full test day."""
    predictions = pd.read_parquet(PROCESSED / "test_predictions.parquet")["model"]
    days = hour_by_day(predictions).dropna(axis=1)
    last = days.columns[-1]
    carbon_clock(days[last].to_numpy(), f"Model forecast for {last:%a %b %d, %Y}").savefig(OUTPUTS / "01_carbon_clock.png", dpi=DPI)


def carbon_stripes():
    """Every day since 2018 (x) by hour of day (y), colored by carbon intensity."""
    grid = hour_by_day(pd.read_parquet(PROCESSED / "dataset.parquet")[TARGET])
    fig, ax = plt.subplots(figsize=(13, 5.5))
    image = ax.imshow(grid.to_numpy(), aspect="auto", cmap=CLEAN_TO_DIRTY, interpolation="nearest",
                      vmin=np.nanpercentile(grid.to_numpy(), 1), vmax=np.nanpercentile(grid.to_numpy(), 99),
                      extent=[0, grid.shape[1], 24, 0])
    first = grid.columns[0]
    years = list(range(first.year + 1, grid.columns[-1].year + 1))
    ax.set_xticks([(pd.Timestamp(f"{y}-01-01") - first).days for y in years])
    ax.set_xticklabels(years)
    ax.set_yticks([0, 6, 12, 18, 24])
    ax.set_yticklabels(["12 AM", "6 AM", "12 PM", "6 PM", "12 AM"])
    ax.set_ylabel("Hour of day (Eastern)")
    ax.set_title("PJM carbon stripes: every day since July 2018")
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, pad=0.01)
    colorbar.set_label("g CO₂/kWh")
    colorbar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "02_carbon_stripes.png", dpi=DPI)


def forecast_vs_actual():
    """Last 7 test days: actual, model and the 2-days-ago baseline."""
    p = pd.read_parquet(PROCESSED / "test_predictions.parquet").iloc[-7 * 24:]
    t = p.index.tz_convert(TZ)
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.plot(t, p["actual"], color=TEXT, lw=2.4, label="Actual")
    ax.plot(t, p["model"], color=BLUE, lw=2, label="LightGBM forecast")
    ax.plot(t, p["lag_2d"], color=ORANGE, lw=1.8, ls="--", label="Baseline: same hour 2 days ago")
    ax.set_ylabel("g CO₂/kWh")
    ax.set_title("Forecast vs. actual, last 7 test days", pad=40)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %b %d", tz=t.tz))
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)  # above the plot, clear of the dates
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUTPUTS / "03_forecast_vs_actual.png", dpi=DPI)


def shap_importance():
    """SHAP summary: which inputs push the forecast up or down."""
    import lightgbm as lgb
    import shap

    model = lgb.Booster(model_file=str(OUTPUTS / "model.txt"))
    X = pd.read_parquet(PROCESSED / "features.parquet").drop(columns=[TARGET]).iloc[-2000:][model.feature_name()]
    values = shap.TreeExplainer(model).shap_values(X)
    # Save the ranking so the website can quote it: average size of each input's effect, and which way it pushes.
    effect = np.abs(values).mean(axis=0)
    direction = [float(np.corrcoef(X[c].fillna(X[c].mean()), values[:, i])[0, 1]) for i, c in enumerate(X.columns)]
    top = np.argsort(effect)[::-1][:12]
    (OUTPUTS / "shap_summary.json").write_text(json.dumps(
        [{"feature": X.columns[i], "mean_abs_effect": float(effect[i]), "higher_value_pushes": "up" if direction[i] > 0 else "down",
          "correlation": direction[i]} for i in top], indent=2))
    plt.figure(figsize=(10, 7))
    shap.summary_plot(values, X, max_display=15, show=False, cmap=plt.get_cmap("coolwarm"))
    plt.title("What drives the forecast (SHAP, last 2,000 hours)", loc="left")
    plt.gca().tick_params(axis="both", labelcolor=TEXT)  # SHAP hardcodes dark gray labels
    plt.xlabel("SHAP value (effect on g CO₂/kWh)")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "04_feature_importance_shap.png", dpi=DPI)


def fuel_mix_gif(days: int = 7):
    """Animated stacked-area chart of one week of generation by fuel."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    grid = pd.read_parquet(PROCESSED / "grid_ci.parquet").iloc[-days * 24:]
    groups = {"Natural gas": ["gen_NG"], "Nuclear": ["gen_NUC"], "Coal": ["gen_COL"],
              "Wind + solar": ["gen_WND", "gen_SUN"], "Hydro + other": ["gen_WAT", "gen_OIL", "gen_OTH", "gen_PS", "gen_BAT"]}
    fuel = pd.DataFrame({name: grid[cols].clip(lower=0).sum(axis=1) / 1000 for name, cols in groups.items()})
    colors = ["#e69f00", "#cc79a7", "#6b7a89", "#009e73", "#56b4e9"]
    t = fuel.index.tz_convert(TZ)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, fuel.sum(axis=1).max() * 1.1)
    ax.set_ylabel("Generation (GW)")
    ax.set_title("PJM fuel mix over one week", pad=40)

    def draw(i):
        for area in list(ax.collections):
            area.remove()
        ax.stackplot(t[:i + 1], fuel.iloc[:i + 1].T.values, colors=colors, labels=fuel.columns)
        if i == 0:
            ax.legend(loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0))

    fig.autofmt_xdate()
    fig.tight_layout()
    FuncAnimation(fig, draw, frames=range(0, len(fuel), 3), interval=60).save(OUTPUTS / "05_fuel_mix.gif", writer=PillowWriter(fps=16), dpi=100)


if __name__ == "__main__":
    for make in (clock_example, carbon_stripes, forecast_vs_actual, shap_importance, fuel_mix_gif):
        make()
        plt.close("all")
        print("saved", make.__name__)

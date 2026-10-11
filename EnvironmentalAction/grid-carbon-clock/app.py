"""Grid Carbon Clock: pick a day and an appliance, see the cleanest time to run it.

Run with:  .venv/bin/python app.py
This file only wires the page together. The forecast is in src/service.py, the appliance math in
src/impact.py, the charts in src/plots.py and the look in src/ui.py.
"""
import random
import sys
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
import numpy as np
import pandas as pd

import findings
import impact
import pages
import ui
from config import OUTPUTS, TZ
from plots import best_window, carbon_clock, hour_label, plan_timeline, profile_plot, trend_plot
from service import CarbonService

service = CarbonService()
facts = findings.compute(service)
first_day, last_day = service.available_range()
today = pd.Timestamp.now(tz=TZ).date()
tomorrow = today + timedelta(days=1)


def to_date(value):
    """The date pickers hand back datetimes; we only want the date. Raises ValueError on half-typed text."""
    try:
        return pd.Timestamp(value).date()
    except (ValueError, TypeError):
        raise ValueError("Type a date like 2026-09-20.")


# ---- tab 1: plan a run ------------------------------------------------------------------------------
def plan_a_run(date_value, appliance, kwh, duration, earliest, finish_by, usual):
    """Returns (carbon clock, run-window chart, summary card) for one day and one appliance."""
    try:
        day = service.day(to_date(date_value))
    except ValueError as error:
        return gr.skip(), gr.skip(), ui.notice(str(error), "warning")

    a = replace(impact.APPLIANCES[appliance], kwh=kwh or impact.APPLIANCES[appliance].kwh, duration_h=int(duration),
                earliest_start=int(earliest), finish_by=int(finish_by), usual_start=int(usual))
    warning = ui.horizon_notice(day)
    plan = impact.plan(day.pred, a)

    if plan is None:
        return carbon_clock(day.pred[:24], f"{day.day:%a %b %d, %Y}"), None, warning + ui.notice("No run fits between those hours.")

    # The clock starts when the appliance can first run and covers the next 24 hours, so its outline matches the plan.
    # If those hours are not all known (the last forecast day), fall back to the plain midnight-to-midnight clock.
    start = a.earliest_start
    around = day.pred[start:start + 24]
    if len(around) == 24 and not np.isnan(around).any():
        run = (plan.best_start - start, a.duration_h) if plan.best_start + a.duration_h <= start + 24 else None
        title = f"{day.day:%a %b %d, %Y}" + (f", starting {hour_label(start)}" if start else "")
        clock = carbon_clock(around, title, start_hour=start, run=run)
    else:
        clock = carbon_clock(day.pred[:24], f"{day.day:%a %b %d, %Y}")
    timeline = plan_timeline(day, plan, f"{appliance}, {a.duration_h} h, {a.kwh:g} kWh")
    return clock, timeline, warning + ui.verdict_card(day, plan, impact.hindsight(day.actual, plan))


def appliance_defaults(name):
    """Picking an appliance fills in its usual numbers (which the user can then change)."""
    a = impact.APPLIANCES[name]
    return a.kwh, a.duration_h, a.earliest_start, a.finish_by, a.usual_start


def random_test_day():
    """A random day from the final 30 days, which the model never trained on."""
    first = service.test_start.tz_convert(TZ).date() + timedelta(days=1)
    last = service.history.index.max().tz_convert(TZ).date() - timedelta(days=1)
    return pd.Timestamp(first + timedelta(days=random.randrange((last - first).days + 1))).to_pydatetime()


# ---- tab 2: explore ------------------------------------------------------------------------------------
def explore(start_value, end_value, daytype):
    """Returns (typical-day chart, trend chart, summary) for a stretch of real data."""
    try:
        start, end = sorted([to_date(start_value), to_date(end_value)])
    except ValueError as error:
        return gr.skip(), gr.skip(), ui.notice(str(error), "warning")
    typical = service.typical_day(start, end, daytype)
    if typical.empty:
        return None, None, "No data in that range."
    n_days = service.actual_between(start, end, daytype).index.normalize().nunique()
    clean = best_window(typical["mean"].to_numpy())
    dirty = best_window(-typical["mean"].to_numpy())
    summary = ui.summary_card(n_days, typical["mean"].mean(), clean, (dirty[0], -dirty[1]))
    return (profile_plot(typical, f"Typical day, {start:%b %Y} to {end:%b %Y}"),
            trend_plot(service.daily_means(start, end), "Daily average"), summary)


# ---- the page --------------------------------------------------------------------------------------------
with gr.Blocks(title="Grid Carbon Clock", js=ui.FORCE_DARK_JS) as demo:
    gr.HTML(ui.hero(last_day))

    with gr.Tab("Plan a run"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                date = gr.DateTime(value=pd.Timestamp(tomorrow).to_pydatetime(), include_time=False, type="datetime",
                                   label="Date", info=f"{first_day:%b %d, %Y} to {last_day:%b %d, %Y}")
                appliance = gr.Dropdown(list(impact.APPLIANCES), value="EV charging", label="Appliance")
                with gr.Row():
                    tomorrow_button = gr.Button("Tomorrow", variant="primary")
                    random_button = gr.Button("Random test day")
                with gr.Accordion("Adjust the appliance", open=False):
                    kwh = gr.Number(value=30.0, label="Energy per run (kWh)")
                    duration = gr.Slider(1, 12, value=4, step=1, label="Run time (hours)")
                    earliest = gr.Slider(0, 23, value=18, step=1, label="Earliest start (hours after midnight)")
                    earliest_time = gr.Markdown()
                    finish_by = gr.Slider(1, 48, value=31, step=1, label="Finish by (hours after midnight)")
                    finish_time = gr.Markdown()
                    usual = gr.Slider(0, 23, value=18, step=1, label="Usual start (hours after midnight)")
                    usual_time = gr.Markdown()
            with gr.Column(scale=6):
                summary_card = gr.HTML()
        with gr.Row(equal_height=False):
            clock_plot = gr.Plot(show_label=False, scale=5)
            timeline_plot = gr.Plot(show_label=False, scale=6)

        inputs = [date, appliance, kwh, duration, earliest, finish_by, usual]
        outputs = [clock_plot, timeline_plot, summary_card]
        date.change(plan_a_run, inputs, outputs, show_progress="hidden", trigger_mode="always_last")
        for control in (kwh, duration, earliest, finish_by, usual):
            control.input(plan_a_run, inputs, outputs, show_progress="hidden", trigger_mode="always_last")
        # Picking an appliance rewrites five fields, so plan once afterwards (.input fires only on user edits).
        appliance.input(appliance_defaults, appliance, [kwh, duration, earliest, finish_by, usual]).then(plan_a_run, inputs, outputs)
        # Show each slider's value as a clock time. .change also fires when picking an appliance rewrites them.
        for slider, caption in ((earliest, earliest_time), (finish_by, finish_time), (usual, usual_time)):
            slider.change(ui.time_caption, slider, caption, show_progress="hidden")
        tomorrow_button.click(lambda: pd.Timestamp(tomorrow).to_pydatetime(), None, date)
        random_button.click(random_test_day, None, date)

    with gr.Tab("Explore"):
        with gr.Row():
            start = gr.DateTime(value=pd.Timestamp(today - timedelta(days=365)).to_pydatetime(), include_time=False, type="datetime", label="From")
            end = gr.DateTime(value=pd.Timestamp(today).to_pydatetime(), include_time=False, type="datetime", label="To")
            daytype = gr.Radio(["All days", "Weekdays", "Weekends"], value="All days", label="Days")
        explore_summary = gr.HTML()
        with gr.Row():
            typical_plot = gr.Plot(show_label=False)
            trend_plot_box = gr.Plot(show_label=False)
        explore_inputs, explore_outputs = [start, end, daytype], [typical_plot, trend_plot_box, explore_summary]
        start.change(explore, explore_inputs, explore_outputs, show_progress="hidden")
        end.change(explore, explore_inputs, explore_outputs, show_progress="hidden")
        daytype.input(explore, explore_inputs, explore_outputs, show_progress="hidden")
        demo.load(explore, explore_inputs, explore_outputs)

    with gr.Tab("Findings"):
        gr.HTML(pages.findings_html(facts))
        gr.Image(str(OUTPUTS / "02_carbon_stripes.png"), show_label=False, interactive=False)
        gr.HTML(pages.gif_html(OUTPUTS / "05_fuel_mix.gif"))
        with gr.Row():
            gr.Image(str(OUTPUTS / "04_feature_importance_shap.png"), show_label=False, interactive=False)
            gr.Image(str(OUTPUTS / "01_carbon_clock.png"), show_label=False, interactive=False)

    with gr.Tab("Models"):
        gr.HTML(pages.models_html(service.recent_accuracy()))
        gr.Image(str(OUTPUTS / "03_forecast_vs_actual.png"), show_label=False, interactive=False)

    with gr.Tab("Data"):
        gr.HTML(ui.DATA_TEXT)

    demo.load(plan_a_run, inputs, outputs)
    demo.load(lambda: [ui.time_caption(h) for h in (18, 31, 18)], None, [earliest_time, finish_time, usual_time])

if __name__ == "__main__":
    demo.launch(theme=ui.THEME, css=ui.page_css())

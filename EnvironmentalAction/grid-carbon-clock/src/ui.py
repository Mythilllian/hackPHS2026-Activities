"""Everything about how the page looks: theme, colors and the small HTML pieces."""
import base64
import json

import gradio as gr

from config import OUTPUTS, ROOT
from plots import hour_label

ACCURACY = json.loads((OUTPUTS / "horizon_accuracy.json").read_text())  # made by horizon_eval.py
METRICS = json.loads((OUTPUTS / "metrics.json").read_text())            # made by train.py

SPLIT_NOTE = {
    "train": "In the model's training data",
    "validation": "Used to tune the model",
    "test": "Held out: the model never saw this day",
    "unseen": "Newer than the training data",
}

# ---- theme: dark in both modes ----------------------------------------------------------------------
DARK = dict(
    body_background_fill="#0b1217", background_fill_primary="#121c24", background_fill_secondary="#0f1820",
    block_background_fill="#121c24", block_border_color="#1f2e39", block_label_background_fill="#121c24",
    block_label_text_color="#8b9bab", block_title_text_color="#e6edf3", body_text_color="#e6edf3",
    body_text_color_subdued="#8b9bab", input_background_fill="#0f1820", input_border_color="#243341",
    border_color_primary="#243341", panel_background_fill="#121c24", button_secondary_background_fill="#1b2a36",
    button_secondary_text_color="#e6edf3", button_primary_background_fill="#0f766e",
    button_primary_background_fill_hover="#14a398", button_primary_text_color="#ffffff", slider_color="#2dd4bf",
)
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.teal, neutral_hue=gr.themes.colors.slate, radius_size=gr.themes.sizes.radius_lg,
    font=[gr.themes.Font("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(**DARK, **{name + "_dark": value for name, value in DARK.items()}, block_shadow="none")

FORCE_DARK_JS = "() => { document.body.classList.add('dark'); }"


def page_css() -> str:
    """The Inter font (embedded, so it works offline) plus assets/style.css."""
    font = base64.b64encode((ROOT / "assets" / "Inter.woff2").read_bytes()).decode()
    face = f"@font-face {{ font-family: 'Inter'; font-weight: 100 900; src: url(data:font/woff2;base64,{font}) format('woff2'); }}\n"
    return face + (ROOT / "assets" / "style.css").read_text()


# ---- pieces of the page ---------------------------------------------------------------------------------
def hero(last_day) -> str:
    return f"""
<div class="hero">
  <h1>Grid Carbon Clock</h1>
  <p>Hourly carbon intensity of the PJM grid, which serves New Jersey.</p>
  <div class="chips">
    <span class="chip"><b>343</b> g CO₂/kWh average</span>
    <span class="chip"><b>208 to 490</b> range</span>
    <span class="chip"><b>July 2018</b> to {last_day:%b %Y}</span>
  </div>
</div>"""


def time_caption(hour) -> str:
    """Turn 'hours after midnight' into a readable time: 18 -> '6 PM', 31 -> '7 AM next day'."""
    hour = int(hour)
    if hour == 24:
        return "= midnight"
    if hour == 48:
        return "= midnight at the end of the next day"
    return f"= {hour_label(hour)}" + (" next day" if hour > 24 else "")


def notice(text: str, level: str = "caution") -> str:
    return f'<div class="notice {level}">{text}</div>'


def horizon_notice(day) -> str:
    """A warning for dates more than 2 days out, with the error measured by horizon_eval.py."""
    if day.kind == "history" or day.horizon_days <= 2:
        return ""
    days_out = day.horizon_days
    buckets = ACCURACY["buckets"]
    near = buckets["1-2 days"]["mae_model"]
    far = buckets["3-7 days" if days_out <= 7 else "8-14 days" if days_out <= 14 else "15-30 days"]
    hit = f"{far['within_1h_model']:.0%}"
    if day.kind == "outlook":
        return notice(
            f"<b>Seasonal outlook, {days_out} days ahead.</b> Past the 16-day weather forecast the model assumes typical "
            f"weather for the date, so this shows the usual shape of the day, not the real one. At 15 to 30 days the typical "
            f"error was {far['mae_model']:.0f} g/kWh (against {near:.0f} for the next two days), and the cleanest window landed "
            f"within an hour only {hit} of the time. Further out was not measured, and these tests used perfect weather.", "warning")
    return notice(
        f"<b>{days_out} days ahead.</b> Typical error at this range is {far['mae_model']:.0f} g/kWh (against {near:.0f} for the "
        f"next two days), and the cleanest window landed within an hour {hit} of the time. "
        f"Those tests used perfect weather, so real forecasts do worse.")


def verdict_card(day, plan, hindsight) -> str:
    """The big 'Best start' card."""
    a = plan.appliance
    when = hour_label(plan.best_start) + (" <small>next day</small>" if plan.best_start >= 24 else "")
    stats = f'<div class="stat"><b>{plan.best_g_kwh:.0f}</b><span>g/kWh during the run</span></div>'
    if plan.usual_g_kwh is not None:
        stats += (f'<div class="stat"><b>{plan.usual_g_kwh:.0f}</b><span>g/kWh if started at {hour_label(a.usual_start)}</span></div>'
                  f'<div class="stat good"><b>{plan.saved_kg:.2f} kg</b><span>CO₂ saved per run</span></div>')
    tags = f'<span class="tag">{SPLIT_NOTE[day.split]}</span>'
    if hindsight is not None:
        tags += f'<span class="tag">Actual grid: saved {hindsight[0]:.2f} kg, best possible {hindsight[1]:.2f} kg</span>'
    if day.cached:
        tags += '<span class="tag">Saved weather, no connection</span>'
    return (f'<div class="card"><div class="eyebrow">Best start</div><div class="big">{when}</div>'
            f'<div class="stats">{stats}</div>{tags}</div>')


def summary_card(n_days, average, clean, dirty) -> str:
    """One-paragraph summary for the Explore tab. clean and dirty are (start hour, average)."""
    return (f'<div class="card summary"><b>{n_days} days</b>, average <b>{average:.0f}</b> g CO₂/kWh. '
            f'Cleanest 3 hours: <b>{hour_label(clean[0])} to {hour_label(clean[0] + 3)}</b> ({clean[1]:.0f}). '
            f'Dirtiest: <b>{hour_label(dirty[0])} to {hour_label(dirty[0] + 3)}</b> ({dirty[1]:.0f}).</div>')


DATA_TEXT = """
<div class="card prose">
<b>Carbon intensity.</b> EIA Form 930 Hourly Electric Grid Monitor, PJM. Hourly CO₂ emissions divided by generation, in g CO₂/kWh.
July 2018 to October 2026: average 343, range 208 to 490. 72,427 hours joined with weather, 2.5% of them missing.
This is the grid average, not the marginal emissions of one extra kWh.<br><br>
<b>Weather.</b> Open-Meteo (CC BY 4.0): temperature, 100 m wind, solar radiation and cloud cover for Philadelphia, Newark,
Washington DC, Pittsburgh, Columbus and Chicago.<br><br>
<b>Credits.</b> EIA, Open-Meteo, LightGBM, SHAP, Gradio.
</div>"""


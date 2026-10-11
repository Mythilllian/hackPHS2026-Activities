"""The Findings and Models pages: HTML built from saved results (metrics, tuning, SHAP) and findings.py."""
import base64
import json

from config import OUTPUTS
from plots import hour_label
from ui import ACCURACY, METRICS

CITY = {"phl": "Philadelphia", "newark": "Newark", "dc": "Washington DC", "pit": "Pittsburgh", "col": "Columbus", "chi": "Chicago"}
WEATHER = {"temperature_2m": "temperature", "wind_speed_100m": "wind speed", "shortwave_radiation": "sunshine", "cloud_cover": "cloud cover"}
LAGS = {"ci_lag48h": "Carbon intensity 2 days earlier", "ci_lag72h": "Carbon intensity 3 days earlier",
        "ci_lag168h": "Carbon intensity 7 days earlier", "ci_roll24_lag48h": "Average of the 24 hours ending 2 days ago",
        "ci_shape_lag48h": "Shape of the day 2 days ago", "ci_shape_lag168h": "Shape of the day 7 days ago"}
CALENDAR = {"hour": "Hour of day", "dow": "Day of week", "month": "Month", "is_holiday": "US holiday", "is_weekend": "Weekend",
            "hour_sin": "Hour (sine)", "hour_cos": "Hour (cosine)", "doy_sin": "Day of year (sine)", "doy_cos": "Day of year (cosine)"}


def readable(name: str) -> str:
    if name in LAGS or name in CALENDAR:
        return LAGS.get(name) or CALENDAR[name]
    city, variable = name.split("_", 1)
    relative = variable.endswith("_anom")
    variable = variable.removesuffix("_anom")
    return f"{CITY[city]} {WEATHER[variable]}" + (" vs. the day's average" if relative else "")


def table(header, rows, highlight=None) -> str:
    head = "".join(f"<th>{h}</th>" for h in header)
    body = "".join(f"<tr{' class=\"pick\"' if i == highlight else ''}>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
                   for i, row in enumerate(rows))
    return f'<table class="table"><tr>{head}</tr>{body}</table>'


def card(title: str, body: str) -> str:
    return f'<div class="card prose gap"><div class="eyebrow">{title}</div>{body}</div>'


def input_list(names) -> str:
    """Every model input, grouped, in a collapsible block."""
    calendar = [CALENDAR[n] for n in names if n in CALENDAR]
    lags = [LAGS[n] for n in names if n in LAGS]
    weather = [n for n in names if n not in CALENDAR and n not in LAGS]
    cities = [CITY[c] for c in CITY if any(n.startswith(c + "_") for n in weather)]
    variables = [WEATHER[v] for v in WEATHER if any(n.split("_", 1)[1].removesuffix("_anom") == v for n in weather)]
    plain = [n for n in weather if not n.endswith("_anom")]
    return (f"<details><summary>See all {len(names)} inputs</summary>"
            f"<p><b>Calendar ({len(calendar)}):</b> {', '.join(calendar)}.</p>"
            f"<p><b>Weather ({len(plain)}):</b> {', '.join(variables)}, in each of {', '.join(cities)}.</p>"
            f"<p><b>Weather relative to its day's average ({len(weather) - len(plain)}):</b> the same {len(plain)} values, each minus that local day's average.</p>"
            f"<p><b>Carbon intensity from 2 to 7 days earlier ({len(lags)}):</b> {', '.join(lags)}.</p></details>")


def kpis(items) -> str:
    return '<div class="kpis">' + "".join(f'<div class="stat"><b>{big}</b><span>{small}</span></div>' for big, small in items) + "</div>"


def window(w) -> str:
    return f"{hour_label(w[0])} to {hour_label(w[0] + 3)} ({w[1]:.0f})"


def gif_html(path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/gif;base64,{data}" style="width:100%;border-radius:12px" alt="Fuel mix over one week">'


# ---- Findings ---------------------------------------------------------------------------------------------
def findings_html(f: dict) -> str:
    shap = json.loads((OUTPUTS / "shap_summary.json").read_text())
    dash = lambda v: "-" if v is None or v != v else f"{v:.0f}"

    top = kpis([(f"{f['mean']:.0f}", "g CO₂/kWh average"), (f"{f['low']:.0f} to {f['high']:.0f}", "range of hourly values"),
                (f"{f['hours']:,}", "hourly readings analyzed"), (f"{f['weekday'] - f['weekend']:.0f}", "g/kWh cleaner on weekends")])

    years = table(["Year", "Whole year", "July and August", "Winter (Dec to Feb)"],
                  [(y, dash(a), dash(b), dash(c)) for y, a, b, c in f["years"]])
    trend = card("The grid is getting cleaner", f"""
<p>Average g CO₂/kWh. 2018 starts in July and 2026 ends in early October. Winter 2019 means December 2018 to February 2019.</p>{years}""")

    mix = table(["Share of generation", f"First 8,760 hours (from {f['first']:%b %Y})", "Latest 8,760 hours"],
                [(k, f"{f['mix_then'][k]:.0%}", f"{f['mix_now'][k]:.0%}") for k in f["mix_then"]])
    fuel = card("What changed on the grid", f"<p>Shares of PJM's generation from the fuels that matter most for carbon intensity.</p>{mix}")

    pattern = card("The daily pattern has shifted", f"""
<p>Cleanest and dirtiest 3 hours of the day, with their average g CO₂/kWh.</p>
{table(["", "Cleanest 3 hours", "Dirtiest 3 hours"],
       [("Whole record", window(f["clean"]), window(f["dirty"])), ("Last 12 months", window(f["clean_year"]), window(f["dirty_year"]))])}
<p>Weekdays average {f['weekday']:.0f} g/kWh and weekends {f['weekend']:.0f}, so weekends are about {100 * (f['weekday'] - f['weekend']) / f['weekday']:.0f}% cleaner.</p>""")

    drivers = table(["Input", "Average effect (g/kWh)", "Higher value pushes the forecast"],
                    [(readable(s["feature"]), f"{s['mean_abs_effect']:.1f}", s["higher_value_pushes"]) for s in shap[:8]])
    why = card("What drives the forecast", f"""
<p>How much each input moves the model's prediction, averaged over the last 2,000 hours (SHAP values). Higher Chicago wind
and more sunshine in Columbus both push the forecast down.</p>{drivers}""")

    appliances = table(["Appliance", "Saved per run by following the forecast", "A perfect pick would save", "Days it helped"],
                       [(a["name"], f"{a['followed']:.2f} kg", f"{a['perfect']:.2f} kg", f"{a['days_helped']:.0%}") for a in f["appliances"]])
    saved = card("What each appliance would have saved", f"""
<p>Over the {f['test_days']} test days, starting at the forecast's best time instead of the appliance's usual time, judged against what the grid really did.
Default run times, energy use and usual start times are rough assumptions.</p>{appliances}""")

    return top + trend + fuel + pattern + why + saved


# ---- Models ---------------------------------------------------------------------------------------------------
def models_html(recent) -> str:
    m, d, p = METRICS, METRICS["data"], METRICS["params"]
    groups = {"Calendar": 0, "Weather in six cities": 0, "Weather relative to its day's average": 0, "Carbon intensity from 2 to 7 days earlier": 0}
    for name in d["features"]:
        key = ("Carbon intensity from 2 to 7 days earlier" if name in LAGS else "Calendar" if name in CALENDAR
               else "Weather relative to its day's average" if name.endswith("_anom") else "Weather in six cities")
        groups[key] += 1
    inputs = table(["Inputs", "Count"], [*groups.items(), ("Total", d["n_features"])])

    tuning = m["tuning"]
    best = min(range(len(tuning)), key=lambda i: tuning[i]["val_mae"])
    tuned = table(["Leaves", "Learning rate", "Recent years count more (half-life)", "Trees", "Validation error"],
                  [(t["num_leaves"], t["learning_rate"], f"{t['half_life_days']} days" if t["half_life_days"] else "no", t["trees"], f"{t['val_mae']:.2f}")
                   for t in tuning], highlight=best)

    main = card("Model 1: LightGBM forecast", f"""
<p>A few hundred small decision trees, each correcting the one before. It predicts g CO₂/kWh for every hour from {d['n_features']} inputs.
Trained on <b>{d['train_rows']:,} hours</b> (July 2018 to {d['train_end'][:10]}), tuned on the next {d['val_rows']} hours and tested on the final {d['test_rows']},
split by time and never shuffled. Chosen settings: {p['num_leaves']} leaves, learning rate {p['learning_rate']},
{p['best_iteration']} trees, and a {p['half_life_days']}-day half-life so recent years count more.
All lagged inputs are at least 48 hours old, because EIA publishes 1 to 2 days late.</p>{inputs}{input_list(d['features'])}""")
    tune = card("Tuning: all 12 settings tried", f"<p>Validation error is the average miss in g/kWh on the 30 days used for tuning. The chosen row is highlighted.</p>{tuned}")

    rows = [("LightGBM model", m["model"]), ("Baseline: same hour 2 days ago", m["baselines"]["lag_2d"]),
            ("Baseline: same hour 7 days ago", m["baselines"]["lag_7d"])]
    results = table(["", "Hourly error", "Error %", "Cleanest window within 1 h", "Exact", "CO₂ saved vs. perfect pick"],
                    [(n, f"{r['MAE']:.1f}", f"{r['MAPE']:.1f}%", f"{r['windows']['hit_rate_within_1h']:.0%}",
                      f"{r['windows']['hit_rate_exact']:.0%}", f"{r['windows']['co2_captured_pct_vs_avg_window']:.0f}%") for n, r in rows])
    w = m["model"]["windows"]
    start, end = (t[:10] for t in m["test_period_utc"])
    since = (f"<p>Since the training data ended, <b>{recent[0]} hours</b> of real grid data have been published. "
             f"The forecast for them was off by <b>{recent[1]:.1f} g/kWh</b> on average.</p>") if recent else ""
    test = card("Test results and baselines", f"""
<p>{start} to {end}: {m['n_test_rows']} hours, {w['n_days']} full days the model never trained on. The two baselines are the simplest forecasts:
repeat the same hour from 2 days or 7 days ago. Error is in g/kWh.</p>{results}
<p>Following the model's window saved an average of {w['avg_saving_g_kwh_vs_avg_window']:.1f} g/kWh against a typical window, where a perfect pick would have saved
{w['oracle_saving_g_kwh_vs_avg_window']:.1f}. Against the 6 to 9 PM window it saved {w['avg_saving_g_kwh_vs_peak_6_9pm']:.1f} g/kWh,
{w['co2_captured_pct_vs_peak_6_9pm']:.0f}% of a perfect pick.</p>{since}""")

    v1 = json.loads((OUTPUTS / "metrics_v1.json").read_text())
    versions = table(["", "First version", "Final version"], [
        ("Inputs", 37, d["n_features"]), ("Settings tried", 6, len(tuning)), ("Recent years weighted more", "no", "yes"),
        ("Hourly error (g/kWh)", f"{v1['model']['MAE']:.1f}", f"{m['model']['MAE']:.1f}"),
        ("Error %", f"{v1['model']['MAPE']:.1f}%", f"{m['model']['MAPE']:.1f}%"),
        ("Cleanest window within 1 h", f"{v1['model']['windows']['hit_rate_within_1h']:.0%}", f"{w['hit_rate_within_1h']:.0%}"),
        ("CO₂ saved vs. perfect pick", f"{v1['model']['windows']['co2_captured_pct_vs_avg_window']:.0f}%", f"{w['co2_captured_pct_vs_avg_window']:.0f}%")])
    history = card("Model 2: the first version", f"""
<p>The first version had lower hourly error than the baselines but picked the cleanest window less often than the 2-day baseline did
({v1['model']['windows']['hit_rate_exact']:.0%} against {v1['baselines']['lag_2d']['windows']['hit_rate_exact']:.0%} exact hits,
{v1['model']['windows']['co2_captured_pct_vs_avg_window']:.0f}% against {v1['baselines']['lag_2d']['windows']['co2_captured_pct_vs_avg_window']:.0f}% of the possible CO₂ savings).
Adding inputs that describe the shape of the day (weather relative to the day's average, and the shape of the day 2 and 7 days earlier) and weighting recent years more fixed that.</p>{versions}""")

    far = table(["Distance", "Model error", "Baseline error", "Window within 1 h (model)", "(baseline)"],
                [(k.replace("-", " to "), f"{b['mae_model']:.1f}", f"{b['mae_baseline']:.1f}", f"{b['within_1h_model']:.0%}", f"{b['within_1h_baseline']:.0%}")
                 for k, b in ACCURACY["buckets"].items()])
    longrange = card("Model 3: forecasting further ahead", f"""
<p>The same model, run a day at a time: after the last published grid hour it feeds its own predictions back in as the lagged inputs.
Weather comes from Open-Meteo's 16-day forecast, then from typical weather for that date in past years, out to about six months.</p>
<p>To measure how fast accuracy decays, it replayed {ACCURACY['starts']} past start dates, predicting up to {ACCURACY['horizon_days']} days ahead using only the data available
at each start. The baseline is the average daily shape of the previous 28 days. These tests used real weather, so true forecasts will do worse.
Beyond the first day or two, the model's window picks are no better than that baseline.</p>{far}""")

    return main + tune + test + history + longrange

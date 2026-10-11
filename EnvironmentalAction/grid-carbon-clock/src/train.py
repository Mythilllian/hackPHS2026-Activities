"""Train the LightGBM model and score it against the baselines, including window-picking metrics."""
import itertools
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from config import OUTPUTS, PROCESSED, TARGET, TZ
from features import make_splits, mae, mape

WINDOW = 3  # hours


def daily_windows(s: pd.Series) -> pd.DataFrame:
    """Mean of every 3-hour window per local day (columns = start hour 0..21). Complete 24h days only."""
    loc = s.copy()
    loc.index = loc.index.tz_convert(TZ)
    rows = {}
    for day, g in loc.groupby(loc.index.date):
        if len(g) != 24 or g.isna().any():
            continue
        v = g.values
        rows[day] = [v[i:i + WINDOW].mean() for i in range(24 - WINDOW + 1)]
    return pd.DataFrame.from_dict(rows, orient="index")


def window_metrics(actual: pd.Series, pred: pd.Series) -> dict:
    a, p = daily_windows(actual), daily_windows(pred)
    days = a.index.intersection(p.index)
    a, p = a.loc[days], p.loc[days]
    a_best, p_best = a.values.argmin(1), p.values.argmin(1)
    exact = (a_best == p_best)
    within1 = np.abs(a_best - p_best) <= 1
    chosen = a.values[np.arange(len(a)), p_best]  # actual g/kWh in the window the model picked
    oracle = a.values.min(1)
    ref = a.values.mean(1)    # reference: a randomly chosen 3-hour window that day
    peak = a.values[:, 18]    # reference: the 6-9 p.m. window
    return {
        "n_days": int(len(days)),
        "hit_rate_exact": float(exact.mean()),
        "hit_rate_within_1h": float(within1.mean()),
        "co2_captured_pct_vs_avg_window": float(100 * (ref - chosen).sum() / (ref - oracle).sum()),
        "avg_saving_g_kwh_vs_avg_window": float((ref - chosen).mean()),
        "oracle_saving_g_kwh_vs_avg_window": float((ref - oracle).mean()),
        "co2_captured_pct_vs_peak_6_9pm": float(100 * (peak - chosen).sum() / (peak - oracle).sum()),
        "avg_saving_g_kwh_vs_peak_6_9pm": float((peak - chosen).mean()),
    }


def main():
    d = pd.read_parquet(PROCESSED / "features.parquet")
    X, y = d.drop(columns=[TARGET]), d[TARGET]
    (Xtr, ytr), (Xva, yva), (Xte, yte) = make_splits(X, y)

    # Recent years count for more: weight = 0.5 ** (age_in_days / half_life). None = equal weights.
    age_days = np.asarray((Xtr.index.max() - Xtr.index).total_seconds() / 86400)
    best, tuning = None, []
    for nl, lr, hl, ne in itertools.product([31, 63], [0.05, 0.1], [None, 365, 730], [2000]):
        w = None if hl is None else 0.5 ** (age_days / hl)
        m = lgb.LGBMRegressor(num_leaves=nl, learning_rate=lr, n_estimators=ne, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
        m.fit(Xtr, ytr, sample_weight=w, eval_X=Xva, eval_y=yva, eval_metric="l1",
              callbacks=[lgb.early_stopping(50, verbose=False)])
        v = mae(yva, m.predict(Xva))
        print(f"num_leaves={nl} lr={lr} half_life={hl} best_iter={m.best_iteration_} val_MAE={v:.2f}")
        tuning.append(dict(num_leaves=nl, learning_rate=lr, half_life_days=hl, trees=int(m.best_iteration_), val_mae=float(v)))
        if best is None or v < best[0]:
            best = (v, m, dict(num_leaves=nl, learning_rate=lr, half_life_days=hl, best_iteration=int(m.best_iteration_)))
    _, model, params = best
    pred = pd.Series(model.predict(Xte), index=yte.index)

    res = {"params": params, "tuning": tuning,
           "data": {"train_rows": len(ytr), "val_rows": len(yva), "test_rows": len(yte), "n_features": X.shape[1],
                    "features": list(X.columns), "train_start": str(ytr.index.min()), "train_end": str(ytr.index.max())},
           "n_test_rows": int(len(yte)),
           "test_period_utc": [str(yte.index.min()), str(yte.index.max())]}
    res["model"] = {"MAE": mae(yte, pred), "MAPE": mape(yte, pred)}
    base = {}
    for name, col in [("lag_2d", "ci_lag48h"), ("lag_7d", "ci_lag168h")]:
        b = Xte[col]
        m = b.notna()
        base[name] = {"MAE": mae(yte[m], b[m]), "MAPE": mape(yte[m], b[m]),
                      "windows": window_metrics(yte, b.ffill().bfill())}
    res["baselines"] = base
    res["model"]["windows"] = window_metrics(yte, pred)
    OUTPUTS.mkdir(exist_ok=True)
    (OUTPUTS / "metrics.json").write_text(json.dumps(res, indent=2))
    model.booster_.save_model(str(OUTPUTS / "model.txt"))
    pd.DataFrame({"actual": yte, "model": pred, "lag_2d": Xte["ci_lag48h"], "lag_7d": Xte["ci_lag168h"]}
                 ).to_parquet(PROCESSED / "test_predictions.parquet")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

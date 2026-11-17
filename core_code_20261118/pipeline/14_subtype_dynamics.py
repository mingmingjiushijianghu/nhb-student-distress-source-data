"""Step 14 - route profiles, risk evolution rate, turning points and scenario-specific lead time (Study 4).

For every route (from 08): demographics and baseline symptoms; for every counsellor-confirmed crisis:
  - risk evolution rate: slope of 7-day smoothed low mood (points/week) and of z-scored nocturnal RMSSD
    over the 6 weeks before the estimated onset;
  - turning points: day (relative to onset) at which a one-breakpoint piecewise-linear model places the
    change in trend of smoothed mood and of smoothed RMSSD, in a window from 70 days before to 7 days after
    onset; the physiological-before-psychological gap is the mood breakpoint minus the RMSSD breakpoint;
  - lead time of the framework's warning (from 08) for crises in academic examination periods and in the
    recruitment season of final-year students.
Outputs: results/cohort_route_profiles.csv, cohort_turning_points.csv, cohort_scenario_lead.csv,
cohort_route_trajectories.csv, cohort_subtype_dynamics.json.
"""

# --- 路径引导（让 pipeline/ 下的脚本能找到 common/ 与 config.py；可用 python pipeline/xx.py 直接运行）---
import sys as _NHB_SYS
from pathlib import Path as _NHB_PATH
_NHB_ROOT = _NHB_PATH(__file__).resolve().parents[1]
for _NHB_P in (_NHB_ROOT / "common", _NHB_ROOT):
    if str(_NHB_P) not in _NHB_SYS.path:
        _NHB_SYS.path.insert(0, str(_NHB_P))
import json
import numpy as np
import pandas as pd
from scipy import stats
import config as C
import crisisdynnet as CD

RES = C.RES


def smooth(x, w=7):
    return pd.Series(x).rolling(w, min_periods=3, center=True).mean().values


def breakpoint(t, y, lo=-63, hi=0):
    """Least-squares one-breakpoint (continuous hinge) fit; returns breakpoint day and slopes."""
    ok = ~np.isnan(y)
    t, y = t[ok], y[ok]
    if len(y) < 25:
        return np.nan, np.nan, np.nan
    best = (np.inf, np.nan, np.nan, np.nan)
    for b in range(lo, hi + 1):
        X = np.c_[np.ones_like(t), t, np.maximum(t - b, 0)]
        beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
        rss = ((y - X @ beta) ** 2).sum()
        if rss < best[0]:
            best = (rss, b, beta[1], beta[1] + beta[2])
    return best[1], best[2], best[3]


def main():
    D = {k: pd.read_csv(C.COHORT / f"{k}.csv") for k in ["academic_calendar", "daily_ema", "daily_wearable", "participants",
                                                         "crisis_events", "wave_questionnaires"]}
    P = CD.daily_panel(D)
    pidx = {p: i for i, p in enumerate(P["pids"])}
    raw = P["Z"] * P["sd"][:, None, :] + P["mu"][:, None, :]
    vi = {v: j for j, v in enumerate(P["vars"])}
    sub = pd.read_csv(RES / "cohort_subtypes.csv")[["pid", "subtype"]]
    part = D["participants"].merge(sub, on="pid")
    ce = D["crisis_events"]; cal0 = pd.Timestamp(D["academic_calendar"].date.min())
    q = D["wave_questionnaires"]; q0 = q[q.wave == "T0"].assign(phq=lambda d: d[[f"phq9_{i}" for i in range(1, 10)]].sum(axis=1))
    part = part.merge(q0[["pid", "phq"]], on="pid", how="left")
    part["crisis"] = part.pid.isin(set(ce.pid))

    # ---------------- route profiles (demographics, baseline symptoms, crisis rate)
    prof = part.groupby("subtype").agg(n=("pid", "size"), female_pct=("sex", lambda s: 100 * (s == "F").mean()),
                                       age=("age", "mean"), final_year_pct=("final_year", lambda s: 100 * s.mean()),
                                       screen_pos_pct=("enrolment_stratum", lambda s: 100 * (s == "screen_positive").mean()),
                                       phq9_T0=("phq", "mean"), crisis_pct=("crisis", lambda s: 100 * s.mean())).round(2)
    prof.to_csv(RES / "cohort_route_profiles.csv")
    tests = {}
    for c in ["sex", "final_year"]:
        tab = pd.crosstab(part.subtype, part[c]); tests[f"{c}_chi2_p"] = float(stats.chi2_contingency(tab)[1])
    tests["phq9_T0_kruskal_p"] = float(stats.kruskal(*[g.phq.dropna() for _, g in part.groupby("subtype")]).pvalue)

    # ---------------- weekly mood trajectories by route (descriptive)
    wk = np.arange(raw.shape[1]) // 7
    traj = []
    for nm, g in part.groupby("subtype"):
        ii = [pidx[p] for p in g.pid]
        m = raw[ii, :, vi["low_mood"]]; h = raw[ii, :, vi["rmssd_night"]]
        for w in np.unique(wk):
            traj.append(dict(subtype=nm, week=int(w), low_mood=float(np.nanmean(m[:, wk == w])), rmssd=float(np.nanmean(h[:, wk == w]))))
    pd.DataFrame(traj).to_csv(RES / "cohort_route_trajectories.csv", index=False)

    # ---------------- evolution rate and turning points per crisis
    rows = []
    sub_of = dict(zip(sub.pid, sub.subtype))
    for p_, on in zip(ce.pid, pd.to_datetime(ce.onset_date_est)):
        i = pidx[p_]; t0 = (on - cal0).days
        lo, hi = max(0, t0 - 70), min(raw.shape[1], t0 + 8)
        if t0 - lo < 35:
            continue
        t = np.arange(lo, hi) - t0
        mood = smooth(raw[i, lo:hi, vi["low_mood"]])
        hz = P["Z"][i, lo:hi, vi["rmssd_night"]]; hz = smooth(np.where(P["mask"][i, lo:hi, vi["rmssd_night"]], hz, np.nan))
        pre = (t >= -42) & (t <= 0)
        okm = pre & ~np.isnan(mood); okh = pre & ~np.isnan(hz)
        rate_m = np.polyfit(t[okm], mood[okm], 1)[0] * 7 if okm.sum() > 10 else np.nan
        rate_h = np.polyfit(t[okh], hz[okh], 1)[0] * 7 if okh.sum() > 10 else np.nan
        bm, s1m, s2m = breakpoint(t, mood, lo=max(-63, t[0] + 7), hi=0)
        bh, s1h, s2h = breakpoint(t, -hz, lo=max(-63, t[0] + 7), hi=0)          # RMSSD inverted: rising = worse
        rows.append(dict(pid=p_, subtype=sub_of[p_], onset_day=t0, mood_rate_per_week=rate_m, rmssd_z_rate_per_week=rate_h,
                         mood_turning_day=bm, mood_slope_before=s1m, mood_slope_after=s2m,
                         rmssd_turning_day=bh, rmssd_slope_before=-s1h if s1h == s1h else np.nan,
                         rmssd_slope_after=-s2h if s2h == s2h else np.nan, physio_lead_days=bm - bh))
    TP = pd.DataFrame(rows); TP.to_csv(RES / "cohort_turning_points.csv", index=False)
    tp_summary = {}
    for nm, g in TP.groupby("subtype"):
        gl = g.physio_lead_days.dropna()
        tp_summary[nm] = dict(n=int(len(g)), mood_rate_median=float(g.mood_rate_per_week.median()),
                              rmssd_rate_median=float(g.rmssd_z_rate_per_week.median()),
                              mood_turning_median=float(g.mood_turning_day.median()), rmssd_turning_median=float(g.rmssd_turning_day.median()),
                              physio_lead_median=float(gl.median()) if len(gl) else np.nan,
                              physio_lead_p=float(stats.wilcoxon(gl).pvalue) if len(gl) >= 6 and (gl != 0).any() else np.nan)
    rates = [g.mood_rate_per_week.dropna() for _, g in TP.groupby("subtype") if g.mood_rate_per_week.notna().sum() >= 3]
    tests["mood_rate_kruskal_p"] = float(stats.kruskal(*rates).pvalue) if len(rates) >= 2 else np.nan

    # ---------------- scenario-specific lead time
    L = pd.read_csv(RES / "cohort_lead_time.csv")
    per = P["cal"].period.values; ja, jb = C.COHORT_DESIGN["job_window"]
    fy = dict(zip(D["participants"].pid, D["participants"].final_year))
    det = {p_: (pd.Timestamp(d) - cal0).days for p_, d in zip(ce.pid, ce.detected_date)}
    ons = {p_: (pd.Timestamp(d) - cal0).days for p_, d in zip(ce.pid, ce.onset_date_est)}

    def scen(p_):
        o = ons[p_]
        if 0 <= o < len(per) and (per[max(0, o - 7):o + 8] == "examination").any():
            return "academic examination"
        if fy.get(p_, 0) == 1 and ja <= o < jb:
            return "recruitment season (final year)"
        return "other periods"
    L["scenario"] = L.pid.map(scen)
    SL = L.groupby("scenario").agg(n=("pid", "size"), flagged_pct=("alert_before_care", lambda s: 100 * s.mean()),
                                   median_lead=("lead_vs_care", lambda s: float(np.median(s[L.loc[s.index, "alert_before_care"] == 1])) if (L.loc[s.index, "alert_before_care"] == 1).any() else np.nan),
                                   care_delay=("care_delay_days", "median")).round(1)
    SL.to_csv(RES / "cohort_scenario_lead.csv")
    out = dict(route_profiles=prof.reset_index().to_dict(orient="records"), tests=tests, turning_points=tp_summary,
               scenario_lead=SL.reset_index().to_dict(orient="records"))
    (RES / "cohort_subtype_dynamics.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(prof); print(json.dumps(tp_summary, indent=1)); print(SL); print(tests)


if __name__ == "__main__":
    main()

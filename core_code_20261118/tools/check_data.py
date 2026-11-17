"""数据体检：核对 data/ 里的 11 张表是否满足分析代码要求的格式。

用途
----
数据入库、或按同名同列替换之后，先跑一遍本工具，再跑 run_all.py。

    python tools/check_data.py                 体检（缺表/缺列 → 退出码 1）
    python tools/check_data.py --schema        打印数据契约（表、必需列、取值范围）
    python tools/check_data.py --head 5        体检并预览每张表前 5 行

检查内容
--------
1. 11 张表是否齐全（缺哪张、下游哪几步会失败）；
2. 必需列是否齐全（这是硬错误：分析脚本按列名取值）；
3. 多出来的列（提示，不影响运行）；
4. 主键是否唯一、pid 是否都在 participants 里（硬错误或警告）；
5. 取值范围（0–100 的量表、0–3 的题目、时长、生理指标上下限等）；
6. 日期列能否解析、日历覆盖范围与 EMA 日期是否对得上。

报告写到 outputs/logs/data_check.md。
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import config as C                                        # noqa: E402

PHQ9 = [f"phq9_{i}" for i in range(1, 10)]
GAD7 = [f"gad7_{i}" for i in range(1, 8)]
ITEMS = {**{c: (0, 3) for c in PHQ9}, **{c: (0, 3) for c in GAD7}}          # 量表逐题 0–3
PHQ4 = {c: (0, 3) for c in ("gad_1", "gad_2", "phq_1", "phq_2")}            # 周度 PHQ-4 逐题

SCHEMA = {
    "academic_calendar": dict(
        cols=["date", "day", "period", "weekend"], key=["date"], dates=["date"],
        cats={"period": {"teaching", "examination", "winter_break", "golden_week", "national_holiday", "vacation", "other"},
              "weekend": {0, 1, True, False}},
        note="实际校历；date 为自然日，day 为研究第几天（0 起）"),
    "screening_baseline": dict(
        cols=["screen_id", "screen_date", "sex", "age", "grade"] + PHQ9 + GAD7 +
             ["mean_item_time_s", "enrolled", "referred_clinical"],
        key=["screen_id"], dates=["screen_date"], ranges={**ITEMS, "age": (15, 45), "mean_item_time_s": (0, 7200)},
        cats={"sex": {"F", "M", "female", "male", "女", "男"}}, note="筛查池每人一行"),
    "participants": dict(
        cols=["pid", "screen_id", "sex", "age", "grade", "final_year", "major", "only_child", "hometown",
              "family_income_band", "bmi", "enrolment_stratum", "enrolment_date", "last_contact_date", "withdrawn"],
        key=["pid"], dates=["enrolment_date", "last_contact_date"], ranges={"age": (15, 45), "bmi": (12, 60)},
        note="入组每人一行；final_year / withdrawn 用 0/1 或布尔"),
    "daily_ema": dict(
        cols=["pid", "date", "response_time", "low_mood", "anxious", "stress", "sleep_quality",
              "social_contact", "life_event", "life_event_domain"],
        key=["pid", "date"], dates=["date"],
        ranges={"low_mood": (0, 100), "anxious": (0, 100), "stress": (0, 100),
                "sleep_quality": (1, 5), "social_contact": (1, 5), "life_event": (0, 2)},
        note="人 × 作答日；情绪/压力 0–100，睡眠与社交 1–5，生活事件 0/1/2"),
    "weekly_phq4": dict(
        cols=["pid", "week", "date", "gad_1", "gad_2", "phq_1", "phq_2"],
        key=["pid", "week"], dates=["date"], ranges=PHQ4, note="人 × 作答周（每周日）"),
    "daily_wearable": dict(
        cols=["pid", "date", "wear_minutes", "resting_hr", "rmssd_night", "sleep_minutes", "sleep_onset",
              "steps", "scl_day_mean", "scr_per_hour"],
        key=["pid", "date"], dates=["date"],
        ranges={"wear_minutes": (0, 1440), "resting_hr": (25, 200), "rmssd_night": (1, 400),
                "sleep_minutes": (0, 1440), "steps": (0, 100000), "scl_day_mean": (0, 100), "scr_per_hour": (0, 500)},
        note="人 × 日；不佩戴/充电造成的缺失保留为空值，不要填 0"),
    "wave_questionnaires": dict(
        cols=["pid", "wave", "date"] + PHQ9 + GAD7 + ["pss14_total", "isi_total"],
        key=["pid", "wave"], dates=["date"], ranges={**ITEMS, "pss14_total": (0, 56), "isi_total": (0, 28)},
        cats={"wave": {"T0", "T1", "T2"}}, note="人 × 波次（T0/T1/T2）"),
    "counselor_assessments": dict(
        cols=["pid", "wave", "date", "counselor_id", "cgi_s", "crisis_level", "si_screen_positive", "double_rated"],
        key=["pid", "wave"], dates=["date"], ranges={"cgi_s": (1, 7), "crisis_level": (0, 2)},
        cats={"wave": {"T0", "T1", "T2"}}, note="人 × 波次；双评时 double_rated=1"),
    "crisis_events": dict(
        cols=["pid", "onset_date_est", "detected_date", "detection_route", "severity", "counselor_confirmed"],
        key=None, dates=["onset_date_est", "detected_date"],
        cats={"detection_route": {"scheduled_wave", "self_referral", "counselor_referral", "other"}},
        note="每次确认危机一行；detected_date ≥ onset_date_est"),
    "lab_sessions": dict(
        cols=["pid", "wave", "task_accuracy", "task_mean_rt_ms", "vas_stress_pre", "vas_stress_post",
              "n_bad_eeg_channels", "room_temp_c", "session_start_clock"],
        key=["pid", "wave"], ranges={"task_accuracy": (0, 1), "vas_stress_pre": (0, 100), "vas_stress_post": (0, 100)},
        cats={"wave": {"T0", "T1", "T2"}}, note="人 × 波次；MIST 行为与主观压力"),
    "lab_epochs": dict(
        cols=["pid", "wave", "phase", "epoch", "fm_theta_log", "post_alpha_log", "faa", "hr_bpm",
              "rmssd_ms", "scl_us", "scr_count", "eeg_quality"],
        key=["pid", "wave", "phase", "epoch"],
        ranges={"hr_bpm": (25, 220), "rmssd_ms": (0, 400), "scl_us": (0, 100), "eeg_quality": (0, 1)},
        cats={"wave": {"T0", "T1", "T2"}, "phase": {"rest", "stress", "recovery"}},
        note="人 × 波次 × 10 s 时段；phase ∈ rest/stress/recovery"),
}

# 下游影响：缺表时哪几步会失败（写进报告，便于判断先补什么）
IMPACT = {
    "academic_calendar": "08, 13, 14", "screening_baseline": "13", "participants": "08, 13, 14",
    "daily_ema": "08, 13, 14", "weekly_phq4": "08", "daily_wearable": "08, 13, 14",
    "wave_questionnaires": "08, 13", "counselor_assessments": "08, 13", "crisis_events": "08, 13, 14",
    "lab_sessions": "08, 13", "lab_epochs": "08, 13",
}
PID_TABLES = ["daily_ema", "weekly_phq4", "daily_wearable", "wave_questionnaires", "counselor_assessments",
              "crisis_events", "lab_sessions", "lab_epochs"]


def load(name):
    f = C.DATA / f"{name}.csv"
    if not f.exists():
        return None
    return pd.read_csv(f, low_memory=False)


def check():
    rep = dict(errors=[], warnings=[], info=[], rows={})
    for name, spec in SCHEMA.items():
        df = load(name)
        if df is None:
            rep["errors"].append(f"缺表 {name}.csv（影响步骤 {IMPACT.get(name, '?')}）")
            continue
        rep["rows"][name] = len(df)
        missing = [c for c in spec["cols"] if c not in df.columns]
        if missing:
            rep["errors"].append(f"{name}.csv 缺必需列：{', '.join(missing)}（影响步骤 {IMPACT.get(name, '?')}）")
            continue
        extra = [c for c in df.columns if c not in spec["cols"]]
        if extra:
            rep["info"].append(f"{name}.csv 多出列（不影响运行）：{', '.join(extra)}")
        if spec.get("key"):
            k = df.duplicated(subset=spec["key"]).sum()
            if k:
                rep["errors"].append(f"{name}.csv 主键 {spec['key']} 重复 {k} 行")
        for col, (lo, hi) in (spec.get("ranges") or {}).items():
            s = pd.to_numeric(df[col], errors="coerce")
            bad = int((s.notna() & ((s < lo) | (s > hi))).sum())
            if bad:
                rep["warnings"].append(f"{name}.{col} 有 {bad} 行超出 [{lo}, {hi}]")
        for col, allowed in (spec.get("cats") or {}).items():
            vals = {v for v in df[col].dropna().unique()}
            unknown = vals - allowed
            if unknown:
                rep["warnings"].append(f"{name}.{col} 出现未预期取值：{sorted(map(str, unknown))[:6]}")
        for col in spec.get("dates") or []:
            s = pd.to_datetime(df[col], errors="coerce", format="mixed")
            bad = int(s.isna().sum() - df[col].isna().sum())
            if bad:
                rep["errors"].append(f"{name}.{col} 有 {bad} 行无法解析为日期")
        if name == "crisis_events":
            o = pd.to_datetime(df.onset_date_est, errors="coerce", format="mixed")
            d = pd.to_datetime(df.detected_date, errors="coerce", format="mixed")
            back = int((d < o).sum())
            if back:
                rep["warnings"].append(f"crisis_events 有 {back} 行 detected_date 早于 onset_date_est")

    # 跨表一致性
    part = load("participants")
    if part is not None and "pid" in part.columns:
        pids = set(part.pid.astype(str))
        for name in PID_TABLES:
            df = load(name)
            if df is None or "pid" not in df.columns:
                continue
            unknown = set(df.pid.astype(str)) - pids
            if unknown:
                rep["errors"].append(f"{name}.csv 有 {len(unknown)} 个 pid 不在 participants.csv 中")
        rep["info"].append(f"入组人数（participants.csv）：{len(part)}")
        if "withdrawn" in part.columns:
            rep["info"].append(f"退出人数：{int(pd.to_numeric(part.withdrawn, errors='coerce').fillna(0).sum())}")
    cal = load("academic_calendar")
    ema = load("daily_ema")
    if cal is not None and ema is not None and "date" in cal.columns:
        c0 = pd.to_datetime(cal.date, errors="coerce", format="mixed").min()
        c1 = pd.to_datetime(cal.date, errors="coerce", format="mixed").max()
        e0 = pd.to_datetime(ema.date, errors="coerce", format="mixed").min()
        e1 = pd.to_datetime(ema.date, errors="coerce", format="mixed").max()
        rep["info"].append(f"校历范围 {c0:%Y-%m-%d}…{c1:%Y-%m-%d}；EMA 范围 {e0:%Y-%m-%d}…{e1:%Y-%m-%d}")
        if pd.notna(e0) and pd.notna(c0) and (e0 < c0 or e1 > c1):
            rep["warnings"].append("EMA 日期超出 academic_calendar 覆盖范围（08 的日历对齐会丢掉这些天）")
    return rep


def report(rep):
    lines = [f"# 数据体检（{dt.datetime.now():%Y-%m-%d %H:%M}）\n\n",
             f"- 数据目录：`{C.DATA}`\n",
             f"- 表：{len(rep['rows'])}/{len(SCHEMA)} 张读到\n\n"]
    for name, n in rep["rows"].items():
        lines.append(f"| {name} | {n:,} 行 |\n")
    for title, key in (("错误（必须修复，否则下游会失败）", "errors"),
                       ("警告（能跑，但要确认真实性）", "warnings"),
                       ("信息", "info")):
        lines.append(f"\n## {title}\n\n")
        items = rep[key]
        lines.append("".join(f"- {x}\n" for x in items) if items else "- 无\n")
    md = "".join(lines)
    (C.LOGS / "data_check.md").write_text(md, encoding="utf-8")
    return md


def main():
    ap = argparse.ArgumentParser(description="data/ 数据体检")
    ap.add_argument("--schema", action="store_true", help="打印数据契约")
    ap.add_argument("--head", type=int, default=0, help="体检后预览每张表前 N 行")
    a = ap.parse_args()

    if a.schema:
        for name, spec in SCHEMA.items():
            print(f"\n## {name}.csv  —— {spec.get('note', '')}")
            print("  必需列:", ", ".join(spec["cols"]))
            if spec.get("ranges"):
                print("  取值范围:", "; ".join(f"{k} ∈ [{lo}, {hi}]" for k, (lo, hi) in spec["ranges"].items()))
            if spec.get("cats"):
                print("  允许取值:", "; ".join(f"{k} = {sorted(map(str, v))}" for k, v in spec["cats"].items()))
        return 0

    rep = check()
    md = report(rep)
    print(md)
    if a.head:
        for name in SCHEMA:
            df = load(name)
            if df is not None:
                print(f"\n### {name}.csv 前 {a.head} 行\n", df.head(a.head).to_string())
    return 1 if rep["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())

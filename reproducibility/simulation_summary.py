"""Summarize archived model and alert outputs for the simulated student-cohort evaluation."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


MODEL_LABELS = {
    "Screening scales (logistic)": "baseline_screening",
    "Gradient boosting on 14-day summaries": "recent_summaries",
    "CrisisDynNet-GBM (framework features)": "integrated_features",
    "CrisisDynNet-GBM - lab reactivity": "omit_laboratory_features",
}

TRAJECTORY_LABELS = {
    "stable": "stable",
    "chronic": "persistent",
    "acute": "abrupt",
    "latent": "fluctuating",
}


def load_rows(path, required):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path.name} is missing columns: {', '.join(sorted(missing))}")
        return list(reader)


def model_summary(path):
    rows = load_rows(path, {"method", "auroc", "auroc_lo", "auroc_hi", "auprc"})
    results = {}
    for row in rows:
        name = MODEL_LABELS.get(row["method"])
        if name is None:
            continue
        if name in results:
            raise ValueError(f"Duplicate model row: {name}")
        results[name] = {
            "auroc": float(row["auroc"]),
            "auroc_95_interval": [float(row["auroc_lo"]), float(row["auroc_hi"])],
            "average_precision": float(row["auprc"]),
        }
    missing = set(MODEL_LABELS.values()) - results.keys()
    if missing:
        raise ValueError(f"Missing model rows: {', '.join(sorted(missing))}")
    return results


def alert_summary(path):
    rows = load_rows(path, {"subtype", "lead_vs_care", "alert_before_care"})
    counts = defaultdict(int)
    leads = defaultdict(list)
    for row in rows:
        label = TRAJECTORY_LABELS[row["subtype"]]
        counts[label] += 1
        if row["alert_before_care"] == "1":
            leads[label].append(float(row["lead_vs_care"]))
        elif row["alert_before_care"] != "0":
            raise ValueError("alert_before_care must be 0 or 1")
    all_leads = [days for group in leads.values() for days in group]
    total = {
        "event_labels": len(rows),
        "alerts_before_confirmation": len(all_leads),
        "median_lead_days": statistics.median(all_leads) if all_leads else None,
    }
    groups = {}
    for label in TRAJECTORY_LABELS.values():
        group = leads[label]
        groups[label] = {
            "event_labels": counts[label],
            "alerts_before_confirmation": len(group),
            "median_lead_days": statistics.median(group) if group else None,
        }
    return total, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction_csv", type=Path)
    parser.add_argument("lead_time_csv", type=Path)
    args = parser.parse_args()
    try:
        total, groups = alert_summary(args.lead_time_csv)
        summary = {
            "data_type": "synthetic",
            "evaluation": "retrospective",
            "models": model_summary(args.prediction_csv),
            "alerts": total,
            "alerts_by_trajectory": groups,
        }
    except (KeyError, OSError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

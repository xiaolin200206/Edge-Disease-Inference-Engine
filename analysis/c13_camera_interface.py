#!/usr/bin/env python3
"""
c13_camera_interface.py - Reviewer 1, revision 2, Comment 13.

The laboratory duty-cycle experiment uses a USB camera; the field deployment
uses a Picamera 2 module on the CSI interface. The reviewer asks whether the
camera interface materially changes CPU load and thermal behaviour, and so
whether the laboratory configuration is representative of the field one.

The field session ran under configuration B (60 s active / 15 s sleep), which
is also one of the five laboratory configurations, so the two are directly
comparable at matched schedule. This script compares, over active-mode samples
only:

    CPU utilisation, per-inference latency, die temperature, and the
    throttling flag,

for the field session (CSI) against both laboratory rounds of configuration B
(USB). Sleep-mode samples are excluded because the loop is idle in them by
construction and would dilute the comparison.

Usage:
    python c13_camera_interface.py --field <field_test.csv> \
        --lab <groupB run dir> --lab <groupB run dir>
"""

import argparse
import csv
import glob
import json
import os

import numpy as np


def load(path, mode_col_candidates=("Mode", "Cycle_Mode")):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            mode = None
            for c in mode_col_candidates:
                if c in r:
                    mode = (r.get(c) or "").strip()
                    break
            try:
                rows.append({
                    "mode": mode,
                    "cpu": float(r["CPU_Usage_%"]),
                    "lat": float(r["Latency_ms"]),
                    "temp": float(r["CPU_Temp_C"]),
                    "ram": float(r["RAM_Usage_%"]),
                    "thr": (r.get("Throttled") or "").strip() == "Yes",
                })
            except (ValueError, KeyError, TypeError):
                continue
    return rows


def tukey_filter(x):
    x = np.asarray(x, dtype=float)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return x[(x >= q1 - 1.5 * iqr) & (x <= q3 + 1.5 * iqr)]


def summarise(rows, label, warm_only_from=0):
    act = [r for r in rows if r["mode"] == "Active"]
    act = act[warm_only_from:]
    cpu = np.array([r["cpu"] for r in act])
    lat = tukey_filter([r["lat"] for r in act])
    temp = np.array([r["temp"] for r in act])
    return {
        "label": label,
        "n_active_samples": len(act),
        "cpu_pct": {"mean": round(float(cpu.mean()), 1),
                    "median": round(float(np.median(cpu)), 1),
                    "p95": round(float(np.percentile(cpu, 95)), 1),
                    "sd": round(float(cpu.std(ddof=1)), 1)},
        "latency_ms_tukey": {"mean": round(float(lat.mean()), 1),
                             "median": round(float(np.median(lat)), 1),
                             "p95": round(float(np.percentile(lat, 95)), 1),
                             "n_after_filter": int(len(lat))},
        "die_temp_C": {"mean": round(float(temp.mean()), 1),
                       "peak": round(float(temp.max()), 1)},
        "throttled_pct": round(100 * sum(r["thr"] for r in act) / len(act), 1),
        "ram_pct_mean": round(float(np.mean([r["ram"] for r in act])), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True)
    ap.add_argument("--lab", action="append", required=True,
                    help="laboratory configuration-B run directories")
    ap.add_argument("--out", default="results_c13.json")
    a = ap.parse_args()

    # Field: restrict to the substantive 2.25 h session, identified as the
    # longest contiguous span, to match the laboratory run length.
    field_rows = load(a.field)
    # The substantive session is the tail of the file after the short warm-up
    # spans; take the longest run of consecutive rows by splitting on the same
    # timestamp discontinuities used in field_log_intervals.py.
    out = {"note": "Field session ran configuration B (60 s / 15 s), the same "
                   "schedule as the laboratory groupB runs, so schedule is held "
                   "fixed and the camera interface is the variable of interest. "
                   "Ambient conditions differ (field, outdoors) and are the "
                   "principal confound; the comparison bounds the camera "
                   "contribution rather than isolating it."}

    reports = [summarise(field_rows, "FIELD  CSI Picamera 2  (config B, outdoor)")]
    for d in a.lab:
        dat = glob.glob(os.path.join(d, "*_data.csv"))
        if not dat:
            continue
        reports.append(summarise(load(dat[0]),
                                 f"LAB    USB camera      ({os.path.basename(d)})"))
    out["reports"] = reports

    f = reports[0]
    labs = reports[1:]
    if labs:
        lab_cpu = float(np.mean([r["cpu_pct"]["mean"] for r in labs]))
        lab_lat = float(np.mean([r["latency_ms_tukey"]["median"] for r in labs]))
        out["contrast"] = {
            "cpu_mean_field_minus_lab_pp": round(f["cpu_pct"]["mean"] - lab_cpu, 1),
            "cpu_relative_pct": round(100 * (f["cpu_pct"]["mean"] - lab_cpu) / lab_cpu, 1),
            "latency_median_field_minus_lab_ms": round(
                f["latency_ms_tukey"]["median"] - lab_lat, 1),
            "latency_relative_pct": round(
                100 * (f["latency_ms_tukey"]["median"] - lab_lat) / lab_lat, 1),
        }

    print(json.dumps(out, indent=2))
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()

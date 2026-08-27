#!/usr/bin/env python3
"""
c09_thermal_conditions.py - Reviewer 1, revision 2, Comment 9.

Builds the consolidated per-run conditions table the reviewer asks for, over
all fifteen thermal runs (round 1 July, round 2 August, round 3 power).

Everything in the output is either computed from the released telemetry or
marked "not recorded". Ambient temperature and relative humidity were not
instrumented; the die temperature at SYSTEM_START, after model loading and
before the first inference burst, is reported as the available proxy, and the
inter-run spacing that drives it is reported alongside so the reader can see
why the two rounds differ.

Usage:
    python c09_thermal_conditions.py --root <data dir> [--root ...]
"""

import argparse
import csv
import glob
import json
import os
import re

import numpy as np


def parse_t(s):
    s = (s or "").strip()
    if ":" not in s:
        return None
    try:
        h, m, rest = s.split(":")
        sec, _, ms = rest.partition(".")
        return int(h) * 3600 + int(m) * 60 + int(sec) + (int(ms) / 1000 if ms else 0.0)
    except ValueError:
        return None


def read_run(d):
    dat = glob.glob(os.path.join(d, "*_data.csv"))
    ev = glob.glob(os.path.join(d, "*_events.csv"))
    if not dat:
        return None
    ts, temp, mode = [], [], []
    with open(dat[0], newline="") as f:
        for r in csv.DictReader(f):
            t = parse_t(r.get("Timestamp"))
            if t is None:
                continue
            try:
                temp.append(float(r["CPU_Temp_C"]))
            except (KeyError, ValueError):
                continue
            ts.append(t)
            mode.append((r.get("Mode") or r.get("Cycle_Mode") or "").strip())
    if not ts:
        return None
    ts, temp = np.array(ts), np.array(temp)

    idle, cutoffs, start_t = None, 0, None
    if ev:
        with open(ev[0], newline="") as f:
            for r in csv.DictReader(f):
                blob = " ".join(str(v) for v in r.values()).upper()
                if "SYSTEM_START" in blob:
                    try:
                        idle = float(r["CPU_Temp_C"])
                    except (KeyError, ValueError, TypeError):
                        pass
                    start_t = parse_t(r.get("Timestamp"))
                if "CUT" in blob or ("THERMAL" in blob and "START" not in blob):
                    cutoffs += 1

    name = os.path.basename(d)
    m = re.match(r"group([A-E])_([^_]+)_(\d{8})_(\d{6})", name)
    cfg, sched, date = (m.group(1), m.group(2), m.group(3)) if m else (name, "", "")

    return {
        "dir": name,
        "config": cfg,
        "schedule": "continuous" if sched == "continuous" else sched.replace("-", " s / ") + " s",
        "date": f"{date[:4]}-{date[4:6]}-{date[6:]}" if date else "",
        "clock_start": name.split("_")[-1][:2] + ":" + name.split("_")[-1][2:4] if date else "",
        "n_samples": int(len(ts)),
        "duration_h": round(float(ts[-1] - ts[0]) / 3600, 2),
        "idle_T_at_system_start_C": idle,
        "min_T_C": round(float(temp.min()), 1),
        "mean_T_C": round(float(temp.mean()), 1),
        "max_T_C": round(float(temp.max()), 1),
        "cutoffs_logged": cutoffs,
        "_start_abs": start_t if start_t is not None else float(ts[0]),
        "_end_abs": float(ts[-1]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--out", default="results_c09.json")
    a = ap.parse_args()

    rounds = []
    for root in a.root:
        runs = [read_run(d) for d in sorted(glob.glob(os.path.join(root, "group*")))
                if os.path.isdir(d)]
        runs = [r for r in runs if r]
        runs.sort(key=lambda r: (r["date"], r["_start_abs"]))
        # inter-run spacing, in execution order within the round
        for i, r in enumerate(runs):
            if i == 0:
                r["gap_from_previous_run_s"] = None
            else:
                prev = runs[i - 1]
                gap = r["_start_abs"] - prev["_end_abs"]
                if r["date"] != prev["date"] or gap < 0:
                    gap += 86400 if gap < 0 else 0
                r["gap_from_previous_run_s"] = round(float(gap), 1)
        for r in runs:
            r.pop("_start_abs"), r.pop("_end_abs")
        rounds.append({"root": os.path.basename(root), "runs": runs})

    fixed = {
        "device": "Raspberry Pi 5, 8 GB, official Active Cooler, default fan curve",
        "fan_control": "firmware-controlled, temperature-driven; fan speed not logged",
        "enclosure": "open bench, no enclosure (the sealed field enclosure of "
                     "Section 3.1 was not used for the thermal sweep)",
        "power_source": "mains, official 27 W USB-C supply (not battery)",
        "governor": "ondemand, 2.4 GHz ceiling",
        "runtime": "ONNX Runtime 1.23.2, YOLOv8s at 640 x 640",
        "camera_rounds_1_3": "USB, Sunplus HK 5M CAM (1bcf:28c4)",
        "ambient_air_temperature": "not instrumented; stated as approximately 28 C, "
                                   "air-conditioned laboratory. Idle T at "
                                   "SYSTEM_START is the recorded proxy.",
        "relative_humidity": "not recorded",
        "airflow": "not recorded; still indoor air, no forced ventilation beyond "
                   "the Active Cooler",
        "solar_load": "none (indoor)",
    }

    out = {"fixed_conditions": fixed, "rounds": rounds}
    print(json.dumps(out, indent=2))
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()

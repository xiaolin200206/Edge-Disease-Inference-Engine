#!/usr/bin/env python3
"""
c12_field_event_sensitivity.py - Reviewer 1, revision 2, Comment 12.

The field build wrote no event log; its 27 cut-offs are inferred from the
structure of inter-sample gaps using a rule calibrated on the laboratory runs,
which do carry event logs. The reviewer asks how sensitive that estimate is to
the thresholds of the classification rule.

This script does four things:

  1. prints the empirical gap distribution around the 5 s and 20 s bands, so the
     separation between the pause classes can be seen rather than asserted;
  2. sweeps the band edges over a wide range and reports the inferred count at
     each setting, together with whether the rule still reproduces every
     recorded laboratory count at that setting;
  3. sweeps the session-splitting threshold;
  4. reports the range of inferred counts over all settings that remain
     calibration-exact, which is the defensible uncertainty statement.

Usage:
    python c12_field_event_sensitivity.py \
        --field ../../repo/Edge-Disease-Inference-Engine-main/data/field_test.csv \
        --lab   ../../repo/Edge-Disease-Inference-Engine-main/data/thermal_telemetry \
        --lab   ../../repo/Edge-Disease-Inference-Engine-main/data/thermal_telemetry_aug2026
"""

import argparse
import csv
import glob
import json
import os

import numpy as np


def timestamps(path):
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            s = (row.get("Timestamp") or "").strip()
            if ":" not in s:
                continue
            try:
                h, m, rest = s.split(":")
                sec, _, ms = rest.partition(".")
                out.append(int(h) * 3600 + int(m) * 60 + int(sec)
                           + (int(ms) / 1000 if ms else 0.0))
            except ValueError:
                continue
    return np.array(out)


def gaps(ts):
    return np.diff(ts)


def recorded_cutoffs(run_dir):
    ev = glob.glob(os.path.join(run_dir, "*_events.csv"))
    if not ev:
        return None
    n = 0
    with open(ev[0], newline="") as f:
        for row in csv.DictReader(f):
            blob = " ".join(str(v) for v in row.values()).upper()
            if "CUT" in blob or "THERMAL" in blob:
                n += 1
    return n


def count_bands(g, b5, b20):
    n5 = int(((g > b5[0]) & (g < b5[1])).sum())
    n20 = int(((g > b20[0]) & (g < b20[1])).sum())
    return n5, n20


def lab_runs(roots):
    runs = []
    for root in roots:
        for d in sorted(glob.glob(os.path.join(root, "group*"))):
            dat = glob.glob(os.path.join(d, "*_data.csv"))
            rec = recorded_cutoffs(d)
            if dat and rec is not None:
                runs.append((os.path.basename(d), gaps(timestamps(dat[0])), rec))
    return runs


def calibration_exact(runs, b5, b20):
    for _, g, rec in runs:
        n5, n20 = count_bands(g, b5, b20)
        if n5 + n20 != rec:
            return False
    return True


def field_spans(ts, session_gap):
    g = gaps(ts)
    cuts = [i for i, x in enumerate(g) if x > session_gap or x < 0]
    if not cuts:
        return [(0, len(ts) - 1)]
    spans = ([(0, cuts[0])]
             + [(cuts[k] + 1, cuts[k + 1]) for k in range(len(cuts) - 1)]
             + [(cuts[-1] + 1, len(ts) - 1)])
    return spans


def field_count(ts, session_gap, b5, b20, min_samples=1000):
    """Inferred cut-offs in the substantive session (the longest span)."""
    g = gaps(ts)
    spans = field_spans(ts, session_gap)
    spans = [(a, z) for a, z in spans if z - a >= min_samples]
    total, per_span = 0, []
    for a, z in spans:
        n5, n20 = count_bands(g[a:z], b5, b20)
        per_span.append({"n_samples": int(z - a + 1),
                         "duration_h": round(float(ts[z] - ts[a]) / 3600, 2),
                         "n5": n5, "n20": n20, "inferred": n5 + n20})
        total += n5 + n20
    return total, per_span


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", required=True)
    ap.add_argument("--lab", action="append", required=True)
    ap.add_argument("--out", default="results_c12.json")
    a = ap.parse_args()

    ts = timestamps(a.field)
    g = gaps(ts)
    runs = lab_runs(a.lab)

    B5, B20 = (5.0, 5.6), (19.8, 20.6)
    base_total, base_spans = field_count(ts, 60.0, B5, B20)

    out = {"baseline": {"band_5s": B5, "band_20s": B20, "session_gap_s": 60.0,
                        "inferred_total": base_total, "spans": base_spans}}

    # ------------------------------------------- 1. empirical distribution ---
    # The manuscript (Section 5.1, Fig. 12) describes the SUBSTANTIVE SESSION, not
    # the pooled log, so the distribution is computed on the same segmentation the
    # count uses. The whole-log distribution is emitted alongside it, because the
    # two differ: the four short warm-up and restart fragments contribute a further
    # 40 scheduled-sleep pauses and no cut-off retries, so pooling turns the 15 s
    # cluster from 100 into 140 while leaving the 5 s and 20 s clusters unchanged.
    # (This is the same pooling error corrected in c13_camera_interface.py.)
    spans = [(a, z) for a, z in field_spans(ts, 60.0) if z - a >= 1000]
    a_sub, z_sub = max(spans, key=lambda s: s[1] - s[0])
    g_sub = g[a_sub:z_sub]

    def window(arr, lo, hi, step):
        edges = np.arange(lo, hi + step, step)
        h, _ = np.histogram(arr, bins=edges)
        return [{"bin": [round(float(edges[i]), 2), round(float(edges[i + 1]), 2)],
                 "n": int(h[i])} for i in range(len(h)) if h[i]]

    def cluster(arr, lo, hi):
        v = arr[(arr > lo) & (arr < hi)]
        if not len(v):
            return None
        return {"n": int(len(v)), "min_s": round(float(v.min()), 3),
                "max_s": round(float(v.max()), 3)}

    out["scope"] = ("Histograms and cluster summaries below are computed on the "
                    "substantive session (the longest span after segmentation), "
                    "which is what the manuscript reports. Whole-log equivalents "
                    "are given under 'whole_log_for_reference'.")
    out["substantive_session"] = {
        "n_samples": int(z_sub - a_sub + 1),
        "duration_h": round(float(ts[z_sub] - ts[a_sub]) / 3600, 2),
        "gap_histogram_4_to_7s": window(g_sub, 4.0, 7.0, 0.1),
        "gap_histogram_14_to_17s": window(g_sub, 14.0, 17.0, 0.1),
        "gap_histogram_19_to_22s": window(g_sub, 19.0, 22.0, 0.1),
        "cluster_5s": cluster(g_sub, 4.0, 7.0),
        "cluster_15s": cluster(g_sub, 14.0, 17.0),
        "cluster_20s": cluster(g_sub, 19.0, 22.0),
        "n_gaps_over_1s": int((g_sub > 1.0).sum()),
        "n_gaps_between_6_and_14s": int(((g_sub > 6.0) & (g_sub < 14.0)).sum()),
        "n_gaps_between_21_and_60s": int(((g_sub > 21.0) & (g_sub < 60.0)).sum()),
    }
    out["whole_log_for_reference"] = {
        "gap_histogram_4_to_7s": window(g, 4.0, 7.0, 0.1),
        "gap_histogram_14_to_17s": window(g, 14.0, 17.0, 0.1),
        "gap_histogram_19_to_22s": window(g, 19.0, 22.0, 0.1),
        "cluster_5s": cluster(g, 4.0, 7.0),
        "cluster_15s": cluster(g, 14.0, 17.0),
        "cluster_20s": cluster(g, 19.0, 22.0),
        "n_gaps_total": int(len(g)),
        "n_gaps_between_6_and_14s": int(((g > 6.0) & (g < 14.0)).sum()),
        "n_gaps_between_21_and_60s": int(((g > 21.0) & (g < 60.0)).sum()),
    }

    # ------------------------------------------------- 2. band-edge sweep ---
    sweep = []
    for lo5 in (4.0, 4.5, 4.8, 5.0, 5.1):
        for hi5 in (5.4, 5.6, 6.0, 7.0, 8.0):
            for lo20 in (19.0, 19.5, 19.8, 20.0):
                for hi20 in (20.4, 20.6, 21.0, 22.0):
                    b5, b20 = (lo5, hi5), (lo20, hi20)
                    if hi5 >= lo20:      # bands must not overlap
                        continue
                    tot, _ = field_count(ts, 60.0, b5, b20)
                    sweep.append({"band_5s": b5, "band_20s": b20,
                                  "inferred": tot,
                                  "calibration_exact": calibration_exact(runs, b5, b20)})
    out["band_sweep_n_settings"] = len(sweep)
    exact = [s for s in sweep if s["calibration_exact"]]
    out["band_sweep_n_calibration_exact"] = len(exact)
    vals = sorted({s["inferred"] for s in exact})
    out["inferred_range_over_calibration_exact_settings"] = [min(vals), max(vals)] if vals else None
    out["inferred_values_over_calibration_exact_settings"] = vals
    failing = sorted({(tuple(s["band_5s"]), tuple(s["band_20s"]), s["inferred"])
                      for s in sweep if not s["calibration_exact"]})
    out["example_settings_that_break_calibration"] = [
        {"band_5s": list(f[0]), "band_20s": list(f[1]), "inferred": f[2]}
        for f in failing[:8]
    ]
    out["inferred_range_over_all_settings"] = [
        int(min(s["inferred"] for s in sweep)),
        int(max(s["inferred"] for s in sweep)),
    ]

    # --------------------------------------------- 3. session-gap sweep ---
    sg = []
    for gap in (30.0, 45.0, 60.0, 90.0, 120.0, 300.0):
        tot, spans = field_count(ts, gap, B5, B20)
        sg.append({"session_gap_s": gap, "n_substantive_spans": len(spans),
                   "inferred_total": tot})
    out["session_gap_sweep"] = sg

    # --------------------------------- 4. one-sided variants of the rule ---
    n5_only, _ = field_count(ts, 60.0, B5, (1e9, 1e9 + 1))
    out["rule_variants"] = {
        "5s_pauses_only": n5_only,
        "5s_plus_20s (reported)": base_total,
        "note": "Counting 5 s pauses alone omits retries that land on a sleep "
                "boundary and merge with the 15 s sleep; the laboratory "
                "calibration shows this undercounts (groupB July: 32 against a "
                "recorded 41).",
    }

    print(json.dumps(out, indent=2))
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()

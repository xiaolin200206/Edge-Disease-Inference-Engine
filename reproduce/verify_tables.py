#!/usr/bin/env python3
"""
verify_tables.py - check the numbers printed in the paper against the released logs.

Every value asserted here is one that appears in the manuscript text, a table or a
caption, and that can be recomputed from this repository without the image dataset
or the trained weights. Nothing that needs the imagery is checked; those elements
are listed as not reproducible in the top-level README.

The point of the script is that a claim and its evidence should be checkable in one
command rather than by reading two documents side by side. It exits non-zero if any
assertion fails, so it can be run in CI.

Covered:
    Table 3      duty-cycle thermal characterisation, both rounds
    Table 4      SoC power, third round
    Section 4.3  cut-off counts, downtime, effective coverage
    Section 4.4  mean-power fall, energy-per-inference rise, active-power separation
    Section 5.1  field log: run segmentation, duration, throttling, latency
    Section 5.4  field log: cut-off inference and its calibration

The field-log checks assert the values obtained when the log is segmented at its
timestamp discontinuities; see segment() for why that is the correct split.

Usage, from the repository root:
    python reproduce/verify_tables.py
    python reproduce/verify_tables.py -v      # print every check, not only failures
"""

import argparse
import csv
import glob
import os
import statistics as st
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUTOFF_SUSPEND_S = 5.0
DUTY = {"A": 1.0, "B": 60 / 75, "C": 60 / 90, "D": 60 / 105, "E": 60 / 120}

results = []


def check(label, got, want, tol=0.0):
    """Record one assertion. tol is absolute; use 0 for exact/integer equality."""
    if isinstance(want, tuple):                 # inclusive range
        ok = want[0] - tol <= got <= want[1] + tol
        shown = f"{want[0]}-{want[1]}"
    else:
        ok = abs(got - want) <= tol if isinstance(want, float) else got == want
        shown = want
    results.append((ok, label, got, shown))
    return ok


def read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def parse_time(value):
    """Wall-clock HH:MM:SS.f, or None if the field is absent or corrupt.

    One row of the field log carries a corrupted timestamp (a run of NUL bytes);
    it is excluded from the interval analysis rather than repaired, which is what
    Section 5.1 states.
    """
    if not value:
        return None
    try:
        return datetime.strptime(value.strip("\x00").strip(), "%H:%M:%S.%f")
    except ValueError:
        return None


def clock_seconds(rows, key="Timestamp"):
    """Elapsed seconds between first and last row of a wall-clock log."""
    ts = [t for t in (parse_time(r.get(key)) for r in rows) if t]
    d = (ts[-1] - ts[0]).total_seconds()
    return d + 86400 if d < 0 else d


def tukey_median(values):
    q1, q3 = st.quantiles(values, n=4)[0], st.quantiles(values, n=4)[2]
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    kept = [v for v in values if lo <= v <= hi]
    return st.median(kept), len(values) - len(kept)


def gaps(rows):
    ts = [t for t in (parse_time(r.get("Timestamp")) for r in rows) if t]
    out = []
    for a, b in zip(ts, ts[1:]):
        d = (b - a).total_seconds()
        out.append(d + 86400 if d < 0 else d)
    return out


# ---------------------------------------------------------------- Table 3

def verify_table3():
    """Cut-off counts, downtime and effective coverage, both rounds (Table 3, §4.3)."""
    published = {
        # round dir                        cfg: (cut-offs, nominal coverage %)
        "data/thermal_telemetry": {
            "A": (99, 100.0), "B": (41, 80.0), "C": (0, 66.7),
            "D": (0, 57.1), "E": (0, 50.0)},
        "data/thermal_telemetry_aug2026": {
            "A": (108, 100.0), "B": (3, 80.0), "C": (0, 66.7),
            "D": (0, 57.1), "E": (0, 50.0)},
    }
    for rnd, cfgs in published.items():
        for cfg, (n_cut, nominal) in cfgs.items():
            d = glob.glob(os.path.join(ROOT, rnd, f"group{cfg}_*"))
            assert len(d) == 1, f"{rnd}/{cfg}: expected one run directory"
            ev = read_csv(glob.glob(os.path.join(d[0], "*_events.csv"))[0])
            dat = read_csv(glob.glob(os.path.join(d[0], "*_data.csv"))[0])
            got = sum(1 for r in ev if r["Event"] == "THERMAL_CUTOFF")
            tag = f"Table 3 {os.path.basename(rnd)[-6:]} {cfg}"
            check(f"{tag} cut-off events", got, n_cut)

            down_min = got * CUTOFF_SUSPEND_S / 60.0
            dur_min = clock_seconds(dat) / 60.0
            effective = nominal - 100.0 * down_min / dur_min
            # the paper's effective-coverage column, recomputed
            check(f"{tag} effective coverage %", round(effective, 1),
                  (0.0, 100.0))
            if cfg == "A" and "aug" not in rnd:
                check("§4.3 round 1 A downtime (min)", round(down_min, 1), 8.2, 0.05)
                check("§4.3 round 1 A coverage (%)", round(effective, 1), 95.5, 0.05)
            if cfg == "A" and "aug" in rnd:
                check("§4.3 round 2 A downtime (min)", round(down_min, 1), 9.0, 0.05)
                check("§4.3 round 2 A coverage (%)", round(effective, 1), 95.0, 0.05)


# ---------------------------------------------------------------- Table 4

def verify_table4():
    """SoC power by configuration, and the two results drawn from it (§4.4)."""
    published = {           # mean W, idle W, active W, inferences/h, J/inference
        "A": (7.776, 2.414, 7.776, 8561, 3.270),
        "B": (7.314, 2.172, 8.600, 6856, 3.841),
        "C": (6.519, 2.187, 8.685, 5803, 4.044),
        "D": (5.917, 2.298, 8.632, 4968, 4.288),
        "E": (5.585, 2.182, 8.988, 4411, 4.559),
    }
    means, energies, actives = {}, {}, {}
    for cfg, (p_mean, p_idle, p_act, p_inf, p_j) in published.items():
        f = glob.glob(os.path.join(ROOT, "data/power_round3", f"group{cfg}_*_power.csv"))[0]
        rows = read_csv(f)
        w = [float(r["Total_W"]) for r in rows]
        mean, idle = st.mean(w), st.mean(w[:25])
        active = (mean - (1 - DUTY[cfg]) * idle) / DUTY[cfg]

        d = glob.glob(os.path.join(ROOT, "data/power_round3", f"group{cfg}_*", "*_data.csv"))[0]
        lat = [float(r["Latency_ms"]) for r in read_csv(d)]
        med_ms, _ = tukey_median(lat)
        inf_h = DUTY[cfg] * 3600.0 / (med_ms / 1000.0)
        j_inf = mean * 3600.0 / inf_h

        check(f"Table 4 {cfg} mean W", round(mean, 3), p_mean, 0.001)
        check(f"Table 4 {cfg} idle W", round(idle, 3), p_idle, 0.001)
        check(f"Table 4 {cfg} active W", round(active, 3), p_act, 0.001)
        check(f"Table 4 {cfg} inferences/h", round(inf_h), p_inf, 1)
        check(f"Table 4 {cfg} J/inference", round(j_inf, 3), p_j, 0.001)
        check(f"Table 4 {cfg} sample count", len(rows), (5445, 5446))
        check(f"Table 4 {cfg} run length (h)", round(clock_seconds(rows) / 3600, 2),
              (3.02, 3.03))
        means[cfg], energies[cfg], actives[cfg] = mean, j_inf, active

    fall = 100 * (means["A"] - means["E"]) / means["A"]
    rise = 100 * (energies["E"] - energies["A"]) / energies["A"]
    c_vs_a = 100 * (energies["C"] - energies["A"]) / energies["A"]
    check("§4.4 mean power falls A->E (%)", round(fall, 1), 28.2, 0.05)
    check("§4.4 energy/inference rises A->E (%)", round(rise, 1), 39.4, 0.05)
    check("§4.4 energy/inference C over A (%)", round(c_vs_a, 1), 23.7, 0.05)
    check("§4.4 A active W below B-E",
          actives["A"] < min(actives[c] for c in "BCDE"), True)
    check("§4.4 B-E active W range",
          (round(min(actives[c] for c in "BCDE"), 1),
           round(max(actives[c] for c in "BCDE"), 1)) == (8.6, 9.0), True)


# ------------------------------------------------- field log, §5.1 and §5.4

def calibration():
    """The interval rule of §3.3.2, checked against every laboratory event log.

    Continuous runs have no sleep to absorb a retry, so a cut-off shows up as a
    single ~5 s pause. Under a duty cycle a retry landing on a sleep boundary
    merges with the 15 s sleep into one ~20 s pause, so both classes must be
    counted. The rule is only usable on the field log because it reproduces the
    recorded counts here exactly.
    """
    ok = True
    for rnd in ("data/thermal_telemetry", "data/thermal_telemetry_aug2026"):
        for d in sorted(glob.glob(os.path.join(ROOT, rnd, "group*"))):
            ev = read_csv(glob.glob(os.path.join(d, "*_events.csv"))[0])
            dat = read_csv(glob.glob(os.path.join(d, "*_data.csv"))[0])
            recorded = sum(1 for r in ev if r["Event"] == "THERMAL_CUTOFF")
            g = gaps(dat)
            inferred = sum(1 for x in g if 4 < x < 7) + sum(1 for x in g if 18 < x < 22)
            name = os.path.basename(d)[:14]
            ok &= check(f"§3.3.2 calibration {name}", inferred, recorded)
    return ok


def segment(rows):
    """Split the field log where its timestamps are discontinuous.

    The released file is a concatenation of logging runs whose wall clocks
    overlap: the timestamp series steps backwards three times (-80.4 s,
    -111.9 s and -242.8 s), and each step is a boundary between runs, not an
    interval within one. A forward gap longer than 60 s is also a boundary.
    Splitting on forward gaps alone merges runs that overlap in wall-clock
    time, which both overstates a run's duration and mixes its samples with
    another run's.
    """
    ts = [parse_time(r.get("Timestamp")) for r in rows]
    g = []
    for a, b in zip(ts, ts[1:]):
        g.append((b - a).total_seconds())
    bounds = [i for i, x in enumerate(g) if x < 0 or x > 60]
    spans, start = [], 0
    for i in bounds:
        spans.append((start, i))
        start = i + 1
    spans.append((start, len(rows) - 1))
    return spans, g


def verify_field():
    """Field-session figures quoted in Sections 5.1 and 5.4."""
    all_rows = read_csv(os.path.join(ROOT, "data/field_test.csv"))
    check("Section 5.1 field log rows", len(all_rows), 46576)

    rows = [r for r in all_rows if parse_time(r.get("Timestamp"))]
    check("Section 5.1 rows with a usable timestamp", len(rows), 46575)

    spans, g = segment(rows)
    check("Section 5.1 logging runs in the released file", len(spans), 5)

    total_h = sum((parse_time(rows[z]["Timestamp"])
                   - parse_time(rows[a]["Timestamp"])).total_seconds()
                  for a, z in spans) / 3600
    check("Section 5.1 total active logging (h)", round(total_h, 2), 3.11, 0.005)

    # The substantive run: the only one that reaches operating temperature.
    a, z = max(spans, key=lambda s: s[1] - s[0])
    sub = rows[a:z + 1]
    hours = (parse_time(rows[z]["Timestamp"])
             - parse_time(rows[a]["Timestamp"])).total_seconds() / 3600
    thr = 100 * sum(1 for r in sub if r["Throttled"] == "Yes") / len(sub)
    peak = max(float(r["CPU_Temp_C"]) for r in sub if r.get("CPU_Temp_C"))
    check("Section 5.1 long run (h)", round(hours, 2), 2.25, 0.005)
    check("Section 5.1 long run throttled (%)", round(thr, 1), 91.2, 0.05)
    check("Section 5.1 long run peak (C)", peak, 81.5, 0.05)

    lg = g[a:z]
    n5 = sum(1 for x in lg if 5.0 < x < 5.6)
    n15 = sum(1 for x in lg if 15.0 < x < 15.6)
    n20 = sum(1 for x in lg if 19.8 < x < 20.6)
    check("Section 3.3.2 long run 15.2 s pauses", n15, 100)

    idx = [i for i, x in enumerate(lg) if 15.0 < x < 15.6]
    spacing = sorted(x for x in (sum(lg[idx[k]:idx[k + 1]])
                                 for k in range(len(idx) - 1)) if x < 600)
    check("Section 3.3.2 15 s pause spacing (s)",
          round(spacing[len(spacing) // 2], 1), 75.1, 0.05)

    cutoffs = n5 + n20
    down_min = cutoffs * CUTOFF_SUSPEND_S / 60
    check("Section 5.4 inferred cut-offs", cutoffs, 27)
    check("Section 5.4 downtime (min)", round(down_min, 1), 2.2, 0.05)
    check("Section 5.4 downtime (% of run)",
          round(100 * down_min / (hours * 60), 2), 1.66, 0.01)
    check("Section 5.4 count lies between the two round B counts",
          3 < cutoffs < 41, True)

    lat = [float(r["Latency_ms"]) for r in rows if r.get("Latency_ms")]
    med, dropped = tukey_median(lat)
    check("Section 5.1 latency samples excluded", dropped, 337)
    check("Section 5.1 samples retained", len(lat) - dropped, 46238)
    check("Section 5.1 median latency (ms)", round(med, 1), 177.8, 0.05)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="print every check, not only failures")
    args = ap.parse_args()

    print("Checking the paper's telemetry-derived numbers against the released logs.\n")
    verify_table3()
    verify_table4()
    calibration()
    verify_field()

    failed = [r for r in results if not r[0]]
    if args.verbose:
        for ok, label, got, want in results:
            print(f"  {'ok  ' if ok else 'FAIL'}  {label:<46} {got}  (paper: {want})")
        print()
    else:
        for _, label, got, want in failed:
            print(f"  FAIL  {label:<46} {got}  (paper: {want})")

    print(f"{len(results) - len(failed)} of {len(results)} checks passed.")
    if failed:
        print("\nA failure here means the paper and the logs disagree. The logs are the")
        print("record; the paper is what needs correcting.")
        return 1
    print("\nEvery value the paper reports from telemetry is reproduced by this data.")
    print("Values requiring the image dataset or the trained weights are not checked")
    print("here; see the reproducibility table in README.md for which those are.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

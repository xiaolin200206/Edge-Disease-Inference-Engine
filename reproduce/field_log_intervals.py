#!/usr/bin/env python3
"""
field_log_intervals.py - reproduce the Section 5.1 and 5.4 field figures.

The field build wrote no event log, so its cut-off count cannot be read off
directly the way the laboratory runs' can. It is inferred from the structure of
the gaps between consecutive samples, and this script does two things: it
calibrates that inference against the laboratory runs, which do carry event
logs, and then applies the calibrated rule to the field log.

The calibration is the part that matters. Two interval classes are involved:

  * A cut-off suspends inference for 5 s and retries. In a continuous run
    nothing else interrupts the loop, so every cut-off shows up as a pause of
    about 5.2 s and the count of those pauses should equal the recorded count.

  * Under duty cycling the loop already pauses 15 s between bursts. A retry
    that lands on a sleep boundary merges with that sleep into a single pause
    of about 20 s, so 5 s and 20 s pauses have to be counted together.

Run it against the released logs and both predictions come out exact:

    groupA (continuous, Jul)   recorded  99   inferred  99
    groupA (continuous, Aug)   recorded 108   inferred 108
    groupB (60/15, Jul)        recorded  41   inferred  41  (32 + 9)
    groupB (60/15, Aug)        recorded   3   inferred   3  (2 + 1)

which is the basis for reporting 27 for the field session rather than the 20
that counting 5 s pauses alone would give.

Usage:
    python reproduce/field_log_intervals.py
    python reproduce/field_log_intervals.py --field data/field_test.csv
"""

import argparse
import csv
import glob
import os
from pathlib import Path

SESSION_GAP_S = 60.0     # sessions are separated by gaps longer than this
BANDS = {"5s": (5.0, 5.6), "15s": (15.0, 15.6), "20s": (19.8, 20.6)}


def timestamps(path):
    """Seconds since midnight for every parseable row."""
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
    return out


def rows(path):
    """Rows with a usable timestamp, plus a count of those without.

    The released log carries one row whose Timestamp field was written as a run
    of null bytes - a second logging artefact, distinct from the 1.05e8 ms
    latency value that Section 5.1 also excludes. It is reported rather than
    silently dropped, so that the sample count here reconciles with the 46,576
    quoted in the paper.
    """
    good, bad = [], 0
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if ":" in (r.get("Timestamp") or ""):
                good.append(r)
            else:
                bad += 1
    return good, bad


def bands(gaps):
    return {k: sum(1 for g in gaps if lo < g < hi) for k, (lo, hi) in BANDS.items()}


def recorded_cutoffs(run_dir):
    """Cut-off events as the engine itself logged them, where an event log exists."""
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


def calibrate(roots):
    print("Calibration against the laboratory runs")
    print("-" * 72)
    print(f"  {'run':<34}{'recorded':>9}{'5s':>6}{'20s':>6}{'inferred':>10}  ok")
    ok = True
    for root in roots:
        for d in sorted(glob.glob(os.path.join(root, "group*"))):
            dat = glob.glob(os.path.join(d, "*_data.csv"))
            rec = recorded_cutoffs(d)
            if not dat or rec is None:
                continue
            ts = timestamps(dat[0])
            g = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
            b = bands(g)
            inferred = b["5s"] + b["20s"]
            good = inferred == rec
            ok &= good
            label = os.path.basename(d)[:20] + "  " + os.path.basename(root)[-7:]
            print(f"  {label:<34}{rec:>9}{b['5s']:>6}{b['20s']:>6}{inferred:>10}  "
                  f"{'yes' if good else 'NO'}")
    print()
    print("  The rule reproduces every recorded count."
          if ok else
          "  The rule does NOT reproduce every recorded count - do not apply it "
          "to the field log.")
    return ok


def field(path):
    ts = timestamps(path)
    rr, corrupt = rows(path)
    g = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
    # A run boundary is any discontinuity in the timestamp series: a forward gap
    # longer than SESSION_GAP_S, or a step backwards. The released file is a
    # concatenation of logging runs whose wall clocks overlap, so it steps back
    # three times (-80.4 s, -111.9 s, -242.8 s). Splitting on forward gaps alone
    # merges those runs, which overstates a run's duration and dilutes its
    # throttling fraction with samples taken before the device warmed up.
    cuts = [i for i, x in enumerate(g) if x > SESSION_GAP_S or x < 0]
    spans = ([(0, cuts[0])] +
             [(cuts[k] + 1, cuts[k + 1]) for k in range(len(cuts) - 1)] +
             [(cuts[-1] + 1, len(ts) - 1)]) if cuts else [(0, len(ts) - 1)]

    print(f"\nField log: {path}")
    print(f"  {len(rr) + corrupt} rows in the file; {corrupt} with an unusable "
          f"timestamp field, leaving {len(rr)} for interval analysis")
    print(f"  {(ts[-1] - ts[0]) / 3600:.2f} h wall clock")
    print(f"  runs split at timestamp discontinuities "
          f"(forward gap > {SESSION_GAP_S:.0f} s, or a step backwards) "
          f"-> {len(spans)}")
    print("-" * 72)
    for a, z in spans:
        sub = rr[a:z + 1]
        dur = ts[z] - ts[a]
        b = bands(g[a:z])
        cut = b["5s"] + b["20s"]
        thr = sum(1 for r in sub if (r.get("Throttled") or "").strip() == "Yes")
        temps = [float(r["CPU_Temp_C"]) for r in sub if r.get("CPU_Temp_C")]
        print(f"  n = {len(sub):>6}   {dur / 3600:>5.2f} h   "
              f"throttled {thr / len(sub) * 100:>5.1f}%   peak {max(temps):>5.1f} C")
        print(f"      pauses  5s = {b['5s']:<4} 15s = {b['15s']:<4} 20s = {b['20s']}")
        if b["15s"] > 5:
            med = median_spacing(g, a, z)
            print(f"      15 s pauses recur every {med:.1f} s "
                  "(60 s active + 15 s sleep = 75 s -> configuration B)")
        print(f"      inferred cut-offs = {cut}"
              + (f"   downtime {cut * 5 / 60:.1f} min = "
                 f"{cut * 5 / dur * 100:.2f}% of the session" if cut else ""))
        print()


def median_spacing(g, a, z):
    idx = [i for i in range(a, z) if BANDS["15s"][0] < g[i] < BANDS["15s"][1]]
    if len(idx) < 2:
        return float("nan")
    # spacing measured in elapsed samples is meaningless; use elapsed time
    gaps = []
    for k in range(len(idx) - 1):
        gaps.append(sum(g[idx[k]:idx[k + 1]]))
    gaps = sorted(x for x in gaps if x < 600)
    return gaps[len(gaps) // 2] if gaps else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="data/field_test.csv")
    ap.add_argument("--round1", default="data/thermal_telemetry")
    ap.add_argument("--round2", default="data/thermal_telemetry_aug2026")
    a = ap.parse_args()

    roots = [r for r in (a.round1, a.round2) if Path(r).exists()]
    if roots:
        calibrate(roots)
    else:
        print("No laboratory telemetry found; the field count below is "
              "uncalibrated and should not be reported.")

    if not Path(a.field).exists():
        raise SystemExit(f"field log not found: {a.field}")
    field(a.field)


if __name__ == "__main__":
    main()

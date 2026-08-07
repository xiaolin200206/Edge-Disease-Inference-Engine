#!/usr/bin/env python3
"""
analyse_power.py - reproduce Table 4 from the third round's logs.

The table has two columns that are not read directly off the sampler and are
worth stating the derivation of.

ACTIVE POWER is recovered rather than measured, and only the separation of the
continuous configuration from the rest should be read from it: each value is
derived from that configuration's own idle estimate, and those estimates span
2.17-2.41 W, which is wider than the spread among B to E.

The sampler reports total SoC power continuously, including the sleep intervals,
so the mean over a run is a duty-weighted mixture of the active and idle draws.
Given the duty fraction and the idle floor measured before the detector starts,
the active draw follows as (mean - (1 - duty) * idle) / duty.

ENERGY PER INFERENCE is the quantity a deployment is planned against, and it
moves in the opposite direction to mean power. Duty cycling reduces both the
power drawn and the number of inferences performed; the ratio is what decides
whether the schedule is free. It is not - the idle floor is paid throughout the
sleep interval and produces nothing, so energy per inference rises with the
sleep length even as mean power falls.

Usage, from the repository root:
    python reproduce/analyse_power.py
    python reproduce/analyse_power.py --dir data/power_round3
"""

import argparse
import csv
import glob
import os
import statistics as st

DUTY = {"A": 1.0, "B": 60 / 75, "C": 60 / 90, "D": 60 / 105, "E": 60 / 120}
LABEL = {"A": "continuous", "B": "60 s / 15 s", "C": "60 s / 30 s",
         "D": "60 s / 45 s", "E": "60 s / 60 s"}
IDLE_SAMPLES = 25          # 50 s at the 2 s interval, before the detector starts


def tukey_median(xs):
    xs = sorted(xs)
    n = len(xs)
    q1, q3 = xs[n // 4], xs[3 * n // 4]
    iqr = q3 - q1
    keep = [x for x in xs if q1 - 1.5 * iqr <= x <= q3 + 1.5 * iqr]
    return st.median(keep), len(xs) - len(keep)


def load_power(path):
    W, T, E = [], [], []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                W.append(float(r["Total_W"]))
                T.append(float(r["CPU_Temp_C"]))
                E.append(float(r["Elapsed_s"]))
            except (ValueError, KeyError):
                continue
    return W, T, E


def load_latency(d, key):
    hits = glob.glob(os.path.join(d, f"group{key}_*", "*_data.csv"))
    if not hits:
        return None, None
    lat = []
    cuts = 0
    with open(hits[0], newline="") as f:
        for r in csv.DictReader(f):
            try:
                v = float(r["Latency_ms"])
                if v < 1e6:
                    lat.append(v)
            except (ValueError, KeyError):
                continue
    ev = glob.glob(os.path.join(d, f"group{key}_*", "*_events.csv"))
    if ev:
        with open(ev[0], newline="") as f:
            cuts = sum(1 for r in csv.DictReader(f)
                       if r.get("Event") == "THERMAL_CUTOFF")
    return lat, cuts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/power_round3")
    a = ap.parse_args()

    print("Table 4 - SoC power by duty-cycle configuration\n")
    print(f"  {'Config':<16}{'Duty':>6}{'Mean W':>9}{'Idle W':>8}"
          f"{'Active W':>10}{'Inf/h':>9}{'J/inf':>8}{'Peak C':>8}")
    rows = []
    for key in "ABCDE":
        hits = glob.glob(os.path.join(a.dir, f"group{key}_*_power.csv"))
        if not hits:
            print(f"  {key}: no power log found")
            continue
        W, T, E = load_power(hits[0])
        duty = DUTY[key]
        idle = st.mean(W[:IDLE_SAMPLES])
        mean = st.mean(W)
        active = (mean - (1 - duty) * idle) / duty
        lat, cuts = load_latency(a.dir, key)
        med, _ = tukey_median(lat)
        inf_h = duty * 3600 / (med / 1000)
        j = mean * 3600 / inf_h
        rows.append((key, duty, mean, idle, active, inf_h, j, max(T), cuts,
                     (E[-1] - E[0]) / 3600, len(W)))
        print(f"  {key + ' ' + LABEL[key]:<16}{duty:>6.3f}{mean:>9.3f}{idle:>8.3f}"
              f"{active:>10.3f}{inf_h:>9.0f}{j:>8.3f}{max(T):>8.1f}")

    if len(rows) != 5:
        return

    a_, e_ = rows[0], rows[-1]
    print(f"\n  Mean power falls {100 * (1 - e_[2] / a_[2]):.1f}% from A to E")
    print(f"  Energy per inference rises {100 * (e_[6] / a_[6] - 1):.1f}% "
          f"({a_[6]:.3f} -> {e_[6]:.3f} J)")
    print(f"  Active power: A {a_[4]:.2f} W against "
          f"{min(r[4] for r in rows[1:]):.2f}-{max(r[4] for r in rows[1:]):.2f} W "
          "for B-E")
    print("    A draws less while active than any other configuration despite")
    print("    doing the most work per unit time. The within-round thermal record")
    print("    below is consistent with clock reduction as the cause; absolute")
    print("    temperatures are not comparable with the other two rounds.")

    print(f"\n  Run lengths: {min(r[9] for r in rows):.2f}-{max(r[9] for r in rows):.2f} h, "
          f"{min(r[10] for r in rows)}-{max(r[10] for r in rows)} samples each")

    print("\n  Thermal record of this round. The ordering is used in Section 4.4;")
    print("  the absolute values are excluded from Table 3:")
    for r in rows:
        print(f"    {r[0]}  peak {r[7]:.1f} C   cut-offs {r[8]}")
    print("    See README.md in the data directory for why: the pack was")
    print("    charging, the sampler was attached, and ambient was uncontrolled.")


if __name__ == "__main__":
    main()

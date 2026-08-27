#!/usr/bin/env python3
"""
c11_coverage_model.py - Reviewer 1, revision 2, Comment 11.

Monitoring coverage is currently reported as a time fraction and then converted
informally into "trees passed unexamined". The reviewer asks for the conversion
to be defined mathematically, its assumptions stated, and its sensitivity to
walking speed and inspection behaviour quantified. This script does that.

Model
-----
Over a round of duration T_run the node is in one of three states:

    active and inferring      t_inf = duty * T_run - D
    sleeping (by schedule)    t_slp = (1 - duty) * T_run
    suspended (thermal)       D     = N * tau

with duty = T_a / (T_a + T_s), N the number of thermal cut-off events and
tau = 5 s the suspension each one imposes.

    Nominal coverage    C_nom = duty
    Effective coverage  C_eff = (duty * T_run - D) / T_run = duty - D / T_run

The conversion to trees rests on three assumptions, which are the substance of
the reviewer's point and are made explicit here:

  A1  The operator walks at constant speed v along rows of uniform tree spacing
      s, so trees are encountered at a constant rate r = v / s.
  A2  A tree is examined if and only if it is within the camera's field of view
      while the node is inferring; the field of view is wide enough that
      framing is not the limiting factor, and dwell time per tree exceeds the
      confirmation window.
  A3  The operator does not pause during inactive periods. This is relaxed
      below by a parameter p, the fraction of inactive time during which the
      operator is stationary; p = 0 is the assumption used in the manuscript,
      p = 1 is an operator who pauses whenever the node is not inferring.

Under A1-A3 the trees passed unexamined over a round are

    U(v, s, p) = (v / s) * (1 - p) * (T_run - t_inf)
               = (v / s) * (1 - p) * (1 - C_eff) * T_run

and those attributable specifically to thermal suspension are

    U_thermal(v, s, p) = (v / s) * (1 - p) * D.

A2 is the assumption most likely to fail in practice and is not testable from
the released logs; it is stated rather than defended.

If a tree is passed k times in a round and examination opportunities are
treated as independent, the probability that it is never examined is
(1 - C_eff)^k, which is reported as a revisit sensitivity.

Usage:  python c11_coverage_model.py
"""

import argparse
import json

import numpy as np

TAU = 5.0  # seconds of suspension per thermal cut-off event

# (label, duty, T_run_s, N_cutoffs)  - Table 3, round 1, and the field session
SCENARIOS = [
    ("A continuous (lab, Jul)", 1.000, 3.04 * 3600, 99),
    ("A continuous (lab, Aug)", 1.000, 3.00 * 3600, 108),
    ("B 60/15 (lab, Jul)",      0.800, 3.00 * 3600, 41),
    ("B 60/15 (lab, Aug)",      0.800, 3.00 * 3600, 3),
    ("B 60/15 (field, 2.25 h)", 0.800, 2.25 * 3600, 27),
    ("C 60/30 (lab, both)",     0.667, 3.00 * 3600, 0),
    ("D 60/45 (lab, both)",     0.571, 3.00 * 3600, 0),
    ("E 60/60 (lab, both)",     0.500, 2.99 * 3600, 0),
]

# Walking speed, m/s: slow inspection pace to brisk walk.
SPEEDS = [0.3, 0.6, 0.9, 1.2, 1.4]
# Durian spacing, m: mature orchards are commonly planted 8-12 m apart.
SPACINGS = [8.0, 10.0, 12.0]


def coverage(duty, T_run, N):
    D = N * TAU
    t_inf = duty * T_run - D
    return {
        "duty": duty,
        "T_run_h": round(T_run / 3600, 2),
        "n_cutoffs": N,
        "downtime_s": round(D, 1),
        "downtime_min": round(D / 60, 2),
        "C_nominal": round(duty, 4),
        "C_effective": round(t_inf / T_run, 4),
        "coverage_lost_to_thermal_pp": round(100 * D / T_run, 2),
    }


def trees(duty, T_run, N, v, s, p=0.0):
    D = N * TAU
    t_inf = duty * T_run - D
    inactive = T_run - t_inf
    r = v / s
    return {
        "trees_encountered": r * T_run,
        "trees_unexamined_total": r * (1 - p) * inactive,
        "trees_unexamined_thermal_only": r * (1 - p) * D,
        "trees_per_cutoff_event": r * (1 - p) * TAU,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_c11.json")
    a = ap.parse_args()

    out = {"tau_s": TAU, "assumptions": ["A1 constant v and spacing s",
                                         "A2 examined iff in FoV while inferring",
                                         "A3 operator does not pause (p = 0)"]}

    out["coverage"] = [{"scenario": lab, **coverage(d, T, N)}
                       for lab, d, T, N in SCENARIOS]

    # ---- sensitivity of the per-event cost, which is the sentence at issue ---
    grid = []
    for s in SPACINGS:
        for v in SPEEDS:
            grid.append({"spacing_m": s, "speed_m_s": v,
                         "metres_per_5s_event": round(v * TAU, 1),
                         "trees_per_5s_event": round(v * TAU / s, 2)})
    out["per_event_cost_grid"] = grid
    out["per_event_cost_range"] = [
        round(min(g["trees_per_5s_event"] for g in grid), 2),
        round(max(g["trees_per_5s_event"] for g in grid), 2),
    ]

    # ---- the field session, in trees, over the grid --------------------------
    lab, duty, T, N = SCENARIOS[4]
    field = []
    for s in SPACINGS:
        for v in SPEEDS:
            t = trees(duty, T, N, v, s)
            field.append({
                "spacing_m": s, "speed_m_s": v,
                "trees_encountered": round(t["trees_encountered"], 0),
                "unexamined_thermal_only": round(t["trees_unexamined_thermal_only"], 1),
                "unexamined_total_incl_schedule": round(t["trees_unexamined_total"], 0),
            })
    out["field_session_trees"] = {"scenario": lab, "grid": field}

    # ---- pause behaviour -----------------------------------------------------
    out["pause_sensitivity"] = [
        {"p": p,
         "unexamined_thermal_only_at_1.2ms_10m":
             round(trees(duty, T, N, 1.2, 10.0, p)["trees_unexamined_thermal_only"], 1)}
        for p in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]

    # ---- revisit sensitivity -------------------------------------------------
    out["revisit_sensitivity"] = []
    for lab_, d_, T_, N_ in SCENARIOS:
        c = coverage(d_, T_, N_)["C_effective"]
        out["revisit_sensitivity"].append({
            "scenario": lab_, "C_effective": c,
            "P_never_examined_k1": round((1 - c) ** 1, 4),
            "P_never_examined_k2": round((1 - c) ** 2, 4),
            "P_never_examined_k3": round((1 - c) ** 3, 4),
        })

    # ---- the manuscript's "three trees" sentence, checked --------------------
    v_needed = 3 * 10.0 / TAU
    out["three_trees_check"] = {
        "claim": "a 5 s suspension costs the walking operator three trees",
        "implied_speed_at_10_m_spacing_m_s": round(v_needed, 1),
        "comment": "That is roughly 22 km/h, which is a sprint, not an "
                   "inspection walk. At a realistic 1.2 m/s and 10 m spacing a "
                   "5 s suspension costs 0.6 of a tree; the sentence should be "
                   "restated per round rather than per event.",
        "trees_per_event_at_1.2_m_s_10_m": round(1.2 * TAU / 10.0, 2),
        "field_round_thermal_cost_trees_at_1.2_m_s_10_m":
            round(trees(duty, T, N, 1.2, 10.0)["trees_unexamined_thermal_only"], 1),
    }

    print(json.dumps(out, indent=2))
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()

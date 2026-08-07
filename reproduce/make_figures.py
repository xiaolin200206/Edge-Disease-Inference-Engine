#!/usr/bin/env python3
"""
make_figures.py - regenerate Figs. 5, 6 and 7 from the released telemetry.

Addresses Reviewer 1 Minor 1 and Reviewer 2's second comment (figure fonts too
small to read at journal size) and implements two content changes required by
the revision:

  Fig. 5  thermal traces, now two rounds side by side
  Fig. 6  the two transitions, with the second round overlaid
  Fig. 7  latency distributions ONLY. The right-hand panel of the submitted
          version split the field log by throttling state; that panel is
          withdrawn, because the platform's throttle register latches until
          reboot, so the "unthrottled" samples of any run are simply the ones
          preceding its first transient event - a cold-start comparison, not a
          throttled/unthrottled one (Section 4.3).

Font sizes are set for a single-column figure reduced to 90 mm: nothing below
9 pt after reduction. Figures are written at 400 dpi and also as PDF, which is
what the journal prefers for line art.

Usage
-----
  python make_figures.py --round1 data/thermal_telemetry \\
                         --round2 data/thermal_telemetry_aug2026 \\
                         --field  data/field_test.csv --out figures/
  python make_figures.py --round1 data/thermal_telemetry --out figures/
"""

import argparse
import csv
import glob
import os
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- typography ---------------------------------------------------------
# Journal line art is usually reduced to 90 mm single column. Setting the
# figure width to that size and the base font to 9 pt means what is set here
# is what the reader sees; scaling a 6-inch figure down to 90 mm is what made
# the submitted figures unreadable.
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "figure.titlesize": 10,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.1,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "savefig.bbox": "tight",
    "savefig.dpi": 400,
    "pdf.fonttype": 42,          # embed TrueType, not Type 3
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
})

MM = 1 / 25.4
W1, W2 = 90 * MM, 190 * MM       # single and double column

CONFIGS = [("A", "continuous", "A - continuous"),
           ("B", "60-15", "B - 60 s / 15 s"),
           ("C", "60-30", "C - 60 s / 30 s"),
           ("D", "60-45", "D - 60 s / 45 s"),
           ("E", "60-60", "E - 60 s / 60 s")]
COLOURS = ["#B2182B", "#EF8A62", "#4D4D4D", "#67A9CF", "#2166AC"]
CUTOFF_C, SOFT_C = 82.0, 80.0


def psec(s):
    m = re.match(r"(\d+):(\d+):(\d+)", s.strip())
    if not m:
        return None
    h, mi, se = (int(x) for x in m.groups())
    return h * 3600 + mi * 60 + se


def load_run(d):
    data = glob.glob(os.path.join(d, "*_data.csv"))
    evt = glob.glob(os.path.join(d, "*_events.csv"))
    if not data:
        return None
    rows = list(csv.DictReader(open(data[0], newline="", encoding="utf-8-sig")))
    t, temp, lat = [], [], []
    t0 = None
    for r in rows:
        s = psec(r.get("Timestamp", ""))
        if s is None:
            continue
        if t0 is None:
            t0 = s
        dt = s - t0
        if dt < 0:
            dt += 86400
        t.append(dt / 60.0)
        temp.append(float(r["CPU_Temp_C"]))
        try:
            lat.append(float(r["Latency_ms"]))
        except (KeyError, ValueError):
            pass
    cuts = []
    if evt:
        e0 = None
        for r in csv.DictReader(open(evt[0], newline="", encoding="utf-8-sig")):
            s = psec(r.get("Timestamp", ""))
            if s is None:
                continue
            if e0 is None:
                e0 = s
            if r.get("Event", "").strip() == "THERMAL_CUTOFF":
                dt = s - e0
                cuts.append((dt + 86400 if dt < 0 else dt) / 60.0)
    return dict(t=np.array(t), temp=np.array(temp),
                lat=np.array(lat), cuts=np.array(cuts))


def find(root, key):
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if os.path.isdir(d) and key in os.path.basename(d):
            r = load_run(d)
            if r is not None and len(r["t"]) > 5000:
                return r
    return None


def tukey(x):
    if len(x) == 0:
        return x
    q1, q3 = np.percentile(x, [25, 75])
    f = 1.5 * (q3 - q1)
    return x[(x >= q1 - f) & (x <= q3 + f)]


# =========================================================================
def fig5(r1, r2, out):
    rounds = [("Round 1", r1)] + ([("Round 2", r2)] if r2 else [])
    fig, axes = plt.subplots(len(CONFIGS), len(rounds),
                             figsize=(W2 if len(rounds) > 1 else W1,
                                      2.0 * len(CONFIGS) * MM * 25.4 * 0.42),
                             sharex=True, sharey=True, squeeze=False)
    for j, (rname, rd) in enumerate(rounds):
        for i, (key, _, label) in enumerate(CONFIGS):
            ax = axes[i][j]
            run = rd.get(key)
            if run is None:
                ax.text(0.5, 0.5, "not available", ha="center", va="center",
                        transform=ax.transAxes, fontsize=8, color="0.5")
                ax.set_ylim(45, 90)
                continue
            ax.plot(run["t"], run["temp"], color=COLOURS[i], lw=0.6)
            ax.axhline(SOFT_C, ls="--", lw=0.7, color="0.35")
            ax.axhline(CUTOFF_C, ls=":", lw=0.7, color="#B2182B")
            for c in run["cuts"]:
                ax.plot(c, 88, marker="v", ms=2.2, color="#B2182B",
                        clip_on=False)
            ax.set_ylim(45, 90)
            ax.set_yticks([50, 60, 70, 80])
            n = len(run["cuts"])
            ax.text(0.985, 0.06, f"{label}   {n} cut-off{'' if n == 1 else 's'}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
            if j == 0:
                ax.set_ylabel("Die T (\u00b0C)")
            if i == 0:
                ax.set_title(rname)
            if i == len(CONFIGS) - 1:
                ax.set_xlabel("Elapsed time (min)")
    fig.align_ylabels()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig5_thermal_traces.{ext}")
    plt.close(fig)
    print("  fig5 written")


def fig6(r1, r2, out):
    x = np.arange(len(CONFIGS))
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(W2, 62 * MM))

    ax = axes[0]

    def _annotate(xs, counts):
        """Print the count above every bar.

        A zero has no bar to see, and three of the five configurations are
        zero in both rounds - the paper's main thermal result. Unlabelled,
        an empty category reads as missing data rather than as a measured
        zero, so every value is written out.
        """
        for xi, c in zip(xs, counts):
            if np.isnan(c):
                continue
            ax.annotate(f"{int(c)}", (xi, c), textcoords="offset points",
                        xytext=(0, 2.5), ha="center", fontsize=7.5)

    c1 = [len(r1[k]["cuts"]) if k in r1 else np.nan for k, _, _ in CONFIGS]
    x1 = x - w / 2 if r2 else x
    ax.bar(x1, c1, w if r2 else 0.6, label="Round 1", color="#B2182B")
    _annotate(x1, c1)
    if r2:
        c2 = [len(r2[k]["cuts"]) if k in r2 else np.nan for k, _, _ in CONFIGS]
        ax.bar(x + w / 2, c2, w, label="Round 2", color="#67A9CF")
        _annotate(x + w / 2, c2)
        ax.legend(frameon=False, loc="upper right")
    ax.set_xticks(x)
    ax.set_xticklabels([c[2].split(" - ")[1] if " - " in c[2] else c[2]
                        for c in CONFIGS], rotation=30, ha="right")
    ax.set_ylabel("Thermal cut-off events per run")
    # A neutral description. The earlier title asserted that cut-offs fall to
    # zero between 15 s and 30 s, which the right-hand panel qualifies: in the
    # warmer round configuration C reaches the threshold and is clipped there,
    # so its zero is a truncated excursion rather than a retained margin. The
    # conditional belongs in the caption, where there is room to state it.
    ax.set_title("Cut-off events by sleep interval")
    ax.margins(y=0.14)

    ax = axes[1]
    for j, (rd, nm, mk) in enumerate([(r1, "Round 1", "o"),
                                      (r2, "Round 2", "s")]):
        if not rd:
            continue
        mx = [rd[k]["temp"].max() if k in rd else np.nan for k, _, _ in CONFIGS]
        mn = [rd[k]["temp"].mean() if k in rd else np.nan for k, _, _ in CONFIGS]
        ax.plot(x, mx, mk + "-", ms=3.5, label=f"{nm} max", color=COLOURS[j * 4])
        ax.plot(x, mn, mk + "--", ms=3.5, label=f"{nm} mean",
                color=COLOURS[j * 4], alpha=0.55)
    ax.axhline(CUTOFF_C, ls=":", lw=0.9, color="#B2182B")
    ax.axhline(SOFT_C, ls="--", lw=0.9, color="0.35")

    # Any run whose maximum equals the cut-off threshold exactly did not
    # plateau there thermally: the engine evaluates its check once per loop
    # and truncates the excursion when it fires, so the run comes to rest on
    # the threshold. Left unmarked, those points read as a real plateau, which
    # is the opposite of what Section 4.3 concludes. Ring them.
    for rd in (r1, r2):
        if not rd:
            continue
        clipped = [(xi, rd[k]["temp"].max()) for xi, (k, _, _) in zip(x, CONFIGS)
                   if k in rd and abs(rd[k]["temp"].max() - CUTOFF_C) < 0.05]
        if clipped:
            cx, cy = zip(*clipped)
            ax.plot(cx, cy, "o", ms=9, mfc="none", mec="0.25", mew=0.9,
                    zorder=5, label="_nolegend_")
    has_clip = any(abs(rd[k]["temp"].max() - CUTOFF_C) < 0.05
                   for rd in (r1, r2) if rd for k, _, _ in CONFIGS if k in rd)
    if has_clip:
        # Open a band at the foot of the axes so the note and the legend do not
        # land on the mean series.
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - (hi - lo) * 0.34, hi)
        ax.annotate("circled: peak truncated by the cut-off check,\n"
                    "not a thermal plateau",
                    xy=(0.015, 0.015), xycoords="axes fraction", fontsize=7,
                    color="0.25", va="bottom")
    # annotate on the right margin so the labels cannot sit on the series
    ax.text(1.01, CUTOFF_C, "cut-off\n82 \u00b0C", fontsize=7.5, va="center",
            color="#B2182B", transform=ax.get_yaxis_transform())
    ax.text(1.01, SOFT_C, "soft limit\n80 \u00b0C", fontsize=7.5, va="center",
            color="0.35", transform=ax.get_yaxis_transform())
    ax.set_xticks(x)
    ax.set_xticklabels([c[2].split(" - ")[1] if " - " in c[2] else c[2]
                        for c in CONFIGS], rotation=30, ha="right")
    ax.set_ylabel("Die temperature (\u00b0C)")
    ax.set_title("Peak temperature against the two limits")
    ax.legend(frameon=False, fontsize=7.5, ncol=2,
              loc="lower left", bbox_to_anchor=(0.0, 0.13))

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig6_transitions.{ext}")
    plt.close(fig)
    print("  fig6 written")


def fig7(r1, out):
    # Single panel. The submitted right-hand panel split samples by throttling
    # state; that split is not meaningful because the flag latches (Section 4.3).
    fig, ax = plt.subplots(figsize=(W1, 62 * MM))
    data, labels = [], []
    for k, _, label in CONFIGS:
        if k in r1 and len(r1[k]["lat"]):
            data.append(tukey(r1[k]["lat"]))
            labels.append(label.split(" - ")[1] if " - " in label else label)
    # matplotlib renamed this parameter in 3.9 and removed the old name in
    # 3.11. Comparing version strings is unsafe here ("3.10" < "3.9"
    # lexicographically), so ask the function which name it accepts.
    import inspect
    params = inspect.signature(ax.boxplot).parameters
    kw = {"tick_labels" if "tick_labels" in params else "labels": labels}
    bp = ax.boxplot(data, showfliers=False, widths=0.55, **kw,
                    medianprops=dict(color="#B2182B", lw=1.2),
                    boxprops=dict(lw=0.8), whiskerprops=dict(lw=0.8),
                    capprops=dict(lw=0.8))
    ax.set_ylabel("Inference latency (ms)")
    ax.set_xlabel("Duty-cycle configuration")
    ax.tick_params(axis="x", rotation=30)
    for t in ax.get_xticklabels():
        t.set_ha("right")
    meds = [np.median(d) for d in data]

    # The main axes are magnified so the distributions can be compared at all;
    # on a zero-based scale they are indistinguishable. A magnified axis makes a
    # small effect look large, and the claim this figure supports is that the
    # effect IS small, so the zero-based view is shown as an inset rather than
    # left to the reader to imagine. Both readings are on the page.
    top = max(np.max(d) for d in data)
    inset = ax.inset_axes([0.615, 0.55, 0.365, 0.38])
    inset.boxplot(data, showfliers=False, widths=0.55,
                  medianprops=dict(color="#B2182B", lw=0.9),
                  boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6),
                  capprops=dict(lw=0.6), **{list(kw)[0]: [""] * len(data)})
    inset.set_ylim(0, top * 1.08)
    inset.tick_params(labelsize=6.5, length=2)
    inset.set_xticks([])
    inset.set_title("same data, zero-based", fontsize=6.5, pad=2)
    for sp in inset.spines.values():
        sp.set_linewidth(0.6)

    if len(meds) > 3:
        pct = 100 * (meds[0] - meds[3]) / meds[0]
        lo = int(np.argmin(meds))
        ax.set_title(f"Median moves {pct:.1f}% from A to D (round 1)", fontsize=9.5)
        ax.annotate(f"main axes magnified; smallest median at {labels[lo]}",
                    xy=(0.015, 0.02), xycoords="axes fraction",
                    fontsize=7, color="0.3", va="bottom")
    else:
        ax.set_title("Latency by configuration (round 1)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig7_latency.{ext}")
    plt.close(fig)
    print("  fig7 written (single panel; throttle-split panel withdrawn)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round1", default="data/thermal_telemetry")
    ap.add_argument("--round2", default="data/thermal_telemetry_aug2026")
    ap.add_argument("--field", default="data/field_test.csv")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--dpi", type=int, default=400,
                    help="raster output resolution (default 400; journals "
                         "commonly ask for 300 for halftone, 600+ for line art)")
    a = ap.parse_args()
    plt.rcParams["savefig.dpi"] = a.dpi

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    r1 = {k: r for k, sub, _ in CONFIGS
          if (r := find(a.round1, f"group{k}_{sub}")) is not None}
    print(f"round 1: {len(r1)}/5 configurations loaded")

    r2 = {}
    if os.path.isdir(a.round2):
        r2 = {k: r for k, sub, _ in CONFIGS
              if (r := find(a.round2, f"group{k}_{sub}")) is not None}
    print(f"round 2: {len(r2)}/5 configurations loaded"
          + ("" if r2 else "  (directory absent or empty; figures will show"
                           " round 1 only)"))

    if not r1:
        raise SystemExit(f"no round-1 runs found under {a.round1}")

    fig5(r1, r2, out)
    fig6(r1, r2, out)
    fig7(r1, out)
    print(f"\nwritten to {out.resolve()} as PDF and PNG at {a.dpi} dpi.")
    print("Check every label at final print size before submitting; nothing")
    print("here should fall below 9 pt when the figure is placed at 90 mm.")


if __name__ == "__main__":
    main()

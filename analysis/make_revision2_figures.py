#!/usr/bin/env python3
"""
make_revision2_figures.py - the five new figures required by revision 2.

  fig8_dataset_flow          Comment 5   source photographs -> corrected partition
  fig9_phash_evidence        Comment 6   threshold-selection evidence
  fig10_power_boundary       Comment 10  what the SoC measurement includes
  fig11_field_gap_structure  Comment 12  the pause structure the 27 rests on
  fig12_coverage_sensitivity Comment 11  coverage -> trees, over the assumption grid

Typography follows reproduce/make_figures.py: sized for a 90 mm single column,
nothing below 9 pt after reduction, 400 dpi PNG and PDF.

Usage:
    python make_revision2_figures.py --repo <repo root> --out <dir>
"""

import argparse
import csv
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8,
    "figure.titlesize": 10, "axes.linewidth": 0.8, "lines.linewidth": 1.1,
    "savefig.bbox": "tight", "savefig.dpi": 400,
})

MM = 1 / 25.4
COL = 90 * MM
DBL = 190 * MM
INK = "#1a1a1a"
ACC = "#b2182b"
MUT = "#6b6b6b"


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote {name}.png / .pdf")


# ------------------------------------------------------------------ fig 8 ---
def fig_dataset_flow(out):
    fig, ax = plt.subplots(figsize=(DBL, 96 * MM))
    ax.set_xlim(0, 100); ax.set_ylim(2, 58); ax.axis("off")

    def box(x, y, w, h, title, lines, edge=INK, face="white", lw=0.9):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.4,rounding_size=0.8",
                                    linewidth=lw, edgecolor=edge, facecolor=face))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top",
                fontsize=8.4, fontweight="bold", color=INK)
        for i, ln in enumerate(lines):
            ax.text(x + w / 2, y + h - 5.6 - i * 2.7, ln, ha="center", va="top",
                    fontsize=7.6, color=MUT)

    def arrow(x1, y1, x2, y2, label=None, color=INK, dy=0.8):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=9, linewidth=0.9, color=color,
                                     shrinkA=1, shrinkB=1))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, label, ha="center",
                    va="bottom", fontsize=7.4, color=color)

    W, H = 27.0, 15.0
    # row 1
    box(1, 41, W, H, "Source photographs",
        ["~750 distinct captures", "recovered by export stem,", "MD5 and pHash d \u2264 2"])
    box(36.5, 41, W, H, "Exported images",
        ["1,121 images", "augmented variants added", "by the annotation platform"])
    box(72, 41, W, H, "Original partition",
        ["995 train / 126 validation", "154 validation instances"])
    # row 2
    box(72, 21, W, H, "Leakage detected",
        ["1 exact (MD5) + 117 near", "(pHash < 10) = 118 pairs", "\u2192 22 straddling components"])
    box(36.5, 21, W, H, "Table 1 re-evaluation",
        ["32 validation images removed", "model and weights fixed", "mAP@0.5  0.402 \u2192 0.367"])
    box(1, 21, W, H, "Corrected partition",
        ["components assigned wholly", "to one side; 125 val instances", "Leaf_rot  0.000 \u2192 0.399"])

    arrow(28, 48.5, 36.5, 48.5, "augment")
    arrow(63.5, 48.5, 72, 48.5, "split")
    arrow(85.5, 41, 85.5, 36.2, None, ACC)
    arrow(72, 28.5, 63.5, 28.5, None, ACC)
    arrow(36.5, 28.5, 28, 28.5, None, ACC)

    ax.text(84.0, 38.6, "audit", ha="right", va="center", fontsize=7.4, color=ACC)
    ax.text(67.75, 30.2, "remove", ha="center", va="bottom", fontsize=7.4, color=ACC)
    ax.text(32.25, 30.2, "regroup", ha="center", va="bottom", fontsize=7.4, color=ACC)
    ax.text(50, 17.2, "augmentation-before-split fault", ha="center", va="center",
            fontsize=7.6, color=ACC, style="italic")
    ax.text(50, 8.5,
            "Augmentation precedes the split, so variants of one source photograph reach both partitions.\n"
            "The two lower-left boxes answer different questions: what the leak cost the published figure\n"
            "(weights held fixed, validation sample reduced) and what the architecture attains under a\n"
            "correct partition (both partitions rebuilt, so the two are not a before-and-after pair).",
            ha="center", va="center", fontsize=7.8, color=INK)
    save(fig, out, "fig8_dataset_flow")


# ------------------------------------------------------------------ fig 9 ---
def fig_phash(repo, out):
    rows = list(csv.DictReader(open(os.path.join(repo, "audit/leakage_report.csv"))))
    d = np.array([int(r["hamming_distance"]) for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(DBL, 62 * MM))

    a = axes[0]
    vals, counts = np.unique(d, return_counts=True)
    a.bar(vals, counts, width=1.4, color="#4d4d4d", edgecolor="white", linewidth=0.6)
    a.axvline(10, color=ACC, linestyle="--", linewidth=1.0)
    a.text(10.2, counts.max() * 0.92, "detection\nthreshold 10", fontsize=7.6, color=ACC)
    a.axvline(2, color="#2166ac", linestyle=":", linewidth=1.0)
    a.text(2.3, counts.max() * 0.62, "grouping\nthreshold 2", fontsize=7.6, color="#2166ac")
    a.set_xlabel("Perceptual Hamming distance")
    a.set_ylabel("Cross-partition pairs")
    a.set_title("(a) Detected pairs", loc="left")
    a.set_xlim(-1.5, 13)
    a.spines[["top", "right"]].set_visible(False)

    b = axes[1]
    th = [0, 2, 4, 6, 8, 10]
    val_imgs = [20, 25, 27, 29, 32, 32]
    b.step(th, val_imgs, where="post", color=INK)
    b.plot(th, val_imgs, "o", color=INK, markersize=3.5)
    b.axhline(32, color=MUT, linewidth=0.7, linestyle=":")
    b.set_xlabel("Threshold (Hamming)")
    b.set_ylabel("Validation images implicated")
    b.set_title("(b) What Table 1 removes", loc="left")
    b.set_ylim(0, 38)
    b.spines[["top", "right"]].set_visible(False)
    b.text(2.2, 8.5, "20 images implicated even at\nexact perceptual identity (d = 0)", fontsize=7.4, color=MUT)

    c = axes[2]
    th2 = [2, 4, 6, 10]
    comps = [750, 743, 732, 723]
    strad = [22, 20, 15, 11]
    c.plot(th2, comps, "o-", color=INK, markersize=3.5, label="source components")
    c.set_xlabel("Threshold (Hamming)")
    c.set_ylabel("Source components", color=INK)
    c.set_ylim(700, 770)
    c2 = c.twinx()
    c2.plot(th2, strad, "s--", color=ACC, markersize=3.5, label="straddling")
    c2.set_ylabel("Straddling components", color=ACC)
    c2.set_ylim(0, 28)
    c2.tick_params(axis="y", colors=ACC)
    c.set_title("(c) Threshold sweep", loc="left")
    c.spines[["top"]].set_visible(False)
    c2.spines[["top"]].set_visible(False)

    fig.tight_layout()
    save(fig, out, "fig9_phash_evidence")


# ----------------------------------------------------------------- fig 10 ---
def fig_power_boundary(out):
    fig, ax = plt.subplots(figsize=(DBL * 0.82, 92 * MM))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    ax.add_patch(FancyBboxPatch((3, 22), 94, 70,
                                boxstyle="round,pad=0.5,rounding_size=1.2",
                                linewidth=0.9, edgecolor=MUT, facecolor="#fafafa"))
    ax.text(6, 88, "System — battery rail, not instrumented", fontsize=8.4,
            color=MUT, va="center")

    ax.add_patch(FancyBboxPatch((7, 28), 46, 52,
                                boxstyle="round,pad=0.5,rounding_size=1.2",
                                linewidth=1.5, edgecolor=ACC, facecolor="white"))
    ax.text(30, 75, "SoC domain — measured", fontsize=8.8, fontweight="bold",
            ha="center", color=ACC)
    ax.text(30, 68.5, "vcgencmd pmic_read_adc, 2 s interval", fontsize=7.8,
            ha="center", color=INK)
    ax.text(30, 64.0, "12 rails reporting V and I, summed", fontsize=7.8,
            ha="center", color=INK)
    for i, lab in enumerate(["cores (VDD_CORE)",
                             "memory (DDR_VDD2, DDR_VDDQ)",
                             "SoC rails (1V1, 1V8, 3V3)"]):
        ax.text(30, 55.5 - i * 5.5, "\u00b7 " + lab, fontsize=7.6, ha="center",
                color=MUT)
    ax.text(30, 34.0, "5,445 samples per run, 3 h", fontsize=7.4, ha="center",
            color=MUT, style="italic")

    ax.text(76, 84, "excluded", fontsize=8.6, ha="center", color=MUT,
            fontweight="bold")
    for i, (lab, note) in enumerate([
            ("USB camera", "enumerated throughout"),
            ("Active Cooler", "temperature-driven"),
            ("Picamera 2 / CSI", "field build only"),
            ("power board, charging", "third round only")]):
        y = 71 - i * 12.0
        ax.add_patch(FancyBboxPatch((59, y - 4.4), 34, 8.8,
                                    boxstyle="round,pad=0.4,rounding_size=0.8",
                                    linewidth=0.8, edgecolor=MUT,
                                    facecolor="white", linestyle="--"))
        ax.text(76, y + 1.6, lab, fontsize=7.9, ha="center", color=INK)
        ax.text(76, y - 2.4, note, fontsize=7.0, ha="center", color=MUT)

    ax.text(50, 12,
            "Table 6 reports SoC power and SoC energy per inference.\n"
            "EXT5V is logged as a supply-stability check and is not summed; BATT is not logged.\n"
            "Battery-rail instrumentation, and therefore system-level energy, remains future work.",
            fontsize=7.8, ha="center", va="center", color=INK)
    save(fig, out, "fig10_power_boundary")


# ----------------------------------------------------------------- fig 11 ---
def fig_field_gaps(repo, out):
    path = os.path.join(repo, "data/field_test.csv")
    ts = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            s = (r.get("Timestamp") or "").strip()
            if ":" not in s:
                continue
            h, m, rest = s.split(":")
            sec, _, ms = rest.partition(".")
            ts.append(int(h) * 3600 + int(m) * 60 + int(sec) + (int(ms) / 1000 if ms else 0))
    ts = np.array(ts)
    g = np.diff(ts)
    cuts = [i for i, x in enumerate(g) if x > 60 or x < 0]
    spans = ([(0, cuts[0])] + [(cuts[k] + 1, cuts[k + 1]) for k in range(len(cuts) - 1)]
             + [(cuts[-1] + 1, len(ts) - 1)])
    a, z = max(spans, key=lambda s: s[1] - s[0])
    sub = g[a:z]
    big = sub[sub > 1.0]

    fig, axes = plt.subplots(1, 2, figsize=(DBL, 66 * MM))

    ax = axes[0]
    ax.hist(big, bins=np.arange(0, 24, 0.25), color="#4d4d4d")
    ax.axvspan(3, 14, color="#2166ac", alpha=0.10)
    ax.axvspan(16, 22, color=ACC, alpha=0.10)
    ax.set_yscale("log")
    ax.set_ylim(0.6, 4000)
    ax.set_xlim(0, 23)
    ax.set_xlabel("Inter-sample gap (s)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title("(a) Pause structure, 2.25 h field session", loc="left")
    ax.text(5.2, 400, "5.18–5.29 s\ncut-off retry\nn = 20", ha="center", va="top",
            fontsize=7.6, color="#2166ac")
    ax.text(11.6, 2400, "15.20–15.35 s\nscheduled sleep, n = 100", ha="center",
            va="top", fontsize=7.6, color=MUT)
    ax.text(21.4, 400, "20.20–20.26 s\nretry merged\nwith sleep\nn = 7", ha="right",
            va="top", fontsize=7.6, color=ACC)
    ax.text(9.0, 1.1, "no gap of any length between 5.6 s and 15.0 s",
            ha="center", fontsize=7.4, color=INK, style="italic")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    variants = [
        ("5 s only", int(((sub > 3) & (sub < 14)).sum()), MUT),
        ("as reported", 27, INK),
        ("widened",
         int(((sub > 3) & (sub < 14)).sum()) + int(((sub > 16) & (sub < 600)).sum()), INK),
        ("narrowed",
         int(((sub > 5.1) & (sub < 5.4)).sum()) + int(((sub > 20.1) & (sub < 20.4)).sum()), INK),
    ]
    xs = np.arange(len(variants))
    ax.bar(xs, [v[1] for v in variants], 0.55,
           color=[v[2] for v in variants], edgecolor="white")
    for x, v in zip(xs, variants):
        ax.text(x, v[1] + 0.9, str(v[1]), ha="center", fontsize=8.2, color=INK)
    ax.axhline(27, color=ACC, linewidth=0.9, linestyle="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([v[0] for v in variants], fontsize=7.8)
    ax.set_ylabel("Inferred cut-off events")
    ax.set_ylim(0, 33)
    ax.set_title("(b) Sensitivity of the inferred count", loc="left")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    save(fig, out, "fig11_field_gap_structure")


# ----------------------------------------------------------------- fig 12 ---
def fig_coverage(out):
    fig, axes = plt.subplots(1, 2, figsize=(DBL, 62 * MM))

    ax = axes[0]
    labels = ["A cont.", "B 60/15", "C 60/30", "D 60/45", "E 60/60"]
    nom = [100.0, 80.0, 66.7, 57.1, 50.0]
    eff = [95.5, 78.1, 66.7, 57.1, 50.0]
    x = np.arange(len(labels))
    ax.bar(x - 0.19, nom, 0.36, label="nominal", color="#c9c9c9", edgecolor="white")
    ax.bar(x + 0.19, eff, 0.36, label="effective", color=INK, edgecolor="white")
    for i, (n, e) in enumerate(zip(nom, eff)):
        if n - e > 0.05:
            ax.text(i + 0.19, e + 2, f"-{n - e:.1f} pp", ha="center",
                    fontsize=7.4, color=ACC)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Monitoring coverage (%)")
    ax.set_title("(a) Nominal against effective, round 1", loc="left")
    ax.set_ylim(0, 112)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    speeds = np.linspace(0.3, 1.4, 60)
    for s, style in ((8.0, ":"), (10.0, "-"), (12.0, "--")):
        trees = speeds / s * 135.0   # 27 events x 5 s of suspension
        ax.plot(speeds, trees, style, color=INK, label=f"spacing {s:.0f} m")
    ax.plot([1.2], [1.2 / 10 * 135], "o", color=ACC, markersize=4.5)
    ax.annotate("1.2 m s$^{-1}$, 10 m spacing:\n16 trees of ~970 passed",
                xy=(1.2, 1.2 / 10 * 135), xytext=(0.66, 5.2), fontsize=7.6,
                color=ACC, ha="left",
                arrowprops=dict(arrowstyle="->", lw=0.7, color=ACC))
    ax.set_xlabel("Walking speed (m s$^{-1}$)")
    ax.set_ylabel("Trees passed during suspension")
    ax.set_title("(b) Field round, thermal loss in trees", loc="left")
    ax.legend(frameon=False, loc="upper left", handlelength=2.4)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    save(fig, out, "fig12_coverage_sensitivity")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig_dataset_flow(a.out)
    fig_phash(a.repo, a.out)
    fig_power_boundary(a.out)
    fig_field_gaps(a.repo, a.out)
    fig_coverage(a.out)


if __name__ == "__main__":
    main()

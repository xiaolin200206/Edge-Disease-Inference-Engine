#!/usr/bin/env python3
"""
make_figures_2_3.py - redraw Figures 2 and 3 for the revision.

Both figures describe the same run: YOLOv8s under the pre-cleanup eight-class
taxonomy, in which `Early_Blight` and `early_blight` are separate indices.

Two problems in the versions supplied with the original submission are fixed
here.

FIGURE 3 - the axes were labelled the wrong way round. The plot was drawn with
predictions on the vertical axis and ground truth on the horizontal, but
labelled "True label" on the vertical and "Predicted label" on the horizontal.
Read as labelled, the Phomopsis row is empty, which would mean the validation
set holds no Phomopsis instances; it holds 54. Read correctly the column totals
reproduce the validation composition exactly - Algal_leave 58, Leaf_rot 26,
Phomopsis 54, Pink_Disease 2, early_blight 11 plus Early_Blight 2 = 13,
root_disease 1, total 154 - which is the figure Supplementary Table S5 reports.
The transposition matters for the paper's argument: as labelled, the detector
appears to emit 54 spurious Phomopsis boxes on background; in fact it emits
none at all and misses all 54 instances. Those are opposite failure modes.

FIGURE 2 - five of the eight classes score exactly 0.000, and a bar of zero
height is invisible. An absent bar reads as missing data, and three of those
zeros are the paper's central observation. Every value is printed.

Usage:
    python make_figures_2_3.py --out figures/
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

MM = 1 / 25.4

# Pre-cleanup eight-class taxonomy, in the index order the run used.
CLASSES = ["Algal_leave", "Early_Blight", "Leaf_rot", "Phomopsis",
           "Pink_Disease", "early_blight", "root_disease", "Anthracnose"]

# Per-class AP@0.5 for that run (Figure 2).
AP = {
    "Algal_leave":  0.87,
    "Early_Blight": 0.00,
    "Leaf_rot":     0.00,
    "Phomopsis":    0.00,
    "Pink_Disease": 0.00,
    "early_blight": 0.90,
    "root_disease": 0.00,
    "Anthracnose":  0.00,
}

# Validation instances per class, from Supplementary Table S5.
N_INST = {
    "Algal_leave": 58, "Early_Blight": 2, "Leaf_rot": 26, "Phomopsis": 54,
    "Pink_Disease": 2, "early_blight": 11, "root_disease": 1, "Anthracnose": 0,
}

# Confusion counts, entered as the plot was actually computed:
# rows = PREDICTED class, columns = TRUE class, with a background row/column.
# Verified against the validation composition - every column sums to N_INST.
ROWS = CLASSES + ["background"]
COLS = CLASSES + ["background"]
COUNTS = {
    ("Algal_leave", "Algal_leave"): 44, ("Algal_leave", "background"): 3,
    ("Leaf_rot", "Algal_leave"): 2, ("Leaf_rot", "Early_Blight"): 1,
    ("Leaf_rot", "background"): 7,
    ("early_blight", "Algal_leave"): 1, ("early_blight", "early_blight"): 11,
    ("early_blight", "background"): 1,
    ("background", "Algal_leave"): 11, ("background", "Early_Blight"): 1,
    ("background", "Leaf_rot"): 26, ("background", "Phomopsis"): 54,
    ("background", "Pink_Disease"): 2, ("background", "root_disease"): 1,
}


def matrix():
    m = np.zeros((len(ROWS), len(COLS)), dtype=int)
    for (r, c), v in COUNTS.items():
        m[ROWS.index(r), COLS.index(c)] = v
    return m


def check(m):
    """The column totals must reproduce the validation composition."""
    bad = []
    for j, c in enumerate(COLS[:-1]):
        got, want = int(m[:, j].sum()), N_INST[c]
        if got != want:
            bad.append(f"{c}: matrix {got}, Table S5 {want}")
    total = int(m[:, :-1].sum())
    if total != 154:
        bad.append(f"total {total}, Table S5 154")
    return bad


def fig2(out, dpi):
    fig, ax = plt.subplots(figsize=(150 * MM, 78 * MM))
    order = sorted(CLASSES, key=lambda c: (-AP[c], -N_INST[c]))
    y = np.arange(len(order))
    vals = [AP[c] for c in order]
    # Classes with too little validation support to interpret are drawn in grey.
    colours = ["#4C7A34" if N_INST[c] >= 10 else "#999999" for c in order]
    ax.barh(y, vals, color=colours, height=0.62)

    for yi, c in zip(y, order):
        v, n = AP[c], N_INST[c]
        note = f"{v:.2f}   (n = {n})" if n else f"{v:.2f}   (no validation instances)"
        ax.text(v + 0.012, yi, note, va="center", fontsize=8)

    ax.set_yticks(y)
    # A grey bar of zero height is as invisible as a green one, so the
    # low-support classes are marked on the label instead of by colour.
    ax.set_yticklabels([c + (" *" if N_INST[c] < 10 else "") for c in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("AP@0.5")
    ax.set_title("Per-class AP@0.5, YOLOv8s under the pre-cleanup eight-class taxonomy")
    ax.grid(axis="x", ls=":", lw=0.5, color="0.8")
    ax.set_axisbelow(True)

    # A zero bar is invisible; say so rather than leaving a blank row.
    ax.annotate("* fewer than 10 validation instances\n"
                "All eight values are printed; five are exactly 0.00 and have "
                "no bar to see.",
                xy=(0.0, -0.30), xycoords="axes fraction", ha="left",
                fontsize=7, color="0.35", va="top")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig2_per_class_ap.{ext}", dpi=dpi)
    plt.close(fig)


def fig3(out, dpi):
    m = matrix()
    fig, ax = plt.subplots(figsize=(150 * MM, 128 * MM))
    ax.imshow(m, cmap="Blues", vmin=0, vmax=m.max())

    for i in range(len(ROWS)):
        for j in range(len(COLS)):
            v = m[i, j]
            if v == 0:
                continue
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                    color="white" if v > m.max() * 0.55 else "#123",
                    fontweight="bold" if v > m.max() * 0.55 else "normal")

    ax.set_xticks(range(len(COLS)))
    ax.set_xticklabels(COLS, rotation=45, ha="right")
    ax.set_yticks(range(len(ROWS)))
    ax.set_yticklabels(ROWS)
    # The orientation the plot was actually computed in.
    ax.set_xlabel("True class")
    ax.set_ylabel("Predicted class")
    ax.set_title("Confusion matrix, YOLOv8s under the pre-cleanup eight-class taxonomy")
    ax.set_xticks(np.arange(-0.5, len(COLS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ROWS), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.1)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.6)

    ax.annotate("Column totals reproduce the validation composition "
                "(154 instances, Table S5).\n"
                "The Phomopsis and Leaf_rot columns are entirely in the "
                "background row: every\ninstance of both classes is missed. "
                "The Phomopsis row is empty: no box is emitted.",
                xy=(0.0, -0.30), xycoords="axes fraction", fontsize=7,
                color="0.3", va="top")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig3_confusion_matrix.{ext}", dpi=dpi)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figures")
    ap.add_argument("--dpi", type=int, default=400)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    m = matrix()
    bad = check(m)
    if bad:
        print("column totals do not match Table S5:")
        for b in bad:
            print("   ", b)
        raise SystemExit(1)
    print(f"column totals check out: {int(m[:, :-1].sum())} instances, "
          "matching Table S5")
    print(f"Leaf_rot boxes emitted: {int(m[ROWS.index('Leaf_rot')].sum())}")
    print(f"Phomopsis boxes emitted: {int(m[ROWS.index('Phomopsis')].sum())}")

    fig2(out, a.dpi)
    fig3(out, a.dpi)
    print(f"\nwritten to {out.resolve()} as PDF and PNG at {a.dpi} dpi")


if __name__ == "__main__":
    main()

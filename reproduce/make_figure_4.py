#!/usr/bin/env python3
"""
make_figure_4.py - redraw Figure 4 with a title that fits.

The version supplied with the original submission had its title clipped at both
ends by the figure boundary, so the reader saw "...omopsis Detection at conf=0.1
(red=Phomopsis, blue=oth...". Nothing about the content changes here: the same
image, the same detection, the same confidence. Only the layout is rebuilt so
that the title, the source filename and the threshold all fit inside the canvas
at a single-column width.

The detection box and its label are the detector's own overlay, carried over
from the original render; nothing is drawn on top of them here.

Usage:
    python make_figure_4.py --image phomopsis_photo.png --out figures/
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

MM = 1 / 25.4

CONF = 0.10
SOURCE = "IMG-20251006-WA0002"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "font.family": "DejaVu Sans",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--dpi", type=int, default=400)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    img = Image.open(a.image).convert("RGB")
    w, h = img.size

    # Single-column width; height follows the image aspect plus room for the title.
    fig_w = 88 * MM
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * h / w))
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
        sp.set_color("0.6")

    ax.set_title(f"Phomopsis detected at confidence {CONF:.2f}", pad=5)
    ax.annotate(f"Source image {SOURCE}. Deployment threshold is 0.20; no box "
                f"is returned there.\nAll AP values in this paper use the "
                f"framework default, which is lower still.",
                xy=(0.0, -0.035), xycoords="axes fraction", fontsize=6.5,
                color="0.35", va="top")

    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig4_phomopsis_lowconf.{ext}", dpi=a.dpi)
    plt.close(fig)
    print(f"written to {out.resolve()} at {a.dpi} dpi")


if __name__ == "__main__":
    main()

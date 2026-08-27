#!/usr/bin/env python3
"""
phash_distribution.py - Reviewer 1, revision 2, Comment 6.

Regenerates the complete cross-partition perceptual-hash distance distribution:
every validation image against every training image. This cannot be run from the
released artefacts, because it needs the imagery; it is released so that any
reader holding the same Roboflow export can produce the figure and check the
threshold choice against the full distribution rather than against the detected
pairs alone.

Outputs a histogram (PNG and PDF) and a CSV of the distance counts.

Usage:
    python phash_distribution.py \
        --train Leave_disease/train/images \
        --val   Leave_disease/valid/images \
        --out   figures/

Dependencies: pillow, imagehash, numpy, matplotlib.
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from PIL import Image
    import imagehash
except ImportError:
    raise SystemExit("pip install pillow imagehash")

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def hashes(d):
    out = []
    for p in sorted(Path(d).iterdir()):
        if p.suffix.lower() in EXTS:
            out.append((p.name, imagehash.phash(Image.open(p))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", default=".")
    ap.add_argument("--detection-threshold", type=int, default=10)
    ap.add_argument("--grouping-threshold", type=int, default=2)
    a = ap.parse_args()

    tr, va = hashes(a.train), hashes(a.val)
    print(f"{len(tr)} training images, {len(va)} validation images, "
          f"{len(tr) * len(va):,} pairs")

    counts = np.zeros(65, dtype=np.int64)
    nearest = []
    for _, hv in va:
        best = 64
        for _, ht in tr:
            d = hv - ht
            counts[d] += 1
            best = min(best, d)
        nearest.append(best)

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "phash_distance_counts.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["hamming_distance", "n_cross_partition_pairs"])
        for d, n in enumerate(counts):
            if n:
                w.writerow([d, int(n)])

    plt.rcParams.update({"font.size": 9, "savefig.dpi": 400,
                         "savefig.bbox": "tight"})
    fig, axes = plt.subplots(1, 2, figsize=(190 / 25.4, 62 / 25.4))

    ax = axes[0]
    ax.bar(np.arange(65), counts, width=1.0, color="#4d4d4d")
    ax.set_yscale("log")
    ax.axvline(a.detection_threshold, color="#b2182b", ls="--", lw=1.0)
    ax.axvline(a.grouping_threshold, color="#2166ac", ls=":", lw=1.0)
    ax.set_xlabel("Perceptual Hamming distance")
    ax.set_ylabel("Cross-partition pairs (log)")
    ax.set_title("(a) All validation x training pairs", loc="left")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.hist(nearest, bins=np.arange(0, 40) - 0.5, color="#4d4d4d")
    ax.axvline(a.detection_threshold, color="#b2182b", ls="--", lw=1.0)
    ax.set_xlabel("Distance to nearest training image")
    ax.set_ylabel("Validation images")
    ax.set_title("(b) Nearest-neighbour distance", loc="left")
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(a.out, f"figS_phash_full_distribution.{ext}"))
    print(f"wrote figS_phash_full_distribution.png/.pdf and "
          f"phash_distance_counts.csv to {a.out}")

    below = int(sum(1 for d in nearest if d <= a.detection_threshold))
    print(f"validation images within distance {a.detection_threshold} of some "
          f"training image: {below} of {len(va)}")


if __name__ == "__main__":
    main()

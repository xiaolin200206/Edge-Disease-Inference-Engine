#!/usr/bin/env python3
"""
regroup_split.py - reconstruct source groups and emit a leak-free partition.

Why
---
The dataset was augmented before splitting, so augmented variants of one source
photograph sit on both sides of the train/validation boundary. The manuscript
declines to retrain on a corrected partition on the ground that the original
source grouping is unrecoverable. Two signals recover much of it:

  1. Roboflow export names are of the form  <original-stem>_jpg.rf.<uuid>.jpg
     The text before ".rf." is the pre-export filename, so two exports of one
     source photograph share it. This is exact and invariant to any geometric
     augmentation, unlike a perceptual hash.

  2. Perceptual hashing over the eight dihedral transforms (four rotations x
     mirror), which catches variants whose stems were rewritten but whose
     content survives flips and right-angle rotations.

Groups are the connected components of the union of both relations, plus exact
MD5 identity. A component is assigned wholly to train or wholly to validation,
so no source photograph can straddle the boundary.

What this does and does not establish
-------------------------------------
The result is an upper-bound correction, not a proof of leak freedom. pHash is
not invariant to cropping, shear, or heavy colour jitter, so some variants will
remain unmatched and some components will be under-merged. Conversely, two
genuinely distinct photographs of the same lesion may be merged. Both are stated
rather than assumed away. The reviewer's own framing - "even if approximate
using perceptual clustering" - is the standard this meets.

If a single component absorbs most of the dataset, a leak-free split of useful
size does not exist, and that is itself the finding: it would mean augmentation
duplication is not confined to the 118 cross-partition pairs already reported.

Usage
-----
  python regroup_split.py --root Leave_disease --out-root Leave_disease_regrouped
  python regroup_split.py --root Leave_disease --report-only
  python regroup_split.py --root Leave_disease --threshold 8 --val-frac 0.2
"""

import argparse
import csv
import hashlib
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    from PIL import Image
    import imagehash
except ImportError:
    sys.exit("pip install pillow imagehash")

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
RF = re.compile(r"^(.*?)(?:_jpg|_png|_jpeg)?\.rf\.[0-9a-f]{8,}", re.I)


# ---------------------------------------------------------------- utilities

class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def source_stem(name):
    """Text before the Roboflow .rf. marker, or the plain stem if absent."""
    m = RF.match(name)
    if m:
        return m.group(1)
    return Path(name).stem


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dihedral_hashes(path):
    """pHash of the image under the eight dihedral transforms, as uint64."""
    im = Image.open(path).convert("L")
    out = []
    for mirror in (False, True):
        base = im.transpose(Image.FLIP_LEFT_RIGHT) if mirror else im
        for k in range(4):
            v = base.rotate(90 * k, expand=True)
            out.append(np.uint64(int(str(imagehash.phash(v)), 16)))
    return out


POPC = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming(a, b):
    """Hamming distance between uint64 arrays a (N,) and b (M,) -> (N,M)."""
    x = np.bitwise_xor(a[:, None], b[None, :]).view(np.uint8).reshape(
        a.size, b.size, 8)
    return POPC[x].sum(axis=2)


def read_classes(label_path):
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text().splitlines():
        p = line.split()
        if p:
            out.append(int(float(p[0])))
    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="dataset root containing train/ and valid/")
    ap.add_argument("--splits", nargs="+", default=["train", "valid"])
    ap.add_argument("--out-root", default=None,
                    help="where to write the regrouped dataset")
    ap.add_argument("--threshold", type=int, default=10,
                    help="pHash Hamming distance for a near-duplicate edge; "
                         "10 matches the threshold used in the original audit")
    ap.add_argument("--val-frac", type=float, default=None,
                    help="target validation fraction; default keeps the "
                         "original ratio")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--report", default="regroup_report.csv")
    a = ap.parse_args()

    root = Path(a.root)
    files, split_of = [], []
    for s in a.splits:
        d = root / s / "images"
        if not d.is_dir():
            print(f"  (no {d}, skipping)")
            continue
        got = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
        files += got
        split_of += [s] * len(got)
        print(f"{s:<8} {len(got):5d} images")
    n = len(files)
    if n == 0:
        sys.exit("no images found")

    orig_val_frac = split_of.count("valid") / n
    target_val = a.val_frac if a.val_frac is not None else orig_val_frac
    print(f"total    {n:5d} images, original validation fraction "
          f"{orig_val_frac:.3f}\n")

    # ---- signals ----
    print("hashing (8 dihedral variants each) ...")
    stems, md5s, hashes = [], [], []
    for i, p in enumerate(files, 1):
        if i % 100 == 0 or i == n:
            print(f"    {i}/{n}", end="\r", flush=True)
        stems.append(source_stem(p.name))
        md5s.append(md5(p))
        hashes.append(dihedral_hashes(p))
    print()

    dsu = DSU(n)

    # (1) shared Roboflow source stem
    by_stem = defaultdict(list)
    for i, s in enumerate(stems):
        by_stem[s].append(i)
    stem_groups = [v for v in by_stem.values() if len(v) > 1]
    for g in stem_groups:
        for j in g[1:]:
            dsu.union(g[0], j)
    print(f"stem signal   : {len(stem_groups)} multi-image stems covering "
          f"{sum(len(g) for g in stem_groups)} images")

    # (2) identical bytes
    by_md5 = defaultdict(list)
    for i, m in enumerate(md5s):
        by_md5[m].append(i)
    md5_groups = [v for v in by_md5.values() if len(v) > 1]
    for g in md5_groups:
        for j in g[1:]:
            dsu.union(g[0], j)
    print(f"md5 signal    : {len(md5_groups)} byte-identical groups covering "
          f"{sum(len(g) for g in md5_groups)} images")

    # (3) perceptual near-duplicate, min over dihedral variants
    canon = np.array([h[0] for h in hashes], dtype=np.uint64)
    variants = np.array(hashes, dtype=np.uint64)          # (n, 8)
    edges = 0
    B = 256
    for s in range(0, n, B):
        e = min(s + B, n)
        # distance of each variant of block rows against canonical of all
        d = np.stack([hamming(variants[s:e, k], canon) for k in range(8)])
        d = d.min(axis=0)                                  # (block, n)
        idx = np.argwhere(d <= a.threshold)
        for r, c in idx:
            i = s + int(r)
            j = int(c)
            if i < j:
                dsu.union(i, j)
                edges += 1
        print(f"    pHash {e}/{n}", end="\r", flush=True)
    print(f"\nphash signal  : {edges} near-duplicate edges at Hamming <= "
          f"{a.threshold}")

    # ---- components ----
    comp = defaultdict(list)
    for i in range(n):
        comp[dsu.find(i)].append(i)
    comps = sorted(comp.values(), key=len, reverse=True)
    sizes = [len(c) for c in comps]
    straddling = sum(1 for c in comps
                     if len({split_of[i] for i in c}) > 1)

    print(f"\n{'='*62}\ncomponents    : {len(comps)}")
    print(f"largest       : {sizes[0]} images ({100*sizes[0]/n:.1f}% of the set)")
    print(f"size histogram: " +
          ", ".join(f"{k}x{v}" for k, v in
                    sorted(Counter(sizes).items())[:12]))
    print(f"singletons    : {sizes.count(1)}")
    print(f"components straddling the current split: {straddling}")

    if sizes[0] > 0.5 * n:
        print("\nWARNING: one component holds more than half the dataset.")
        print("A leak-free split of useful size may not exist; treat the")
        print("component structure itself as the result.")

    # ---- report ----
    lbl_of = {}
    for i, p in enumerate(files):
        lp = root / split_of[i] / "labels" / (p.stem + ".txt")
        lbl_of[i] = read_classes(lp)

    with open(a.report, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["component", "size", "straddles_current_split",
                    "image", "current_split", "source_stem", "classes"])
        for ci, c in enumerate(comps):
            strad = len({split_of[i] for i in c}) > 1
            for i in c:
                w.writerow([ci, len(c), int(strad), files[i].name,
                            split_of[i], stems[i],
                            " ".join(map(str, sorted(set(lbl_of[i]))))])
    print(f"\nwritten: {a.report}")

    if a.report_only:
        return

    # ---- group-level split ----
    rng = np.random.default_rng(a.seed)
    order = list(range(len(comps)))
    rng.shuffle(order)

    # rare classes first, so single-component classes are not stranded
    cls_comp = defaultdict(set)
    for ci, c in enumerate(comps):
        for i in c:
            for k in lbl_of[i]:
                cls_comp[k].add(ci)
    rare = sorted(cls_comp, key=lambda k: len(cls_comp[k]))

    val_comps, want = set(), int(round(target_val * n))
    got, uncovered = 0, []
    for k in rare:      # give every class a presence in validation if possible,
        if any(ci in val_comps for ci in cls_comp[k]):   # without overshooting
            continue
        pick = min(cls_comp[k], key=lambda ci: len(comps[ci]))
        if got + len(comps[pick]) > want * 1.25:
            uncovered.append(k)
            continue
        val_comps.add(pick)
        got += len(comps[pick])
    if uncovered:
        print(f"\nclasses not placeable in validation within the size budget: "
              f"{uncovered}")
        print("  their source components are too large; raise --val-frac or "
              "accept their absence")
    for ci in order:
        if got >= want:
            break
        if ci not in val_comps:
            val_comps.add(ci)
            got += len(comps[ci])

    assign = {}
    for ci, c in enumerate(comps):
        for i in c:
            assign[i] = "valid" if ci in val_comps else "train"
    moved = sum(1 for i in range(n) if assign[i] != split_of[i])
    print(f"\nnew split: train {sum(1 for v in assign.values() if v=='train')}, "
          f"valid {got}  ({got/n:.3f})   images changing side: {moved}")

    cc = {s: Counter() for s in ("train", "valid")}
    for i in range(n):
        for k in set(lbl_of[i]):
            cc[assign[i]][k] += 1
    print(f"{'class':>6}{'train imgs':>12}{'valid imgs':>12}")
    for k in sorted(set(cc['train']) | set(cc['valid'])):
        print(f"{k:>6}{cc['train'][k]:>12}{cc['valid'][k]:>12}")
    empty = [k for k in cc['train'] if cc['valid'][k] == 0]
    if empty:
        print(f"classes absent from validation: {empty}")

    out = Path(a.out_root or (str(root) + "_regrouped"))
    for s in ("train", "valid"):
        (out / s / "images").mkdir(parents=True, exist_ok=True)
        (out / s / "labels").mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(files):
        dst = assign[i]
        shutil.copy2(p, out / dst / "images" / p.name)
        lp = root / split_of[i] / "labels" / (p.stem + ".txt")
        if lp.exists():
            shutil.copy2(lp, out / dst / "labels" / lp.name)

    src_yaml = next((y for y in root.parent.glob("*.yaml")), None)
    names_line = ""
    if src_yaml:
        txt = src_yaml.read_text()
        m = re.search(r"^names:.*?(?=^\w)", txt, re.S | re.M)
        if m:
            names_line = m.group(0)
    yml = out / "data_regrouped.yaml"
    yml.write_text(
        f"# leak-free partition by source component; see {a.report}\n"
        f"train: {(out/'train'/'images').resolve().as_posix()}\n"
        f"val: {(out/'valid'/'images').resolve().as_posix()}\n"
        + (names_line or "# copy names: and nc: from your original yaml\n"))
    print(f"\nwritten: {out}\n         {yml}")
    print("Check nc: and names: in the yaml before training.")


if __name__ == "__main__":
    main()

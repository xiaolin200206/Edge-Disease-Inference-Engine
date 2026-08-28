#!/usr/bin/env python3
"""
check_within_split_dupes.py — duplicates INSIDE a partition.

WHY THIS EXISTS
---------------
check_leakage_leafrot.py answers "does any image appear on both sides of the
train/validation boundary?". That is the question the paper is about, and it is
the question a leakage audit is normally asked. It has a blind spot: a file
duplicated inside one partition never crosses the boundary, so the audit is
silent about it by construction.

Run against this study's own dataset, that blind spot was not empty. Twenty-one
groups of byte-identical images sit inside the training partition and three
inside the validation partition, so 1,121 exported files correspond to 1,096
distinct images. Three validation files carry a Windows ' - Copy' suffix, and
two of those pair an annotated original with an unannotated duplicate, so the
same image is counted once as a disease instance and once as background.

None of this changes the paper's conclusions, and the duplicates are retained in
the released dataset because the reported numbers were computed on them. It is
released as a script because the general point is worth acting on: audit inside
each partition as well as across the boundary.

USAGE
-----
    python audit/check_within_split_dupes.py --root /path/to/dataset

    --splits train valid          which subdirectories to check
    --out    dupes.csv            where to write the inventory

DEPS
----
    Standard library. Pillow and imagehash optional; with them, near-duplicates
    within a split are reported too, not just byte-identical ones.
"""

import argparse
import collections
import csv
import hashlib
import itertools
import sys
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

try:
    from PIL import Image
    import imagehash
    HAVE_PHASH = True
except Exception:
    HAVE_PHASH = False


def md5_of(p, chunk=1 << 20):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "valid_clean", "test"])
    ap.add_argument("--threshold", type=int, default=10,
                    help="perceptual-hash Hamming distance for near-duplicates")
    ap.add_argument("--out", default="within_split_dupes.csv")
    a = ap.parse_args()

    root = Path(a.root)
    if not root.is_dir():
        sys.exit(f"not found: {root}")

    print(f"perceptual hashing: {'on' if HAVE_PHASH else 'OFF (pip install pillow imagehash)'}")
    out_rows = []
    grand = collections.Counter()

    for sp in a.splits:
        imgdir, lbldir = root / sp / "images", root / sp / "labels"
        if not imgdir.is_dir():
            continue
        imgs = [p for p in sorted(imgdir.iterdir())
                if p.is_file() and (p.suffix.lower() in IMG_EXT
                                    or any(e in p.name.lower() for e in IMG_EXT))]
        labelled = set()
        if lbldir.is_dir():
            for p in lbldir.iterdir():
                if p.suffix.lower() == ".txt":
                    try:
                        if any(ln.strip() for ln in p.read_text(errors="ignore").splitlines()):
                            labelled.add(p.stem)
                    except OSError:
                        pass

        by_md5 = collections.defaultdict(list)
        for p in imgs:
            by_md5[md5_of(p)].append(p)
        exact = {k: v for k, v in by_md5.items() if len(v) > 1}
        redundant = sum(len(v) - 1 for v in exact.values())
        distinct = len(by_md5)

        print(f"\n{sp}: {len(imgs)} files, {distinct} distinct images, "
              f"{len(exact)} identical-bytes groups ({redundant} redundant)")
        grand["files"] += len(imgs)
        grand["distinct"] += distinct
        grand["redundant"] += redundant

        for k, v in sorted(exact.items()):
            states = ["labelled" if p.stem in labelled else "background" for p in v]
            mixed = len(set(states)) > 1
            if mixed:
                grand["mixed"] += 1
            print(f"  md5 {k[:12]}  {'MIXED LABEL STATE' if mixed else ''}")
            for p, st in zip(v, states):
                print(f"    [{st:10s}] {p.name}")
                out_rows.append([sp, "exact", k, p.name, st, 0])

        if HAVE_PHASH:
            hashes = {}
            for p in imgs:
                try:
                    hashes[p] = imagehash.phash(Image.open(p))
                except Exception:
                    pass
            seen = {frozenset(x.name for x in v) for v in exact.values()}
            near = 0
            for p, q in itertools.combinations(sorted(hashes), 2):
                d = hashes[p] - hashes[q]
                if 0 < d <= a.threshold:
                    if frozenset((p.name, q.name)) in seen:
                        continue
                    near += 1
                    for x in (p, q):
                        st = "labelled" if x.stem in labelled else "background"
                        out_rows.append([sp, "near", f"d={d}", x.name, st, d])
            print(f"  near-duplicate pairs within this split "
                  f"(0 < distance <= {a.threshold}): {near}")
            grand["near"] += near

    print("\n" + "=" * 66)
    print(f"total files {grand['files']}, distinct images {grand['distinct']}, "
          f"redundant {grand['redundant']}")
    print(f"identical-bytes groups whose members disagree on label state: "
          f"{grand['mixed']}")
    if HAVE_PHASH:
        print(f"near-duplicate pairs within a split: {grand['near']}")
    print("A group with a MIXED LABEL STATE means the same image is counted once")
    print("as an annotated instance and once as background inside one partition.")

    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "type", "key", "filename", "label_state", "hamming"])
        w.writerows(out_rows)
    print(f"\ninventory -> {a.out}")


if __name__ == "__main__":
    main()

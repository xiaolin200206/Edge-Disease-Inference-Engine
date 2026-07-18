#!/usr/bin/env python3
"""
add_hashes.py — add MD5 and perceptual-hash columns to an existing
leakage_report.csv without re-running the full audit.

WHY THIS EXISTS
---------------
The manuscript undertakes to release the inventory of implicated image pairs
"by MD5 and perceptual hash rather than by image", so that the reported leak
can be checked by anyone holding the same images — or the same Roboflow
export — without the dataset itself being distributed. Filenames alone do not
satisfy that undertaking: a filename is not evidence about pixel content.

The audit script (check_leakage_leafrot.py) computes both hashes but earlier
versions wrote only filenames and the Hamming distance. This script fills in
the missing columns for a report that has already been produced, so the
verified audit result is preserved rather than regenerated.

Use check_leakage_leafrot.py instead if you want to re-run the audit from
scratch; it now emits the hash columns directly.

USAGE
-----
    python audit/add_hashes.py \
        --report audit/leakage_report.csv \
        --train_images Leave_disease/train/images \
        --val_images   Leave_disease/valid/images

Writes leakage_report_hashed.csv beside the input. Review it, then replace
the original.

DEPS
----
    pip install pillow imagehash
"""

import argparse
import csv
import hashlib
import os
import sys

try:
    from PIL import Image
    import imagehash
except ImportError:
    sys.exit("pip install pillow imagehash")


def md5(path):
    return hashlib.md5(open(path, 'rb').read()).hexdigest()


def phash(path):
    # Must match check_leakage_leafrot.py exactly: imagehash defaults
    # (8x8 DCT, 64-bit). Changing the hash size here would make the
    # published hashes incomparable with the reported Hamming distances.
    return str(imagehash.phash(Image.open(path)))


def index(*dirs):
    """basename -> full path, over every image directory given."""
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    m = {}
    for d in dirs:
        if not os.path.isdir(d):
            sys.exit(f"not a directory: {d}")
        for f in os.listdir(d):
            if f.lower().endswith(exts):
                m[f] = os.path.join(d, f)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--report', default='audit/leakage_report.csv')
    ap.add_argument('--train_images', required=True)
    ap.add_argument('--val_images', required=True)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    out = a.out or a.report.replace('.csv', '_hashed.csv')
    files = index(a.train_images, a.val_images)
    print(f"indexed {len(files)} images")

    rows = list(csv.DictReader(open(a.report, newline='', encoding='utf-8')))
    print(f"{len(rows)} pairs in {a.report}")

    cache, missing = {}, set()

    def hashes(name):
        if name in cache:
            return cache[name]
        p = files.get(name)
        if p is None:
            missing.add(name)
            cache[name] = ('', '')
        else:
            cache[name] = (md5(p), phash(p))
        return cache[name]

    cols = ['type', 'val_image', 'train_image', 'hamming_distance',
            'val_md5', 'train_md5', 'val_phash', 'train_phash']
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, r in enumerate(rows, 1):
            vm, vp = hashes(r['val_image'])
            tm, tp = hashes(r['train_image'])
            w.writerow({**{k: r.get(k, '') for k in cols[:4]},
                        'val_md5': vm, 'train_md5': tm,
                        'val_phash': vp, 'train_phash': tp})
            if i % 25 == 0:
                print(f"  {i}/{len(rows)}")

    print(f"\n-> {out}")

    # Consistency check: for exact pairs the two MD5s must agree, and the
    # recomputed pHash distance must reproduce the distance already in the
    # report. If either fails, the report and the images have diverged and
    # the audit should be re-run rather than patched.
    bad_md5 = bad_dist = 0
    for r in csv.DictReader(open(out, newline='', encoding='utf-8')):
        if not r['val_phash'] or not r['train_phash']:
            continue
        if r['type'] == 'exact' and r['val_md5'] != r['train_md5']:
            bad_md5 += 1
        d = imagehash.hex_to_hash(r['val_phash']) - \
            imagehash.hex_to_hash(r['train_phash'])
        if d != int(r['hamming_distance']):
            bad_dist += 1

    print("\nconsistency:")
    print(f"  exact pairs with mismatched MD5      : {bad_md5}  "
          f"{'OK' if bad_md5 == 0 else 'FAIL'}")
    print(f"  rows whose pHash distance disagrees  : {bad_dist}  "
          f"{'OK' if bad_dist == 0 else 'FAIL'}")
    if missing:
        print(f"\n  {len(missing)} filenames not found in the image dirs; "
              f"their hash cells are blank:")
        for m in sorted(missing)[:10]:
            print(f"    {m}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")
        print("  These rows cannot be released as hashes. Either point the "
              "script at the correct directories or re-run the audit.")


if __name__ == '__main__':
    main()

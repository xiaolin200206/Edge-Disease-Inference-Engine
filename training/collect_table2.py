#!/usr/bin/env python3
"""
collect_table2.py - evaluate every Table 2 run and summarise by architecture.

Picks up all runs matching  */runs_table2/table2_<arch>_seed<n>/weights/best.pt
regardless of how deeply ultralytics nested the project directory, evaluates each
on the same validation partition with the same routine, and reports per-seed rows
plus a per-architecture spread.

The spread is computed within each architecture, never across them: pooling
YOLOv8s, YOLOv11s and RT-DETR into one standard deviation would describe nothing.
Where the same seeds exist for two architectures, the paired difference is also
reported, since that is the quantity the architecture comparison actually rests
on.

Usage
-----
  python collect_table2.py
  python collect_table2.py --data data_orig_abs.yaml --out table2_all.csv
"""

import argparse
import csv
import glob
import re
import statistics as st
from pathlib import Path


def parse_tag(path):
    """table2_v11s_seed2 -> ('v11s', 2)"""
    tag = Path(path).parts[-3]
    m = re.match(r"table2_(.+)_seed(\d+)$", tag)
    if not m:
        return tag, None
    return m.group(1), int(m.group(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_orig_abs.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default="table2_all.csv")
    ap.add_argument("--pattern",
                    default="**/runs_table2/table2_*/weights/best.pt")
    a = ap.parse_args()

    from ultralytics import YOLO

    paths = sorted(set(glob.glob(a.pattern, recursive=True)))
    if not paths:
        raise SystemExit(f"no runs matched {a.pattern}")
    print(f"{len(paths)} run(s) found\n")

    rows = []
    for w in paths:
        arch, seed = parse_tag(w)
        m = YOLO(w).val(data=a.data, imgsz=a.imgsz, device=a.device,
                        workers=0, verbose=False)
        r = {"arch": arch, "seed": seed,
             "mAP50": round(float(m.box.map50), 4),
             "mAP50_95": round(float(m.box.map), 4)}
        for i, c in enumerate(m.box.ap_class_index):
            r[m.names[int(c)]] = round(float(m.box.ap50[i]), 4)
        rows.append(r)
        print(f"  {arch:<8} seed {seed}   mAP50 {r['mAP50']:.4f}   "
              f"mAP50-95 {r['mAP50_95']:.4f}")

    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)

    metrics = [k for k in keys if k not in ("arch", "seed")]
    archs = list(dict.fromkeys(r["arch"] for r in rows))

    # ---- per-seed table ----
    print("\n" + "=" * 78)
    print("Per-seed results")
    print("=" * 78)
    hdr = f"{'arch':<8}{'seed':>5}" + "".join(f"{m[:11]:>13}" for m in metrics)
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: (x["arch"], x["seed"] if x["seed"] is not None else -1)):
        line = f"{r['arch']:<8}{str(r['seed']):>5}"
        for m in metrics:
            v = r.get(m)
            line += f"{v:>13.4f}" if isinstance(v, float) else f"{'-':>13}"
        print(line)

    # ---- per-architecture spread ----
    print("\n" + "=" * 78)
    print("Spread within each architecture")
    print("=" * 78)
    for arch in archs:
        sub = [r for r in rows if r["arch"] == arch]
        print(f"\n{arch}   n = {len(sub)}   seeds {[r['seed'] for r in sub]}")
        if len(sub) < 2:
            print("  single run; no spread to report")
            for m in metrics:
                v = sub[0].get(m)
                if isinstance(v, float):
                    print(f"    {m:<18}{v:>9.4f}")
            continue
        print(f"    {'metric':<18}{'mean':>9}{'sd':>9}{'min':>9}{'max':>9}{'range':>9}")
        for m in metrics:
            v = [r[m] for r in sub if isinstance(r.get(m), float)]
            if len(v) < 2:
                continue
            print(f"    {m:<18}{st.mean(v):>9.4f}{st.stdev(v):>9.4f}"
                  f"{min(v):>9.4f}{max(v):>9.4f}{max(v)-min(v):>9.4f}")

    # ---- paired differences on shared seeds ----
    if len(archs) > 1:
        print("\n" + "=" * 78)
        print("Paired differences on shared seeds")
        print("=" * 78)
        by = {(r["arch"], r["seed"]): r for r in rows}
        for i, x in enumerate(archs):
            for y in archs[i + 1:]:
                shared = sorted(s for s in {r["seed"] for r in rows}
                                if (x, s) in by and (y, s) in by)
                if not shared:
                    continue
                d = [by[(x, s)]["mAP50"] - by[(y, s)]["mAP50"] for s in shared]
                print(f"\n  {x} minus {y}   seeds {shared}")
                for s, v in zip(shared, d):
                    print(f"    seed {s}: {v:+.4f}")
                if len(d) > 1:
                    mean, sd = st.mean(d), st.stdev(d)
                    half = 1.96 * sd / len(d) ** 0.5
                    print(f"    mean {mean:+.4f}   sd {sd:.4f}   "
                          f"95% CI [{mean-half:+.4f}, {mean+half:+.4f}]")
                    signs = {v > 0 for v in d}
                    if len(signs) > 1:
                        print("    the sign of the difference is not stable "
                              "across seeds")

    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()

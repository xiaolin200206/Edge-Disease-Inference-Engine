#!/usr/bin/env python3
"""
val_rtdetr.py - validate the three RT-DETR-l seeds on the original partition.

Table 2 reports RT-DETR-l over three seeds. `results.csv` records the per-epoch
history but not the final validation pass, so the per-class figures in Table 2
and Supplementary Table S4 come from a separate validation of each selected
checkpoint. This script performs it, and writes the numbers to JSON so that the
table can be checked against them.

The interesting result is visible in the output rather than in the aggregate:
seeds 0 and 1 score 0.995 on root_disease and 0.166 / 0.028 on Pink_Disease,
while seed 2 inverts the pattern. Those two classes hold one and two validation
instances. Nothing differs across the three runs but the seed.

The weights are not released (see Data availability), so this script cannot be
run from the repository alone; it is included because it defines exactly how the
reported numbers were obtained. `rtdetr_3seeds.json` in
data/training_logs/multiseed/ carries its output.

Usage, from the directory the runs were launched in:
    python reproduce/val_rtdetr.py --runs runs/detect/runs/detect/runs_table2
"""

import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/detect/runs/detect/runs_table2",
                    help="directory holding table2_rtdetr_seed{0,1,2}")
    ap.add_argument("--data", default="data_orig_abs.yaml")
    ap.add_argument("--out", default="rtdetr_3seeds.json")
    a = ap.parse_args()

    from ultralytics import RTDETR

    out = {}
    for s in (0, 1, 2):
        p = os.path.join(a.runs, f"table2_rtdetr_seed{s}", "weights", "best.pt")
        if not os.path.exists(p):
            print(f"seed{s}: best.pt not found at {p}")
            continue
        m = RTDETR(p)
        # workers=0: the default dataloader workers deadlock on Windows unless
        # the caller guards __main__, and this validation is 126 images.
        r = m.val(data=a.data, split="val", workers=0, verbose=False)
        names = m.names
        ap50 = {names[i]: float(v) for i, v in zip(r.box.ap_class_index, r.box.ap50)}
        ap95 = {names[i]: float(v) for i, v in zip(r.box.ap_class_index, r.box.ap)}
        out[f"seed{s}"] = {"map50": float(r.box.map50), "map": float(r.box.map),
                           "per_class_ap50": ap50, "per_class_ap": ap95}
        print(f"\nseed{s}: mAP50={r.box.map50:.4f}  mAP50-95={r.box.map:.4f}")
        for k in ap50:
            print(f"    {k:<14} AP50={ap50[k]:.4f}  AP50-95={ap95[k]:.4f}")

    if len(out) == 3:
        m50 = [out[f"seed{s}"]["map50"] for s in (0, 1, 2)]
        mean = sum(m50) / 3
        sd = (sum((x - mean) ** 2 for x in m50) / 2) ** 0.5
        print(f"\nmAP@0.5 over three seeds: {mean:.4f} +/- {sd:.4f}")
        for c in ("root_disease", "Pink_Disease"):
            v = [out[f"seed{s}"]["per_class_ap50"][c] for s in (0, 1, 2)]
            print(f"  {c:<14} {v[0]:.4f}  {v[1]:.4f}  {v[2]:.4f}   range {max(v)-min(v):.4f}")

    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    main()

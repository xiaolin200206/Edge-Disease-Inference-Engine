#!/usr/bin/env python3
"""
run_m1.py - retrain on a source-level, leak-free partition.

Answers Reviewer 1's Major Comment 1: the manuscript identifies train/validation
leakage but continues to report performance from a model whose weights were
learned with the leaked images present, so the learned representation may still
carry information from them.

The partition is produced by audit/regroup_split.py, which groups images into
source components using three signals - the Roboflow export stem preserved
before the ".rf." marker, MD5 identity, and perceptual hash within Hamming
distance 2 over the eight dihedral transforms - and assigns each component
wholly to one side of the boundary. Three seeds are trained under the Table 2
configuration, on the same machine and framework version.

What this does and does not measure
-----------------------------------
Re-partitioning necessarily changes the validation set as well as the training
set: instance counts move from 154 to 125, Algal_leave from 57 images to 46,
Pink_Disease from 2 to 7. The difference between this result and the original
split therefore confounds the removal of leakage with a change of test sample,
and the two must not be reported as a before-and-after.

Table 1 already answers "what did the leakage cost", by holding the model,
weights and protocol fixed and removing only the implicated images from the
validation set. This script answers the separate question the manuscript's own
future-work section posed: what the architecture attains under a correct
partition. What transfers across the two partitions is the per-class
qualitative result, which is where the interesting finding lies - Leaf_rot
scores 0.000 with zero variance across seven runs on the original split and
recovers on the regrouped one.

Usage
-----
  python ../audit/regroup_split.py --root Leave_disease --threshold 2 \\
      --out-root Leave_disease_regrouped
  python run_m1.py
  python run_m1.py --seeds 0 1 2 --dry-run
"""

import argparse
import csv
import statistics as st
from datetime import datetime
from pathlib import Path

# Identical to the Table 2 runs; only the dataset partition differs.
BASE = dict(
    model="yolov8s.pt",
    epochs=150,
    patience=0,
    batch=4,
    imgsz=640,
    device="0",
    workers=8,
    optimizer="auto",
    warmup_epochs=3.0,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",
                    default="Leave_disease_regrouped/data_regrouped.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--project", default="runs_m1")
    ap.add_argument("--out", default="m1_results.csv")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    data = Path(a.data)
    if not data.exists():
        raise SystemExit(
            f"dataset YAML not found: {data}\n"
            "Run audit/regroup_split.py first, and check that the YAML it\n"
            "writes carries the nc: and names: fields.")

    text = data.read_text()
    if "nc:" not in text or "names:" not in text:
        raise SystemExit(
            f"{data} is missing nc: or names:.\n"
            "regroup_split.py copies these from the source YAML when it can\n"
            "find one; add them by hand before training, or the class indices\n"
            "will not line up with Table 2.")

    print("configuration (identical to the Table 2 runs)")
    for k, v in BASE.items():
        print(f"  {k:<16}{v}")
    print(f"  {'data':<16}{data}")
    print(f"  {'seeds':<16}{a.seeds}")

    if a.dry_run:
        print("\ndry run - nothing trained")
        return

    from ultralytics import YOLO

    for s in a.seeds:
        name = f"m1_regrouped_seed{s}"
        if (Path(a.project) / name / "weights" / "best.pt").exists():
            print(f"skip {name} (already trained)")
            continue
        print(f"\n{'='*62}\n[{datetime.now():%H:%M:%S}] {name}\n{'='*62}")
        YOLO(BASE["model"]).train(
            data=str(data), epochs=BASE["epochs"], patience=BASE["patience"],
            batch=BASE["batch"], imgsz=BASE["imgsz"], device=BASE["device"],
            workers=BASE["workers"], optimizer=BASE["optimizer"],
            warmup_epochs=BASE["warmup_epochs"], seed=s, deterministic=True,
            project=a.project, name=name, exist_ok=True, verbose=False)

    # ---- collect ----
    rows = []
    for w in sorted(Path(".").glob(f"**/{a.project}/m1_regrouped_seed*/weights/best.pt")):
        seed = int(w.parts[-3].replace("m1_regrouped_seed", ""))
        m = YOLO(str(w)).val(data=str(data), imgsz=BASE["imgsz"],
                             device=BASE["device"], workers=0, verbose=False)
        r = {"seed": seed,
             "mAP50": round(float(m.box.map50), 4),
             "mAP50_95": round(float(m.box.map), 4)}
        for i, c in enumerate(m.box.ap_class_index):
            r[m.names[int(c)]] = round(float(m.box.ap50[i]), 4)
        rows.append(r)
        print(f"  seed {seed}: mAP@0.5 {r['mAP50']:.4f}")

    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(rows)

    print(f"\n{'metric':<20}{'mean':>9}{'sd':>9}{'min':>9}{'max':>9}")
    for k in keys:
        if k == "seed":
            continue
        v = [r[k] for r in rows if isinstance(r.get(k), float)]
        if len(v) > 1:
            print(f"{k:<20}{st.mean(v):>9.4f}{st.stdev(v):>9.4f}"
                  f"{min(v):>9.4f}{max(v):>9.4f}")

    print(f"\nwritten: {a.out}")
    print("Compare per-class values against the original split qualitatively;\n"
          "the aggregates are not a before-and-after (see module docstring).")


if __name__ == "__main__":
    main()

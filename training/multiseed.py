#!/usr/bin/env python3
"""
multiseed.py - retrain the matched configuration under several random seeds.

Answers Reviewer 1's Major Comment 6: the manuscript attributes roughly 83% of
the observed architecture improvement to training configuration rather than
architecture, but that figure rests on a single seed and a single dataset, so it
is unclear whether the same proportion would hold under different random
initialisations.

Every hyperparameter is taken verbatim from data/training_logs/yolov8s_matched/
args.yaml - the configuration that produced the run reported in Table 2 - with
only `seed` varied. Seed 0 is the published run; it is re-run here so that the
reported spread includes a reproduction of the original rather than treating it
as a fixed reference point.

Each completed run is validated on the same validation split and the per-class
and aggregate metrics are collected into one CSV, so the spread across seeds can
be compared directly against the architecture difference reported in Table 2.

Usage
-----
  python multiseed.py --seeds 0 1 2 3 4
  python multiseed.py --seeds 1 2 3 4 --data data_orig_abs.yaml
  python multiseed.py --seeds 1 --dry-run
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

# Hyperparameters copied from the matched run's args.yaml. Do not edit casually:
# the claim "architecture is the only variable" depends on these being identical
# to the YOLOv11s run.
BASE = dict(
    model="yolov8s.pt",
    epochs=150,
    patience=0,          # early stopping disabled
    batch=4,
    imgsz=640,
    device="0",
    workers=8,
    optimizer="auto",
    warmup_epochs=3.0,
)


def one_run(seed, data_yaml, project, exist_ok):
    from ultralytics import YOLO

    name = f"seed{seed}"
    print(f"\n{'='*66}\n[{datetime.now():%H:%M:%S}] seed {seed}\n{'='*66}")
    t0 = time.time()

    model = YOLO(BASE["model"])
    model.train(
        data=str(data_yaml),
        epochs=BASE["epochs"],
        patience=BASE["patience"],
        batch=BASE["batch"],
        imgsz=BASE["imgsz"],
        device=BASE["device"],
        workers=BASE["workers"],
        optimizer=BASE["optimizer"],
        warmup_epochs=BASE["warmup_epochs"],
        seed=seed,
        deterministic=True,
        project=project,
        name=name,
        exist_ok=exist_ok,
        verbose=False,
    )

    # validate the selected checkpoint on the same split
    best = Path(project) / name / "weights" / "best.pt"
    metrics = YOLO(str(best)).val(data=str(data_yaml), imgsz=BASE["imgsz"],
                                  device=BASE["device"], verbose=False)

    elapsed = (time.time() - t0) / 60
    names = metrics.names
    row = dict(seed=seed, minutes=round(elapsed, 1),
               mAP50=round(float(metrics.box.map50), 4),
               mAP50_95=round(float(metrics.box.map), 4),
               precision=round(float(metrics.box.mp), 4),
               recall=round(float(metrics.box.mr), 4),
               weights=str(best))
    for i, c in enumerate(metrics.box.ap_class_index):
        row[f"AP50_{names[int(c)]}"] = round(float(metrics.box.ap50[i]), 4)

    print(f"seed {seed}: mAP@0.5 {row['mAP50']:.4f}  "
          f"mAP@0.5:0.95 {row['mAP50_95']:.4f}  ({elapsed:.0f} min)")
    return row


def summarise(rows, out_csv):
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    import statistics as st
    print(f"\n{'='*66}\nSpread across {len(rows)} seeds\n{'='*66}")
    print(f"{'metric':<26}{'mean':>9}{'sd':>9}{'min':>9}{'max':>9}{'range':>9}")
    print("-" * 71)
    for k in keys:
        if k in ("seed", "weights", "minutes"):
            continue
        v = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        if len(v) < 2:
            continue
        print(f"{k:<26}{st.mean(v):>9.4f}{st.stdev(v):>9.4f}"
              f"{min(v):>9.4f}{max(v):>9.4f}{max(v)-min(v):>9.4f}")

    m = [r["mAP50"] for r in rows]
    m95 = [r["mAP50_95"] for r in rows]
    print(f"\nFor comparison, Table 2 reports an architecture difference of 0.017")
    print(f"in mAP@0.5. The seed-to-seed range here is {max(m)-min(m):.4f} "
          f"(mAP@0.5) and {max(m95)-min(m95):.4f} (mAP@0.5:0.95).")
    if max(m) - min(m) > 0.017:
        print("The seed spread exceeds the reported architecture difference.")
    print(f"\nwritten: {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--data", default="data_orig_abs.yaml",
                    help="dataset yaml used for Table 2")
    ap.add_argument("--project", default="runs/detect/multiseed")
    ap.add_argument("--out", default="multiseed_results.csv")
    ap.add_argument("--exist-ok", action="store_true",
                    help="overwrite an existing run directory")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    data_yaml = Path(a.data)
    if not data_yaml.exists():
        raise SystemExit(f"dataset yaml not found: {data_yaml}\n"
                         f"pass the one used for Table 2 with --data")

    print("configuration (from the matched run's args.yaml):")
    for k, v in BASE.items():
        print(f"  {k:<16}{v}")
    print(f"  {'data':<16}{data_yaml}")
    print(f"  {'seeds':<16}{a.seeds}")
    print(f"\nestimated: about 45 min per seed -> "
          f"{len(a.seeds)*45/60:.1f} h total")

    if a.dry_run:
        print("\ndry run, nothing trained")
        return

    rows, t0 = [], time.time()
    for s in a.seeds:
        try:
            rows.append(one_run(s, data_yaml, a.project, a.exist_ok))
        except Exception as e:                       # keep going; salvage the rest
            print(f"seed {s} FAILED: {type(e).__name__}: {e}")
        if rows:
            summarise(rows, a.out)                   # checkpoint after every seed
    print(f"\ntotal wall time: {(time.time()-t0)/3600:.2f} h")


if __name__ == "__main__":
    main()

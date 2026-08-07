#!/usr/bin/env python3
"""
run_rtdetr.py - train RT-DETR-l under the Table 2 configuration.

Answers Reviewer 1's Major Comment 3: only two YOLO architectures are
investigated, so it cannot be determined whether the observed rare-class
suppression is architecture-independent or specific to the YOLO family.

RT-DETR is a transformer-based, NMS-free detector, i.e. a different detection
paradigm rather than another member of the same family. Every training
hyperparameter is held identical to the YOLOv8s and YOLOv11s runs of Table 2 -
150 epochs, early stopping disabled, batch 4, imgsz 640, optimiser auto, warm-up
3 epochs, seed 0, deterministic - and the run executes on the same machine under
the same framework version, so the architecture and its detection paradigm are
the only variables that differ.

The question this run answers is qualitative, not comparative: whether Leaf_rot
and Phomopsis, which return AP@0.5 = 0.000 in all eight YOLO runs, remain at zero
under a detector with no anchor assignment and no NMS. A single seed is
sufficient for that question. It is not sufficient for an aggregate performance
comparison, and none is drawn.

Usage
-----
  python run_rtdetr.py
  python run_rtdetr.py --data data_orig_abs.yaml --epochs 150
  python run_rtdetr.py --dry-run
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

# Held identical to the Table 2 runs. Do not edit: the comparison depends on it.
BASE = dict(
    epochs=150,
    patience=0,            # early stopping disabled
    batch=4,
    imgsz=640,
    device="0",
    workers=8,
    optimizer="auto",
    warmup_epochs=3.0,
    seed=0,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rtdetr-l.pt")
    ap.add_argument("--data", default="data_orig_abs.yaml",
                    help="must be the YAML used for Table 2")
    ap.add_argument("--project", default="runs/detect/runs_table2")
    ap.add_argument("--name", default="table2_rtdetr_seed0")
    ap.add_argument("--epochs", type=int, default=BASE["epochs"])
    ap.add_argument("--batch", type=int, default=BASE["batch"])
    ap.add_argument("--workers", type=int, default=BASE["workers"])
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    data = Path(a.data)
    if not data.exists():
        raise SystemExit(f"dataset YAML not found: {data}")

    out = Path(a.project) / a.name / "weights" / "best.pt"
    if out.exists():
        print(f"already trained: {out}\nDelete it to retrain.")
        return

    import torch
    from ultralytics import YOLO
    import ultralytics

    print("environment")
    print(f"  ultralytics   {ultralytics.__version__}")
    print(f"  torch         {torch.__version__}  cuda {torch.version.cuda}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  gpu           {p.name}  {p.total_memory/1e9:.1f} GB")
    else:
        print("  gpu           NONE - training will fall back to CPU and be slow")

    print("\nconfiguration (identical to the Table 2 YOLO runs)")
    cfg = dict(BASE, epochs=a.epochs, batch=a.batch, workers=a.workers)
    for k, v in cfg.items():
        print(f"  {k:<16}{v}")
    print(f"  {'model':<16}{a.model}")
    print(f"  {'data':<16}{data}")
    print(f"  {'output':<16}{Path(a.project) / a.name}")

    if a.dry_run:
        print("\ndry run - nothing trained")
        return

    print(f"\n[{datetime.now():%H:%M:%S}] starting; RT-DETR-l is heavier than "
          f"YOLOv8s, expect roughly 2-3x the wall time on a 6 GB card.\n")
    t0 = time.time()

    try:
        model = YOLO(a.model)
        model.train(
            data=str(data),
            epochs=cfg["epochs"],
            patience=cfg["patience"],
            batch=cfg["batch"],
            imgsz=cfg["imgsz"],
            device=cfg["device"],
            workers=cfg["workers"],
            optimizer=cfg["optimizer"],
            warmup_epochs=cfg["warmup_epochs"],
            seed=cfg["seed"],
            deterministic=True,
            project=a.project,
            name=a.name,
            exist_ok=True,
            verbose=False,
        )
    except torch.cuda.OutOfMemoryError:
        raise SystemExit(
            "CUDA out of memory.\n"
            "Do NOT reduce batch to fit: batch 4 is part of the matched\n"
            "configuration and changing it breaks comparability with Table 2.\n"
            "Options: close other GPU processes, lower --workers to 4 or 2,\n"
            "or run this one job on a larger card.")

    print(f"\n[{datetime.now():%H:%M:%S}] done in {(time.time()-t0)/3600:.2f} h")
    print(f"weights: {out}")
    print("\nNow run collect_table2.py; its glob picks this run up alongside\n"
          "the YOLO runs, so the per-class table is produced in one pass.")


if __name__ == "__main__":
    main()

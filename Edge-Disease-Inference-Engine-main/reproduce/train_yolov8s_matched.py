#!/usr/bin/env python3
"""
train_yolov8s_matched.py — the controlled architecture ablation of Table 2.

WHAT THIS IS FOR
----------------
The study's original YOLOv8s run and its YOLOv11s run differed in more than
architecture: 50 epochs at batch 16 with early stopping permitted, against 150
epochs at batch 4 with early stopping disabled. Comparing them answers "which
of these two training histories produced a better model", which is not the
question the paper asks.

This script retrains YOLOv8s from COCO-pretrained weights under the YOLOv11s
configuration in full, so that architecture is the only variable that differs.
Table 2 reports this run, not the original one.

WHAT IT CHANGED
---------------
Under matched training the aggregate gap between the two architectures falls
from 0.101 to 0.017 in mAP@0.5, and from an apparent difference to 0.003 in
mAP@0.5:0.95. Pink_Disease, which carried the largest single share of the
unmatched gap, scores 0.495 under both — identical to three decimals. Roughly
83% of what had looked like an architecture effect was training configuration.

What survives the control is the finding the paper rests on: with training
schedule, label space, validation partition, evaluation threshold and scoring
routine all held fixed, Leaf_rot scores 0.000 under both architectures and
Phomopsis scores 0.005 and 0.000, on 26 and 54 validation instances.

EVALUATION THRESHOLD
--------------------
Validation is run at the framework's default detector confidence threshold,
which is also what the training loop reports. Every detection figure in the
paper is at this threshold. Scoring one architecture at a stricter threshold
than the other silently changes the comparison, which is precisely the fault
the audit in Section 4.1 was written to catch.

USAGE
-----
    python reproduce/train_yolov8s_matched.py --data config/data_orig_abs.yaml

Requires the image dataset, which is not distributed (see README).

DEPS
----
    pip install ultralytics
"""

import argparse
import json
import time

# The YOLOv11s configuration, reproduced exactly. Every value here is part of
# the control: changing any one of them reopens the confound this script exists
# to close.
CFG = dict(
    epochs=150,
    patience=0,          # early stopping disabled
    batch=4,
    imgsz=640,
    workers=8,
    seed=0,
    optimizer='auto',    # selects AdamW and derives lr0 from the class count,
                         # so nc must be 7 for the setting to match
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    close_mosaic=10,     # mosaic withdrawn for the final ten epochs
    pretrained=True,
    cos_lr=False,
    rect=False,
)


def preflight(data_yaml):
    """Fail before training rather than after, on the things that silently
    invalidate the comparison."""
    import yaml
    d = yaml.safe_load(open(data_yaml, encoding='utf-8'))

    nc = d.get('nc')
    assert nc == 7, (
        f"nc is {nc}, expected 7. The repaired label space has seven classes "
        f"(six carry validation instances). With optimizer='auto' the initial "
        f"learning rate is derived from the class count, so a different nc "
        f"gives a different learning rate and the run is no longer matched."
    )

    expected = ['Algal_leave', 'Leaf_rot', 'Phomopsis', 'Pink_Disease',
                'early_blight', 'root_disease', 'Anthracnose']
    assert list(d['names']) == expected, (
        f"class order differs from the YOLOv11s run.\n"
        f"  expected {expected}\n"
        f"  found    {list(d['names'])}\n"
        f"Label indices are positional; a different order silently relabels "
        f"every annotation."
    )
    print(f"preflight OK: nc={nc}, class order matches")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True,
                    help='dataset YAML with the repaired seven-class taxonomy')
    ap.add_argument('--weights', default='yolov8s.pt')
    ap.add_argument('--project', default='runs/paper2_matched')
    ap.add_argument('--name', default='yolov8s_matched')
    ap.add_argument('--device', default=0)
    ap.add_argument('--out', default='v8s_matched_result.json')
    a = ap.parse_args()

    preflight(a.data)

    from ultralytics import YOLO
    import ultralytics

    t0 = time.time()
    m = YOLO(a.weights)
    m.train(data=a.data, project=a.project, name=a.name,
            exist_ok=True, device=a.device, **CFG)
    train_s = time.time() - t0

    # Default confidence threshold, matching the YOLOv11s evaluation.
    r = m.val(data=a.data, imgsz=CFG['imgsz'])
    per_class = {r.names[c]: float(r.box.ap50[i])
                 for i, c in enumerate(r.ap_class_index)}

    print("\nper-class AP@0.5 (default confidence threshold)")
    for k, v in per_class.items():
        print(f"  {k:<16}{v:.4f}")
    print(f"\n  mAP@0.5      {r.box.map50:.5f}")
    print(f"  mAP@0.5:0.95 {r.box.map:.5f}")

    json.dump({
        'ultralytics': ultralytics.__version__,
        'train_time_s': round(train_s, 1),
        'config': CFG,
        'mAP50': float(r.box.map50),
        'mAP50_95': float(r.box.map),
        'precision': float(r.box.mp),
        'recall': float(r.box.mr),
        'per_class_ap50': per_class,
    }, open(a.out, 'w'), indent=2)
    print(f"\n-> {a.out}")

    print("\nreference (Table 2):")
    print("  YOLOv8s original 50 ep b16   mAP@0.5 0.301")
    print("  YOLOv8s matched  150 ep b4   mAP@0.5 0.385   <- this run")
    print("  YOLOv11s         150 ep b4   mAP@0.5 0.402")
    print("\nthe claim under test:")
    for k in ('Leaf_rot', 'Phomopsis'):
        if k in per_class:
            v = per_class[k]
            print(f"  {k:<12}{v:.4f}  "
                  f"{'still zero' if v < 0.01 else 'NOT zero — re-examine'}")

    # Environment differences that remain and cannot be removed after the
    # fact. Recorded in the paper's limitations; recorded here too so the
    # JSON is self-describing.
    print("\nNot controlled: this run may execute on different hardware and "
          "under a different minor release of the framework than the "
          "YOLOv11s run. Both affect throughput rather than the optimisation "
          "performed.")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
bootstrap_ap.py - image-level bootstrap confidence intervals for per-class AP.

Answers Reviewer 1's Major Comment 4: several validation classes hold one or two
images, so their AP values are unstable, and the manuscript reports no confidence
intervals, no repeated experiments, and no significance testing for the
architecture comparison.

Method
------
The model is run over the validation set once. Predictions and ground truth are
cached in memory, then the *images* are resampled with replacement B times; AP is
recomputed on each resample. No retraining and no repeated inference are
required, so the whole procedure costs one validation pass plus a few minutes of
CPU.

This quantifies sampling uncertainty in the validation set - the thing that makes
a one-image class unstable. It does not capture training-run variance; that needs
multiple seeds and is a separate experiment (Reviewer 1, Major Comment 6).

AP is computed at IoU 0.5 by 101-point interpolation over all predictions, with
no confidence threshold applied, matching the manuscript's stated convention that
every candidate box enters the precision-recall computation.

Paired comparison
-----------------
With --compare, both checkpoints are evaluated on the *same* bootstrap resamples,
so the per-resample difference in mAP is a paired quantity. The reported interval
for the architecture difference is therefore the correct one: an interval on the
difference, not the overlap of two marginal intervals.

Usage
-----
  python bootstrap_ap.py --weights yolov8s_matched/weights/best.pt \\
      --data Leave_disease/valid --n-boot 2000 --out boot_v8s.csv

  python bootstrap_ap.py \\
      --weights   yolov8s_matched/weights/best.pt \\
      --compare   runs/detect/runs_paper2_ablation/yolov11s_wholeleaf_v1/weights/best.pt \\
      --data Leave_disease/valid --n-boot 2000 --out boot_compare.csv
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# ground truth
# ---------------------------------------------------------------------------

def load_labels(label_path, w, h):
    """YOLO txt -> (cls, x1, y1, x2, y2) in pixels."""
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text().strip().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c = int(float(p[0]))
        xc, yc, bw, bh = (float(v) for v in p[1:5])
        out.append((c, (xc - bw / 2) * w, (yc - bh / 2) * h,
                    (xc + bw / 2) * w, (yc + bh / 2) * h))
    return out


def iou_matrix(a, b):
    """a: (N,4), b: (M,4) -> (N,M)"""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-12)


# ---------------------------------------------------------------------------
# one inference pass, cached
# ---------------------------------------------------------------------------

def collect(weights, images, labels_dir, imgsz, device, iou_thr):
    """
    Run the model once. For every image return, per class:
        matched: array of (confidence, is_true_positive)
        n_gt:    number of ground-truth boxes
    Matching is greedy by descending confidence at the given IoU, one GT per
    prediction, which is the standard COCO/VOC assignment.
    """
    from ultralytics import YOLO
    model = YOLO(weights)

    per_image = []
    for i, img in enumerate(images, 1):
        if i % 25 == 0 or i == len(images):
            print(f"    {i}/{len(images)}", end="\r", flush=True)

        r = model.predict(str(img), imgsz=imgsz, device=device,
                          conf=0.001, verbose=False)[0]
        h, w = r.orig_shape
        gt = load_labels(labels_dir / (img.stem + ".txt"), w, h)

        boxes = r.boxes
        pred_cls = boxes.cls.cpu().numpy().astype(int) if len(boxes) else np.array([], int)
        pred_conf = boxes.conf.cpu().numpy() if len(boxes) else np.array([])
        pred_xyxy = boxes.xyxy.cpu().numpy() if len(boxes) else np.zeros((0, 4))

        rec = {}
        classes = set(pred_cls.tolist()) | {g[0] for g in gt}
        for c in classes:
            g = [b[1:] for b in gt if b[0] == c]
            sel = pred_cls == c
            pc, pb = pred_conf[sel], pred_xyxy[sel]
            order = np.argsort(-pc)
            pc, pb = pc[order], pb[order]

            tp = np.zeros(len(pc))
            if len(g) and len(pb):
                ious = iou_matrix(pb, g)
                taken = np.zeros(len(g), bool)
                for k in range(len(pb)):
                    j = int(np.argmax(np.where(taken, -1, ious[k])))
                    if ious[k, j] >= iou_thr and not taken[j]:
                        taken[j] = True
                        tp[k] = 1
            rec[c] = dict(conf=pc, tp=tp, n_gt=len(g))
        per_image.append(rec)

    print()
    return per_image, model.names


def ap_from_pool(conf, tp, n_gt):
    """101-point interpolated AP for one class from pooled detections."""
    if n_gt == 0:
        return np.nan            # class absent from this resample
    if len(conf) == 0:
        return 0.0
    o = np.argsort(-conf)
    tp = tp[o]
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1 - tp)
    rec = ctp / n_gt
    prec = ctp / (ctp + cfp + 1e-12)
    # monotone envelope
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    grid = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, grid, side="left")
    q = np.where(idx < len(prec), prec[np.clip(idx, 0, len(prec) - 1)], 0.0)
    return float(q.mean())


def eval_resample(per_image, idx, class_ids):
    """AP per class and mAP over a bootstrap resample of image indices."""
    aps = {}
    for c in class_ids:
        conf, tp, n_gt = [], [], 0
        for i in idx:
            r = per_image[i].get(c)
            if r is None:
                continue
            conf.append(r["conf"])
            tp.append(r["tp"])
            n_gt += r["n_gt"]
        conf = np.concatenate(conf) if conf else np.array([])
        tp = np.concatenate(tp) if tp else np.array([])
        aps[c] = ap_from_pool(conf, tp, n_gt)
    vals = [v for v in aps.values() if not np.isnan(v)]
    return aps, (float(np.mean(vals)) if vals else np.nan)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--compare", default=None,
                    help="second checkpoint, evaluated on the same resamples")
    ap.add_argument("--data", required=True,
                    help="validation folder containing images/ and labels/")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0", help="'0' for the GPU, 'cpu' otherwise")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="bootstrap_ap.csv")
    a = ap.parse_args()

    root = Path(a.data)
    img_dir, lbl_dir = root / "images", root / "labels"
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        sys.exit(f"expected {img_dir} and {lbl_dir}")
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    if not images:
        sys.exit(f"no images in {img_dir}")
    print(f"validation images: {len(images)}")

    print(f"inference pass 1: {a.weights}")
    pi_a, names = collect(a.weights, images, lbl_dir, a.imgsz, a.device, a.iou)

    pi_b = None
    if a.compare:
        print(f"inference pass 2: {a.compare}")
        pi_b, _ = collect(a.compare, images, lbl_dir, a.imgsz, a.device, a.iou)

    class_ids = sorted({c for rec in pi_a for c in rec})
    if pi_b:
        class_ids = sorted(set(class_ids) | {c for rec in pi_b for c in rec})

    # ground-truth support per class, the quantity that drives the instability
    support_img = {c: sum(1 for r in pi_a if r.get(c, {}).get("n_gt", 0) > 0)
                   for c in class_ids}
    support_box = {c: sum(r.get(c, {}).get("n_gt", 0) for r in pi_a)
                   for c in class_ids}

    n = len(images)
    rng = np.random.default_rng(a.seed)
    print(f"bootstrapping {a.n_boot} resamples over {n} images ...")

    boot_a = defaultdict(list)
    boot_b = defaultdict(list)
    map_a, map_b, map_d = [], [], []
    for b in range(a.n_boot):
        if (b + 1) % 200 == 0:
            print(f"    {b+1}/{a.n_boot}", end="\r", flush=True)
        idx = rng.integers(0, n, n)
        aps, m = eval_resample(pi_a, idx, class_ids)
        for c, v in aps.items():
            boot_a[c].append(v)
        map_a.append(m)
        if pi_b:
            apsb, mb = eval_resample(pi_b, idx, class_ids)
            for c, v in apsb.items():
                boot_b[c].append(v)
            map_b.append(mb)
            map_d.append(m - mb)
    print()

    def ci(v):
        v = np.asarray([x for x in v if not np.isnan(x)])
        if v.size == 0:
            return (np.nan,) * 4
        return (float(v.mean()), float(np.percentile(v, 2.5)),
                float(np.percentile(v, 97.5)), float(v.std(ddof=1)))

    # point estimates on the full validation set
    full = np.arange(n)
    point_a, pm_a = eval_resample(pi_a, full, class_ids)
    point_b, pm_b = (eval_resample(pi_b, full, class_ids) if pi_b else ({}, np.nan))

    hdr = ["class", "n_val_images", "n_val_boxes", "AP_A", "AP_A_mean",
           "AP_A_lo95", "AP_A_hi95", "AP_A_sd", "CI_width_A"]
    if pi_b:
        hdr += ["AP_B", "AP_B_mean", "AP_B_lo95", "AP_B_hi95", "AP_B_sd",
                "diff_A_minus_B", "diff_lo95", "diff_hi95"]

    rows = []
    for c in class_ids:
        m, lo, hi, sd = ci(boot_a[c])
        row = [names.get(c, c), support_img[c], support_box[c],
               round(point_a.get(c, float("nan")), 4), round(m, 4),
               round(lo, 4), round(hi, 4), round(sd, 4), round(hi - lo, 4)]
        if pi_b:
            mb, lob, hib, sdb = ci(boot_b[c])
            d = [x - y for x, y in zip(boot_a[c], boot_b[c])
                 if not (np.isnan(x) or np.isnan(y))]
            dm, dlo, dhi, _ = ci(d)
            row += [round(point_b.get(c, float("nan")), 4), round(mb, 4),
                    round(lob, 4), round(hib, 4), round(sdb, 4),
                    round(dm, 4), round(dlo, 4), round(dhi, 4)]
        rows.append(row)

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(rows)

    # ---- console summary ----
    print()
    print(f"{'class':<16}{'imgs':>6}{'boxes':>7}{'AP':>8}{'95% CI':>20}{'width':>8}")
    print("-" * 65)
    for r in rows:
        print(f"{str(r[0]):<16}{r[1]:>6}{r[2]:>7}{r[3]:>8.3f}"
              f"{'[' + format(r[5], '.3f') + ', ' + format(r[6], '.3f') + ']':>20}"
              f"{r[8]:>8.3f}")

    m, lo, hi, _ = ci(map_a)
    print(f"\nmAP@0.5 (A): {pm_a:.4f}   bootstrap 95% CI [{lo:.4f}, {hi:.4f}]")
    if pi_b:
        mb, lob, hib, _ = ci(map_b)
        dm, dlo, dhi, _ = ci(map_d)
        print(f"mAP@0.5 (B): {pm_b:.4f}   bootstrap 95% CI [{lob:.4f}, {hib:.4f}]")
        print(f"\nPaired difference A - B: {pm_a - pm_b:+.4f}   "
              f"95% CI [{dlo:+.4f}, {dhi:+.4f}]")
        crosses = dlo <= 0 <= dhi
        print("The interval " + ("includes" if crosses else "excludes") +
              " zero, so the architecture difference is "
              + ("not distinguishable from sampling variation in the validation set."
                 if crosses else
                 "larger than validation-set sampling variation alone."))
        print("Note this is validation-set uncertainty only; training-run variance")
        print("requires multiple seeds and is not captured here.")

    print(f"\nwritten: {a.out}")


if __name__ == "__main__":
    main()

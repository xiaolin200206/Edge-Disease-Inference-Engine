# Training logs (Sections 3.3.1, 4.2.1, 4.2.3)

Per-epoch validation records for the two runs compared in Table 2, released so that
the claims made about their stability can be checked rather than taken on trust.

```
yolov11s_wholeleaf/
  results.csv                # 150 epochs
yolov8s_matched/
  results.csv                # 150 epochs
  args.yaml                  # the full training configuration actually used
  v8s_matched_result.json    # final validation, per-class AP@0.5
  BoxPR_curve.png            # Supplementary Figure S1
  BoxP_curve.png             # Supplementary Figure S2
  BoxR_curve.png             # Supplementary Figure S3
  results.png                # Supplementary Figure S4
```

## What these files are for

**`args.yaml` is the evidence for the controlled ablation.** Section 3.3.1 states that
YOLOv8s was retrained under the YOLOv11s configuration in full, so that architecture is
the only variable that differs. That is a claim about fourteen hyperparameters, and this
file records what was actually passed to the trainer: 150 epochs, `patience=0`, batch 4,
`imgsz=640`, `seed=0`, `optimizer=auto`, `lr0=0.01`, `lrf=0.01`, `momentum=0.937`,
`weight_decay=0.0005`, `warmup_epochs=3.0`, `close_mosaic=10`, `cos_lr=False`,
`rect=False`.

Two of these matter more than they look. With `optimizer=auto` the initial learning rate
is derived from the class count, so a different `nc` would silently unmatch the runs. And
label indices are positional, so a different class order would relabel every annotation.

**`results.csv` is the evidence that the aggregate is unstable.** Over the final hundred
epochs, on a validation set that does not change, mAP@0.5 varies by 0.178 in the matched
YOLOv8s run and 0.225 in the YOLOv11s run, while mAP@0.5:0.95 varies by only 0.076 and
0.099. The reported architecture difference is 0.017 — an order of magnitude below the
run's own epoch-to-epoch variation, which is why Section 4.2.1 declines to interpret it.

To reproduce those figures:

```bash
python - <<'EOF'
import csv
for run in ('yolov11s_wholeleaf', 'yolov8s_matched'):
    r = list(csv.DictReader(open(f'{run}/results.csv')))
    k5  = [c for c in r[0] if 'mAP50(B)'    in c and '95' not in c][0]
    k95 = [c for c in r[0] if 'mAP50-95(B)' in c][0]
    m5  = [float(x[k5])  for x in r][49:]
    m95 = [float(x[k95]) for x in r][49:]
    print(f'{run:<22} mAP@0.5 {min(m5):.3f}-{max(m5):.3f} (range {max(m5)-min(m5):.3f})  '
          f'mAP@0.5:0.95 range {max(m95)-min(m95):.3f}')
EOF
```

## Why the reported checkpoint is not the best epoch by mAP@0.5

Peak mAP@0.5 during training is 0.438 (epoch 64) for the matched YOLOv8s run and 0.462
(epoch 139) for YOLOv11s. The checkpoints reported in Table 2 score 0.385 and 0.402.

This is not a discrepancy. The framework selects `best.pt` on a fitness criterion weighted
nine-to-one toward mAP@0.5:0.95, which is the more stable of the two quantities, and the
same criterion selected both checkpoints — YOLOv8s at epoch 64, YOLOv11s at epoch 82. A
selection rule that chased peak mAP@0.5 would be chasing the noise documented above.

## What is not here

Per-class AP is not recorded per epoch; `results.csv` carries only precision, recall,
mAP@0.5 and mAP@0.5:0.95. Per-class values exist only for the final checkpoint, in
`v8s_matched_result.json` and in Table 2. Any claim about how an individual class behaved
during training cannot be checked from these files, and none is made in the paper.

The images and weights are not released (see the repository README).

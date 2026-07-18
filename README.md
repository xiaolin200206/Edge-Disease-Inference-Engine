# Edge-Based Durian Disease Detection — Deployment Engine & Reproducibility Package

Code, telemetry, and audit tools accompanying the manuscript:

> **Thermal Cut-off, Not Throttling: Duty-Cycle Scheduling and a Dataset-Integrity
> Audit for On-Farm Edge Disease Detection in Durian**
> Lin Ding Shan, Institute of Computer Science and Digital Innovation, UCSI University
> ORCID [0009-0009-6031-8479](https://orcid.org/0009-0009-6031-8479)
> Under review, *Smart Agricultural Technology*, 2026.

This repository contains (1) the offline field-deployment inference engine, (2) the
dataset-integrity audit used to detect and quantify a train/validation leak, and (3)
scripts and telemetry that reproduce the paper's analyses from raw data.

---

## Read this first: what is and is not reproducible here

The paper reports two kinds of result, and they are not equally checkable. Saying so
plainly is more useful than a blanket claim of reproducibility.

**Fully reproducible from this repository, with no additional assets.**
The duty-cycle thermal characterisation — Table 3, and the cut-off counts and coverage
figures on which the paper's operational recommendation rests. The raw telemetry is
released unfiltered; one command recomputes every value. Also the epoch-to-epoch
stability analysis of §4.2.1, which is what licenses the paper's refusal to interpret
the small difference between the two architectures.

**Not reproducible from this repository.**
Everything requiring the durian images or the trained weights: the leakage
quantification (Table 1), the architecture ablation (Table 2), and the per-class
figures. The dataset and weights are assets of an ongoing commercialisation effort and
are not released.

For those results the repository provides the *procedure* rather than the *evidence*:
every script is released so it can be run against any other detection dataset, and the
leakage inventory is published **as hashes**, so the specific leak reported in the paper
can be checked by anyone holding the same images — or the same export — without the
images being distributed by us.

---

## What reproduces what

| Paper element | Script / data | Needs the dataset? |
|---|---|---|
| **§4.1 Integrity audit** — 1 byte-identical + 117 near-duplicate pairs | `audit/check_leakage_leafrot.py` → `audit/leakage_report.csv` | yes |
| **Table 1** — leak-free re-evaluation (Algal_leave +35.0%, mAP@0.5 +9.5%, mAP@0.5:0.95 +14.8%) | `reproduce/clean_val_and_revalidate.py` | yes |
| **Table 2** — controlled architecture ablation, YOLOv8s vs YOLOv11s | `reproduce/train_yolov8s_matched.py`, `reproduce/train_yolov11s_paper2.py`, `reproduce/count_classes.py` | yes |
| **Table 3** — duty-cycle thermal, 99 cut-offs to 0 | `reproduce/reproduce_table3.py` + `data/thermal_telemetry/` | **no** |
| **§3.2** — taxonomy repair (`Early_Blight` / `early_blight`, nc 8 to 7) | `reproduce/remove_class_and_reindex.py` | yes |
| **§5.1** — field-deployment latency and thermal motivation (177.8 ms median, CV 5.2%, 91% throttled) | `data/field_test.csv` | **no** |
| **§4.2.1, §4.2.3** — epoch-to-epoch instability of the aggregate (0.178 and 0.225 in mAP@0.5) | `data/training_logs/` | **no** |
| **§5** — field deployment system | `deployment/detection.py` | weights only |

---

## Two conventions that decide whether a number is comparable

**Evaluation threshold.** Every detection figure in the paper — every table, every
figure — is scored at the framework's *default* detector confidence threshold, at which
essentially every candidate box enters the precision–recall computation. This matters
more than it sounds. Scoring one architecture at a stricter threshold than another
silently changes the comparison, and a class that scores 0.000 at a strict threshold has
not necessarily failed; it may simply have been filtered out. The claim in §4.2.3 is that
two classes score zero *at the loosest available threshold*, which is a different and far
stronger statement.

One exception is disclosed in the paper and repeated here: Ultralytics assembles its
confusion matrix at a fixed display threshold of 0.25 even when AP is computed at the
default. Fig. 3 therefore shows where detections go, not how well they score, and its
cell counts are not a second view of the AP values.

**Which YOLOv8s run.** There are two, and they are not interchangeable.

| | epochs | batch | early stopping | reported in |
|---|---|---|---|---|
| original | 50 | 16 | permitted | Figs. 2 and 3 |
| **matched** | **150** | **4** | **disabled** | **Table 2** |

Figures 2 and 3 document the taxonomy fault in the *original* run. Table 2 is the
controlled ablation and uses the *matched* retraining, under which architecture is the
only variable that differs from the YOLOv11s run. Reading a per-class value out of
Fig. 2 and comparing it against Table 2 produces a discrepancy that is not an error.

"Architecture is the only variable" is a claim about fourteen hyperparameters, so the
configuration actually passed to the trainer is released as
`data/training_logs/yolov8s_matched/args.yaml` rather than left as an assertion in the
paper.

---

## Repository layout

```
.
├── deployment/
│   └── detection.py                  # offline field inference engine (§5): thermal-aware
│                                     # duty-cycling, 82 C cut-off, temporal-confirmation
│                                     # buffering, class-specific thresholds, CSV telemetry
├── audit/
│   ├── check_leakage_leafrot.py      # §4.1 leakage + Leaf_rot distribution audit
│   ├── add_hashes.py                 # adds MD5/pHash columns to an existing report
│   └── leakage_report.csv            # all 118 cross-partition pairs, by hash
├── reproduce/
│   ├── reproduce_table3.py           # Table 3 from raw telemetry (one command)
│   ├── clean_val_and_revalidate.py   # Table 1: quantify the leakage effect
│   ├── train_yolov8s_matched.py      # Table 2: matched-configuration ablation
│   ├── train_yolov11s_paper2.py      # Table 2: the YOLOv11s run
│   ├── count_classes.py              # per-class instance / source-image counts
│   └── remove_class_and_reindex.py   # taxonomy repair, nc 8 to 7
├── config/
│   ├── data_orig_abs.yaml            # original validation split
│   └── data_clean.yaml               # leak-free validation split
└── data/
    ├── field_test.csv                # 46,576-sample field-deployment log (§5.1)
    ├── thermal_telemetry/            # five three-hour duty-cycle benchmarks (Table 3)
    └── training_logs/                # per-epoch records + configuration for both runs
                                      # in Table 2; see its own README
```

---

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.10+ on Raspberry Pi OS (deployment) and desktop Linux/Windows
(reproduction).

---

## Reproducing the results

### 1. Duty-cycle thermal characterisation (Table 3) — no dataset needed

```bash
python reproduce/reproduce_table3.py --data-dir data/thermal_telemetry
```

Recomputes, for all five configurations, the sample count, mean and maximum CPU
temperature, time to first throttle, throttle-flag percentage, and monitoring coverage.
The released logs are unfiltered.

This is the reproduction that matters most, because the paper's operational conclusion
rests on it: continuous inference triggered 99 software thermal cut-offs and a 15 s sleep
interval still triggered 41, whereas a 30 s interval eliminated them entirely. Throttling
itself cost only 2.7% of median latency, so what duty cycling buys is uninterrupted
operation rather than frame rate.

### 2. Dataset-integrity audit (§4.1) — needs the images

```bash
python audit/check_leakage_leafrot.py \
    --train_images Leave_disease/train/images \
    --val_images   Leave_disease/valid/images \
    --train_labels Leave_disease/train/labels \
    --val_labels   Leave_disease/valid/labels \
    --classes_yaml config/data_orig_abs.yaml
```

Regenerates `leakage_report.csv` with MD5 and pHash for both members of every pair, plus
the Leaf_rot / Phomopsis instance-to-image distribution.

To add hash columns to an existing report without re-running the audit:

```bash
python audit/add_hashes.py --report audit/leakage_report.csv \
    --train_images Leave_disease/train/images \
    --val_images   Leave_disease/valid/images
```

This also verifies that the recomputed pHash distances reproduce the distances already in
the report, and that exact pairs really do share an MD5.

### 3. Quantify the leakage (Table 1) — needs images + weights

```bash
python reproduce/clean_val_and_revalidate.py --weights <trained.pt> --dataset Leave_disease
```

Removes every leaking validation image listed in `leakage_report.csv`, then validates the
*same* weights on the original and cleaned validation sets. This isolates the leak's
contribution: Algal_leave inflated by 35.0%, aggregate mAP@0.5 by 9.5%.

`<trained.pt>` is the study's trained checkpoint, not the COCO-pretrained `yolo11s.pt`.

### 4. Controlled architecture ablation (Table 2) — needs the images

```bash
python reproduce/train_yolov8s_matched.py --data config/data_orig_abs.yaml
python reproduce/train_yolov11s_paper2.py
python reproduce/count_classes.py
```

`train_yolov8s_matched.py` runs a preflight check before training and refuses to start if
the class count or the class order differs from the YOLOv11s run. Both matter: with
`optimizer='auto'` the initial learning rate is derived from the class count, so a
different `nc` silently unmatches the configuration; and label indices are positional, so
a different class order relabels every annotation.

`count_classes.py` also confirms the case-sensitivity duplicate (`Early_Blight` vs
`early_blight`) discussed in §4.2.

> **Paths.** The `config/*.yaml` files and several scripts contain absolute Windows paths
> from the original environment. Edit `train:` / `val:` in the YAMLs and the `BASE_DIR` /
> `--dataset` arguments before running any dataset-dependent step.

---

### 5. Stability of the aggregate metric (§4.2.1) — no dataset needed

```bash
cd data/training_logs
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

Over the final hundred epochs, on a validation set that does not change, mAP@0.5 varies by
0.178 in the matched YOLOv8s run and 0.225 in the YOLOv11s run, while mAP@0.5:0.95 varies
by only 0.076 and 0.099. The architecture difference reported in Table 2 is 0.017 — an
order of magnitude below the runs' own epoch-to-epoch variation, which is why §4.2.1
declines to interpret it. The instability is concentrated at the looser IoU threshold
because classes holding one or two validation instances swing between zero and near-unity,
and each such swing moves a six-class mean by up to 0.167.

This also explains a discrepancy a reader will otherwise notice: peak mAP@0.5 during
training is 0.438 and 0.462, while the checkpoints reported in Table 2 score 0.385 and
0.402. The framework selects `best.pt` on a fitness criterion weighted nine-to-one toward
mAP@0.5:0.95, the more stable quantity, and the same criterion selected both checkpoints.
A rule that chased peak mAP@0.5 would be chasing the variation documented above.

---

## What the leakage inventory contains

`audit/leakage_report.csv` lists all 118 cross-partition pairs — 1 byte-identical and 117
near-duplicate — with, for each member:

| column | meaning |
|---|---|
| `type` | `exact` (identical MD5) or `near` (pHash distance < 10) |
| `val_image`, `train_image` | filenames as they appear in the export |
| `hamming_distance` | perceptual-hash distance between the pair |
| `val_md5`, `train_md5` | MD5 of the file bytes |
| `val_phash`, `train_phash` | 64-bit perceptual hash (imagehash defaults, 8x8 DCT) |

The hashes are what makes the claim checkable. Anyone holding the same images can
recompute them and confirm — or refute — that these specific pairs straddle the
train/validation boundary, without us distributing a single image.

The fault itself is diagnosable from the inventory: the near-duplicates concentrate in one
group of source photographs, and the training partition holds augmented variants of images
whose siblings sit on the other side of the split. That is an augmentation-before-split
error, a workflow hazard of browser-based annotation tools rather than a mistake peculiar
to this dataset.

---

## Data availability

**Released.** The duty-cycle thermal telemetry (`data/thermal_telemetry/`), the
field-deployment telemetry log (`data/field_test.csv`), the per-epoch training logs and
configuration for both runs in Table 2 (`data/training_logs/`), the complete leakage-pair
inventory as hashes (`audit/leakage_report.csv`), and all audit, reproduction and
deployment code.

**Not released.** The durian leaf-disease images and the trained weights (`.pt` / `.onnx`),
which are assets of an ongoing commercialisation effort. Steps 2–4 above therefore require
the original images; step 1 is fully self-contained.

---

## Citation

```
L. D. Shan, "Thermal Cut-off, Not Throttling: Duty-Cycle Scheduling and a
Dataset-Integrity Audit for On-Farm Edge Disease Detection in Durian,"
under review, Smart Agricultural Technology, 2026.
```

## License

Apache-2.0. Code and telemetry are released for research reproducibility; the image
dataset and model weights are not covered by this license and are not distributed.

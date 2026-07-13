# Edge-Based Durian Disease Detection — Deployment Engine & Reproducibility Package

Code, telemetry, and audit tools accompanying the manuscript:

> **From Bench to Mud: A Failure-Driven Engineering Analysis of Edge-Based Durian Disease Detection, with a Dataset Integrity Audit and Duty-Cycle Thermal Characterization**
> Lin Ding Shan, under review, *Computers and Electronics in Agriculture*, 2026.

This repository contains (1) the offline field-deployment inference engine, (2) the dataset-integrity audit used to detect and quantify a train/validation data leak, and (3) scripts and telemetry that reproduce every table in the paper from raw data. The durian image dataset and trained model weights are **proprietary assets of an ongoing commercialization effort and are not released**; everything required to reproduce the reported *analyses* is provided.

---

## What reproduces what

| Paper element | Script / data | Notes |
|---|---|---|
| **§4.1 Integrity audit** (1 byte-identical + 117 near-duplicate pairs) | `audit/check_leakage_leafrot.py` → `audit/leakage_report.csv` | MD5 + perceptual-hash duplicate detection across train/val |
| **Table 1** (leak-free re-evaluation; 35.0% / 9.5% / 14.8% inflation) | `reproduce/clean_val_and_revalidate.py` | Removes the leaking val images and re-runs validation, same weights |
| **Table 2** (per-class AP + instance counts) | `reproduce/train_yolov11s_paper2.py`, `reproduce/count_classes.py` | Model-capacity ablation on the whole-leaf dataset; instance/image counts |
| **Table 4** (duty-cycle thermal, 97.5% → 0.0%) | `reproduce/reproduce_table4.py` + `data/thermal_telemetry/` | One command reproduces all five configurations |
| **Dataset preprocessing** (remove duplicate `Early_Blight` class) | `reproduce/remove_class_and_reindex.py` | nc 8 → 7, label re-indexing |
| **Field deployment system** (§5) | `deployment/detection.py` | The always-on edge engine (add your own weights) |

---

## Repository layout

```
.
├── deployment/
│   └── detection.py            # offline field inference engine (§5): thermal-aware
│                               # duty-cycling, 82 °C cutoff, temporal-confirmation
│                               # buffering, class-specific thresholds, CSV telemetry
├── audit/
│   ├── check_leakage_leafrot.py    # §4.1 leakage + Leaf_rot distribution audit
│   └── leakage_report.csv          # audit output: all 118 cross-partition pairs
├── reproduce/
│   ├── reproduce_table4.py         # Table 4 from raw telemetry (one command)
│   ├── clean_val_and_revalidate.py # Table 1: quantify leakage effect on AP
│   ├── train_yolov11s_paper2.py    # Table 2: capacity-ablation training (v11s)
│   ├── count_classes.py            # Table 2: per-class instance / image counts
│   └── remove_class_and_reindex.py # dataset preprocessing
├── config/
│   ├── data_orig_abs.yaml          # original validation split
│   └── data_clean.yaml             # leak-free validation split (valid_clean)
└── data/
    └── thermal_telemetry/          # five three-hour duty-cycle benchmarks (Table 4)
```

---

## Setup

```bash
pip install -r requirements.txt
```
Tested with Python 3.10+ on Raspberry Pi OS (deployment) and desktop Linux/Windows (reproduction).

---

## Reproducing the results

### 1. Duty-cycle thermal characterization (Table 4) — no dataset needed
```bash
python reproduce/reproduce_table4.py --data-dir data/thermal_telemetry
```
Recomputes, for all five configurations, the sample count, mean/max CPU temperature, time to first throttle, throttle-flag percentage, and monitoring coverage — reproducing Table 4 exactly. The released logs are unfiltered.

### 2. Dataset-integrity audit (§4.1) — needs the (proprietary) image dataset
```bash
python audit/check_leakage_leafrot.py \
    --train_images Leave_disease/train/images \
    --val_images   Leave_disease/valid/images \
    --train_labels Leave_disease/train/labels \
    --val_labels   Leave_disease/valid/labels \
    --classes_yaml config/data_orig_abs.yaml
```
Regenerates `leakage_report.csv` (1 byte-identical + 117 near-duplicate pairs) and the Leaf_rot/Phomopsis instance-to-image distribution. The provided `audit/leakage_report.csv` is the exact inventory used in Supplementary Table S1.

### 3. Quantify the leakage (Table 1) — needs dataset + weights
```bash
python reproduce/clean_val_and_revalidate.py --weights yolo11s.pt --dataset Leave_disease
```
Removes every leaking validation image listed in `leakage_report.csv`, then validates the *same* weights on the original and cleaned validation sets, isolating the leak's contribution to AP (Algal_leave +35.0%, aggregate mAP@0.5 +9.5%).

### 4. Capacity ablation and counts (Table 2) — needs dataset
```bash
python reproduce/count_classes.py                 # per-class instance & image counts
python reproduce/train_yolov11s_paper2.py         # YOLOv11s on the whole-leaf dataset
```
`count_classes.py` also confirms the case-sensitivity duplicate (`Early_Blight` vs `early_blight`) discussed in §4.2.

> **Paths:** the `config/*.yaml` files and several scripts contain absolute Windows paths from the original environment. Edit `train:` / `val:` in the YAMLs and the `BASE_DIR` / `--dataset` arguments to point to your local dataset before running dataset-dependent steps.

---

## Data availability

- **Released:** the duty-cycle thermal telemetry logs (`data/thermal_telemetry/`), the complete leakage-pair inventory (`audit/leakage_report.csv`), and all audit / reproduction / deployment code.
- **Not released:** the durian leaf-disease image dataset and the trained model weights (`.pt` / `.onnx`), which are proprietary assets of an ongoing commercialization effort. Dataset-dependent steps (2–4) therefore require access to the original images; the thermal reproduction (step 1) is fully self-contained.

---

## Citation

```
L. D. Shan, "Thermal Cut-off, Not Throttling: Duty-Cycle Scheduling and a
Dataset-Integrity Audit for Edge-Based Durian Disease Detection," under
review, Computers and Electronics in Agriculture, 2026.
```

## License

Apache-2.0. Code and telemetry are released for research reproducibility; the image dataset and model weights are not covered by this license and are not distributed.


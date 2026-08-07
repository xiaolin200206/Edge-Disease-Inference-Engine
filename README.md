# Edge-Based Durian Disease Detection — Deployment Engine & Reproducibility Package

Code, telemetry, and audit tools accompanying the manuscript:

> **Thermal Cut-off, Not Frequency Throttling: Duty-Cycle Scheduling and a
> Dataset-Integrity Audit for On-Farm Edge Disease Detection in Durian**
> Lin Ding Shan, Institute of Computer Science and Digital Innovation, UCSI University
> ORCID [0009-0009-6031-8479](https://orcid.org/0009-0009-6031-8479)
> Under review, *Smart Agricultural Technology*, 2026 (ATECH-D-26-02454, revision 1).

This repository contains (1) the offline field-deployment inference engine, (2) the
dataset-integrity audit used to detect, quantify and correct a train/validation leak,
and (3) scripts and telemetry that reproduce the paper's analyses from raw data.

---

## What changed in revision 1

Ten changes are material enough that anyone who used the first release should know
about them. Five are corrections to this repository's own code or claims.

**Power is now measured.** The first release asserted that energy was not the binding
constraint without measuring it. All five duty-cycle configurations were re-run a
third time with the SoC's power-management controller sampled every two seconds
(`data/power_round3/`, `reproduce/analyse_power.py`). The result does not go the way
the assertion implied: duty cycling lowers mean SoC power 28.2% but raises energy per
inference 39.4%, because the idle floor is paid throughout the sleep interval. The
configuration that runs hottest also draws the least power while active, because it is
the only one being thermally limited. That round's thermal record is released but is
not used for any thermal claim - see `data/power_round3/README.md` for why.

**RT-DETR-l now has three seeds.** The first release reported it on one, which has no
spread. Seeds 1 and 2 are added (`data/training_logs/multiseed/`,
`data/training_logs/multiseed/rtdetr_3seeds.json`, `reproduce/val_rtdetr.py`), giving
mAP@0.5 = 0.4530 +/- 0.0150. The replication also produced an observation that is now
in Section 4.2.1: on this architecture the two smallest classes exchange their scores
between seeds, root_disease and Pink_Disease swapping 0.995 for 0.045 and 0.028 with
nothing changed but the seed. They hold one and two validation instances.

**The field log did not predate the duty-cycle build.** The first release described
`data/field_test.csv` as having been collected under continuous inference with a build
carrying neither the scheduler nor the cut-off. Its own timestamps disprove that: the
log contains 138 pauses of 15.2 s at a median spacing of 75.1 s, which is the
60 s / 15 s period of configuration B, and 20 pauses of 5.2 s occurring only between
79.3 and 81.5 degrees C, which is the cut-off retry. Counting 5 s and 20 s pauses
together — the rule that reproduces the laboratory event-log counts exactly (99, 108,
32 + 9 = 41, 2 + 1 = 3) — gives 27 cut-off events in the field.

**The field log is a concatenation of runs, and must be split at its timestamp
discontinuities.** The series steps backwards three times (-80.4 s, -111.9 s,
-242.8 s), each step a boundary between runs whose wall clocks overlap. Splitting on
forward gaps alone merges three of them, which overstates the substantive run's
duration and dilutes its throttling fraction with samples taken before the device
warmed up. `reproduce/field_log_intervals.py` splits on both. The substantive run is
2.25 h, 91.2% throttled, peak 81.5 degrees C, 27 inferred cut-offs costing 1.66% of
it. The first release's 2.25 h and 91% were right; its 82 degrees C rounded up from
81.5, and its description of the build was wrong.

**Figure 3 had its axis labels transposed.** The confusion matrix was computed with
predictions on the vertical axis and ground truth on the horizontal, and labelled the
other way round. Read as labelled it showed 54 spurious Phomopsis detections on
background; read correctly it shows all 54 Phomopsis instances missed and no Phomopsis
box emitted anywhere. `reproduce/make_figures_2_3.py` redraws it and checks its column
totals against the validation composition before writing.

**Replication of the thermal characterisation.** All five duty-cycle configurations
were re-run four weeks after the first round, on the same device with the same
software stack and the same camera, with an automatic wait for the SoC to fall below
55 °C before each run. Both rounds are released. The direction of the effect
replicates without exception; the magnitude of the intermediate counts does not.

**A bug in the throttle flag, and its consequence.** `vcgencmd get_throttled` returns
a bit field whose bits 16–19 latch until reboot. The first release tested the raw word
against zero, so once any transient event occurred the flag never cleared for the rest
of the session. The reported "percentage of samples throttled" was therefore not an
independent measurement: it equals `100 × (1 − t_first / duration)`, a restatement of
the time-to-first-throttle column beside it. Verified across all released runs — zero
set-to-clear transitions. `deployment/detection.py` now masks with `0b1111` and logs
the raw word separately as `Throttled_Raw`. The revised manuscript drops the
percentage column and reports cut-off counts, which the engine records independently
at its own 82 °C threshold.

**A silent camera fallback, and the runs it invalidated.** `_init_camera` fell back to
`cv2.VideoCapture(0)` without checking that the device had opened. With no camera
attached the capture object exists but yields nothing, and the inference framework
substitutes its own bundled sample images. A three-hour benchmark in that state
completes and writes plausible-looking telemetry; it was caught only because median
latency was 819 ms against 410 ms with the camera present. The engine now aborts.
The affected runs are retained under `data/thermal_telemetry_INVALID_no_camera/`
rather than deleted, because the failure mode is worth being able to recognise.

**Retraining on a leak-free partition.** The first release stated that pre-augmentation
source photographs could no longer be recovered as a distinct set, so a source-level
re-partition would have to rely on perceptual clustering alone. That was too strong.
Roboflow preserves the pre-export filename before the `.rf.` marker, and of 1,121
images 223 multi-image stems cover 452 of them; 25 further groups are byte-identical.
`audit/regroup_split.py` combines those provenance signals with dihedral-invariant
perceptual hashing, and `training/run_m1.py` retrains on the resulting partition.

**Replication across seeds and a third architecture family.** `training/multiseed.py`
trains the matched configuration under several seeds; `training/run_rtdetr.py` adds a
transformer-based, NMS-free detector under the identical configuration;
`training/collect_table2.py` evaluates every run and reports the spread within each
architecture and the paired difference on shared seeds.

**Bootstrap intervals on per-class AP.** `audit/bootstrap_ap.py` resamples the
validation set at the image level, evaluating both checkpoints on identical resamples
so that the architecture difference is a paired quantity. It requires one inference
pass and no retraining.

---

## Read this first: what is and is not reproducible here

The paper reports two kinds of result, and they are not equally checkable.

**Fully reproducible from this repository, with no additional assets.**
Everything that rests on telemetry rather than on imagery, which is more of the paper
than the split first suggests:

- the duty-cycle thermal characterisation across both rounds, and the cut-off counts
  and coverage figures the operational recommendation rests on (Table 3);
- the SoC power measurement and every derived quantity in Table 4, including the
  energy-per-inference result (§4.4);
- the field-session analysis: session segmentation, latency, throttling fraction and
  the calibrated cut-off inference (§5.1, §5.4);
- Figures 5, 6 and 7, redrawn from the released telemetry;
- Figures 2 and 3, whose values are carried in the plotting script itself.

The raw telemetry is released unfiltered; one command recomputes every value in each
case. `verify_tables.py` checks the numbers that appear in the paper against the logs.

**Not reproducible from this repository.**
Everything requiring the durian images or the trained weights: the leakage
quantification, the architecture comparison including the RT-DETR-l seeds, the seed and
bootstrap analyses, the leak-free retraining, and the taxonomy repair. The dataset and
weights are assets of an ongoing commercialisation effort and are not released.

For those results the repository provides the *procedure* rather than the *evidence*.
Every script runs against any other detection dataset in YOLO format, and the leakage
inventory is published **as hashes**, so the specific leak reported in the paper can be
checked by anyone holding the same images — or the same export — without the images
being distributed here.

---

## What reproduces what

| Paper element | Script / data | Needs the dataset? |
|---|---|---|
| **§4.1** Integrity audit — 1 byte-identical + 117 near-duplicate pairs | `audit/check_leakage_leafrot.py` → `audit/leakage_report.csv` | yes |
| **§4.1** Source-component reconstruction, threshold sweep, ≈750 sources behind 1,121 images | `audit/regroup_split.py --report-only` | yes |
| **Table 1** Leak-free re-evaluation, model held fixed | `reproduce/clean_val_and_revalidate.py` | yes |
| **Table 2** Architecture comparison across seeds and families | `training/multiseed.py`, `training/run_rtdetr.py`, `training/collect_table2.py` | yes |
| **Supplementary Table S3** (S4.2.2) Bootstrap confidence intervals, paired across checkpoints | `audit/bootstrap_ap.py` | yes |
| **§4.4, Table 4** SoC power by duty-cycle configuration | `reproduce/analyse_power.py` + `data/power_round3/` | **no** |
| **Table 2, S4** RT-DETR-l across three seeds | `training/train_rtdetr_seeds.py`, `reproduce/val_rtdetr.py` | yes |
| **§4.2.4** Retraining on the leak-free partition | `audit/regroup_split.py` then `training/run_m1.py` | yes |
| **Table 3** Duty-cycle thermal, both rounds | `reproduce/reproduce_table3.py` + `data/thermal_telemetry*/` | **no** |
| **§3.2** Taxonomy repair (`Early_Blight`/`early_blight`, nc 8 → 7) | `reproduce/remove_class_and_reindex.py` | yes |
| **§5.1, §5.4** Field session: segmentation, latency, throttling, inferred cut-offs | `reproduce/field_log_intervals.py` + `data/field_test.csv` | **no** |
| **Figs. 2, 3** Per-class AP and confusion matrix, pre-cleanup taxonomy | `reproduce/make_figures_2_3.py` | **no** |
| **Figs. 5, 6, 7** Thermal traces, transitions, latency | `reproduce/make_figures.py` + `data/thermal_telemetry*/` | **no** |
| **§5** Field deployment system | `deployment/detection.py` | weights only |

---

## Two conventions that decide whether a number is comparable

**Evaluation threshold.** Every detection figure in the paper is computed at the
framework's default detector confidence threshold, at which essentially every
candidate box enters the precision–recall computation. A figure computed at a tuned
operating point is not comparable to one here. `audit/bootstrap_ap.py` uses an
independent AP implementation (101-point interpolation, IoU 0.5, no confidence
threshold) whose point estimates differ from the framework's by up to 0.13 in
aggregate; its intervals are reported as relative uncertainty, not as a
re-estimation of the framework's values.

**Training environment.** Every run in the revised Table 2 was trained on one machine
under one framework version. This was not true of the first release: the two
architectures were trained thirteen days apart under ultralytics 8.4.100 and 8.4.87
respectively, which the manuscript described as a comparison in which architecture was
the only variable. The discrepancy is recoverable from the `version` and `date` fields
that the framework embeds in every checkpoint:

```python
import torch
ck = torch.load("weights/best.pt", map_location="cpu", weights_only=False)
print(ck["version"], ck["date"], ck["train_args"]["data"])
```

Run this on any checkpoint before comparing it to another.

---

## Layout

```
audit/
  check_leakage_leafrot.py   MD5 + pHash cross-partition duplicate detection
  add_hashes.py              hash inventory for the released report
  regroup_split.py           source-component reconstruction and leak-free split
  bootstrap_ap.py            image-level bootstrap CIs on per-class AP
  leakage_report.csv         released inventory, hashes only
training/
  multiseed.py               retrain the matched configuration under N seeds
  run_rtdetr.py              third architecture family, identical configuration
  run_m1.py                  retrain on the leak-free partition
  collect_table2.py          evaluate every run; spread and paired differences
reproduce/
  reproduce_table3.py        duty-cycle table from raw telemetry, both rounds
  analyse_power.py           Table 4, SoC power by duty cycle (third round)
  val_rtdetr.py              RT-DETR-l three-seed validation
  verify_tables.py           checks the paper's telemetry numbers against the logs
  make_figure_4.py           Fig. 4
  field_log_intervals.py     field session: calibrates the cut-off inference
                             against the laboratory event logs, then applies it
  make_figures.py            Figs. 5, 6, 7
  make_figures_2_3.py        Figs. 2, 3; checks its own column totals
  clean_val_and_revalidate.py
  remove_class_and_reindex.py
  count_classes.py
  train_yolov8s_matched.py   first-release training entry points, retained
  train_yolov11s_paper2.py
deployment/
  detection.py               field inference engine, duty-cycle scheduler
data/
  thermal_telemetry/                   round 1, July 2026
  thermal_telemetry_aug2026/           round 2, August 2026
  power_round3/                        round 3, power-instrumented; see its README
  thermal_telemetry_INVALID_no_camera/ retained failed runs, see above
  training_logs/
  field_test.csv
config/
  data_clean.yaml, data_orig_abs.yaml
```

---

## Quick start

Reproduce the thermal table, which needs nothing but this repository:

```bash
pip install -r requirements.txt
python reproduce/reproduce_table3.py --data-dir data/thermal_telemetry
python reproduce/reproduce_table3.py --data-dir data/thermal_telemetry --published-only
```

The second command reproduces the first release's table exactly, so the effect of
adding the replicate runs is visible as a difference rather than asserted.

Run the audit against your own dataset:

```bash
python audit/check_leakage_leafrot.py --root /path/to/dataset
python audit/regroup_split.py --root /path/to/dataset --report-only --threshold 2
```

The threshold matters. At a Hamming distance of 10 the components chain-merge: in the
largest component of this dataset 74% of pairs exceeded the threshold and the median
pairwise distance was 28, which is what transitive closure over a visually homogeneous
class produces. A genuine duplicate group has a pairwise distance near zero. Sweep the
threshold and inspect component diameter before trusting a partition.

Field deployment:

```bash
python deployment/detection.py

EDIE_CYCLE_ACTIVE_SEC=60 EDIE_CYCLE_SLEEP_SEC=30 \
EDIE_RUN_LABEL=groupC_60-30 EDIE_LOG_DIR=./logs/groupC \
  python deployment/detection.py
```

The engine aborts if no camera is available. `EDIE_ALLOW_NO_CAMERA=1` bypasses that
check; telemetry from such a run must not be reported.

---

## Data availability

The durian disease image dataset and the trained model weights are proprietary assets
of an ongoing commercialisation effort and are not publicly released. All code,
telemetry and hash inventories required to reproduce the reported analyses are
provided here.

## Licence

See `LICENSE`.

# Edge-Based Durian Disease Detection — Deployment Engine & Reproducibility Package

Code, telemetry, and audit tools accompanying the manuscript:

> **Thermal Cut-off, Not Frequency Throttling: Duty-Cycle Scheduling and a
> Dataset-Integrity Audit for On-Farm Edge Disease Detection in Durian**
> Lin Ding Shan, Institute of Computer Science and Digital Innovation, UCSI University
> ORCID [0009-0009-6031-8479](https://orcid.org/0009-0009-6031-8479)
> Under review, *Smart Agricultural Technology*, 2026 (ATECH-D-26-02454, revision 2).
>
> Dataset and weights: [https://doi.org/10.5281/zenodo.22138420](https://doi.org/10.5281/zenodo.22138420) (CC BY 4.0)

This repository contains (1) the offline field-deployment inference engine, (2) the
dataset-integrity audit used to detect, quantify and correct a train/validation leak,
and (3) scripts and telemetry that reproduce the paper's analyses from raw data.

---

## What changed in revision 2

Nothing in the data changed. The second round of review asked for claims to be
brought back inside the evidence supporting them, and for evidence already held
to be presented more directly. Five analyses were added to do the second of
those; all of them run on files that were already in this repository at
revision 1.

**The architecture comparison now carries a formal test, and the test says the
experiment cannot resolve the difference.** On the three seeds shared by all
three families, every 95% confidence interval on paired mAP@0.5 includes zero.
More usefully, with three paired seeds the exact sign-flip permutation test has
eight possible outcomes, so the smallest attainable two-sided p-value is 0.25 —
no architectural difference of any size could have reached significance under
this design. The minimum difference the design could have detected is 0.174,
against the 0.017 the original submission attributed to architecture. See
`analysis/c04_architecture_stats.py`.

**The 27 field cut-offs are insensitive to the rule that infers them.** All 127
inter-sample gaps longer than 1 s in the 2.25 h session fall into one of three
tight clusters — 5.18-5.29 s (n = 20), 15.20-15.35 s (n = 100) and 20.20-20.26 s
(n = 7) — with nothing at all between 5.6 s and 15.0 s or between 21 s and 60 s.
Four hundred combinations of the band edges were swept; all four hundred return
27, and 240 of them still reproduce every laboratory event-log count exactly.
The count is still an inference and is labelled as one throughout, but its
uncertainty is interpretive rather than numerical: excluding merged 20 s pauses
would give 20, which is the floor. See
`analysis/c12_field_event_sensitivity.py`.

> **Correction, revision 2c.** The gap histograms `c12_field_event_sensitivity.py`
> emitted were computed over the whole log while the counts beside them were
> computed over the substantive session, so the 15 s cluster appeared as 140 in
> `results_c12.json` against the 100 the manuscript and Fig. 12 report. The 5 s
> and 20 s clusters were unaffected, since all of those pauses fall inside the
> substantive session. This is the same pooling error corrected in c13 at
> revision 2b. The script now segments before it bins, emits the whole-log
> distribution separately under `whole_log_for_reference`, and
> `reproduce/verify_tables.py` asserts the session-scoped cluster counts and
> their ranges. No number in the manuscript changed.

**The laboratory USB camera configuration is conservative relative to the field
CSI one.** Both ran configuration B, so the schedule is held fixed. Over
active-mode samples the field build recorded 95.1% mean CPU utilisation, 77.0
degrees C mean die temperature and an 81.5 degrees C peak; the two laboratory
rounds recorded 97.4% and 97.6%, 77.0 and 77.8 degrees C, and peaks of 83.2 and
82.0 degrees C. The laboratory setup sits at or above the field one on every
load and thermal indicator, which is the direction that matters for results that
are limits rather than optima. See `analysis/c13_camera_interface.py`.

> **Correction, revision 2b.** `analysis/c13_camera_interface.py` originally
> summarised the field log without splitting it into its five logging runs, so
> the JSON it emitted pooled the substantive 2.25 h session with four short
> warm-up and restart fragments. Pooling diluted the throttling fraction from
> 91.2% to 66.2% and shifted the mean die temperature (77.0 to 75.6 C) and the
> median latency (173.0 to 177.8 ms). The manuscript reports the substantive
> session throughout; the script now applies the same segmentation rule used in
> `c12_field_event_sensitivity.py` and `reproduce/verify_tables.py`, emits the
> whole-log and per-session summaries alongside it, and `verify_tables.py`
> asserts the substantive-session figures so the two cannot drift apart again.
> No number in the manuscript changed.

**The leakage threshold is not doing the work.** Of the 32 validation images
Table 1 removes, 20 are implicated at exact perceptual identity (Hamming
distance 0), rising to 25 at distance 2, 27 at 4, 29 at 6 and 32 at 8. No pair
falls at distance 9 or 10, so tightening the detection threshold from 10 to 8
changes nothing at all. The full cross-partition distance distribution needs the
imagery and cannot be produced here; `audit/phash_distribution.py` regenerates
it for anyone holding the same export.

**Monitoring coverage is now defined rather than used informally, and the
definition exposed an error.** Revision 1 stated that a 5 s suspension costs a
walking operator three trees. Three trees in 5 s implies 6 m/s at 10 m row
spacing, which is a sprint. At 1.2 m/s a suspension costs six tenths of a tree,
and the 27 events of the recorded session cost about 16 trees of roughly 970
encountered. The manuscript is corrected accordingly. See
`analysis/c11_coverage_model.py`, which states the three assumptions the
conversion rests on and sweeps walking speed, row spacing and pause behaviour.

**Thermal run conditions are consolidated, including what was not recorded.**
`analysis/c09_thermal_conditions.py` builds a per-run table over all fifteen
runs from the released telemetry. Ambient air temperature, relative humidity,
airflow and fan speed were not instrumented in any round; the die temperature at
`SYSTEM_START` is reported as the available proxy and is labelled as a proxy.
The interval since the previous run is tabulated alongside it, because that is
what drives the difference between rounds.

**The audit was extended inside each partition, and found something.** The
leakage check answers "does an image cross the train/validation boundary?", so a
file duplicated inside one partition is invisible to it by construction. Running
the same hashes within each partition finds 21 groups of byte-identical images in
train and 3 in valid, so the 1,121 exported files are 1,096 distinct images. Two
of the validation groups pair an annotated original with an unannotated
duplicate, so one image is counted once as an instance and once as background.
No conclusion changes and the duplicates are retained in the archive, because the
reported numbers were computed on them. See `audit/check_within_split_dupes.py`.

**The checkpoint inventory was missing two rows.** `checkpoint_metadata.csv` was
assembled by hand and omitted RT-DETR-l seeds 1 and 2, although both runs appear
under `data/training_logs/multiseed/` and both are in Table 2. Section 3.5 offers
that file as the evidence for the single-environment claim, so a gap in it is a
gap in the claim. `reproduce/extract_checkpoint_metadata.py` now regenerates the
file from the checkpoints; run it against the released weights rather than
editing the CSV by hand.

**Five figures were added** (`figures/fig8` through `figures/fig12`),
regenerated by `analysis/make_revision2_figures.py`.

**Claims narrowed.** The recommendation is now "at least 30 s under the tested
conditions, with 45 s providing greater thermal margin" rather than "eliminates
cut-offs"; power is consistently "SoC power" and "SoC energy per inference" with
a measurement-boundary diagram; the failure taxonomy is a candidate diagnostic
category rather than an established one; and the transferability claim is stated
as a procedure offered for use elsewhere, not a result shown to generalise.

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
log contains 140 pauses of 15.2 s at a median spacing of 75.1 s, which is the
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

## Read this first: what the release covers

**As of revision 2 the dataset and the trained weights are public.** They are
archived at [https://doi.org/10.5281/zenodo.22138420](https://doi.org/10.5281/zenodo.22138420) — 1,121 exported images with three label sets, the
leak-free validation partition, the source-level re-partition of Section 4.2.4,
and one checkpoint per run in Table 2. Every detection result in the paper can
now be regenerated, not merely inspected.

Earlier releases of this repository stated that the detection results were
"procedurally auditable but not independently reproducible" because the imagery
and weights were withheld as commercialisation assets. That is no longer the
case. The competing-interest declaration stands — the author does hold a
financial interest in those assets — but the assets themselves are released, so
the findings can be checked by anyone.

Two things the release does **not** do, and which no amount of data sharing
could:

- It does not supply an independent, source-level held-out test partition. None
  was constructed, and publishing the existing partitions does not create one.
  Every accuracy figure here remains a validation figure.
- It does not make the small classes interpretable. Two classes hold one and two
  validation instances; releasing them changes nothing about what they can
  support.

The dataset is published with its defects documented rather than corrected: the
train/validation leak, the within-partition duplicates, the 565 unannotated
training images, and the near-zero scene diversity behind the rare classes are
all listed in the archive's README. The point of releasing a flawed dataset
alongside the audit that found the flaws is that the audit becomes checkable.

Everything resting on telemetry rather than imagery was already reproducible and
remains so: the duty-cycle thermal characterisation across both rounds, the SoC
power measurement, the field-session analysis, and the figures drawn from them.

## What reproduces what

| Paper element | Script / data | Needs the archive? |
|---|---|---|
| **§4.1** Integrity audit — 1 byte-identical + 117 near-duplicate pairs | `audit/check_leakage_leafrot.py` → `audit/leakage_report.csv` | yes (archived) |
| **§4.1** Source-component reconstruction, threshold sweep, ≈750 sources behind 1,121 images | `audit/regroup_split.py --report-only` | yes (archived) |
| **Table 1** Leak-free re-evaluation, model held fixed | `reproduce/clean_val_and_revalidate.py` | yes (archived) |
| **Table 2** Architecture comparison across seeds and families | `training/multiseed.py`, `training/run_rtdetr.py`, `training/collect_table2.py` | yes (archived) |
| **Supplementary Table S3** (S4.2.2) Bootstrap confidence intervals, paired across checkpoints | `audit/bootstrap_ap.py` | yes (archived) |
| **§4.4, Table 6** SoC power by duty-cycle configuration | `reproduce/analyse_power.py` + `data/power_round3/` | **no** |
| **Table 2, S4** RT-DETR-l across three seeds | `training/train_rtdetr_seeds.py`, `reproduce/val_rtdetr.py` | yes (archived) |
| **§4.2.4** Retraining on the leak-free partition | `audit/regroup_split.py` then `training/run_m1.py` | yes (archived) |
| **Table 3** Duty-cycle thermal, both rounds | `reproduce/reproduce_table3.py` + `data/thermal_telemetry*/` | **no** |
| **§3.2** Taxonomy repair (`Early_Blight`/`early_blight`, nc 8 → 7) | `reproduce/remove_class_and_reindex.py` | yes (archived) |
| **§5.1, §5.4** Field session: segmentation, latency, throttling, inferred cut-offs | `reproduce/field_log_intervals.py` + `data/field_test.csv` | **no** |
| **Figs. 4, 5** Per-class AP and confusion matrix, pre-cleanup taxonomy (files `fig2`, `fig3`) | `reproduce/make_figures_2_3.py` | **no** |
| **Figs. 7, 8, 9** Thermal traces, transitions, latency (files `fig5`, `fig6`, `fig7`) | `reproduce/make_figures.py` + `data/thermal_telemetry*/` | **no** |
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
  phash_distribution.py      full cross-partition distance distribution;
                             needs the imagery, so cannot run from this repo
  check_within_split_dupes.py
                             duplicates INSIDE a partition, which the
                             cross-boundary audit cannot see by construction
  regroup_split.py           source-component reconstruction and leak-free split
  bootstrap_ap.py            image-level bootstrap CIs on per-class AP
  leakage_report.csv         released inventory, hashes only
training/
  multiseed.py               retrain the matched configuration under N seeds
  run_rtdetr.py              third architecture family, identical configuration
  run_m1.py                  retrain on the leak-free partition
  collect_table2.py          evaluate every run; spread and paired differences
## Figure and table numbering

The manuscript numbers figures and tables in order of first citation. The file
names in `figures/` keep the working numbers they were generated under and are
**not** renamed, so that every previously released artefact resolves. Map:

| Manuscript | File | Manuscript | File |
|---|---|---|---|
| Fig. 1 | (field photograph, not in repository) | Fig. 7 | `fig5_thermal_traces` |
| Fig. 2 | `fig8_dataset_flow` | Fig. 8 | `fig6_transitions` |
| Fig. 3 | `fig9_phash_evidence` | Fig. 9 | `fig7_latency` |
| Fig. 4 | `fig2_per_class_ap` | Fig. 10 | `fig12_coverage` |
| Fig. 5 | `fig3_confusion` | Fig. 11 | `fig10_power_boundary` |
| Fig. 6 | `fig4_phomopsis` | Fig. 12 | `fig11_field_gap_structure` |

Tables 1, 2 and 3 keep their revision-1 numbers. Revision 2 adds Table 4 (the
dataset-differences table, Section 4.2.3), Table 5 (per-run thermal conditions,
Section 4.3) and Table 7 (demonstrated against unvalidated, Section 5.4). The
SoC power table introduced in revision 1 is **Table 6** in the revision-2
manuscript; some check labels in `reproduce/verify_tables.py` still call it
Table 6, which is correct. Supplementary Table S6 carries the architecture
statistics.

analysis/
  c04_architecture_stats.py  paired tests across architectures; power statement
  c09_thermal_conditions.py  per-run conditions for all fifteen thermal runs
  c11_coverage_model.py      coverage definition, assumptions, sensitivity grid
  c12_field_event_sensitivity.py
                             threshold sensitivity of the inferred cut-off count
  c13_camera_interface.py    field CSI against laboratory USB, schedule matched
  make_revision2_figures.py  Figs. 2, 3, 10, 11, 12 (files fig8-fig12)
  results_*.json             the outputs the manuscript's numbers were taken from
reproduce/
  reproduce_table3.py        duty-cycle table from raw telemetry, both rounds
  analyse_power.py           Table 6, SoC power by duty cycle (third round)
  val_rtdetr.py              RT-DETR-l three-seed validation
  verify_tables.py           checks the paper's telemetry numbers against the logs
  extract_checkpoint_metadata.py
                             rebuilds checkpoint_metadata.csv from the
                             checkpoints themselves
  make_figure_4.py           Fig. 6 (file fig4)
  field_log_intervals.py     field session: calibrates the cut-off inference
                             against the laboratory event logs, then applies it
  make_figures.py            Figs. 7, 8, 9 (files fig5, fig6, fig7)
  make_figures_2_3.py        Figs. 4, 5 (files fig2, fig3); checks its own column totals
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

Reproduce the revision-2 analyses, none of which need anything outside this
repository:

```bash
python analysis/c04_architecture_stats.py
python analysis/c09_thermal_conditions.py \
    --root data/thermal_telemetry --root data/thermal_telemetry_aug2026 \
    --root data/power_round3
python analysis/c11_coverage_model.py
python analysis/c12_field_event_sensitivity.py \
    --field data/field_test.csv \
    --lab data/thermal_telemetry --lab data/thermal_telemetry_aug2026
python analysis/c13_camera_interface.py \
    --field data/field_test.csv \
    --lab data/thermal_telemetry/groupB_60-15_20260706_132828 \
    --lab data/thermal_telemetry_aug2026/groupB_60-15_20260804_115108
python analysis/make_revision2_figures.py --repo . --out figures/
```

Each writes a `results_*.json` into the working directory alongside its console
output. The copies committed under `analysis/` are the ones the manuscript's
numbers were taken from, so a discrepancy after re-running is a discrepancy
worth reporting.

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

All artefacts of this study are public.

**Telemetry and analysis** — thermal logs for all five duty-cycle configurations
across three rounds, the power samples, the field-deployment log, and every
analysis and audit script — are in this repository.

**Imagery and weights** are archived at [https://doi.org/10.5281/zenodo.22138420](https://doi.org/10.5281/zenodo.22138420) under CC BY 4.0: the
1,121 exported images, three label sets spanning the taxonomy revisions of
Section 3.2, the leak-free validation partition behind Table 1, the source-level
re-partition of Section 4.2.4, and the checkpoints behind Table 2 and Table 1.
The archive carries a MANIFEST and MD5 checksums so every file can be verified.

The durian images were collected by collaborating growers who have given written
permission for their public release.

## Licence

See `LICENSE`.

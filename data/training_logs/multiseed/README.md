# Training logs - revision 1 runs

`args.yaml` and `results.csv` for every run reported in Table 2 and Section
4.2.4. `args.yaml` records every hyperparameter as the framework received it;
`results.csv` is the full per-epoch history.

One substitution has been made in these files: absolute paths written by the
training framework have been replaced with `<local>/`, in `args.yaml`, in
`checkpoint_metadata.csv` and in this file. Nothing else is altered — every
hyperparameter, metric and timestamp is as the framework recorded it.

Model weights are not included. They are derived from the unreleasable imagery
and are excluded for the same reason (see Data availability). What the weights
would otherwise let a reader check - which framework version trained each run,
when, and against which dataset configuration - is extracted into
`checkpoint_metadata.csv` instead, so the claim in Section 3.5 that all runs
reported in this revision share one environment can be verified without
them. The two first-release runs are listed in the same file under
`group = first release` precisely because they do *not* share it.

Framework versions across the runs collected here: 8.4.87
Dataset configuration files referenced: 2

## First-release runs

Metadata for the first-release runs is included in `checkpoint_metadata.csv` under `group = first release`, so the discrepancy Section 3.5 describes can be checked here rather than taken on trust:

- `first_release_v8s_matched`: ultralytics 8.4.100, 2026-07-18, data `/content/data.yaml`
- `first_release_v11s_wholeleaf`: ultralytics 8.4.87, 2026-07-05, data `<local>/Leave_disease\data.yaml`

They differ in framework version (8.4.100 against 8.4.87) and in execution environment: the `data_yaml` column shows the first was trained under `/content/`, a hosted Colab runtime, and the second on a local workstation. That is the difference Section 3.5 discloses and eliminates. Their log directories are left where they are and nothing was copied from them.

## Columns in checkpoint_metadata.csv

| column | meaning |
|---|---|
| `run` | run label, matching the rows of Table 2 and Supplementary Table S4 |
| `source` | path the run was collected from |
| `ultralytics_version` | framework version embedded in `best.pt` |
| `trained_utc` | timestamp embedded in `best.pt` |
| `data_yaml` | dataset configuration file the run was trained against |
| `epochs`, `batch`, `imgsz`, `seed`, `optimizer` | as received by the trainer |
| `group` | `revision` for the runs reported in this revision, `first release` for the two runs of the original submission |
| `n_classes`, `class_order` | label space, to confirm index alignment across runs |

## Two YOLOv8s groups under the same seeds

`v8s_seed0` to `v8s_seed4` and `table2_v8s_seed0` to `table2_v8s_seed2` are two separate executions of the same configuration under the same seeds; 3 seeds (seed0, seed1, seed2) appear in both. The first group was run to characterise seed-to-seed variation, the second as part of the architecture comparison alongside YOLOv11s and RT-DETR-l, before it was noticed that the first group already covered those seeds.

Both are retained, and neither was selected on its results. Training is deterministic (`deterministic=True`, fixed seed, identical data and environment), so the two executions should return identical results on the shared seeds.

The two executions were compared epoch by epoch on the mAP@0.5 column of `results.csv`. The largest absolute difference at any epoch of any shared seed is **0.000000** (seed0 0.000000 over 150 epochs; seed1 0.000000 over 150 epochs; seed2 0.000000 over 150 epochs). The curves are identical, as a deterministic setting requires.

That is why both groups can be published without ambiguity about which one the paper reports: on the shared seeds there is nothing to choose between them, and the comparison above is an independent check that the determinism the Methods claim actually holds.

Table 2 and Supplementary Table S4 report YOLOv8s over five seeds, which only the first group provides. Supplementary Table S5 additionally reports the three-seed subset, because the leak-free retraining (`m1_regrouped_seed0` to `_seed2`) was run under seeds 0-2 only and the like-for-like comparator is therefore the same three seeds.

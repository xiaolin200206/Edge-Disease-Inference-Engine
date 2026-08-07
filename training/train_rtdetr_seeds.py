#!/usr/bin/env python3
"""
train_rtdetr_seeds.py - add the missing RT-DETR-l seeds to Table 2.

Table 2 currently reports RT-DETR-l on one seed while both YOLO families carry
three or five. That asymmetry is exactly what Major Comment 4 asks about: a
single run has no spread, so the third architecture cannot be said to confirm
anything. Two more seeds close it.

The point of this script is that it does not invent a configuration. It reads
args.yaml from the seed-0 run already in the repository and reuses it verbatim,
changing only `seed` and `name`. Anything else - a different learning rate, a
different data yaml, a different epoch count - would make the three runs
incomparable and defeat the purpose. The script prints what it is about to
change and refuses to run if the reference args are missing.

One thing worth knowing before starting: `optimizer: auto` derives the initial
learning rate from the class count, so the three runs are only matched as long
as the data yaml is the same file. That is checked.

Usage, from the machine with the GPU:

    python train_rtdetr_seeds.py --dry-run
    python train_rtdetr_seeds.py --seeds 1,2

Expect roughly the same wall-clock per seed as the seed-0 run took.
"""

import argparse
import shutil
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("pyyaml not installed:  pip install pyyaml")

# Keys that must not be carried over: they name the run or are runtime state.
DROP = {"name", "seed", "save_dir", "resume", "exist_ok", "model"}

# Keys whose value decides whether two runs are comparable at all. If any of
# these differs from the reference, the new run is not a replicate.
CRITICAL = ["data", "epochs", "batch", "imgsz", "optimizer", "lr0", "lrf",
            "momentum", "weight_decay", "warmup_epochs", "close_mosaic",
            "cos_lr", "rect", "patience", "deterministic", "amp"]


def load_reference(path):
    if not path.exists():
        sys.exit(f"reference args not found: {path}\n"
                 "Point --reference at the seed-0 run's args.yaml.")
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference",
                    default="data/training_logs/multiseed/table2_rtdetr_seed0/args.yaml",
                    help="args.yaml of the run being replicated")
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--model", default="rtdetr-l.pt")
    ap.add_argument("--project", default="runs/detect/runs_table2")
    ap.add_argument("--name-fmt", default="table2_rtdetr_seed{seed}")
    ap.add_argument("--data", default="",
                    help="override the data yaml if the reference path no "
                         "longer resolves on this machine")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    ref = load_reference(Path(a.reference))
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]

    kwargs = {k: v for k, v in ref.items()
              if k not in DROP and v is not None}
    kwargs["project"] = a.project
    if a.data:
        kwargs["data"] = a.data

    data_path = Path(str(kwargs.get("data", "")))
    if not data_path.exists():
        print(f"  data yaml does not resolve here: {kwargs.get('data')}")
        print("  pass --data with the correct path, or run from the directory")
        print("  the reference run was launched from.")
        if not a.dry_run:
            sys.exit(1)

    print("Replicating the seed-0 RT-DETR-l run. Carried over verbatim:")
    for k in CRITICAL:
        if k in ref:
            print(f"    {k:<16} {ref[k]}")
    print(f"\nChanged per run: seed, name")
    print(f"Model: {a.model}   project: {a.project}")
    print(f"Seeds to run: {seeds}\n")

    if a.dry_run:
        for s in seeds:
            print(f"  would train  seed={s}  name={a.name_fmt.format(seed=s)}")
        print("\nDry run only. Drop --dry-run to start.")
        return

    from ultralytics import RTDETR
    import ultralytics
    print(f"ultralytics {ultralytics.__version__}")
    if ref.get("model") and Path(str(ref["model"])).name != a.model:
        print(f"  note: reference used {ref['model']}, this run uses {a.model}")

    for s in seeds:
        name = a.name_fmt.format(seed=s)
        out = Path(a.project) / name
        if (out / "weights" / "best.pt").exists():
            print(f"\n{name} already has weights - skipping. Delete the "
                  "directory to redo it.")
            continue
        print(f"\n{'=' * 60}\n{name}  (seed {s})\n{'=' * 60}")
        model = RTDETR(a.model)
        model.train(seed=s, name=name, exist_ok=True, **kwargs)

    print("\nDone. Collect the logs into the repository with:")
    print("    python collect_training_logs.py --repo <repo> --search .")


if __name__ == "__main__":
    main()

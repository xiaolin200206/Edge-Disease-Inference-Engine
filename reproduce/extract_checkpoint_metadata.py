#!/usr/bin/env python3
"""
extract_checkpoint_metadata.py — rebuild data/training_logs/multiseed/
checkpoint_metadata.csv from the checkpoints themselves.

WHY THIS EXISTS
---------------
Section 3.5 undertakes that the framework version and training timestamp
embedded in every checkpoint are extracted and released, so that the claim of a
single training environment can be checked without the weights. The committed
CSV was assembled by hand and two RT-DETR-l rows (seeds 1 and 2) were omitted,
even though both runs exist under data/training_logs/multiseed/ and both appear
in Table 2. This script regenerates the file from the checkpoints so the
inventory cannot drift from the runs again.

The values are read out of the checkpoint, never typed in. Run it wherever the
weights live.

USAGE
-----
    python reproduce/extract_checkpoint_metadata.py \
        --runs /path/to/runs \
        --extra yolov8s_matched/weights/best.pt=first_release_v8s_matched \
        --out data/training_logs/multiseed/checkpoint_metadata.csv

    --dry-run   print the rows without writing

DEPS
----
    torch, and the ultralytics classes on the import path (the checkpoints are
    pickled ultralytics objects). Run it in the training environment.
"""

import argparse
import csv
import sys
from pathlib import Path


def read_ckpt(p):
    import torch
    ck = torch.load(str(p), map_location="cpu", weights_only=False)
    ta = ck.get("train_args", {}) or {}
    names = ck.get("names") or {}
    if isinstance(names, dict):
        order = " ".join(names[k] for k in sorted(names))
        nc = len(names)
    else:
        order = " ".join(names)
        nc = len(names)
    return {
        "ultralytics_version": ck.get("version", ""),
        "trained_utc": str(ck.get("date", "")),
        "data_yaml": Path(str(ta.get("data", ""))).name,
        "epochs": ta.get("epochs", ""),
        "batch": ta.get("batch", ""),
        "imgsz": ta.get("imgsz", ""),
        "seed": ta.get("seed", ""),
        "optimizer": ta.get("optimizer", ""),
        "n_classes": nc,
        "class_order": order,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help="directory searched recursively for weights/best.pt")
    ap.add_argument("--extra", action="append", default=[],
                    help="PATH=run_label for checkpoints outside --runs")
    ap.add_argument("--out", default="checkpoint_metadata.csv")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    # The directory name is not always the label the paper uses. These two
    # groups predate the revision-2 naming and are referred to by their older
    # labels in Table 1, Table 2, the Zenodo archive's weights/ folder and
    # Supplementary Table S4, so the mapping is pinned rather than derived.
    LABEL = {"seed0": "v8s_seed0", "seed1": "v8s_seed1", "seed2": "v8s_seed2",
             "seed3": "v8s_seed3", "seed4": "v8s_seed4",
             "yolov11s_wholeleaf_v1": "first_release_v11s_wholeleaf"}

    targets = []
    for p in sorted(Path(a.runs).rglob("weights/best.pt")):
        d = p.parent.parent.name
        targets.append((p, LABEL.get(d, d)))
    for spec in a.extra:
        path, _, label = spec.partition("=")
        targets.append((Path(path), label or Path(path).parent.parent.name))

    if not targets:
        sys.exit(f"no weights/best.pt found under {a.runs}")

    cols = ["run", "group", "source", "files", "ultralytics_version", "trained_utc",
            "data_yaml", "epochs", "batch", "imgsz", "seed", "optimizer",
            "n_classes", "class_order"]
    rows, failed = [], []
    for p, label in targets:
        try:
            info = read_ckpt(p)
        except Exception as e:
            failed.append(f"{p}: {e.__class__.__name__}: {e}")
            continue
        rows.append({
            "run": label,
            "group": "first release" if label.startswith("first_release") else "revision",
            "source": str(p.parent.parent),
            "files": "args.yaml results.csv",
            **info,
        })
        print(f"  {label:44s} v{info['ultralytics_version']:8s} "
              f"seed={info['seed']}  {info['trained_utc']}")

    if failed:
        print(f"\n{len(failed)} checkpoint(s) could not be read:")
        for f in failed:
            print("  !", f)

    print(f"\n{len(rows)} rows")
    versions = {r["ultralytics_version"] for r in rows if r["group"] == "revision"}
    print(f"framework versions across revision runs: {sorted(versions)}"
          + ("   <- single environment, as Section 3.5 claims"
             if len(versions) == 1 else
             "   <- MORE THAN ONE; Section 3.5's claim does not hold"))

    if a.dry_run:
        print("\ndry run: nothing written.")
        return
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\r\n")
        w.writeheader()
        w.writerows(rows)
    print(f"written: {a.out}")


if __name__ == "__main__":
    main()

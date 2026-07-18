"""
train_yolov11s_paper2.py

Purpose
-------
Train YOLOv11s on paper 2's ORIGINAL (whole-leaf annotated) dataset.
This is the model-capacity arm of the ablation:
  - Does a stronger architecture (v11s) rescue the zero-AP rare classes
    (Phomopsis, Pink_Disease, root_disease) WITHOUT changing the
    whole-leaf annotation protocol?
  - If yes  -> the original diagnosis ("model too shallow") still holds partially.
  - If no   -> supports the new argument that annotation granularity,
               not model capacity, is the dominant cause of Dominant
               Class Suppression.

Do NOT run this on the Muar dataset — that's already covered in Paper 4's
Table III. This script is only for the OLD dataset folder shown in your
screenshot (train/valid/data.yaml).

Setup (run once)
----------------
    pip install ultralytics

Usage
-----
    python train_yolov11s_paper2.py
"""

from ultralytics import YOLO
import json
import os

# ---------------- CONFIG — edit these before running ----------------

# Point this at the data.yaml inside your Leave_disease folder
DATA_YAML = r"C:\Users\Lim Ding Shan\Desktop\Durian project and paper\second paper\Leave_disease\data.yaml"

MODEL_WEIGHTS = "yolo11s.pt"   # pretrained COCO checkpoint, auto-downloads first run

# IMPORTANT: match these to whatever you used to train YOLOv8n / YOLOv8s
# originally in this paper. If you don't remember the exact numbers,
# these are reasonable Ultralytics defaults for a small custom dataset.
EPOCHS     = 150
IMG_SIZE   = 640
BATCH      = -1      # auto: Ultralytics picks the largest batch that fits ~60% of VRAM
                      # (was 16 — caused CUDA OOM on 6GB laptop GPU; if -1 still OOMs,
                      # hardcode a small fixed value like 4 instead)
SEED       = 0       # keep this fixed so results are reproducible

PROJECT_DIR = "runs_paper2_ablation"
RUN_NAME    = "yolov11s_wholeleaf_v1"

# ----------------------------------------------------------------------


def main():
    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(
            f"data.yaml not found at:\n  {DATA_YAML}\n"
            "Fix the DATA_YAML path at the top of this script."
        )

    model = YOLO(MODEL_WEIGHTS)

    print("=" * 60)
    print("Training YOLOv11s on the ORIGINAL whole-leaf dataset")
    print("=" * 60)

    train_results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        seed=SEED,
        project=PROJECT_DIR,
        name=RUN_NAME,
        patience=0,      # DISABLED — force the full EPOCHS run to match v8n/v8s protocol
        exist_ok=True,
        plots=True,      # saves confusion matrix, PR curves etc. for the paper
    )

    print("\n" + "=" * 60)
    print("Running validation for per-class AP50")
    print("=" * 60)

    # Use the ACTUAL save directory Ultralytics used, not a guessed path
    best_weights = os.path.join(str(train_results.save_dir), "weights", "best.pt")
    val_model = YOLO(best_weights)
    metrics = val_model.val(data=DATA_YAML, imgsz=IMG_SIZE, split="val")

    class_names = metrics.names
    ap50_per_class = metrics.box.ap50   # array indexed by class id
    ap5095_per_class = metrics.box.ap   # AP@0.5:0.95 per class

    results_table = []
    print(f"\n{'Class':20s} {'AP50':>8s} {'AP50-95':>10s}")
    print("-" * 42)
    for i, name in class_names.items():
        ap50 = ap50_per_class[i] if i < len(ap50_per_class) else float("nan")
        ap5095 = ap5095_per_class[i] if i < len(ap5095_per_class) else float("nan")
        print(f"{name:20s} {ap50:8.3f} {ap5095:10.3f}")
        results_table.append({"class": name, "AP50": float(ap50), "AP50_95": float(ap5095)})

    print("-" * 42)
    print(f"{'mAP50 (all classes)':20s} {metrics.box.map50:8.3f}")
    print(f"{'mAP50-95':20s} {metrics.box.map:8.3f}")

    # Save results to JSON so you can drop it straight into the paper's table
    out_path = os.path.join(PROJECT_DIR, RUN_NAME, "per_class_ap_summary.json")
    with open(out_path, "w") as f:
        json.dump(
            {
                "model": "YOLOv11s",
                "dataset": "paper2_wholeleaf_original",
                "per_class": results_table,
                "mAP50": float(metrics.box.map50),
                "mAP50_95": float(metrics.box.map),
            },
            f,
            indent=2,
        )
    print(f"\nSaved summary to: {out_path}")
    print("\nCompare the Phomopsis / Pink_Disease / root_disease rows above")
    print("against Paper 2's original Table 1 (all 0.00 under YOLOv8n/YOLOv8s).")


if __name__ == "__main__":
    main()
